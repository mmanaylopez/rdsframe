"""Command-line interface."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from ._core import RDSError
from .api import RDSCatalog, inspect_r_file, list_rds_tables, read_r_object, to_parquet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rdsframe", description="Inspect or convert RDS data.frames"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="inspect an R file header")
    inspect_parser.add_argument("input", type=Path)
    list_parser = subparsers.add_parser(
        "list", help="list tables without materializing their column payloads"
    )
    list_parser.add_argument("input", type=Path)
    list_parser.add_argument("--json", action="store_true", dest="as_json")
    list_parser.add_argument("--catalog", type=Path, help="save a reusable catalog")
    convert_parser = subparsers.add_parser("convert", help="convert RDS data.frames to Parquet")
    convert_parser.add_argument("input", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.add_argument("--basename")
    convert_parser.add_argument("--compression", default="zstd")
    convert_parser.add_argument("--memory-limit", default="1GB")
    convert_parser.add_argument("--temp-directory", type=Path)
    convert_parser.add_argument("--row-group-size", type=int, default=250_000)
    convert_parser.add_argument("--stage-max-columns", type=int, default=16)
    convert_parser.add_argument("--stage-max-bytes", type=int, default=128 * 1024 * 1024)
    convert_parser.add_argument("--gc-collect-every", type=int, default=16)
    convert_parser.add_argument("--max-tables", type=int)
    convert_parser.add_argument("--max-root-items", type=int)
    convert_parser.add_argument(
        "--posixct-mode", choices=["preserve", "utc_naive"], default="preserve"
    )
    convert_parser.add_argument(
        "--invalid-timestamp", choices=["error", "null"], default="error"
    )
    convert_parser.add_argument(
        "--list-column-mode", choices=["infer", "json", "string"], default="infer"
    )
    convert_parser.add_argument(
        "--table-index", type=int, action="append", default=[], help="zero-based table index"
    )
    convert_parser.add_argument(
        "--table-name", action="append", default=[], help="exact table name"
    )
    convert_parser.add_argument(
        "--catalog", type=Path, help="catalog previously created by 'rdsframe list'"
    )
    convert_parser.add_argument(
        "--encoding", help="codec for unflagged native-encoding strings"
    )
    dump_parser = subparsers.add_parser(
        "dump",
        help="print any R object (not only data.frames) as a readable tree or JSON",
    )
    dump_parser.add_argument("input", type=Path)
    dump_parser.add_argument(
        "--json", action="store_true", dest="as_json", help="emit the full object as JSON"
    )
    dump_parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="tree mode: children shown per list/vector before truncating (default 10)",
    )
    dump_parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="tree mode: nesting levels expanded before summarizing (default 8)",
    )
    dump_parser.add_argument(
        "--encoding", help="codec for unflagged native-encoding strings"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except (RDSError, FileNotFoundError, ImportError, ValueError) as exc:
        print(f"rdsframe: error: {exc}", file=sys.stderr)
        return 2


def _run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        payload = asdict(inspect_r_file(args.input))
        payload["path"] = str(payload["path"])
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "list":
        list_catalog = list_rds_tables(args.input)
        if args.catalog:
            list_catalog.save(args.catalog)
        if args.as_json:
            print(json.dumps(list_catalog.to_dict(), ensure_ascii=False, indent=2))
        else:
            print("index\tname\trows\tcolumns")
            for listed_table in list_catalog.tables:
                rows = "?" if listed_table.rows is None else str(listed_table.rows)
                print(
                    f"{listed_table.index}\t{listed_table.name}\t"
                    f"{rows}\t{listed_table.columns}"
                )
        return 0
    if args.command == "dump":
        if args.max_items < 1:
            raise ValueError("--max-items must be at least 1")
        if args.max_depth < 1:
            raise ValueError("--max-depth must be at least 1")
        value = read_r_object(args.input, encoding=args.encoding)
        if args.as_json:
            print(json.dumps(_dump_json_ready(value), ensure_ascii=False, indent=2))
        else:
            for line in _dump_lines(value, 0, args.max_depth, args.max_items):
                print(line)
        return 0
    selectors = [*args.table_index, *args.table_name]
    loaded_catalog = RDSCatalog.load(args.catalog) if args.catalog else None
    tables = to_parquet(
        args.input,
        args.output,
        basename=args.basename,
        compression=args.compression,
        memory_limit=args.memory_limit,
        temp_directory=args.temp_directory,
        row_group_size=args.row_group_size,
        stage_max_columns=args.stage_max_columns,
        stage_max_bytes=args.stage_max_bytes,
        gc_collect_every=args.gc_collect_every,
        max_tables=args.max_tables,
        max_root_items=args.max_root_items,
        posixct_mode=args.posixct_mode,
        invalid_timestamp=args.invalid_timestamp,
        list_column_mode=args.list_column_mode,
        tables=selectors or None,
        catalog=loaded_catalog,
        progress=lambda value: print(f"\r{value:3d}%", end="", flush=True),
        encoding=args.encoding,
    )
    print()
    for output_table in tables:
        print(
            f"{output_table.path}\t{output_table.rows} rows\t"
            f"{output_table.columns} columns"
        )
    return 0


def _summarize(value: Any) -> str:
    """One-line description of a value for the tree renderer."""
    if isinstance(value, pd.DataFrame):
        columns = ", ".join(
            f"{name}:{dtype}" for name, dtype in value.dtypes.astype(str).items()
        )
        return f"<data.frame {len(value)} rows x {value.shape[1]} cols> [{columns}]"
    if isinstance(value, dict):
        return f"<named list, {len(value)} entries>"
    if isinstance(value, (list, tuple)):
        return f"<list, {len(value)} items>"
    if isinstance(value, np.ndarray):
        return f"<matrix {'x'.join(str(size) for size in value.shape)}>"
    if isinstance(value, str):
        return repr(value if len(value) <= 120 else value[:117] + "...")
    return repr(value)


def _dump_lines(
    value: Any, depth: int, max_depth: int, max_items: int, label: str = ""
) -> list[str]:
    pad = "  " * depth
    prefix = f"{pad}{label}" if label else pad
    if isinstance(value, dict) and depth < max_depth:
        lines = [f"{prefix}{_summarize(value)}"]
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                lines.append(f"{pad}  ... (+{len(value) - max_items} more entries)")
                break
            lines.extend(_dump_lines(item, depth + 1, max_depth, max_items, f"{key}: "))
        return lines
    if isinstance(value, (list, tuple)) and depth < max_depth:
        nested = any(isinstance(item, (dict, list, tuple, pd.DataFrame)) for item in value)
        if not nested:
            shown = ", ".join(_summarize(item) for item in value[:max_items])
            extra = f", ... (+{len(value) - max_items} more)" if len(value) > max_items else ""
            return [f"{prefix}[{shown}{extra}]"]
        lines = [f"{prefix}{_summarize(value)}"]
        for index, item in enumerate(value):
            if index >= max_items:
                lines.append(f"{pad}  ... (+{len(value) - max_items} more items)")
                break
            lines.extend(_dump_lines(item, depth + 1, max_depth, max_items, f"[{index}] "))
        return lines
    if isinstance(value, pd.DataFrame):
        lines = [f"{prefix}{_summarize(value)}"]
        preview = value.head(min(max_items, 5)).to_string()
        lines.extend(f"{pad}  {row}" for row in preview.splitlines())
        if len(value) > min(max_items, 5):
            lines.append(f"{pad}  ... (+{len(value) - min(max_items, 5)} more rows)")
        return lines
    return [f"{prefix}{_summarize(value)}"]


def _dump_json_ready(value: Any) -> Any:
    """Convert :func:`read_r_object` output into JSON-serializable data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else {"$float": str(value)}
    if isinstance(value, pd.DataFrame):
        return {
            "$r_type": "data.frame",
            "rows": len(value),
            "columns": {
                str(name): [_dump_json_ready(item) for item in value[name].tolist()]
                for name in value.columns
            },
        }
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return _dump_json_ready(value.item())
    if isinstance(value, np.ndarray):
        return [_dump_json_ready(item) for item in value.tolist()]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, bytes):
        return {"$binary_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _dump_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump_json_ready(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
