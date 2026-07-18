"""Repeatable rdsframe benchmark: fresh process per run, median + spread.

Each (operation, repetition) executes in its own child process so no run
benefits from another's imports, interned strings, or allocator state; only
the OS file cache stays warm, which is stated in the output. Categories are
measured separately so catalog scanning, full parsing, and each conversion
target are never conflated:

- ``catalog``  -- ``list_rds_tables(cache=False)`` (structural scan)
- ``pandas``   -- ``read_rds()``
- ``arrow``    -- ``read_rds_arrow()`` (needs pyarrow)
- ``polars``   -- ``read_rds_polars()`` (needs polars)

Usage::

    python benchmarks/bench.py FILE.rds [more.rds ...] [--reps 5]
        [--ops catalog,pandas,arrow,polars]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

OPERATIONS = ("catalog", "pandas", "arrow", "polars")


def run_single(operation: str, path: str) -> None:
    """Child-process entry point: run one operation once, print JSON."""
    import rdsframe

    start = time.perf_counter()
    if operation == "catalog":
        result = rdsframe.list_rds_tables(path, cache=False)
        detail = f"{len(result.tables)} tables"
    elif operation == "pandas":
        frames = rdsframe.read_rds(path)
        frames = frames if isinstance(frames, dict) else {"data": frames}
        detail = f"{sum(len(f) for f in frames.values()):,} rows"
    elif operation == "arrow":
        tables = rdsframe.read_rds_arrow(path)
        tables = tables if isinstance(tables, dict) else {"data": tables}
        detail = f"{sum(t.num_rows for t in tables.values()):,} rows"
    elif operation == "polars":
        frames = rdsframe.read_rds_polars(path)
        frames = frames if isinstance(frames, dict) else {"data": frames}
        detail = f"{sum(len(f) for f in frames.values()):,} rows"
    else:  # pragma: no cover - guarded by the parent
        raise SystemExit(f"unknown operation: {operation}")
    print(json.dumps({"seconds": time.perf_counter() - start, "detail": detail}))


def _operation_available(operation: str) -> bool:
    import importlib.util

    if operation == "arrow":
        return importlib.util.find_spec("pyarrow") is not None
    if operation == "polars":
        return importlib.util.find_spec("polars") is not None
    return True


def bench(path: Path, operations: list[str], reps: int) -> None:
    import rdsframe

    info = rdsframe.inspect_r_file(path)
    print(
        f"\n=== {path.name}: {info.size_bytes / 1_048_576:.1f} MiB, "
        f"compression={info.compression}, {info.serialization} ==="
    )
    print(
        f"{reps} repetitions per operation, one fresh Python process each "
        "(OS file cache warm after the first run)."
    )
    header = f"{'operation':10} {'median s':>10} {'min s':>10} {'max s':>10}  detail"
    print(header)
    print("-" * len(header))
    for operation in operations:
        if not _operation_available(operation):
            print(f"{operation:10} {'skipped':>10}  (dependency not installed)")
            continue
        seconds: list[float] = []
        detail = ""
        for _ in range(reps):
            child = subprocess.run(
                [sys.executable, __file__, "--single", operation, str(path)],
                capture_output=True,
                text=True,
            )
            if child.returncode != 0:
                print(f"{operation:10} {'FAILED':>10}  {child.stderr.strip()[:200]}")
                seconds = []
                break
            payload = json.loads(child.stdout)
            seconds.append(payload["seconds"])
            detail = payload["detail"]
        if seconds:
            print(
                f"{operation:10} {statistics.median(seconds):10.3f} "
                f"{min(seconds):10.3f} {max(seconds):10.3f}  {detail}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--ops", default="catalog,pandas,arrow,polars")
    parser.add_argument("--single", metavar="OPERATION", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single:
        run_single(args.single, args.files[0])
        return 0
    if args.reps < 1:
        raise SystemExit("--reps must be at least 1")
    operations = [op.strip() for op in args.ops.split(",") if op.strip()]
    unknown = [op for op in operations if op not in OPERATIONS]
    if unknown:
        raise SystemExit(f"unknown operations: {unknown}; choose from {OPERATIONS}")
    for name in args.files:
        path = Path(name)
        if not path.is_file():
            raise SystemExit(f"not a file: {path}")
        bench(path, operations, args.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
