"""Column-staged RDS to Parquet conversion powered by DuckDB."""

from __future__ import annotations

import base64
import gc
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ._core import (
    CPLXSXP,
    DIFFTIME_SECONDS_PER_UNIT,
    INTSXP,
    LGLSXP,
    NA_INTEGER,
    RAWSXP,
    REALSXP,
    STRSXP,
    VECSXP,
    RDSCatalogError,
    RDSLimitError,
    Reader,
    SerializedObject,
    UnsupportedRDS,
    as_strings,
    as_value,
)
from ._scan import scan_dataframe_from_header

PosixctMode = Literal["preserve", "utc_naive"]
InvalidTimestampMode = Literal["error", "null"]
ListColumnMode = Literal["infer", "json", "string"]


@dataclass(frozen=True, slots=True)
class StagedColumns:
    path: Path
    indices: tuple[int, ...]


def stream_root_to_parquet(
    reader: Reader,
    root_header: tuple[int, bool, bool, bool, int],
    *,
    out_dir: Path,
    basename: str,
    compression: str,
    row_group_size: int,
    memory_limit: str | None,
    temp_directory: Path | None,
    stage_max_columns: int,
    stage_max_bytes: int,
    gc_collect_every: int,
    max_tables: int | None,
    max_root_items: int | None,
    posixct_mode: PosixctMode,
    invalid_timestamp: InvalidTimestampMode,
    list_column_mode: ListColumnMode,
    selected_indices: frozenset[int] | None,
    selected_names: dict[int, str] | None,
    safe_name: Any,
    unique_name: Any,
) -> list[tuple[str, Path, int, int]]:
    """Convert a root data.frame while retaining at most one parsed column.

    R serializes a data.frame column-by-column and writes its names afterwards.
    Columns are staged in memory-bounded Parquet batches. DuckDB then combines
    those batches with a positional join into the final Parquet output.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "Streaming Parquet export requires: pip install 'rdsframe[parquet]'"
        ) from exc

    sexp_type, _is_object, has_attr, _has_tag, _flags = root_header
    if sexp_type != VECSXP:
        raise UnsupportedRDS("root object is not a data.frame or list of data.frames")
    count = reader.length()
    if count == 0:
        raise UnsupportedRDS("root object is empty")
    if max_root_items is not None and count > max_root_items:
        raise RDSLimitError(
            f"root object contains {count:,} items; configured limit is "
            f"{max_root_items:,}"
        )

    stage_root = Path(tempfile.mkdtemp(prefix=".rdsframe-", dir=out_dir))
    connection = duckdb.connect()
    try:
        if memory_limit:
            connection.execute(f"SET memory_limit = {_sql_literal(memory_limit)}")
        duck_temp = temp_directory or (stage_root / "duckdb-temp")
        duck_temp.mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory = {_sql_literal(str(duck_temp))}")

        first_header = reader.flags()
        if selected_indices is not None and 0 not in selected_indices:
            first_frame = scan_dataframe_from_header(reader, first_header)
            if first_frame is None:
                raise RDSCatalogError(
                    "the RDS contains one root data.frame; table index 0 must be selected"
                )
            _validate_selected_indices(selected_indices, count)
            if max_tables is not None and count > max_tables:
                raise RDSLimitError(
                    f"RDS contains {count:,} data.frames; configured limit is "
                    f"{max_tables:,}. No partial output was written."
                )
            return _convert_frame_list(
                reader,
                None,
                first_scanned=True,
                selected_indices=selected_indices,
                selected_names=selected_names,
                count=count,
                root_has_attr=has_attr,
                connection=connection,
                stage_root=stage_root,
                out_dir=out_dir,
                basename=basename,
                compression=compression,
                row_group_size=row_group_size,
                stage_max_columns=stage_max_columns,
                stage_max_bytes=stage_max_bytes,
                gc_collect_every=gc_collect_every,
                posixct_mode=posixct_mode,
                invalid_timestamp=invalid_timestamp,
                list_column_mode=list_column_mode,
                safe_name=safe_name,
                unique_name=unique_name,
            )
        if first_header[0] == VECSXP:
            # Could be a named list of data.frames or a data.frame list-column.
            # Parse only this ambiguous first child; ordinary flat data.frames
            # remain strictly one-column-at-a-time.
            first = reader.read_item_from_header(first_header)
            if _is_dataframe(first):
                _validate_selected_indices(selected_indices, count)
                if max_tables is not None and count > max_tables:
                    raise RDSLimitError(
                        f"RDS contains {count:,} data.frames; configured limit is "
                        f"{max_tables:,}. No partial output was written."
                    )
                results = _convert_frame_list(
                    reader,
                    first,
                    first_scanned=False,
                    selected_indices=selected_indices,
                    selected_names=selected_names,
                    count=count,
                    root_has_attr=has_attr,
                    connection=connection,
                    stage_root=stage_root,
                    out_dir=out_dir,
                    basename=basename,
                    compression=compression,
                    row_group_size=row_group_size,
                    stage_max_columns=stage_max_columns,
                    stage_max_bytes=stage_max_bytes,
                    gc_collect_every=gc_collect_every,
                    posixct_mode=posixct_mode,
                    invalid_timestamp=invalid_timestamp,
                    list_column_mode=list_column_mode,
                    safe_name=safe_name,
                    unique_name=unique_name,
                )
                return results
            first_column = first
        else:
            first_column = reader.read_item_from_header(first_header)

        if selected_indices is not None and selected_indices != frozenset({0}):
            raise RDSCatalogError(
                "the RDS contains one root data.frame; only table index 0 is valid"
            )

        stage = stage_root / "table"
        stage.mkdir()
        staged, row_count = _stage_columns(
            reader,
            connection,
            stage,
            count=count,
            first_column=first_column,
            stage_max_columns=stage_max_columns,
            stage_max_bytes=stage_max_bytes,
            gc_collect_every=gc_collect_every,
            posixct_mode=posixct_mode,
            invalid_timestamp=invalid_timestamp,
            list_column_mode=list_column_mode,
        )
        attributes = reader.read_attributes() if has_attr else {}
        if "data.frame" not in as_strings(attributes.get("class")):
            raise UnsupportedRDS("root VECSXP is not a data.frame")
        names = _column_names(attributes, count, unique_name)
        final = out_dir / f"{basename}.parquet"
        _merge_columns(
            connection,
            staged,
            names,
            final,
            compression=compression,
            row_group_size=row_group_size,
        )
        return [("data", final, row_count, count)]
    finally:
        connection.close()
        shutil.rmtree(stage_root, ignore_errors=True)


def _convert_frame_list(
    reader: Reader,
    first: SerializedObject | None,
    *,
    first_scanned: bool,
    selected_indices: frozenset[int] | None,
    selected_names: dict[int, str] | None,
    count: int,
    root_has_attr: bool,
    connection: Any,
    stage_root: Path,
    out_dir: Path,
    basename: str,
    compression: str,
    row_group_size: int,
    stage_max_columns: int,
    stage_max_bytes: int,
    gc_collect_every: int,
    posixct_mode: PosixctMode,
    invalid_timestamp: InvalidTimestampMode,
    list_column_mode: ListColumnMode,
    safe_name: Any,
    unique_name: Any,
) -> list[tuple[str, Path, int, int]]:
    pending: list[tuple[int, Path, int, int]] = []
    last_selected = max(selected_indices) if selected_indices else None
    for table_index in range(count):
        selected = selected_indices is None or table_index in selected_indices
        table_stage = stage_root / f"table-{table_index}"
        if selected:
            table_stage.mkdir()
        if table_index == 0:
            if first_scanned:
                if selected_names is not None and table_index == last_selected:
                    break
                continue
            if first is None:
                raise UnsupportedRDS("first data.frame payload is unavailable")
            if not selected:
                continue
            columns = first.value
            column_count = len(columns)
            staged, rows = _stage_materialized_columns(
                connection,
                table_stage,
                columns,
                stage_max_columns=stage_max_columns,
                stage_max_bytes=stage_max_bytes,
                gc_collect_every=gc_collect_every,
                posixct_mode=posixct_mode,
                invalid_timestamp=invalid_timestamp,
                list_column_mode=list_column_mode,
            )
            attributes = first.attributes
        else:
            header = reader.flags()
            if header[0] != VECSXP:
                raise UnsupportedRDS("root list contains a non-data.frame element")
            if not selected:
                if scan_dataframe_from_header(reader, header) is None:
                    raise UnsupportedRDS("root list contains a non-data.frame VECSXP")
                continue
            column_count = reader.length()
            staged, rows = _stage_columns(
                reader,
                connection,
                table_stage,
                count=column_count,
                first_column=None,
                stage_max_columns=stage_max_columns,
                stage_max_bytes=stage_max_bytes,
                gc_collect_every=gc_collect_every,
                posixct_mode=posixct_mode,
                invalid_timestamp=invalid_timestamp,
                list_column_mode=list_column_mode,
            )
            attributes = reader.read_attributes() if header[2] else {}
            if "data.frame" not in as_strings(attributes.get("class")):
                raise UnsupportedRDS("root list contains a non-data.frame VECSXP")
        names = _column_names(attributes, column_count, unique_name)
        pending_path = stage_root / f"merged-{table_index}.parquet"
        _merge_columns(
            connection,
            staged,
            names,
            pending_path,
            compression=compression,
            row_group_size=row_group_size,
        )
        pending.append((table_index, pending_path, rows, column_count))

        if selected_names is not None and table_index == last_selected:
            break

    if selected_names is None:
        root_attributes = reader.read_attributes() if root_has_attr else {}
        table_names = as_strings(root_attributes.get("names"))
    else:
        table_names = []
    results: list[tuple[str, Path, int, int]] = []
    used: dict[str, object] = {}
    for table_index, pending_path, rows, columns in pending:
        raw_name = (
            selected_names.get(table_index, "")
            if selected_names is not None
            else (table_names[table_index] if table_index < len(table_names) else "")
        )
        name = unique_name(safe_name(raw_name, table_index), used)
        used[name] = object()
        final = out_dir / f"{basename}__{name}.parquet"
        os.replace(pending_path, final)
        results.append((name, final, rows, columns))
    return results


def _validate_selected_indices(
    selected_indices: frozenset[int] | None, table_count: int
) -> None:
    if selected_indices is None:
        return
    invalid = sorted(index for index in selected_indices if index < 0 or index >= table_count)
    if invalid:
        raise RDSCatalogError(
            f"table indices out of range for {table_count} tables: {invalid}"
        )


def _stage_materialized_columns(
    connection: Any,
    stage: Path,
    columns: list[Any],
    *,
    stage_max_columns: int,
    stage_max_bytes: int,
    gc_collect_every: int,
    posixct_mode: PosixctMode,
    invalid_timestamp: InvalidTimestampMode,
    list_column_mode: ListColumnMode,
) -> tuple[list[StagedColumns], int]:
    staged: list[StagedColumns] = []
    rows: int | None = None
    batch: list[tuple[int, Any]] = []
    batch_bytes = 0
    next_gc = gc_collect_every if gc_collect_every > 0 else 0
    for index, column in enumerate(columns):
        array = _column_to_arrow(
            column,
            posixct_mode=posixct_mode,
            invalid_timestamp=invalid_timestamp,
            list_column_mode=list_column_mode,
        )
        rows = _validate_rows(rows, len(array))
        array_bytes = _array_nbytes(array)
        if _batch_is_full(
            batch,
            batch_bytes,
            array_bytes,
            max_columns=stage_max_columns,
            max_bytes=stage_max_bytes,
        ):
            staged.append(_write_columns(connection, stage, len(staged), batch))
            batch = []
            batch_bytes = 0
            next_gc = _maybe_collect(
                index, gc_collect_every, next_gc
            )
        batch.append((index, array))
        batch_bytes += array_bytes
        columns[index] = None
        del column
    if batch:
        staged.append(_write_columns(connection, stage, len(staged), batch))
        batch = []
        _maybe_collect(
            len(columns), gc_collect_every, next_gc
        )
    return staged, rows or 0


def _stage_columns(
    reader: Reader,
    connection: Any,
    stage: Path,
    *,
    count: int,
    first_column: Any | None,
    stage_max_columns: int,
    stage_max_bytes: int,
    gc_collect_every: int,
    posixct_mode: PosixctMode,
    invalid_timestamp: InvalidTimestampMode,
    list_column_mode: ListColumnMode,
) -> tuple[list[StagedColumns], int]:
    staged: list[StagedColumns] = []
    rows: int | None = None
    batch: list[tuple[int, Any]] = []
    batch_bytes = 0
    next_gc = gc_collect_every if gc_collect_every > 0 else 0
    for index in range(count):
        column = first_column if index == 0 and first_column is not None else reader.read_item()
        array = _column_to_arrow(
            column,
            posixct_mode=posixct_mode,
            invalid_timestamp=invalid_timestamp,
            list_column_mode=list_column_mode,
        )
        rows = _validate_rows(rows, len(array))
        array_bytes = _array_nbytes(array)
        if _batch_is_full(
            batch,
            batch_bytes,
            array_bytes,
            max_columns=stage_max_columns,
            max_bytes=stage_max_bytes,
        ):
            staged.append(_write_columns(connection, stage, len(staged), batch))
            batch = []
            batch_bytes = 0
            next_gc = _maybe_collect(
                index, gc_collect_every, next_gc
            )
        batch.append((index, array))
        batch_bytes += array_bytes
        del column
        reader.tick()
    if batch:
        staged.append(_write_columns(connection, stage, len(staged), batch))
        batch = []
        _maybe_collect(
            count, gc_collect_every, next_gc
        )
    return staged, rows or 0


def _write_columns(
    connection: Any,
    stage: Path,
    batch_index: int,
    columns: list[tuple[int, Any]],
) -> StagedColumns:
    import pyarrow as pa  # type: ignore[import-untyped]

    path = stage / f"columns-{batch_index}.parquet"
    table = pa.table({f"c{index}": array for index, array in columns})
    connection.register("_rdsframe_columns", table)
    try:
        connection.execute(
            f"COPY _rdsframe_columns TO {_sql_literal(str(path))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.unregister("_rdsframe_columns")
    return StagedColumns(path, tuple(index for index, _array in columns))


def _batch_is_full(
    batch: list[tuple[int, Any]],
    current_bytes: int,
    next_bytes: int,
    *,
    max_columns: int,
    max_bytes: int,
) -> bool:
    return bool(batch) and (
        len(batch) >= max_columns or current_bytes + next_bytes > max_bytes
    )


def _array_nbytes(array: Any) -> int:
    size = getattr(array, "nbytes", 0)
    return max(0, int(size))


def _maybe_collect(
    processed_columns: int,
    interval: int,
    next_at: int,
) -> int:
    if interval > 0 and next_at > 0 and processed_columns >= next_at:
        gc.collect()
        while next_at <= processed_columns:
            next_at += interval
    return next_at


def _column_to_arrow(
    column: Any,
    *,
    posixct_mode: PosixctMode = "preserve",
    invalid_timestamp: InvalidTimestampMode = "error",
    list_column_mode: ListColumnMode = "infer",
) -> Any:
    """Map a parsed R vector to Arrow, reusing native buffers where possible."""
    import pyarrow as pa

    if not isinstance(column, SerializedObject):
        return pa.array(column)
    value, attributes, sexp_type = column.value, column.attributes, column.sexp_type
    if attributes.get("dim") is not None:
        raise UnsupportedRDS(
            "matrix- or array-valued data.frame columns are not supported"
        )
    classes = as_strings(attributes.get("class"))
    if sexp_type == INTSXP:
        mask = value == NA_INTEGER
        levels = attributes.get("levels")
        if "factor" in classes and levels is not None:
            codes = value.astype(np.int64) - 1
            codes[mask] = 0
            indices = pa.array(codes, mask=mask)
            return pa.DictionaryArray.from_arrays(
                indices, pa.array(as_strings(levels)), ordered="ordered" in classes
            )
        if "Date" in classes:
            clean = value.copy()
            clean[mask] = 0
            return pa.array(clean, mask=mask, type=pa.date32())
        return pa.array(value, mask=mask)
    if sexp_type == REALSXP:
        if "Date" in classes:
            mask = np.isnan(value)
            clean = np.nan_to_num(value).astype(np.int32)
            return pa.array(clean, mask=mask, type=pa.date32())
        if "difftime" in classes:
            unit = next(iter(as_strings(attributes.get("units"))), "secs")
            seconds_per_unit = DIFFTIME_SECONDS_PER_UNIT.get(unit)
            if seconds_per_unit is None:
                raise UnsupportedRDS(f"difftime unit {unit!r} is not supported")
            mask = np.isnan(value)
            micros = np.zeros(len(value), dtype=np.int64)
            micros[~mask] = np.rint(value[~mask] * seconds_per_unit * 1_000_000).astype(
                np.int64
            )
            return pa.array(micros, mask=mask, type=pa.duration("us"))
        if "POSIXct" in classes:
            null_mask = np.isnan(value)
            finite = np.isfinite(value)
            max_seconds = np.iinfo(np.int64).max / 1_000_000
            representable = finite & (np.abs(value) <= max_seconds)
            invalid = ~null_mask & ~representable
            if invalid_timestamp == "error" and np.any(invalid):
                raise UnsupportedRDS(
                    "POSIXct contains infinite or out-of-range values; "
                    "use invalid_timestamp='null' to coerce them explicitly"
                )
            micros = np.zeros(len(value), dtype=np.int64)
            micros[representable] = np.rint(
                value[representable] * 1_000_000
            ).astype(np.int64)
            timezone = next(iter(as_strings(attributes.get("tzone"))), None)
            arrow_timezone = timezone if posixct_mode == "preserve" else None
            return pa.array(
                micros,
                mask=~representable,
                type=pa.timestamp("us", tz=arrow_timezone or None),
            )
        return pa.array(value)
    if sexp_type == LGLSXP:
        return pa.array(value == 1, mask=value == NA_INTEGER)
    if sexp_type == STRSXP:
        return value if value.__class__.__module__.startswith("pyarrow") else pa.array(value)
    if sexp_type == RAWSXP:
        return pa.array(value, type=pa.uint8())
    if sexp_type == CPLXSXP:
        return pa.StructArray.from_arrays(
            [pa.array(value.real), pa.array(value.imag)],
            names=["real", "imag"],
        )
    if sexp_type == VECSXP:
        if "POSIXlt" in classes:
            from .api import _posixlt_to_pandas  # local import: api.py imports this module

            return pa.array(_posixlt_to_pandas(column))
        values = [_element_to_python(item) for item in value]
        if list_column_mode == "json":
            return pa.array([_to_json(item) for item in values], type=pa.string())
        if list_column_mode == "string":
            return pa.array(
                [None if item is None else str(item) for item in values],
                type=pa.string(),
            )
        try:
            return pa.array(values)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
            raise UnsupportedRDS(
                "heterogeneous list-column cannot be represented losslessly by Arrow; "
                "use list_column_mode='json' or 'string' explicitly"
            ) from exc
    raise UnsupportedRDS(f"Parquet export does not support SEXP column type {sexp_type}")


def _element_to_python(element: Any) -> Any:
    value = as_value(element)
    if value.__class__.__module__.startswith("pyarrow") and hasattr(value, "to_pylist"):
        value = value.to_pylist()
    if isinstance(value, np.ndarray):
        return value[0].item() if value.size == 1 else value.tolist()
    if isinstance(value, list):
        return value[0] if len(value) == 1 else value
    return value


def _to_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_compatible(value: Any) -> Any:
    if value.__class__.__module__.startswith("pyarrow") and hasattr(value, "to_pylist"):
        return [_json_compatible(item) for item in value.to_pylist()]
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, bytes):
        return {"$binary_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"$float": str(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"$python_type": type(value).__name__, "$value": str(value)}


def _merge_columns(
    connection: Any,
    staged: list[StagedColumns],
    names: list[str],
    destination: Path,
    *,
    compression: str,
    row_group_size: int,
) -> None:
    if not staged:
        raise UnsupportedRDS("data.frame has no columns")
    sources = [
        f"read_parquet({_sql_literal(str(item.path))}) AS p{index}"
        for index, item in enumerate(staged)
    ]
    from_clause = " POSITIONAL JOIN ".join(sources)
    selections: list[str] = []
    for stage_index, item in enumerate(staged):
        selections.extend(
            f"p{stage_index}.c{column_index} AS {_quote_identifier(names[column_index])}"
            for column_index in item.indices
        )
    select = ", ".join(selections)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    try:
        connection.execute(
            f"COPY (SELECT {select} FROM {from_clause}) TO {_sql_literal(str(temporary))} "
            f"(FORMAT PARQUET, COMPRESSION {_quote_option(compression)}, "
            f"ROW_GROUP_SIZE {int(row_group_size)})"
        )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _column_names(attributes: dict[str, Any], count: int, unique_name: Any) -> list[str]:
    raw = as_strings(attributes.get("names"))
    if raw and len(raw) != count:
        raise UnsupportedRDS("data.frame column-name count does not match column count")
    raw = raw or [f"V{index + 1}" for index in range(count)]
    result: list[str] = []
    existing: dict[str, object] = {}
    for index, name in enumerate(raw):
        candidate = name or f"V{index + 1}"
        candidate = unique_name(candidate, existing)
        existing[candidate] = object()
        result.append(candidate)
    return result


def _validate_rows(expected: int | None, actual: int) -> int:
    if expected is not None and expected != actual:
        raise UnsupportedRDS("data.frame columns have different lengths")
    return actual


def _is_dataframe(value: Any) -> bool:
    return (
        isinstance(value, SerializedObject)
        and value.sexp_type == VECSXP
        and "data.frame" in as_strings(value.attributes.get("class"))
    )


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_option(value: str) -> str:
    normalized = value.upper()
    allowed = {"UNCOMPRESSED", "SNAPPY", "GZIP", "ZSTD", "LZ4", "BROTLI"}
    if normalized not in allowed:
        raise ValueError(f"unsupported Parquet compression: {value}")
    return normalized
