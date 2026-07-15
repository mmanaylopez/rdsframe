"""Public API for reading RDS data frames and exporting them to Parquet."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, TypeAlias, cast

import numpy as np
import pandas as pd

from ._core import (
    CPLXSXP,
    DIFFTIME_SECONDS_PER_UNIT,
    ENVSXP,
    INTSXP,
    LGLSXP,
    NA_INTEGER,
    RAWSXP,
    REALSXP,
    S4SXP,
    STRSXP,
    VECSXP,
    InvalidRDS,
    ProgressCallback,
    RDSCatalogError,
    Reader,
    ReaderLimits,
    SerializedObject,
    SkippedObject,
    UnsupportedRDS,
    as_strings,
    as_value,
    decode_header,
    is_buffer_source,
    open_rds_stream,
    posixlt_wall_clock_components,
    resolve_native_encoding,
    symbol_name,
)
from ._scan import ScannedVector, as_dataframe, scan_dataframe_from_header, scan_vector_from_header

# Sources accepted by the in-memory read functions: a filesystem path, raw
# RDS bytes, or an already-open seekable binary stream.
RDSSource: TypeAlias = (
    "os.PathLike[str] | str | bytes | bytearray | memoryview | BinaryIO"
)


@dataclass(frozen=True, slots=True)
class RFileInfo:
    path: Path
    size_bytes: int
    compression: Literal["none", "gzip", "bzip2", "xz"]
    container: Literal["rds", "rdata", "unknown"]
    serialization: Literal["xdr", "native", "ascii", "unknown"]
    fast_supported: bool


@dataclass(frozen=True, slots=True)
class ParquetTable:
    name: str
    path: Path
    rows: int
    columns: int


@dataclass(frozen=True, slots=True)
class RTableInfo:
    """Cheap structural metadata for one data.frame inside an RDS file."""

    index: int
    name: str
    rows: int | None
    columns: int
    column_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RDSCatalog:
    """Table catalog tied to a source file's path, size, and modification time."""

    path: Path
    size_bytes: int
    mtime_ns: int
    compression: str
    tables: tuple[RTableInfo, ...]

    def matches(self, path: os.PathLike[str] | str) -> bool:
        candidate = Path(path).expanduser().resolve()
        try:
            stat = candidate.stat()
        except OSError:
            return False
        return (
            candidate == self.path
            and stat.st_size == self.size_bytes
            and stat.st_mtime_ns == self.mtime_ns
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "rdsframe.catalog",
            "version": 1,
            "source": {
                "path": str(self.path),
                "size_bytes": self.size_bytes,
                "mtime_ns": self.mtime_ns,
                "compression": self.compression,
            },
            "tables": [
                {
                    "index": table.index,
                    "name": table.name,
                    "rows": table.rows,
                    "columns": table.columns,
                    "column_names": list(table.column_names),
                }
                for table in self.tables
            ],
        }

    def save(self, destination: os.PathLike[str] | str) -> Path:
        """Atomically persist the catalog for reuse across processes."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def load(cls, source: os.PathLike[str] | str) -> RDSCatalog:
        """Load a catalog; source-file freshness is checked when it is used."""
        try:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
            if payload.get("format") != "rdsframe.catalog" or payload.get("version") != 1:
                raise ValueError("unsupported catalog format or version")
            metadata = payload["source"]
            tables = tuple(
                RTableInfo(
                    int(item["index"]),
                    str(item["name"]),
                    None if item["rows"] is None else int(item["rows"]),
                    int(item["columns"]),
                    tuple(str(name) for name in item["column_names"]),
                )
                for item in payload["tables"]
            )
            return cls(
                Path(metadata["path"]),
                int(metadata["size_bytes"]),
                int(metadata["mtime_ns"]),
                str(metadata["compression"]),
                tables,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RDSCatalogError(f"invalid RDS catalog: {source}") from exc


@dataclass(frozen=True, slots=True)
class _SelectionPlan:
    indices: frozenset[int]
    names: dict[int, str] | None


def materialize_uncompressed(
    path: os.PathLike[str] | str,
    destination: os.PathLike[str] | str | None = None,
) -> Path:
    """Decompress an RDS container to disk once so later reads can seek.

    Compressed RDS (gzip/bzip2/xz) cannot seek: skipping an unselected
    payload still decompresses its bytes. Applications that repeatedly and
    selectively access one large compressed file can pay the decompression
    once, then run every later `list`/`extract`/`columns=` operation against
    the uncompressed copy where payload skipping is a real ``seek()``.

    Returns the path of the uncompressed file. If the source is already
    uncompressed it is returned unchanged (no copy is made). Without a
    ``destination`` a fresh file is created in the system temp directory;
    the caller owns its lifetime either way. An explicit destination is
    written atomically (temp file + rename).
    """
    source = _validate_source(path)
    with open_rds_stream(source) as (stream, _raw, compression):
        if compression == "none":
            return source
        if destination is None:
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f"{source.stem}.", suffix=".uncompressed.rds"
            )
            target = Path(temp_name)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    shutil.copyfileobj(stream, output, 4 * 1024 * 1024)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            return target
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
        temporary = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                shutil.copyfileobj(stream, output, 4 * 1024 * 1024)
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target


def inspect_r_file(path: os.PathLike[str] | str) -> RFileInfo:
    """Inspect compression and serialization headers without parsing the object."""
    source = _validate_source(path)
    with source.open("rb") as raw:
        magic = raw.read(6)
    if magic.startswith(b"\x1f\x8b"):
        compression = "gzip"
    elif magic.startswith(b"BZh"):
        compression = "bzip2"
    elif magic.startswith(b"\xfd7zXZ\x00"):
        compression = "xz"
    else:
        compression = "none"
    with open_rds_stream(source) as (stream, _raw, _compression):
        head = stream.read(5)
    if head.startswith(b"RDX"):
        container, serialization = "rdata", "xdr"
    elif head.startswith(b"RDB"):
        container, serialization = "rdata", "native"
    elif head.startswith(b"RDA"):
        container, serialization = "rdata", "ascii"
    elif head.startswith(b"X\n"):
        container, serialization = "rds", "xdr"
    elif head.startswith(b"B\n"):
        container, serialization = "rds", "native"
    elif head.startswith((b"A\n", b"A\r\n")):
        container, serialization = "rds", "ascii"
    else:
        container, serialization = "unknown", "unknown"
    return RFileInfo(
        source,
        source.stat().st_size,
        compression,  # type: ignore[arg-type]
        container,  # type: ignore[arg-type]
        serialization,  # type: ignore[arg-type]
        container == "rds" and serialization in {"xdr", "native"},
    )


def list_rds_tables(
    path: os.PathLike[str] | str,
    *,
    progress: ProgressCallback | None = None,
    limits: ReaderLimits | None = None,
) -> RDSCatalog:
    """List data.frames without allocating their vector payloads.

    The stream is still traversed because R stores list names after its elements.
    Numeric, raw, complex, and character payloads are discarded with bounded
    memory; no pandas, NumPy column, Arrow table, or temporary Parquet is created.
    """
    source = _validate_source(path).resolve()
    stat = source.stat()
    file_info = inspect_r_file(source)
    _emit_progress(progress, 0)
    with open_rds_stream(source) as (stream, raw, _compression):
        version, byteorder, _encoding = decode_header(stream)
        if version not in {2, 3}:
            raise UnsupportedRDS(f"serialization version {version} is not supported")
        reader = Reader(
            stream,
            byteorder=byteorder,
            limits=limits or ReaderLimits(),
            progress=progress,
            total_bytes=stat.st_size,
            compressed_position=raw.tell,
            seekable_discard=stream is raw,
        )
        root_header = reader.flags()
        if root_header[0] != VECSXP:
            raise UnsupportedRDS(
                "root object is not a data.frame or list of data.frames; "
                "use read_r_object() to read it as a general R value"
            )
        root_count = reader.length()
        if root_count == 0:
            raise UnsupportedRDS("root object is empty")

        first_scan = scan_vector_from_header(reader, reader.flags())
        first_frame = as_dataframe(first_scan)
        if first_frame is not None:
            frames = [first_frame]
            for _index in range(1, root_count):
                frame = scan_dataframe_from_header(reader, reader.flags())
                if frame is None:
                    raise UnsupportedRDS("root list contains non-data.frame elements")
                frames.append(frame)
            root_attributes = reader.read_attributes() if root_header[2] else {}
            names = as_strings(root_attributes.get("names"))
            table_info = tuple(
                RTableInfo(
                    index,
                    names[index] if index < len(names) and names[index] else f"table_{index + 1}",
                    frame.rows,
                    frame.columns,
                    frame.column_names,
                )
                for index, frame in enumerate(frames)
            )
        else:
            first_length = first_scan.summary.length
            for _index in range(1, root_count):
                reader.skip_item()
            root_attributes = reader.read_attributes() if root_header[2] else {}
            root_scan = ScannedVector(
                summary=SkippedObject(VECSXP, root_count),
                attributes=root_attributes,
                first_element_length=first_length,
            )
            frame = as_dataframe(root_scan)
            if frame is None:
                raise UnsupportedRDS(
                    "root VECSXP is not a data.frame; "
                    "use read_r_object() to read it as a general R value"
                )
            table_info = (
                RTableInfo(0, "data", frame.rows, frame.columns, frame.column_names),
            )
    _emit_progress(progress, 100)
    return RDSCatalog(
        source,
        stat.st_size,
        stat.st_mtime_ns,
        file_info.compression,
        table_info,
    )


def read_rds(
    path: RDSSource,
    *,
    progress: ProgressCallback | None = None,
    limits: ReaderLimits | None = None,
    strings: Literal["object", "string", "pyarrow"] = "object",
    columns: Sequence[int | str] | None = None,
    encoding: str | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Read a data.frame or a named list of data.frames directly into pandas.

    Atomic numeric vectors are filled directly in their final NumPy allocation,
    avoiding a second full-column bytes buffer. This is memory-efficient parsing,
    not row streaming: a returned DataFrame must still fit in memory.

    ``columns`` restricts parsing to a subset of a single data.frame, by
    zero-based index or exact name. Unselected columns are structurally
    skipped (never allocated as NumPy/pandas objects), which is the main lever
    for cutting time and memory on wide tables when only a few fields are
    needed. It requires the RDS root to be a single data.frame; use
    :func:`extract_rds_tables` or :func:`to_parquet` first to narrow a
    multi-table RDS down to one table.

    ``encoding`` overrides the codec used for CHARSXP elements with no
    explicit UTF-8/ASCII/latin-1 flag (R's "native encoding" strings). By
    default a version-3 RDS header's own declared encoding is used when
    present and recognized, falling back to UTF-8 otherwise; pass e.g.
    ``encoding="windows-1252"`` for older Windows-authored files where that
    default guess is wrong.

    ``path`` may also be a bytes-like object or a seekable binary stream
    (e.g. RDS content received over HTTP), read without touching disk.
    """
    source, total_bytes = _coerce_read_source(path)
    if strings not in {"object", "string", "pyarrow"}:
        raise ValueError("strings must be 'object', 'string', or 'pyarrow'")
    if columns is not None:
        return _read_dataframe_selective(
            source,
            total_bytes=total_bytes,
            progress=progress,
            limits=limits,
            strings=strings,
            columns=columns,
            encoding=encoding,
        )
    _emit_progress(progress, 0)
    with open_rds_stream(source) as (stream, raw, _compression):
        version, byteorder, declared_encoding = decode_header(stream)
        if version not in {2, 3}:
            raise UnsupportedRDS(f"serialization version {version} is not supported")
        reader = Reader(
            stream,
            byteorder=byteorder,
            limits=limits or ReaderLimits(),
            progress=progress,
            total_bytes=total_bytes,
            compressed_position=raw.tell,
            arrow_strings=strings == "pyarrow",
            native_encoding=resolve_native_encoding(declared_encoding, encoding),
            utf8_fallback=encoding,
        )
        root = reader.read_item()
    result = _root_to_frames(root, strings=strings)
    _emit_progress(progress, 100)
    return result


def _read_dataframe_selective(
    source: Any,
    *,
    total_bytes: int,
    progress: ProgressCallback | None,
    limits: ReaderLimits | None,
    strings: Literal["object", "string", "pyarrow"],
    columns: Sequence[int | str],
    encoding: str | None = None,
) -> pd.DataFrame:
    if isinstance(columns, (str, bytes)):
        raise TypeError("columns must be a sequence of column names or integer indices")
    requested = list(columns)
    if not requested:
        raise ValueError("columns cannot be empty")
    effective_limits = limits or ReaderLimits()
    column_names: tuple[str, ...] | None = None
    if any(isinstance(item, str) for item in requested):
        column_names = _scan_single_dataframe_column_names(
            source, effective_limits, total_bytes
        )

    _emit_progress(progress, 0)
    with open_rds_stream(source) as (stream, raw, _compression):
        version, byteorder, declared_encoding = decode_header(stream)
        if version not in {2, 3}:
            raise UnsupportedRDS(f"serialization version {version} is not supported")
        reader = Reader(
            stream,
            byteorder=byteorder,
            limits=effective_limits,
            progress=progress,
            total_bytes=total_bytes,
            compressed_position=raw.tell,
            arrow_strings=strings == "pyarrow",
            seekable_discard=stream is raw,
            native_encoding=resolve_native_encoding(declared_encoding, encoding),
            utf8_fallback=encoding,
        )
        root_header = reader.flags()
        if root_header[0] != VECSXP:
            raise UnsupportedRDS("root object is not a data.frame")
        count = reader.length()
        if count == 0:
            raise UnsupportedRDS("root object is empty")

        order = _resolve_column_selection(requested, column_names, count)
        wanted = set(order)
        raw_columns: list[Any] = [None] * count
        for index in range(count):
            if index in wanted:
                raw_columns[index] = reader.read_item()
            else:
                reader.skip_item()

        attributes = reader.read_attributes() if root_header[2] else {}
        if "data.frame" not in as_strings(attributes.get("class")):
            raise UnsupportedRDS(
                "root object is not a single data.frame; columns= requires a "
                "single data.frame root (use extract_rds_tables/to_parquet to "
                "select one table first)"
            )
        names = as_strings(attributes.get("names"))
        if names and len(names) != count:
            raise InvalidRDS("data.frame column-name count does not match column count")

        data: dict[str, Any] = {}
        for index in order:
            name = names[index] if index < len(names) and names[index] else f"V{index + 1}"
            key = _unique_name(name, data)
            data[key] = _column_to_pandas(raw_columns[index], strings=strings)
        frame = pd.DataFrame(data, copy=False)
        frame = _apply_row_names(frame, attributes)
    _emit_progress(progress, 100)
    return frame


def _scan_single_dataframe_column_names(
    source: Any, limits: ReaderLimits, total_bytes: int
) -> tuple[str, ...]:
    """One structural pass to resolve column names to indices (no payloads)."""
    with open_rds_stream(source) as (stream, raw, _compression):
        version, byteorder, _encoding = decode_header(stream)
        if version not in {2, 3}:
            raise UnsupportedRDS(f"serialization version {version} is not supported")
        reader = Reader(
            stream,
            byteorder=byteorder,
            limits=limits,
            total_bytes=total_bytes,
            compressed_position=raw.tell,
            seekable_discard=stream is raw,
        )
        frame = scan_dataframe_from_header(reader, reader.flags())
    if frame is None:
        raise UnsupportedRDS(
            "root object is not a single data.frame; column-name selection "
            "requires a single data.frame root (use integer indices, or select "
            "one table first with extract_rds_tables/to_parquet)"
        )
    return frame.column_names


def _resolve_column_selection(
    requested: list[int | str],
    column_names: tuple[str, ...] | None,
    count: int,
) -> list[int]:
    resolved: list[int] = []
    seen: set[int] = set()
    for selector in requested:
        if isinstance(selector, bool):
            raise TypeError("boolean column selectors are not valid indices")
        if isinstance(selector, int):
            index = selector
            if index < 0 or index >= count:
                raise ValueError(f"column index out of range for {count} columns: {index}")
        elif isinstance(selector, str):
            if column_names is None:  # pragma: no cover - defensive
                raise ValueError("column name selection requires resolved column names")
            matches = [i for i, name in enumerate(column_names) if name == selector]
            if not matches:
                raise ValueError(f"column name not found: {selector!r}")
            if len(matches) > 1:
                raise ValueError(f"column name is ambiguous: {selector!r}; use an integer index")
            index = matches[0]
        else:
            raise TypeError(f"unsupported column selector: {selector!r}")
        if index not in seen:
            seen.add(index)
            resolved.append(index)
    return resolved


def read_rds_dataframe(
    path: RDSSource,
    progress: ProgressCallback | None = None,
    *,
    limits: ReaderLimits | None = None,
    strings: Literal["object", "string", "pyarrow"] = "object",
    columns: Sequence[int | str] | None = None,
    encoding: str | None = None,
) -> pd.DataFrame:
    """Compatibility helper returning the first data.frame in an RDS file."""
    result = read_rds(
        path,
        progress=progress,
        limits=limits,
        strings=strings,
        columns=columns,
        encoding=encoding,
    )
    return result if isinstance(result, pd.DataFrame) else next(iter(result.values()))


def read_r_object(
    path: RDSSource,
    *,
    progress: ProgressCallback | None = None,
    limits: ReaderLimits | None = None,
    strings: Literal["object", "string", "pyarrow"] = "object",
    encoding: str | None = None,
) -> Any:
    """Read any supported R value, not only a data.frame or list of them.

    Many real RDS files are not tabular at all (an R help-alias index, an
    ``R CMD check`` result list, a plain nested list of vectors, ...); using
    :func:`read_rds` on them raises :class:`UnsupportedRDS` by design. This
    function recursively converts whatever the file contains into native
    Python: ``NULL`` becomes ``None``, an R list becomes a ``dict`` (if
    named) or a ``list`` (if not), a nested ``data.frame`` still becomes a
    pandas ``DataFrame``, and atomic vectors follow the same type rules as a
    data.frame column (factor, ``Date``, ``POSIXct``, ``difftime``, NA
    handling) but are unwrapped to a plain Python scalar or list rather than
    a pandas Series, and a length-1 result is returned as a bare scalar.

    Object types outside R's data-representation model -- environments,
    closures, S4/R6 objects, promises, language calls, external pointers --
    still are not supported and raise :class:`UnsupportedRDS` or
    :class:`InvalidRDS`, matching the parser's overall "fail explicitly"
    contract. This is a general-purpose, exploratory reader, not a
    performance-tuned path: it is not streaming and materializes the whole
    structure, like :func:`read_rds`.

    ``path`` may also be a bytes-like object or a seekable binary stream.
    """
    source, total_bytes = _coerce_read_source(path)
    if strings not in {"object", "string", "pyarrow"}:
        raise ValueError("strings must be 'object', 'string', or 'pyarrow'")
    _emit_progress(progress, 0)
    with open_rds_stream(source) as (stream, raw, _compression):
        version, byteorder, declared_encoding = decode_header(stream)
        if version not in {2, 3}:
            raise UnsupportedRDS(f"serialization version {version} is not supported")
        reader = Reader(
            stream,
            byteorder=byteorder,
            limits=limits or ReaderLimits(),
            progress=progress,
            total_bytes=total_bytes,
            compressed_position=raw.tell,
            arrow_strings=strings == "pyarrow",
            native_encoding=resolve_native_encoding(declared_encoding, encoding),
            utf8_fallback=encoding,
        )
        root = reader.read_item()
    result = _serialized_to_python(root, strings=strings)
    _emit_progress(progress, 100)
    return result


def _serialized_to_python(
    node: Any, *, strings: Literal["object", "string", "pyarrow"]
) -> Any:
    if node is None:
        return None
    if isinstance(node, tuple) and len(node) == 2:
        kind, payload = node
        if kind == "sym":
            name = symbol_name(node)
            return name if name else payload
        if kind == "extptr":
            raise UnsupportedRDS("external pointers cannot be represented in Python")
        if kind == "r_env":
            return {"$r_environment": payload}
        if kind in {"r_namespace", "r_package"}:
            return {f"${kind[2:]}": payload}
        if kind == "environment":  # pragma: no cover - defensive
            return payload
    if isinstance(node, SerializedObject) and node.sexp_type == ENVSXP:
        return {
            key: _serialized_to_python(item, strings=strings)
            for key, item in node.value.items()
        }
    if isinstance(node, SerializedObject) and node.sexp_type == S4SXP:
        result: dict[str, Any] = {
            "$r_class": as_strings(node.attributes.get("class")) or None
        }
        for key, item in node.attributes.items():
            if key != "class":
                result[key] = _serialized_to_python(item, strings=strings)
        return result
    if isinstance(node, SerializedObject) and node.sexp_type == VECSXP:
        if _is_dataframe(node):
            return _to_dataframe(node, strings=strings)
        if "POSIXlt" in as_strings(node.attributes.get("class")):
            return _vector_to_python(node, strings=strings)
        names = (
            as_strings(node.attributes["names"]) if "names" in node.attributes else []
        )
        items = [_serialized_to_python(item, strings=strings) for item in node.value]
        if names and len(names) == len(items):
            result: dict[str, Any] = {}
            for index, (name, item) in enumerate(zip(names, items, strict=True)):
                key = _unique_name(name or f"_unnamed_{index + 1}", result)
                result[key] = item
            return result
        return items
    if isinstance(node, SerializedObject):
        return _vector_to_python(node, strings=strings)
    if isinstance(node, np.ndarray):
        return node.tolist()[0] if node.size == 1 else node.tolist()
    return node


def _vector_to_python(
    column: SerializedObject, *, strings: Literal["object", "string", "pyarrow"]
) -> Any:
    dim = column.attributes.get("dim")
    if dim is not None:
        stripped = SerializedObject(
            column.value,
            {key: value for key, value in column.attributes.items() if key != "dim"},
            column.sexp_type,
        )
        flat = _vector_to_python(stripped, strings=strings)
        shape = tuple(int(size) for size in as_value(dim).tolist())
        flat_list = flat if isinstance(flat, list) else [flat]
        return np.array(flat_list, dtype=object).reshape(shape, order="F")
    series = pd.Series(_column_to_pandas(column, strings=strings))
    values = series.tolist()
    return values[0] if len(values) == 1 else values


def to_parquet(
    path: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    *,
    basename: str | None = None,
    compression: str = "zstd",
    row_group_size: int = 250_000,
    memory_limit: str | None = "1GB",
    temp_directory: os.PathLike[str] | str | None = None,
    stage_max_columns: int = 16,
    stage_max_bytes: int = 128 * 1024 * 1024,
    gc_collect_every: int = 16,
    max_tables: int | None = None,
    max_root_items: int | None = None,
    posixct_mode: Literal["preserve", "utc_naive"] = "preserve",
    invalid_timestamp: Literal["error", "null"] = "error",
    list_column_mode: Literal["infer", "json", "string"] = "infer",
    tables: Sequence[int | str] | None = None,
    catalog: RDSCatalog | None = None,
    progress: ProgressCallback | None = None,
    limits: ReaderLimits | None = None,
    encoding: str | None = None,
) -> list[ParquetTable]:
    """Incrementally convert RDS data.frames to query-ready Parquet.

    Columns are parsed and staged one at a time, then DuckDB combines them into
    the final file. Peak parser memory is therefore tied mainly to the largest
    individual column rather than the complete table. Existing destination files
    are replaced only after a complete result has been written.

    ``encoding`` overrides the codec for CHARSXP elements with no explicit
    UTF-8/ASCII/latin-1 flag; see :func:`read_rds` for the default-resolution
    rule (the RDS header's own declared encoding, when present, else UTF-8).
    """
    source = _validate_source(path)
    if stage_max_columns < 1:
        raise ValueError("stage_max_columns must be at least 1")
    if stage_max_bytes < 1:
        raise ValueError("stage_max_bytes must be at least 1")
    if gc_collect_every < 0:
        raise ValueError("gc_collect_every cannot be negative")
    if max_tables is not None and max_tables < 1:
        raise ValueError("max_tables must be at least 1 or None")
    if max_root_items is not None and max_root_items < 1:
        raise ValueError("max_root_items must be at least 1 or None")
    if posixct_mode not in {"preserve", "utc_naive"}:
        raise ValueError("posixct_mode must be 'preserve' or 'utc_naive'")
    if invalid_timestamp not in {"error", "null"}:
        raise ValueError("invalid_timestamp must be 'error' or 'null'")
    if list_column_mode not in {"infer", "json", "string"}:
        raise ValueError("list_column_mode must be 'infer', 'json', or 'string'")
    selection = _resolve_table_selection(source, tables, catalog)
    out_dir = Path(destination)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(basename or source.stem, 0)
    _emit_progress(progress, 0)
    with open_rds_stream(source) as (stream, raw, _compression):
        version, byteorder, declared_encoding = decode_header(stream)
        if version not in {2, 3}:
            raise UnsupportedRDS(f"serialization version {version} is not supported")
        reader = Reader(
            stream,
            byteorder=byteorder,
            limits=limits or ReaderLimits(),
            progress=progress,
            total_bytes=source.stat().st_size,
            compressed_position=raw.tell,
            arrow_strings=True,
            seekable_discard=stream is raw,
            native_encoding=resolve_native_encoding(declared_encoding, encoding),
            utf8_fallback=encoding,
        )
        from ._parquet import stream_root_to_parquet

        raw_results = stream_root_to_parquet(
            reader,
            reader.flags(),
            out_dir=out_dir,
            basename=stem,
            compression=compression,
            row_group_size=row_group_size,
            memory_limit=memory_limit,
            temp_directory=Path(temp_directory) if temp_directory else None,
            stage_max_columns=stage_max_columns,
            stage_max_bytes=stage_max_bytes,
            gc_collect_every=gc_collect_every,
            max_tables=max_tables,
            max_root_items=max_root_items,
            posixct_mode=posixct_mode,
            invalid_timestamp=invalid_timestamp,
            list_column_mode=list_column_mode,
            selected_indices=selection.indices if selection else None,
            selected_names=selection.names if selection else None,
            safe_name=_safe_name,
            unique_name=_unique_name,
        )
    _emit_progress(progress, 100)
    return [ParquetTable(name, path, rows, columns) for name, path, rows, columns in raw_results]


def extract_rds_tables(
    path: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    tables: Sequence[int | str],
    *,
    catalog: RDSCatalog | None = None,
    **conversion_options: Any,
) -> list[ParquetTable]:
    """Convert only selected tables, identified by zero-based index or name.

    Integer-only selection can run in one pass. Name selection requires a table
    catalog; when omitted, a low-allocation catalog scan is performed first.
    Reusing the result of :func:`list_rds_tables` avoids that extra scan.
    """
    return to_parquet(
        path,
        destination,
        tables=tables,
        catalog=catalog,
        **conversion_options,
    )


def convert_rds(
    path: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
    base: str,
    progress: ProgressCallback | None = None,
    *,
    max_tables: int | None = None,
) -> list[tuple[str, str, int, int]]:
    """Backward-compatible wrapper around :func:`to_parquet`."""
    return [
        (table.name, str(table.path), table.rows, table.columns)
        for table in to_parquet(
            path,
            out_dir,
            basename=base,
            progress=progress,
            max_tables=max_tables,
        )
    ]


def _validate_source(path: os.PathLike[str] | str) -> Path:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _coerce_read_source(path_or_buffer: Any) -> tuple[Any, int]:
    """Accept a path, bytes-like object, or seekable binary stream.

    Returns the source to hand to :func:`open_rds_stream` plus its total size
    in bytes (used only for progress percentages; 0 disables them).
    """
    if isinstance(path_or_buffer, (bytes, bytearray, memoryview)):
        return path_or_buffer, len(path_or_buffer)
    if is_buffer_source(path_or_buffer):
        try:
            position = path_or_buffer.tell()
            size = path_or_buffer.seek(0, os.SEEK_END)
            path_or_buffer.seek(position)
            return path_or_buffer, int(size)
        except (OSError, AttributeError):
            return path_or_buffer, 0
    source = _validate_source(path_or_buffer)
    return source, source.stat().st_size


def _resolve_table_selection(
    path: Path,
    selectors: Sequence[int | str] | None,
    catalog: RDSCatalog | None,
) -> _SelectionPlan | None:
    if selectors is None:
        if catalog is not None and not catalog.matches(path):
            raise RDSCatalogError("catalog does not match the current source file")
        return None
    if isinstance(selectors, (str, bytes)):
        raise TypeError("tables must be a sequence of integer indices or names")
    requested = list(selectors)
    if not requested:
        raise ValueError("tables cannot be empty")
    if catalog is not None and not catalog.matches(path):
        raise RDSCatalogError("catalog is stale or belongs to a different source file")
    if any(isinstance(item, str) for item in requested) and catalog is None:
        catalog = list_rds_tables(path)

    by_name: dict[str, list[int]] = {}
    if catalog is not None:
        for table in catalog.tables:
            by_name.setdefault(table.name, []).append(table.index)

    resolved: set[int] = set()
    for selector in requested:
        if isinstance(selector, bool):
            raise TypeError("boolean table selectors are not valid indices")
        if isinstance(selector, int):
            resolved.add(selector)
            continue
        if not isinstance(selector, str):
            raise TypeError(f"unsupported table selector: {selector!r}")
        matches = by_name.get(selector, [])
        if not matches:
            raise RDSCatalogError(f"table name not found: {selector!r}")
        if len(matches) > 1:
            raise RDSCatalogError(
                f"table name is ambiguous: {selector!r}; use an integer index"
            )
        resolved.add(matches[0])
    catalog_names = (
        {table.index: table.name for table in catalog.tables}
        if catalog is not None
        else None
    )
    return _SelectionPlan(frozenset(resolved), catalog_names)


def _emit_progress(callback: ProgressCallback | None, value: int) -> None:
    if callback is not None:
        with suppress(Exception):
            callback(value)


def _root_to_frames(
    root: Any, *, strings: Literal["object", "string", "pyarrow"]
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    if _is_dataframe(root):
        return _to_dataframe(root, strings=strings)
    if not isinstance(root, SerializedObject) or root.sexp_type != VECSXP:
        raise UnsupportedRDS(
            "root object is not a data.frame or list of data.frames; "
            "use read_r_object() to read it as a general R value"
        )
    items = root.value
    if not items or not all(_is_dataframe(item) for item in items):
        raise UnsupportedRDS(
            "root list contains non-data.frame elements; "
            "use read_r_object() to read it as a general R value"
        )
    names = as_strings(root.attributes.get("names"))
    result: dict[str, pd.DataFrame] = {}
    for index, item in enumerate(items):
        name = _unique_name(_safe_name(names[index] if index < len(names) else "", index), result)
        result[name] = _to_dataframe(item, strings=strings)
    return result


def _is_dataframe(obj: Any) -> bool:
    return (
        isinstance(obj, SerializedObject)
        and obj.sexp_type == VECSXP
        and "data.frame" in as_strings(obj.attributes.get("class"))
    )


def _to_dataframe(
    obj: SerializedObject, *, strings: Literal["object", "string", "pyarrow"]
) -> pd.DataFrame:
    columns = obj.value
    names = as_strings(obj.attributes.get("names"))
    if not names:
        names = [f"V{index + 1}" for index in range(len(columns))]
    if len(names) != len(columns):
        raise InvalidRDS("data.frame column-name count does not match column count")
    data: dict[str, Any] = {}
    for index, (name, column) in enumerate(zip(names, columns, strict=True)):
        key = _unique_name(name or f"V{index + 1}", data)
        data[key] = _column_to_pandas(column, strings=strings)
    frame = pd.DataFrame(data, copy=False)
    lengths = {len(series) for series in data.values()}
    if len(lengths) > 1:
        raise InvalidRDS("data.frame columns have different lengths")
    return _apply_row_names(frame, obj.attributes)


def _apply_row_names(frame: pd.DataFrame, attributes: dict[str, Any]) -> pd.DataFrame:
    """Use R's row.names as the pandas index when they carry real information.

    R's compact encoding for the default sequential case is ``c(NA, -n)``:
    two integers, not real row labels, so the default `RangeIndex` is left
    alone. Character row names (the common case for meaningful labels) or a
    genuine non-default integer vector become the DataFrame's index instead
    of being silently discarded.
    """
    row_names = attributes.get("row.names")
    if not isinstance(row_names, SerializedObject):
        return frame
    value = row_names.value
    if row_names.sexp_type == INTSXP and isinstance(value, np.ndarray):
        if len(value) == 2 and value[0] == NA_INTEGER and value[1] <= 0:
            return frame
        if len(value) == len(frame):
            frame.index = pd.Index(value)
        return frame
    if (
        row_names.sexp_type == STRSXP
        and isinstance(value, list)
        and len(value) == len(frame)
    ):
        frame.index = pd.Index(value)
    return frame


def _column_to_pandas(
    column: Any, *, strings: Literal["object", "string", "pyarrow"]
) -> pd.Series[Any] | pd.Categorical:
    if not isinstance(column, SerializedObject):
        return pd.Series(column if isinstance(column, np.ndarray) else np.asarray(column))
    value, attributes, sexp_type = column.value, column.attributes, column.sexp_type
    if attributes.get("dim") is not None:
        raise UnsupportedRDS(
            "matrix- or array-valued data.frame columns are not supported"
        )
    classes = as_strings(attributes.get("class"))
    if sexp_type == INTSXP:
        levels = attributes.get("levels")
        if "factor" in classes and levels is not None:
            categories = as_strings(levels)
            codes = np.where(value == NA_INTEGER, -1, value.astype(np.int64) - 1).astype(
                np.int64
            )
            codes[(codes < -1) | (codes >= len(categories))] = -1
            return cast(
                pd.Categorical,
                pd.Categorical.from_codes(
                    codes, categories=pd.Index(categories), ordered="ordered" in classes
                ),
            )
        if "Date" in classes:
            numeric = value.astype(np.float64)
            numeric[value == NA_INTEGER] = np.nan
            return pd.Series(pd.to_datetime(numeric, unit="D", origin="unix"))
        if np.any(value == NA_INTEGER):
            return pd.Series(pd.array(value, dtype="Int32")).mask(value == NA_INTEGER)
        return pd.Series(value, copy=False)
    if sexp_type == REALSXP:
        if "Date" in classes:
            return pd.Series(pd.to_datetime(value, unit="D", origin="unix"))
        if "POSIXct" in classes:
            timezone = next(iter(as_strings(attributes.get("tzone"))), "")
            safe = value.copy()
            safe[~np.isfinite(safe)] = np.nan
            dates = pd.to_datetime(
                safe,
                unit="s",
                origin="unix",
                utc=bool(timezone),
                errors="coerce",
            )
            if timezone and timezone not in {"UTC", "GMT"}:
                with suppress(KeyError, ValueError):
                    dates = dates.tz_convert(timezone)
            return pd.Series(dates)
        if "difftime" in classes:
            unit = next(iter(as_strings(attributes.get("units"))), "secs")
            seconds_per_unit = DIFFTIME_SECONDS_PER_UNIT.get(unit)
            if seconds_per_unit is None:
                raise UnsupportedRDS(f"difftime unit {unit!r} is not supported")
            return pd.Series(pd.to_timedelta(value * seconds_per_unit, unit="s"))
        return pd.Series(value, copy=False)
    if sexp_type == LGLSXP:
        result = pd.array(value == 1, dtype="boolean")
        result[value == NA_INTEGER] = pd.NA
        return pd.Series(result)
    if sexp_type == STRSXP:
        dtype = "string[pyarrow]" if strings == "pyarrow" else (
            "string" if strings == "string" else "object"
        )
        return pd.Series(value, dtype=dtype, copy=False)
    if sexp_type == CPLXSXP:
        return pd.Series(value, copy=False)
    if sexp_type == RAWSXP:
        return pd.Series(value, dtype="uint8", copy=False)
    if sexp_type == VECSXP:
        if "POSIXlt" in classes:
            return _posixlt_to_pandas(column)
        return pd.Series([_element_to_python(item) for item in value], dtype="object")
    raise UnsupportedRDS(f"data.frame column SEXP type {sexp_type} is unsupported")


def _posixlt_to_pandas(column: SerializedObject) -> pd.Series[Any]:
    """Reconstruct wall-clock `pandas.Timestamp`s from a `POSIXlt` list.

    Each row's `year`/`mon`/`mday`/`hour`/`min`/`sec` components are combined
    directly into the timestamp they represent; no time zone conversion is
    attempted; a POSIXlt already stores the local wall-clock reading, and R's
    own `zone` component does not reliably map onto an IANA zone name pandas
    can resolve, so reproducing it faithfully (rather than guessing) is the
    safer choice. A row with a non-finite component or an invalid date/time
    (e.g. day 30 of February) becomes `NaT`, never a raised exception.
    """
    components = posixlt_wall_clock_components(column.attributes, column.value)
    year, mon, mday = components["year"], components["mon"], components["mday"]
    hour, minute, sec = components["hour"], components["min"], components["sec"]
    length = len(sec)
    timestamps: list[Any] = [pd.NaT] * length
    for index in range(length):
        if not (
            np.isfinite(year[index])
            and np.isfinite(mon[index])
            and np.isfinite(mday[index])
            and np.isfinite(hour[index])
            and np.isfinite(minute[index])
            and np.isfinite(sec[index])
        ):
            continue
        whole_seconds = int(sec[index])
        microseconds = round((sec[index] - whole_seconds) * 1_000_000)
        try:
            timestamps[index] = pd.Timestamp(
                year=int(year[index]) + 1900,
                month=int(mon[index]) + 1,
                day=int(mday[index]),
                hour=int(hour[index]),
                minute=int(minute[index]),
                second=whole_seconds,
                microsecond=microseconds,
            )
        except (ValueError, OverflowError):
            continue
    return pd.Series(pd.to_datetime(timestamps))


def _element_to_python(element: Any) -> Any:
    value = as_value(element)
    if value.__class__.__module__.startswith("pyarrow") and hasattr(value, "to_pylist"):
        value = value.to_pylist()
    if isinstance(value, np.ndarray):
        return value[0].item() if value.size == 1 else value.tolist()
    if isinstance(value, list):
        return value[0] if len(value) == 1 else value
    return value


def _safe_name(name: str, index: int) -> str:
    result = re.sub(r"[^\w.-]+", "_", str(name or ""), flags=re.UNICODE).strip("_.").lower()
    result = result or f"table_{index + 1}"
    return f"t_{result}" if result[0].isdigit() else result


def _unique_name(name: str, existing: Mapping[str, Any]) -> str:
    if name not in existing:
        return name
    number = 2
    while f"{name}_{number}" in existing:
        number += 1
    return f"{name}_{number}"
