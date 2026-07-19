"""Validation and comparison of RDS files for audit-style workflows.

Two public entry points:

- :func:`validate_rds` answers "can rdsframe read this file, and what will
  need an explicit policy?" without materializing column payloads. It runs
  the structural catalog scan (or, for non-tabular roots, a full structural
  skip) and reports issues with stable machine-readable codes.
- :func:`diff_rds` compares two RDS files. The structural tier works from
  catalogs alone -- tables, row counts, column names, storage and logical
  types, factor levels -- so it never allocates column data. The opt-in
  ``content=True`` tier reads both files as Arrow and counts differing rows
  per column, with two audit-specific equality rules: rows are compared
  positionally (RDS preserves order, so reordering is a difference), and
  float NaN equals NaN (R's ``NA_real_`` reads back as NaN, and a missing
  value has not "changed" between two files that both miss it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ._core import (
    InvalidRDS,
    RDSLimitError,
    Reader,
    ReaderLimits,
    UnsupportedRDS,
    decode_header,
    open_rds_stream,
    resolve_native_encoding,
)
from .api import (
    RFileInfo,
    RTableInfo,
    _validate_source,
    inspect_r_file,
    list_rds_tables,
)

IssueSeverity = Literal["error", "warning", "info"]

DiffKind = Literal[
    "compression_changed",
    "table_added",
    "table_removed",
    "rows_changed",
    "columns_reordered",
    "column_added",
    "column_removed",
    "type_changed",
    "levels_changed",
    "values_changed",
]


@dataclass(frozen=True, slots=True)
class RDSValidationIssue:
    """One finding from :func:`validate_rds`, with a stable ``code``."""

    severity: IssueSeverity
    code: str
    message: str
    table: str | None = None
    column: str | None = None


@dataclass(frozen=True, slots=True)
class RDSValidationReport:
    path: Path
    file: RFileInfo
    tables: tuple[RTableInfo, ...]
    issues: tuple[RDSValidationIssue, ...]

    @property
    def ok(self) -> bool:
        """True when rdsframe can read the file (warnings/info allowed)."""
        return all(issue.severity != "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[RDSValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[RDSValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "ok": self.ok,
            "compression": self.file.compression,
            "container": self.file.container,
            "serialization": self.file.serialization,
            "size_bytes": self.file.size_bytes,
            "tables": [
                {
                    "index": table.index,
                    "name": table.name,
                    "rows": table.rows,
                    "columns": table.columns,
                }
                for table in self.tables
            ],
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                    "table": issue.table,
                    "column": issue.column,
                }
                for issue in self.issues
            ],
        }


@dataclass(frozen=True, slots=True)
class RDSDiffEntry:
    """One difference between two RDS files."""

    kind: DiffKind
    table: str | None = None
    column: str | None = None
    before: str | None = None
    after: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RDSDiffReport:
    left: Path
    right: Path
    content_checked: bool
    entries: tuple[RDSDiffEntry, ...] = field(default=())

    @property
    def identical(self) -> bool:
        """No differences at the level that was actually checked."""
        return not self.entries

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": str(self.left),
            "right": str(self.right),
            "content_checked": self.content_checked,
            "identical": self.identical,
            "entries": [
                {
                    "kind": entry.kind,
                    "table": entry.table,
                    "column": entry.column,
                    "before": entry.before,
                    "after": entry.after,
                    "detail": entry.detail,
                }
                for entry in self.entries
            ],
        }


def validate_rds(
    path: Any,
    *,
    limits: ReaderLimits | None = None,
    encoding: str | None = None,
) -> RDSValidationReport:
    """Check whether rdsframe can read *path* and flag policy-sensitive columns.

    Structural only: no column payload is materialized. ``report.ok`` means
    the whole stream parsed within limits; warnings mark columns that need
    an explicit conversion policy (list columns), and info entries record
    representation notes (POSIXlt reconstruction, complex-as-struct) and
    non-tabular roots that require :func:`rdsframe.read_r_object`.
    """
    source = _validate_source(path)
    issues: list[RDSValidationIssue] = []
    tables: tuple[RTableInfo, ...] = ()
    try:
        file_info = inspect_r_file(source)
    except InvalidRDS as exc:
        # A corrupt compressed container can fail as early as the header
        # peek; the report still identifies what compression was attempted.
        issues.append(RDSValidationIssue("error", "invalid", str(exc)))
        return RDSValidationReport(
            source, _fallback_file_info(source), tables, tuple(issues)
        )

    if file_info.container == "rdata":
        issues.append(
            RDSValidationIssue(
                "error",
                "rdata-container",
                "this is an RData workspace, not an RDS file; read it with "
                "read_rdata()",
            )
        )
    elif file_info.serialization == "ascii":
        issues.append(
            RDSValidationIssue(
                "error",
                "ascii-serialization",
                "ASCII R serialization is not supported; rewrite with "
                "saveRDS(..., ascii = FALSE)",
            )
        )
    elif file_info.container == "unknown":
        issues.append(
            RDSValidationIssue(
                "error",
                "unrecognized-container",
                "not a recognizable RDS or RData stream",
            )
        )

    if issues:
        return RDSValidationReport(source, file_info, tables, tuple(issues))

    try:
        catalog = list_rds_tables(
            source, cache=False, limits=limits, encoding=encoding
        )
    except UnsupportedRDS as exc:
        issues.extend(_validate_non_tabular(source, exc, limits, encoding))
    except InvalidRDS as exc:
        issues.append(RDSValidationIssue("error", "invalid", str(exc)))
    except RDSLimitError as exc:
        issues.append(RDSValidationIssue("error", "limit", str(exc)))
    else:
        tables = catalog.tables
        for table in tables:
            issues.extend(_column_issues(table))
        # The catalog pass stops at the end of the root object without
        # checking what follows; the trailing-bytes probe needs its own
        # structural traversal.
        issues.extend(_skip_root_and_probe(source, limits, encoding))
    return RDSValidationReport(source, file_info, tables, tuple(issues))


def _fallback_file_info(source: Path) -> RFileInfo:
    """File info when the container is too corrupt for a full inspection."""
    with source.open("rb") as raw:
        magic = raw.read(6)
    if magic.startswith(b"\x1f\x8b"):
        compression = "gzip"
    elif magic.startswith(b"BZh"):
        compression = "bzip2"
    elif magic.startswith(b"\xfd7zXZ\x00"):
        compression = "xz"
    elif magic.startswith(b"\x28\xb5\x2f\xfd"):
        compression = "zstd"
    else:
        compression = "none"
    return RFileInfo(
        source,
        source.stat().st_size,
        compression,  # type: ignore[arg-type]
        "unknown",
        "unknown",
        False,
    )


def _skip_root_and_probe(
    source: Path,
    limits: ReaderLimits | None,
    encoding: str | None,
) -> list[RDSValidationIssue]:
    """Structurally skip the whole root object and probe for trailing bytes.

    Traverses every byte with bounded memory, so truncation, bad lengths,
    and genuinely unsupported R types surface as issues. An empty result
    means the stream is exactly one clean object.
    """
    issues: list[RDSValidationIssue] = []
    try:
        with open_rds_stream(source) as (stream, raw, _compression):
            version, byteorder, declared = decode_header(stream)
            if version not in {2, 3}:
                return [
                    RDSValidationIssue(
                        "error",
                        "unsupported-version",
                        f"serialization version {version} is not supported",
                    )
                ]
            reader = Reader(
                stream,
                byteorder=byteorder,
                limits=limits or ReaderLimits(),
                total_bytes=source.stat().st_size,
                compressed_position=raw.tell,
                seekable_discard=stream is raw,
                native_encoding=resolve_native_encoding(declared, encoding),
                utf8_fallback=encoding,
            )
            reader.skip_item()
            # Trailing detection must go through the reader: the batched
            # string parser reads ahead in large chunks, so leftover bytes
            # may sit in its pending buffer rather than in the stream.
            try:
                reader.raw(1)
            except InvalidRDS:
                pass  # clean EOF exactly at the end of the root object
            else:
                issues.append(
                    RDSValidationIssue(
                        "warning",
                        "trailing-data",
                        "extra bytes follow the root object; the file may be "
                        "corrupted or concatenated",
                    )
                )
    except UnsupportedRDS as exc:
        issues.append(RDSValidationIssue("error", "unsupported", str(exc)))
    except InvalidRDS as exc:
        issues.append(RDSValidationIssue("error", "invalid", str(exc)))
    except RDSLimitError as exc:
        issues.append(RDSValidationIssue("error", "limit", str(exc)))
    return issues


def _validate_non_tabular(
    source: Path,
    original: UnsupportedRDS,
    limits: ReaderLimits | None,
    encoding: str | None,
) -> list[RDSValidationIssue]:
    """Integrity-check a file whose root is not a data.frame.

    The catalog pass stopped at a structural mismatch, which is not the same
    thing as a broken file: a plain list or vector is valid RDS that simply
    needs ``read_r_object()``.
    """
    issues = _skip_root_and_probe(source, limits, encoding)
    if not any(issue.severity == "error" for issue in issues):
        issues.append(RDSValidationIssue("info", "non-tabular", str(original)))
    return issues


def _column_issues(table: RTableInfo) -> list[RDSValidationIssue]:
    issues: list[RDSValidationIssue] = []
    for column in table.schema:
        if column.logical_type == "list":
            issues.append(
                RDSValidationIssue(
                    "warning",
                    "list-column",
                    "list column: read_rds() yields Python objects; Parquet "
                    "export may need an explicit list_column_mode",
                    table=table.name,
                    column=column.name,
                )
            )
        elif column.logical_type == "datetime_components":
            issues.append(
                RDSValidationIssue(
                    "info",
                    "posixlt-column",
                    "POSIXlt column: wall-clock components are reconstructed "
                    "as naive timestamps",
                    table=table.name,
                    column=column.name,
                )
            )
        elif column.r_type == "complex":
            issues.append(
                RDSValidationIssue(
                    "info",
                    "complex-column",
                    "complex column: becomes STRUCT(real, imag) in Parquet",
                    table=table.name,
                    column=column.name,
                )
            )
        elif column.logical_type.startswith("sexp_"):
            issues.append(
                RDSValidationIssue(
                    "warning",
                    "unknown-column-type",
                    f"column with unrecognized storage type {column.r_type}",
                    table=table.name,
                    column=column.name,
                )
            )
    return issues


def diff_rds(
    left: Any,
    right: Any,
    *,
    content: bool = False,
    limits: ReaderLimits | None = None,
    encoding: str | None = None,
) -> RDSDiffReport:
    """Compare two RDS files structurally and, optionally, value by value.

    The structural tier uses catalog scans only (no column payloads):
    compression, table sets matched by name, row counts, column
    added/removed/reordered, storage/logical type changes, and factor level
    changes. With ``content=True`` both files are additionally read as
    Arrow (requires ``rdsframe[arrow]``, and enough memory for both) and
    each structurally comparable column reports how many rows differ.
    Comparison is positional, and float NaN equals NaN -- see the module
    docstring for why both rules fit auditing.
    """
    left_source = _validate_source(left)
    right_source = _validate_source(right)
    left_info = inspect_r_file(left_source)
    right_info = inspect_r_file(right_source)
    left_catalog = list_rds_tables(
        left_source, cache=False, limits=limits, encoding=encoding
    )
    right_catalog = list_rds_tables(
        right_source, cache=False, limits=limits, encoding=encoding
    )

    entries: list[RDSDiffEntry] = []
    if left_info.compression != right_info.compression:
        entries.append(
            RDSDiffEntry(
                "compression_changed",
                before=left_info.compression,
                after=right_info.compression,
            )
        )

    left_tables = {table.name: table for table in left_catalog.tables}
    right_tables = {table.name: table for table in right_catalog.tables}
    for name in left_tables:
        if name not in right_tables:
            entries.append(RDSDiffEntry("table_removed", table=name))
    for name in right_tables:
        if name not in left_tables:
            entries.append(RDSDiffEntry("table_added", table=name))

    common = [name for name in left_tables if name in right_tables]
    comparable_columns: dict[str, list[tuple[str, int, int]]] = {}
    for name in common:
        table_entries, comparable = _diff_table(
            left_tables[name], right_tables[name]
        )
        entries.extend(table_entries)
        comparable_columns[name] = comparable

    if content:
        entries.extend(
            _diff_content(
                left_source,
                right_source,
                left_catalog.tables,
                right_catalog.tables,
                comparable_columns,
                limits=limits,
                encoding=encoding,
            )
        )
    return RDSDiffReport(
        left_source, right_source, content, tuple(entries)
    )


def _describe_column(column: Any) -> str:
    if column.logical_type == column.r_type:
        return str(column.r_type)
    return f"{column.r_type}/{column.logical_type}"


def _preview(values: list[str], limit: int = 5) -> str:
    shown = ", ".join(repr(value) for value in values[:limit])
    if len(values) > limit:
        shown += f", ... (+{len(values) - limit})"
    return shown


def _keyed_columns(
    table: RTableInfo,
) -> list[tuple[tuple[str, int], int, Any]]:
    """Schema columns keyed by ``(name, occurrence)`` in serialized order.

    R tolerates duplicate column names, so a plain name-keyed dict would
    silently collapse duplicates -- an audit tool must keep each occurrence
    distinct and compare positionally.
    """
    counts: dict[str, int] = {}
    keyed: list[tuple[tuple[str, int], int, Any]] = []
    for position, column in enumerate(table.schema):
        occurrence = counts.get(column.name, 0)
        counts[column.name] = occurrence + 1
        keyed.append(((column.name, occurrence), position, column))
    return keyed


def _display_name(key: tuple[str, int]) -> str:
    name, occurrence = key
    return name if occurrence == 0 else f"{name} (occurrence {occurrence + 1})"


def _diff_table(
    left: RTableInfo, right: RTableInfo
) -> tuple[list[RDSDiffEntry], list[tuple[str, int, int]]]:
    """Structural entries for one table plus its value-comparable columns.

    The comparable list carries ``(display_name, left_position,
    right_position)`` so the content tier can address columns positionally
    -- including duplicated names, which Arrow cannot look up by name.
    """
    entries: list[RDSDiffEntry] = []
    if left.rows is not None and right.rows is not None and left.rows != right.rows:
        entries.append(
            RDSDiffEntry(
                "rows_changed",
                table=left.name,
                before=str(left.rows),
                after=str(right.rows),
            )
        )
    left_keyed = _keyed_columns(left)
    right_keyed = _keyed_columns(right)
    left_map = {key: (position, column) for key, position, column in left_keyed}
    right_map = {key: (position, column) for key, position, column in right_keyed}
    for key, _position, column in left_keyed:
        if key not in right_map:
            entries.append(
                RDSDiffEntry(
                    "column_removed",
                    table=left.name,
                    column=_display_name(key),
                    before=_describe_column(column),
                )
            )
    for key, _position, column in right_keyed:
        if key not in left_map:
            entries.append(
                RDSDiffEntry(
                    "column_added",
                    table=left.name,
                    column=_display_name(key),
                    after=_describe_column(column),
                )
            )
    common_left_order = [key for key, _p, _c in left_keyed if key in right_map]
    common_right_order = [key for key, _p, _c in right_keyed if key in left_map]
    if common_left_order != common_right_order:
        entries.append(
            RDSDiffEntry(
                "columns_reordered",
                table=left.name,
                before=", ".join(_display_name(key) for key in common_left_order),
                after=", ".join(_display_name(key) for key in common_right_order),
            )
        )

    comparable: list[tuple[str, int, int]] = []
    for key in common_left_order:
        name = _display_name(key)
        left_position, before_column = left_map[key]
        right_position, after_column = right_map[key]
        if (
            before_column.r_type != after_column.r_type
            or before_column.logical_type != after_column.logical_type
        ):
            entries.append(
                RDSDiffEntry(
                    "type_changed",
                    table=left.name,
                    column=name,
                    before=_describe_column(before_column),
                    after=_describe_column(after_column),
                )
            )
            continue
        if before_column.levels != after_column.levels:
            removed = [
                level
                for level in before_column.levels
                if level not in set(after_column.levels)
            ]
            added = [
                level
                for level in after_column.levels
                if level not in set(before_column.levels)
            ]
            if added or removed:
                pieces = []
                if added:
                    pieces.append(f"added: {_preview(added)}")
                if removed:
                    pieces.append(f"removed: {_preview(removed)}")
                detail = "; ".join(pieces)
            else:
                detail = "level order changed"
            entries.append(
                RDSDiffEntry(
                    "levels_changed",
                    table=left.name,
                    column=name,
                    before=f"{len(before_column.levels)} levels",
                    after=f"{len(after_column.levels)} levels",
                    detail=detail,
                )
            )
            # Levels changed but codes may still decode to equal values;
            # the column stays comparable at the value level.
        comparable.append((name, left_position, right_position))
    return entries, comparable


def _diff_content(
    left_source: Path,
    right_source: Path,
    left_tables: tuple[RTableInfo, ...],
    right_tables: tuple[RTableInfo, ...],
    comparable_columns: dict[str, list[tuple[str, int, int]]],
    *,
    limits: ReaderLimits | None,
    encoding: str | None,
) -> list[RDSDiffEntry]:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "diff_rds(content=True) requires: pip install 'rdsframe[arrow]'"
        ) from exc

    from .api import read_rds_arrow

    def tables_by_catalog(source: Path, catalog: tuple[RTableInfo, ...]) -> dict[str, Any]:
        result = read_rds_arrow(
            source, limits=limits, encoding=encoding
        )
        # read_rds_arrow sanitizes dict keys differently from catalog names;
        # serialization order is shared, so align by position instead.
        values = list(result.values()) if isinstance(result, dict) else [result]
        if len(values) != len(catalog):
            raise InvalidRDS(
                "Arrow table count does not match the structural catalog"
            )
        return {info.name: table for info, table in zip(catalog, values, strict=True)}

    left_arrow = tables_by_catalog(left_source, left_tables)
    right_arrow = tables_by_catalog(right_source, right_tables)

    entries: list[RDSDiffEntry] = []
    for table_name, columns in comparable_columns.items():
        left_table = left_arrow.get(table_name)
        right_table = right_arrow.get(table_name)
        if left_table is None or right_table is None:  # pragma: no cover - defensive
            continue
        for column_name, left_position, right_position in columns:
            try:
                left_column = left_table.column(left_position)
                right_column = right_table.column(right_position)
            except (KeyError, IndexError):  # pragma: no cover - defensive
                continue
            if len(left_column) != len(right_column):
                continue  # rows_changed is already reported structurally
            if left_column.type != right_column.type:
                entries.append(
                    RDSDiffEntry(
                        "type_changed",
                        table=table_name,
                        column=column_name,
                        before=str(left_column.type),
                        after=str(right_column.type),
                        detail="Arrow representation differs",
                    )
                )
                continue
            differing = _differing_rows(pa, left_column, right_column)
            if differing:
                detail = (
                    f"{differing} of {len(left_column)} rows differ"
                    if differing > 0
                    else "values differ"
                )
                entries.append(
                    RDSDiffEntry(
                        "values_changed",
                        table=table_name,
                        column=column_name,
                        detail=detail,
                    )
                )
    return entries


def _differing_rows(pa: Any, left: Any, right: Any) -> int:
    """Count positionally differing rows; -1 when only inequality is known.

    Nulls compare as equal to nulls, and float NaN equals NaN: R's NA_real_
    surfaces as NaN in Arrow doubles, and "both files miss this value" is
    not a difference an audit should flag.
    """
    import pyarrow.compute as pc  # type: ignore[import-untyped]

    # Normalize chunk layout so binary kernels see aligned inputs even on
    # older pyarrow versions.
    left = left.combine_chunks()
    right = right.combine_chunks()
    if pa.types.is_dictionary(left.type):
        # Factor codes are an encoding detail; compare decoded values.
        try:
            left = left.cast(left.type.value_type)
            right = right.cast(right.type.value_type)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            return 0 if left.equals(right) else -1
    if left.equals(right):
        return 0
    try:
        unequal = pc.fill_null(pc.not_equal(left, right), False)
        null_mismatch = pc.xor(pc.is_null(left), pc.is_null(right))
        mask = pc.or_(unequal, null_mismatch)
        if pa.types.is_floating(left.type):
            both_nan = pc.and_(
                pc.fill_null(pc.is_nan(left), False),
                pc.fill_null(pc.is_nan(right), False),
            )
            mask = pc.and_(mask, pc.invert(both_nan))
        count = pc.sum(mask).as_py()
        return int(count) if count else 0
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        # Types without comparison kernels (e.g. structs on old pyarrow):
        # equals() already said the columns differ, only the count is unknown.
        return -1
