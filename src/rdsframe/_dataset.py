"""Multi-file RDS collections: one logical dataset over many .rds files.

Data lakes, yearly archives, and survey waves often store one RDS per
period with (nearly) the same schema. :func:`open_rds_dataset` aggregates
them behind a single handle with the same design rules as the rest of the
library:

- Schema compatibility is validated from *structural catalogs* before any
  column payload is read. ``schema_mode="strict"`` requires identical
  column names, order, and storage/logical types; ``"union"`` fills columns
  missing in some files with nulls. A same-named column with a different
  type is always an explicit :class:`RDSCatalogError` -- never a silent
  promotion.
- Factor columns whose level sets differ across files are decoded to plain
  strings everywhere (values are preserved exactly; only the dictionary
  encoding is dropped). Identical levels keep the dictionary/categorical
  type. The rule is decided from catalogs, so it is deterministic and
  applies identically to every output format.
- Parallelism happens *across files* (one process per file), which is the
  axis RDS actually allows: catalog scans and Parquet export accept
  ``workers=``. In-memory readers stay sequential on purpose -- shipping
  whole tables between processes would double memory for little gain.
- ``iter_arrow()`` / ``to_record_batch_reader()`` hold one *file* in
  memory at a time -- the whole file, including any tables and columns
  that the selection then discards, because RDS offers no selective Arrow
  read yet. ``to_arrow()``/``to_pandas()``/``to_polars()`` materialize the
  whole collection and say so.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from glob import glob
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from ._core import RDSCatalogError, RDSError, ReaderLimits, UnsupportedRDS
from .api import (
    ParquetTable,
    RDSCatalog,
    RTableInfo,
    _safe_name,
    _unique_name,
    list_rds_tables,
    read_rds,
    read_rds_arrow,
)

SchemaMode = Literal["strict", "union"]

# Column kinds that may be null-filled when union mode finds them missing
# in some files. Types whose Arrow form cannot be derived from the catalog
# alone (datetime time zones, list layouts, ...) are deliberately excluded:
# filling them would require guessing.
_FILLABLE = {
    "integer",
    "double",
    "logical",
    "character",
    "factor",
    "ordered_factor",
}


@dataclass(frozen=True, slots=True)
class _CollectionPlan:
    """Everything needed to align one file's table to the collection schema."""

    columns: tuple[str, ...]
    fills: tuple[tuple[str, str], ...]  # (column, logical kind)
    decode_factors: frozenset[str]
    table_indices: tuple[int, ...]  # aligned with the files tuple
    schema: tuple[Any, ...]  # RDSColumnInfo for the unified columns


def _resolve_files(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    if isinstance(source, (str, os.PathLike)):
        text = str(source)
        candidate = Path(text).expanduser()
        if candidate.is_dir():
            files = sorted(candidate.glob("*.rds"))
        elif any(character in text for character in "*?["):
            files = sorted(Path(match) for match in glob(text, recursive=True))
        else:
            files = [candidate]
    else:
        files = [Path(item).expanduser() for item in source]
    if not files:
        raise FileNotFoundError(f"no RDS files matched: {source!r}")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"not a file: {missing}")
    return tuple(path.resolve() for path in files)


def _catalog_worker(
    args: tuple[str, ReaderLimits | None, str | None, bool],
) -> RDSCatalog:
    path, limits, encoding, cache = args
    try:
        return list_rds_tables(path, cache=cache, limits=limits, encoding=encoding)
    except RDSError as exc:
        raise type(exc)(f"{Path(path).name}: {exc}") from None


@dataclass(frozen=True, slots=True)
class _ParquetJob:
    """Picklable per-file Parquet conversion job (runs in worker processes)."""

    path: str
    table_index: int
    columns: tuple[str, ...]
    fills: tuple[tuple[str, str], ...]
    decode_factors: tuple[str, ...]
    constants: tuple[tuple[str, str | None], ...]
    destination: str
    compression: str | None
    row_group_size: int
    limits: ReaderLimits | None
    encoding: str | None
    posixct_mode: str
    invalid_timestamp: str
    list_column_mode: str


def _arrow_fill_type(kind: str) -> Any:
    import pyarrow as pa  # type: ignore[import-untyped]

    if kind == "integer":
        return pa.int32()
    if kind == "double":
        return pa.float64()
    if kind == "logical":
        return pa.bool_()
    if kind == "character":
        return pa.large_string()
    return pa.string()  # decoded factor levels


def _align_arrow(
    table: Any,
    *,
    columns: tuple[str, ...],
    fills: dict[str, str],
    decode_factors: frozenset[str],
    constants: Sequence[tuple[str, str | None]],
    source_name: str,
) -> Any:
    import pyarrow as pa

    length = table.num_rows
    names: list[str] = []
    arrays: list[Any] = []
    for name in columns:
        index = table.schema.get_field_index(name)
        if index >= 0:
            array = table.column(index)
            if name in decode_factors and pa.types.is_dictionary(array.type):
                array = array.cast(pa.string())
            arrays.append(array)
        else:
            kind = fills.get(name)
            if kind is None:  # pragma: no cover - guarded by the plan
                raise RDSCatalogError(
                    f"{source_name}: column {name!r} is missing and not fillable"
                )
            arrays.append(pa.nulls(length, type=_arrow_fill_type(kind)))
        names.append(name)
    for name, value in constants:
        arrays.append(pa.array([value] * length, type=pa.string()))
        names.append(name)
    return pa.table(dict(zip(names, arrays, strict=True)))


def _parquet_worker(job: _ParquetJob) -> tuple[str, int, int]:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    try:
        result = read_rds_arrow(
            job.path,
            limits=job.limits,
            encoding=job.encoding,
            posixct_mode=cast(Any, job.posixct_mode),
            invalid_timestamp=cast(Any, job.invalid_timestamp),
            list_column_mode=cast(Any, job.list_column_mode),
        )
        tables = list(result.values()) if isinstance(result, dict) else [result]
        aligned = _align_arrow(
            tables[job.table_index],
            columns=job.columns,
            fills=dict(job.fills),
            decode_factors=frozenset(job.decode_factors),
            constants=job.constants,
            source_name=Path(job.path).name,
        )
        destination = Path(job.destination)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            pq.write_table(
                aligned,
                temporary,
                compression=job.compression,
                row_group_size=job.row_group_size,
            )
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return str(destination), aligned.num_rows, aligned.num_columns
    except RDSError as exc:
        raise type(exc)(f"{Path(job.path).name}: {exc}") from None


class RDSCollection:
    """Deferred handle over many RDS files validated as one dataset."""

    __slots__ = (
        "_cache",
        "_catalogs",
        "_columns",
        "_encoding",
        "_files",
        "_invalid_timestamp",
        "_limits",
        "_list_column_mode",
        "_partitions",
        "_plan",
        "_posixct_mode",
        "_schema_mode",
        "_source_column",
        "_table",
        "_workers",
    )

    def __init__(
        self,
        source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        table: int | str | None = None,
        columns: Sequence[str] | None = None,
        source_column: str | None = None,
        partitions: str | None = None,
        schema_mode: SchemaMode = "strict",
        cache: bool = False,
        workers: int | None = None,
        limits: ReaderLimits | None = None,
        encoding: str | None = None,
        posixct_mode: Literal["preserve", "utc_naive"] = "preserve",
        invalid_timestamp: Literal["error", "null"] = "error",
        list_column_mode: Literal["infer", "json", "string"] = "infer",
    ) -> None:
        if schema_mode not in {"strict", "union"}:
            raise ValueError("schema_mode must be 'strict' or 'union'")
        if isinstance(columns, (str, bytes)):
            raise TypeError("columns must be a sequence of names")
        if isinstance(table, bool):
            raise TypeError("boolean table selectors are not valid indices")
        self._files = _resolve_files(source)
        self._table = table
        self._columns = tuple(columns) if columns is not None else None
        self._source_column = source_column
        self._partitions = re.compile(partitions) if partitions else None
        if self._partitions is not None and not self._partitions.groupindex:
            raise ValueError(
                "partitions must be a regex with named groups, e.g. "
                r"r'(?P<year>\d{4})'"
            )
        self._schema_mode: SchemaMode = schema_mode
        self._cache = bool(cache)
        self._workers = workers
        self._limits = limits
        self._encoding = encoding
        self._posixct_mode = posixct_mode
        self._invalid_timestamp = invalid_timestamp
        self._list_column_mode = list_column_mode
        self._catalogs: tuple[RDSCatalog, ...] | None = None
        self._plan: _CollectionPlan | None = None

    @property
    def files(self) -> tuple[Path, ...]:
        return self._files

    def catalogs(self) -> tuple[RDSCatalog, ...]:
        """Structural catalogs for every file (parallel when ``workers>1``)."""
        if self._catalogs is None:
            jobs = [
                (str(path), self._limits, self._encoding, self._cache)
                for path in self._files
            ]
            if self._workers and self._workers > 1 and len(jobs) > 1:
                with ProcessPoolExecutor(max_workers=self._workers) as pool:
                    self._catalogs = tuple(pool.map(_catalog_worker, jobs))
            else:
                self._catalogs = tuple(_catalog_worker(job) for job in jobs)
        return self._catalogs

    def _selected_tables(self) -> tuple[tuple[RTableInfo, ...], tuple[int, ...]]:
        infos: list[RTableInfo] = []
        indices: list[int] = []
        for path, catalog in zip(self._files, self.catalogs(), strict=True):
            tables = catalog.tables
            if self._table is None:
                if len(tables) != 1:
                    raise RDSCatalogError(
                        f"{path.name} contains {len(tables)} tables; select one "
                        "with table="
                    )
                selected = tables[0]
            elif isinstance(self._table, int):
                if self._table < 0 or self._table >= len(tables):
                    raise RDSCatalogError(
                        f"{path.name}: table index out of range: {self._table}"
                    )
                selected = tables[self._table]
            else:
                matches = [t for t in tables if t.name == self._table]
                if not matches:
                    raise RDSCatalogError(
                        f"{path.name}: table name not found: {self._table!r}"
                    )
                if len(matches) > 1:
                    raise RDSCatalogError(
                        f"{path.name}: table name is ambiguous: {self._table!r}"
                    )
                selected = matches[0]
            infos.append(selected)
            indices.append(selected.index)
        return tuple(infos), tuple(indices)

    def _build_plan(self) -> _CollectionPlan:
        if self._plan is not None:
            return self._plan
        infos, indices = self._selected_tables()
        by_file: list[dict[str, Any]] = []
        for path, info in zip(self._files, infos, strict=True):
            names = [column.name for column in info.schema]
            if len(set(names)) != len(names):
                raise RDSCatalogError(
                    f"{path.name}: duplicate column names are not supported "
                    "in collections"
                )
            by_file.append({column.name: column for column in info.schema})

        first_path, first_info = self._files[0], infos[0]
        reference = [column.name for column in first_info.schema]
        if self._schema_mode == "strict":
            for path, info in zip(self._files[1:], infos[1:], strict=True):
                names = [column.name for column in info.schema]
                if names != reference:
                    raise RDSCatalogError(
                        f"{path.name}: column layout {names} differs from "
                        f"{first_path.name} {reference}; use "
                        "schema_mode='union' to align by name"
                    )
            unified = list(reference)
        else:
            unified = list(reference)
            seen = set(reference)
            for info in infos[1:]:
                for column in info.schema:
                    if column.name not in seen:
                        seen.add(column.name)
                        unified.append(column.name)

        # Projection first: validating only the selected columns is the
        # point of columns= -- an unselected column with incompatible types
        # across files must not block a projection that never touches it.
        if self._columns is not None:
            unknown = [name for name in self._columns if name not in unified]
            if unknown:
                raise ValueError(f"column names not found: {unknown}")
            unified = list(self._columns)

        schema: list[Any] = []
        fills: list[tuple[str, str]] = []
        decode: set[str] = set()
        for name in unified:
            present = [
                (path, columns[name])
                for path, columns in zip(self._files, by_file, strict=True)
                if name in columns
            ]
            first_seen = present[0][1]
            for path, column in present[1:]:
                if (
                    column.r_type != first_seen.r_type
                    or column.logical_type != first_seen.logical_type
                ):
                    raise RDSCatalogError(
                        f"column {name!r} is {first_seen.r_type}/"
                        f"{first_seen.logical_type} in {present[0][0].name} but "
                        f"{column.r_type}/{column.logical_type} in {path.name}; "
                        "collections never coerce types silently"
                    )
            missing_somewhere = len(present) != len(self._files)
            is_factor = first_seen.logical_type in {"factor", "ordered_factor"}
            if missing_somewhere:
                kind = first_seen.logical_type
                if kind not in _FILLABLE:
                    raise RDSCatalogError(
                        f"column {name!r} ({kind}) is missing from some files "
                        "and cannot be null-filled deterministically; only "
                        f"{sorted(_FILLABLE)} columns can"
                    )
                fills.append(
                    (name, "factor" if is_factor else kind)
                )
            if is_factor:
                level_sets = {column.levels for _path, column in present}
                if len(level_sets) > 1 or missing_somewhere:
                    decode.add(name)
            schema.append(
                replace(first_seen, levels=())
                if name in decode
                else first_seen
            )

        reserved = set(unified)
        for extra in self._constant_names():
            if extra in reserved:
                raise ValueError(
                    f"generated column {extra!r} collides with a data column"
                )
            reserved.add(extra)

        self._plan = _CollectionPlan(
            tuple(unified), tuple(fills), frozenset(decode), indices, tuple(schema)
        )
        return self._plan

    def _constant_names(self) -> list[str]:
        names: list[str] = []
        if self._partitions is not None:
            names.extend(self._partitions.groupindex)
        if self._source_column is not None:
            names.append(self._source_column)
        return names

    def _constants_for(self, path: Path) -> tuple[tuple[str, str | None], ...]:
        values: list[tuple[str, str | None]] = []
        if self._partitions is not None:
            match = self._partitions.search(path.name)
            if match is None:
                raise ValueError(
                    f"{path.name} does not match the partitions pattern "
                    f"{self._partitions.pattern!r}"
                )
            values.extend(match.groupdict().items())
        if self._source_column is not None:
            values.append((self._source_column, str(path)))
        return tuple(values)

    @property
    def schema(self) -> tuple[Any, ...]:
        """Unified column schema (factor levels cleared where files differ)."""
        return self._build_plan().schema

    @property
    def columns(self) -> tuple[str, ...]:
        plan = self._build_plan()
        return plan.columns + tuple(self._constant_names())

    @property
    def rows(self) -> int | None:
        infos, _indices = self._selected_tables()
        total = 0
        for info in infos:
            if info.rows is None:
                return None
            total += info.rows
        return total

    def iter_arrow(self) -> Iterator[tuple[Path, Any]]:
        """Yield ``(path, aligned_arrow_table)`` one file at a time.

        Peak memory is one complete *file* (every table it contains):
        table and column selection apply after the file is materialized.
        """
        plan = self._build_plan()
        fills = dict(plan.fills)
        for path, index in zip(self._files, plan.table_indices, strict=True):
            result = read_rds_arrow(
                path,
                limits=self._limits,
                encoding=self._encoding,
                posixct_mode=self._posixct_mode,
                invalid_timestamp=self._invalid_timestamp,
                list_column_mode=self._list_column_mode,
            )
            tables = list(result.values()) if isinstance(result, dict) else [result]
            yield (
                path,
                _align_arrow(
                    tables[index],
                    columns=plan.columns,
                    fills=fills,
                    decode_factors=plan.decode_factors,
                    constants=self._constants_for(path),
                    source_name=path.name,
                ),
            )

    def to_arrow(self) -> Any:
        """Concatenate every file into one Arrow table (all in memory)."""
        import pyarrow as pa

        tables: list[Any] = []
        reference_schema: Any = None
        reference_path: Path | None = None
        for path, table in self.iter_arrow():
            if reference_schema is None:
                reference_schema, reference_path = table.schema, path
            elif table.schema != reference_schema:
                assert reference_path is not None
                raise RDSCatalogError(
                    f"{path.name}: Arrow schema differs from "
                    f"{reference_path.name} beyond what catalogs can detect "
                    "(typically POSIXct time zones); read files individually "
                    "or use posixct_mode='utc_naive'"
                )
            tables.append(table)
        return pa.concat_tables(tables)

    def to_record_batch_reader(self) -> Any:
        """Stream the collection as one RecordBatchReader (one file resident)."""
        import pyarrow as pa

        iterator = self.iter_arrow()
        first_path, first = next(iterator)
        schema = first.schema

        def batches() -> Iterator[Any]:
            yield from first.to_batches()
            for path, table in iterator:
                if table.schema != schema:
                    raise RDSCatalogError(
                        f"{path.name}: Arrow schema differs from "
                        f"{first_path.name}; see to_arrow() notes"
                    )
                yield from table.to_batches()

        return pa.RecordBatchReader.from_batches(schema, batches())

    def _file_to_pandas(self, path: Path, table_index: int) -> pd.DataFrame:
        plan = self._build_plan()
        fills = dict(plan.fills)
        single_frame = len(self.catalogs()[self._files.index(path)].tables) == 1
        frame: pd.DataFrame | None = None
        if single_frame and self._schema_mode == "strict":
            try:
                frame = cast(
                    pd.DataFrame,
                    read_rds(
                        path,
                        columns=list(plan.columns),
                        limits=self._limits,
                        encoding=self._encoding,
                    ),
                )
            except UnsupportedRDS:
                # A one-table catalog does not guarantee a data.frame root:
                # a named list holding a single data.frame also catalogs one
                # table, and the selective reader rejects that root shape.
                frame = None
        if frame is None:
            result = read_rds(path, limits=self._limits, encoding=self._encoding)
            frames = list(result.values()) if isinstance(result, dict) else [result]
            frame = frames[table_index]
        assert isinstance(frame, pd.DataFrame)
        length = len(frame)
        data: dict[str, Any] = {}
        for name in plan.columns:
            if name in frame.columns:
                column = frame[name]
                if name in plan.decode_factors and isinstance(
                    column.dtype, pd.CategoricalDtype
                ):
                    column = pd.Series(
                        [None if pd.isna(v) else str(v) for v in column],
                        dtype=object,
                    )
                if self._posixct_mode == "utc_naive" and isinstance(
                    column.dtype, pd.DatetimeTZDtype
                ):
                    # The documented policy applies to the pandas path too:
                    # the UTC instant is kept, only display-zone metadata is
                    # dropped, which also aligns files with mixed zones.
                    column = column.dt.tz_convert("UTC").dt.tz_localize(None)
                data[name] = column.reset_index(drop=True)
            else:
                data[name] = _pandas_fill(fills[name], length)
        for name, value in self._constants_for(path):
            data[name] = pd.Series([value] * length, dtype=object)
        return pd.DataFrame(data)

    def _check_concat_dtypes(self, frames: list[pd.DataFrame]) -> None:
        """Reject datetime dtype mismatches that pandas would hide as object.

        ``pd.concat`` silently promotes a column mixing time zones (or
        aware and naive values) to ``dtype=object`` -- exactly the silent
        loss this module promises never to perform.
        """
        plan = self._build_plan()
        for name in plan.columns:
            dtypes = {str(frame[name].dtype) for frame in frames}
            if len(dtypes) > 1 and any("datetime" in item for item in dtypes):
                raise RDSCatalogError(
                    f"column {name!r} mixes datetime dtypes across files "
                    f"({sorted(dtypes)}); concatenating would silently "
                    "produce dtype=object. Use posixct_mode='utc_naive' to "
                    "normalize the files, or read them individually"
                )

    def to_pandas(self) -> pd.DataFrame:
        """Concatenate every file into one DataFrame (all in memory)."""
        plan = self._build_plan()
        frames = [
            self._file_to_pandas(path, index)
            for path, index in zip(self._files, plan.table_indices, strict=True)
        ]
        self._check_concat_dtypes(frames)
        return pd.concat(frames, ignore_index=True)

    def head(self, rows: int = 5) -> pd.DataFrame:
        """First *rows* rows, reading only as many files as needed."""
        if rows < 0:
            raise ValueError("rows cannot be negative")
        plan = self._build_plan()
        collected: list[pd.DataFrame] = []
        total = 0
        for path, index in zip(self._files, plan.table_indices, strict=True):
            frame = self._file_to_pandas(path, index)
            collected.append(frame)
            total += len(frame)
            if total >= rows:
                break
        self._check_concat_dtypes(collected)
        return pd.concat(collected, ignore_index=True).head(rows)

    def to_polars(self) -> Any:
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "Polars integration requires: pip install 'rdsframe[polars]'"
            ) from exc
        return pl.from_arrow(self.to_arrow())

    def to_parquet(
        self,
        destination: os.PathLike[str] | str,
        *,
        compression: str = "zstd",
        row_group_size: int = 250_000,
        workers: int | None = None,
    ) -> list[ParquetTable]:
        """Write one Parquet file per input file (parallel with ``workers``).

        Peak memory is one complete input file per worker process (all its
        tables; selection applies after materialization). Output files are
        named after each source's stem (uniquified on collision), include
        the partition/source columns, and are written atomically.
        """
        if row_group_size < 1:
            raise ValueError("row_group_size must be at least 1")
        plan = self._build_plan()
        out_dir = Path(destination)
        out_dir.mkdir(parents=True, exist_ok=True)
        normalized: str | None = compression.lower()
        if normalized in {"none", "uncompressed"}:
            normalized = None

        used: dict[str, Any] = {}
        jobs: list[_ParquetJob] = []
        for path, index in zip(self._files, plan.table_indices, strict=True):
            stem = _unique_name(_safe_name(path.stem, len(used)), used)
            used[stem] = object()
            jobs.append(
                _ParquetJob(
                    path=str(path),
                    table_index=index,
                    columns=plan.columns,
                    fills=plan.fills,
                    decode_factors=tuple(sorted(plan.decode_factors)),
                    constants=self._constants_for(path),
                    destination=str(out_dir / f"{stem}.parquet"),
                    compression=normalized,
                    row_group_size=row_group_size,
                    limits=self._limits,
                    encoding=self._encoding,
                    posixct_mode=self._posixct_mode,
                    invalid_timestamp=self._invalid_timestamp,
                    list_column_mode=self._list_column_mode,
                )
            )
        if workers and workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_parquet_worker, jobs))
        else:
            results = [_parquet_worker(job) for job in jobs]
        return [
            ParquetTable(Path(out_path).stem, Path(out_path), rows, columns)
            for (out_path, rows, columns) in results
        ]

    def __repr__(self) -> str:
        projection = "*" if self._columns is None else ", ".join(self._columns)
        return (
            f"RDSCollection({len(self._files)} files, mode="
            f"{self._schema_mode!r}, columns=[{projection}])"
        )


def _pandas_fill(kind: str, length: int) -> Any:
    if kind == "integer":
        return pd.array([pd.NA] * length, dtype="Int32")
    if kind == "double":
        return pd.array([float("nan")] * length, dtype="float64")
    if kind == "logical":
        return pd.array([pd.NA] * length, dtype="boolean")
    return pd.Series([None] * length, dtype=object)  # character / factor


def open_rds_dataset(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    table: int | str | None = None,
    columns: Sequence[str] | None = None,
    source_column: str | None = None,
    partitions: str | None = None,
    schema_mode: SchemaMode = "strict",
    cache: bool = False,
    workers: int | None = None,
    limits: ReaderLimits | None = None,
    encoding: str | None = None,
    posixct_mode: Literal["preserve", "utc_naive"] = "preserve",
    invalid_timestamp: Literal["error", "null"] = "error",
    list_column_mode: Literal["infer", "json", "string"] = "infer",
) -> RDSCollection:
    """Open many RDS files (glob pattern, directory, or explicit paths) as one dataset.

    See :class:`RDSCollection` for the schema-compatibility rules and
    terminals. ``partitions`` is a regex with named groups applied to each
    file name (``r"(?P<year>\\d{4})"`` adds a ``year`` column);
    ``source_column`` names a column holding each row's origin path;
    ``workers`` parallelizes the catalog scan across processes.
    """
    return RDSCollection(
        source,
        table=table,
        columns=columns,
        source_column=source_column,
        partitions=partitions,
        schema_mode=schema_mode,
        cache=cache,
        workers=workers,
        limits=limits,
        encoding=encoding,
        posixct_mode=posixct_mode,
        invalid_timestamp=invalid_timestamp,
        list_column_mode=list_column_mode,
    )


__all__ = ["RDSCollection", "open_rds_dataset"]
