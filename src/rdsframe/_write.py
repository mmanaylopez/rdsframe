"""Serialize pandas data.frames to R's RDS format (XDR serialization v3).

This is a deliberately *limited* writer: a flat ``pandas.DataFrame`` (or a
named mapping of them) with atomic column types -- integers, doubles,
logicals, strings, factors, ``Date``/``POSIXct``/``difftime`` -- becomes an R
``data.frame`` that R itself can ``readRDS()``. Anything outside that set
fails with :class:`RDSWriteError` naming the column, and every lossy
coercion is an explicit policy choice (see ``int64=``), never silent. That
mirrors the reader's contract: a narrow, verified round trip is worth more
than a general writer that quietly corrupts edge cases.

Fidelity notes (the details R itself cares about):

- Missing values use R's exact sentinels: ``NA_integer_``/``NA`` (logical)
  are INT32_MIN, ``NA_character_`` is a length ``-1`` CHARSXP, and
  ``NA_real_`` is the IEEE NaN whose low word is 1954 -- so ``is.na()``
  distinguishes it from an ordinary NaN. Because a plain float64 column
  cannot tell ``pd.NA`` from NaN, *all* its NaNs become ``NA_real_`` (pandas
  semantics treat NaN as missing); complex NaNs stay ordinary NaNs.
- ``-2**31`` inside an integer column is indistinguishable from R's NA
  sentinel and is therefore rejected (or written as double under
  ``int64="double"``) instead of silently becoming a missing value.
- Strings are written as UTF-8 with the UTF-8 encoding flag set; factor
  levels are stringified the same way ``factor()`` does in R.
- POSIXct is epoch *seconds* as a double -- R's own representation -- so
  sub-microsecond precision truncates exactly as it would in R.
- Output is deterministic: fixed header, fixed attribute order, and a
  zeroed gzip mtime, so identical inputs produce identical bytes (useful
  for content diffing and caching).

Peak memory is roughly one encoded column plus the compressor's buffer:
columns are encoded and written one at a time, matching the reader's
memory posture.
"""

from __future__ import annotations

import bz2
import datetime as _dt
import gzip
import lzma
import os
import struct
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
import pandas as pd

from ._core import (
    _R_NA_REAL_BITS,
    CHARSXP,
    CPLXSXP,
    INTSXP,
    LGLSXP,
    NA_INTEGER,
    REALSXP,
    STRSXP,
    SYMSXP,
    UTF8_MASK,
    VECSXP,
    RDSError,
)

LISTSXP = 2
NILVALUE = 254

# R's integer is always 32-bit and INT32_MIN is its NA sentinel, so the
# valid data range excludes it.
_INT32_MAX = 2**31 - 1
_INT32_MIN_VALID = -(2**31 - 1)

_OBJECT_BIT = 1 << 8
_ATTR_BIT = 1 << 9
_TAG_BIT = 1 << 10

_SECONDS_PER_UNIT = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
_UNITS_PER_DAY = {
    "s": 86_400,
    "ms": 86_400 * 10**3,
    "us": 86_400 * 10**6,
    "ns": 86_400 * 10**9,
}

Compression = Literal["gzip", "bzip2", "xz", "zstd", "none"]
Int64Policy = Literal["error", "double"]


class RDSWriteError(RDSError):
    """A value cannot be represented by the limited RDS writer."""


def _i32(value: int) -> bytes:
    return struct.pack(">i", value)


def _flags(
    sexp_type: int, *, is_object: bool = False, has_attr: bool = False, gp: int = 0
) -> bytes:
    word = sexp_type | (gp << 12)
    if is_object:
        word |= _OBJECT_BIT
    if has_attr:
        word |= _ATTR_BIT
    return _i32(word)


def _charsxp(value: str | None, column: str) -> bytes:
    if value is None:
        return _flags(CHARSXP) + _i32(-1)  # NA_character_
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RDSWriteError(
            f"column {column!r} contains a string that is not encodable as "
            f"UTF-8: {exc}"
        ) from exc
    return _flags(CHARSXP, gp=UTF8_MASK) + _i32(len(encoded)) + encoded


def _strsxp(
    values: list[str | None],
    column: str,
    *,
    is_object: bool = False,
    has_attr: bool = False,
) -> bytes:
    header = _flags(STRSXP, is_object=is_object, has_attr=has_attr)
    parts = [header, _i32(len(values))]
    parts.extend(_charsxp(value, column) for value in values)
    return b"".join(parts)


def _symbol(name: str) -> bytes:
    return _flags(SYMSXP) + _charsxp(name, name)


def _attributes(items: list[tuple[str, bytes]]) -> bytes:
    """Encode a tagged attribute pairlist terminated by NIL.

    Symbols are written in full each time instead of as back-references;
    that is valid serialization (R interns each symbol on read), merely a
    few bytes larger than R's own reference-compressed output.
    """
    parts: list[bytes] = []
    for name, payload in items:
        parts.append(_i32(LISTSXP | _TAG_BIT))
        parts.append(_symbol(name))
        parts.append(payload)
    parts.append(_i32(NILVALUE))
    return b"".join(parts)


def _int32_be(values: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]) -> bytes:
    out = values.astype(np.int32, copy=True)
    if mask.any():
        out[mask] = NA_INTEGER
    return out.astype(">i4").tobytes()


def _double_be(values: np.ndarray[Any, Any], na_mask: np.ndarray[Any, Any]) -> bytes:
    """Big-endian doubles with masked entries set to R's NA_real_ bit pattern.

    The uint64 view round-trip is deliberate: byte order is swapped with
    integer semantics so NaN payload bits (including the 1954 low word that
    distinguishes NA_real_ from NaN) survive exactly.
    """
    out = np.ascontiguousarray(values, dtype=np.float64).copy()
    bits = out.view(np.uint64)
    if na_mask.any():
        bits[na_mask] = _R_NA_REAL_BITS
    return bits.astype(">u8").tobytes()


def _length_checked(count: int, column: str) -> int:
    if count > _INT32_MAX:
        raise RDSWriteError(
            f"column {column!r} has {count:,} elements; long vectors "
            "(> 2^31-1) are not supported by this writer"
        )
    return count


def _datetime_ints(series: pd.Series) -> tuple[np.ndarray[Any, Any], str]:
    """Raw int64 counts plus their unit for a (naive) datetime64 series."""
    values = series.to_numpy()
    unit, _step = np.datetime_data(values.dtype)
    if unit not in _SECONDS_PER_UNIT:
        raise RDSWriteError(f"unsupported datetime unit: {unit!r}")
    return values.view(np.int64), unit


def _timezone_name(dtype: pd.DatetimeTZDtype, column: str) -> str:
    tz = dtype.tz
    name = getattr(tz, "key", None) or getattr(tz, "zone", None)
    if name:
        return str(name)
    if str(tz) == "UTC":
        return "UTC"
    raise RDSWriteError(
        f"column {column!r} uses a fixed-offset timezone ({tz!r}); R's tzone "
        "needs an IANA name -- convert with .dt.tz_convert() first"
    )


def _encode_integerish(
    series: pd.Series, column: str, int64: Int64Policy, sexp_type: int
) -> bytes:
    mask = series.isna().to_numpy(dtype=bool)
    try:
        values = series.to_numpy(dtype="int64", na_value=0)
    except (OverflowError, ValueError, TypeError):
        values = None
    out_of_range = True
    if values is not None:
        valid = values[~mask] if mask.any() else values
        out_of_range = bool(
            ((valid < _INT32_MIN_VALID) | (valid > _INT32_MAX)).any()
        )
    if not out_of_range and values is not None:
        return (
            _flags(sexp_type)
            + _i32(_length_checked(len(values), column))
            + _int32_be(values, mask)
        )
    if int64 == "error":
        raise RDSWriteError(
            f"column {column!r} has integer values outside R's 32-bit range "
            f"[{_INT32_MIN_VALID}, {_INT32_MAX}] (note: -2**31 is R's NA "
            "sentinel); pass int64='double' to write them as R doubles "
            "(exact up to 2**53)"
        )
    doubles = series.to_numpy(dtype="float64", na_value=np.nan)
    return (
        _flags(REALSXP)
        + _i32(_length_checked(len(doubles), column))
        + _double_be(doubles, mask)
    )


def _encode_column(
    label: str, series: pd.Series, *, int64: Int64Policy, as_date: bool
) -> bytes:
    dtype = series.dtype

    if isinstance(dtype, pd.CategoricalDtype):
        return _encode_factor(label, series)

    if isinstance(dtype, pd.DatetimeTZDtype):
        if as_date:
            raise RDSWriteError(
                f"date_columns includes {label!r}, but it is timezone-aware; "
                "R Dates are calendar days -- drop the timezone first"
            )
        tzone = _timezone_name(dtype, label)
        naive = series.dt.tz_convert("UTC").dt.tz_localize(None)
        return _encode_posixct(label, naive, series.isna(), tzone)

    if pd.api.types.is_datetime64_any_dtype(dtype):
        if as_date:
            return _encode_date(label, series)
        return _encode_posixct(label, series, series.isna(), None)

    if as_date:
        raise RDSWriteError(
            f"date_columns includes {label!r}, but its dtype is {dtype}; "
            "only naive datetime64 columns can be forced to R Date"
        )

    if pd.api.types.is_timedelta64_dtype(dtype):
        return _encode_difftime(label, series)

    if pd.api.types.is_bool_dtype(dtype):
        mask = series.isna().to_numpy(dtype=bool)
        values = series.to_numpy(dtype="int64", na_value=0)
        return (
            _flags(LGLSXP)
            + _i32(_length_checked(len(values), label))
            + _int32_be(values, mask)
        )

    if pd.api.types.is_integer_dtype(dtype):
        return _encode_integerish(series, label, int64, INTSXP)

    if pd.api.types.is_float_dtype(dtype):
        mask = series.isna().to_numpy(dtype=bool)
        values = series.to_numpy(dtype="float64", na_value=np.nan)
        # A plain float64 column cannot distinguish pd.NA from NaN, and in
        # pandas semantics NaN *is* the missing value -- so every NaN
        # becomes R's NA_real_ rather than R's distinct NaN.
        mask = mask | np.isnan(values)
        return (
            _flags(REALSXP)
            + _i32(_length_checked(len(values), label))
            + _double_be(values, mask)
        )

    if pd.api.types.is_complex_dtype(dtype):
        values = np.ascontiguousarray(series.to_numpy(), dtype=np.complex128)
        payload = values.view(np.float64).view(np.uint64).astype(">u8").tobytes()
        return (
            _flags(CPLXSXP) + _i32(_length_checked(len(values), label)) + payload
        )

    if isinstance(dtype, pd.StringDtype):
        items = [None if pd.isna(value) else str(value) for value in series]
        return _strsxp(items, label)

    if dtype == np.dtype(object):
        return _encode_object_column(label, series)

    raise RDSWriteError(
        f"column {label!r} has dtype {dtype}, which this writer does not "
        "support (supported: integer, float, boolean, string, categorical, "
        "datetime64, timedelta64, complex)"
    )


def _encode_object_column(label: str, series: pd.Series) -> bytes:
    """Object columns: all-strings become STRSXP; all-dates become R Date."""
    values = list(series)
    kinds = {type(v) for v in values if v is not None and not pd.isna(v)}
    if kinds <= {str}:
        items = [None if v is None or pd.isna(v) else str(v) for v in values]
        return _strsxp(items, label)
    if kinds == {_dt.date}:
        # datetime.datetime is a date subclass; the exact-type check above
        # keeps timestamps out of the Date path on purpose.
        epoch = _dt.date(1970, 1, 1)
        days = np.zeros(len(values), dtype=np.float64)
        mask = np.zeros(len(values), dtype=bool)
        for index, value in enumerate(values):
            if value is None or pd.isna(value):
                mask[index] = True
            else:
                days[index] = float((value - epoch).days)
        payload = (
            _flags(REALSXP, is_object=True, has_attr=True)
            + _i32(_length_checked(len(values), label))
            + _double_be(days, mask)
        )
        return payload + _attributes([("class", _strsxp(["Date"], label))])
    offenders = sorted(k.__name__ for k in kinds if k is not str)
    raise RDSWriteError(
        f"column {label!r} is an object column mixing types {offenders}; "
        "only all-string or all-datetime.date object columns are supported"
    )


def _encode_factor(label: str, series: pd.Series) -> bytes:
    categories = series.cat.categories
    # Mirror R's factor(): non-string level values are stringified exactly
    # as as.character() would produce them for the common cases.
    levels = [str(category) for category in categories]
    codes = series.cat.codes.to_numpy(dtype="int64") + 1  # R codes are 1-based
    mask = codes == 0
    payload = (
        _flags(INTSXP, is_object=True, has_attr=True)
        + _i32(_length_checked(len(codes), label))
        + _int32_be(codes, mask)
    )
    class_values: list[str | None] = (
        ["ordered", "factor"] if series.cat.ordered else ["factor"]
    )
    return payload + _attributes(
        [
            ("levels", _strsxp(list(levels), label)),
            ("class", _strsxp(class_values, label)),
        ]
    )


def _encode_posixct(
    label: str, naive: pd.Series, na_mask_series: pd.Series, tzone: str | None
) -> bytes:
    ints, unit = _datetime_ints(naive)
    mask = na_mask_series.to_numpy(dtype=bool)
    seconds = ints.astype(np.float64) * _SECONDS_PER_UNIT[unit]
    payload = (
        _flags(REALSXP, is_object=True, has_attr=True)
        + _i32(_length_checked(len(ints), label))
        + _double_be(seconds, mask)
    )
    items: list[tuple[str, bytes]] = []
    if tzone is not None:
        items.append(("tzone", _strsxp([tzone], label)))
    items.append(("class", _strsxp(["POSIXct", "POSIXt"], label)))
    return payload + _attributes(items)


def _encode_date(label: str, series: pd.Series) -> bytes:
    ints, unit = _datetime_ints(series)
    mask = series.isna().to_numpy(dtype=bool)
    per_day = _UNITS_PER_DAY[unit]
    remainder = ints % per_day
    if bool((remainder[~mask] != 0).any()):
        raise RDSWriteError(
            f"date_columns includes {label!r}, but it contains non-midnight "
            "timestamps; normalize them or write the column as POSIXct"
        )
    days = (ints // per_day).astype(np.float64)
    payload = (
        _flags(REALSXP, is_object=True, has_attr=True)
        + _i32(_length_checked(len(ints), label))
        + _double_be(days, mask)
    )
    return payload + _attributes([("class", _strsxp(["Date"], label))])


def _encode_difftime(label: str, series: pd.Series) -> bytes:
    values = series.to_numpy()
    unit, _step = np.datetime_data(values.dtype)
    if unit not in _SECONDS_PER_UNIT:
        raise RDSWriteError(f"unsupported timedelta unit: {unit!r}")
    mask = series.isna().to_numpy(dtype=bool)
    seconds = values.view(np.int64).astype(np.float64) * _SECONDS_PER_UNIT[unit]
    payload = (
        _flags(REALSXP, is_object=True, has_attr=True)
        + _i32(_length_checked(len(values), label))
        + _double_be(seconds, mask)
    )
    # Seconds is the canonical unit: any difftime read by rdsframe was
    # normalized to a Timedelta, so the round trip preserves the duration
    # even though R's display unit is not retained.
    return payload + _attributes(
        [
            ("units", _strsxp(["secs"], label)),
            ("class", _strsxp(["difftime"], label)),
        ]
    )


def _row_names(frame: pd.DataFrame) -> bytes:
    index = frame.index
    if isinstance(index, pd.MultiIndex):
        raise RDSWriteError(
            "MultiIndex row labels cannot be represented as R row.names; "
            "reset_index() first"
        )
    if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
        if len(index) == 0:
            return _flags(INTSXP) + _i32(0)  # R stores integer(0) here
        # R's compact encoding for default sequential row names.
        return _flags(INTSXP) + _i32(2) + _i32(NA_INTEGER) + _i32(-len(index))
    if pd.api.types.is_integer_dtype(index.dtype) and not index.hasnans:
        values = index.to_numpy(dtype="int64")
        if (
            (values >= _INT32_MIN_VALID) & (values <= _INT32_MAX)
        ).all():
            mask = np.zeros(len(values), dtype=bool)
            return (
                _flags(INTSXP)
                + _i32(_length_checked(len(values), "row.names"))
                + _int32_be(values, mask)
            )
    labels = [None if pd.isna(value) else str(value) for value in index]
    return _strsxp(labels, "row.names")


def _write_frame(
    out: Any,
    frame: pd.DataFrame,
    *,
    int64: Int64Policy,
    date_columns: frozenset[str],
) -> None:
    labels = [str(column) for column in frame.columns]
    out.write(_flags(VECSXP, is_object=True, has_attr=True) + _i32(len(labels)))
    # Positional iteration: frame.items() misbehaves under duplicate column
    # labels (R tolerates duplicates, so the writer must too).
    for position, label in enumerate(labels):
        out.write(
            _encode_column(
                label,
                frame.iloc[:, position],
                int64=int64,
                as_date=label in date_columns,
            )
        )
    out.write(
        _attributes(
            [
                ("names", _strsxp(list(labels), "names")),
                ("row.names", _row_names(frame)),
                ("class", _strsxp(["data.frame"], "class")),
            ]
        )
    )


def _write_stream(
    out: Any,
    data: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    int64: Int64Policy,
    date_columns: frozenset[str],
) -> None:
    # XDR RDS header, serialization version 3 (R >= 3.5.0 reads it), with
    # the native-encoding field every modern R writes.
    out.write(b"X\n" + struct.pack(">iii", 3, 0x040500, 0x030500))
    out.write(_i32(5) + b"UTF-8")
    if isinstance(data, pd.DataFrame):
        _write_frame(out, data, int64=int64, date_columns=date_columns)
        return
    names = list(data)
    out.write(_flags(VECSXP, has_attr=True) + _i32(len(names)))
    for name in names:
        _write_frame(out, data[name], int64=int64, date_columns=date_columns)
    out.write(
        _attributes([("names", _strsxp(list(names), "names"))])
    )


def _open_compressor(raw: BinaryIO, compress: Compression) -> Any:
    if compress == "gzip":
        # mtime=0 keeps the output deterministic for identical inputs.
        return gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
    if compress == "bzip2":
        return bz2.BZ2File(raw, "wb")
    if compress == "xz":
        return lzma.LZMAFile(raw, "wb")
    if compress == "zstd":
        try:
            from compression import zstd  # type: ignore[import-not-found]
        except ImportError:
            pass
        else:
            return zstd.ZstdFile(raw, "wb")
        try:
            import zstandard
        except ImportError as exc:
            raise ImportError(
                "zstd compression requires Python >= 3.14 or: "
                "pip install 'rdsframe[zstd]'"
            ) from exc
        return zstandard.ZstdCompressor().stream_writer(raw, closefd=False)
    return raw


def _validate_data(
    data: pd.DataFrame | Mapping[str, pd.DataFrame],
) -> None:
    if isinstance(data, pd.DataFrame):
        return
    if not isinstance(data, Mapping):
        raise TypeError(
            "write_rds() takes a pandas DataFrame or a mapping of names to "
            f"DataFrames, not {type(data).__name__}"
        )
    if not data:
        raise ValueError("cannot write an empty mapping of data.frames")
    for name, frame in data.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"table names must be non-empty strings, got {name!r}")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"table {name!r} is {type(frame).__name__}, not a DataFrame"
            )


def write_rds(
    data: pd.DataFrame | Mapping[str, pd.DataFrame],
    path: os.PathLike[str] | str | BinaryIO,
    *,
    compress: Compression = "gzip",
    int64: Int64Policy = "error",
    date_columns: Sequence[str] = (),
) -> Path | None:
    """Write a DataFrame (or a named mapping of them) as an RDS file.

    The result is a plain R ``data.frame`` -- or a named list of them --
    readable by ``readRDS()`` in R >= 3.5 and by :func:`rdsframe.read_rds`.
    See the module docstring for the exact type mapping and NA fidelity
    rules. Files are written atomically (temp file + rename); ``path`` may
    also be a writable binary stream, in which case nothing is renamed and
    ``None`` is returned.

    ``int64`` controls integer columns with values outside R's 32-bit
    range: ``"error"`` (default) fails explicitly, ``"double"`` writes the
    column as R doubles (exact up to 2**53).

    ``date_columns`` names naive datetime64 columns to write as R ``Date``
    (calendar days) instead of ``POSIXct``; non-midnight values in those
    columns are an error, never a silent truncation.
    """
    _validate_data(data)
    if compress not in {"gzip", "bzip2", "xz", "zstd", "none"}:
        raise ValueError(
            "compress must be 'gzip', 'bzip2', 'xz', 'zstd', or 'none'"
        )
    if int64 not in {"error", "double"}:
        raise ValueError("int64 must be 'error' or 'double'")
    if isinstance(date_columns, (str, bytes)):
        raise TypeError("date_columns must be a sequence of column names")
    forced_dates = frozenset(str(name) for name in date_columns)
    frames = [data] if isinstance(data, pd.DataFrame) else list(data.values())
    known = {str(column) for frame in frames for column in frame.columns}
    missing = sorted(forced_dates - known)
    if missing:
        raise ValueError(f"date_columns not found in the data: {missing}")

    if hasattr(path, "write") and not isinstance(path, (str, os.PathLike)):
        stream = _open_compressor(path, compress)
        try:
            _write_stream(
                stream, data, int64=int64, date_columns=forced_dates
            )
        finally:
            if stream is not path:
                stream.close()
        return None

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            stream = _open_compressor(raw, compress)
            try:
                _write_stream(
                    stream, data, int64=int64, date_columns=forced_dates
                )
            finally:
                if stream is not raw:
                    stream.close()
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
