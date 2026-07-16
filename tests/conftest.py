from __future__ import annotations

import gzip
import struct
from pathlib import Path

import pytest

NIL = 254
SYM = 1
LIST = 2
CHAR = 9
LGL = 10
INT = 13
REAL = 14
COMPLEX = 15
STR = 16
VEC = 19
RAW = 24
HAS_ATTR = 1 << 9
HAS_TAG = 1 << 10


def i32(value: int) -> bytes:
    return struct.pack(">i", value)


NULL_BYTES = i32(NIL)


def flags(sexp_type: int, *, attr: bool = False, tag: bool = False) -> bytes:
    return i32(sexp_type | (HAS_ATTR if attr else 0) | (HAS_TAG if tag else 0))


def chars(value: str | None) -> bytes:
    if value is None:
        return flags(CHAR) + i32(-1)
    encoded = value.encode()
    return flags(CHAR) + i32(len(encoded)) + encoded


def symbol(value: str) -> bytes:
    return flags(SYM) + chars(value)


def strings(values: list[str | None], attrs: dict[str, bytes] | None = None) -> bytes:
    payload = flags(STR, attr=bool(attrs)) + i32(len(values)) + b"".join(chars(v) for v in values)
    return payload + (attributes(attrs) if attrs else b"")


def integers(values: list[int], attrs: dict[str, bytes] | None = None) -> bytes:
    payload = flags(INT, attr=bool(attrs)) + i32(len(values))
    payload += struct.pack(f">{len(values)}i", *values)
    return payload + (attributes(attrs) if attrs else b"")


def logicals(values: list[int]) -> bytes:
    return flags(LGL) + i32(len(values)) + struct.pack(f">{len(values)}i", *values)


def reals(values: list[float], attrs: dict[str, bytes] | None = None) -> bytes:
    payload = flags(REAL, attr=bool(attrs)) + i32(len(values))
    payload += struct.pack(f">{len(values)}d", *values)
    return payload + (attributes(attrs) if attrs else b"")


def complexes(values: list[complex]) -> bytes:
    payload = flags(COMPLEX) + i32(len(values))
    payload += b"".join(struct.pack(">dd", value.real, value.imag) for value in values)
    return payload


def raw(values: bytes) -> bytes:
    return flags(RAW) + i32(len(values)) + values


def vectors(values: list[bytes], attrs: dict[str, bytes] | None = None) -> bytes:
    payload = flags(VEC, attr=bool(attrs)) + i32(len(values)) + b"".join(values)
    return payload + (attributes(attrs) if attrs else b"")


def attributes(values: dict[str, bytes] | None) -> bytes:
    if not values:
        return flags(NIL)
    output = b""
    for index, (name, value) in enumerate(values.items()):
        output += flags(LIST, tag=True) if index == 0 else b""
        output += symbol(name) + value
        output += flags(LIST, tag=True) if index < len(values) - 1 else flags(NIL)
    return output


def dataframe(columns: list[bytes], names: list[str]) -> bytes:
    attrs = {"names": strings(names), "class": strings(["data.frame"])}
    return flags(VEC, attr=True) + i32(len(columns)) + b"".join(columns) + attributes(attrs)


def dataframe_list(frames: list[bytes], names: list[str]) -> bytes:
    return vectors(frames, {"names": strings(names)})


def rds(object_bytes: bytes) -> bytes:
    encoding = b"UTF-8"
    return (
        b"X\n"
        + struct.pack(">iii", 3, 0x040300, 0x030500)
        + i32(len(encoding))
        + encoding
        + object_bytes
    )


@pytest.fixture
def sample_rds(tmp_path: Path) -> Path:
    factor_attrs = {"levels": strings(["low", "high"]), "class": strings(["factor"])}
    date_attrs = {"class": strings(["Date"])}
    payload = dataframe(
        [
            integers([1, 2, -(2**31)]),
            reals([0.0, 1.0, float("nan")], date_attrs),
            logicals([1, 0, -(2**31)]),
            integers([1, 2, -(2**31)], factor_attrs),
            strings(["á", "á", None]),
        ],
        ["id", "date", "active", "level", "label"],
    )
    path = tmp_path / "sample.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def compressed_rds(sample_rds: Path, tmp_path: Path) -> Path:
    path = tmp_path / "sample.rds.gz"
    path.write_bytes(gzip.compress(sample_rds.read_bytes()))
    return path


@pytest.fixture
def zstd_rds(sample_rds: Path, tmp_path: Path) -> Path:
    """The container R >= 4.5 writes for saveRDS(..., compress = "zstd")."""
    zstandard = pytest.importorskip("zstandard")
    path = tmp_path / "sample.rds.zst"
    path.write_bytes(zstandard.compress(sample_rds.read_bytes()))
    return path


@pytest.fixture
def na_level_factor_rds(tmp_path: Path) -> Path:
    """addNA(): NA is an explicit factor level, next to a genuine "" level.

    Row codes: 1 -> "", 2 -> "a", 3 -> the NA level, NA_integer_ -> missing.
    """
    attrs = {"levels": strings(["", "a", None]), "class": strings(["factor"])}
    payload = dataframe([integers([1, 2, 3, -(2**31)], attrs)], ["f"])
    path = tmp_path / "na_level_factor.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def named_vector_rds(tmp_path: Path) -> Path:
    """A named atomic vector: c(a = 1L, b = 2L, 3L) with one blank name."""
    payload = integers([1, 2, 3], {"names": strings(["a", "b", ""])})
    path = tmp_path / "named_vector.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def multi_frame_rds(tmp_path: Path) -> Path:
    payload = dataframe_list(
        [
            dataframe([integers([1, 2])], ["id"]),
            dataframe([strings(["a", "b"])], ["label"]),
        ],
        ["numbers", "labels"],
    )
    path = tmp_path / "multiple.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def advanced_types_rds(tmp_path: Path) -> Path:
    posixct_attrs = {
        "class": strings(["POSIXct", "POSIXt"]),
        "tzone": strings(["America/Lima"]),
    }
    payload = dataframe(
        [
            reals([0.0, float("nan"), float("inf")], posixct_attrs),
            vectors([integers([1, 2]), strings(["x"]), strings([None])]),
            complexes([1 + 2j, 3 - 4j, 0j]),
            raw(bytes([1, 2, 255])),
        ],
        ["when", "nested", "z", "payload"],
    )
    path = tmp_path / "advanced.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def ordered_factor_rds(tmp_path: Path) -> Path:
    ordered_attrs = {
        "levels": strings(["low", "medium", "high"]),
        "class": strings(["ordered", "factor"]),
    }
    unordered_attrs = {
        "levels": strings(["red", "green"]),
        "class": strings(["factor"]),
    }
    payload = dataframe(
        [
            integers([1, 3, 2], ordered_attrs),
            integers([2, 1, 2], unordered_attrs),
        ],
        ["severity", "color"],
    )
    path = tmp_path / "ordered_factor.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def difftime_rds(tmp_path: Path) -> Path:
    difftime_attrs = {
        "class": strings(["difftime"]),
        "units": strings(["days"]),
    }
    payload = dataframe(
        [reals([1.0, 2.5, float("nan")], difftime_attrs)],
        ["elapsed"],
    )
    path = tmp_path / "difftime.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def matrix_column_rds(tmp_path: Path) -> Path:
    matrix_attrs = {"dim": integers([2, 3])}
    payload = dataframe(
        [integers([1, 2]), integers([10, 20, 30, 40, 50, 60], matrix_attrs)],
        ["id", "mat"],
    )
    path = tmp_path / "matrix_column.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def nested_dataframe_column_rds(tmp_path: Path) -> Path:
    """A data.frame whose `sub` column is itself a 2x2 data.frame.

    The square case (nested rows == nested columns) is the dangerous one:
    every column of the outer frame has length 2, so nothing trips the
    length check and a naive read silently returns the nested frame's
    *columns* as if they were row values.
    """
    nested = dataframe([integers([3, 4]), integers([5, 6])], ["a", "b"])
    payload = dataframe([integers([1, 2]), nested], ["id", "sub"])
    path = tmp_path / "nested_dataframe_column.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def dataframe_of_dataframes_rds(tmp_path: Path) -> Path:
    """A data.frame in which *every* column is a nested data.frame.

    The streaming Parquet path classifies a root by its first child before
    the root's own class attribute arrives, so this shape used to be
    misread as a list of independent tables.
    """
    payload = dataframe(
        [
            dataframe([integers([1, 2])], ["a"]),
            dataframe([integers([3, 4])], ["b"]),
        ],
        ["s1", "s2"],
    )
    path = tmp_path / "dataframe_of_dataframes.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def integer_posixct_rds(tmp_path: Path) -> Path:
    """POSIXct stored with integer storage.mode (legal, e.g. DB imports)."""
    posixct_attrs = {
        "class": strings(["POSIXct", "POSIXt"]),
        "tzone": strings(["UTC"]),
    }
    payload = dataframe(
        [integers([1, 86400, -(2**31)], posixct_attrs)],
        ["t"],
    )
    path = tmp_path / "integer_posixct.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def integer_difftime_rds(tmp_path: Path) -> Path:
    """difftime stored with integer storage.mode."""
    difftime_attrs = {
        "class": strings(["difftime"]),
        "units": strings(["days"]),
    }
    payload = dataframe(
        [integers([1, 2, -(2**31)], difftime_attrs)],
        ["elapsed"],
    )
    path = tmp_path / "integer_difftime.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def named_list_rds(tmp_path: Path) -> Path:
    """A plain named list that is not a data.frame."""
    payload = vectors(
        [
            strings(["x", "y", "z"]),
            integers([1, 2, 3]),
            NULL_BYTES,
        ],
        {"names": strings(["letters", "numbers", "nothing"])},
    )
    path = tmp_path / "named_list.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def unnamed_nested_list_rds(tmp_path: Path) -> Path:
    """An unnamed list of unnamed lists (nested-structure case)."""
    inner = vectors([integers([1, 2]), strings(["a"])])
    payload = vectors([inner, inner, inner])
    path = tmp_path / "unnamed_nested_list.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def mixed_object_rds(tmp_path: Path) -> Path:
    """A named list mixing a nested data.frame, a factor, and a matrix."""
    matrix_attrs = {"dim": integers([2, 2])}
    factor_attrs = {"levels": strings(["low", "high"]), "class": strings(["factor"])}
    payload = vectors(
        [
            dataframe([integers([1, 2])], ["id"]),
            integers([1, 2, 1], factor_attrs),
            integers([1, 2, 3, 4], matrix_attrs),
            NULL_BYTES,
        ],
        {"names": strings(["table", "level", "grid", "empty"])},
    )
    path = tmp_path / "mixed_object.rds"
    path.write_bytes(rds(payload))
    return path


def _posixlt_vecsxp(sec: list[float], components: dict[str, list[int]]) -> bytes:
    attrs = {
        "names": strings(["sec", *components.keys()]),
        "class": strings(["POSIXlt", "POSIXt"]),
    }
    return vectors(
        [reals(sec), *(integers(values) for values in components.values())],
        attrs,
    )


@pytest.fixture
def posixlt_rds(tmp_path: Path) -> Path:
    # Row 0: 2024-03-15 10:30:45.5 (mon is 0-based, year is since 1900).
    # Row 1: all-NA (sec is NaN), must become NaT rather than a wrong date.
    posixlt = _posixlt_vecsxp(
        [45.5, float("nan")],
        {"min": [30, 0], "hour": [10, 0], "mday": [15, 1], "mon": [2, 0], "year": [124, 70]},
    )
    payload = dataframe([posixlt], ["when"])
    path = tmp_path / "posixlt.rds"
    path.write_bytes(rds(payload))
    return path


@pytest.fixture
def standalone_posixlt_rds(tmp_path: Path) -> Path:
    posixlt = _posixlt_vecsxp(
        [0.0], {"min": [0], "hour": [0], "mday": [1], "mon": [0], "year": [100]}
    )
    path = tmp_path / "standalone_posixlt.rds"
    path.write_bytes(rds(posixlt))
    return path
