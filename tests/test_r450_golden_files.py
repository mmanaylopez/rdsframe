"""Golden-file regression tests against RDS files written by real R 4.5.0.

The binaries under ``tests/data/r450`` were produced by ``tests/data/gen_fixtures.R``
with R 4.5.0 on Windows. They pin down behavior that synthetic byte-built
fixtures cannot vouch for: genuine ALTREP serializations (compact sequences,
deferred strings, sort() wrappers), real POSIXlt layout, R's own row-names
encodings, native (non-XDR) byte order, and version-2 streams.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rdsframe import read_r_object, read_rds

DATA = Path(__file__).parent / "data" / "r450"


def test_altrep_compact_intseq() -> None:
    values = read_r_object(DATA / "altrep_compact_intseq.rds")
    assert len(values) == 100_000
    assert values[:3] == [1, 2, 3]
    assert values[-1] == 100_000


def test_altrep_compact_realseq() -> None:
    values = read_r_object(DATA / "altrep_compact_realseq.rds")
    assert len(values) == 1000
    assert values[0] == 1.5
    assert values[-1] == 1000.5


def test_altrep_deferred_string() -> None:
    values = read_r_object(DATA / "altrep_deferred_string.rds")
    assert len(values) == 5000
    assert values[:3] == ["1", "2", "3"]
    assert values[-1] == "5000"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("altrep_sorted_int_wrapper", [10, 20, 30, 40]),
        ("altrep_sorted_real_wrapper", [1.25, 2.75, 3.5]),
        ("altrep_sorted_string_wrapper", ["a", "b", "c"]),
    ],
)
def test_altrep_sort_wrappers(name: str, expected: list[object]) -> None:
    assert read_r_object(DATA / f"{name}.rds") == expected


def test_dataframe_with_altrep_columns() -> None:
    frame = read_rds(DATA / "df_with_altrep_columns.rds")
    assert frame.shape == (50_000, 2)
    assert frame["id"].iloc[0] == 1
    assert frame["id"].iloc[-1] == 50_000
    assert frame["value"].iloc[0] == "1"
    assert frame["value"].iloc[-1] == "50000"


def test_character_row_names_become_index() -> None:
    frame = read_rds(DATA / "df_character_rownames.rds")
    assert list(frame.index) == ["alpha", "beta", "gamma"]
    assert frame.loc["beta", "x"] == 20


def test_default_row_names_stay_rangeindex() -> None:
    frame = read_rds(DATA / "df_default_rownames.rds")
    assert isinstance(frame.index, pd.RangeIndex)


def test_real_difftime_hours() -> None:
    frame = read_rds(DATA / "df_difftime_hours.rds")
    assert frame["d"].dt.total_seconds().tolist() == [5400.0, 7200.0]


def test_real_ordered_factor() -> None:
    frame = read_rds(DATA / "df_ordered_factor.rds")
    assert frame["f"].cat.ordered is True
    assert list(frame["f"].cat.categories) == ["low", "medium", "high"]


def test_native_non_xdr_format() -> None:
    frame = read_rds(DATA / "df_native_format.rds")
    assert frame["a"].tolist() == [1, 2, 3]
    assert frame["b"].tolist() == ["x", "y", "z"]


def test_version2_stream() -> None:
    frame = read_rds(DATA / "df_version2.rds")
    assert frame["a"].tolist() == [1, 2, 3]


def test_real_posixlt_column() -> None:
    frame = read_rds(DATA / "df_posixlt_column.rds")
    ts = frame["when"].iloc[0]
    assert (ts.year, ts.month, ts.day, ts.hour, ts.minute) == (2024, 3, 15, 10, 30)
    assert ts.second == 45 and ts.microsecond == 500_000
    assert pd.isna(frame["when"].iloc[1])


def test_real_posixlt_standalone() -> None:
    values = read_r_object(DATA / "posixlt_standalone.rds")
    assert values[0] == pd.Timestamp("2024-03-15 10:30:45.5")
    assert pd.isna(values[1])


def test_nested_general_object() -> None:
    result = read_r_object(DATA / "nested_general_object.rds")
    assert result["meta"] == {"created": "2026-07-12", "n": 42}
    assert result["tables"][0]["v"].tolist() == [1, 2]
    assert result["tables"][1]["w"].tolist() == ["a", "b"]
    assert result["matrix"].tolist() == [[1, 3, 5], [2, 4, 6]]


def test_intseq_with_step() -> None:
    values = read_r_object(DATA / "altrep_intseq_step.rds")
    assert values[:3] == [2, 4, 6]
    assert values[-1] == 200


def test_s4_object_becomes_slot_dict() -> None:
    result = read_r_object(DATA / "s4_person.rds")
    assert result["$r_class"] == ["Person"]
    assert result["name"] == "Ana"
    assert result["age"] == 31.0


def test_s4_object_with_nested_dataframe_slot() -> None:
    result = read_r_object(DATA / "s4_with_dataframe.rds")
    assert result["$r_class"] == ["Study"]
    assert result["title"] == "catch survey"
    assert result["data"]["x"].tolist() == [1, 2, 3]


def test_environment_becomes_dict() -> None:
    result = read_r_object(DATA / "environment_simple.rds")
    assert result["alpha"] == [1, 2, 3]
    assert result["beta"] == "hello"
    assert result["gamma"]["v"].tolist() == [1.5, 2.5]


def test_environment_shared_reference_alignment() -> None:
    result = read_r_object(DATA / "environment_shared_ref.rds")
    assert result["first"]["beta"] == "hello"
    assert result["second"]["alpha"] == [1, 2, 3]


def test_environment_with_namespace_parent() -> None:
    result = read_r_object(DATA / "environment_ns_parent.rds")
    assert result == {"value": 42}


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("closure_function", "closure"),
        ("language_call", "language call"),
        ("formula_object", "language call"),
    ],
)
def test_code_objects_fail_with_named_type(name: str, message: str) -> None:
    from rdsframe import UnsupportedRDS

    with pytest.raises(UnsupportedRDS, match=message):
        read_r_object(DATA / f"{name}.rds")
