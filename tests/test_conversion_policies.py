from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rdsframe import RDSLimitError, UnsupportedRDS, read_rds, to_parquet


def test_posixlt_becomes_wall_clock_timestamps(posixlt_rds: Path) -> None:
    frame = read_rds(posixlt_rds)
    assert pd.api.types.is_datetime64_any_dtype(frame["when"])
    ts = frame["when"].iloc[0]
    assert (ts.year, ts.month, ts.day) == (2024, 3, 15)
    assert (ts.hour, ts.minute, ts.second) == (10, 30, 45)
    assert ts.microsecond == 500000
    assert pd.isna(frame["when"].iloc[1])


def test_posixlt_to_parquet_timestamps(posixlt_rds: Path, tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    table = to_parquet(posixlt_rds, tmp_path)[0]
    rows = duckdb.sql(
        f"SELECT \"when\" FROM read_parquet('{table.path.as_posix()}')"
    ).fetchall()
    first = rows[0][0]
    assert (first.year, first.month, first.day) == (2024, 3, 15)
    assert rows[1][0] is None


def test_in_memory_complex_raw_and_arrow_list_elements(advanced_types_rds: Path) -> None:
    frame = read_rds(advanced_types_rds)
    assert isinstance(frame, pd.DataFrame)
    assert frame["z"].tolist() == [1 + 2j, 3 - 4j, 0j]
    assert frame["payload"].tolist() == [1, 2, 255]
    assert frame["nested"].tolist() == [[1, 2], "x", None]


def test_ordered_factor_preserves_order(ordered_factor_rds: Path) -> None:
    frame = read_rds(ordered_factor_rds)
    assert frame["severity"].cat.ordered is True
    assert list(frame["severity"].cat.categories) == ["low", "medium", "high"]
    assert frame["severity"].tolist() == ["low", "high", "medium"]
    assert frame["color"].cat.ordered is False


def test_ordered_factor_arrow_array_is_ordered_dictionary(
    ordered_factor_rds: Path,
) -> None:
    """`_column_to_arrow` itself must mark the dictionary as ordered.

    The `to_parquet()` pipeline stages columns through DuckDB, which already
    (independently of this fix) re-materializes any dictionary-encoded Arrow
    column as a plain string when it writes the final Parquet file -- this
    happens for ordinary unordered factors too, not just ordered ones, so
    it is a pre-existing DuckDB-staging characteristic rather than something
    this fix changes. Values must still be correct end to end (see below);
    the `ordered` flag is verified where it is actually preserved.
    """
    pytest.importorskip("pyarrow")
    from rdsframe._core import Reader, ReaderLimits, decode_header, open_rds_stream
    from rdsframe._parquet import _column_to_arrow

    with open_rds_stream(ordered_factor_rds) as (stream, _raw, _compression):
        _version, byteorder, _encoding = decode_header(stream)
        reader = Reader(stream, byteorder=byteorder, limits=ReaderLimits())
        root = reader.read_item()
    severity, color = root.value[0], root.value[1]
    ordered_array = _column_to_arrow(severity)
    unordered_array = _column_to_arrow(color)
    assert ordered_array.type.ordered is True
    assert unordered_array.type.ordered is False
    assert ordered_array.dictionary.to_pylist() == ["low", "medium", "high"]


def test_ordered_factor_to_parquet_values_are_correct(
    ordered_factor_rds: Path, tmp_path: Path
) -> None:
    duckdb = pytest.importorskip("duckdb")
    table = to_parquet(ordered_factor_rds, tmp_path)[0]
    rows = duckdb.sql(f"SELECT severity FROM read_parquet('{table.path.as_posix()}')").fetchall()
    assert [row[0] for row in rows] == ["low", "high", "medium"]


def test_difftime_becomes_timedelta(difftime_rds: Path) -> None:
    frame = read_rds(difftime_rds)
    assert pd.api.types.is_timedelta64_dtype(frame["elapsed"])
    assert frame["elapsed"].dt.total_seconds().tolist()[:2] == [86400.0, 216000.0]
    assert pd.isna(frame["elapsed"].iloc[2])


def test_difftime_to_parquet_duration_type(difftime_rds: Path, tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    table = to_parquet(difftime_rds, tmp_path)[0]
    rows = duckdb.sql(
        f"SELECT epoch(elapsed) AS s FROM read_parquet('{table.path.as_posix()}')"
    ).fetchall()
    assert rows[0][0] == 86400.0
    assert rows[1][0] == 216000.0


def test_matrix_column_raises_clear_error(matrix_column_rds: Path, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedRDS, match="matrix"):
        read_rds(matrix_column_rds)
    with pytest.raises(UnsupportedRDS, match="matrix"):
        to_parquet(matrix_column_rds, tmp_path)


def test_table_limit_fails_without_partial_results(
    multi_frame_rds: Path, tmp_path: Path
) -> None:
    output = tmp_path / "limited"
    with pytest.raises(RDSLimitError, match="No partial output"):
        to_parquet(multi_frame_rds, output, max_tables=1)
    assert not list(output.glob("*.parquet"))
    assert not list(output.glob(".rdsframe-*"))


def test_root_item_limit_fails_before_conversion(sample_rds: Path, tmp_path: Path) -> None:
    output = tmp_path / "root-limit"
    with pytest.raises(RDSLimitError, match="root object contains 5 items"):
        to_parquet(sample_rds, output, max_root_items=4)
    assert not list(output.glob("*.parquet"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_tables": 0}, "max_tables"),
        ({"max_root_items": 0}, "max_root_items"),
        ({"posixct_mode": "local"}, "posixct_mode"),
        ({"invalid_timestamp": "guess"}, "invalid_timestamp"),
        ({"list_column_mode": "silent"}, "list_column_mode"),
    ],
)
def test_invalid_conversion_policies(
    sample_rds: Path, tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        to_parquet(sample_rds, tmp_path, **kwargs)  # type: ignore[arg-type]


def test_heterogeneous_list_column_requires_explicit_policy(
    advanced_types_rds: Path, tmp_path: Path
) -> None:
    with pytest.raises(UnsupportedRDS, match="list_column_mode"):
        to_parquet(
            advanced_types_rds,
            tmp_path / "infer",
            invalid_timestamp="null",
        )


def test_invalid_timestamp_requires_explicit_coercion(
    advanced_types_rds: Path, tmp_path: Path
) -> None:
    with pytest.raises(UnsupportedRDS, match="invalid_timestamp='null'"):
        to_parquet(
            advanced_types_rds,
            tmp_path / "timestamp-error",
            list_column_mode="json",
        )


@pytest.mark.parametrize(
    ("posixct_mode", "expected_type"),
    [("preserve", "TIMESTAMP WITH TIME ZONE"), ("utc_naive", "TIMESTAMP")],
)
def test_explicit_loss_policies_and_extended_parquet_types(
    advanced_types_rds: Path,
    tmp_path: Path,
    posixct_mode: str,
    expected_type: str,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    table = to_parquet(
        advanced_types_rds,
        tmp_path / posixct_mode,
        posixct_mode=posixct_mode,  # type: ignore[arg-type]
        invalid_timestamp="null",
        list_column_mode="json",
    )[0]
    relation = duckdb.sql(f"SELECT * FROM read_parquet('{table.path.as_posix()}')")
    schema = {row[0]: row[1] for row in duckdb.sql(f"DESCRIBE {relation.sql_query()}").fetchall()}
    assert schema["when"] == expected_type
    assert schema["z"] in {
        "STRUCT(real DOUBLE, imag DOUBLE)",
        'STRUCT("real" DOUBLE, imag DOUBLE)',
    }
    assert schema["payload"] == "UTINYINT"
    rows = relation.select('nested, z.real AS real, z.imag AS imag, payload').fetchall()
    assert rows == [
        ("[1,2]", 1.0, 2.0, 1),
        ('"x"', 3.0, -4.0, 2),
        (None, 0.0, 0.0, 255),
    ]
