from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rdsframe import (
    InvalidRDS,
    UnsupportedRDS,
    inspect_r_file,
    read_r_object,
    read_rds,
    read_rds_dataframe,
    to_parquet,
)


@pytest.mark.parametrize("fixture_name", ["sample_rds", "compressed_rds"])
def test_read_dataframe(request: pytest.FixtureRequest, fixture_name: str) -> None:
    path: Path = request.getfixturevalue(fixture_name)
    frame = read_rds(path)
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (3, 5)
    assert frame["id"].dtype.name == "Int32"
    assert frame["id"].tolist() == [1, 2, pd.NA]
    assert str(frame["date"].iloc[0].date()) == "1970-01-01"
    assert frame["active"].dtype.name == "boolean"
    assert frame["active"].tolist() == [True, False, pd.NA]
    assert list(frame["level"].cat.categories) == ["low", "high"]
    assert frame["label"].tolist() == ["á", "á", None]


def test_selective_columns_by_index(sample_rds: Path) -> None:
    frame = read_rds(sample_rds, columns=[0, 4])
    assert list(frame.columns) == ["id", "label"]
    assert frame["id"].tolist() == [1, 2, pd.NA]
    assert frame["label"].tolist() == ["á", "á", None]


def test_selective_columns_by_name_preserves_request_order(sample_rds: Path) -> None:
    frame = read_rds(sample_rds, columns=["label", "id"])
    assert list(frame.columns) == ["label", "id"]
    assert list(frame.columns) != list(read_rds(sample_rds).columns[:2])


def test_selective_columns_matches_full_read(sample_rds: Path) -> None:
    full = read_rds(sample_rds)
    subset = read_rds(sample_rds, columns=["level", "active"])
    assert subset["level"].tolist() == full["level"].tolist()
    assert subset["active"].tolist() == full["active"].tolist()


def test_selective_columns_rejects_unknown_name(sample_rds: Path) -> None:
    with pytest.raises(ValueError, match="column name not found"):
        read_rds(sample_rds, columns=["missing"])


def test_selective_columns_rejects_out_of_range_index(sample_rds: Path) -> None:
    with pytest.raises(ValueError, match="column index out of range"):
        read_rds(sample_rds, columns=[99])


def test_selective_columns_rejects_empty_selection(sample_rds: Path) -> None:
    with pytest.raises(ValueError, match="columns cannot be empty"):
        read_rds(sample_rds, columns=[])


def test_selective_columns_rejects_string_as_whole_sequence(sample_rds: Path) -> None:
    with pytest.raises(TypeError, match="sequence of column names"):
        read_rds(sample_rds, columns="id")  # type: ignore[arg-type]


def test_selective_columns_requires_single_dataframe_root(multi_frame_rds: Path) -> None:
    with pytest.raises(UnsupportedRDS, match=r"single data\.frame"):
        read_rds(multi_frame_rds, columns=[0])
    with pytest.raises(UnsupportedRDS, match=r"single data\.frame"):
        read_rds(multi_frame_rds, columns=["numbers"])


def test_read_r_object_named_list(named_list_rds: Path) -> None:
    result = read_r_object(named_list_rds)
    assert isinstance(result, dict)
    assert result["letters"] == ["x", "y", "z"]
    assert result["numbers"] == [1, 2, 3]
    assert result["nothing"] is None


def test_read_r_object_unnamed_nested_list(unnamed_nested_list_rds: Path) -> None:
    result = read_r_object(unnamed_nested_list_rds)
    assert isinstance(result, list)
    assert len(result) == 3
    for inner in result:
        assert isinstance(inner, list)
        assert inner == [[1, 2], "a"]


def test_read_r_object_mixed_dataframe_factor_and_matrix(mixed_object_rds: Path) -> None:
    result = read_r_object(mixed_object_rds)
    assert isinstance(result, dict)
    assert isinstance(result["table"], pd.DataFrame)
    assert result["table"]["id"].tolist() == [1, 2]
    assert result["level"] == ["low", "high", "low"]
    grid = result["grid"]
    assert grid.shape == (2, 2)
    # R fills matrices column-major: matrix(1:4, nrow=2) is [[1,3],[2,4]].
    assert grid.tolist() == [[1, 3], [2, 4]]
    assert result["empty"] is None


def test_read_r_object_still_reads_a_plain_dataframe(sample_rds: Path) -> None:
    result = read_r_object(sample_rds)
    assert isinstance(result, pd.DataFrame)


def test_read_r_object_standalone_posixlt(standalone_posixlt_rds: Path) -> None:
    result = read_r_object(standalone_posixlt_rds)
    assert isinstance(result, pd.Timestamp)
    assert (result.year, result.month, result.day) == (2000, 1, 1)


def test_read_rds_from_bytes(sample_rds: Path) -> None:
    payload = sample_rds.read_bytes()
    frame = read_rds(payload)
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (3, 5)
    assert frame["label"].tolist() == ["á", "á", None]


def test_read_rds_from_compressed_bytes(compressed_rds: Path) -> None:
    frame = read_rds(compressed_rds.read_bytes())
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (3, 5)


def test_read_rds_from_binary_stream(sample_rds: Path) -> None:
    from io import BytesIO

    stream = BytesIO(sample_rds.read_bytes())
    frame = read_rds(stream)
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (3, 5)
    assert not stream.closed  # caller-owned streams stay open


def test_read_rds_from_bytes_with_column_selection(sample_rds: Path) -> None:
    frame = read_rds(sample_rds.read_bytes(), columns=["label", "id"])
    assert list(frame.columns) == ["label", "id"]


def test_read_r_object_from_bytes(named_list_rds: Path) -> None:
    result = read_r_object(named_list_rds.read_bytes())
    assert result["numbers"] == [1, 2, 3]


def test_read_rds_rejects_text_stream(sample_rds: Path) -> None:
    from io import StringIO

    with pytest.raises(TypeError, match="binary mode"):
        read_rds(StringIO("not binary"))


def test_materialize_uncompressed_passthrough(sample_rds: Path) -> None:
    from rdsframe import materialize_uncompressed

    assert materialize_uncompressed(sample_rds) == sample_rds


def test_materialize_uncompressed_decompresses(
    sample_rds: Path, compressed_rds: Path, tmp_path: Path
) -> None:
    from rdsframe import inspect_r_file, materialize_uncompressed

    target = tmp_path / "expanded" / "plain.rds"
    result = materialize_uncompressed(compressed_rds, target)
    assert result == target
    assert inspect_r_file(target).compression == "none"
    assert target.read_bytes() == sample_rds.read_bytes()
    frame = read_rds(target)
    assert frame.shape == (3, 5)


def test_materialize_uncompressed_to_temp(compressed_rds: Path) -> None:
    from rdsframe import inspect_r_file, materialize_uncompressed

    result = materialize_uncompressed(compressed_rds)
    try:
        assert result != compressed_rds
        assert inspect_r_file(result).compression == "none"
    finally:
        result.unlink(missing_ok=True)


def test_read_r_object_invalid_string_backend(named_list_rds: Path) -> None:
    with pytest.raises(ValueError, match="strings must be"):
        read_r_object(named_list_rds, strings="invalid")  # type: ignore[arg-type]


def test_progress_is_monotonic_enough(sample_rds: Path) -> None:
    events: list[int] = []
    read_rds_dataframe(sample_rds, progress=events.append)
    assert events[0] == 0
    assert events[-1] == 100
    assert all(0 <= event <= 100 for event in events)


def test_arrow_backed_strings(sample_rds: Path) -> None:
    pytest.importorskip("pyarrow")
    frame = read_rds(sample_rds, strings="pyarrow")
    assert isinstance(frame, pd.DataFrame)
    assert getattr(frame["label"].dtype, "storage", None) == "pyarrow"
    assert frame["label"].tolist() == ["á", "á", pd.NA]


def test_invalid_string_backend(sample_rds: Path) -> None:
    with pytest.raises(ValueError, match="strings must be"):
        read_rds(sample_rds, strings="invalid")  # type: ignore[arg-type]


def test_inspect_plain_and_gzip(sample_rds: Path, compressed_rds: Path) -> None:
    plain = inspect_r_file(sample_rds)
    compressed = inspect_r_file(compressed_rds)
    assert plain.container == compressed.container == "rds"
    assert plain.serialization == compressed.serialization == "xdr"
    assert plain.compression == "none"
    assert compressed.compression == "gzip"
    assert plain.fast_supported and compressed.fast_supported


def test_truncated_input_is_rejected(sample_rds: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken.rds"
    broken.write_bytes(sample_rds.read_bytes()[:-7])
    with pytest.raises(InvalidRDS):
        read_rds(broken)


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        read_rds("does-not-exist.rds")


@pytest.mark.parametrize(
    ("stage_max_columns", "stage_max_bytes"),
    [(1, 1_000_000), (2, 1_000_000), (16, 1)],
)
def test_streaming_parquet_is_queryable(
    sample_rds: Path,
    tmp_path: Path,
    stage_max_columns: int,
    stage_max_bytes: int,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    output = tmp_path / "parquet"
    tables = to_parquet(
        sample_rds,
        output,
        memory_limit="256MB",
        stage_max_columns=stage_max_columns,
        stage_max_bytes=stage_max_bytes,
    )
    assert len(tables) == 1
    assert tables[0].rows == 3
    assert tables[0].columns == 5
    result = duckdb.sql(
        f"SELECT id, level, label FROM read_parquet('{tables[0].path.as_posix()}') "
        "WHERE active IS TRUE"
    ).fetchall()
    assert result == [(1, "low", "á")]
    assert not list(output.glob(".rdsframe-*"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stage_max_columns": 0}, "stage_max_columns"),
        ({"stage_max_bytes": 0}, "stage_max_bytes"),
        ({"gc_collect_every": -1}, "gc_collect_every"),
    ],
)
def test_invalid_staging_configuration(
    sample_rds: Path, tmp_path: Path, kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        to_parquet(sample_rds, tmp_path, **kwargs)
