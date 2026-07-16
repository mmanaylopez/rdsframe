from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rdsframe import read_rds_arrow, to_parquet


def test_read_rds_arrow_returns_table_without_pandas(sample_rds: Path) -> None:
    table = read_rds_arrow(sample_rds)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 3
    assert table.num_columns == 5
    assert table["id"].to_pylist() == [1, 2, None]
    assert table["label"].to_pylist() == ["á", "á", None]
    assert pa.types.is_dictionary(table.schema.field("level").type)


def test_read_rds_arrow_accepts_bytes_and_named_frame_lists(
    multi_frame_rds: Path,
) -> None:
    tables = read_rds_arrow(multi_frame_rds.read_bytes())
    assert list(tables) == ["numbers", "labels"]
    assert tables["numbers"]["id"].to_pylist() == [1, 2]
    assert tables["labels"]["label"].to_pylist() == ["a", "b"]


def test_pyarrow_parquet_engine_needs_no_duckdb(
    sample_rds: Path, tmp_path: Path
) -> None:
    output = tmp_path / "parquet"
    results = to_parquet(sample_rds, output, engine="pyarrow")
    assert [(item.name, item.rows, item.columns) for item in results] == [
        ("data", 3, 5)
    ]
    table = pq.read_table(results[0].path)
    assert table["label"].to_pylist() == ["á", "á", None]
    assert not list(output.glob(".rdsframe-*"))


def test_pyarrow_parquet_preserves_multi_table_names(
    multi_frame_rds: Path, tmp_path: Path
) -> None:
    results = to_parquet(
        multi_frame_rds,
        tmp_path,
        engine="pyarrow",
        tables=["labels"],
    )
    assert [item.name for item in results] == ["labels"]
    assert pq.read_table(results[0].path)["label"].to_pylist() == ["a", "b"]

