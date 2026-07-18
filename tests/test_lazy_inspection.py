from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pytest

from rdsframe import (
    RDSDataset,
    inspect_rds,
    open_rds,
    read_rds_duckdb,
    read_rds_polars,
)
from rdsframe._core import Reader


def test_metadata_inspection_reports_schema_without_materializing_columns(
    sample_rds: Path,
) -> None:
    with patch.object(
        Reader,
        "numeric_array",
        side_effect=AssertionError("metadata inspection materialized a column"),
    ):
        inspection = inspect_rds(sample_rds)

    assert inspection.rows == 3
    assert inspection.columns == 5
    assert inspection.compression == "none"
    assert not inspection.statistics_complete
    assert not inspection.estimate_complete
    table = inspection.tables[0]
    assert table.column_names == ("id", "date", "active", "level", "label")
    assert [column.r_type for column in table.schema] == [
        "integer",
        "double",
        "logical",
        "integer",
        "character",
    ]
    assert [column.logical_type for column in table.schema] == [
        "integer",
        "date",
        "logical",
        "factor",
        "character",
    ]
    assert table.schema[3].factor
    assert table.schema[3].levels == ("low", "high")
    assert table.schema[0].estimated_bytes == 12
    assert table.schema[4].estimated_bytes is None


def test_scan_inspection_adds_exact_missing_and_buffer_statistics(
    sample_rds: Path,
) -> None:
    inspection = inspect_rds(sample_rds, mode="scan")
    assert inspection.statistics_complete
    assert inspection.estimate_complete
    schema = inspection.tables[0].schema
    assert [column.missing_count for column in schema] == [1, 1, 1, 1, 1]
    assert all(column.data_bytes is not None for column in schema)
    assert schema[3].arrow_type is not None
    assert "dictionary" in schema[3].arrow_type


def test_inspection_rejects_unknown_mode(sample_rds: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        inspect_rds(sample_rds, mode="deep")  # type: ignore[arg-type]


def test_open_rds_defers_then_projects_and_collects(sample_rds: Path) -> None:
    dataset = open_rds(sample_rds)
    assert isinstance(dataset, RDSDataset)
    assert dataset.shape == (3, 5)
    assert dataset.columns == ("id", "date", "active", "level", "label")

    selected = dataset[["label", "id"]]
    assert selected.columns == ("label", "id")
    frame = selected.collect()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["label", "id"]
    assert selected.head(2)["label"].tolist() == ["á", "á"]
    assert "columns=[label, id]" in repr(selected)


def test_lazy_dataset_selects_named_table(multi_frame_rds: Path) -> None:
    dataset = open_rds(multi_frame_rds)
    assert [table.name for table in dataset.tables] == ["numbers", "labels"]
    labels = dataset.table("labels")
    assert labels.columns == ("label",)
    assert labels.collect()["label"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="select one"):
        _ = dataset.columns


def test_lazy_dataset_validates_projection(sample_rds: Path) -> None:
    dataset = open_rds(sample_rds)
    with pytest.raises(ValueError, match="not found"):
        dataset.select(["missing"])
    with pytest.raises(ValueError, match="duplicates"):
        dataset.select(["id", "id"])
    with pytest.raises(TypeError, match="sequence"):
        open_rds(sample_rds, columns="id")  # type: ignore[arg-type]


def test_lazy_dataset_arrow_polars_and_duckdb_adapters(sample_rds: Path) -> None:
    pytest.importorskip("polars")
    pytest.importorskip("duckdb")
    dataset = open_rds(sample_rds).select(["id", "label"])
    arrow = dataset.to_arrow()
    assert isinstance(arrow, pa.Table)
    assert arrow.column_names == ["id", "label"]

    polars = dataset.to_polars()
    assert polars.columns == ["id", "label"]
    assert polars["label"].to_list() == ["á", "á", None]

    relation = dataset.to_duckdb()
    assert relation.filter("id = 2").fetchall() == [(2, "á")]
    connection = dataset.register_duckdb("measurements")
    assert connection.sql(
        "SELECT label FROM measurements WHERE id = 1"
    ).fetchall() == [("á",)]
    assert read_rds_duckdb(sample_rds, columns=["id"]).fetchall() == [
        (1,),
        (2,),
        (None,),
    ]


def test_read_rds_polars_preserves_named_table_lists(multi_frame_rds: Path) -> None:
    pytest.importorskip("polars")
    result = read_rds_polars(multi_frame_rds)
    assert list(result) == ["numbers", "labels"]
    assert result["numbers"]["id"].to_list() == [1, 2]
