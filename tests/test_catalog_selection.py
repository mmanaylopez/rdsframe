from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from rdsframe import (
    RDSCatalogError,
    extract_rds_tables,
    list_rds_tables,
    to_parquet,
)
from rdsframe._core import Reader
from rdsframe._parquet import _column_to_arrow


def test_catalog_lists_named_frames_without_numeric_materialization(
    multi_frame_rds: Path,
) -> None:
    with patch.object(
        Reader,
        "numeric_array",
        side_effect=AssertionError("table listing materialized a numeric vector"),
    ):
        catalog = list_rds_tables(multi_frame_rds)
    assert [table.name for table in catalog.tables] == ["numbers", "labels"]
    assert [(table.rows, table.columns) for table in catalog.tables] == [(2, 1), (2, 1)]
    assert catalog.tables[0].column_names == ("id",)
    assert catalog.tables[1].column_names == ("label",)
    assert catalog.matches(multi_frame_rds)


def test_catalog_round_trip_and_freshness(multi_frame_rds: Path, tmp_path: Path) -> None:
    catalog = list_rds_tables(multi_frame_rds)
    catalog_path = catalog.save(tmp_path / "multiple.rdsframe.json")
    loaded = catalog.load(catalog_path)
    assert loaded == catalog
    assert loaded.matches(multi_frame_rds)

    catalog_path.write_text("not json", encoding="utf-8")
    with pytest.raises(RDSCatalogError, match="invalid RDS catalog"):
        catalog.load(catalog_path)


def test_catalog_lists_single_root_dataframe(sample_rds: Path) -> None:
    catalog = list_rds_tables(sample_rds)
    assert len(catalog.tables) == 1
    assert catalog.tables[0].name == "data"
    assert catalog.tables[0].rows == 3
    assert catalog.tables[0].columns == 5


def test_integer_selection_materializes_only_selected_frame(
    multi_frame_rds: Path, tmp_path: Path
) -> None:
    with patch("rdsframe._parquet._column_to_arrow", wraps=_column_to_arrow) as convert:
        result = extract_rds_tables(multi_frame_rds, tmp_path, [1])
    assert convert.call_count == 1
    assert [table.name for table in result] == ["labels"]
    assert duckdb.sql(
        f"SELECT * FROM read_parquet('{result[0].path.as_posix()}')"
    ).fetchall() == [("a",), ("b",)]


def test_name_selection_reuses_catalog_without_rescanning(
    multi_frame_rds: Path, tmp_path: Path
) -> None:
    catalog = list_rds_tables(multi_frame_rds)
    with patch(
        "rdsframe.api.list_rds_tables",
        side_effect=AssertionError("catalog should have been reused"),
    ):
        result = extract_rds_tables(
            multi_frame_rds,
            tmp_path,
            ["numbers"],
            catalog=catalog,
        )
    assert [table.name for table in result] == ["numbers"]
    assert duckdb.sql(
        f"SELECT * FROM read_parquet('{result[0].path.as_posix()}')"
    ).fetchall() == [(1,), (2,)]


def test_validated_catalog_stops_after_last_selected_table(
    multi_frame_rds: Path, tmp_path: Path
) -> None:
    catalog = list_rds_tables(multi_frame_rds)
    with patch(
        "rdsframe._parquet.scan_dataframe_from_header",
        side_effect=AssertionError("tables after the selection should not be scanned"),
    ):
        result = extract_rds_tables(
            multi_frame_rds,
            tmp_path,
            [0],
            catalog=catalog,
        )
    assert [table.name for table in result] == ["numbers"]


def test_name_selection_can_build_catalog_automatically(
    multi_frame_rds: Path, tmp_path: Path
) -> None:
    result = to_parquet(multi_frame_rds, tmp_path, tables=["labels"])
    assert [table.name for table in result] == ["labels"]


def test_stale_catalog_and_invalid_selectors_are_rejected(
    multi_frame_rds: Path, tmp_path: Path
) -> None:
    catalog = list_rds_tables(multi_frame_rds)
    stale = replace(catalog, mtime_ns=catalog.mtime_ns - 1)
    with pytest.raises(RDSCatalogError, match="stale"):
        extract_rds_tables(
            multi_frame_rds,
            tmp_path / "stale",
            ["labels"],
            catalog=stale,
        )
    with pytest.raises(RDSCatalogError, match="out of range"):
        extract_rds_tables(multi_frame_rds, tmp_path / "range", [99])
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_rds_tables(multi_frame_rds, tmp_path / "empty", [])
    with pytest.raises(TypeError, match="boolean"):
        extract_rds_tables(multi_frame_rds, tmp_path / "bool", [True])
