from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rdsframe import RDSCatalogError, open_rds_dataset, write_rds


@pytest.fixture
def yearly_files(tmp_path: Path) -> Path:
    """Three same-schema files named like survey waves."""
    for year, offset in [(2021, 0), (2022, 10), (2023, 20)]:
        frame = pd.DataFrame(
            {
                "id": [offset + 1, offset + 2],
                "amount": [offset + 0.5, offset + 1.5],
                "region": pd.Categorical(
                    ["north", "south"], categories=["north", "south"]
                ),
            }
        )
        write_rds(frame, tmp_path / f"survey_{year}.rds")
    return tmp_path


def test_glob_strict_collection(yearly_files: Path) -> None:
    dataset = open_rds_dataset(str(yearly_files / "survey_*.rds"))
    assert len(dataset.files) == 3
    assert dataset.rows == 6
    assert dataset.columns == ("id", "amount", "region")
    assert [column.name for column in dataset.schema] == ["id", "amount", "region"]

    frame = dataset.to_pandas()
    assert len(frame) == 6
    assert frame["id"].tolist() == [1, 2, 11, 12, 21, 22]
    # Identical level sets across files: categorical typing is preserved.
    assert isinstance(frame["region"].dtype, pd.CategoricalDtype)


def test_directory_and_explicit_paths(yearly_files: Path) -> None:
    from_dir = open_rds_dataset(yearly_files)
    assert len(from_dir.files) == 3
    explicit = open_rds_dataset(
        [yearly_files / "survey_2023.rds", yearly_files / "survey_2021.rds"]
    )
    assert [path.name for path in explicit.files] == [
        "survey_2023.rds",
        "survey_2021.rds",
    ]
    with pytest.raises(FileNotFoundError):
        open_rds_dataset(str(yearly_files / "nothing_*.rds"))


def test_source_and_partition_columns(yearly_files: Path) -> None:
    dataset = open_rds_dataset(
        str(yearly_files / "survey_*.rds"),
        source_column="origin",
        partitions=r"survey_(?P<year>\d{4})",
    )
    assert dataset.columns == ("id", "amount", "region", "year", "origin")
    frame = dataset.to_pandas()
    assert frame["year"].tolist() == ["2021"] * 2 + ["2022"] * 2 + ["2023"] * 2
    assert frame["origin"][0].endswith("survey_2021.rds")

    with pytest.raises(ValueError, match="named groups"):
        open_rds_dataset(str(yearly_files / "survey_*.rds"), partitions=r"\d{4}")
    bad = open_rds_dataset(
        str(yearly_files / "survey_*.rds"), partitions=r"(?P<code>census_\d+)"
    )
    with pytest.raises(ValueError, match="does not match"):
        bad.to_pandas()


def test_arrow_paths(yearly_files: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    dataset = open_rds_dataset(
        str(yearly_files / "survey_*.rds"), source_column="origin"
    )
    per_file = list(dataset.iter_arrow())
    assert len(per_file) == 3
    assert all(table.num_rows == 2 for _path, table in per_file)

    combined = dataset.to_arrow()
    assert combined.num_rows == 6
    assert combined.column("id").to_pylist() == [1, 2, 11, 12, 21, 22]
    assert pa.types.is_dictionary(combined.schema.field("region").type)

    reader = dataset.to_record_batch_reader()
    streamed = reader.read_all()
    assert streamed.num_rows == 6
    assert streamed.schema == combined.schema


def test_union_mode_fills_and_strict_rejects(tmp_path: Path) -> None:
    write_rds(pd.DataFrame({"id": [1, 2], "extra": [1.5, 2.5]}), tmp_path / "a.rds")
    write_rds(pd.DataFrame({"id": [3, 4]}), tmp_path / "b.rds")

    with pytest.raises(RDSCatalogError, match="schema_mode='union'"):
        open_rds_dataset(str(tmp_path / "*.rds")).to_pandas()

    dataset = open_rds_dataset(str(tmp_path / "*.rds"), schema_mode="union")
    frame = dataset.to_pandas()
    assert frame["extra"].tolist()[:2] == [1.5, 2.5]
    assert frame["extra"].isna().tolist() == [False, False, True, True]

    pytest.importorskip("pyarrow")
    combined = dataset.to_arrow()
    assert combined.column("extra").null_count == 2


@pytest.mark.parametrize("mode", ["strict", "union"])
def test_type_conflict_is_always_an_error(tmp_path: Path, mode: str) -> None:
    write_rds(pd.DataFrame({"x": [1, 2]}), tmp_path / "a.rds")
    write_rds(pd.DataFrame({"x": ["one", "two"]}), tmp_path / "b.rds")
    dataset = open_rds_dataset(str(tmp_path / "*.rds"), schema_mode=mode)  # type: ignore[arg-type]
    with pytest.raises(RDSCatalogError):
        _ = dataset.schema


def test_differing_factor_levels_decode_to_strings(tmp_path: Path) -> None:
    write_rds(
        pd.DataFrame({"g": pd.Categorical(["a"], categories=["a", "b"])}),
        tmp_path / "a.rds",
    )
    write_rds(
        pd.DataFrame({"g": pd.Categorical(["c"], categories=["a", "b", "c"])}),
        tmp_path / "b.rds",
    )
    dataset = open_rds_dataset(str(tmp_path / "*.rds"))
    assert dataset.schema[0].levels == ()
    frame = dataset.to_pandas()
    assert frame["g"].tolist() == ["a", "c"]
    assert frame["g"].dtype == object

    pa = pytest.importorskip("pyarrow")
    combined = dataset.to_arrow()
    assert not pa.types.is_dictionary(combined.schema.field("g").type)
    assert combined.column("g").to_pylist() == ["a", "c"]


def test_multi_table_files_need_selector(tmp_path: Path) -> None:
    tables = {
        "sales": pd.DataFrame({"v": [1.0]}),
        "costs": pd.DataFrame({"v": [2.0]}),
    }
    write_rds(tables, tmp_path / "a.rds")
    write_rds(tables, tmp_path / "b.rds")
    with pytest.raises(RDSCatalogError, match="select one"):
        _ = open_rds_dataset(str(tmp_path / "*.rds")).schema
    dataset = open_rds_dataset(str(tmp_path / "*.rds"), table="costs")
    assert dataset.to_pandas()["v"].tolist() == [2.0, 2.0]


def test_column_projection(yearly_files: Path) -> None:
    dataset = open_rds_dataset(
        str(yearly_files / "survey_*.rds"), columns=["amount", "id"]
    )
    frame = dataset.to_pandas()
    assert list(frame.columns) == ["amount", "id"]
    with pytest.raises(ValueError, match="not found"):
        _ = open_rds_dataset(
            str(yearly_files / "survey_*.rds"), columns=["ghost"]
        ).schema


def test_head_reads_only_needed_files(yearly_files: Path) -> None:
    dataset = open_rds_dataset(str(yearly_files / "survey_*.rds"))
    head = dataset.head(3)
    assert len(head) == 3
    assert head["id"].tolist() == [1, 2, 11]


def test_to_parquet_per_file(yearly_files: Path, tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    out = tmp_path / "parquet_out"
    dataset = open_rds_dataset(
        str(yearly_files / "survey_*.rds"),
        partitions=r"survey_(?P<year>\d{4})",
    )
    results = dataset.to_parquet(out)
    assert [table.name for table in results] == [
        "survey_2021",
        "survey_2022",
        "survey_2023",
    ]
    first = pq.read_table(results[0].path)
    assert first.num_rows == 2
    assert first.column("year").to_pylist() == ["2021", "2021"]


def test_parallel_workers(yearly_files: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    dataset = open_rds_dataset(str(yearly_files / "survey_*.rds"), workers=2)
    assert len(dataset.catalogs()) == 3  # parallel catalog scan
    results = dataset.to_parquet(tmp_path / "par_out", workers=2)
    assert len(results) == 3
    assert sum(table.rows for table in results) == 6


def test_to_polars(yearly_files: Path) -> None:
    pytest.importorskip("polars")
    dataset = open_rds_dataset(str(yearly_files / "survey_*.rds"))
    frame = dataset.to_polars()
    assert frame.shape == (6, 3)


def test_single_table_inside_named_list(tmp_path: Path) -> None:
    """One catalog table does not mean the root is a data.frame: a named
    list holding a single data.frame used to break the selective path."""
    for year in (2021, 2022):
        frame = pd.DataFrame({"id": [year, year + 1], "amount": [1.0, 2.0]})
        write_rds({"wave": frame}, tmp_path / f"list_{year}.rds")
    dataset = open_rds_dataset(str(tmp_path / "list_*.rds"))
    frame = dataset.to_pandas()
    assert frame["id"].tolist() == [2021, 2022, 2022, 2023]


def test_mixed_timezones_never_become_object(tmp_path: Path) -> None:
    stamps = pd.to_datetime(["2020-06-01 12:00", "2020-06-02 12:00"])
    utc = pd.DataFrame({"id": [1, 2], "t": stamps.tz_localize("UTC")})
    lima = pd.DataFrame(
        {"id": [3, 4], "t": stamps.tz_localize("UTC").tz_convert("America/Lima")}
    )
    write_rds(utc, tmp_path / "utc.rds")
    write_rds(lima, tmp_path / "lima.rds")

    preserve = open_rds_dataset([tmp_path / "utc.rds", tmp_path / "lima.rds"])
    with pytest.raises(RDSCatalogError, match="mixes datetime dtypes"):
        preserve.to_pandas()

    normalized = open_rds_dataset(
        [tmp_path / "utc.rds", tmp_path / "lima.rds"],
        posixct_mode="utc_naive",
    )
    frame = normalized.to_pandas()
    assert str(frame["t"].dtype).startswith("datetime64")
    assert "UTC" not in str(frame["t"].dtype)  # naive, same instants
    assert frame["t"].tolist() == list(stamps) * 2


def test_projection_skips_incompatible_unselected_columns(tmp_path: Path) -> None:
    """Selecting only compatible columns must not fail because an
    unselected column mixes types across files."""
    write_rds(
        pd.DataFrame({"id": [1, 2], "x": [10, 20]}), tmp_path / "ints.rds"
    )
    write_rds(
        pd.DataFrame({"id": [3, 4], "x": ["a", "b"]}), tmp_path / "strs.rds"
    )
    files = [tmp_path / "ints.rds", tmp_path / "strs.rds"]

    projected = open_rds_dataset(files, columns=["id"], schema_mode="union")
    assert projected.to_pandas()["id"].tolist() == [1, 2, 3, 4]

    conflicting = open_rds_dataset(files, columns=["x"], schema_mode="union")
    with pytest.raises(RDSCatalogError, match="never coerce"):
        conflicting.to_pandas()
