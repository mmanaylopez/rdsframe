from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest
from conftest import dataframe, integers, rds, strings, vectors

from rdsframe import diff_rds, validate_rds, write_rds
from rdsframe.cli import main as cli_main


def _entry_kinds(report) -> list[str]:
    return [entry.kind for entry in report.entries]


def test_validate_ok_tabular(tmp_path: Path) -> None:
    write_rds(pd.DataFrame({"x": [1, 2], "s": ["a", "b"]}), tmp_path / "ok.rds")
    report = validate_rds(tmp_path / "ok.rds")
    assert report.ok
    assert report.errors == ()
    assert len(report.tables) == 1
    assert report.tables[0].columns == 2
    payload = report.to_dict()
    assert payload["ok"] is True and payload["compression"] == "gzip"


def test_validate_corrupt_compressed(tmp_path: Path) -> None:
    healthy = tmp_path / "ok.rds"
    write_rds(pd.DataFrame({"x": list(range(200))}), healthy)
    corrupt = tmp_path / "corrupt.rds"
    corrupt.write_bytes(healthy.read_bytes()[:60])
    report = validate_rds(corrupt)
    assert not report.ok
    assert any(issue.code == "invalid" for issue in report.errors)


def test_validate_non_tabular_root(tmp_path: Path) -> None:
    target = tmp_path / "vector.rds"
    target.write_bytes(rds(strings(["a", "b", None])))
    report = validate_rds(target)
    assert report.ok  # readable, just not tabular
    assert any(issue.code == "non-tabular" for issue in report.issues)
    assert report.tables == ()


def test_validate_trailing_data(tmp_path: Path) -> None:
    target = tmp_path / "trailing.rds"
    target.write_bytes(rds(strings(["a"])) + b"junk")
    report = validate_rds(target)
    assert report.ok
    assert any(issue.code == "trailing-data" for issue in report.warnings)


def test_validate_list_column_warning(tmp_path: Path) -> None:
    payload = dataframe(
        [integers([1, 2]), vectors([integers([1]), strings(["x"])])],
        ["id", "payload"],
    )
    target = tmp_path / "listcol.rds"
    target.write_bytes(rds(payload))
    report = validate_rds(target)
    assert report.ok
    warning = next(issue for issue in report.warnings if issue.code == "list-column")
    assert warning.column == "payload"


def test_validate_rejects_other_containers(tmp_path: Path) -> None:
    rdata = tmp_path / "workspace.rdata"
    rdata.write_bytes(gzip.compress(b"RDX3\nrest"))
    report = validate_rds(rdata)
    assert not report.ok
    assert report.errors[0].code == "rdata-container"

    ascii_rds = tmp_path / "ascii.rds"
    ascii_rds.write_bytes(b"A\nrest of file")
    assert validate_rds(ascii_rds).errors[0].code == "ascii-serialization"

    junk = tmp_path / "junk.rds"
    junk.write_bytes(b"\x00\x01\x02\x03 not an rds at all")
    assert validate_rds(junk).errors[0].code == "unrecognized-container"


def test_diff_identical_files(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1, 2], "v": [1.5, float("nan")], "s": ["a", None]})
    write_rds(frame, tmp_path / "a.rds")
    write_rds(frame, tmp_path / "b.rds")
    structural = diff_rds(tmp_path / "a.rds", tmp_path / "b.rds")
    assert structural.identical

    pytest.importorskip("pyarrow")
    with_content = diff_rds(tmp_path / "a.rds", tmp_path / "b.rds", content=True)
    # Float NaN must compare equal to NaN: both files miss the same value.
    assert with_content.identical


def test_diff_structural_changes(tmp_path: Path) -> None:
    before = pd.DataFrame(
        {
            "keep": [1, 2],
            "quantity": pd.array([1, 2], dtype="Int32"),
            "gone": [True, False],
            "region": pd.Categorical(["n", "s"], categories=["n", "s"]),
        }
    )
    after = pd.DataFrame(
        {
            "keep": [1, 2, 3],
            "quantity": [1.0, 2.0, 3.0],
            "region": pd.Categorical(["n", "s", "e"], categories=["n", "s", "e"]),
            "brand_new": ["x", "y", "z"],
        }
    )
    write_rds(before, tmp_path / "before.rds")
    write_rds(after, tmp_path / "after.rds", compress="xz")
    report = diff_rds(tmp_path / "before.rds", tmp_path / "after.rds")
    kinds = _entry_kinds(report)
    assert "compression_changed" in kinds
    assert "rows_changed" in kinds
    assert "column_removed" in kinds and "column_added" in kinds
    assert "type_changed" in kinds  # quantity: integer -> double
    levels = next(e for e in report.entries if e.kind == "levels_changed")
    assert levels.column == "region" and "added" in (levels.detail or "")
    assert not report.identical


def test_diff_content_counts_differing_rows(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    before = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "price": [1.0, 2.0, 3.0, 4.0],
            "label": ["a", "b", "c", None],
        }
    )
    after = before.copy()
    after.loc[1, "price"] = 2.5
    after.loc[3, "price"] = 9.0
    after.loc[0, "label"] = "CHANGED"
    write_rds(before, tmp_path / "before.rds")
    write_rds(after, tmp_path / "after.rds")
    report = diff_rds(tmp_path / "before.rds", tmp_path / "after.rds", content=True)
    changed = {
        entry.column: entry.detail
        for entry in report.entries
        if entry.kind == "values_changed"
    }
    assert changed == {
        "price": "2 of 4 rows differ",
        "label": "1 of 4 rows differ",
    }


def test_diff_multiple_tables(tmp_path: Path) -> None:
    write_rds(
        {"one": pd.DataFrame({"x": [1]}), "two": pd.DataFrame({"y": [2]})},
        tmp_path / "a.rds",
    )
    write_rds(
        {"one": pd.DataFrame({"x": [1]}), "three": pd.DataFrame({"z": [3]})},
        tmp_path / "b.rds",
    )
    report = diff_rds(tmp_path / "a.rds", tmp_path / "b.rds")
    kinds = _entry_kinds(report)
    assert kinds.count("table_removed") == 1 and kinds.count("table_added") == 1


def test_cli_validate_and_diff(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    frame = pd.DataFrame({"x": [1, 2]})
    write_rds(frame, tmp_path / "a.rds")
    write_rds(frame, tmp_path / "b.rds")
    write_rds(pd.DataFrame({"x": [1, 2, 3]}), tmp_path / "c.rds")

    assert cli_main(["validate", str(tmp_path / "a.rds")]) == 0
    assert "OK" in capsys.readouterr().out

    assert cli_main(["diff", str(tmp_path / "a.rds"), str(tmp_path / "b.rds")]) == 0
    assert "identical" in capsys.readouterr().out

    assert cli_main(["diff", str(tmp_path / "a.rds"), str(tmp_path / "c.rds")]) == 1
    assert "rows_changed" in capsys.readouterr().out

    corrupt = tmp_path / "corrupt.rds"
    corrupt.write_bytes((tmp_path / "a.rds").read_bytes()[:40])
    assert cli_main(["validate", str(corrupt)]) == 1
    assert cli_main(["validate", str(tmp_path / "missing.rds")]) == 2
