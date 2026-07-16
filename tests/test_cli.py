from __future__ import annotations

import json
from pathlib import Path

import pytest

import rdsframe
from rdsframe.cli import main


def test_inspect_cli(sample_rds: Path, capsys) -> None:
    assert main(["inspect", str(sample_rds)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["container"] == "rds"
    assert output["fast_supported"] is True


def test_version_cli(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])
    assert rdsframe.__version__ in capsys.readouterr().out


def test_cli_reports_domain_errors(capsys) -> None:
    assert main(["inspect", "missing.rds"]) == 2
    assert "missing.rds" in capsys.readouterr().err


def test_dump_cli_tree_mode_named_list(named_list_rds: Path, capsys) -> None:
    assert main(["dump", str(named_list_rds)]) == 0
    output = capsys.readouterr().out
    assert "<named list, 3 entries>" in output
    assert "letters:" in output
    assert "'x'" in output
    assert "numbers:" in output


def test_dump_cli_tree_mode_dataframe(sample_rds: Path, capsys) -> None:
    assert main(["dump", str(sample_rds)]) == 0
    output = capsys.readouterr().out
    assert "<data.frame 3 rows x 5 cols>" in output
    assert "id:" in output


def test_dump_cli_truncates_with_max_items(named_list_rds: Path, capsys) -> None:
    assert main(["dump", str(named_list_rds), "--max-items", "1"]) == 0
    output = capsys.readouterr().out
    assert "(+2 more entries)" in output


def test_dump_cli_json_mode(named_list_rds: Path, capsys) -> None:
    assert main(["dump", str(named_list_rds), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["letters"] == ["x", "y", "z"]
    assert payload["numbers"] == [1, 2, 3]
    assert payload["nothing"] is None


def test_dump_cli_json_dataframe(sample_rds: Path, capsys) -> None:
    assert main(["dump", str(sample_rds), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["$r_type"] == "data.frame"
    assert payload["rows"] == 3
    assert payload["columns"]["id"] == [1, 2, None]
    assert payload["columns"]["label"] == ["á", "á", None]


def test_dump_cli_rejects_bad_limits(named_list_rds: Path, capsys) -> None:
    assert main(["dump", str(named_list_rds), "--max-items", "0"]) == 2
    assert "--max-items" in capsys.readouterr().err


def test_convert_cli_exposes_safe_loss_policies(
    advanced_types_rds: Path, tmp_path: Path
) -> None:
    assert (
        main(
            [
                "convert",
                str(advanced_types_rds),
                str(tmp_path / "output"),
                "--list-column-mode",
                "json",
                "--invalid-timestamp",
                "null",
                "--posixct-mode",
                "utc_naive",
                "--max-tables",
                "4",
            ]
        )
        == 0
    )


def test_list_and_select_cli_reuse_catalog(
    multi_frame_rds: Path, tmp_path: Path, capsys
) -> None:
    catalog_path = tmp_path / "tables.json"
    assert (
        main(
            [
                "list",
                str(multi_frame_rds),
                "--catalog",
                str(catalog_path),
            ]
        )
        == 0
    )
    listing = capsys.readouterr().out
    assert "numbers" in listing and "labels" in listing

    output = tmp_path / "selected"
    assert (
        main(
            [
                "convert",
                str(multi_frame_rds),
                str(output),
                "--table-name",
                "labels",
                "--catalog",
                str(catalog_path),
            ]
        )
        == 0
    )
    assert (output / "multiple__labels.parquet").is_file()
    assert not (output / "multiple__numbers.parquet").exists()


def test_list_cli_opt_in_cache(multi_frame_rds: Path, capsys) -> None:
    assert main(["list", str(multi_frame_rds), "--cache"]) == 0
    assert "numbers" in capsys.readouterr().out
    assert multi_frame_rds.with_suffix(".rdsframe.json").is_file()
