from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdsframe import (
    InvalidRDS,
    UnsupportedRDS,
    inspect_r_file,
    read_rdata,
    read_rds,
)
from rdsframe.cli import main as cli_main

R450 = Path(__file__).parent / "data" / "r450"


def _check_df1(frame: pd.DataFrame) -> None:
    """df1 in the golden workspaces: 4 rows of int/double/character."""
    assert list(frame.columns) == ["id", "value", "label"]
    assert frame["id"].tolist() == [1, 2, 3, 4]
    assert frame["value"][0] == 1.5 and np.isnan(frame["value"][1])
    assert frame["label"].tolist() == ["a", "b", None, "día"]


def test_reads_two_frame_workspace() -> None:
    result = read_rdata(R450 / "workspace_two_frames.RData")
    assert list(result) == ["df1", "df2"]
    _check_df1(result["df1"])
    df2 = result["df2"]
    assert list(df2["code"].cat.categories) == ["x", "y", "z"]
    assert df2["code"].tolist() == ["x", "y", "x"]
    assert df2["flag"][0] == True and pd.isna(df2["flag"][1])  # noqa: E712


def test_reads_mixed_workspace() -> None:
    result = read_rdata(R450 / "workspace_mixed.RData")
    assert list(result) == ["df1", "measurements", "title", "config"]
    _check_df1(result["df1"])
    assert result["measurements"] == {"first": 1.5, "second": 2.5}
    assert result["title"] == "workspace title"
    assert result["config"] == {"alpha": 1, "beta": ["m", "n"]}


@pytest.mark.parametrize(
    "name",
    ["workspace_v2.RData", "workspace_xz.RData", "workspace_plain.RData"],
)
def test_reads_version_and_compression_variants(name: str) -> None:
    result = read_rdata(R450 / name)
    _check_df1(result["df1"])


def test_reads_empty_workspace() -> None:
    assert read_rdata(R450 / "workspace_empty.RData") == {}


def test_selective_load_skips_other_objects() -> None:
    # df2's attribute symbols ("names", "class", ...) arrive as REFSXP
    # back-references to entries registered while df1 was being *skipped*;
    # correct decoding proves the skip/read reference-table alignment.
    result = read_rdata(R450 / "workspace_two_frames.RData", select=["df2"])
    assert list(result) == ["df2"]
    assert result["df2"]["code"].tolist() == ["x", "y", "x"]

    mixed = read_rdata(
        R450 / "workspace_mixed.RData", select=["title", "config"]
    )
    assert list(mixed) == ["title", "config"]
    assert mixed["title"] == "workspace title"


def test_select_errors() -> None:
    with pytest.raises(ValueError, match="available: df1, df2"):
        read_rdata(R450 / "workspace_two_frames.RData", select=["missing"])
    with pytest.raises(TypeError, match="sequence"):
        read_rdata(R450 / "workspace_two_frames.RData", select="df1")
    with pytest.raises(ValueError, match="empty"):
        read_rdata(R450 / "workspace_two_frames.RData", select=[])


def test_ascii_workspace_is_rejected() -> None:
    with pytest.raises(UnsupportedRDS, match="ASCII RData"):
        read_rdata(R450 / "workspace_ascii.RData")


def test_container_cross_errors() -> None:
    # An RDS handed to read_rdata() points back at the right function...
    with pytest.raises(UnsupportedRDS, match="read_rds"):
        read_rdata(R450 / "df_default_rownames.rds")
    # ... and an RData handed to read_rds() keeps its explicit error.
    with pytest.raises(UnsupportedRDS, match="RData"):
        read_rds(R450 / "workspace_two_frames.RData")
    with pytest.raises(InvalidRDS, match="not a supported RData"):
        read_rdata(b"garbage that is not an R file at all")


def test_reads_from_bytes() -> None:
    payload = (R450 / "workspace_two_frames.RData").read_bytes()
    result = read_rdata(payload)
    assert list(result) == ["df1", "df2"]


def test_inspect_reports_rdata_as_fast_supported() -> None:
    info = inspect_r_file(R450 / "workspace_two_frames.RData")
    assert info.container == "rdata"
    assert info.serialization == "xdr"
    assert info.fast_supported


def test_cli_dump_routes_rdata(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["dump", str(R450 / "workspace_mixed.RData")]) == 0
    output = capsys.readouterr().out
    assert "df1" in output and "workspace title" in output
