"""Equivalence and memory-bound tests for the batched string parsers.

The Arrow path drains parsed elements into its buffers in bounded chunks of
``_STRING_CHUNK`` rows (a full-column Python list would put peak memory at
~2x the column). These tests pin down that chunked draining produces output
identical to the object-strings path, including across chunk boundaries.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import dataframe, rds, strings

import rdsframe._core as core
from rdsframe import list_rds_tables, read_rds


def _text_fixture(tmp_path: Path, values: list[str | None], name: str) -> Path:
    payload = dataframe([strings(values)], ["label"])
    path = tmp_path / f"{name}.rds"
    path.write_bytes(rds(payload))
    return path


def test_arrow_matches_object_strings(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    values: list[str | None] = [
        "repeated", "repeated", None, "único-á", "", "repeated", None, "z" * 300,
    ]
    path = _text_fixture(tmp_path, values, "mixed")
    object_frame = read_rds(path)
    arrow_frame = read_rds(path, strings="pyarrow")
    assert object_frame["label"].tolist() == values
    assert arrow_frame["label"].fillna("@NA@").tolist() == (
        object_frame["label"].fillna("@NA@").tolist()
    )


def test_python_string_fallback_matches_optional_accelerator(tmp_path: Path) -> None:
    """The package remains usable when a platform cannot compile Cython."""
    values = [None if index % 11 == 0 else "same" for index in range(2_048)]
    path = _text_fixture(tmp_path, values, "fallback")
    accelerated = list_rds_tables(path)
    with patch("rdsframe._core._skip_string_chunk", None):
        fallback = list_rds_tables(path)
    assert accelerated == fallback


def test_arrow_chunk_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a tiny chunk so one column spans many drain cycles."""
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(core, "_STRING_CHUNK", 7)
    values: list[str | None] = [
        (None if index % 5 == 0 else f"value_{index}") for index in range(40)
    ]
    path = _text_fixture(tmp_path, values, "chunked")
    frame = read_rds(path, strings="pyarrow")
    assert frame["label"].fillna("@NA@").tolist() == [
        "@NA@" if value is None else value for value in values
    ]


def test_arrow_chunk_boundary_with_giant_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload larger than the batch buffer, landing on a chunk edge."""
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(core, "_STRING_CHUNK", 3)
    big = "x" * (2 * 1024 * 1024)  # larger than _BATCH_CHUNK (1 MiB)
    values: list[str | None] = ["a", "b", big, None, "c"]
    path = _text_fixture(tmp_path, values, "giant")
    frame = read_rds(path, strings="pyarrow")
    result = frame["label"].tolist()
    assert result[2] == big
    assert result[0] == "a" and result[4] == "c"


def test_arrow_chunked_draining_bounds_peak_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounded chunks must beat whole-column materialization on peak memory.

    Same file, same parser, two drain granularities: one chunk per column
    (the pre-fix behavior: a full Python list of bytes objects lives next to
    the growing Arrow data buffer) versus bounded chunks. Unique strings
    ensure interning could not have helped either way. The comparative
    assertion avoids brittle absolute budgets while pinning the mechanism.
    """
    pytest.importorskip("pyarrow")
    import tracemalloc

    rows = 40_000
    values = [f"unique-value-{index:06d}" for index in range(rows)]
    path = _text_fixture(tmp_path, values, "peak")

    def measure(chunk_rows: int) -> int:
        monkeypatch.setattr(core, "_STRING_CHUNK", chunk_rows)
        tracemalloc.start()
        frame = read_rds(path, strings="pyarrow")
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert frame.shape == (rows, 1)
        return peak

    whole_column_peak = measure(rows)
    bounded_peak = measure(2048)
    # 40k transient bytes objects (~55 bytes each incl. allocator rounding)
    # plus the 320 KB list slot array should shrink to a ~2048-element chunk.
    saved = whole_column_peak - bounded_peak
    assert saved > 1_000_000, (
        f"bounded draining saved only {saved:,} bytes "
        f"({whole_column_peak:,} -> {bounded_peak:,})"
    )
