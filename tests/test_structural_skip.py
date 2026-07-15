from __future__ import annotations

from io import BytesIO

import pytest

from rdsframe import InvalidRDS, ReaderLimits
from rdsframe._core import Reader


def test_seekable_discard_advances_without_reading_payload() -> None:
    stream = BytesIO(b"x" * 128)
    reader = Reader(
        stream,
        byteorder=">",
        limits=ReaderLimits(),
        total_bytes=128,
        seekable_discard=True,
    )
    reader.discard(96)
    assert stream.tell() == 96
    assert reader.bytes_read == 96


def test_seekable_discard_still_detects_truncation() -> None:
    stream = BytesIO(b"x" * 16)
    reader = Reader(
        stream,
        byteorder=">",
        limits=ReaderLimits(),
        total_bytes=16,
        seekable_discard=True,
    )
    with pytest.raises(InvalidRDS, match="end of file"):
        reader.discard(17)
