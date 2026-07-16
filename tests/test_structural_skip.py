from __future__ import annotations

from io import BytesIO

import conftest as helpers
import numpy as np
import pytest

from rdsframe import InvalidRDS, RDSLimitError, ReaderLimits
from rdsframe._core import REFSXP, Reader


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


def _reader(payload: bytes, *, limits: ReaderLimits | None = None) -> Reader:
    return Reader(BytesIO(payload), byteorder=">", limits=limits or ReaderLimits())


def _nested_vectors(depth: int) -> bytes:
    payload = helpers.flags(helpers.NIL)
    for _ in range(depth):
        payload = helpers.flags(helpers.VEC) + helpers.i32(1) + payload
    return payload


def test_read_rejects_zero_reference_index() -> None:
    reader = _reader(helpers.i32(0))
    reader.references.append("sentinel")

    with pytest.raises(InvalidRDS, match="invalid reference index: 0"):
        reader.read_item_from_header((REFSXP, False, False, False, REFSXP))


def test_string_batch_rejects_zero_reference_index() -> None:
    reader = _reader(helpers.i32(REFSXP) + helpers.i32(0))
    reader.references.append("sentinel")

    with pytest.raises(InvalidRDS, match="invalid reference index: 0"):
        reader.read_string_elements(1, utf8=False)


def test_structural_skip_respects_max_depth() -> None:
    reader = _reader(_nested_vectors(20), limits=ReaderLimits(max_depth=10))

    with pytest.raises(RDSLimitError, match="nesting exceeds configured limit"):
        reader.skip_item()


def test_numeric_array_allocation_limit_is_checked_before_payload_read() -> None:
    reader = _reader(helpers.i32(5), limits=ReaderLimits(max_allocation_bytes=16))

    with pytest.raises(RDSLimitError, match="vector allocation"):
        reader.numeric_array(np.dtype(np.int32))


def test_string_array_allocation_limit_is_checked_before_element_list() -> None:
    reader = _reader(helpers.i32(3), limits=ReaderLimits(max_allocation_bytes=16))

    with pytest.raises(RDSLimitError, match="string element list allocation"):
        reader.string_array()
