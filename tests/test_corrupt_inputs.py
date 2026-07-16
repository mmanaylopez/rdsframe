from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rdsframe import RDSError, ReaderLimits, read_r_object

TIGHT_LIMITS = ReaderLimits(
    max_vector_length=10_000,
    max_string_bytes=64 * 1024,
    max_allocation_bytes=1 * 1024 * 1024,
    max_references=1_000,
    max_depth=32,
)


def test_truncated_and_bit_flipped_files_fail_inside_rds_error_domain(
    sample_rds: Path,
) -> None:
    payload = sample_rds.read_bytes()
    cut_points = sorted(
        {0, 1, 2, 3, 4, 8, 16, len(payload) // 4, len(payload) // 2, len(payload) - 1}
    )
    corruptions = [payload[:point] for point in cut_points]
    for offset in sorted({0, 1, 2, 3, 8, len(payload) // 3, len(payload) - 1}):
        changed = bytearray(payload)
        changed[offset] ^= 0xFF
        corruptions.append(bytes(changed))

    for corrupted in corruptions:
        with suppress(RDSError):
            read_r_object(corrupted, limits=TIGHT_LIMITS)


@pytest.mark.fuzz
@settings(max_examples=150, deadline=None)
@given(st.binary(min_size=0, max_size=512))
def test_fuzzed_xdr_payload_never_leaks_implementation_exceptions(tail: bytes) -> None:
    with suppress(RDSError):
        read_r_object(b"X\n" + tail, limits=TIGHT_LIMITS)
