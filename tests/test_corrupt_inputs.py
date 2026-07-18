from __future__ import annotations

import bz2
import gzip
import lzma
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


def _compress(label: str, plain: bytes) -> bytes:
    """Container bytes for every algorithm R can write via ``compress=``.

    R saves RDS gzip-compressed by default; bzip2/xz/zstd are the alternatives.
    A corrupt or truncated container must still fail inside the ``RDSError``
    domain, not leak the decompressor's own ``EOFError``/``BadGzipFile``/
    ``zlib.error``/``LZMAError``/zstd error.
    """
    if label == "gzip":
        return gzip.compress(plain)
    if label == "bzip2":
        return bz2.compress(plain)
    if label == "xz":
        return lzma.compress(plain)
    zstandard = pytest.importorskip("zstandard")
    return zstandard.compress(plain)


@pytest.mark.parametrize("label", ["gzip", "bzip2", "xz", "zstd"])
def test_corrupt_compressed_containers_stay_in_rds_error_domain(
    sample_rds: Path, label: str
) -> None:
    blob = _compress(label, sample_rds.read_bytes())
    # A valid container must still read cleanly through the guard.
    read_r_object(blob)
    cut_points = sorted(
        {
            0,
            1,
            2,
            len(blob) // 4,
            len(blob) // 2,
            (3 * len(blob)) // 4,
            len(blob) - 1,
        }
    )
    corruptions = [blob[:point] for point in cut_points if point >= 0]
    for offset in sorted({4, 8, len(blob) // 3, (2 * len(blob)) // 3, len(blob) - 2}):
        if 0 <= offset < len(blob):
            changed = bytearray(blob)
            changed[offset] ^= 0xFF
            corruptions.append(bytes(changed))
    for corrupted in corruptions:
        # Only RDSError (or a clean success) is allowed to escape; any other
        # exception type means a decompressor error leaked the contract.
        with suppress(RDSError):
            read_r_object(corrupted, limits=TIGHT_LIMITS)
