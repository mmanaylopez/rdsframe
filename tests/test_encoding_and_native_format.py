from __future__ import annotations

import struct
import sys
from io import BytesIO
from pathlib import Path

import pytest

from rdsframe import InvalidRDS, RDSLimitError, ReaderLimits, read_rds
from rdsframe._core import CHARSXP, Reader, decode_header

CURRENT = "<" if sys.byteorder == "little" else ">"
OPPOSITE = ">" if CURRENT == "<" else "<"


def test_native_format_matching_endianness_decodes_directly() -> None:
    header = b"B\n" + struct.pack(f"{CURRENT}iii", 2, 0x030500, 0x020300)
    version, byteorder, encoding = decode_header(BytesIO(header))
    assert (version, byteorder, encoding) == (2, CURRENT, None)


def test_native_format_opposite_endianness_self_corrects() -> None:
    header = b"B\n" + struct.pack(f"{OPPOSITE}iii", 2, 0x030500, 0x020300)
    version, byteorder, encoding = decode_header(BytesIO(header))
    assert (version, byteorder, encoding) == (2, OPPOSITE, None)


def test_native_format_garbage_header_raises_clear_error() -> None:
    header = b"B\n" + bytes([0x12, 0x34, 0x56, 0x78]) * 3
    with pytest.raises(InvalidRDS, match="native-format"):
        decode_header(BytesIO(header))


def test_encoding_override_fixes_misdecoded_native_string(tmp_path: Path) -> None:
    # Reuse conftest's byte-level helpers for everything except the one
    # CHARSXP that must hold raw windows-1252 bytes -- chars() always
    # UTF-8-encodes, so that one element is hand-built with no gp flags.
    import conftest as helpers

    label_bytes = b"caf\xe9"  # "café" in windows-1252 (0xE9 = 'é'); invalid UTF-8 alone
    raw_char = helpers.flags(helpers.CHAR) + helpers.i32(len(label_bytes)) + label_bytes
    strsxp = helpers.flags(helpers.STR) + helpers.i32(1) + raw_char
    payload = helpers.dataframe([strsxp], ["label"])
    path = tmp_path / "cp1252.rds"
    path.write_bytes(helpers.rds(payload))

    default_frame = read_rds(path)
    assert default_frame["label"].tolist() != ["café"]

    fixed_frame = read_rds(path, encoding="windows-1252")
    assert fixed_frame["label"].tolist() == ["café"]


def test_encoding_override_rejects_unknown_codec(tmp_path: Path) -> None:
    from rdsframe._core import resolve_native_encoding

    with pytest.raises(ValueError, match="unknown text encoding"):
        resolve_native_encoding(None, "not-a-real-codec")


def test_encoding_override_covers_mislabeled_utf8_flag(tmp_path: Path) -> None:
    """A CHARSXP that *claims* UTF-8 but carries cp1252 bytes.

    Without an override the fast path must keep trusting the flag (zero-cost
    default); with an override the bytes are validated and re-decoded, so the
    user's stated encoding wins over the file's wrong label.
    """
    import conftest as helpers

    utf8_flag = 1 << 3  # UTF8_MASK, stored in the gp bits (flags >> 12)
    label_bytes = b"caf\xe9"  # cp1252 'café'; invalid as UTF-8
    raw_char = (
        helpers.i32(helpers.CHAR | (utf8_flag << 12))
        + helpers.i32(len(label_bytes))
        + label_bytes
    )
    strsxp = helpers.flags(helpers.STR) + helpers.i32(1) + raw_char
    payload = helpers.dataframe([strsxp], ["label"])
    path = tmp_path / "mislabeled_utf8.rds"
    path.write_bytes(helpers.rds(payload))

    default_frame = read_rds(path)
    assert default_frame["label"].tolist() != ["café"]

    fixed_frame = read_rds(path, encoding="windows-1252")
    assert fixed_frame["label"].tolist() == ["café"]


def test_encoding_override_covers_mislabeled_utf8_flag_arrow(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    import conftest as helpers

    utf8_flag = 1 << 3
    label_bytes = b"caf\xe9"
    raw_char = (
        helpers.i32(helpers.CHAR | (utf8_flag << 12))
        + helpers.i32(len(label_bytes))
        + label_bytes
    )
    strsxp = helpers.flags(helpers.STR) + helpers.i32(1) + raw_char
    payload = helpers.dataframe([strsxp], ["label"])
    path = tmp_path / "mislabeled_utf8_arrow.rds"
    path.write_bytes(helpers.rds(payload))

    fixed_frame = read_rds(path, strings="pyarrow", encoding="windows-1252")
    assert fixed_frame["label"].tolist() == ["café"]


def test_declared_header_encoding_is_used_when_recognized() -> None:
    from rdsframe._core import resolve_native_encoding

    assert resolve_native_encoding("latin1", None) == "latin-1"
    assert resolve_native_encoding("UTF-8", None) == "utf-8"
    assert resolve_native_encoding("unknown", None) == "utf-8"
    assert resolve_native_encoding(None, None) == "utf-8"
    assert resolve_native_encoding("totally-bogus-name", None) == "utf-8"
    assert resolve_native_encoding("latin1", "utf-8") == "utf-8"  # override wins


def test_standalone_charsxp_honors_native_encoding() -> None:
    import conftest as helpers

    data = b"caf\xe9"
    payload = helpers.i32(CHARSXP) + helpers.i32(len(data)) + data
    reader = Reader(
        BytesIO(payload),
        byteorder=">",
        limits=ReaderLimits(),
        native_encoding="windows-1252",
    )
    value = reader.read_item()
    assert value.value == "caf\u00e9"


def test_invalid_charsxp_length_is_malformed_not_a_limit() -> None:
    import conftest as helpers

    reader = Reader(
        BytesIO(helpers.i32(CHARSXP) + helpers.i32(-2)),
        byteorder=">",
        limits=ReaderLimits(max_string_bytes=1),
    )
    with pytest.raises(InvalidRDS, match="invalid character length") as error:
        reader.read_item()
    assert not isinstance(error.value, RDSLimitError)
