"""Low-level reader for R binary serialization streams.

This module deliberately implements a conservative subset of R serialization.
Unsupported structures fail explicitly instead of returning silently corrupted
data.  Public users should import from :mod:`rdsframe`, not from here.
"""

from __future__ import annotations

import bz2
import codecs
import gzip
import io
import lzma
import os
import struct
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

try:
    if os.environ.get("RDSFRAME_DISABLE_CYTHON") == "1":
        raise ImportError
    from ._cython_core import skip_string_chunk as _skip_string_chunk
except ImportError:  # Pure-Python wheels and systems without a compiler.
    _skip_string_chunk = None  # type: ignore[assignment]

CYTHON_ACCELERATOR_AVAILABLE = _skip_string_chunk is not None

NILSXP = 0
SYMSXP = 1
LISTSXP = 2
CLOSXP = 3
ENVSXP = 4
PROMSXP = 5
LANGSXP = 6
CHARSXP = 9
LGLSXP = 10
INTSXP = 13
REALSXP = 14
CPLXSXP = 15
STRSXP = 16
DOTSXP = 17
VECSXP = 19
EXPRSXP = 20
BCODESXP = 21
EXTPTRSXP = 22
WEAKREFSXP = 23
RAWSXP = 24
S4SXP = 25
ALTREP_SXP = 238
EMPTYENV_SXP = 242
BASEENV_SXP = 241
GLOBALENV_SXP = 253
UNBOUNDVALUE_SXP = 252
MISSINGARG_SXP = 251
BASENAMESPACE_SXP = 250
NAMESPACESXP = 249
PACKAGESXP = 248
NILVALUE_SXP = 254
REFSXP = 255

# Human-readable names for R object types the fast reader deliberately does
# not convert. Callers surface these so a user understands *why* a file needs
# a slower general-purpose fallback instead of just perceiving slowness.
SEXP_TYPE_NAMES = {
    CLOSXP: "closure (R function)",
    PROMSXP: "promise (lazy R value)",
    LANGSXP: "language call (unevaluated R code)",
    DOTSXP: "dot-dot-dot argument list",
    EXPRSXP: "expression vector",
    BCODESXP: "compiled R bytecode",
    WEAKREFSXP: "weak reference",
    7: "special (R internal function)",
    8: "builtin (R internal function)",
}

NA_INTEGER = -(2**31)
BYTES_MASK = 1 << 1
LATIN1_MASK = 1 << 2
UTF8_MASK = 1 << 3
ASCII_MASK = 1 << 6

# R's difftime stores an elapsed-time count plus a "units" attribute naming
# what each unit represents (see ?difftime). Seconds is the common base for
# converting to a fixed-duration type (pandas Timedelta / Arrow duration).
DIFFTIME_SECONDS_PER_UNIT = {
    "secs": 1.0,
    "mins": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "weeks": 604800.0,
}

ProgressCallback = Callable[[int], None]

# Chunk size for the batched STRSXP parsers. Large enough to amortize stream
# calls over thousands of small string headers, small enough that the
# overshoot parked in the pending buffer stays negligible.
_BATCH_CHUNK = 1 << 20

# Rows per drain cycle when building Arrow string buffers: bounds the
# transient list of Python bytes objects so peak memory tracks one column
# plus one chunk, never two whole columns. 262,144 matches the default
# Parquet row-group granularity.
_STRING_CHUNK = 1 << 18

# Crossing the Python/C boundary and constructing a typed memoryview costs more
# than the loop saves for the tiny vectors common in deeply nested R objects.
# Keep those on the already-batched Python path; large analytical columns take
# the compiled loop where the fixed entry cost is negligible.
_CYTHON_STRING_MIN_ELEMENTS = 1024


class RDSError(Exception):
    """Base exception for rdsframe."""


class InvalidRDS(RDSError):
    """The input is truncated, malformed, or not an R serialization stream."""


class UnsupportedRDS(RDSError):
    """The input contains a valid R structure unsupported by the fast reader."""


class RDSLimitError(RDSError):
    """A configured safety or resource limit was exceeded."""


class RDSCatalogError(RDSError):
    """A table catalog is stale, ambiguous, or incompatible with its source."""


@dataclass(frozen=True, slots=True)
class ReaderLimits:
    """Defensive parsing limits.

    Defaults are intentionally generous for analytical datasets. Set a field to
    a smaller value when reading untrusted files in a constrained environment.
    """

    max_vector_length: int = 2_500_000_000
    max_string_bytes: int = 256 * 1024 * 1024
    max_allocation_bytes: int = 1 * 1024 * 1024 * 1024
    max_references: int = 10_000_000
    max_depth: int = 256


@dataclass(frozen=True, slots=True)
class SerializedObject:
    value: Any
    attributes: dict[str, Any]
    sexp_type: int


@dataclass(frozen=True, slots=True)
class SkippedObject:
    """Structural metadata returned when an object payload is discarded."""

    sexp_type: int
    length: int | None = None


class Reader:
    """Sequential parser that materializes atomic vectors directly into NumPy."""

    def __init__(
        self,
        stream: BinaryIO,
        *,
        byteorder: str,
        limits: ReaderLimits,
        progress: ProgressCallback | None = None,
        total_bytes: int = 0,
        compressed_position: Callable[[], int] | None = None,
        arrow_strings: bool = False,
        seekable_discard: bool = False,
        native_encoding: str = "utf-8",
        utf8_fallback: str | None = None,
    ) -> None:
        self.stream = stream
        self.byteorder = byteorder
        self.limits = limits
        self.references: list[Any] = []
        self.bytes_read = 0
        self.progress = progress
        self.total_bytes = total_bytes
        self.compressed_position = compressed_position
        self.last_progress = -1
        self.depth = 0
        # Fallback codec for CHARSXP elements with no explicit UTF-8/ASCII/
        # latin-1 gp flag (R's CE_NATIVE strings). Resolved by the caller from
        # the RDS header's declared encoding (version 3) or an explicit
        # override; "utf-8" matches modern R defaults.
        self.native_encoding = native_encoding
        # When the user supplied an explicit encoding override, strings whose
        # gp flags *claim* UTF-8 but whose bytes fail UTF-8 validation are
        # re-decoded with this codec instead of being trusted blindly. None
        # (the default) keeps the zero-validation fast path.
        self.utf8_fallback = utf8_fallback
        self.arrow_strings = arrow_strings
        self.seekable_discard = seekable_discard
        self._discard_buffer = bytearray(1024 * 1024)
        self._discard_view = memoryview(self._discard_buffer)
        # Cached bound methods and a precompiled struct avoid repeated attribute
        # lookups and format-string (re)compilation in the per-element hot loops
        # (one CHARSXP header read per string, potentially tens of millions of
        # times for wide, text-heavy tables).
        self._read = stream.read
        self._readinto = getattr(stream, "readinto", None)
        self._unpack_i32 = struct.Struct(f"{byteorder}i").unpack
        self._unpack_i32_from = struct.Struct(f"{byteorder}i").unpack_from
        # Overshoot buffer for the batched STRSXP parsers: they read the
        # stream in large chunks and stash the unconsumed tail here. Every
        # other read primitive drains this before touching the stream. The
        # bytes were already counted into bytes_read when first read.
        self._pending: bytes = b""
        self._pending_pos = 0
        self._pending_len = 0

    def _reference(self, index: int, *, message: str = "invalid reference index") -> Any:
        if index < 1 or index > len(self.references):
            raise InvalidRDS(f"{message}: {index}")
        return self.references[index - 1]

    def _check_allocation(self, count: int, itemsize: int, label: str) -> None:
        if count < 0:
            raise InvalidRDS(f"invalid {label} length: {count}")
        size = count * itemsize
        if size > self.limits.max_allocation_bytes:
            raise RDSLimitError(
                f"{label} allocation {size:,} bytes exceeds configured limit "
                f"{self.limits.max_allocation_bytes:,}"
            )

    def raw(self, size: int) -> bytes:
        if size < 0:
            raise InvalidRDS(f"negative read size: {size}")
        if self._pending_pos < self._pending_len:
            return self._raw_from_pending(size)
        data = self._read(size)
        if len(data) != size:
            raise InvalidRDS("unexpected end of file")
        self.bytes_read += size
        return data

    def _raw_from_pending(self, size: int) -> bytes:
        pos = self._pending_pos
        take = min(size, self._pending_len - pos)
        part = self._pending[pos : pos + take]
        pos += take
        if pos == self._pending_len:
            self._pending, self._pending_pos, self._pending_len = b"", 0, 0
        else:
            self._pending_pos = pos
        if take == size:
            return part
        rest = self._read(size - take)
        if len(rest) != size - take:
            raise InvalidRDS("unexpected end of file")
        self.bytes_read += size - take
        return part + rest

    def read_into(self, target: memoryview) -> None:
        """Fill *target* without first allocating a second bytes-sized buffer."""
        offset = 0
        length = len(target)
        if self._pending_pos < self._pending_len:
            take = min(length, self._pending_len - self._pending_pos)
            target[:take] = self._pending[self._pending_pos : self._pending_pos + take]
            self._pending_pos += take
            if self._pending_pos == self._pending_len:
                self._pending, self._pending_pos, self._pending_len = b"", 0, 0
            if take == length:
                return
            offset = take
        readinto = self._readinto
        has_progress = self.progress is not None
        while offset < length:
            view = target[offset:]
            count = readinto(view) if readinto is not None else None
            if count is None:
                chunk = self._read(min(len(view), 8 * 1024 * 1024))
                count = len(chunk)
                if count:
                    view[:count] = chunk
            if not count:
                raise InvalidRDS("unexpected end of file")
            offset += count
            self.bytes_read += count
            if has_progress:
                self.tick()

    def discard(self, size: int) -> None:
        """Consume *size* bytes with bounded memory and progress reporting."""
        if size < 0:
            raise InvalidRDS(f"negative discard size: {size}")
        if size == 0:
            return
        if self._pending_pos < self._pending_len:
            take = min(size, self._pending_len - self._pending_pos)
            self._pending_pos += take
            if self._pending_pos == self._pending_len:
                self._pending, self._pending_pos, self._pending_len = b"", 0, 0
            size -= take
            if size == 0:
                return
        if self.seekable_discard:
            current = self.stream.tell()
            if self.total_bytes and current + size > self.total_bytes:
                raise InvalidRDS("unexpected end of file")
            self.stream.seek(size, 1)
            self.bytes_read += size
            if self.progress is not None:
                self.tick()
            return
        readinto = self._readinto
        buffer = self._discard_view
        buffer_len = len(buffer)
        if size <= buffer_len:
            # Common case (one CHARSXP payload): a single bounded read, no loop
            # bookkeeping. io.BufferedReader/BytesIO.readinto() only returns
            # fewer bytes than requested at EOF, never a spurious short read,
            # so one call is sufficient here (open_rds_stream always hands the
            # Reader a buffered stream; tests use BytesIO, same guarantee).
            target = buffer if size == buffer_len else buffer[:size]
            count = readinto(target) if readinto is not None else len(self._read(size))
            if count != size:
                raise InvalidRDS("unexpected end of file")
            self.bytes_read += size
            if self.progress is not None:
                self.tick()
            return
        remaining = size
        has_progress = self.progress is not None
        while remaining:
            target = buffer if remaining >= buffer_len else buffer[:remaining]
            count = readinto(target) if readinto is not None else None
            if count is None:
                chunk = self._read(len(target))
                count = len(chunk)
            if not count:
                raise InvalidRDS("unexpected end of file")
            remaining -= count
            self.bytes_read += count
            if has_progress:
                self.tick()

    def tick(self, forced: int | None = None) -> None:
        if self.progress is None:
            return
        try:
            if forced is not None:
                percent = forced
            elif self.total_bytes and self.compressed_position is not None:
                percent = min(99, int(self.compressed_position() * 100 / self.total_bytes))
            elif self.total_bytes:
                percent = min(99, int(self.bytes_read * 100 / self.total_bytes))
            else:
                return
            if percent != self.last_progress:
                self.last_progress = percent
                self.progress(percent)
        except Exception:
            # Progress reporting must never make a successful parse fail.
            return

    def i32(self) -> int:
        return int(self._unpack_i32(self.raw(4))[0])

    def length(self) -> int:
        length = self.i32()
        if length == -1:
            high = self.i32() & 0xFFFFFFFF
            low = self.i32() & 0xFFFFFFFF
            length = (high << 32) | low
        elif length < 0:
            raise InvalidRDS(f"invalid vector length: {length}")
        if length > self.limits.max_vector_length:
            raise RDSLimitError(
                f"vector length {length:,} exceeds configured limit "
                f"{self.limits.max_vector_length:,}"
            )
        return length

    def flags(self) -> tuple[int, bool, bool, bool, int]:
        flags = self.i32()
        return (
            flags & 0xFF,
            bool((flags >> 8) & 1),
            bool((flags >> 9) & 1),
            bool((flags >> 10) & 1),
            flags,
        )

    def numeric_array(self, dtype: np.dtype[Any]) -> np.ndarray:
        length = self.length()
        native_dtype = np.dtype(dtype).newbyteorder("=")
        self._check_allocation(length, native_dtype.itemsize, "vector")
        array = np.empty(length, dtype=native_dtype)
        self.read_into(memoryview(array).cast("B"))
        source_is_native = (self.byteorder == "<") == (sys.byteorder == "little")
        if not source_is_native and array.itemsize > 1:
            array.byteswap(inplace=True)
        return array

    def char(self) -> str | None:
        """Read one standalone STRSXP element (rare path; batch loops below)."""
        flags = self._unpack_i32(self.raw(4))[0]
        sexp_type = flags & 0xFF
        if sexp_type == REFSXP:
            index = flags >> 8 or self.i32()
            referenced = self._reference(index)
            if referenced is None or isinstance(referenced, str):
                return referenced
            raise InvalidRDS(f"reference {index} does not point to a character value")
        if sexp_type in {NILSXP, NILVALUE_SXP}:
            return None
        if sexp_type != CHARSXP:
            raise UnsupportedRDS(f"unexpected STRSXP element type: {sexp_type}")
        length = self._unpack_i32(self.raw(4))[0]
        if length == -1:
            return None
        if length < -1:
            raise InvalidRDS(f"invalid character length: {length}")
        if length > self.limits.max_string_bytes:
            raise RDSLimitError(
                f"string length {length:,} exceeds configured limit "
                f"{self.limits.max_string_bytes:,}"
            )
        data = self.raw(length)
        return self._decode_charsxp(data, flags)

    def _decode_charsxp(self, data: bytes, flags: int) -> str:
        """Decode one CHARSXP payload using its gp flags and native codec."""
        gp = flags >> 12
        encoding = "latin-1" if gp & (BYTES_MASK | LATIN1_MASK) else self.native_encoding
        if gp & (UTF8_MASK | ASCII_MASK):
            encoding = "utf-8"
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            if encoding == "utf-8" and self.utf8_fallback is not None:
                return data.decode(self.utf8_fallback, errors="replace")
            return data.decode(encoding, errors="replace")

    def _take_batch_buffer(self) -> tuple[bytes, int]:
        """Move any pending overshoot into a local batch buffer."""
        if self._pending_pos < self._pending_len:
            buf, pos = self._pending, self._pending_pos
        else:
            buf, pos = b"", 0
        self._pending, self._pending_pos, self._pending_len = b"", 0, 0
        return buf, pos

    def _store_batch_buffer(self, buf: bytes, pos: int) -> None:
        if pos < len(buf):
            self._pending, self._pending_pos, self._pending_len = buf, pos, len(buf)

    def _refill_batch(self, buf: bytes, pos: int, need: int) -> bytes:
        """Return a buffer holding at least *need* bytes starting at offset 0.

        Reads the stream in large chunks on purpose: the overshoot is what
        the batched parsers bank on, and whatever is left after the batch is
        parked in the pending buffer for the ordinary primitives to drain.
        """
        parts = [buf[pos:]]
        have = len(parts[0])
        while have < need:
            chunk = self._read(max(_BATCH_CHUNK, need - have))
            if not chunk:
                raise InvalidRDS("unexpected end of file")
            self.bytes_read += len(chunk)
            parts.append(chunk)
            have += len(chunk)
        if self.progress is not None:
            self.tick()
        return parts[0] if len(parts) == 1 else b"".join(parts)

    def skip_string_elements(self, count: int) -> None:
        """Structurally skip *count* STRSXP elements with a batched parser.

        Each CHARSXP carries its own header, so element-by-element traversal
        is inherent to the RDS format -- but doing it through per-element
        stream reads costs three I/O calls per string. This loop parses whole
        chunks with ``unpack_from`` instead; payloads that overflow the chunk
        fall back to :meth:`discard`, which turns into a real ``seek()`` on
        uncompressed sources.
        """
        if _skip_string_chunk is not None and count >= _CYTHON_STRING_MIN_ELEMENTS:
            self._skip_string_elements_compiled(count)
            return
        unpack_from = self._unpack_i32_from
        max_string = self.limits.max_string_bytes
        references_len = len(self.references)
        buf, pos = self._take_batch_buffer()
        buf_len = len(buf)
        for _ in range(count):
            if buf_len - pos < 4:
                buf = self._refill_batch(buf, pos, 4)
                pos, buf_len = 0, len(buf)
            flags = unpack_from(buf, pos)[0]
            pos += 4
            sexp_type = flags & 0xFF
            if sexp_type == CHARSXP:
                if buf_len - pos < 4:
                    buf = self._refill_batch(buf, pos, 4)
                    pos, buf_len = 0, len(buf)
                length = unpack_from(buf, pos)[0]
                pos += 4
                if length <= 0:
                    if length == 0 or length == -1:
                        continue
                    raise InvalidRDS(f"invalid character length: {length}")
                if length > max_string:
                    raise RDSLimitError(
                        f"string length {length:,} exceeds configured limit "
                        f"{max_string:,}"
                    )
                available = buf_len - pos
                if length <= available:
                    pos += length
                else:
                    buf, pos, buf_len = b"", 0, 0
                    self.discard(length - available)
                continue
            if sexp_type == REFSXP:
                index = flags >> 8
                if index == 0:
                    if buf_len - pos < 4:
                        buf = self._refill_batch(buf, pos, 4)
                        pos, buf_len = 0, len(buf)
                    index = unpack_from(buf, pos)[0]
                    pos += 4
                if index < 1 or index > references_len:
                    raise InvalidRDS(f"invalid character reference index: {index}")
                continue
            if sexp_type in {NILSXP, NILVALUE_SXP}:
                continue
            raise UnsupportedRDS(f"unexpected STRSXP element type: {sexp_type}")
        self._store_batch_buffer(buf, pos)

    def _skip_string_elements_compiled(self, count: int) -> None:
        """Skip STRSXP elements through the optional Cython chunk scanner."""
        scanner = _skip_string_chunk
        if scanner is None:  # pragma: no cover - guarded by the caller
            raise AssertionError("compiled string scanner is unavailable")
        remaining = count
        references_len = len(self.references)
        max_string = self.limits.max_string_bytes
        little_endian = self.byteorder == "<"
        buf, pos = self._take_batch_buffer()
        while remaining:
            processed, pos, status, value = scanner(
                buf,
                pos,
                remaining,
                max_string,
                references_len,
                little_endian,
            )
            remaining -= processed
            if status == 0:
                if remaining == 0:
                    break
                # The scanner stopped at an incomplete element header. Refill
                # from that exact boundary; eight bytes cover CHARSXP and the
                # long-form REFSXP header.
                buf = self._refill_batch(buf, pos, 8)
                pos = 0
                continue
            if status == 1:
                # A character payload crosses the batch boundary. Bytes still
                # in ``buf`` are already consumed; discard only the tail.
                available = len(buf) - pos
                buf, pos = b"", 0
                self.discard(value - available)
                remaining -= 1
                continue
            if status == 2:
                raise InvalidRDS(f"invalid character length: {value}")
            if status == 3:
                raise RDSLimitError(
                    f"string length {value:,} exceeds configured limit "
                    f"{max_string:,}"
                )
            if status == 4:
                raise InvalidRDS(f"invalid character reference index: {value}")
            raise UnsupportedRDS(f"unexpected STRSXP element type: {value}")
        self._store_batch_buffer(buf, pos)

    def read_string_elements(self, count: int, *, utf8: bool) -> list[Any]:
        """Read *count* STRSXP elements with a batched parser.

        With ``utf8=False`` returns interned ``str | None`` values; with
        ``utf8=True`` returns UTF-8 ``bytes | None`` (R's own gp flags mark
        already-valid UTF-8/ASCII payloads, which pass through byte-exact).
        When an explicit ``utf8_fallback`` codec is configured, bytes that
        claim UTF-8 but fail validation are re-decoded with it instead of
        being trusted -- covering mislabeled files at an opt-in cost.

        Interning applies only to the ``str`` mode, where the returned list
        is the long-lived result and deduplication saves real memory. In
        ``utf8`` mode the pieces are drained into Arrow buffers and dropped
        chunk by chunk (see :meth:`arrow_string_array`), so an interning
        dict would cost a lookup per element for transient objects.
        """
        self._check_allocation(count, 8, "string element list")
        unpack_from = self._unpack_i32_from
        max_string = self.limits.max_string_bytes
        native = self.native_encoding
        fallback = self.utf8_fallback
        references = self.references
        interned: dict[Any, Any] | None = None if utf8 else {}
        values: list[Any] = [None] * count
        buf, pos = self._take_batch_buffer()
        buf_len = len(buf)
        flag_mask = UTF8_MASK | ASCII_MASK
        latin_mask = BYTES_MASK | LATIN1_MASK
        for index in range(count):
            if buf_len - pos < 4:
                buf = self._refill_batch(buf, pos, 4)
                pos, buf_len = 0, len(buf)
            flags = unpack_from(buf, pos)[0]
            pos += 4
            sexp_type = flags & 0xFF
            if sexp_type == CHARSXP:
                if buf_len - pos < 4:
                    buf = self._refill_batch(buf, pos, 4)
                    pos, buf_len = 0, len(buf)
                length = unpack_from(buf, pos)[0]
                pos += 4
                if length < 0:
                    if length == -1:
                        continue  # NA_character_
                    raise InvalidRDS(f"invalid character length: {length}")
                if length > max_string:
                    raise RDSLimitError(
                        f"string length {length:,} exceeds configured limit "
                        f"{max_string:,}"
                    )
                available = buf_len - pos
                if length <= available:
                    data = buf[pos : pos + length]
                    pos += length
                else:
                    head = buf[pos:]
                    tail = self._read(length - available)
                    if len(tail) != length - available:
                        raise InvalidRDS("unexpected end of file")
                    self.bytes_read += len(tail)
                    data = head + tail
                    buf, pos, buf_len = b"", 0, 0
                gp = flags >> 12
                value: Any
                if utf8:
                    if gp & flag_mask:
                        if fallback is not None:
                            try:
                                data.decode("utf-8")
                            except UnicodeDecodeError:
                                data = data.decode(fallback, errors="replace").encode(
                                    "utf-8"
                                )
                        value = data
                    elif gp & latin_mask:
                        value = data.decode("latin-1").encode("utf-8")
                    elif native == "utf-8":
                        try:
                            data.decode("utf-8")
                            value = data
                        except UnicodeDecodeError:
                            value = data.decode("utf-8", errors="replace").encode(
                                "utf-8"
                            )
                    else:
                        value = data.decode(native, errors="replace").encode("utf-8")
                else:
                    if gp & flag_mask:
                        encoding = "utf-8"
                    elif gp & latin_mask:
                        encoding = "latin-1"
                    else:
                        encoding = native
                    try:
                        value = data.decode(encoding)
                    except UnicodeDecodeError:
                        if encoding == "utf-8" and fallback is not None:
                            value = data.decode(fallback, errors="replace")
                        else:
                            value = data.decode(encoding, errors="replace")
                values[index] = (
                    value if interned is None else interned.setdefault(value, value)
                )
                continue
            if sexp_type == REFSXP:
                ref_index = flags >> 8
                if ref_index == 0:
                    if buf_len - pos < 4:
                        buf = self._refill_batch(buf, pos, 4)
                        pos, buf_len = 0, len(buf)
                    ref_index = unpack_from(buf, pos)[0]
                    pos += 4
                if ref_index < 1 or ref_index > len(references):
                    raise InvalidRDS(f"invalid reference index: {ref_index}")
                referenced = references[ref_index - 1]
                if referenced is None:
                    continue
                if isinstance(referenced, str):
                    values[index] = referenced.encode("utf-8") if utf8 else referenced
                    continue
                raise InvalidRDS(
                    f"reference {ref_index} does not point to a character value"
                )
            if sexp_type in {NILSXP, NILVALUE_SXP}:
                continue
            raise UnsupportedRDS(f"unexpected STRSXP element type: {sexp_type}")
        self._store_batch_buffer(buf, pos)
        return values

    def string_array(self) -> Any:
        if self.arrow_strings:
            return self.arrow_string_array()
        return self.read_string_elements(self.length(), utf8=False)

    def arrow_string_array(self) -> Any:
        """Build a LargeStringArray without retaining a Python str per row.

        The persistent representation consists of Arrow-compatible offset,
        UTF-8 data, and validity buffers. Elements are parsed with the
        batched reader but drained into those buffers in bounded chunks of
        ``_STRING_CHUNK`` rows: materializing the whole column as a Python
        list first would put peak memory at roughly twice the column's text
        (list of bytes objects + the data buffer), which is exactly the
        regression a memory-limited conversion subprocess cannot afford.
        Peak stays at ~one column plus one bounded chunk.
        """
        try:
            import pyarrow as pa  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - guarded by the extra
            raise ImportError(
                "Arrow string parsing requires: pip install 'rdsframe[arrow]'"
            ) from exc
        length = self.length()
        self._check_allocation(length + 1, np.dtype(np.int64).itemsize, "Arrow offsets")
        self._check_allocation((length + 7) // 8, 1, "Arrow validity bitmap")
        offsets = np.empty(length + 1, dtype=np.int64)
        offsets[0] = 0
        data = bytearray()
        validity = bytearray((length + 7) // 8)
        null_count = 0
        position = 0
        index = 0
        remaining = length
        while remaining:
            batch = _STRING_CHUNK if remaining > _STRING_CHUNK else remaining
            pieces = self.read_string_elements(batch, utf8=True)
            for piece in pieces:
                if piece is None:
                    null_count += 1
                else:
                    self._check_allocation(position + len(piece), 1, "Arrow string data")
                    data += piece
                    position += len(piece)
                    validity[index >> 3] |= 1 << (index & 7)
                offsets[index + 1] = position
                index += 1
            remaining -= batch
        buffers = [
            None if null_count == 0 else pa.py_buffer(validity),
            pa.py_buffer(offsets),
            pa.py_buffer(data),
        ]
        return pa.Array.from_buffers(
            pa.large_string(), length, buffers, null_count=null_count
        )

    def add_reference(self, value: Any) -> None:
        if len(self.references) >= self.limits.max_references:
            raise RDSLimitError("reference table exceeds configured limit")
        self.references.append(value)

    def read_item(self) -> Any:
        self.depth += 1
        if self.depth > self.limits.max_depth:
            self.depth -= 1
            raise RDSLimitError("object nesting exceeds configured limit")
        try:
            return self._read_item_from_header(self.flags())
        finally:
            self.depth -= 1

    def read_item_from_header(
        self, header: tuple[int, bool, bool, bool, int]
    ) -> Any:
        """Read an item after its flags have already been consumed.

        This entry point lets the Parquet pipeline inspect container boundaries
        while preserving the exact same parsing rules as :meth:`read_item`.
        """
        self.depth += 1
        if self.depth > self.limits.max_depth:
            self.depth -= 1
            raise RDSLimitError("object nesting exceeds configured limit")
        try:
            return self._read_item_from_header(header)
        finally:
            self.depth -= 1

    def _read_item_from_header(
        self, header: tuple[int, bool, bool, bool, int]
    ) -> Any:
        sexp_type, _is_object, has_attr, has_tag, flags = header
        if sexp_type in {NILSXP, NILVALUE_SXP}:
            return None
        if sexp_type == REFSXP:
            index = flags >> 8 or self.i32()
            return self._reference(index)
        if sexp_type == SYMSXP:
            symbol = ("sym", self.read_item())
            self.add_reference(symbol)
            return symbol
        if sexp_type == LISTSXP:
            return self.read_pairlist(has_tag)
        if sexp_type == ALTREP_SXP:
            # ALTREP carries its own attributes in a trailing slot; it
            # returns a complete SerializedObject typed as the underlying
            # vector, so downstream column conversion never sees ALTREP.
            return self.read_altrep()
        if sexp_type == ENVSXP:
            return self.read_environment()
        if sexp_type == GLOBALENV_SXP:
            return ("r_env", "global")
        if sexp_type == EMPTYENV_SXP:
            return ("r_env", "empty")
        if sexp_type == BASEENV_SXP:
            return ("r_env", "base")
        if sexp_type == BASENAMESPACE_SXP:
            return ("r_namespace", "base")
        if sexp_type in {NAMESPACESXP, PACKAGESXP}:
            return self.read_namespace_spec(sexp_type)
        if sexp_type in {UNBOUNDVALUE_SXP, MISSINGARG_SXP}:
            return None

        value = self.read_payload(sexp_type, flags=flags)
        attributes = self.read_attributes() if has_attr else {}
        return SerializedObject(value, attributes, sexp_type)

    def read_environment(self) -> SerializedObject:
        """Read an ENVSXP as a plain name-to-value mapping.

        The result object is registered as a reference *before* its parts are
        read because environments can contain references back to themselves.
        The enclosure (parent scope) and the rarely-used attribute slot are
        consumed but not represented: an environment's data content is its
        frame plus hash table.
        """
        contents: dict[str, Any] = {}
        result = SerializedObject(contents, {}, ENVSXP)
        self.add_reference(result)
        self.i32()  # locked flag
        # The enclosure is fully read even though it is not represented in
        # the result. Skipping it is not safe: a parent scope is usually an
        # environment itself, which registers in the reference table, and a
        # later REFSXP may point at it -- a skipped-but-registered parent
        # would resolve to an empty dict, silently wrong. Depth is bounded
        # by limits.max_depth.
        self.read_item()  # enclosure
        frame = self.read_item()
        hashtab = self.read_item()
        self.read_item()  # attributes
        if isinstance(frame, dict):
            contents.update(frame)
        if isinstance(hashtab, SerializedObject) and hashtab.sexp_type == VECSXP:
            for bucket in hashtab.value:
                if isinstance(bucket, dict):
                    contents.update(bucket)
        return result

    def read_namespace_spec(self, sexp_type: int) -> tuple[str, str]:
        """Read a namespace/package reference (a version-tagged string vector)."""
        if self.i32() != 0:
            raise InvalidRDS("invalid namespace/package specification")
        count = self.i32()
        # R writes exactly (name, version) here -- two entries. The bound is
        # a defensive sanity check against corrupted streams, deliberately
        # far above anything R produces, not a protocol constant.
        if count < 1 or count > 1024:
            raise InvalidRDS(f"invalid namespace/package name count: {count}")
        names = [self.char() for _ in range(count)]
        kind = "r_namespace" if sexp_type == NAMESPACESXP else "r_package"
        marker = (kind, names[0] or "")
        self.add_reference(marker)
        return marker

    def skip_item(self) -> SkippedObject:
        """Consume one serialized item without allocating its vector payload."""
        self.depth += 1
        if self.depth > self.limits.max_depth:
            self.depth -= 1
            raise RDSLimitError("object nesting exceeds configured limit")
        try:
            return self._skip_item_from_header(self.flags())
        finally:
            self.depth -= 1

    def skip_item_from_header(
        self, header: tuple[int, bool, bool, bool, int]
    ) -> SkippedObject:
        self.depth += 1
        if self.depth > self.limits.max_depth:
            self.depth -= 1
            raise RDSLimitError("object nesting exceeds configured limit")
        try:
            return self._skip_item_from_header(header)
        finally:
            self.depth -= 1

    def _skip_item_from_header(
        self, header: tuple[int, bool, bool, bool, int]
    ) -> SkippedObject:
        sexp_type, _is_object, has_attr, has_tag, flags = header
        if sexp_type in {NILSXP, NILVALUE_SXP}:
            return SkippedObject(sexp_type)
        if sexp_type == REFSXP:
            index = flags >> 8 or self.i32()
            self._reference(index)
            return SkippedObject(sexp_type)
        if sexp_type == SYMSXP:
            symbol = ("sym", self.read_item())
            self.add_reference(symbol)
            return SkippedObject(sexp_type)
        if sexp_type == LISTSXP:
            self.skip_pairlist(has_tag)
            return SkippedObject(sexp_type)

        length: int | None = None
        if sexp_type in {INTSXP, LGLSXP}:
            length = self.length()
            self.discard(4 * length)
        elif sexp_type == REALSXP:
            length = self.length()
            self.discard(8 * length)
        elif sexp_type == CPLXSXP:
            length = self.length()
            self.discard(16 * length)
        elif sexp_type == RAWSXP:
            length = self.length()
            self._check_allocation(length, np.dtype(np.uint8).itemsize, "raw vector")
            self.discard(length)
        elif sexp_type == STRSXP:
            length = self.length()
            self.skip_string_elements(length)
        elif sexp_type == VECSXP:
            length = self.length()
            for _index in range(length):
                self.skip_item()
        elif sexp_type == CHARSXP:
            length = self.i32()
            if length < -1:
                raise InvalidRDS(f"invalid character length: {length}")
            if length > self.limits.max_string_bytes:
                raise RDSLimitError(
                    f"string length {length:,} exceeds configured limit "
                    f"{self.limits.max_string_bytes:,}"
                )
            if length >= 0:
                self.discard(length)
        elif sexp_type == ALTREP_SXP:
            self.skip_item()  # class metadata
            self.skip_item()  # state
            self.skip_item()  # ALTREP attributes
        elif sexp_type == EXTPTRSXP:
            self.add_reference(("extptr", None))
            self.skip_item()
            self.skip_item()
        elif sexp_type in {
            ENVSXP,
            S4SXP,
            NAMESPACESXP,
            PACKAGESXP,
            GLOBALENV_SXP,
            EMPTYENV_SXP,
            BASEENV_SXP,
            BASENAMESPACE_SXP,
            UNBOUNDVALUE_SXP,
            MISSINGARG_SXP,
        }:
            # These register reference-table entries during a full read; a
            # structural skip must register the exact same entries or every
            # later REFSXP index in the stream is off by one. They are small
            # (an environment frame, S4 slots), so delegating to the reader
            # keeps the two passes provably aligned.
            self._read_item_from_header(header)
            return SkippedObject(sexp_type)
        else:
            type_name = SEXP_TYPE_NAMES.get(sexp_type)
            raise UnsupportedRDS(
                f"R {type_name} objects are not supported"
                if type_name
                else f"cannot structurally skip SEXP type {sexp_type}"
            )

        if has_attr:
            self.skip_attributes()
        return SkippedObject(sexp_type, length)

    def skip_pairlist(self, has_tag: bool) -> None:
        current_has_tag = has_tag
        while True:
            if current_has_tag:
                self.skip_item()
            self.skip_item()
            header = self.flags()
            if header[0] in {NILSXP, NILVALUE_SXP}:
                return
            if header[0] != LISTSXP:
                # Dotted pair (ALTREP wrapper/deferred state): the CDR is a
                # plain object, not another cell; skip it and stop.
                self.skip_item_from_header(header)
                return
            current_has_tag = header[3]

    def skip_attributes(self) -> None:
        sexp_type, _is_object, _has_attr, has_tag, _flags = self.flags()
        if sexp_type in {NILSXP, NILVALUE_SXP}:
            return
        if sexp_type != LISTSXP:
            raise InvalidRDS("attribute payload is not a pairlist")
        self.skip_pairlist(has_tag)

    def read_payload(self, sexp_type: int, *, flags: int = 0) -> Any:
        if sexp_type in {INTSXP, LGLSXP}:
            return self.numeric_array(np.dtype(np.int32))
        if sexp_type == REALSXP:
            return self.numeric_array(np.dtype(np.float64))
        if sexp_type == STRSXP:
            return self.string_array()
        if sexp_type == VECSXP:
            length = self.length()
            self._check_allocation(length, 8, "list vector")
            return [self.read_item() for _ in range(length)]
        if sexp_type == CHARSXP:
            length = self.i32()
            if length == -1:
                return None
            if length < -1:
                raise InvalidRDS(f"invalid character length: {length}")
            if length > self.limits.max_string_bytes:
                raise RDSLimitError(
                    f"string length {length:,} exceeds configured limit "
                    f"{self.limits.max_string_bytes:,}"
                )
            return self._decode_charsxp(self.raw(length), flags)
        if sexp_type == RAWSXP:
            length = self.length()
            self._check_allocation(length, np.dtype(np.uint8).itemsize, "raw vector")
            array = np.empty(length, dtype=np.uint8)
            self.read_into(memoryview(array))
            return array
        if sexp_type == CPLXSXP:
            return self.numeric_array(np.dtype(np.complex128))
        if sexp_type == EXTPTRSXP:
            marker = ("extptr", None)
            self.add_reference(marker)
            self.read_item()
            self.read_item()
            return None
        if sexp_type == S4SXP:
            # An S4 object's data lives entirely in its attributes (the
            # slots), which the caller reads next via the has_attr bit.
            return None
        type_name = SEXP_TYPE_NAMES.get(sexp_type)
        raise UnsupportedRDS(
            f"R {type_name} objects are not supported"
            if type_name
            else f"SEXP type {sexp_type} is not supported"
        )

    def read_cons_chain(self, has_tag: bool) -> tuple[list[Any], Any]:
        """Read LISTSXP cells, tolerating untagged cars and a dotted CDR.

        ALTREP serialization uses raw cons structures that the tagged-only
        :meth:`read_pairlist` cannot represent: the class info is a proper but
        *untagged* list ``(class_sym pkg_sym type)``, and wrapper/deferred
        states are dotted pairs ``(payload . metadata)`` whose CDR is a plain
        vector rather than another cell. Returns ``(car_values, dotted_tail)``
        with ``dotted_tail=None`` for NIL-terminated chains.
        """
        values: list[Any] = []
        current_has_tag = has_tag
        while True:
            if current_has_tag:
                self.read_item()  # tag symbol; registered as a reference
            values.append(self.read_item())
            header = self.flags()
            if header[0] in {NILSXP, NILVALUE_SXP}:
                return values, None
            if header[0] != LISTSXP:
                return values, self.read_item_from_header(header)
            current_has_tag = header[3]

    def read_altrep(self) -> SerializedObject:
        """Materialize one serialized ALTREP object.

        The stream holds three items: class info, class-specific state, and
        the object's attributes. The attributes slot is genuinely load-bearing
        (``sort()`` of a factor wraps the integer codes and moves ``levels``/
        ``class`` here), so it is merged into the result rather than dropped.
        """
        info_header = self.flags()
        if info_header[0] == LISTSXP:
            info_values, _info_tail = self.read_cons_chain(info_header[3])
        else:
            info_values = [self.read_item_from_header(info_header)]
        class_name = ""
        for candidate in info_values:
            class_name = symbol_name(candidate)
            if class_name:
                break

        state_header = self.flags()
        state_tail: Any = None
        if state_header[0] == LISTSXP:
            state_values, state_tail = self.read_cons_chain(state_header[3])
            state = state_values[0] if state_values else None
        else:
            state = self.read_item_from_header(state_header)

        attributes_item = self.read_item()
        attributes = attributes_item if isinstance(attributes_item, dict) else {}

        if class_name in {"compact_intseq", "compact_realseq"}:
            array = as_array(state)
            if array is None or len(array) < 3:
                raise InvalidRDS(f"malformed {class_name} ALTREP state")
            length, start, step = int(array[0]), array[1], array[2]
            if length > self.limits.max_vector_length:
                raise RDSLimitError("ALTREP sequence exceeds configured limit")
            self._check_allocation(length, np.dtype(np.float64).itemsize, "ALTREP sequence")
            sequence = start + step * np.arange(length)
            if class_name == "compact_intseq":
                return SerializedObject(sequence.astype(np.int32), attributes, INTSXP)
            return SerializedObject(sequence, attributes, REALSXP)
        if class_name == "deferred_string":
            return SerializedObject(
                _deferred_string_values(state), attributes, STRSXP
            )
        if class_name.startswith("wrap_"):
            if isinstance(state, SerializedObject):
                # Wrapper metadata (the dotted CDR) only records sortedness /
                # no-NA hints; the payload and its attributes are the data.
                merged = {**state.attributes, **attributes}
                return SerializedObject(state.value, merged, state.sexp_type)
            raise UnsupportedRDS(f"ALTREP wrapper state is not a vector: {class_name}")
        _ = state_tail
        raise UnsupportedRDS(f"ALTREP class is not supported: {class_name or '?'}")

    def read_pairlist(self, has_tag: bool) -> dict[str, Any]:
        result: dict[str, Any] = {}
        current_has_tag = has_tag
        while True:
            tag = symbol_name(self.read_item()) if current_has_tag else None
            value = self.read_item()
            if tag is not None:
                result[tag] = value
            sexp_type, _is_object, _has_attr, next_has_tag, _flags = self.flags()
            if sexp_type in {NILSXP, NILVALUE_SXP}:
                break
            if sexp_type != LISTSXP:
                raise UnsupportedRDS("pairlist tail is not a LISTSXP")
            current_has_tag = next_has_tag
        return result

    def read_attributes(self) -> dict[str, Any]:
        sexp_type, _is_object, _has_attr, has_tag, _flags = self.flags()
        if sexp_type in {NILSXP, NILVALUE_SXP}:
            return {}
        if sexp_type != LISTSXP:
            raise InvalidRDS("attribute payload is not a pairlist")
        return self.read_pairlist(has_tag)

    def read_selected_attributes(self, names: frozenset[str]) -> dict[str, Any]:
        """Retain selected attributes while structurally skipping other values."""
        sexp_type, _is_object, _has_attr, has_tag, _flags = self.flags()
        if sexp_type in {NILSXP, NILVALUE_SXP}:
            return {}
        if sexp_type != LISTSXP:
            raise InvalidRDS("attribute payload is not a pairlist")
        result: dict[str, Any] = {}
        current_has_tag = has_tag
        while True:
            tag = symbol_name(self.read_item()) if current_has_tag else None
            if tag is not None and tag in names:
                result[tag] = self.read_item()
            else:
                self.skip_item()
            sexp_type, _is_object, _has_attr, next_has_tag, _flags = self.flags()
            if sexp_type in {NILSXP, NILVALUE_SXP}:
                return result
            if sexp_type != LISTSXP:
                raise UnsupportedRDS("pairlist tail is not a LISTSXP")
            current_has_tag = next_has_tag


# Bit pattern of R's NA_real_: an IEEE NaN whose low word is 1954. It must be
# told apart from an ordinary NaN because as.character() renders NA as missing
# but NaN as the string "NaN".
_R_NA_REAL_BITS = 0x7FF00000000007A2


def _deferred_string_values(state: Any) -> list[str | None]:
    """Expand a deferred_string ALTREP the way as.character() would.

    The state payload is the original numeric vector (itself often a compact
    sequence, already materialized by the recursive read). R formats doubles
    with up to 15 significant digits, which ``%.15g`` reproduces.
    """
    value = state.value if isinstance(state, SerializedObject) else state
    if not isinstance(value, np.ndarray):
        raise UnsupportedRDS("deferred_string ALTREP state is not a numeric vector")
    if value.dtype.kind == "i":
        return [None if item == NA_INTEGER else str(item) for item in value.tolist()]
    if value.dtype.kind == "f":
        bits = value.view(np.uint64).tolist()
        result: list[str | None] = []
        for item, bit_pattern in zip(value.tolist(), bits, strict=True):
            if bit_pattern == _R_NA_REAL_BITS:
                result.append(None)
            elif item != item:  # NaN check without importing math
                result.append("NaN")
            elif item == float("inf"):
                result.append("Inf")
            elif item == float("-inf"):
                result.append("-Inf")
            else:
                result.append(f"{item:.15g}")
        return result
    raise UnsupportedRDS(
        f"deferred_string ALTREP over dtype {value.dtype} is not supported"
    )


def as_value(obj: Any) -> Any:
    return obj.value if isinstance(obj, SerializedObject) else obj


def as_array(obj: Any) -> np.ndarray[Any, Any] | None:
    value = as_value(obj)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.asarray(value)
    return None


def as_strings(obj: Any) -> list[str]:
    value = as_value(obj)
    if value.__class__.__module__.startswith("pyarrow") and hasattr(value, "to_pylist"):
        value = value.to_pylist()
    if isinstance(value, (list, tuple, np.ndarray)):
        return ["" if item is None else str(item) for item in value]
    return []


def as_optional_strings(obj: Any) -> list[str | None]:
    """Like :func:`as_strings`, but keeps missing elements as ``None``."""
    value = as_value(obj)
    if value.__class__.__module__.startswith("pyarrow") and hasattr(value, "to_pylist"):
        value = value.to_pylist()
    if isinstance(value, (list, tuple, np.ndarray)):
        return [None if item is None else str(item) for item in value]
    return []


def factor_codes_and_categories(
    values: np.ndarray, levels: Any
) -> tuple[np.ndarray, list[str]]:
    """Zero-based codes (-1 = missing) plus non-null categories for a factor.

    R factors may carry an explicit NA level (``addNA()``); pandas and Arrow
    dictionaries cannot hold a null category, so codes pointing at an NA
    level become missing values here instead of decaying to the string ""
    (which would be indistinguishable from a genuine empty-string level).
    Out-of-range codes in a corrupted stream also become missing.
    """
    level_values = as_optional_strings(levels)
    count = len(level_values)
    codes = np.where(values == NA_INTEGER, -1, values.astype(np.int64) - 1)
    codes[(codes < -1) | (codes >= count)] = -1
    if any(level is None for level in level_values):
        mapping = np.empty(count + 1, dtype=np.int64)
        categories: list[str] = []
        for index, level in enumerate(level_values):
            if level is None:
                mapping[index] = -1
            else:
                mapping[index] = len(categories)
                categories.append(level)
        mapping[count] = -1  # the slot indexed by codes == -1
        return mapping[codes], categories
    return codes, [level for level in level_values if level is not None]


POSIXLT_COMPONENTS = ("sec", "min", "hour", "mday", "mon", "year")


def posixlt_wall_clock_components(
    attributes: dict[str, Any], value: list[Any]
) -> dict[str, np.ndarray]:
    """Extract POSIXlt's wall-clock component vectors as float64 arrays.

    R's `POSIXlt` is a plain named list of parallel vectors (`sec`, `min`,
    `hour`, `mday`, `mon` zero-based, `year` since 1900, plus `wday`/`yday`/
    `isdst`/`zone`/`gmtoff` that are not needed to reconstruct a timestamp).
    NA in an integer component (R's `NA_INTEGER` sentinel) and NaN in `sec`
    both become `float("nan")` here; callers treat any non-finite value as a
    missing row.
    """
    names = as_strings(attributes.get("names"))
    if len(names) != len(value):
        raise InvalidRDS("POSIXlt component count does not match its names")
    by_name = dict(zip(names, value, strict=True))
    missing = [name for name in POSIXLT_COMPONENTS if name not in by_name]
    if missing:
        raise UnsupportedRDS(f"POSIXlt is missing required component(s): {missing}")
    result: dict[str, np.ndarray] = {}
    length: int | None = None
    for name in POSIXLT_COMPONENTS:
        component = by_name[name]
        if isinstance(component, SerializedObject) and component.sexp_type == INTSXP:
            array = component.value.astype(np.float64)
            array[component.value == NA_INTEGER] = np.nan
        elif isinstance(component, SerializedObject) and component.sexp_type == REALSXP:
            array = component.value.astype(np.float64)
        else:
            array = np.asarray(as_value(component), dtype=np.float64)
        if length is None:
            length = len(array)
        elif len(array) != length:
            raise InvalidRDS("POSIXlt components have inconsistent lengths")
        result[name] = array
    return result


def symbol_name(obj: Any) -> str:
    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "sym":
        value = as_value(obj[1])
        return value if isinstance(value, str) else ""
    return ""


def decode_header(stream: BinaryIO) -> tuple[int, str, str | None]:
    magic = stream.read(2)
    if magic.startswith(b"RD"):
        rest = stream.read(3)
        raise UnsupportedRDS(f"RData container is not supported ({(magic + rest)!r})")
    if magic in {b"A\n", b"A\r"}:
        raise UnsupportedRDS("ASCII R serialization is not supported")
    if magic == b"B\n":
        # "Native" RDS has no self-describing byte order; R itself just
        # assumes the reading machine matches the writer. Rather than trust
        # that blindly, validate it: the version field must be 1-3, so if
        # the assumed byte order decodes something else, retry with the
        # opposite order before giving up. This turns a cross-architecture
        # native file into a clear error instead of silently wrong data.
        assumed = "<" if sys.byteorder == "little" else ">"
        header = stream.read(12)
        if len(header) != 12:
            raise InvalidRDS("truncated RDS header")
        version, _writer, _minimum = struct.unpack(f"{assumed}iii", header)
        if version in {1, 2, 3}:
            byteorder = assumed
        else:
            opposite = ">" if assumed == "<" else "<"
            alt_version, _alt_writer, _alt_minimum = struct.unpack(f"{opposite}iii", header)
            if alt_version not in {1, 2, 3}:
                raise InvalidRDS(
                    "native-format RDS header is not valid in either byte order; "
                    "the file may be corrupted, or written on a different-endian "
                    "machine than R's non-portable native format can describe"
                )
            byteorder, version = opposite, alt_version
    elif magic == b"X\n":
        byteorder = ">"
        header = stream.read(12)
        if len(header) != 12:
            raise InvalidRDS("truncated RDS header")
        version, _writer, _minimum = struct.unpack(f"{byteorder}iii", header)
    else:
        raise InvalidRDS(f"not a supported binary RDS stream (magic={magic!r})")
    native_encoding: str | None = None
    if version >= 3:
        length_raw = stream.read(4)
        if len(length_raw) != 4:
            raise InvalidRDS("truncated RDS encoding header")
        length = struct.unpack(f"{byteorder}i", length_raw)[0]
        if length < 0 or length > 1024:
            raise InvalidRDS(f"invalid native encoding length: {length}")
        native_encoding = stream.read(length).decode("ascii", "replace") if length else None
    return version, byteorder, native_encoding


_R_ENCODING_ALIASES = {
    "latin1": "latin-1",
    "latin-1": "latin-1",
    "iso-8859-1": "latin-1",
    "utf8": "utf-8",
    "utf-8": "utf-8",
}


def resolve_native_encoding(declared: str | None, override: str | None) -> str:
    """Pick the codec for CHARSXP elements with no explicit encoding flag.

    An explicit *override* always wins. Otherwise, prefer the encoding a
    version-3 RDS header actually declares (``declared``) over blindly
    assuming UTF-8: this is exactly the information R 3.5+ started
    recording for this purpose. Falls back to UTF-8 for version-2 files
    (no such header field exists) or a declared name Python cannot resolve.
    """
    if override is not None:
        try:
            codecs.lookup(override)
        except LookupError as exc:
            raise ValueError(f"unknown text encoding: {override!r}") from exc
        return override
    if not declared or declared.strip().lower() in {"", "unknown", "bytes", "native.enc"}:
        return "utf-8"
    normalized = declared.strip().lower()
    normalized = _R_ENCODING_ALIASES.get(normalized, normalized)
    try:
        codecs.lookup(normalized)
    except LookupError:
        return "utf-8"
    return normalized


_STREAM_BUFFER_SIZE = 1024 * 1024

# Frame magic of the zstd format, which R >= 4.5 writes for
# saveRDS(..., compress = "zstd").
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _open_zstd_reader(raw: Any) -> Any:
    """Open a decompressing reader over a zstd RDS container.

    Prefers the standard library (Python >= 3.14); otherwise uses the
    optional ``zstandard`` package. The source stream is never closed by
    the returned reader -- the caller manages its lifetime.
    """
    try:
        from compression import zstd  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        return zstd.ZstdFile(raw)
    try:
        import zstandard
    except ImportError as exc:
        raise ImportError(
            "this RDS is zstd-compressed (R >= 4.5, compress='zstd'); reading "
            "it requires Python >= 3.14 or: pip install 'rdsframe[zstd]'"
        ) from exc
    return zstandard.ZstdDecompressor().stream_reader(raw, closefd=False)


def is_buffer_source(source: Any) -> bool:
    """True for in-memory sources: bytes-like objects or binary file objects."""
    return isinstance(source, (bytes, bytearray, memoryview)) or (
        hasattr(source, "read") and not isinstance(source, (str, os.PathLike))
    )


class _DecompressionReadGuard(io.RawIOBase):
    """Re-raise decompressor read failures as :class:`InvalidRDS`.

    R writes RDS gzip-compressed by default (bzip2/xz/zstd are also common). A
    truncated or corrupted container makes the decompressor raise its own
    library-specific error -- ``EOFError``, ``gzip.BadGzipFile``,
    ``zlib.error``, ``lzma.LZMAError``, a zstd error, a bare ``OSError`` --
    *lazily*, deep inside the parse when the bytes are finally pulled. None of
    those subclass :class:`RDSError`, so without translation they leak past the
    documented "malformed input raises ``InvalidRDS``" contract precisely for
    the most common on-disk form (the uncompressed path already fails cleanly
    through the parser's own length/EOF checks). This raw wrapper sits between
    the decompressor and the outer :class:`io.BufferedReader` so every
    decompressed read is guarded at a single choke point without touching the
    hot per-element read primitives.
    """

    def __init__(self, decompressor: Any) -> None:
        super().__init__()
        self._decompressor = decompressor

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        try:
            readinto = getattr(self._decompressor, "readinto", None)
            if readinto is not None:
                count = readinto(buffer)
                return 0 if count is None else count
            chunk = self._decompressor.read(len(buffer))
        except (RDSError, MemoryError):
            # Our own contract errors and genuine resource exhaustion must not
            # be masked as a decompression failure.
            raise
        except Exception as exc:
            raise InvalidRDS("corrupt or truncated compressed RDS container") from exc
        count = len(chunk)
        buffer[:count] = chunk
        return count

    def close(self) -> None:
        try:
            self._decompressor.close()
        finally:
            super().close()


@contextmanager
def open_rds_stream(source: Any) -> Iterator[tuple[BinaryIO, BinaryIO, str]]:
    # The parser issues very many small reads (a flags/length pair per string
    # element, for instance). A large buffer keeps most of those hitting
    # already-buffered memory instead of the raw file or the decompressor.
    # A hand-rolled Python buffer was tried here and measured slower than
    # io.BufferedReader for this access pattern (~0.56s vs ~0.46s per 2M
    # small reads in a microbenchmark) despite avoiding a `raw.closed`
    # property check per call, so the standard library wrapper stays.
    owns_raw = True
    if isinstance(source, (bytes, bytearray, memoryview)):
        raw: Any = io.BytesIO(source if isinstance(source, bytes) else bytes(source))
    elif hasattr(source, "read") and not isinstance(source, (str, os.PathLike)):
        raw = source
        owns_raw = False  # the caller opened it; the caller closes it
        if not (hasattr(raw, "seek") and hasattr(raw, "tell")):
            raise TypeError("an RDS stream source must be seekable")
        if not isinstance(raw.read(0), bytes):
            raise TypeError("an RDS stream source must be opened in binary mode")
        raw.seek(0)
    else:
        raw = Path(source).open("rb", buffering=_STREAM_BUFFER_SIZE)  # noqa: SIM115
    stream: Any = raw
    try:
        magic = raw.read(6)
        raw.seek(0)
        decompressor: Any | None
        if magic.startswith(b"\x1f\x8b"):
            decompressor, compression = gzip.GzipFile(fileobj=raw), "gzip"
        elif magic.startswith(b"BZh"):
            decompressor, compression = bz2.BZ2File(raw), "bzip2"
        elif magic.startswith(b"\xfd7zXZ\x00"):
            decompressor, compression = lzma.LZMAFile(raw), "xz"  # noqa: SIM115
        elif magic.startswith(ZSTD_MAGIC):
            decompressor, compression = _open_zstd_reader(raw), "zstd"
        else:
            decompressor, compression = None, "none"
        if decompressor is not None:
            stream = io.BufferedReader(
                _DecompressionReadGuard(decompressor), buffer_size=_STREAM_BUFFER_SIZE
            )
        yield stream, raw, compression
    finally:
        if stream is not raw:
            stream.close()
        if owns_raw:
            raw.close()
