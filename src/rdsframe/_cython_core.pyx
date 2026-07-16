# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Optional Cython accelerator for structural STRSXP skipping only.

The extension deliberately owns no I/O and no parser state. It scans only the
bytes already buffered by :class:`rdsframe._core.Reader`, returning at an
incomplete boundary so the Python reader remains responsible for refills,
limits, progress, and public exceptions.
"""

from libc.stdint cimport int32_t, uint32_t


cdef inline int32_t _i32(
    const unsigned char[::1] data, Py_ssize_t position, bint little_endian
) noexcept nogil:
    cdef uint32_t value
    if little_endian:
        value = (
            <uint32_t>data[position]
            | (<uint32_t>data[position + 1] << 8)
            | (<uint32_t>data[position + 2] << 16)
            | (<uint32_t>data[position + 3] << 24)
        )
    else:
        value = (
            (<uint32_t>data[position] << 24)
            | (<uint32_t>data[position + 1] << 16)
            | (<uint32_t>data[position + 2] << 8)
            | <uint32_t>data[position + 3]
        )
    return <int32_t>value


def skip_string_chunk(
    bytes buffer,
    Py_ssize_t position,
    Py_ssize_t count,
    Py_ssize_t max_string,
    Py_ssize_t references_len,
    bint little_endian,
):
    """Scan complete STRSXP elements in *buffer*.

    Returns ``(processed, position, status, value)``. Status 0 means complete
    or needs refill, 1 means a payload crosses the buffer boundary, and 2-5
    report malformed length, configured limit, reference, or SEXP type.
    """
    cdef const unsigned char[::1] data = buffer
    cdef Py_ssize_t size = data.shape[0]
    cdef Py_ssize_t processed = 0
    cdef Py_ssize_t start
    cdef int32_t flags
    cdef int32_t length
    cdef int32_t reference
    cdef int sexp_type

    while processed < count:
        start = position
        if size - position < 4:
            break
        flags = _i32(data, position, little_endian)
        position += 4
        sexp_type = flags & 0xFF
        if sexp_type == 9:  # CHARSXP
            if size - position < 4:
                position = start
                break
            length = _i32(data, position, little_endian)
            position += 4
            if length < -1:
                return processed, position, 2, length
            if length > max_string:
                return processed, position, 3, length
            if length > 0:
                if size - position < length:
                    return processed, position, 1, length
                position += length
            processed += 1
            continue
        if sexp_type == 255:  # REFSXP
            reference = flags >> 8
            if reference == 0:
                if size - position < 4:
                    position = start
                    break
                reference = _i32(data, position, little_endian)
                position += 4
            if reference < 1 or reference > references_len:
                return processed, position, 4, reference
            processed += 1
            continue
        if sexp_type == 0 or sexp_type == 254:  # NILSXP / NILVALUE_SXP
            processed += 1
            continue
        return processed, position, 5, sexp_type
    return processed, position, 0, 0
