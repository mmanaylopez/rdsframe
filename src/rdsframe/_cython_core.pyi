def skip_string_chunk(
    buffer: bytes,
    position: int,
    count: int,
    max_string: int,
    references_len: int,
    little_endian: bool,
) -> tuple[int, int, int, int]: ...
