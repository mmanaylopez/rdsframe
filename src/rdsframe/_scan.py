"""Low-allocation structural scans for RDS table discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._core import (
    INTSXP,
    NA_INTEGER,
    VECSXP,
    Reader,
    SerializedObject,
    SkippedObject,
    as_strings,
    as_value,
)


@dataclass(frozen=True, slots=True)
class ScannedFrame:
    rows: int | None
    columns: int
    column_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScannedVector:
    summary: SkippedObject
    attributes: dict[str, Any]
    first_element_length: int | None = None


def scan_vector_from_header(
    reader: Reader, header: tuple[int, bool, bool, bool, int]
) -> ScannedVector:
    """Skip a vector-like object but retain its small attribute pairlist."""
    sexp_type, _is_object, has_attr, _has_tag, _flags = header
    if sexp_type != VECSXP:
        return ScannedVector(reader.skip_item_from_header(header), {})
    length = reader.length()
    first_element_length: int | None = None
    for index in range(length):
        item = reader.skip_item()
        if index == 0:
            first_element_length = item.length
    attributes = reader.read_attributes() if has_attr else {}
    return ScannedVector(
        SkippedObject(VECSXP, length), attributes, first_element_length
    )


def as_dataframe(scan: ScannedVector) -> ScannedFrame | None:
    if scan.summary.sexp_type != VECSXP:
        return None
    if "data.frame" not in as_strings(scan.attributes.get("class")):
        return None
    columns = scan.summary.length or 0
    names = as_strings(scan.attributes.get("names"))
    if not names:
        names = [f"V{index + 1}" for index in range(columns)]
    rows = _rows_from_attributes(scan.attributes)
    if rows is None:
        rows = scan.first_element_length
    return ScannedFrame(rows, columns, tuple(names))


def scan_dataframe_from_header(
    reader: Reader, header: tuple[int, bool, bool, bool, int]
) -> ScannedFrame | None:
    return as_dataframe(scan_vector_from_header(reader, header))


def _rows_from_attributes(attributes: dict[str, Any]) -> int | None:
    row_names = attributes.get("row.names")
    if row_names is None:
        return None
    value = as_value(row_names)
    if (
        isinstance(row_names, SerializedObject)
        and row_names.sexp_type == INTSXP
        and isinstance(value, np.ndarray)
    ):
        if len(value) == 2 and value[0] == NA_INTEGER and value[1] <= 0:
            return int(-value[1])
        return len(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value)
    return None
