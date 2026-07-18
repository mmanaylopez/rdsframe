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

_FRAME_ATTRIBUTES = frozenset(
    {"class", "dim", "levels", "names", "row.names", "tzone", "units"}
)
_COLUMN_ATTRIBUTES = frozenset({"class", "dim", "levels", "tzone", "units"})


@dataclass(frozen=True, slots=True)
class ScannedFrame:
    rows: int | None
    columns: int
    column_names: tuple[str, ...]
    column_scans: tuple[ScannedColumn, ...] = ()


@dataclass(frozen=True, slots=True)
class ScannedColumn:
    summary: SkippedObject
    attributes: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScannedVector:
    summary: SkippedObject
    attributes: dict[str, Any]
    first_element_length: int | None = None
    column_scans: tuple[ScannedColumn, ...] = ()


def scan_vector_from_header(
    reader: Reader, header: tuple[int, bool, bool, bool, int]
) -> ScannedVector:
    """Skip a vector-like object but retain its small attribute pairlist."""
    sexp_type, _is_object, has_attr, _has_tag, _flags = header
    if sexp_type != VECSXP:
        column = scan_column_from_header(reader, header)
        return ScannedVector(column.summary, column.attributes)
    length = reader.length()
    first_element_length: int | None = None
    column_scans: list[ScannedColumn] = []
    for index in range(length):
        item = scan_column_from_header(reader, reader.flags())
        column_scans.append(item)
        if index == 0:
            first_element_length = item.summary.length
    attributes = reader.read_selected_attributes(_FRAME_ATTRIBUTES) if has_attr else {}
    return ScannedVector(
        SkippedObject(VECSXP, length),
        attributes,
        first_element_length,
        tuple(column_scans),
    )


def scan_column_from_header(
    reader: Reader, header: tuple[int, bool, bool, bool, int]
) -> ScannedColumn:
    """Skip one column payload while retaining its small attribute pairlist."""
    sexp_type, is_object, has_attr, has_tag, flags = header
    payload_header = (sexp_type, is_object, False, has_tag, flags)
    summary = reader.skip_item_from_header(payload_header)
    attributes = (
        reader.read_selected_attributes(_COLUMN_ATTRIBUTES) if has_attr else {}
    )
    return ScannedColumn(summary, attributes)


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
    return ScannedFrame(rows, columns, tuple(names), scan.column_scans)


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
