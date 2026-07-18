"""Functional smoke test executed against every built wheel.

cibuildwheel runs this inside the wheel-test environment (see
``[tool.cibuildwheel] test-command`` in pyproject.toml). Asserting
``compiled_backend_available()`` alone proves the extension *imports*; this
script proves it *works*: it builds a synthetic RDS in memory (no R needed),
reads it through the compiled scanner, re-runs the same reads in a child
process with ``RDSFRAME_DISABLE_CYTHON=1``, and requires byte-identical
results from both backends. The string column has more elements than the
adaptive threshold (1024) and is structurally skipped via ``columns=`` so the
compiled skip loop genuinely executes.
"""

from __future__ import annotations

import gzip
import json
import os
import struct
import subprocess
import sys
import tempfile
from typing import Any

STRING_ROWS = 1500  # above _CYTHON_STRING_MIN_ELEMENTS so the compiled loop runs
NA_INTEGER = -(2**31)


def _i32(value: int) -> bytes:
    return struct.pack(">i", value)


def _chars(value: str | None) -> bytes:
    if value is None:
        return _i32(9) + _i32(-1)
    encoded = value.encode()
    return _i32(9) + _i32(len(encoded)) + encoded


def _strings(values: list[str | None]) -> bytes:
    return _i32(16) + _i32(len(values)) + b"".join(_chars(v) for v in values)


def _integers(values: list[int]) -> bytes:
    return _i32(13) + _i32(len(values)) + struct.pack(f">{len(values)}i", *values)


def _symbol(name: str) -> bytes:
    return _i32(1) + _chars(name)


def _attributes(values: dict[str, bytes]) -> bytes:
    output = b""
    names = list(values)
    for index, name in enumerate(names):
        output += _i32(2 | (1 << 10)) if index == 0 else b""
        output += _symbol(name) + values[name]
        output += _i32(2 | (1 << 10)) if index < len(names) - 1 else _i32(254)
    return output


def build_rds() -> bytes:
    text = [
        None if i % 97 == 0 else ("día-中" if i % 5 == 0 else f"v{i % 13}")
        for i in range(STRING_ROWS)
    ]
    ints = [NA_INTEGER if i % 41 == 0 else i - 700 for i in range(STRING_ROWS)]
    frame = (
        _i32(19 | (1 << 9))  # VECSXP with attributes
        + _i32(2)
        + _integers(ints)
        + _strings(text)
        + _attributes(
            {"names": _strings(["i", "s"]), "class": _strings(["data.frame"])}
        )
    )
    return b"X\n" + struct.pack(">iii", 2, 0x040300, 0x030500) + frame


def collect() -> dict[str, Any]:
    import pandas as pd

    import rdsframe

    payload = build_rds()
    descriptor, path = tempfile.mkstemp(suffix=".rds")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)

        full = rdsframe.read_rds(path)
        assert isinstance(full, pd.DataFrame), type(full)
        # Structurally skips the 1500-element string column: with the
        # accelerator loaded this is the compiled skip loop.
        selective = rdsframe.read_rds(path, columns=[0])
        assert isinstance(selective, pd.DataFrame), type(selective)
        catalog = rdsframe.list_rds_tables(path)
        compressed = rdsframe.read_rds(gzip.compress(payload))
        assert isinstance(compressed, pd.DataFrame), type(compressed)

        return {
            "full_i": [None if pd.isna(v) else int(v) for v in full["i"].tolist()],
            "full_s": [None if v is None else str(v) for v in full["s"].tolist()],
            "selective_columns": list(selective.columns),
            "selective_i": [
                None if pd.isna(v) else int(v) for v in selective["i"].tolist()
            ],
            "catalog": [
                {
                    "name": table.name,
                    "rows": table.rows,
                    "columns": list(table.column_names),
                    "types": [column.r_type for column in table.schema],
                }
                for table in catalog.tables
            ],
            "gzip_equal": bool(compressed.equals(full)),
        }
    finally:
        os.unlink(path)


def main() -> int:
    import rdsframe

    if os.environ.get("RDSFRAME_SMOKE_CHILD") == "1":
        assert not rdsframe.compiled_backend_available(), (
            "RDSFRAME_DISABLE_CYTHON=1 must force the pure-Python backend"
        )
        print(json.dumps(collect(), sort_keys=True))
        return 0

    assert rdsframe.compiled_backend_available(), (
        "the built wheel did not load its compiled scanner"
    )
    accelerated = collect()
    child_env = dict(os.environ)
    child_env["RDSFRAME_DISABLE_CYTHON"] = "1"
    child_env["RDSFRAME_SMOKE_CHILD"] = "1"
    child = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=child_env,
        capture_output=True,
        text=True,
        check=True,
    )
    fallback = json.loads(child.stdout)
    if fallback != accelerated:
        raise SystemExit(
            "compiled and pure-Python backends disagree:\n"
            f"compiled: {json.dumps(accelerated, sort_keys=True)[:2000]}\n"
            f"fallback: {json.dumps(fallback, sort_keys=True)[:2000]}"
        )
    expected = sum(1 for i in range(STRING_ROWS) if i % 97 != 0)
    observed = sum(1 for v in accelerated["full_s"] if v is not None)
    assert observed == expected, (observed, expected)
    assert accelerated["gzip_equal"] is True
    print(f"wheel smoke OK: {STRING_ROWS} rows, compiled == fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
