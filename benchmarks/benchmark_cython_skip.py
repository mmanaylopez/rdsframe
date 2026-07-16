from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

CHILD = r"""
import json
import sys
import time
from rdsframe import compiled_backend_available, list_rds_tables

started = time.perf_counter()
catalog = list_rds_tables(sys.argv[1])
elapsed = time.perf_counter() - started
signature = [
    [table.index, table.name, table.rows, table.columns, list(table.column_names)]
    for table in catalog.tables
]
print(json.dumps({
    "seconds": elapsed,
    "compiled": compiled_backend_available(),
    "signature": signature,
}))
"""


def _run(path: Path, *, cython: bool) -> dict[str, Any]:
    environment = os.environ.copy()
    if cython:
        environment.pop("RDSFRAME_DISABLE_CYTHON", None)
    else:
        environment["RDSFRAME_DISABLE_CYTHON"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", CHILD, str(path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the Cython skip_string_elements scanner with Python"
    )
    parser.add_argument("path", type=Path, help="RDS file used by list_rds_tables()")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    path = args.path.expanduser().resolve()
    if not path.is_file():
        parser.error(f"file not found: {path}")

    samples: dict[str, list[float]] = {"python": [], "cython": []}
    expected: Any = None
    for iteration in range(args.repeat):
        order = (False, True) if iteration % 2 == 0 else (True, False)
        for cython in order:
            result = _run(path, cython=cython)
            if cython and not result["compiled"]:
                raise RuntimeError("the compiled Cython backend is unavailable")
            if expected is None:
                expected = result["signature"]
            elif result["signature"] != expected:
                raise RuntimeError("Python and Cython produced different catalogs")
            key = "cython" if cython else "python"
            samples[key].append(float(result["seconds"]))

    python_median = statistics.median(samples["python"])
    cython_median = statistics.median(samples["cython"])
    print(
        json.dumps(
            {
                "file": str(path),
                "operation": "list_rds_tables (skip_string_elements)",
                "repeat": args.repeat,
                "python_seconds": samples["python"],
                "cython_seconds": samples["cython"],
                "python_median_seconds": python_median,
                "cython_median_seconds": cython_median,
                "speedup": python_median / cython_median,
                "catalogs_equal": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
