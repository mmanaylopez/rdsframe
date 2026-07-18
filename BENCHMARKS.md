# Preliminary benchmarks

These numbers validate architecture, not universal performance claims. Run the
same comparison on real files before tuning defaults.

## Reproducing (median + spread)

The historical tables below were single warm-process runs. For new
measurements use `benchmarks/bench.py`, which runs every (operation,
repetition) in a fresh Python process and reports median/min/max per
category -- catalog scan, full pandas read, Arrow read, and Polars read are
timed separately, and the file's compression is printed with the results so
compressed and uncompressed inputs are never mixed in one table:

```bash
python benchmarks/bench.py file.rds --reps 5
python benchmarks/bench.py file.rds --reps 7 --ops catalog,pandas
```

## Synthetic 15-table RDS

Environment: Linux, Python 3.12, DuckDB 1.5.4, PyArrow 25.0.0. Input was an
uncompressed 17.17 MiB XDR RDS containing 15 data.frames of 100,000 rows and two
numeric columns each. Every measurement was one warm process run on 2026-07-11.

| Operation | Tables written | Seconds |
| --- | ---: | ---: |
| Structural catalog | 0 | 0.0020 |
| Convert all tables | 15 | 0.6327 |
| Extract first 3 with catalog | 3 | 0.1314 |
| Extract tables 1, 8, and 15 with catalog | 3 | 0.1683 |

The uncompressed catalog benefits from direct seek-based payload skipping. The
spread selection must traverse farther through the source but still avoids
Arrow conversion, temporary staging, and final Parquet generation for 12 of 15
tables.

Compressed RDS input cannot skip compressed bytes randomly: listing must still
decompress the stream, and extraction time depends on the last selected table.
Text cardinality, compression, storage, row width, and DuckDB configuration may
materially change these ratios.

## Real-world 123 MiB gzip RDS (0.4.0a2 CHARSXP hot-path rework)

Environment: Windows, Python 3.11, DuckDB 1.5.4, PyArrow 23.0.1, pandas 2.0.3.
Input: a real production RDS (gzip, XDR, 6 `data.frame`s, ~5.5M rows total),
with many text-heavy columns of highly repeated string values. One warm
process run per measurement.

| Version | Operation | Seconds |
| --- | --- | ---: |
| 0.4.0a1 | `list_rds_tables()` (structural catalog, all 6 tables) | 165.4 |
| 0.4.0a2 | `list_rds_tables()` (same file, identical output) | 66.7 |

The 2.5x improvement comes entirely from the CHARSXP hot path (see
`AUDITORIA_TECNICA.md`): cached read methods, a precompiled `struct.Struct`,
a single-read fast path in `discard()`, skipping `tick()` when there is no
progress callback, and a larger buffer around the decompressor. No public
behavior or output changed; the full test suite (52/52, including the
DuckDB-backed tests) stayed green, and `to_parquet()` was re-run end-to-end
against the same file with row counts cross-checked via `duckdb.sql(...)`.

`read_rds(path, columns=[...])` was also benchmarked on a synthetic
uncompressed 300,000-row / 30-column table (2 text columns, 28 numeric):
selecting 3 columns by index took 0.74s versus 0.79s to read all 30 — a
real but modest gain, because most of the skipped columns were already
cheap. Skipping large numeric columns in an uncompressed file is close to
free (a `seek()`); skipping a text column still costs one header visit per
row, matching the cost of reading it. See the "Memory model" section of
`README.md` for what this means in practice.

## Batched STRSXP parsing (0.4.0a6)

Every CHARSXP carries its own header, so element-by-element traversal is
inherent to the RDS format — but paying three stream calls per string is
not. 0.4.0a6 parses string vectors from large chunks with `unpack_from`,
banking the overshoot in a pending buffer that the ordinary read primitives
drain. Same environment and files as above, one warm run each:

| Operation | 0.4.0a5 | 0.4.0a6 | vs. 0.4.0a1 |
| --- | ---: | ---: | ---: |
| `list_rds_tables()`, 123 MiB gzip, 6 tables, ~5.5M rows | 66.7s | 17.1s | 165.4s → 17.1s (9.7x) |
| `read_rds()`, 186 MiB uncompressed, 557k rows x 33 cols | 4.6s | 3.0s | — |
| `to_parquet()` of one 589k-row table with catalog | — | 5.0s | — |

The remaining listing time was dominated by per-element loop iterations in
pure Python over ~44M string headers; the stream-call overhead was gone.

## Optional Cython scanner and pandas tiny-table path (2026-07-16)

Environment: Windows, Python 3.12.13, pandas 3.0.3, NumPy 2.5.1. Each number is
one fresh-process run against the real local audit files. The compiled module
owns no I/O or parser state: it scans buffered CHARSXP elements only and
returns to the Python Reader for refills, limits, progress, and exceptions.

| Input / operation | Python fallback | Cython | Speedup |
| --- | ---: | ---: | ---: |
| `datos_limpios.rds`: `list_rds_tables()` | 16.692s | 3.866s | 4.32x |

The fallback was forced with `RDSFRAME_DISABLE_CYTHON=1` and the full suite was
run both ways. Vectors below 1,024 elements stay on the Python batch loop: the
extension-call/memoryview setup costs more than it saves at that size. For
`archive.rds` (27,315 tiny tables), replacing Series/index
alignment and pandas datetime conversion with array-like/NumPy paths reduced
the unprofiled run from 35.160s to 20.248s. Its residual cost is dominated by
the 27,315 requested pandas DataFrame objects rather than RDS parsing.

The skip-only comparison is reproducible (and verifies identical catalogs)
with:

```console
python benchmarks/benchmark_cython_skip.py ../datos_limpios.rds --repeat 3
```

A post-change validation run on 2026-07-16 (`--repeat 1`) measured 32.559s for
Python and 7.147s for Cython, a **4.56x** speedup, with identical catalogs. The
absolute times moved with machine load; the ratio agrees with the controlled
4.32x measurement above.

## Cross-reader comparison (2026-07-16)

Environment: Windows 11, Python 3.12.13, pandas 3.0.3, NumPy 2.5.1, pyreadr
0.5.6 (librdata, C), rdata 1.1.0. Each cell is one cold run in a fresh
subprocess; peak RSS is the process `PeakWorkingSetSize`. Synthetic inputs
were generated with R 4.5.0 (2M x 8 numeric; 1M x 5 text with mixed
cardinality); the real inputs are nflverse `play_by_play_2023.rds` (gzip,
49,665 x 372) and a 123 MiB gzip production file whose root is a named list
of 6 data.frames (~5.2M rows total).

| Input / reader | rdsframe | pyreadr | rdata |
| --- | ---: | ---: | ---: |
| numeric 2M x 8, uncompressed | 0.66 s / 199 MB | 1.82 s / 469 MB | 1.27 s / 490 MB |
| numeric 2M x 8, gzip | 1.25 s / 208 MB | 2.27 s / 469 MB | 1.78 s / 515 MB |
| text 1M x 5, uncompressed | 2.63 s / 228 MB | 16.5-17.3 s / 418 MB | 39.3 s / 1,744 MB |
| text 1M x 5, gzip | 2.83 s / 229 MB | 2.46 s / 418 MB | -- |
| play-by-play 49,665 x 372 | 4.31 s / 322 MB | 4.73 s / 766 MB | 94.0 s / 3,678 MB |
| 123 MiB gzip, 6-table list root | 32 s / ~1.7 GB | **`{}`, silent** | impractical |
| 5.5 MiB gzip, 27,315 tiny tables | 21 s / 773 MB | **`{}`, silent** | 59.4 s / 1,621 MB |

Notes and caveats:

- pyreadr/librdata does not support an RDS whose root is a list of
  data.frames: it returns an empty dict without raising. The timings shown
  for those two rows are therefore not comparable -- pyreadr did no work.
- pyreadr's uncompressed text case was re-run twice (16.5 s, 17.3 s); its
  gzip path is the fast one. This is a librdata characteristic, not noise.
- Correctness was cross-checked, not assumed: 372/372 play-by-play columns
  compare equal against pyreadr, and column checksums (string lengths, sums,
  level counts) match R 4.5.0 itself on the synthetic text table.
- `strings="pyarrow"` lowers rdsframe's peak on the text table further
  (165 MB); `read_rds_arrow()` avoids pandas entirely.
