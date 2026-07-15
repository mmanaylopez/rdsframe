# Preliminary benchmarks

These numbers validate architecture, not universal performance claims. Run the
same comparison on real files before tuning defaults.

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

The remaining listing time is dominated by per-element loop iterations in
pure Python over ~44M string headers; the stream-call overhead is gone.
Parallelizing independent columns across processes is the next lever, but
requires either a seekable (uncompressed) source or per-column
re-decompression, and is out of scope for this release.
