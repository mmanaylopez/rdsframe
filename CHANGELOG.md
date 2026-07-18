# Changelog

## Unreleased

- Release engineering (external review follow-up): removed the stale
  `publish.yml` workflow -- it raced `release.yml` on every `v*` tag and had
  failed its Trusted Publishing OIDC exchange on every release since
  v0.4.0b1, leaving a spurious red X; `release.yml` is now the single
  publishing path. All GitHub Actions are pinned to full commit SHAs instead
  of mutable tags. Wheel testing goes beyond asserting the compiled scanner
  imports: `tools/wheel_smoke.py` reads a synthetic RDS through the compiled
  path (including the >=1024-element string skip loop) and requires identical
  results from the pure-Python fallback in a `RDSFRAME_DISABLE_CYTHON=1`
  child process. The minimum-dependencies CI job now also runs the full suite
  with the compiled backend against NumPy 1.23/pandas 1.5, not only the
  fallback.
- The pandas < 2 pyarrow-strings fallback is now an explicit version gate
  (`_PANDAS_BEFORE_2` plus a `to_pylist` capability check) instead of
  catching broad `TypeError`/`ValueError`: unforeseen construction failures
  surface instead of being silently retried.
- Added `benchmarks/bench.py`: one fresh process per (operation, repetition),
  median/min/max per category -- catalog scan, pandas read, Arrow read,
  Polars read -- with the input's compression stated alongside the results.

## 0.4.0b2 - 2026-07-18

- Fixed: a corrupt or truncated **compressed** RDS container (gzip/bzip2/xz/
  zstd) leaked the decompressor's own exception -- `EOFError`,
  `gzip.BadGzipFile`, `zlib.error`, `lzma.LZMAError`, a zstd error, or a bare
  `OSError` -- past the documented "malformed input raises `RDSError`"
  contract. Because R saves RDS gzip-compressed by *default*, this affected the
  most common on-disk form (the uncompressed path already failed cleanly
  through the parser's own length/EOF checks, so existing corrupt-input tests
  never exercised the decompressor error surface). Decompression is now guarded
  at a single choke point (`_DecompressionReadGuard`) so every such failure
  surfaces as `InvalidRDS`. A parametrized regression test corrupts and
  truncates every compression container and asserts only `RDSError` escapes.
- Added deferred `open_rds()` / `RDSDataset`: cached structural `schema`,
  `columns`, `shape`, table and column selection, `head()`/`collect()`, and
  terminal pandas, Arrow, Polars, and DuckDB conversions. Single-root column
  projections reuse the structural skip path; `head()` limits the result but
  does not pretend RDS supports row-random access.
- Added `inspect_rds(mode="metadata"|"scan")`. Metadata mode records storage
  and logical types, factors/levels, rows, compression, and fixed-width memory
  estimates without materializing columns. Scan mode explicitly adds exact
  Arrow null counts, buffer bytes, and Arrow types.
- Added `read_rds_polars()`, `read_rds_duckdb()`, `RDSDataset.to_polars()`,
  `.to_duckdb()`, and `.register_duckdb()`, plus a `rdsframe[polars]` extra.
  The DuckDB integration exposes an Arrow relation/view; a native SQL
  `read_rds(path)` extension remains future work.
- The catalog sidecar format is now version 2 (adds per-column schema:
  storage/logical type, factor levels, and fixed-width estimates); version 1
  sidecars still load.
- Packaging: the Trove classifier is now `Development Status :: 4 - Beta`,
  matching the beta version line.

## 0.4.0b1 - 2026-07-16

### Upgrade notes

- **`rdsframe[parquet]` no longer installs DuckDB.** It now brings only
  PyArrow, and `to_parquet()` defaults to `engine="auto"`: DuckDB when
  installed, PyArrow otherwise. Pipelines that rely on the memory-bounded,
  column-staged conversion should install `rdsframe[duckdb]` and may pass
  `engine="duckdb"` to make the requirement explicit. Outputs are equivalent;
  the PyArrow engine materializes each table in memory before writing, while
  the DuckDB engine keeps peak memory tied to a column batch. As a side
  effect, the PyArrow engine preserves Arrow dictionary (factor/categorical)
  types in the final Parquet, which the DuckDB staging step still
  re-materializes as plain strings.
- Exceeding a configured `ReaderLimits` bound now raises `RDSLimitError`
  everywhere; previously some limit violations surfaced as `UnsupportedRDS`.
  Code catching `RDSError` is unaffected.

### Changes

- Security/correctness: standalone `CHARSXP` values now honor their serialized
  encoding flags; invalid negative string lengths are classified as malformed
  data rather than configured-limit failures; environments preserve shared
  identity and self-cycles during Python conversion. The suite now includes
  bounded corrupt-file mutation and Hypothesis fuzz tests.
- Added public `read_rds_arrow()`, returning a `pyarrow.Table` (or a named dict
  of tables) without constructing pandas objects.
- Parquet no longer requires DuckDB: `rdsframe[parquet]` installs the PyArrow
  writer, while `rdsframe[duckdb]` enables the existing memory-bounded staged
  engine. `to_parquet(engine="auto"|"pyarrow"|"duckdb")` and the CLI expose
  the choice.
- Name-based table selection now automatically writes/reuses a validated
  catalog sidecar. CI adds strict mypy, an 80% coverage floor, NumPy 1.23.5 /
  pandas 1.5.3 minimum-dependency tests, and corrupt/fuzz coverage.
- Added zstd container support: R >= 4.5 writes `saveRDS(..., compress =
  "zstd")` (R-universe already ships zstd `PACKAGES.rds`), and such files
  previously failed with a misleading "not a supported binary RDS stream"
  while `inspect_r_file()` reported `compression="none"`. Reading now works
  through the standard library on Python >= 3.14 or the new
  `rdsframe[zstd]` extra (`zstandard`); without either, the error names the
  compression and the install command. `inspect_r_file()` reports "zstd".
- Fixed: a factor with an explicit NA level (`addNA()`) decoded that level
  as the string "", indistinguishable from a genuine empty-string level.
  Codes pointing at an NA level now become missing values (pandas and
  Arrow dictionaries cannot represent a null category), and out-of-range
  codes in corrupted streams become missing in the Arrow path too instead
  of producing an invalid dictionary index.
- `read_r_object()` fidelity: a *named* atomic vector (`c(a = 1, b = 2)`)
  now becomes a `dict` (matching named lists) instead of silently dropping
  its names; a class-less matrix keeps its native NumPy dtype instead of
  always decaying to `dtype=object` (`dimnames` are still not represented);
  a `dim` attribute inconsistent with the vector length now raises
  `InvalidRDS` instead of an unhandled reshape error.
- Performance: converting R lists that contain many small vectors was
  dominated by pandas Series construction (measured ~80% of wall time on a
  real 24k-entry index file; 27.5s -> 5.0s after the change). Class-less
  atomic vectors now convert to Python values directly, with the exact
  same missing-value sentinels as the pandas round-trip (`pd.NA` for
  integer/logical NA, `None`/`pd.NA` for character NA depending on the
  strings mode). `_column_to_pandas` now returns arrays rather than
  Series so DataFrame assembly skips per-column index alignment, and the
  column-length consistency check runs before DataFrame construction so a
  malformed file still raises `InvalidRDS` rather than a pandas error.
- Performance: Date/POSIXct/difftime columns now use direct NumPy temporal
  arrays instead of one `pd.to_datetime`/`pd.to_timedelta` setup per tiny
  column, and object/list columns no longer require an intermediate Series.
  On the real 27,315-table `archive.rds`, the already-optimized 0.4.0a7 worktree
  dropped from 35.2s to 20.2s while preserving pandas dtypes and missing values.
- Added an optional Cython prototype limited to the CHARSXP structural-skip loop. It is built
  from generated portable C when a compiler is available and falls back to the
  fully tested Python loop when it is not. An adaptive threshold keeps small
  vectors on Python because extension-call setup outweighs the compiled loop
  there. On the real 123 MiB text-heavy file,
  catalog scanning improved from 16.7s to 3.9s (4.3x); a later isolated run
  measured 32.6s versus 7.1s (4.56x) with identical output.
  `compiled_backend_available()` reports which path was loaded;
  `RDSFRAME_DISABLE_CYTHON=1` supports fallback diagnostics.
- `list_rds_tables(..., cache=True)` now atomically writes and reuses the
  validated `<source>.rdsframe.json` sidecar. A custom cache path is accepted;
  corrupt or stale automatic caches are rebuilt. The CLI exposes `list --cache`.

- Fixed: a data.frame column that is itself a data.frame was read (and
  written to Parquet) **silently transposed** whenever the nested frame was
  square -- its columns were presented as row values. The README already
  promised a clear `UnsupportedRDS` for this shape; now `read_rds()`,
  `read_r_object()`, `to_parquet()`, and `list_rds_tables()` all raise
  "data.frame-valued data.frame columns are not supported". A data.frame
  whose *every* column is a data.frame was additionally misclassified by
  the streaming Parquet path as a list of independent tables; the root
  class attribute (which R serializes last) is now checked before any
  output is renamed into place, so no partial results are left behind.
  Unselected nested-frame columns can still be skipped with
  `read_rds(..., columns=[...])`.
- Fixed: `POSIXct` and `difftime` columns stored with integer
  `storage.mode` (legal in R; common in DB-imported frames) decayed to
  plain integer columns, silently dropping the time semantics. The
  integer payload is now normalized to the same rules as the double
  representation in both the pandas and Arrow paths (`NA_integer_`
  becomes `NaT`/null; the `units` and `tzone` attributes are honored).

## 0.4.0a7 - 2026-07-13

- Fixed a peak-memory regression introduced in 0.4.0a6's Arrow string path
  (reported by a downstream review): `arrow_string_array()` materialized
  the whole column as a Python list of bytes objects plus an interning dict
  before building the Arrow buffers, putting peak memory at roughly twice
  the column's text. Elements are now drained into the Arrow buffers in
  bounded chunks of 262,144 rows (`_STRING_CHUNK`, the default Parquet
  row-group granularity), restoring the ~one-column peak of 0.4.0a5 while
  keeping the batched parser's speed; conversion timings are unchanged.
  A comparative tracemalloc test pins the mechanism, and chunk-boundary
  tests (tiny patched chunks, a payload larger than the batch buffer
  landing on a chunk edge) pin output equivalence with the object path.
- Interning now applies only to the object-strings mode, where the returned
  list is the long-lived result; in the Arrow mode the pieces are transient
  per chunk, so the per-element dict lookup was pure overhead.
- Removed a no-op try/except in `read_item_from_header()` that caught
  `InvalidRDS`/`UnsupportedRDS` only to re-raise them.
- Documented two deliberate trade-offs flagged by the same review: an
  environment's enclosure chain is fully read (not skipped) because parent
  scopes register in the reference table and a later REFSXP resolving to a
  skipped-but-registered parent would be silently wrong; and the
  namespace/package name-count bound (1024) is a defensive sanity check far
  above the two entries R actually writes, not a protocol constant.

## 0.4.0a6 - 2026-07-12

- Batched STRSXP parsing: string vectors are now decoded (and structurally
  skipped) from large chunks with `struct.unpack_from` instead of three
  stream calls per element, with the overshoot parked in a pending buffer
  that every other read primitive drains first. On the real 123 MiB gzip
  benchmark file, `list_rds_tables()` dropped from 66.7s to 17.1s (9.7x
  total versus 0.4.0a1's 165.4s), `read_rds()` on a 186 MiB / 557k-row
  dataset from 4.6s to 3.0s, and a catalog-assisted `to_parquet()` of a
  589k-row table runs in 5.0s.
- S4 objects are now readable through `read_r_object()`/`rdsframe dump` as
  a dict of their slots (class under `"$r_class"`), including nested
  data.frame slots. Validated against files written by R 4.5.0.
- Environments (ENVSXP) are now readable as a dict of their frame and hash
  table contents, with reference-table alignment preserved when the same
  environment appears multiple times, and namespace/package parent scopes
  handled. Global/base/empty environment markers decode as
  `{"$r_environment": ...}`.
- Closures, language calls, promises, bytecode, and the remaining
  non-data R types now fail with an error that names the R type
  ("R closure (R function) objects are not supported") instead of a bare
  SEXP number, so applications can tell users exactly why a slower
  general-purpose fallback is being engaged.
- The `encoding=` override now also covers strings whose gp flags claim
  UTF-8 but whose bytes are not valid UTF-8 (mislabeled files): with an
  override present they are validated and re-decoded; without one the
  zero-cost trust-the-flag fast path is unchanged.
- Added `materialize_uncompressed()`: one-time decompression of a
  gzip/bzip2/xz RDS to disk (atomic when given a destination) so that
  repeated selective access can use the seek-based skip path afterwards.
- Documented the representation trade-offs Parquet imposes: complex
  columns as `STRUCT(real, imag)`, `difftime` normalized to
  `duration("us")` (exact value, display unit not preserved in the type),
  and the deliberate explicit-only policy for heterogeneous list-columns.

## 0.4.0a5 - 2026-07-12

- Rewrote ALTREP deserialization after validating against files written by
  real R 4.5.0 (a corpus generated with `tests/data/gen_fixtures.R` now lives in
  `tests/data/r450` as golden regression files). The previous parser lost
  the ALTREP class name (the info pairlist is untagged, and tagless entries
  were dropped) and could not represent the dotted-pair state of wrapper
  classes, so **any modern data.frame containing a plain `1:n` column
  failed** with "ALTREP class is not supported: ?". Compact integer/real
  sequences, deferred `as.character` strings (formatted exactly as R does,
  including the NA-vs-NaN distinction), and `wrap_*` wrappers now all read
  correctly.
- The ALTREP attributes slot is now honored instead of discarded; it is
  load-bearing (e.g. `sort()` of a factor stores `levels`/`class` there),
  so dropping it could silently strip a factor down to bare integers.
- Added `POSIXlt` support: the component list (`year`/`mon`/`mday`/`hour`/
  `min`/`sec`) is reconstructed into timezone-naive wall-clock timestamps in
  both the pandas and Parquet paths; invalid or NA components become NaT.
- Character row names now become the pandas index; R's compact default
  numbering (`c(NA, -n)`) still yields a plain `RangeIndex`.
- Native (non-XDR) RDS headers are validated in both byte orders: a
  cross-endian file now produces a correct read or a clear `InvalidRDS`
  instead of silently wrong numbers.
- Strings with no explicit encoding flag now default to the encoding the
  version-3 RDS header itself declares (when recognized) instead of always
  UTF-8, and all read functions plus `rdsframe convert`/`dump` accept an
  `encoding=` override for older files (e.g. `windows-1252`).
- `read_rds()`, `read_rds_dataframe()`, and `read_r_object()` accept raw
  `bytes`/`bytearray`/`memoryview` and seekable binary streams in addition
  to paths, so RDS content received from an API never has to touch disk.
- Added `rdsframe dump`: prints any supported R object as a truncated,
  indented tree (`--max-items`, `--max-depth`) or as complete JSON
  (`--json`), for exploring unknown RDS files from the terminal.

## 0.4.0a4 - 2026-07-11

- Added `read_r_object()`: reads any supported R value, not only a
  data.frame or a list of data.frames. A named list becomes a `dict`, an
  unnamed list becomes a `list`, nested `data.frame`s inside either are
  still converted to pandas DataFrames, and atomic vectors get the same
  type handling as a data.frame column (factor, ordered factor, `Date`,
  `POSIXct`, `difftime`, matrices via `dim`) but unwrapped to a plain Python
  scalar/list instead of a pandas Series.
- `read_rds()`/`list_rds_tables()` now mention `read_r_object()` in their
  `UnsupportedRDS` message when the root is not a data.frame or list of
  data.frames, instead of leaving the caller to guess at an alternative.
- Motivated by two real files that `read_rds()` correctly rejects because
  they are not tabular: a plain named list (24,206 entries, serialization
  version 2) and a nested list of results tables. Both are now fully
  readable: the named list becomes a `dict[str, dict[str, str]]`, and the
  nested file resolves into a list of lists containing pandas DataFrames
  that a shallow structural inspection during the 0.4.0a3 work had missed
  entirely.

## 0.4.0a3 - 2026-07-11

- Fixed: ordered factors (`class(x) = c("ordered", "factor")`) now become an
  ordered `pd.Categorical` / an `ordered` Arrow dictionary instead of always
  silently dropping the ordering and producing an unordered categorical.
- Fixed: `difftime` columns now become a proper pandas `timedelta64` Series
  (Arrow `duration("us")` for `to_parquet()`), honoring the R `units`
  attribute (`secs`/`mins`/`hours`/`days`/`weeks`). Previously the elapsed
  count was kept but the unit was silently discarded, so the values read
  back as a plain, unlabeled float.
- Fixed: a data.frame column carrying a `dim` attribute (an R matrix or
  array stored as one column) now raises a clear
  `UnsupportedRDS("matrix- or array-valued data.frame columns are not
  supported")` instead of either a confusing "columns have different
  lengths" error, or -- for the `dim = c(n, 1)` case specifically -- silently
  succeeding as if it were a plain length-n vector, discarding the fact that
  it was a matrix.
- Validated against 7 real-world RDS files beyond the original benchmark
  file, gathered specifically to avoid over-fitting to one dataset: a
  24,206-entry plain named list (serialization v2 -- correctly rejected as
  not a data.frame), a 27,315-table metadata snapshot, a 186 MB uncompressed
  557,691-row dataset, two tool-generated result files, and a plain nested
  list-of-lists object -- all either convert correctly or fail with the same
  clear, unchanged error as before.

## 0.4.0a2 - 2026-07-11

- Added `read_rds(..., columns=[...])`: structurally skip unselected columns
  of a single data.frame instead of materializing every column. Integer
  indices take one pass; names take a bounded-memory structural pass to
  resolve positions first, matching the existing table-selection contract.
  The saving is largest in RAM and for wide tables with many unwanted
  numeric columns (skipping one is a real `seek()` on uncompressed RDS);
  skipping a text column still costs one `skip_char()` per row, the same
  count as reading it, so the time saved there is smaller and mainly avoids
  the decode/intern/pandas-object cost, not the row traversal itself.
- Reworked the CHARSXP hot path (`char()`, `char_utf8()`, `skip_char()`),
  which runs once per string element and dominates parsing time for
  text-heavy tables: cached the read/readinto bound methods and a
  precompiled `struct.Struct` instead of re-deriving a format string per
  call, and dropped the unused is_object/has_attr/has_tag tuple that
  `flags()` built on every element.
- Added `Reader.char_utf8()`, used by the Arrow string builder, which keeps
  already-UTF-8/ASCII bytes (per R's own gp flags) as-is instead of a
  decode-then-re-encode round trip; only genuine latin-1 or unflagged
  native-encoding strings still pay for a transcode.
- Gave `discard()` a single-read fast path for payloads that fit in one
  buffer (the common case: one string) instead of always entering the
  chunked copy loop, and made both `discard()` and `read_into()` skip the
  `tick()` call entirely when no `progress` callback is set.
- Wrapped compressed streams (gzip/bzip2/xz) in an outer `io.BufferedReader`
  and widened the raw file's buffer, so the very many small header reads
  hit an in-memory buffer instead of the decompressor or the OS per call.
- Benchmarked on a real 123 MB gzip RDS (6 tables, ~5.5M rows, many
  repeated-text columns): `list_rds_tables()` dropped from 165.4s to 66.7s
  (~2.5x) with identical catalog output; no public behavior changed.
- Fixed a bug found while validating `columns=`: its internal `Reader` did
  not propagate `seekable_discard`, so skipped numeric columns never took
  the fast `seek()` path even on uncompressed RDS. Fixed before release.
- Full validation before this release: 52/52 tests green (including the
  DuckDB-backed ones), `to_parquet()` run end-to-end against the same real
  123 MB file with results cross-checked via `duckdb.sql(...)`, and
  `python -m build` + `twine check` both pass on the built sdist/wheel.

## 0.4.0a1 - 2026-07-11

- Added `list_rds_tables()` with bounded-memory structural payload skipping.
- Added `RDSCatalog` and `RTableInfo` with rows, columns, and field names.
- Added atomic JSON catalog persistence and stale-source validation.
- Added `extract_rds_tables()` and `to_parquet(..., tables=..., catalog=...)`.
- Added one-pass integer selection and catalog-assisted early termination.
- Added exact-name selection with missing and duplicate-name diagnostics.
- Added CLI `list`, `--table-index`, `--table-name`, and `--catalog` workflows.
- Added `RDSCatalogError` and tests proving unselected tables do not reach Arrow.
- Added direct seek-based payload skipping for uncompressed RDS with truncation checks.

## 0.3.0a1 - 2026-07-10

- Added fail-fast `max_tables` and `max_root_items` limits with no partial output.
- Added fidelity-preserving and UTC-naive policies for `POSIXct` export.
- Added explicit error/null handling for infinite or out-of-range timestamps.
- Added opt-in deterministic JSON and string policies for heterogeneous list-columns.
- Added raw-vector support and lossless complex representation as an Arrow struct.
- Expanded the CLI to expose memory, staging, limit, timestamp, and list policies.
- Replaced tracebacks for expected CLI errors with concise diagnostics and exit code 2.

## 0.2.0 - 2026-07-10

- Replaced one-temporary-file-per-column with adaptive Arrow staging batches.
- Added `stage_max_columns`, `stage_max_bytes`, and `gc_collect_every` controls.
- Reduced explicit cyclic-GC calls from every column to a configurable interval.
- Added Arrow-backed strings to the in-memory API with `strings="pyarrow"`.
- Clarified the intentional RData and unsupported-ALTREP fallback contract.

## 0.1.0 - 2026-07-10

- Initial public package structure and typed API.
- Direct NumPy allocation for atomic numeric vectors and Arrow-native strings.
- RDS inspection, pandas conversion, column-staged Parquet export, and CLI.
- Configurable parser limits and explicit malformed/unsupported exceptions.
- Test suite for XDR, gzip, nullable dtypes, factors, dates, CLI, and truncation.
