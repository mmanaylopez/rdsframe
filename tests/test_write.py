from __future__ import annotations

import datetime as dt
import io
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdsframe import RDSWriteError, read_rds, write_rds


def _find_rscript() -> str | None:
    found = shutil.which("Rscript")
    if found:
        return found
    for root in (Path("C:/Program Files/R"), Path("C:/Program Files (x86)/R")):
        if root.is_dir():
            candidates = sorted(root.glob("R-*/bin/Rscript.exe"), reverse=True)
            if candidates:
                return str(candidates[0])
    return None


RSCRIPT = _find_rscript()
requires_r = pytest.mark.skipif(RSCRIPT is None, reason="Rscript is not available")


def _roundtrip(frame: pd.DataFrame, tmp_path: Path, **options) -> pd.DataFrame:
    target = write_rds(frame, tmp_path / "out.rds", **options)
    assert target is not None and target.is_file()
    result = read_rds(target)
    assert isinstance(result, pd.DataFrame)
    return result


def test_roundtrip_basic_types(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "i": np.array([1, -2, 3], dtype=np.int32),
            "im": pd.array([1, None, 3], dtype="Int32"),
            "v": [1.5, float("nan"), -2.25],
            "b": [True, False, True],
            "bm": pd.array([True, None, False], dtype="boolean"),
            "s": ["a", None, "día-中"],
            "sd": pd.array(["x", None, "z"], dtype="string"),
        }
    )
    result = _roundtrip(frame, tmp_path)
    assert list(result.columns) == list(frame.columns)
    assert result["i"].tolist() == [1, -2, 3]
    assert result["im"][0] == 1 and pd.isna(result["im"][1])
    assert result["v"][0] == 1.5 and np.isnan(result["v"][1])
    assert result["b"].tolist() == [True, False, True]
    assert result["bm"][0] is True or result["bm"][0] == True  # noqa: E712
    assert pd.isna(result["bm"][1])
    assert result["s"].tolist() == ["a", None, "día-中"]
    assert result["sd"].tolist() == ["x", None, "z"]


def test_factor_roundtrip(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "f": pd.Categorical(
                ["low", "high", None], categories=["high", "low"]
            ),
            "o": pd.Categorical(
                [1, 2, 2], categories=[1, 2, 3], ordered=True
            ),
        }
    )
    result = _roundtrip(frame, tmp_path)
    factor = result["f"]
    assert list(factor.cat.categories) == ["high", "low"]
    assert factor.tolist()[:2] == ["low", "high"] and pd.isna(factor[2])
    assert not factor.cat.ordered
    ordered = result["o"]
    # R factor levels are always character; integer categories stringify
    # exactly like factor() in R would.
    assert list(ordered.cat.categories) == ["1", "2", "3"]
    assert ordered.cat.ordered
    assert ordered.tolist() == ["1", "2", "2"]


def test_datetime_roundtrip(tmp_path: Path) -> None:
    naive = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-06-01 12:30:00.5"),
            pd.NaT,
            pd.Timestamp("1969-12-31 23:59:59"),
        ]
    )
    aware = naive.tz_localize("UTC").tz_convert("America/Lima")
    frame = pd.DataFrame({"t": naive, "tz": aware})
    result = _roundtrip(frame, tmp_path, naive_timezone="UTC")
    written = pd.DatetimeIndex(result["t"])
    assert str(written.tz) == "UTC"  # the declared zone travels with the file
    assert written[0] == pd.Timestamp("2020-06-01 12:30:00.5", tz="UTC")
    assert pd.isna(written[1])
    assert written[2] == pd.Timestamp("1969-12-31 23:59:59", tz="UTC")
    zoned = pd.DatetimeIndex(result["tz"])
    assert str(zoned.tz) == "America/Lima"
    assert zoned[0] == aware[0]
    assert pd.isna(zoned[1])


def test_naive_datetime_requires_timezone_policy(tmp_path: Path) -> None:
    """R displays tzone-less POSIXct in the reader's own timezone, so a
    naive column written blindly shows a different wall time there."""
    frame = pd.DataFrame({"t": pd.to_datetime(["2020-06-01 12:30:00"])})
    with pytest.raises(RDSWriteError, match="naive_timezone"):
        write_rds(frame, tmp_path / "naive.rds")
    with pytest.raises(ValueError, match="IANA zone"):
        write_rds(frame, tmp_path / "naive.rds", naive_timezone="Marte/Olympus")
    result = _roundtrip(frame, tmp_path, naive_timezone="America/Lima")
    written = pd.DatetimeIndex(result["t"])
    assert str(written.tz) == "America/Lima"
    # The wall-clock reading is preserved under the declared zone.
    assert written[0].tz_localize(None) == pd.Timestamp("2020-06-01 12:30:00")


def test_timedelta_roundtrip(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"dur": pd.to_timedelta(["1h", None, "-30s", "2.5s"])}
    )
    result = _roundtrip(frame, tmp_path)
    assert result["dur"][0] == pd.Timedelta(hours=1)
    assert pd.isna(result["dur"][1])
    assert result["dur"][2] == pd.Timedelta(seconds=-30)
    assert result["dur"][3] == pd.Timedelta(seconds=2.5)


def test_date_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"d": pd.to_datetime(["2020-01-01", None, "1969-12-31"])}
    )
    result = _roundtrip(frame, tmp_path, date_columns=["d"])
    assert result["d"][0] == pd.Timestamp("2020-01-01")
    assert pd.isna(result["d"][1])
    assert result["d"][2] == pd.Timestamp("1969-12-31")

    objects = pd.DataFrame(
        {"d": [dt.date(2020, 1, 1), None, dt.date(1969, 12, 31)]}
    )
    from_objects = _roundtrip(objects, tmp_path)
    assert from_objects["d"][0] == pd.Timestamp("2020-01-01")

    not_midnight = pd.DataFrame({"d": pd.to_datetime(["2020-01-01 08:00"])})
    with pytest.raises(RDSWriteError, match="non-midnight"):
        write_rds(not_midnight, tmp_path / "bad.rds", date_columns=["d"])

    with pytest.raises(ValueError, match="date_columns not found"):
        write_rds(frame, tmp_path / "bad.rds", date_columns=["missing"])


def test_int64_policy(tmp_path: Path) -> None:
    frame = pd.DataFrame({"big": [1, 2**40]})
    with pytest.raises(RDSWriteError, match="32-bit range"):
        write_rds(frame, tmp_path / "big.rds")
    result = _roundtrip(frame, tmp_path, int64="double")
    assert result["big"].tolist() == [1.0, float(2**40)]

    # -2**31 is R's NA sentinel: writing it as data would silently become NA.
    sentinel = pd.DataFrame({"x": np.array([-(2**31)], dtype=np.int64)})
    with pytest.raises(RDSWriteError, match="NA"):
        write_rds(sentinel, tmp_path / "sentinel.rds")


def test_int64_double_rejects_precision_loss(tmp_path: Path) -> None:
    """2**53 + 1 rounds to 2**53 as float64; 'double' must refuse, and the
    check must run on the original integers (the converted value hides it)."""
    exact_edge = pd.DataFrame({"x": [2**53, -(2**53)]})
    result = _roundtrip(exact_edge, tmp_path, int64="double")
    assert result["x"].tolist() == [float(2**53), float(-(2**53))]

    beyond = pd.DataFrame({"x": [2**53 + 1]})
    with pytest.raises(RDSWriteError, match="2\\*\\*53"):
        write_rds(beyond, tmp_path / "beyond.rds", int64="double")
    lossy = _roundtrip(beyond, tmp_path, int64="lossy_double")
    assert lossy["x"].tolist() == [float(2**53)]  # the documented rounding

    unsigned = pd.DataFrame({"x": np.array([2**63 + 8], dtype=np.uint64)})
    with pytest.raises(RDSWriteError, match="lossy_double"):
        write_rds(unsigned, tmp_path / "unsigned.rds", int64="double")
    lossy_unsigned = _roundtrip(unsigned, tmp_path, int64="lossy_double")
    assert lossy_unsigned["x"].tolist() == [float(2**63)]


def test_factor_levels_match_r_and_reject_collisions(tmp_path: Path) -> None:
    booleans = pd.DataFrame({"f": pd.Categorical([True, False, True])})
    result = _roundtrip(booleans, tmp_path)
    # R's as.character() renders TRUE/FALSE, not Python's True/False.
    assert sorted(result["f"].cat.categories) == ["FALSE", "TRUE"]

    colliding = pd.DataFrame({"f": pd.Categorical.from_codes([0, 1], [1, "1"])})
    with pytest.raises(RDSWriteError, match="collide"):
        write_rds(colliding, tmp_path / "collide.rds")

    unsupported = pd.DataFrame(
        {"f": pd.Categorical.from_codes([0], [(1, 2)])}
    )
    with pytest.raises(RDSWriteError, match="unsupported type"):
        write_rds(unsupported, tmp_path / "tuple.rds")


def test_row_names_must_be_unique_and_present(tmp_path: Path) -> None:
    duplicated = pd.DataFrame({"x": [1, 2]}, index=["a", "a"])
    with pytest.raises(RDSWriteError, match="duplicates"):
        write_rds(duplicated, tmp_path / "dup.rds")

    missing = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index([0.5, float("nan")]))
    with pytest.raises(RDSWriteError, match="missing"):
        write_rds(missing, tmp_path / "nan.rds")

    colliding = pd.DataFrame({"x": [1, 2]}, index=pd.Index([1, "1"], dtype=object))
    with pytest.raises(RDSWriteError, match="collide"):
        write_rds(colliding, tmp_path / "collide.rds")


def test_unsupported_list_column_fails_as_write_error(tmp_path: Path) -> None:
    """A cell holding a list used to escape as an ambiguous ValueError from
    pd.isna(); it must surface as RDSWriteError naming the column."""
    frame = pd.DataFrame({"lst": [[1, 2], [3]]})
    with pytest.raises(RDSWriteError, match="'lst'"):
        write_rds(frame, tmp_path / "lst.rds")


@pytest.mark.parametrize("compress", ["gzip", "bzip2", "xz", "none"])
def test_compressions(tmp_path: Path, compress: str) -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "s": ["a", "b"]})
    result = _roundtrip(frame, tmp_path, compress=compress)
    assert result["x"].tolist() == [1.0, 2.0]
    assert result["s"].tolist() == ["a", "b"]


def test_zstd_compression(tmp_path: Path) -> None:
    pytest.importorskip("zstandard")
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    result = _roundtrip(frame, tmp_path, compress="zstd")
    assert result["x"].tolist() == [1.0, 2.0]


def test_dict_of_frames(tmp_path: Path) -> None:
    data = {
        "first": pd.DataFrame({"a": [1, 2]}),
        "second": pd.DataFrame({"s": ["x", "y"]}),
    }
    target = write_rds(data, tmp_path / "multi.rds")
    result = read_rds(target)
    assert isinstance(result, dict)
    assert list(result) == ["first", "second"]
    assert result["first"]["a"].tolist() == [1, 2]
    assert result["second"]["s"].tolist() == ["x", "y"]


def test_row_names(tmp_path: Path) -> None:
    default = _roundtrip(pd.DataFrame({"x": [1, 2, 3]}), tmp_path)
    assert isinstance(default.index, pd.RangeIndex)

    labeled = pd.DataFrame({"x": [1.0, 2.0]}, index=["r1", "r2"])
    assert _roundtrip(labeled, tmp_path).index.tolist() == ["r1", "r2"]

    numbered = pd.DataFrame({"x": [1.0, 2.0]}, index=[10, 20])
    assert _roundtrip(numbered, tmp_path).index.tolist() == [10, 20]


def test_stream_output(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1, 2]})
    buffer = io.BytesIO()
    assert write_rds(frame, buffer, compress="none") is None
    result = read_rds(buffer.getvalue())
    assert isinstance(result, pd.DataFrame)
    assert result["x"].tolist() == [1, 2]


def test_deterministic_output(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1.5, 2.5], "s": ["a", "b"]})
    write_rds(frame, tmp_path / "a.rds")
    write_rds(frame, tmp_path / "b.rds")
    assert (tmp_path / "a.rds").read_bytes() == (tmp_path / "b.rds").read_bytes()


def test_na_real_bit_pattern(tmp_path: Path) -> None:
    """A float NaN must be written as R's NA_real_, not an ordinary NaN."""
    target = write_rds(
        pd.DataFrame({"v": [float("nan")]}), tmp_path / "na.rds", compress="none"
    )
    payload = target.read_bytes()
    assert struct.pack(">Q", 0x7FF00000000007A2) in payload


def test_write_errors(tmp_path: Path) -> None:
    out = tmp_path / "x.rds"
    with pytest.raises(RDSWriteError, match="mixing types"):
        write_rds(pd.DataFrame({"m": ["a", 1]}), out)
    with pytest.raises(RDSWriteError, match="MultiIndex"):
        frame = pd.DataFrame(
            {"x": [1]}, index=pd.MultiIndex.from_tuples([("a", 1)])
        )
        write_rds(frame, out)
    with pytest.raises(RDSWriteError, match="does not support"):
        write_rds(
            pd.DataFrame({"p": pd.period_range("2020", periods=2, freq="Y")}), out
        )
    with pytest.raises(TypeError, match="DataFrame"):
        write_rds([1, 2, 3], out)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty mapping"):
        write_rds({}, out)
    with pytest.raises(TypeError, match="non-empty strings"):
        write_rds({"": pd.DataFrame({"x": [1]})}, out)
    with pytest.raises(RDSWriteError, match="fixed-offset"):
        index = pd.to_datetime(["2020-01-01"]).tz_localize(
            dt.timezone(dt.timedelta(hours=5))
        )
        write_rds(pd.DataFrame({"t": index}), out)
    with pytest.raises(ValueError, match="compress"):
        write_rds(pd.DataFrame({"x": [1]}), out, compress="lz4")  # type: ignore[arg-type]


def test_empty_frames(tmp_path: Path) -> None:
    empty_rows = _roundtrip(pd.DataFrame({"x": pd.array([], dtype="Int32")}), tmp_path)
    assert list(empty_rows.columns) == ["x"] and len(empty_rows) == 0


_R_CHECK_SCRIPT = """\
args <- commandArgs(trailingOnly = TRUE)
x <- readRDS(args[[1]])
stopifnot(is.data.frame(x), nrow(x) == 3L)
stopifnot(is.integer(x$i), is.na(x$i[2]), identical(x$i[c(1L, 3L)], c(1L, 3L)))
stopifnot(is.double(x$v), identical(x$v[1], 1.5))
stopifnot(is.na(x$v[3]), !is.nan(x$v[3]))  # NA_real_, not a plain NaN
stopifnot(is.logical(x$b), is.na(x$b[2]), isTRUE(x$b[1]), identical(x$b[3], FALSE))
stopifnot(is.character(x$s), is.na(x$s[2]), identical(x$s[1], "a"))
stopifnot(identical(nchar(x$s[3], type = "chars"), 5L))  # UTF-8 survived
stopifnot(is.factor(x$f), identical(levels(x$f), c("high", "low")), is.na(x$f[3]))
stopifnot(inherits(x$o, "ordered"), identical(levels(x$o), c("1", "2", "3")))
stopifnot(inherits(x$t, "POSIXct"), identical(attr(x$t, "tzone"), "America/Lima"))
stopifnot(abs(as.numeric(x$t[1]) - 1590969600.5) < 1e-6, is.na(x$t[2]))
stopifnot(inherits(x$d, "Date"), identical(as.integer(x$d[1]), 18262L), is.na(x$d[2]))
stopifnot(inherits(x$dur, "difftime"), identical(attr(x$dur, "units"), "secs"))
stopifnot(abs(as.numeric(x$dur[1]) - 3600) < 1e-9, is.na(x$dur[3]))
saveRDS(x, args[[2]], compress = "gzip")
cat("R_OK\\n")
"""


@requires_r
def test_r_reads_written_file_and_roundtrips(tmp_path: Path) -> None:
    """R itself must read the written file, validate it, and round-trip it."""
    frame = pd.DataFrame(
        {
            "i": pd.array([1, None, 3], dtype="Int32"),
            "v": [1.5, 2.5, float("nan")],
            "b": pd.array([True, None, False], dtype="boolean"),
            "s": ["a", None, "día-中"],
            "f": pd.Categorical(["low", "high", None], categories=["high", "low"]),
            "o": pd.Categorical([1, 2, 2], categories=[1, 2, 3], ordered=True),
            "t": pd.DatetimeIndex(
                [
                    pd.Timestamp("2020-06-01 00:00:00.5"),
                    pd.NaT,
                    pd.Timestamp("2021-01-01"),
                ]
            )
            .tz_localize("UTC")
            .tz_convert("America/Lima"),
            "d": pd.to_datetime(["2020-01-01", None, "1969-12-31"]),
            "dur": pd.to_timedelta(["1h", "90s", None]),
        }
    )
    written = tmp_path / "from_python.rds"
    rewritten = tmp_path / "from_r.rds"
    write_rds(frame, written, date_columns=["d"])

    script = tmp_path / "check.R"
    script.write_text(_R_CHECK_SCRIPT, encoding="utf-8")
    assert RSCRIPT is not None
    completed = subprocess.run(
        [RSCRIPT, str(script), str(written), str(rewritten)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "R_OK" in completed.stdout

    ours = read_rds(written)
    theirs = read_rds(rewritten)
    assert isinstance(ours, pd.DataFrame) and isinstance(theirs, pd.DataFrame)
    pd.testing.assert_frame_equal(ours, theirs, check_dtype=False)
