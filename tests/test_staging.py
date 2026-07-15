from __future__ import annotations

from unittest.mock import patch

from rdsframe._parquet import _batch_is_full, _maybe_collect


def test_adaptive_batch_limits() -> None:
    batch = [(0, object()), (1, object())]
    assert not _batch_is_full(batch, 40, 20, max_columns=4, max_bytes=100)
    assert _batch_is_full(batch, 40, 70, max_columns=4, max_bytes=100)
    assert _batch_is_full(batch * 2, 40, 20, max_columns=4, max_bytes=100)
    assert not _batch_is_full([], 0, 1_000, max_columns=4, max_bytes=100)


def test_gc_runs_at_interval_boundaries() -> None:
    with patch("rdsframe._parquet.gc.collect") as collect:
        next_at = _maybe_collect(8, 16, 16)
        assert next_at == 16
        collect.assert_not_called()

        next_at = _maybe_collect(20, 16, next_at)
        assert next_at == 32
        collect.assert_called_once()


def test_gc_can_be_disabled() -> None:
    with patch("rdsframe._parquet.gc.collect") as collect:
        assert _maybe_collect(1_000, 0, 0) == 0
        collect.assert_not_called()
