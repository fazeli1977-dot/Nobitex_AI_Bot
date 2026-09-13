"""Deterministic tests for indicators.py (Nobitex_AI_Bot). Python 3.12, stdlib only."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.indicators import (  # noqa: E402
    IndicatorSnapshot,
    atr,
    build_snapshot,
    ema,
    rsi,
    sma,
)

NAN = float("nan")


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def last_valid(series):
    return next(v for v in reversed(series) if not math.isnan(v))


class TestSMA(unittest.TestCase):
    def test_basic_values(self):
        got = sma([1, 2, 3, 4, 5], 3)
        self.assertEqual(len(got), 5)
        self.assertTrue(math.isnan(got[0]) and math.isnan(got[1]))
        self.assertTrue(close(got[2], 2.0))
        self.assertTrue(close(got[3], 3.0))
        self.assertTrue(close(got[4], 4.0))

    def test_period_equal_to_length(self):
        got = sma([2.0, 4.0, 6.0], 3)
        self.assertEqual(len(got), 3)
        self.assertTrue(close(got[-1], 4.0))

    def test_insufficient_data(self):
        with self.assertRaises(ValueError):
            sma([1, 2], 3)

    def test_invalid_period(self):
        for bad in (0, -1, 2.5, "3", True):
            with self.assertRaises(ValueError):
                sma([1, 2, 3], bad)

    def test_bad_values(self):
        for bad in ([], [1, float("nan"), 3], [1, float("inf")], [1, "2"], None):
            with self.assertRaises(ValueError):
                sma(bad, 2)


class TestEMA(unittest.TestCase):
    def test_seed_is_sma_then_recursion(self):
        vals = [10, 11, 12, 13, 14, 15]
        got = ema(vals, 3)
        self.assertEqual(len(got), len(vals))
        self.assertTrue(math.isnan(got[0]) and math.isnan(got[1]))
        self.assertTrue(close(got[2], 11.0))  # SMA seed of [10,11,12]
        alpha = 2.0 / 4.0
        expected = 11.0
        for v in vals[3:]:
            expected = alpha * v + (1 - alpha) * expected
            # progressively check each step
        self.assertTrue(close(got[-1], expected, tol=1e-12))
        # step-by-step check
        e3 = alpha * 13 + (1 - alpha) * 11.0
        e4 = alpha * 14 + (1 - alpha) * e3
        e5 = alpha * 15 + (1 - alpha) * e4
        self.assertTrue(close(got[3], e3))
        self.assertTrue(close(got[4], e4))
        self.assertTrue(close(got[5], e5))

    def test_insufficient_data(self):
        with self.assertRaises(ValueError):
            ema([1, 2], 3)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ema([], 1)
        with self.assertRaises(ValueError):
            ema([1, None, 2], 2)
        with self.assertRaises(ValueError):
            ema([1, 2, 3], 0)


class TestRSI(unittest.TestCase):
    def test_fully_rising_is_100(self):
        vals = list(range(1, 31))  # strictly increasing
        got = rsi(vals, 14)
        self.assertEqual(len(got), len(vals))
        self.assertTrue(math.isnan(got[13]))  # first valid at index 14
        self.assertTrue(close(got[14], 100.0))
        self.assertTrue(all(close(v, 100.0) for v in got[14:]))

    def test_constant_series_is_50(self):
        vals = [100.0] * 20
        got = rsi(vals, 14)
        self.assertTrue(math.isnan(got[13]))
        self.assertTrue(close(got[14], 50.0))
        self.assertTrue(all(close(v, 50.0) for v in got[14:]))

    def test_first_valid_index_is_period(self):
        vals = [1, 2, 3, 2, 4, 5, 4, 6, 7, 6, 8, 9, 8, 10, 11, 12]
        got = rsi(vals, 14)
        self.assertTrue(math.isnan(got[13]))
        self.assertFalse(math.isnan(got[14]))

    def test_manual_wilder_check(self):
        # Hand-computed: period=2, vals=[1,2,3,2,1]
        # initial (indices 1,2): +1,+1 -> avg_gain=1, avg_loss=0 -> rsi=100
        # i=3: change=-1 -> avg_gain=0.5, avg_loss=0.5 -> rsi=50
        # i=4: change=-1 -> avg_gain=0.25, avg_loss=0.75 -> rs=1/3 -> rsi=25
        got = rsi([1, 2, 3, 2, 1], 2)
        self.assertTrue(close(got[2], 100.0))
        self.assertTrue(close(got[3], 50.0))
        self.assertTrue(close(got[4], 25.0))

    def test_insufficient_data(self):
        with self.assertRaises(ValueError):
            rsi([1] * 14, 14)  # needs 15
        with self.assertRaises(ValueError):
            rsi([], 14)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            rsi([1, 2, 3, 4], 2.0)
        with self.assertRaises(ValueError):
            rsi([1, 2, float("nan"), 4], 2)


class TestATR(unittest.TestCase):
    def test_basic_wilder(self):
        highs = [10, 11, 12, 13, 14]
        lows = [9, 10, 11, 12, 13]
        closes = [9.5, 10.5, 11.5, 12.5, 13.5]
        got = atr(highs, lows, closes, 3)
        self.assertEqual(len(got), 5)
        self.assertTrue(math.isnan(got[0]) and math.isnan(got[1]))
        # TRs: 1, max(1,|11-9.5|,|10-9.5|)=1.5, max(1,|12-10.5|,|11-10.5|)=1.5,
        #      max(1,1.5,1.5)=1.5, max(1,1.5,1.5)=1.5
        self.assertTrue(close(got[2], (1 + 1.5 + 1.5) / 3))  # first ATR = SMA of TRs
        # Wilder: (1.6667*2 + 1.5)/3
        prev = (1 + 1.5 + 1.5) / 3
        e3 = (prev * 2 + 1.5) / 3
        e4 = (e3 * 2 + 1.5) / 3
        self.assertTrue(close(got[3], e3))
        self.assertTrue(close(got[4], e4))

    def test_first_tr_is_high_minus_low(self):
        highs = [15.0, 15.0]
        lows = [10.0, 10.0]
        closes = [12.0, 14.0]
        got = atr(highs, lows, closes, 2)
        self.assertTrue(close(got[1], (5.0 + max(5.0, 3.0, 2.0)) / 2))

    def test_size_mismatch(self):
        with self.assertRaises(ValueError):
            atr([1, 2], [1, 2, 3], [1, 2], 2)

    def test_low_gt_high(self):
        with self.assertRaises(ValueError):
            atr([1, 2], [3, 1], [1, 2], 2)

    def test_insufficient_data(self):
        with self.assertRaises(ValueError):
            atr([1, 2], [1, 2], [1, 2], 3)

    def test_non_finite(self):
        with self.assertRaises(ValueError):
            atr([1, float("inf")], [1, 2], [1, 2], 2)


class TestBuildSnapshot(unittest.TestCase):
    def setUp(self):
        self.closes = [100 + i * 1.0 for i in range(30)]
        self.highs = [c + 1.0 for c in self.closes]
        self.lows = [c - 1.0 for c in self.closes]

    def test_fields_and_trend_bullish(self):
        snap = build_snapshot(self.closes, self.highs, self.lows,
                              fast_period=5, slow_period=10)
        self.assertIsInstance(snap, IndicatorSnapshot)
        self.assertEqual(snap.trend, "bullish")  # rising series -> ema_fast > ema_slow
        self.assertTrue(close(snap.sma_fast, last_valid(sma(self.closes, 5))))
        self.assertTrue(close(snap.ema_slow, last_valid(ema(self.closes, 10))))
        self.assertTrue(close(snap.rsi, 100.0))  # fully rising
        self.assertIsNotNone(snap.atr)
        self.assertTrue(close(snap.volatility_pct, (snap.atr / self.closes[-1]) * 100))
        self.assertEqual(
            snap.to_dict(),
            {
                "sma_fast": snap.sma_fast,
                "sma_slow": snap.sma_slow,
                "ema_fast": snap.ema_fast,
                "ema_slow": snap.ema_slow,
                "rsi": snap.rsi,
                "atr": snap.atr,
                "volatility_pct": snap.volatility_pct,
                "trend": snap.trend,
            },
        )

    def test_trend_bearish_and_neutral(self):
        falling = [100 - i for i in range(30)]
        snap = build_snapshot(falling, fast_period=5, slow_period=10,
                              compute_atr=False)
        self.assertEqual(snap.trend, "bearish")
        self.assertIsNone(snap.atr)
        flat = [50.0] * 30
        snap2 = build_snapshot(flat, fast_period=5, slow_period=10,
                               compute_atr=False)
        self.assertEqual(snap2.trend, "neutral")
        self.assertTrue(close(snap2.rsi, 50.0))
        self.assertEqual(snap2.volatility_pct, 0.0)

    def test_fast_must_be_smaller_than_slow(self):
        with self.assertRaises(ValueError):
            build_snapshot(self.closes, fast_period=10, slow_period=10)
        with self.assertRaises(ValueError):
            build_snapshot(self.closes, fast_period=21, slow_period=9)

    def test_atr_requires_highs_lows(self):
        with self.assertRaises(ValueError):
            build_snapshot(self.closes, compute_atr=True)

    def test_mismatched_ohlc(self):
        with self.assertRaises(ValueError):
            build_snapshot(self.closes, self.highs, self.lows[:-1],
                           compute_atr=True)

    def test_bad_periods(self):
        with self.assertRaises(ValueError):
            build_snapshot(self.closes, fast_period=0, slow_period=10)
        with self.assertRaises(ValueError):
            build_snapshot(self.closes, fast_period=3, slow_period=10,
                           rsi_period=-1)

    def test_insufficient_closes(self):
        with self.assertRaises(ValueError):
            build_snapshot([1, 2, 3], fast_period=3, slow_period=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
