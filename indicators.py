"""indicators.py — Pure standard-library technical indicators for Nobitex_AI_Bot.

Python 3.12 compatible. No pandas, no third-party dependencies.

Public API:
    sma(values, period) -> list[float]
    ema(values, period) -> list[float]
    rsi(values, period) -> list[float]         (Wilder's smoothing)
    atr(highs, lows, closes, period) -> list[float]  (Wilder's smoothing)
    IndicatorSnapshot (dataclass)
    build_snapshot(closes, highs=None, lows=None, fast_period=9,
                   slow_period=21, rsi_period=14, atr_period=14,
                   compute_atr=True) -> IndicatorSnapshot
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "sma",
    "ema",
    "rsi",
    "atr",
    "IndicatorSnapshot",
    "build_snapshot",
]


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _validate_series(values, name: str) -> list[float]:
    """Validate that `values` is a non-empty sequence of finite real numbers."""
    if values is None:
        raise ValueError(f"{name} must not be None")
    try:
        iterator = list(values)
    except TypeError as exc:  # not iterable / scalar
        raise ValueError(f"{name} must be a sequence of numbers") from exc
    if len(iterator) == 0:
        raise ValueError(f"{name} must not be empty")
    out: list[float] = []
    for i, v in enumerate(iterator):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name}[{i}] must be a real number, got {v!r}")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"{name}[{i}] must be finite, got {f}")
        out.append(f)
    return out


def _validate_period(period, name: str = "period") -> int:
    if isinstance(period, bool) or not isinstance(period, int):
        raise ValueError(f"{name} must be a positive int, got {period!r}")
    if period <= 0:
        raise ValueError(f"{name} must be a positive int, got {period!r}")
    return period


# --------------------------------------------------------------------------- #
# SMA
# --------------------------------------------------------------------------- #
def sma(values, period: int) -> list[float]:
    """Simple moving average.

    Returns a list with (period - 1) leading NaN values, then valid values.
    Raises ValueError for insufficient data (len(values) < period).
    """
    _validate_period(period)
    vals = _validate_series(values, "values")
    if len(vals) < period:
        raise ValueError(
            f"insufficient data: need at least {period} points, got {len(vals)}"
        )
    out: list[float] = [float("nan")] * (period - 1)
    window_sum = sum(vals[:period])
    out.append(window_sum / period)
    for i in range(period, len(vals)):
        window_sum += vals[i] - vals[i - period]
        out.append(window_sum / period)
    return out


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
def ema(values, period: int) -> list[float]:
    """Exponential moving average, seeded with the SMA of the first `period` values.

    Returns a list with (period - 1) leading NaN values, then valid values.
    Raises ValueError for insufficient data (len(values) < period).
    """
    _validate_period(period)
    vals = _validate_series(values, "values")
    if len(vals) < period:
        raise ValueError(
            f"insufficient data: need at least {period} points, got {len(vals)}"
        )
    alpha = 2.0 / (period + 1.0)
    out: list[float] = [float("nan")] * (period - 1)
    prev = sum(vals[:period]) / period  # SMA seed
    out.append(prev)
    for i in range(period, len(vals)):
        prev = alpha * vals[i] + (1.0 - alpha) * prev
        out.append(prev)
    return out


# --------------------------------------------------------------------------- #
# RSI (Wilder)
# --------------------------------------------------------------------------- #
def rsi(values, period: int) -> list[float]:
    """Relative Strength Index using Wilder's smoothing.

    The first valid output is at index `period` (it needs `period` changes,
    i.e. period + 1 prices). Leading values are NaN.
    Fully monotonic rising series -> 100. Constant series ->  Constant series ->  ValueError for insufficient data (len(values) < period + 1).
    """
    _validate_period(period)
    vals = _validate_series(values, "values")
    if len(vals) < period + 1:
        raise ValueError(
            f"insufficient data: RSI needs at least {period + 1} points, "
            f"got {len(vals)}"
        )
    out: list[float] = [float("nan")] * period  # indices 0 .. period-1
    # Initial average gain / loss over the first `period` changes
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = vals[i] - vals[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    out.append(_rsi_value(avg_gain, avg_loss))
    # Wilder smoothing for the rest
    for i in range(period + 1, len(vals)):
        change = vals[i] - vals[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out.append(_rsi_value(avg_gain, avg_loss))
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        if avg_gain == 0.0:
            return 50.0  # constant series
        return 100.0  # strictly rising series
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# --------------------------------------------------------------------------- #
# ATR (Wilder)
# --------------------------------------------------------------------------- #
def atr(highs, lows, closes, period: int) -> list[float]:
    """Average True Range using Wilder's smoothing.

    True Range for the first candle is simply high - low (no previous close).
    The first valid ATR is at index `period - 1` (SMA of the first `period`
    TRs); leading values are NaN.
    Raises ValueError for insufficient data (len < period) and for
    mismatched sizes or low > high.
    """
    _validate_period(period)
    h = _validate_series(highs, "highs")
    l = _validate_series(lows, "lows")
    c = _validate_series(closes, "closes")
    if not (len(h) == len(l) == len(c)):
        raise ValueError(
            f"highs/lows/closes must be the same size "
            f"({len(h)}, {len(l)}, {len(c)})"
        )
    for i in range(len(h)):
        if l[i] > h[i]:
            raise ValueError(f"lows[{i}] must be <= highs[{i}] ({l[i]} > {h[i]})")
    if len(h) < period:
        raise ValueError(
            f"insufficient data: need at least {period} candles, got {len(h)}"
        )
    n = len(h)
    trs: list[float] = [0.0] * n
    trs[0] = h[0] - l[0]
    for i in range(1, n):
        trs[i] = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )
    out: list[float] = [float("nan")] * (period - 1)
    prev = sum(trs[:period]) / period  # SMA of first `period` TRs
    out.append(prev)
    for i in range(period, n):
        prev = (prev * (period - 1) + trs[i]) / period
        out.append(prev)
    return out


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IndicatorSnapshot:
    """A point-in-time snapshot of computed indicators (last values)."""

    sma_fast: float
    sma_slow: float
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float | None
    volatility_pct: float
    trend: str  # "bullish" | "bearish" | "neutral"

    def to_dict(self) -> dict:
        return {
            "sma_fast": self.sma_fast,
            "sma_slow": self.sma_slow,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "rsi": self.rsi,
            "atr": self.atr,
            "volatility_pct": self.volatility_pct,
            "trend": self.trend,
        }


def _last_valid(series: list[float], name: str) -> float:
    for v in reversed(series):
        if not math.isnan(v):
            return v
    raise ValueError(f"{name} produced no valid values")


def build_snapshot(
    closes,
    highs=None,
    lows=None,
    fast_period: int = 9,
    slow_period: int = 21,
    rsi_period: int = 14,
    atr_period: int = 14,
    compute_atr: bool = True,
) -> IndicatorSnapshot:
    """Build an IndicatorSnapshot from OHLC data (last available values).

    If compute_atr is True, highs and lows must be provided and must match
    the size of closes. Trend is "bullish" if ema_fast > ema_slow,
    "bearish" if ema_fast < ema_slow, otherwise "neutral".
    volatility_pct = (ema_slow - sma_slow) / sma_slow * 100 ... replaced by
    ATR-based definition when ATR is available:
        volatility_pct = (last_atr / last_close) * 100
    otherwise falls back to std-dev-of-returns based measure:
        volatility_pct = (stdev of last `slow_period` simple returns) * 100
    """
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be smaller than "
            f"slow_period ({slow_period})"
        )
    _validate_period(rsi_period, "rsi_period")
    _validate_period(atr_period, "atr_period")
    c = _validate_series(closes, "closes")

    sma_fast_s = sma(c, fast_period)
    sma_slow_s = sma(c, slow_period)
    ema_fast_s = ema(c, fast_period)
    ema_slow_s = ema(c, slow_period)
    rsi_s = rsi(c, rsi_period)

    sma_fast = _last_valid(sma_fast_s, "sma_fast")
    sma_slow = _last_valid(sma_slow_s, "sma_slow")
    ema_fast = _last_valid(ema_fast_s, "ema_fast")
    ema_slow = _last_valid(ema_slow_s, "ema_slow")
    rsi_last = _last_valid(rsi_s, "rsi")

    atr_value: float | None = None
    if compute_atr:
        if highs is None or lows is None:
            raise ValueError("highs and lows are required when compute_atr=True")
        h = _validate_series(highs, "highs")
        l = _validate_series(lows, "lows")
        atr_s = atr(h, l, c, atr_period)
        atr_value = _last_valid(atr_s, "atr")
        volatility_pct = (atr_value / c[-1]) * 100.0 if c[-1] != 0 else 0.0
    else:
        # stdev of the last slow_period simple returns
        window = min(slow_period, len(c) - 1)
        returns = [
            (c[i] / c[i - 1] - 1.0) for i in range(len(c) - window, len(c)) if c[i - 1] != 0
        ]
        if not returns:
            volatility_pct = 0.0
        else:
            mean = sum(returns) / len(returns)
            var = sum((r - mean) ** 2 for r in returns) / len(returns)
            volatility_pct = math.sqrt(var) * 100.0

    if ema_fast > ema_slow:
        trend = "bullish"
    elif ema_fast < ema_slow:
        trend = "bearish"
    else:
        trend = "neutral"

    return IndicatorSnapshot(
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi=rsi_last,
        atr=atr_value,
        volatility_pct=volatility_pct,
        trend=trend,
    )
