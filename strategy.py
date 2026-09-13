"""strategy.py — Deterministic signal engine for Nobitex_AI_Bot.

Python 3.12, standard library only (plus app.indicators).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

from app.indicators import IndicatorSnapshot, build_snapshot

__all__ = ["StrategyConfig", "StrategySignal", "Strategy"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class StrategyConfig:
    fast_period: int = 9
    slow_period: int = 21
    rsi_period: int = 14
    atr_period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    atr_stop_multiplier: float = 1.5
    atr_target_multiplier: float = 2.0
    allow_short: bool = True

    def __post_init__(self) -> None:
        for name in ("fast_period", "slow_period", "rsi_period", "atr_period"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be smaller than slow_period")
        for name in ("oversold", "overbought"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
        if not (0.0 <= float(self.oversold) <= 100.0 and 0.0 <= float(self.overbought) <= 100.0):
            raise ValueError("oversold/overbought must be within [0, 100]")
        if float(self.oversold) >= float(self.overbought):
            raise ValueError("oversold must be smaller than overbought")
        for name in ("atr_stop_multiplier", "atr_target_multiplier"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or v <= 0:
                raise ValueError(f"{name} must be a positive finite number, got {v!r}")
        if not isinstance(self.allow_short, bool):
            raise ValueError("allow_short must be a bool")


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
@dataclass
class StrategySignal:
    action: Literal["buy", "sell", "hold"]
    symbol: str
    price: float
    reason: str
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    snapshot: Optional[IndicatorSnapshot] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.action not in ("buy", "sell", "hold"):
            raise ValueError(f"invalid action: {self.action!r}")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(self.price, (int, float)) or isinstance(self.price, bool) \
                or not math.isfinite(float(self.price)) or self.price <= 0:
            raise ValueError("price must be a positive finite number")
        c = float(self.confidence)
        if not math.isfinite(c) or not (0.0 <= c <= 1.0):
            raise ValueError("confidence must be within [0, 1]")

    @property
    def is_entry(self) -> bool:
        return self.action in ("buy", "sell") and self.stop_loss is not None and self.take_profit is not None


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
class Strategy:
    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config if config is not None else StrategyConfig()

    # -- helpers ----------------------------------------------------------- #
    def _levels(self, price: float, atr: float | None, side: str):
        if atr is None or not math.isfinite(atr) or atr <= 0:
            return None, None
        stop_mult = float(self.config.atr_stop_multiplier)
        target_mult = float(self.config.atr_target_multiplier)
        if side == "long":
            stop = price - stop_mult * atr
            target = price + target_mult * atr
        else:
            stop = price + stop_mult * atr
            target = price - target_mult * atr
        return (stop if stop > 0 else None), (target if target > 0 else None)

    @staticmethod
    def _confidence(snapshot: IndicatorSnapshot, direction_ok: bool) -> float:
        base = 0.5
        if direction_ok:
            base += 0.2
        # RSI distance from the extremes adds conviction
        rsi = snapshot.rsi
        if math.isfinite(rsi):
            base += min(abs(rsi - 50.0) / 100.0, 0.2)
        else:
            base += 0.1
        return max(0.0, min(1.0, base))

    # -- public ------------------------------------------------------------ #
    def evaluate(
        self,
        symbol: str,
        closes: Sequence[float],
        highs: Sequence[float] | None = None,
        lows: Sequence[float] | None = None,
        position_side: Literal["long", "short"] | None = None,
    ) -> StrategySignal:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if position_side not in (None, "long", "short"):
            raise ValueError(f"invalid position_side: {position_side!r}")

        if closes is None:
            raise ValueError("closes must not be None")
        try:
            n = len(closes)
        except TypeError as exc:
            raise ValueError("closes must be a sequence of numbers") from exc
        need = max(self.config.slow_period, self.config.rsi_period) + 1
        if n < need:
            raise ValueError(f"insufficient data: need at least {need} closes, got {n}")
        if (highs is None) != (lows is None):
            raise ValueError("highs and lows must be provided together")
        for name, seq in (("highs", highs), ("lows", lows)):
            if seq is not None and len(seq) != n:
                raise ValueError(f"{name} must have the same length as closes")

        cfg = self.config
        snapshot = build_snapshot(
            closes, highs, lows,
            fast_period=cfg.fast_period, slow_period=cfg.slow_period,
            rsi_period=cfg.rsi_period, atr_period=cfg.atr_period,
        )
        price = float(closes[-1])
        rsi = snapshot.rsi
        in_band = math.isfinite(rsi) and cfg.oversold <= rsi <= cfg.overbought
        if not math.isfinite(rsi):
            in_band = True  # RSI undefined (flat data): don't block on it

        trend = snapshot.trend
        # Exit conditions first
        if position_side == "long":
            if trend == "bearish":
                return StrategySignal("sell", symbol, price, "exit long: bearish trend",
                                      self._confidence(snapshot, True), snapshot=snapshot)
            if in_band and math.isfinite(rsi) and rsi >= cfg.overbought:
                return StrategySignal("sell", symbol, price, "exit long: overbought RSI",
                                      self._confidence(snapshot, True), snapshot=snapshot)
            return StrategySignal("hold", symbol, price, "hold long position",
                                  self._confidence(snapshot, trend == "bullish"), snapshot=snapshot)
        if position_side == "short":
            if trend == "bullish":
                return StrategySignal("buy", symbol, price, "exit short: bullish trend",
                                      self._confidence(snapshot, True), snapshot=snapshot)
            if math.isfinite(rsi) and rsi <= cfg.oversold:
                return StrategySignal("buy", symbol, price, "exit short: oversold RSI",
                                      self._confidence(snapshot, True), snapshot=snapshot)
            return StrategySignal("hold", symbol, price, "hold short position",
                                  self._confidence(snapshot, trend == "bearish"), snapshot=snapshot)

        # Entries
        if trend == "bullish" and snapshot.ema_fast > snapshot.ema_slow \
                and snapshot.sma_fast >= snapshot.sma_slow and in_band:
            stop, target = self._levels(price, snapshot.atr, "long")
            return StrategySignal("buy", symbol, price, "long entry: bullish alignment",
                                  self._confidence(snapshot, True), stop, target, snapshot=snapshot)
        if cfg.allow_short and trend == "bearish" and snapshot.ema_fast < snapshot.ema_slow \
                and snapshot.sma_fast <= snapshot.sma_slow and in_band:
            stop, target = self._levels(price, snapshot.atr, "short")
            return StrategySignal("sell", symbol, price, "short entry: bearish alignment",
                                  self._confidence(snapshot, True), stop, target, snapshot=snapshot)
        return StrategySignal("hold", symbol, price, "no clear setup",
                              self._confidence(snapshot, trend != "neutral"), snapshot=snapshot)
