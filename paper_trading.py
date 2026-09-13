
"""paper_trading.py — Paper-trading engine for Nobitex_AI_Bot.

Python 3.12, standard library only. No network, no real keys.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Literal, Optional

__all__ = [
    "PaperTradingConfig", "Position", "TradeExecution",
    "PortfolioSnapshot", "PaperTrader",
]

Side = Literal["long", "short"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _check_positive(v, name: str, allow_zero: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) \
            or not math.isfinite(float(v)) or v < 0 or (v == 0 and not allow_zero):
        raise ValueError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} finite number, got {v!r}")
    return float(v)


# --------------------------------------------------------------------------- #
# Config & dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class PaperTradingConfig:
    initial_cash: float = 3_000_000
    fee_rate: float = 0.001
    min_order_value: float = 300_000
    position_size_pct: float = 0.10
    max_positions: int = 5
    leverage: float = 1.0
    trailing_stop_pct: float = 0.01
    allow_short: bool = True

    def __post_init__(self) -> None:
        _check_positive(self.initial_cash, "initial_cash")
        f = _check_positive(self.fee_rate, "fee_rate")
        if f >= 1.0:
            raise ValueError("fee_rate must be < 1.0")
        _check_positive(self.min_order_value, "min_order_value")
        p = _check_positive(self.position_size_pct, "position_size_pct")
        if p > 1.0:
            raise ValueError("position_size_pct must be within (0, 1]")
        if isinstance(self.max_positions, bool) or not isinstance(self.max_positions, int) \
                or self.max_positions <= 0:
            raise ValueError("max_positions must be a positive int")
        _check_positive(self.leverage, "leverage")
        t = _check_positive(self.trailing_stop_pct, "trailing_stop_pct", allow_zero=True)
        if t >= 1.0:
            raise ValueError("trailing_stop_pct must be < 1.0")
        if not isinstance(self.allow_short, bool):
            raise ValueError("allow_short must be a bool")


@dataclass
class Position:
    symbol: str
    side: Side
    amount: float
    entry_price: float
    notional: float
    margin: float
    entry_fee: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    high_water: float = 0.0
    low_water: float = 0.0
    opened_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if self.side not in ("long", "short"):
            raise ValueError(f"invalid side: {self.side!r}")
        for name, v in (("amount", self.amount), ("entry_price", self.entry_price),
                        ("notional", self.notional), ("margin", self.margin),
                        ("entry_fee", self.entry_fee)):
            _check_positive(v, name, allow_zero=(name == "entry_fee"))
        for name in ("stop_loss", "take_profit", "trailing_stop_pct"):
            v = getattr(self, name)
            if v is not None:
                _check_positive(v, name, allow_zero=(name == "trailing_stop_pct"))


@dataclass
class TradeExecution:
    action: Literal["open", "close"]
    symbol: str
    side: Side
    price: float
    amount: float
    notional: float
    fee: float
    realized_pnl: float
    reason: str
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass
class PortfolioSnapshot:
    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    open_positions: int
    daily_pnl: float


# --------------------------------------------------------------------------- #
# Trader
# --------------------------------------------------------------------------- #
class PaperTrader:
    def __init__(self, config: PaperTradingConfig | None = None) -> None:
        self.config = config if config is not None else PaperTradingConfig()
        self.reset_day()
        self.positions: dict[str, Position] = {}
        self.trades: list[TradeExecution] = []

    # -- internals --------------------------------------------------------- #
    def _new_price(self, price, name: str = "price") -> float:
        return _check_positive(price, name)

    def _get_position(self, symbol: str) -> Position:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non-empty string")
        if symbol not in self.positions:
            raise KeyError(f"no open position for {symbol!r}")
        return self.positions[symbol]

    def _position_upnl(self, pos: Position, mark: float) -> float:
        if pos.side == "long":
            return (mark - pos.entry_price) * pos.amount
        return (pos.entry_price - mark) * pos.amount

    # -- open / close ------------------------------------------------------- #
    def open_position(self, symbol, side, price, allocation=None,
                      stop_loss=None, take_profit=None, timestamp=None) -> TradeExecution:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if side not in ("long", "short"):
            raise ValueError(f"invalid side: {side!r}")
        price = self._new_price(price)
        if side == "short" and not self.config.allow_short:
            raise ValueError("short positions are not allowed")
        if symbol in self.positions:
            raise ValueError(f"duplicate position for {symbol!r}")
        if len(self.positions) >= self.config.max_positions:
            raise ValueError(f"max_positions ({self.config.max_positions}) reached")

        cfg = self.config
        if allocation is None:
            allocation = max(cfg.min_order_value, cfg.initial_cash * cfg.position_size_pct)
        allocation = _check_positive(allocation, "allocation")
        allocation = min(allocation, self.cash)
        notional = allocation * cfg.leverage
        if notional < cfg.min_order_value:
            raise ValueError(
                f"order value {notional:.2f} below min_order_value {cfg.min_order_value:.2f}"
            )
        for name, v in (("stop_loss", stop_loss), ("take_profit", take_profit)):
            if v is not None:
                _check_positive(v, name)

        fee = notional * cfg.fee_rate
        amount = notional / price
        margin = notional / cfg.leverage
        if margin + fee > self.cash + 1e-9:
            raise ValueError("insufficient cash for margin and fee")
        ts = timestamp or _utc_now()
        pos = Position(
            symbol=symbol, side=side, amount=amount, entry_price=price,
            notional=notional, margin=margin, entry_fee=fee,
            stop_loss=stop_loss, take_profit=take_profit,
            trailing_stop_pct=cfg.trailing_stop_pct if cfg.trailing_stop_pct > 0 else None,
            high_water=price, low_water=price, opened_at=ts,
        )
        self.cash -= fee  # entry fee charged from cash
        self.fees_paid += fee
        self.positions[symbol] = pos
        exec_ = TradeExecution("open", symbol, side, price, amount, notional,
                               fee, 0.0, "opened", ts)
        self.trades.append(exec_)
        return exec_

    def close_position(self, symbol, price, reason="manual", timestamp=None) -> TradeExecution:
        pos = self._get_position(symbol)
        price = self._new_price(price)
        ts = timestamp or _utc_now()
        gross = self._position_upnl(pos, price)
        exit_fee = pos.notional * self.config.fee_rate
        realized = gross - pos.entry_fee - exit_fee
        self.cash += pos.margin + gross - exit_fee
        self.realized_pnl += realized
        self.fees_paid += exit_fee
        del self.positions[symbol]
        exec_ = TradeExecution("close", symbol, pos.side, price, pos.amount,
                               pos.notional, exit_fee, realized, reason, ts)
        self.trades.append(exec_)
        self.daily_pnl += realized
        return exec_

    # -- price updates ------------------------------------------------------ #
    def update_price(self, symbol, price, timestamp=None) -> TradeExecution | None:
        pos = self._get_position(symbol)
        price = self._new_price(price)
        if pos.side == "long":
            pos.high_water = max(pos.high_water, price)
            if pos.trailing_stop_pct:
                trail = pos.high_water * (1 - pos.trailing_stop_pct)
                pos.stop_loss = max(pos.stop_loss, trail) if pos.stop_loss else trail
            if pos.stop_loss is not None and price <= pos.stop_loss:
                return self.close_position(symbol, pos.stop_loss, "stop_loss", timestamp)
            if pos.take_profit is not None and price >= pos.take_profit:
                return self.close_position(symbol, pos.take_profit, "take_profit", timestamp)
        else:
            pos.low_water = min(pos.low_water, price)
            if pos.trailing_stop_pct:
                trail = pos.low_water * (1 + pos.trailing_stop_pct)
                pos.stop_loss = min(pos.stop_loss, trail) if pos.stop_loss else trail
            if pos.stop_loss is not None and price >= pos.stop_loss:
                return self.close_position(symbol, pos.stop_loss, "stop_loss", timestamp)
            if pos.take_profit is not None and price <= pos.take_profit:
                return self.close_position(symbol, pos.take_profit, "take_profit", timestamp)
        return None

    def update_bar(self, symbol, high, low, close, timestamp=None) -> TradeExecution | None:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non-empty string")
        high = self._new_price(high, "high")
        low = self._new_price(low, "low")
        close = self._new_price(close, "close")
        if low > high:
            raise ValueError("low must not exceed high")
        pos = self._get_position(symbol)
        # Conservative: evaluate the adverse extreme first.
        if pos.side == "long":
            stop = pos.stop_loss
            if stop is not None and low <= stop:
                return self.close_position(symbol, stop, "stop_loss", timestamp)
            if pos.take_profit is not None and high >= pos.take_profit:
                return self.close_position(symbol, pos.take_profit, "take_profit", timestamp)
        else:
            stop = pos.stop_loss
            if stop is not None and high >= stop:
                return self.close_position(symbol, stop, "stop_loss", timestamp)
            if pos.take_profit is not None and low <= pos.take_profit:
                return self.close_position(symbol, pos.take_profit, "take_profit", timestamp)
        return self.update_price(symbol, close, timestamp)

    # -- signals ------------------------------------------------------------ #
    def process_signal(self, signal, timestamp=None) -> TradeExecution | None:
        if signal is None or not hasattr(signal, "action"):
            raise ValueError("signal must be a StrategySignal-like object")
        action = signal.action
        symbol = signal.symbol
        price = self._new_price(getattr(signal, "price", None))
        ts = timestamp or _utc_now()
        has = symbol in self.positions
        side = self.positions[symbol].side if has else None
        if action == "buy":
            if has and side == "short":
                return self.close_position(symbol, price, "signal exit short", ts)
            if not has:
                sl = getattr(signal, "stop_loss", None)
                tp = getattr(signal, "take_profit", None)
                return self.open_position(symbol, "long", price,
                                          stop_loss=sl, take_profit=tp, timestamp=ts)
            return None
        if action == "sell":
            if has and side == "long":
                return self.close_position(symbol, price, "signal exit long", ts)
            if not has and self.config.allow_short:
                sl = getattr(signal, "stop_loss", None)
                tp = getattr(signal, "take_profit", None)
                return self.open_position(symbol, "short", price,
                                          stop_loss=sl, take_profit=tp, timestamp=ts)
            return None
        return None  # hold

    # -- dashboard ---------------------------------------------------------- #
    def snapshot(self, mark_prices=None) -> PortfolioSnapshot:
        if mark_prices is None:
            mark_prices = {}
        if not isinstance(mark_prices, dict):
            raise ValueError("mark_prices must be a dict")
        unrealized = 0.0
        for sym, pos in self.positions.items():
            mark = mark_prices.get(sym, pos.entry_price)
            mark = self._new_price(mark, f"mark[{sym}]")
            unrealized += self._position_upnl(pos, mark)
        equity = self.cash + sum(
            p.margin for p in self.positions.values()
        ) + unrealized
        return PortfolioSnapshot(
            cash=self.cash, equity=equity, realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized, fees_paid=self.fees_paid,
            open_positions=len(self.positions), daily_pnl=self.daily_pnl,
        )

    def reset_day(self, mark_prices=None) -> PortfolioSnapshot:
        snap = self.snapshot(mark_prices) if getattr(self, "positions", None) is not None else None
        daily = snap.daily_pnl if snap else 0.0
        self.daily_pnl = 0.0
        return PortfolioSnapshot(
            cash=getattr(self, "cash", self.config.initial_cash),
            equity=getattr(self, "cash", self.config.initial_cash),
            realized_pnl=getattr(self, "realized_pnl", 0.0),
            unrealized_pnl=0.0,
            fees_paid=getattr(self, "fees_paid", 0.0),
            open_positions=len(getattr(self, "positions", {})),
            daily_pnl=daily,
        )
