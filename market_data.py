from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from time import monotonic, sleep
from typing import Any

import httpx


class MarketDataError(RuntimeError):
    """خطا هنگام دریافت داده بازار از نوبیتکس."""


@dataclass
class _RateLimiter:
    limit: int
    window_seconds: float = 60.0
    calls: deque[float] = field(default_factory=deque)

    def wait(self) -> None:
        """رعایت محدودیت درخواست در بازه زمانی متحرک."""

        while True:
            now = monotonic()

            while (
                self.calls
                and now - self.calls[0] >= self.window_seconds
            ):
                self.calls.popleft()

            if len(self.calls) < self.limit:
                self.calls.append(now)
                return

            wait_time = self.window_seconds - (
                now - self.calls[0]
            )
            sleep(max(wait_time, 0.05))


class MarketDataClient:
    """کلاینت داده‌های عمومی بازار نوبیتکس."""

    _SYMBOL_PATTERN = re.compile(
        r"^[a-z0-9]+-[a-z0-9]+$"
    )

    def __init__(
        self,
        base_url: str = "https://apiv2.nobitex.ir",
        timeout: float = 10.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max(0, max_retries)

        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers={
                "Accept": "application/json",
                "User-Agent": "NobitexAI-Bot/0.1",
            },
        )

        self._limiters = {
            "orderbook": _RateLimiter(300),
            "depth": _RateLimiter(300),
            "trades": _RateLimiter(60),
        }

    def __enter__(self) -> "MarketDataClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    @staticmethod
    def _validate_symbol(
        symbol: str,
        allow_all: bool = False,
    ) -> str:
        symbol = symbol.strip().lower()

        if allow_all and symbol == "all":
            return symbol

        if not MarketDataClient._SYMBOL_PATTERN.fullmatch(symbol):
            raise ValueError(
                "نماد باید مانند btc-usdt باشد؛ "
                "مقدار all فقط برای orderbook مجاز است."
            )

        return symbol

    @staticmethod
    def _retry_delay(
        response: httpx.Response,
        attempt: int,
    ) -> float:
        retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                return min(
                    max(float(retry_after), 0.25),
                    10.0,
                )
            except ValueError:
                pass

        return min(2**attempt, 5.0)

    def _get(self, path: str, limiter_name: str) -> Any:
        limiter = self._limiters[limiter_name]

        for attempt in range(self.max_retries + 1):
            limiter.wait()

            try:
                response = self._http.get(path)

                if (
                    response.status_code == 429
                    or response.status_code >= 500
                ):
                    if attempt < self.max_retries:
                        delay = self._retry_delay(
                            response,
                            attempt,
                        )
                        response.close()
                        sleep(delay)
                        continue

                response.raise_for_status()

                try:
                    data = response.json()
                except ValueError as exc:
                    raise MarketDataError(
                        "پاسخ نوبیتکس JSON معتبر نیست."
                    ) from exc

                if (
                    isinstance(data, dict)
                    and str(data.get("status", "")).lower()
                    == "failed"
                ):
                    raise MarketDataError(
                        "خطای API نوبیتکس: "
                        f"{data.get('message', data)}"
                    )

                return data

            except httpx.TimeoutException as exc:
                if attempt >= self.max_retries:
                    raise MarketDataError(
                        "مهلت دریافت داده بازار تمام شد."
                    ) from exc

                sleep(min(2**attempt, 5.0))

            except httpx.NetworkError as exc:
                if attempt >= self.max_retries:
                    raise MarketDataError(
                        "اتصال به API نوبیتکس برقرار نشد."
                    ) from exc

                sleep(min(2**attempt, 5.0))

            except httpx.HTTPStatusError as exc:
                raise MarketDataError(
                    "پاسخ ناموفق API نوبیتکس: "
                    f"HTTP {exc.response.status_code}"
                ) from exc

        raise MarketDataError(
            "دریافت داده بازار ناموفق بود."
        )

    def get_orderbook(self, symbol: str = "all") -> Any:
        """دریافت دفتر سفارشات یک نماد یا تمام بازارها."""

        symbol = self._validate_symbol(
            symbol,
            allow_all=True,
        )

        return self._get(
            f"/v3/orderbook/{symbol}",
            limiter_name="orderbook",
        )

    def get_depth(self, symbol: str) -> Any:
        """دریافت عمق بازار."""

        symbol = self._validate_symbol(symbol)

        return self._get(
            f"/v2/depth/{symbol}",
            limiter_name="depth",
        )

    def get_trades(self, symbol: str) -> Any:
        """دریافت حداکثر ۲۰ معامله اخیر."""

        symbol = self._validate_symbol(symbol)

        return self._get(
            f"/v2/trades/{symbol}",
            limiter_name="trades",
        )
