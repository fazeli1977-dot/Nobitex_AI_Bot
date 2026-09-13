from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "paper"
    base_url: str = "https://apiv2.nobitex.ir"

    paper_trading: bool = True
    live_trading: bool = False

    symbol: str = "BTCIRT"
    timeframe: int = 15
    initial_balance: float = 3_000_000

    max_open_positions: int = 5
    risk_per_trade: float = 0.01
    max_daily_loss: float = 0.03

    fee_rate: float = 0.0025
    slippage_rate: float = 0.001

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
