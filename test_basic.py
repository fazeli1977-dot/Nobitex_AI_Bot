from app.config import get_settings


def test_safe_mode_is_enabled():
    settings = get_settings()

    assert settings.paper_trading is True
    assert settings.live_trading is False
    assert settings.initial_balance == 3_000_000
