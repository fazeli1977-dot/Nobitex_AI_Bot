import httpx

from app.market_data import MarketDataClient


def test_market_data_endpoints_use_correct_paths():
    requested_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)

        return httpx.Response(
            200,
            json={
                "status": "ok",
                "bids": [],
                "asks": [],
            },
        )

    transport = httpx.MockTransport(handler)

    with MarketDataClient(transport=transport) as client:
        client.get_orderbook("BTC-USDT")
        client.get_depth("BTC-USDT")
        client.get_trades("BTC-USDT")

    assert requested_paths == [
        "/v3/orderbook/btc-usdt",
        "/v2/depth/btc-usdt",
        "/v2/trades/btc-usdt",
    ]


def test_orderbook_all_is_allowed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    with MarketDataClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        result = client.get_orderbook("all")

    assert result["status"] == "ok"


def test_trades_all_is_rejected():
    with MarketDataClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"status": "ok"})
        )
    ) as client:
        try:
            client.get_trades("all")
            assert False, "نماد all برای trades نباید پذیرفته شود."
        except ValueError:
            pass
