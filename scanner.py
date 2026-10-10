"""
Signal scanner — scan a list of tickers and print consensus signals.

Uses Webull OpenAPI for market data.
Docs: https://developer.webull.com.my/apis/llms.txt

Usage:
    python scanner.py AAPL MSFT NVDA SPY QQQ
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv()

from webull.core.http.endpoint.api_client import ApiClient
from webull.data.data_client import DataClient
from webull.common.category import Category
import pandas as pd

from strategies import (
    ema_crossover_signal, rsi_macd_signal,
    bollinger_mean_reversion_signal, aggregate_signals,
)


def _get_data_client():
    app_key = os.getenv("WEBULL_APP_KEY", "").strip()
    app_secret = os.getenv("WEBULL_APP_SECRET", "").strip()
    region_id = os.getenv("WEBULL_REGION_ID", "my").strip()
    host = os.getenv("WEBULL_API_HOST", "").strip() or None

    api_client = ApiClient(app_key, app_secret, region_id)
    if host:
        api_client.add_endpoint(region_id, host)
    return DataClient(api_client)


def _get_bars(data_client, ticker: str, count: int = 400):
    """Fetch daily bars from Webull and return as a pandas DataFrame."""
    try:
        bars = data_client.get_bars(
            category=Category.US_STOCK.value,
            symbol=ticker,
            timespan="D",
            count=count,
        )
        if not bars:
            return None
        df = pd.DataFrame(bars)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        return df if len(df) >= 30 else None
    except Exception:
        return None


def scan(tickers: list[str]) -> None:
    data_client = _get_data_client()
    print(f"\nScanning {len(tickers)} tickers...\n")
    print(f"{'Ticker':<8} {'Signal':<6} {'Strength':<10} {'Reason'}")
    print("-" * 70)

    for ticker in tickers:
        df = _get_bars(data_client, ticker)
        if df is None:
            continue

        signals = [
            ema_crossover_signal(df, ticker),
            rsi_macd_signal(df, ticker),
            bollinger_mean_reversion_signal(df, ticker),
        ]
        consensus = aggregate_signals(signals)

        if consensus.action != "HOLD":
            print(f"{ticker:<8} {consensus.action:<6} {consensus.strength:<10.2f} {consensus.reason}")

    print()


if __name__ == "__main__":
    tickers = [t.upper() for t in sys.argv[1:]] or [
        "AAPL", "MSFT", "NVDA", "SPY", "QQQ", "AMZN", "GOOGL", "META", "AMD", "AVGO"
    ]
    scan(tickers)
