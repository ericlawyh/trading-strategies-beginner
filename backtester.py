"""
Basic backtester — iterate daily bars, emit signals, compute hit rate and P&L.

Uses Webull OpenAPI for historical data.
Assumes market-on-close fills, no position sizing (1 share per trade).

Usage:
    python backtester.py AAPL
    python backtester.py AAPL MSFT SPY
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
    bollinger_mean_reversion_signal, aggregate_signals, Signal,
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


def _get_bars(data_client, ticker: str, count: int = 500):
    """Fetch daily bars from Webull."""
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
        return df if len(df) >= 60 else None
    except Exception:
        return None


def backtest(ticker: str, data_client) -> dict:
    df = _get_bars(data_client, ticker)
    if df is None:
        return {"ticker": ticker, "error": "Insufficient data"}

    trades = []
    position = None  # None or {"entry_price": float}

    for i in range(30, len(df)):
        window = df.iloc[:i+1]
        signals = [
            ema_crossover_signal(window, ticker),
            rsi_macd_signal(window, ticker),
            bollinger_mean_reversion_signal(window, ticker),
        ]
        consensus = aggregate_signals(signals)
        price = float(df["Close"].iloc[i])

        if consensus.action == "BUY" and position is None:
            position = {"entry_price": price}

        elif consensus.action == "SELL" and position is not None:
            pnl = price - position["entry_price"]
            pnl_pct = pnl / position["entry_price"] * 100
            trades.append({"entry": position["entry_price"], "exit": price,
                           "pnl": pnl, "pnl_pct": pnl_pct})
            position = None

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)

    return {
        "ticker": ticker,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": f"{len(wins)/len(trades)*100:.1f}%" if trades else "N/A",
        "total_pnl": f"${total_pnl:+.2f}",
        "avg_win": f"+{sum(t['pnl_pct'] for t in wins)/len(wins):.1f}%" if wins else "N/A",
        "avg_loss": f"{sum(t['pnl_pct'] for t in losses)/len(losses):.1f}%" if losses else "N/A",
    }


if __name__ == "__main__":
    tickers = [t.upper() for t in sys.argv[1:]] or ["AAPL", "MSFT", "SPY"]
    data_client = _get_data_client()
    print(f"\nBacktesting {len(tickers)} tickers (1 share per trade)...\n")

    for ticker in tickers:
        result = backtest(ticker, data_client)
        if "error" in result:
            print(f"  {ticker}: {result['error']}")
        else:
            print(f"  {ticker}: {result['total_trades']} trades, "
                  f"win rate {result['win_rate']}, "
                  f"P&L {result['total_pnl']}, "
                  f"avg win {result['avg_win']}, avg loss {result['avg_loss']}")
    print()
