"""
Signal Fundamentals — 3 Core Technical Analysis Strategies

Strategies:
  1. EMA Crossover (9/21)          — trend following via moving average crosses
  2. RSI + MACD Combo              — momentum confirmation (oversold/overbought + MACD cross)
  3. Bollinger Band Mean Reversion — fade price extremes in sideways markets

These three strategies form the foundation of technical analysis. They are
well-understood, widely backtested, and behave predictably enough that a new
trader can learn from their failures as well as their successes.

Usage:
    from strategies import ema_crossover_signal, rsi_macd_signal, bollinger_mean_reversion_signal
    from strategies import aggregate_signals, Signal

    signals = [
        ema_crossover_signal(df, "AAPL"),
        rsi_macd_signal(df, "AAPL"),
        bollinger_mean_reversion_signal(df, "AAPL"),
    ]
    consensus = aggregate_signals(signals)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


import numpy as np
import pandas as pd
from typing import Optional


def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 14) -> pd.Series:
    """Average Directional Index."""
    try:
        hs, ls, cs = high.shift(1), low.shift(1), close.shift(1)
        tr = pd.concat([high-low, (high-cs).abs(), (low-cs).abs()], axis=1).max(axis=1)
        dp = (high - hs).clip(lower=0)
        dn = (ls - low).clip(lower=0)
        dp = dp.where(dp > dn, 0)
        dn = dn.where(dn > dp, 0)
        atr = tr.ewm(span=period, adjust=False).mean()
        di_p = 100 * dp.ewm(span=period, adjust=False).mean() / atr
        di_n = 100 * dn.ewm(span=period, adjust=False).mean() / atr
        dx = 100 * (di_p - di_n).abs() / (di_p + di_n).replace(0, np.nan)
        return dx.ewm(span=period, adjust=False).mean().fillna(20.0)
    except Exception:
        return pd.Series([20.0] * len(close), index=close.index)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    try:
        h, l, c = df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze()
        cs = c.shift(1)
        tr = pd.concat([h-l, (h-cs).abs(), (l-cs).abs()], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean().fillna(0.0)
    except Exception:
        return pd.Series([0.0] * len(df), index=df.index)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()


def _market_regime(df: Optional[pd.DataFrame], vix: Optional[float] = None) -> str:
    """Classify market into Trending / Ranging / Volatile."""
    if df is None or len(df) < 30:
        return "Ranging"
    h, l, c = df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze()
    adx_now = _calculate_adx(h, l, c, 14).iloc[-1]
    real_vol_pct = (c.iloc[-20:].std() / c.iloc[-20:].mean()) * 100
    if (vix is not None and vix > 30) or real_vol_pct > 4.0:
        return "Volatile"
    if adx_now > 25 and real_vol_pct <= 4.0:
        return "Trending"
    return "Ranging"


# ─────────────────────────────────────────────────────────────────────
# Signal dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    action: Literal["BUY", "SELL", "HOLD"]
    strategy: str
    ticker: str
    reason: str
    strength: float  # 0.0–1.0


# ─────────────────────────────────────────────────────────────────────
# Strategy 1: EMA Crossover 9/21
# ─────────────────────────────────────────────────────────────────────

def ema_crossover_signal(df: pd.DataFrame, ticker: str) -> Signal:
    """
    EMA 9/21 crossover with volume confirmation.

    BUY:  EMA9 crosses above EMA21 (golden cross)
    SELL: EMA9 crosses below EMA21 (death cross)
    HOLD: No crossover on this bar
    """
    if len(df) < 30:
        return Signal("HOLD", "EMA Crossover", ticker, "Insufficient data", 0.0)

    close = df["Close"].squeeze()
    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()

    fast_now, fast_prev = ema_fast.iloc[-1], ema_fast.iloc[-2]
    slow_now, slow_prev = ema_slow.iloc[-1], ema_slow.iloc[-2]
    separation = abs(fast_now - slow_now) / slow_now

    vol_confirmed = False
    if "Volume" in df.columns:
        vol = df["Volume"].squeeze()
        avg_vol = vol.rolling(20).mean().iloc[-1]
        vol_confirmed = vol.iloc[-1] >= 1.2 * avg_vol if avg_vol > 0 else False

    if fast_prev <= slow_prev and fast_now > slow_now:
        vol_note = " + vol surge" if vol_confirmed else ""
        strength = min(separation * 100, 1.0)
        return Signal("BUY", "EMA Crossover", ticker,
                      f"EMA9 crossed above EMA21{vol_note}", round(strength, 2))

    if fast_prev >= slow_prev and fast_now < slow_now:
        vol_note = " + vol surge" if vol_confirmed else ""
        strength = min(separation * 100, 1.0)
        return Signal("SELL", "EMA Crossover", ticker,
                      f"EMA9 crossed below EMA21{vol_note}", round(strength, 2))

    return Signal("HOLD", "EMA Crossover", ticker,
                  "No fresh crossover", 0.3)


# ─────────────────────────────────────────────────────────────────────
# Strategy 2: RSI + MACD
# ─────────────────────────────────────────────────────────────────────

def rsi_macd_signal(df: pd.DataFrame, ticker: str,
                    rsi_oversold: float = 30, rsi_overbought: float = 70) -> Signal:
    """
    RSI 30/70 + MACD crossover confirmation.

    BUY:  RSI < 30 (oversold) AND MACD bullish cross
    SELL: RSI > 70 (overbought) AND MACD bearish cross
    """
    if len(df) < 35:
        return Signal("HOLD", "RSI+MACD", ticker, "Insufficient data", 0.0)

    close = df["Close"].squeeze()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_now = rsi.iloc[-1]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    macd_now, macd_prev = macd_line.iloc[-1], macd_line.iloc[-2]
    sig_now, sig_prev = signal_line.iloc[-1], signal_line.iloc[-2]

    macd_bullish = macd_prev < sig_prev and macd_now > sig_now
    macd_bearish = macd_prev > sig_prev and macd_now < sig_now

    if rsi_now < rsi_oversold and macd_bullish:
        base = (rsi_oversold - rsi_now) / rsi_oversold * 0.6 + 0.4
        return Signal("BUY", "RSI+MACD", ticker,
                      f"RSI={rsi_now:.1f} (oversold) + MACD bullish cross",
                      round(min(base, 1.0), 2))

    if rsi_now > rsi_overbought and macd_bearish:
        base = (rsi_now - rsi_overbought) / (100 - rsi_overbought) * 0.6 + 0.4
        return Signal("SELL", "RSI+MACD", ticker,
                      f"RSI={rsi_now:.1f} (overbought) + MACD bearish cross",
                      round(min(base, 1.0), 2))

    return Signal("HOLD", "RSI+MACD", ticker,
                  f"RSI={rsi_now:.1f}, no actionable setup", 0.1)


# ─────────────────────────────────────────────────────────────────────
# Strategy 3: Bollinger Band Mean Reversion
# ─────────────────────────────────────────────────────────────────────

def bollinger_mean_reversion_signal(df: pd.DataFrame, ticker: str,
                                     period: int = 20, std_dev: float = 2.0) -> Signal:
    """
    Bollinger Band mean reversion with regime filter.

    BUY:  Price below lower band in a Ranging market
    SELL: Price above upper band in a Ranging market
    HOLD: Trending or Volatile market (bands don't mean-revert reliably)
    """
    if len(df) < period + 10:
        return Signal("HOLD", "BB Mean Reversion", ticker, "Insufficient data", 0.0)

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std

    adx = _calculate_adx(high, low, close, period=14)
    adx_now = adx.iloc[-1] if not adx.empty else 20.0
    real_vol_pct = (close.iloc[-period:].std() / close.iloc[-period:].mean()) * 100

    price_now = close.iloc[-1]
    upper_now = upper.iloc[-1]
    lower_now = lower.iloc[-1]

    # Regime check — only trade mean reversion in ranging markets
    if adx_now > 25 and real_vol_pct < 3.0:
        regime = "Trending"
    elif real_vol_pct > 4.0:
        regime = "Volatile"
    else:
        regime = "Ranging"

    if regime != "Ranging":
        return Signal("HOLD", "BB Mean Reversion", ticker,
                      f"Regime={regime} — skip mean reversion", 0.1)

    if price_now < lower_now:
        below_pct = (lower_now - price_now) / lower_now
        strength = round(min(below_pct * 20 + 0.5, 1.0), 2)
        return Signal("BUY", "BB Mean Reversion", ticker,
                      f"Price below lower band, regime=Ranging", strength)

    if price_now > upper_now:
        above_pct = (price_now - upper_now) / upper_now
        strength = round(min(above_pct * 20 + 0.5, 1.0), 2)
        return Signal("SELL", "BB Mean Reversion", ticker,
                      f"Price above upper band, regime=Ranging", strength)

    return Signal("HOLD", "BB Mean Reversion", ticker,
                  "Price within bands", 0.1)


# ─────────────────────────────────────────────────────────────────────
# Consensus — simple majority vote with equal weights
# ─────────────────────────────────────────────────────────────────────

def aggregate_signals(signals: list[Signal]) -> Signal:
    """
    Simple majority-vote consensus.

    Counts BUY vs SELL votes (HOLD = abstain).
    Requires ≥2 agreeing votes to fire. Strength = average of agreeing signals.
    """
    if not signals:
        return Signal("HOLD", "Consensus", "UNKNOWN", "No signals", 0.0)

    buy_signals = [s for s in signals if s.action == "BUY"]
    sell_signals = [s for s in signals if s.action == "SELL"]

    if len(buy_signals) >= 2:
        avg_str = sum(s.strength for s in buy_signals) / len(buy_signals)
        return Signal("BUY", "Consensus", signals[0].ticker,
                      f"{len(buy_signals)}/{len(signals)} strategies agree BUY",
                      round(avg_str, 2))

    if len(sell_signals) >= 2:
        avg_str = sum(s.strength for s in sell_signals) / len(sell_signals)
        return Signal("SELL", "Consensus", signals[0].ticker,
                      f"{len(sell_signals)}/{len(signals)} strategies agree SELL",
                      round(avg_str, 2))

    return Signal("HOLD", "Consensus", signals[0].ticker,
                  "No majority agreement", 0.1)
