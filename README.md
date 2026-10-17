# Signal Fundamentals — 3 Core Technical Analysis Strategies

Three foundational trading strategies for US equities: EMA Crossover, RSI + MACD, and Bollinger Band Mean Reversion.

These three strategies form the foundation of technical analysis. They are well-understood, widely backtested, and behave predictably enough that a new trader can learn from their failures as well as their successes.

## Strategies

| # | Strategy | Type | Signal |
|---|----------|------|--------|
| 1 | EMA Crossover (9/21) | Trend following | Golden/death cross of 9-period and 21-period EMAs |
| 2 | RSI + MACD | Momentum | RSI oversold/overbought confirmed by MACD crossover |
| 3 | Bollinger Band Mean Reversion | Mean reversion | Price outside ±2σ bands in ranging markets |

## Quick Start

```bash
pip install -r requirements.txt

# Configure Webull API credentials
cp .env.example .env
# Edit .env with your Webull OpenAPI keys
# Apply at: https://developer.webull.com.my/apis/docs/authentication/individual-application
# API docs: https://developer.webull.com.my/apis/llms.txt

# Scan for signals
python scanner.py AAPL MSFT NVDA SPY QQQ

# Backtest
python backtester.py AAPL MSFT SPY
```

## How Consensus Works

Signals are combined by simple majority vote:
- If 2+ strategies agree on BUY → BUY
- If 2+ strategies agree on SELL → SELL
- Otherwise → HOLD

## What's Not Included (by design)

- No position sizing or risk management
- No regime-weighted scoring
- No VIX filtering

These are covered in the [Intermediate](https://github.com/ericlawyh/trading-strategies-intermediate) and [Advanced](https://github.com/ericlawyh/trading-strategies-advanced) tiers.

## Credentials

Get your Webull OpenAPI keys at: https://developer.webull.com.my/apis/docs/authentication/individual-application

API docs: https://developer.webull.com.my/apis/llms.txt

## Disclaimer

For educational and research purposes only. Past performance does not guarantee future results.
