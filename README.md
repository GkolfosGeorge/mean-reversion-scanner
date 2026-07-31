# Mean-Reversion Scanner

Contrarian, 100%-technical mean-reversion scanner and backtester for the
S&P 500 (or any configurable universe): RSI, Bollinger %B, z-score mean
reversion, Stochastic RSI, Williams %R, and options put/call-ratio
sentiment, combined into a single composite score with sector-diversified
position sizing.

## Files

```
TradingScript_MR.ipynb   → main entry point: config, scan, backtest, diagnostics
scorer_mr.py              → composite scoring logic (RSI/BB/MR/PCR/StochRSI/Williams)
backtester.py             → walk-forward backtester, 3-phase stop-loss logic
stop_loss_forensics.py    → post-mortem analysis of stopped-out trades
option_move_tracker.py    → synthetic Black-Scholes option P&L simulation on top of MR signals
requirements.txt
```

`ticker_provider`, `data_engine`, `sector_lookup`, `regime_detector`, and
`options_scanner` are not vendored here — they're installed via the
[`trading-shared-data`](https://github.com/GkolfosGeorge/shared-data-layer)
package (see `requirements.txt`), shared with the trend-following repo and
future strategies.

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook TradingScript_MR.ipynb
```

You'll also need a `data/sp500_membership.csv` file (point-in-time index
membership, including delisted tickers — this is what makes the backtest
survivorship-bias-free). Copy it from the `shared-data-layer` repo's export,
or generate your own with `export_membership_to_csv.py` there if you have
database access.

No database is required to run the scanner or backtest — everything reads
from that static CSV plus live OHLCV/options data from yfinance.

## Notebook structure

The notebook uses a boolean-flag cell pattern: each analysis section (options
scan, backtest with/without regime, rolling-window stress test, stop-loss
forensics, option-move backtest, signal inspection) is gated by explicit
`RUN_*` / `SHOW_*` flags at the top of its cell, defaulting to `False` except
for the core scan. Set the relevant flag to `True` to run that section.
