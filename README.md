# Mean-Reversion Scanner

Contrarian, 100%-technical mean-reversion scanner and backtester for the
S&P 500 (or any configurable universe): RSI, Bollinger %B, z-score mean
reversion, Stochastic RSI, Williams %R, and options put/call-ratio
sentiment, combined into a single composite score with sector-diversified
position sizing.

## Files

```
TradingScript_MR.ipynb          → main entry point: config, scan, backtest, diagnostics
scorer_mr.py                     → composite scoring logic (RSI/BB/MR/PCR/StochRSI/Williams)
backtester.py                    → walk-forward backtester, 3-phase stop-loss logic
stop_loss_forensics.py           → post-mortem analysis of stopped-out trades
option_move_tracker.py           → synthetic Black-Scholes option P&L simulation on top of MR signals
positions.csv                    → manually-recorded live trades (see Demo Portfolio Tracker below)
portfolio_history.csv            → auto-generated daily portfolio value log
update_portfolio_tracker.py      → daily valuation script, run by the workflow below
requirements.txt
.github/workflows/
    daily-portfolio-tracker.yml  → daily cron, commits portfolio_history.csv back automatically
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

## Demo Portfolio Tracker

Tracks real, live-entered trades from the scanner's output against VOO —
the project's primary track-record / marketing asset.

**Recording a new position (manual — your trading decision, automated
capture):** right after the scoring cell (`scored = scorer_mr.compute_scores(...)`),
set `RECORD_NEW_POSITION = True` with a ticker and share count in the
`record_position()` cell. Every indicator value at entry (RSI, Bollinger
%B, z-score, StochRSI, Williams %R, ATR percentile/regime, PCR, sector,
composite score) is captured into `positions.csv` automatically — no
manual retyping of indicator values.

**Daily valuation (fully automated):** `daily-portfolio-tracker.yml` runs
`update_portfolio_tracker.py` daily via GitHub Actions, computing current
portfolio value and comparing it to what the same capital would be worth
in SPY since the first trade, appending one row to `portfolio_history.csv`
— committed back to the repo automatically, no manual updates needed.

**Ad-hoc in-notebook view:** the `track_positions_vs_benchmark()` cell (near
the end of the notebook) reads `positions.csv` directly and shows each
position's return vs. what the same holding period would have returned in
VOO, alongside the indicator values recorded at entry — a quick way to
review trades without leaving the notebook.

Closing a position: edit its row in `positions.csv` — set `status` to
`closed` and fill in `exit_date`/`exit_price`.
