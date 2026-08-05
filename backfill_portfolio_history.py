"""
backfill_portfolio_history.py

Reconstructs the FULL day-by-day portfolio value trajectory, from each
position's entry_date to today, using real historical daily closing
prices — not just "today's" snapshot like update_portfolio_tracker.py.

Same output schema as update_portfolio_tracker.py's portfolio_history.csv,
so the two stay compatible: run this once (or re-run any time — it fully
reconstructs, safe and idempotent) to fill in the trajectory since you
started recording trades. Once the daily cron is enabled,
update_portfolio_tracker.py continues appending new days forward from
wherever this leaves off.

While the cron is still off (manual-tuning phase), you can just re-run
this script whenever you want an updated chart — no need to remember to
run update_portfolio_tracker.py daily; this always recomputes the whole
line correctly from positions.csv + historical prices.

Usage:
    python backfill_portfolio_history.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

INITIAL_CAPITAL   = 10_000
POSITIONS_FILE    = "positions.csv"
HISTORY_FILE      = "portfolio_history.csv"
BENCHMARK_TICKER  = "SPY"


def load_positions(path: str = POSITIONS_FILE) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["entry_date", "exit_date"])


def get_price_history(ticker: str, start_date: pd.Timestamp) -> pd.Series:
    hist = yf.Ticker(ticker).history(start=start_date.strftime("%Y-%m-%d"))
    if hist.empty:
        return pd.Series(dtype=float)
    hist.index = hist.index.tz_localize(None).normalize()
    return hist["Close"]


def main():
    positions = load_positions()
    if positions.empty:
        print("⚠️  positions.csv is empty — nothing to backfill.")
        return

    earliest_entry = positions["entry_date"].min().normalize()
    today = pd.Timestamp.today().normalize()

    # Use the benchmark's own trading days as the reference calendar —
    # avoids needing a separate market-holiday calendar dependency.
    bench_close = get_price_history(BENCHMARK_TICKER, earliest_entry)
    if bench_close.empty:
        print(f"⚠️  No price data for benchmark {BENCHMARK_TICKER}.")
        return

    trading_days = bench_close.index[(bench_close.index >= earliest_entry) & (bench_close.index <= today)]
    print(f"🔍 Reconstructing {len(trading_days)} trading days ({earliest_entry.date()} -> {today.date()})...")

    # Pre-fetch price history for every ticker once, not once per day
    price_cache = {}
    for ticker in positions["ticker"].unique():
        earliest_for_ticker = positions.loc[positions["ticker"] == ticker, "entry_date"].min()
        price_cache[ticker] = get_price_history(ticker, earliest_for_ticker)

    bench_entry_price = float(bench_close.iloc[0])
    rows = []

    for day in trading_days:
        cash = INITIAL_CAPITAL
        open_value = 0.0
        n_open = 0

        for _, pos in positions.iterrows():
            entry_date = pos["entry_date"].normalize()
            if entry_date > day:
                continue   # not opened yet as of this day

            cost_basis = pos["entry_price"] * pos["shares"]
            exit_date = pos["exit_date"].normalize() if pd.notna(pos.get("exit_date")) else None

            if exit_date is not None and exit_date <= day:
                # already closed by this day — realized P&L booked, no open exposure
                realized_pnl = (pos["exit_price"] - pos["entry_price"]) * pos["shares"]
                cash += realized_pnl
                continue

            # open as of this day
            cash -= cost_basis
            series = price_cache.get(pos["ticker"], pd.Series(dtype=float))
            price_on_day = series.asof(day)   # last known close on/before `day`
            if pd.isna(price_on_day):
                continue
            open_value += price_on_day * pos["shares"]
            n_open += 1

        portfolio_value = cash + open_value
        portfolio_return_pct = (portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        bench_price_on_day = bench_close.asof(day)
        spy_equivalent_value = (INITIAL_CAPITAL / bench_entry_price) * bench_price_on_day
        spy_return_pct = (spy_equivalent_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        alpha_pct = portfolio_return_pct - spy_return_pct

        rows.append({
            "date":                  day.strftime("%Y-%m-%d"),
            "portfolio_value":       round(portfolio_value, 2),
            "portfolio_return_pct":  round(portfolio_return_pct, 2),
            "spy_equivalent_value":  round(spy_equivalent_value, 2),
            "spy_return_pct":        round(spy_return_pct, 2),
            "alpha_pct":             round(alpha_pct, 2),
            "n_open_positions":      n_open,
        })

    history = pd.DataFrame(rows)
    history.to_csv(HISTORY_FILE, index=False)
    print(f"💾 Saved: {HISTORY_FILE} ({len(history)} days)")


if __name__ == "__main__":
    main()
