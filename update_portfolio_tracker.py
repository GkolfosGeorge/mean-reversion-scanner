"""
update_portfolio_tracker.py

Daily automated update of the demo portfolio's tracked performance vs SPY.
Reads positions.csv (manually maintained — you add a row each time you open
a position), fetches current prices for open positions, computes total
portfolio value, and appends one row per day to portfolio_history.csv.

Nothing here decides which positions to open — that stays your judgment
call, entered manually into positions.csv. This only automates the
day-to-day valuation and benchmark comparison afterward.

positions.csv columns:
    ticker, entry_date, entry_price, shares, stop_loss, target,
    status ("open" or "closed"), exit_date, exit_price

portfolio_history.csv columns (appended daily):
    date, portfolio_value, spy_equivalent_value, alpha_pct, n_open_positions
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

INITIAL_CAPITAL   = 10_000
POSITIONS_FILE    = "positions.csv"
HISTORY_FILE      = "portfolio_history.csv"
BENCHMARK_TICKER  = "SPY"


def load_positions(path: str = POSITIONS_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    return df


def current_price(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No price data for {ticker}")
    return float(hist["Close"].iloc[-1])


def compute_portfolio_value(positions: pd.DataFrame) -> tuple[float, int, list[dict]]:
    """
    Total value = cash remaining (initial capital minus money committed to
    still-open positions) + current market value of open positions +
    realized P&L already booked from closed positions.

    Also returns an informational (non-acting) exit-suggestion flag per open
    position — the tracker never closes anything on its own; you decide.
    This mirrors stop_loss_forensics.py's philosophy: observe what the exit
    methodology WOULD do, without forcing it, so you can watch full
    post-signal behavior if that's what you want for a given position.
    """
    cash = INITIAL_CAPITAL
    open_value = 0.0
    n_open = 0
    flags = []

    for _, pos in positions.iterrows():
        cost_basis = pos["entry_price"] * pos["shares"]

        if pos["status"] == "closed":
            realized_pnl = (pos["exit_price"] - pos["entry_price"]) * pos["shares"]
            cash += realized_pnl
            continue

        # Open position: subtract cost basis from cash, add current market value
        cash -= cost_basis
        price = current_price(pos["ticker"])
        open_value += price * pos["shares"]
        n_open += 1

        suggestion = None
        if pd.notna(pos.get("stop_loss")) and price <= pos["stop_loss"]:
            suggestion = "below stop_loss"
        elif pd.notna(pos.get("target")) and price >= pos["target"]:
            suggestion = "target reached"

        if suggestion:
            flags.append({
                "ticker": pos["ticker"],
                "current_price": round(price, 2),
                "stop_loss": pos.get("stop_loss"),
                "target": pos.get("target"),
                "flag": suggestion,
            })

    total_value = cash + open_value
    return total_value, n_open, flags


def compute_spy_equivalent(positions: pd.DataFrame) -> float:
    """
    What INITIAL_CAPITAL would be worth today if it had simply been put
    into SPY on the date of the very first trade, instead of running the
    strategy. This is the benchmark line for the alpha calculation.
    """
    if positions.empty:
        return INITIAL_CAPITAL

    first_entry = positions["entry_date"].min()
    spy_hist = yf.Ticker(BENCHMARK_TICKER).history(
        start=first_entry.strftime("%Y-%m-%d"),
    )
    if spy_hist.empty:
        raise RuntimeError("No SPY price data available")

    spy_entry_price   = float(spy_hist["Close"].iloc[0])
    spy_current_price = float(spy_hist["Close"].iloc[-1])
    shares_equivalent  = INITIAL_CAPITAL / spy_entry_price
    return shares_equivalent * spy_current_price


def main():
    positions = load_positions()

    if positions.empty:
        print("⚠️  positions.csv is empty — nothing to track yet.")
        sys.exit(0)

    portfolio_value, n_open, flags = compute_portfolio_value(positions)
    spy_value = compute_spy_equivalent(positions)

    portfolio_return_pct = (portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    spy_return_pct       = (spy_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    alpha_pct            = portfolio_return_pct - spy_return_pct

    today = date.today().isoformat()

    row = pd.DataFrame([{
        "date":                  today,
        "portfolio_value":       round(portfolio_value, 2),
        "portfolio_return_pct":  round(portfolio_return_pct, 2),
        "spy_equivalent_value":  round(spy_value, 2),
        "spy_return_pct":        round(spy_return_pct, 2),
        "alpha_pct":             round(alpha_pct, 2),
        "n_open_positions":      n_open,
    }])

    history_path = Path(HISTORY_FILE)
    if history_path.exists():
        history = pd.read_csv(history_path)
        # Overwrite today's row if the script already ran today (safe to
        # re-run mid-day), otherwise append.
        history = history[history["date"] != today]
        history = pd.concat([history, row], ignore_index=True)
    else:
        history = row

    history.to_csv(history_path, index=False)

    print(f"📊 {today}")
    print(f"   Portfolio: ${portfolio_value:,.2f}  ({portfolio_return_pct:+.2f}%)")
    print(f"   SPY equiv: ${spy_value:,.2f}  ({spy_return_pct:+.2f}%)")
    print(f"   Alpha:     {alpha_pct:+.2f}pp")
    print(f"   Open positions: {n_open}")
    print(f"💾 Saved: {HISTORY_FILE}")

    if flags:
        print(f"\nℹ️  Informational only — nothing closed automatically:")
        for f in flags:
            print(f"   {f['ticker']}: {f['flag']} (price=${f['current_price']})")


if __name__ == "__main__":
    main()
