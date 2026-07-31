# option_move_tracker.py
"""
Option Move Tracker — simulates synthetic call-option P&L on top of MR
signals, to estimate whether "buy a call on the top-score signal" has a
real edge, before committing real capital.

Two measurements per signal:
  1. raw stock path (day-by-day stock % move)
  2. synthetic option value (Black-Scholes, ATM call, configurable
     strike/DTE/IV) — this is what actually determines a real option
     trade's P&L. The same stock % move is worth very different things
     depending on how much time has elapsed (theta decay).

Exit logic (checked in this order every day):
  0. Guard period (first GUARD_DAYS days)       — ONLY a catastrophic floor is active (GUARD_HARD_FLOOR_PCT)
  1. Stop loss (hard floor)                      — after the guard period, option value <= entry * (1 - STOP_LOSS_PCT)
  2. Trailing stop (armed only after profit)     — after the guard period, armed only after profit — see ARM_PROFIT_PCT / TRAIL_PCT
  3. Time exit                                   — no exit inside MAX_HOLD_DAYS

IMPORTANT for trailing stop:
  The trailing stop does NOT arm from entry — if it did, normal theta
  bleed (value drops even with a flat stock) would trigger an almost
  immediate exit on nearly every signal, regardless of whether the thesis
  was right. So it only arms after first reaching ARM_PROFIT_PCT profit,
  and from then on trails TRAIL_PCT below the running peak.

Usage:
    from option_move_tracker import run_option_move_backtest, summarize_option_backtest

    signals_df = pd.DataFrame({
        "ticker":      [...],
        "signal_date": [...],   # scoring day
        "signal_score": [...],  # optional
    })

    results = run_option_move_backtest(data, signals_df)
    summary = summarize_option_backtest(results)
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from trading.data_engine import get_ticker_data
except ImportError:
    from data_engine import get_ticker_data


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — everything overridable from the notebook CONFIG cell
# ─────────────────────────────────────────────────────────────────────────────

ASSUMED_DTE          = 45     # days to expiry at entry
MAX_HOLD_DAYS        = 40     # trading days to follow the signal
ASSUMED_IV           = 0.38   # slightly elevated vs a typical 30% — "penalizes" rather than inflates
RISK_FREE_RATE       = 0.04

STOP_LOSS_PCT        = 0.20   # hard stop: -20% on option value
ARM_PROFIT_PCT       = 0.15   # trailing stop only "arms" after +15% profit
TRAIL_PCT            = 0.05   # after arming, sell at -5% from the peak

# ── Guard period (same idea as the stock backtester's guard_days) ──────────
# An ATM 45-DTE call at IV~38% has so much leverage that a completely
# normal 1-stdev daily stock move (~2.4%) already translates into a
# ~±23% swing in option value — beyond both the stop_loss AND arm level.
# Without a guard period, the tracker "resolves" almost every signal
# within 1-2 days from pure noise, not the actual multi-week thesis.
# During the guard period only a very wide catastrophic floor applies —
# no stop_loss/arm/trailing checks until GUARD_DAYS have passed.
GUARD_DAYS           = 5
GUARD_HARD_FLOOR_PCT = 0.40   # catastrophic floor active ONLY during the guard period

MIN_FORWARD_DAYS     = 5      # fewer than this available forward -> skip (insufficient data)


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES — ATM call pricer
# ─────────────────────────────────────────────────────────────────────────────

def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European call. T in years (e.g. 10 days -> 10/365).

    At expiry (T<=0) returns intrinsic value only.
    """
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


# ─────────────────────────────────────────────────────────────────────────────
# CORE: single-signal path tracker
# ─────────────────────────────────────────────────────────────────────────────

def track_option_path(
    ticker_df:            pd.DataFrame,
    signal_date:          pd.Timestamp,
    assumed_dte:          int   = ASSUMED_DTE,
    max_hold_days:        int   = MAX_HOLD_DAYS,
    assumed_iv:           float = ASSUMED_IV,
    risk_free_rate:       float = RISK_FREE_RATE,
    stop_loss_pct:        float = STOP_LOSS_PCT,
    arm_profit_pct:       float = ARM_PROFIT_PCT,
    trail_pct:            float = TRAIL_PCT,
    guard_days:           int   = GUARD_DAYS,
    guard_hard_floor_pct: float = GUARD_HARD_FLOOR_PCT,
) -> dict | None:
    """
    Follows ONE signal day-by-day and returns how a synthetic ATM call
    on it would have played out.

    Entry convention: score on signal_date (prior close), fill at the
    NEXT trading day's open — same convention as the MR backtester, to
    avoid look-ahead bias.
    """
    if ticker_df is None or ticker_df.empty:
        return None

    idx = ticker_df.index
    after = idx[idx > pd.Timestamp(signal_date)]
    if len(after) == 0:
        return None

    entry_date = after[0]
    entry_pos  = idx.get_loc(entry_date)

    forward_available = len(idx) - entry_pos - 1
    if forward_available < MIN_FORWARD_DAYS:
        return None

    entry_price = ticker_df["Open"].iloc[entry_pos]
    if entry_price <= 0 or pd.isna(entry_price):
        return None

    strike = entry_price  # ATM at entry

    entry_premium = _bs_call(
        S=entry_price, K=strike, T=assumed_dte / 365,
        r=risk_free_rate, sigma=assumed_iv,
    )
    if entry_premium <= 0:
        return None

    stop_floor        = entry_premium * (1 - stop_loss_pct)
    arm_level         = entry_premium * (1 + arm_profit_pct)
    guard_hard_floor  = entry_premium * (1 - guard_hard_floor_pct)

    armed        = False
    running_peak = None  # only starts tracking AFTER the guard period

    n_days = min(max_hold_days, forward_available)

    exit_reason      = "time_exit"
    exit_offset      = n_days
    exit_option_value = None
    exit_stock_price  = None

    for offset in range(1, n_days + 1):
        pos = entry_pos + offset
        remaining_days = max(assumed_dte - offset, 1)  # floor at 1 day so it doesn't zero out artificially
        T = remaining_days / 365

        day_high  = ticker_df["High"].iloc[pos]
        day_low   = ticker_df["Low"].iloc[pos]
        day_close = ticker_df["Close"].iloc[pos]

        val_high  = _bs_call(day_high,  strike, T, risk_free_rate, assumed_iv)
        val_low   = _bs_call(day_low,   strike, T, risk_free_rate, assumed_iv)
        val_close = _bs_call(day_close, strike, T, risk_free_rate, assumed_iv)

        in_guard = offset <= guard_days

        if in_guard:
            # Guard period: ONLY the catastrophic floor — no stop_loss/
            # arm/trailing check, AND we don't update running_peak. If we
            # did, an in-guard spike would silently "bank" a high peak,
            # and the first day after the guard would immediately trigger
            # the trailing stop off it — capturing an early spike instead
            # of genuine subsequent development.
            if val_low <= guard_hard_floor:
                exit_reason       = "guard_floor"
                exit_offset       = offset
                exit_option_value = guard_hard_floor
                exit_stock_price  = day_low
                break

        else:
            # Peak + trailing check on Close (not High/Low): using High
            # for the peak and Low for the check would make the trailing
            # stop fire almost immediately from a single day's own
            # intraday range — comparing that day's best point against its
            # own worst point. Close gives a consistent day-to-day
            # reference point.
            running_peak = val_close if running_peak is None else max(running_peak, val_close)

            # 1) Stop loss — checked against the worst point of the day (Low)
            if val_low <= stop_floor:
                exit_reason       = "stop_loss"
                exit_offset       = offset
                exit_option_value = stop_floor
                exit_stock_price  = day_low
                break

            # 2) Arm the trailing stop (intraday touch on High, once, stays armed)
            if not armed and val_high >= arm_level:
                armed = True

            # 3) Trailing stop — checked against the Close, only if armed
            if armed:
                trail_floor = running_peak * (1 - trail_pct)
                if val_close <= trail_floor:
                    exit_reason       = "trailing_stop"
                    exit_offset       = offset
                    exit_option_value = trail_floor
                    exit_stock_price  = day_close
                    break

        # last day of the window with no exit -> time exit at close
        if offset == n_days:
            exit_reason       = "time_exit"
            exit_offset       = offset
            exit_option_value = val_close
            exit_stock_price  = day_close

    exit_pos   = entry_pos + exit_offset
    exit_date  = idx[exit_pos]

    option_pnl_pct = (exit_option_value - entry_premium) / entry_premium * 100
    stock_pnl_pct  = (exit_stock_price  - entry_price)    / entry_price    * 100

    # If it exited during the guard period, the post-guard peak never
    # started tracking — report entry_premium as a fallback (informational
    # only, doesn't affect any exit logic).
    reported_peak = entry_premium if running_peak is None else running_peak

    return {
        "signal_date":     pd.Timestamp(signal_date),
        "entry_date":      entry_date,
        "entry_price":     round(entry_price, 2),
        "strike":          round(strike, 2),
        "assumed_dte":     assumed_dte,
        "assumed_iv":      assumed_iv,
        "entry_premium":   round(entry_premium, 2),
        "exit_date":       exit_date,
        "days_held":       exit_offset,
        "exit_reason":     exit_reason,
        "exit_option_value": round(exit_option_value, 2),
        "option_pnl_pct":  round(option_pnl_pct, 2),
        "exit_stock_price": round(exit_stock_price, 2),
        "stock_pnl_pct":   round(stock_pnl_pct, 2),
        "armed":           armed,
        "peak_option_value": round(reported_peak, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCH: over historical signals
# ─────────────────────────────────────────────────────────────────────────────

def run_option_move_backtest(
    data:        pd.DataFrame,
    signals_df:  pd.DataFrame,   # columns: ticker, signal_date, [signal_score]
    **tracker_kwargs,
) -> pd.DataFrame:
    """
    Runs track_option_path over every row of signals_df.
    """
    records = []
    skipped = 0

    for _, row in signals_df.iterrows():
        ticker = row["ticker"]
        ticker_df = get_ticker_data(data, ticker)
        if ticker_df is None:
            skipped += 1
            continue

        result = track_option_path(
            ticker_df=ticker_df,
            signal_date=row["signal_date"],
            **tracker_kwargs,
        )

        if result is None:
            skipped += 1
            continue

        result["ticker"] = ticker
        if "signal_score" in row and pd.notna(row["signal_score"]):
            result["signal_score"] = row["signal_score"]

        records.append(result)

    if skipped:
        print(f"⚠️ {skipped} signals skipped (insufficient data / not in universe)")

    if not records:
        return pd.DataFrame()

    cols_order = [
        "ticker", "signal_date", "entry_date", "entry_price", "strike",
        "assumed_dte", "assumed_iv", "entry_premium",
        "exit_date", "days_held", "exit_reason",
        "exit_option_value", "option_pnl_pct",
        "exit_stock_price", "stock_pnl_pct",
        "armed", "peak_option_value", "signal_score",
    ]
    df = pd.DataFrame(records)
    df = df[[c for c in cols_order if c in df.columns]]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY / REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def summarize_option_backtest(results: pd.DataFrame) -> dict:
    """
    Aggregate stats — hit rate per exit_reason, option vs raw stock %
    comparison (to show how much the "naive" estimate differs).
    """
    if results is None or results.empty:
        return {"n_signals": 0}

    n = len(results)
    by_reason = {}
    for reason, grp in results.groupby("exit_reason"):
        by_reason[reason] = {
            "count":            len(grp),
            "pct_of_total":     round(len(grp) / n * 100, 1),
            "avg_option_pnl":   round(grp["option_pnl_pct"].mean(), 2),
            "avg_stock_pnl":    round(grp["stock_pnl_pct"].mean(), 2),
            "avg_days_held":    round(grp["days_held"].mean(), 1),
        }

    return {
        "n_signals":          n,
        "avg_option_pnl_pct": round(results["option_pnl_pct"].mean(), 2),
        "avg_stock_pnl_pct":  round(results["stock_pnl_pct"].mean(), 2),
        "median_option_pnl_pct": round(results["option_pnl_pct"].median(), 2),
        "win_rate_option":    round((results["option_pnl_pct"] > 0).mean() * 100, 1),
        "win_rate_stock":     round((results["stock_pnl_pct"]  > 0).mean() * 100, 1),
        "pct_armed_ever":     round(results["armed"].mean() * 100, 1),
        "by_exit_reason":     by_reason,
    }


def print_option_backtest_report(summary: dict) -> None:
    if summary.get("n_signals", 0) == 0:
        print("❌ No valid signals.")
        return

    print(f"\n{'═'*60}")
    print("  OPTION MOVE TRACKER — SUMMARY")
    print(f"{'═'*60}")
    print(f"  Signals tracked:       {summary['n_signals']}")
    print(f"  Avg option P&L:        {summary['avg_option_pnl_pct']:>+7.2f}%")
    print(f"  Median option P&L:     {summary['median_option_pnl_pct']:>+7.2f}%")
    print(f"  Option win rate:       {summary['win_rate_option']:>7.1f}%")
    print(f"  (raw stock win rate:   {summary['win_rate_stock']:>7.1f}%  <- naive proxy, for comparison)")
    print(f"  % ever armed (trail):  {summary['pct_armed_ever']:>7.1f}%")

    print(f"\n  🚪 EXIT REASONS")
    for reason, d in summary["by_exit_reason"].items():
        print(
            f"    {reason:<15} {d['count']:>4} ({d['pct_of_total']:>5.1f}%)  "
            f"avg option: {d['avg_option_pnl']:>+7.2f}%  "
            f"avg stock: {d['avg_stock_pnl']:>+7.2f}%  "
            f"avg days held: {d['avg_days_held']:>5.1f}"
        )
    print(f"{'═'*60}\n")
