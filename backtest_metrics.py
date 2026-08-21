# backtest_metrics.py
"""
Standardized backtest performance metrics — Phase 2.9 of the roadmap.

Single source of truth for all performance statistics so that
run_backtest(), the rolling-windows loop, the out-of-sample holdout,
the Monte Carlo/bootstrap resampler, and the transaction-cost stress
test all report EXACTLY the same numbers, computed the same way.

Design notes
------------
- total_return / annual_return are computed from `final_capital` (the
  actual ending cash after all positions are closed) vs `initial_capital`,
  using a CALENDAR-DAY CAGR (days/365.25) — this matches the exact
  semantics run_backtest() already had, so those two numbers are
  unchanged by this refactor.
- max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio are computed
  from the EQUITY CURVE's periodic returns (equity_df['portfolio_value']
  at each review date), NOT from individual trade pnl_pct. This is a
  deliberate change from the old inline calculation: the old Sharpe used
  trade-level pnl_pct x sqrt(12), which implicitly (and incorrectly)
  assumed "1 trade per month" — the new version reflects actual
  portfolio-level volatility between review dates. Expect small numeric
  differences vs previous runs; this version is the more standard one.
- expectancy_pct / expectancy_abs are new (previously not reported at
  all).
"""

import numpy as np
import pandas as pd


def compute_standard_metrics(
    trades_df:        pd.DataFrame,
    equity_df:        pd.DataFrame,
    initial_capital:  float,
    final_capital:    float | None = None,
    start_date:        str | pd.Timestamp | None = None,
    end_date:          str | pd.Timestamp | None = None,
    periods_per_year:  int   = 12,   # 12 = monthly review dates (current cadence)
    risk_free_rate:    float = 0.0,  # annualized, e.g. 0.04 for 4%
) -> dict:
    """
    Parameters
    ----------
    trades_df : DataFrame with at least ['pnl', 'pnl_pct'] (one row/trade).
    equity_df : DataFrame with at least ['portfolio_value'], one row per
                review date, chronological (same shape as run_backtest()['equity']).
    initial_capital : starting capital.
    final_capital : ending cash after all positions are closed. If None,
                falls back to equity_df['portfolio_value'].iloc[-1] (less
                accurate — that value is marked at the last REVIEW date,
                not after the final end-of-backtest close-out).
    start_date, end_date : backtest window bounds, for calendar CAGR. If
                None, falls back to equity_df.index[0] / index[-1].
    periods_per_year : how many equity_df rows correspond to one year —
                12 for the current monthly review cadence. Pass 252 if
                you ever switch to daily equity marking.
    risk_free_rate : annualized risk-free rate used in Sharpe/Sortino.
                Defaults to 0 (matches the previous report's assumption).

    Returns
    -------
    dict of metrics — safe to merge directly into a `summary` dict via
    `{**old_summary_fields, **compute_standard_metrics(...)}`.
    """
    metrics = {
        "n_trades":       0,
        "win_rate":       0.0,
        "avg_win":        0.0,
        "avg_loss":       0.0,
        "profit_factor":  0.0,
        "expectancy_pct": 0.0,
        "expectancy_abs": 0.0,
        "total_return":   0.0,
        "annual_return":  0.0,
        "max_drawdown":   0.0,
        "sharpe_ratio":   0.0,
        "sortino_ratio":  0.0,
        "calmar_ratio":   0.0,
    }

    # ── Trade-level stats ────────────────────────────────────────────────
    if trades_df is not None and len(trades_df) > 0:
        n = len(trades_df)
        wins   = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        win_rate = len(wins) / n * 100 if n else 0.0
        avg_win  = wins["pnl_pct"].mean()   if len(wins)   > 0 else 0.0
        avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0.0

        profit_factor = (
            (len(wins) * avg_win) / abs(len(losses) * avg_loss)
            if len(losses) > 0 and avg_loss != 0 else 0.0
        )

        # Expectancy: expected pnl per trade — % terms and absolute €.
        p_win  = len(wins) / n
        p_loss = len(losses) / n
        expectancy_pct = p_win * avg_win + p_loss * avg_loss
        expectancy_abs = trades_df["pnl"].mean()

        metrics.update({
            "n_trades":       n,
            "win_rate":       round(win_rate, 2),
            "avg_win":        round(avg_win, 2),
            "avg_loss":       round(avg_loss, 2),
            "profit_factor":  round(profit_factor, 2),
            "expectancy_pct": round(expectancy_pct, 2),
            "expectancy_abs": round(expectancy_abs, 2),
        })

    # ── Return stats (calendar CAGR, matches previous run_backtest logic) ──
    if final_capital is None and equity_df is not None and len(equity_df) > 0:
        final_capital = equity_df["portfolio_value"].iloc[-1]

    if final_capital is not None and initial_capital:
        if start_date is None and equity_df is not None and len(equity_df) > 0:
            start_date = equity_df.index[0]
        if end_date is None and equity_df is not None and len(equity_df) > 0:
            end_date = equity_df.index[-1]

        total_return = (final_capital - initial_capital) / initial_capital * 100

        annual_return = 0.0
        if start_date is not None and end_date is not None:
            n_years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
            if n_years > 0 and final_capital > 0:
                annual_return = ((final_capital / initial_capital) ** (1 / n_years) - 1) * 100

        metrics.update({
            "total_return":  round(total_return, 2),
            "annual_return": round(annual_return, 2),
        })

    # ── Equity-curve risk stats (drawdown, Sharpe, Sortino, Calmar) ────────
    if equity_df is not None and len(equity_df) > 0 and "portfolio_value" in equity_df.columns:
        equity_vals = equity_df["portfolio_value"].astype(float)

        rolling_max  = equity_vals.cummax()
        drawdowns    = (equity_vals - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min() if not drawdowns.empty else 0.0

        period_returns = equity_vals.pct_change().dropna()
        rf_per_period  = risk_free_rate / periods_per_year if periods_per_year else 0.0
        excess_returns = period_returns - rf_per_period

        sharpe = 0.0
        if len(excess_returns) > 1 and excess_returns.std() > 0:
            sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

        downside = excess_returns[excess_returns < 0]
        sortino = 0.0
        if len(downside) > 0 and downside.std() > 0:
            sortino = (excess_returns.mean() / downside.std()) * np.sqrt(periods_per_year)
        elif len(excess_returns) > 1 and excess_returns.mean() > 0:
            # No losing periods at all — undefined downside risk. Report as
            # +inf rather than 0, so a genuinely flawless run doesn't look
            # like a zero-risk-adjusted-return run.
            sortino = float("inf")

        calmar = 0.0
        if max_drawdown != 0:
            calmar = (metrics["annual_return"] / 100) / abs(max_drawdown / 100)

        metrics.update({
            "max_drawdown":  round(max_drawdown, 2),
            "sharpe_ratio":  round(sharpe, 2) if np.isfinite(sharpe) else sharpe,
            "sortino_ratio": round(sortino, 2) if np.isfinite(sortino) else sortino,
            "calmar_ratio":  round(calmar, 2) if np.isfinite(calmar) else calmar,
        })

    return metrics


def print_standard_metrics(metrics: dict, title: str = "STANDARD METRICS") -> None:
    """Compact, consistent console report — usable standalone or embedded
    inside a larger print_backtest_report()."""
    print(f"\n  📐 {title}")
    print(f"    Total return:      {metrics['total_return']:>+10.2f}%")
    print(f"    Annual return:     {metrics['annual_return']:>+10.2f}%")
    print(f"    Max drawdown:      {metrics['max_drawdown']:>+10.2f}%")
    print(f"    Sharpe ratio:      {metrics['sharpe_ratio']:>10.2f}")
    print(f"    Sortino ratio:     {metrics['sortino_ratio']:>10.2f}")
    print(f"    Calmar ratio:      {metrics['calmar_ratio']:>10.2f}")
    print(f"    Win rate:          {metrics['win_rate']:>10.1f}%")
    print(f"    Avg win / loss:    {metrics['avg_win']:>+9.2f}% / {metrics['avg_loss']:>+.2f}%")
    print(f"    Profit factor:     {metrics['profit_factor']:>10.2f}")
    print(f"    Expectancy/trade:  {metrics['expectancy_pct']:>+9.2f}%  (€{metrics['expectancy_abs']:>+.2f})")
    print(f"    N trades:          {metrics['n_trades']:>10}")
