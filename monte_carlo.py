# monte_carlo.py
"""
Monte Carlo / bootstrap resampling of trades — Phase 2.8 of the roadmap.

Takes an EXISTING trades_df (from a completed run_backtest() call — this
module never re-runs the backtest itself) and randomizes trade order/
sample to show the RANGE of plausible outcomes, instead of just the one
historical path that happened to occur.

Two methods:
- "shuffle"   : permutes the ORDER of the existing trades (same set of
                trades, different sequence). Isolates pure sequence risk —
                "given this exact edge, how much does the order you
                happened to hit these trades in matter?"
- "bootstrap" : resamples trades WITH replacement (same count, but some
                trades appear 0x, some appear 2-3x). Also captures sample
                variability — "what if my sample of trades had looked
                slightly different?" This is the wider, more conservative
                distribution and the default.

IMPORTANT SIMPLIFICATION (documented, not hidden): this module compounds
trade pnl_pct SEQUENTIALLY, each trade risking a fixed fraction
(1/top_n) of CURRENT equity. It does NOT replicate run_backtest()'s exact
overlapping-position, capital-allocation mechanics (multiple concurrent
positions, real share counts, real commission per fill). That level of
fidelity would require re-running the full backtest per simulation, which
is not what "trade-sequence Monte Carlo" is for. This approximation is
standard for measuring sequence risk and the SHAPE/SPREAD of the outcome
distribution — treat the dollar figures as indicative, not exact.
"""

import numpy as np
import pandas as pd


def _simulate_equity_path(pnl_pct_sequence, initial_capital: float, capital_fraction: float) -> np.ndarray:
    """
    Sequential compounding: each trade risks `capital_fraction` of
    CURRENT total equity. See module docstring for the simplification
    this implies vs the real overlapping-position backtester.
    """
    capital = initial_capital
    path = [capital]
    for pnl_pct in pnl_pct_sequence:
        slice_ = capital * capital_fraction
        capital = capital + slice_ * (pnl_pct / 100.0)
        path.append(capital)
    return np.array(path)


def run_monte_carlo(
    trades_df:        pd.DataFrame,
    initial_capital:   float,
    top_n:             int   = 5,      # same top_n as the backtest — sets capital_fraction = 1/top_n
    n_simulations:     int   = 2000,
    method:            str   = "bootstrap",   # "bootstrap" or "shuffle"
    random_seed:       int | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame, one row per simulation, with:
        simulation, final_capital, total_return, max_drawdown
    """
    if trades_df is None or len(trades_df) == 0:
        raise ValueError("trades_df is empty — run a backtest first.")
    if method not in ("bootstrap", "shuffle"):
        raise ValueError("method must be 'bootstrap' or 'shuffle'")

    rng = np.random.default_rng(random_seed)
    pnl_pcts = trades_df["pnl_pct"].to_numpy()
    n_trades = len(pnl_pcts)
    capital_fraction = 1.0 / top_n

    rows = []
    for i in range(n_simulations):
        if method == "bootstrap":
            sample = rng.choice(pnl_pcts, size=n_trades, replace=True)
        else:  # shuffle
            sample = rng.permutation(pnl_pcts)

        path = _simulate_equity_path(sample, initial_capital, capital_fraction)
        final_capital = path[-1]
        total_return  = (final_capital - initial_capital) / initial_capital * 100

        running_max = np.maximum.accumulate(path)
        drawdowns   = (path - running_max) / running_max * 100
        max_dd      = drawdowns.min()

        rows.append({
            "simulation":    i,
            "final_capital": round(final_capital, 2),
            "total_return":  round(total_return, 2),
            "max_drawdown":  round(max_dd, 2),
        })

    return pd.DataFrame(rows)


def summarize_monte_carlo(
    mc_df:               pd.DataFrame,
    known_total_return:  float | None = None,
    known_max_drawdown:  float | None = None,
) -> dict:
    """
    Percentile bands on total_return and max_drawdown, probability of a
    negative outcome, and — if you pass the ACTUAL historical result from
    run_backtest()['summary'] — where that single historical path ranks
    within the simulated distribution (mirrors the percentile-rank idea
    already used in the rolling-windows cell for KNOWN_OUTPERFORMANCE).
    """
    ret = mc_df["total_return"]
    dd  = mc_df["max_drawdown"]

    summary = {
        "n_simulations":        len(mc_df),
        "return_p5":            round(ret.quantile(0.05), 2),
        "return_p25":           round(ret.quantile(0.25), 2),
        "return_median":        round(ret.quantile(0.50), 2),
        "return_p75":           round(ret.quantile(0.75), 2),
        "return_p95":           round(ret.quantile(0.95), 2),
        "drawdown_p5":          round(dd.quantile(0.05), 2),   # shallowest 5% of outcomes
        "drawdown_p50":         round(dd.quantile(0.50), 2),
        "drawdown_p95":         round(dd.quantile(0.95), 2),   # worst 5% of outcomes
        "prob_negative_return": round((ret < 0).mean() * 100, 1),
    }
    if known_total_return is not None:
        summary["known_return_percentile"] = round((ret < known_total_return).mean() * 100, 1)
    if known_max_drawdown is not None:
        summary["known_drawdown_percentile"] = round((dd < known_max_drawdown).mean() * 100, 1)

    return summary


def print_monte_carlo_report(summary: dict, title: str = "MONTE CARLO / BOOTSTRAP RESULTS") -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}  ({summary['n_simulations']} simulations)")
    print(f"{'═'*60}")

    print(f"\n  📊 TOTAL RETURN DISTRIBUTION")
    print(f"    5th percentile (bad case):    {summary['return_p5']:>+8.2f}%")
    print(f"    25th percentile:              {summary['return_p25']:>+8.2f}%")
    print(f"    Median:                       {summary['return_median']:>+8.2f}%")
    print(f"    75th percentile:              {summary['return_p75']:>+8.2f}%")
    print(f"    95th percentile (good case):  {summary['return_p95']:>+8.2f}%")
    print(f"    P(negative return):           {summary['prob_negative_return']:>8.1f}%")

    # NOTE: drawdown values are negative, so the 5th percentile (most
    # negative) is the SEVERE end and the 95th percentile (closest to 0)
    # is the MILD end — opposite of the return distribution above.
    print(f"\n  ⚠️  MAX DRAWDOWN DISTRIBUTION")
    print(f"    5th percentile (severe):      {summary['drawdown_p5']:>+8.2f}%")
    print(f"    Median:                       {summary['drawdown_p50']:>+8.2f}%")
    print(f"    95th percentile (mild):       {summary['drawdown_p95']:>+8.2f}%")

    if "known_return_percentile" in summary:
        print(f"\n  🎯 HISTORICAL RESULT vs SIMULATED DISTRIBUTION")
        print(f"    Your actual historical total return sits at the "
              f"{summary['known_return_percentile']:.0f}th percentile of {summary['n_simulations']} simulations.")
        if summary['known_return_percentile'] >= 90:
            print(f"    ⚠️  The historical result is near the OPTIMISTIC tail — a meaningful")
            print(f"       part of it may reflect favorable trade ordering/sampling, not just edge.")
        elif summary['known_return_percentile'] <= 10:
            print(f"    The historical result sits in the pessimistic tail — the underlying")
            print(f"    edge may be understated by what actually happened.")
        else:
            print(f"    The historical result is unremarkable relative to the simulated range —")
            print(f"    a reasonable sign it's not an artifact of lucky sequencing.")

    if "known_drawdown_percentile" in summary:
        print(f"\n    Your actual historical max drawdown sits at the "
              f"{summary['known_drawdown_percentile']:.0f}th percentile of simulated drawdowns.")

    print(f"\n{'═'*60}\n")


def plot_monte_carlo_distribution(mc_df: pd.DataFrame, known_total_return: float | None = None):
    """
    Histogram of simulated total_return, with the actual historical result
    (if provided) marked as a vertical line — mirrors the style of
    StopLossForensics.plot_distributions() elsewhere in the repo.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(mc_df["total_return"], bins=50, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(mc_df["total_return"].median(), color="black", linestyle="--", linewidth=1,
                label=f"Median = {mc_df['total_return'].median():+.1f}%")
    if known_total_return is not None:
        ax.axvline(known_total_return, color="crimson", linewidth=2,
                    label=f"Actual historical = {known_total_return:+.1f}%")
    ax.set_xlabel("Total return (%)")
    ax.set_ylabel("Simulations")
    ax.set_title(f"Monte Carlo distribution of total return ({len(mc_df)} simulations)")
    ax.legend()
    fig.tight_layout()
    plt.show()
