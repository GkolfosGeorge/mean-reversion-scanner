# cost_stress_test.py
"""
Transaction cost stress test — Phase 2.11 of the roadmap.

Reruns run_backtest() at N multiples of the current slippage_pct and
commission_per_share assumptions (default 1x / 2x / 3x) and reports how
much of the edge survives under worse, more realistic execution
conditions. This does NOT touch commission_min or commission_max_pct —
those are IBKR's contractual fee structure, not a "worse market"
assumption, so scaling them wouldn't represent the same thing as scaling
slippage (adverse price impact) and per-share commission.

Usage
-----
    from cost_stress_test import run_cost_stress_test, print_stress_test_report

    stress_df, stress_results = run_cost_stress_test(
        backtester_module = backtester,
        base_kwargs        = backtest_kwargs,   # the dict already built in cell #10
        regime_detector    = regime_detector,   # or None
        membership         = membership,
        multipliers         = (1, 2, 3),
    )
    print_stress_test_report(stress_df)
"""

import contextlib
import io

import pandas as pd


# Columns pulled from each run's summary for the comparison table — all
# already provided by backtest_metrics.compute_standard_metrics() (2.9).
_METRIC_COLS = [
    "total_return", "annual_return", "sharpe_ratio", "sortino_ratio",
    "calmar_ratio", "max_drawdown", "win_rate", "profit_factor",
    "expectancy_pct", "n_trades", "total_transaction_costs",
]


def run_cost_stress_test(
    backtester_module,
    base_kwargs:      dict,
    regime_detector   = None,
    membership:       pd.DataFrame | None = None,
    multipliers:      tuple = (1, 2, 3),
) -> tuple[pd.DataFrame, dict]:
    """
    Parameters
    ----------
    backtester_module : the imported `backtester` module.
    base_kwargs : the SAME kwargs dict you already build for a normal
        backtest run (data, start_date, end_date, top_n, thresholds,
        stop config, scorer_weights, etc.) — must include slippage_pct
        and commission_per_share (or their run_backtest() defaults are
        used as the 1x baseline).
    regime_detector, membership : passed through to run_backtest() same
        as any other call — kept as explicit args (not folded into
        base_kwargs) to match the calling convention used elsewhere in
        the notebook (cell #10, #13).
    multipliers : which multiples of the baseline slippage/commission to
        test. Default (1, 2, 3) = current assumptions, double, triple.

    Returns
    -------
    (stress_df, results_by_multiplier)
        stress_df : one row per multiplier, all standard metrics.
        results_by_multiplier : {multiplier: full run_backtest() dict}
            in case you need the underlying trades_df/equity_df for a
            specific multiplier afterward.
    """
    base_slippage   = base_kwargs.get("slippage_pct", 0.0005)
    base_commission = base_kwargs.get("commission_per_share", 0.005)

    rows = []
    results_by_multiplier = {}

    for m in multipliers:
        kwargs = dict(base_kwargs)
        kwargs["slippage_pct"]           = round(base_slippage * m, 6)
        kwargs["commission_per_share"]   = round(base_commission * m, 6)
        # commission_min / commission_max_pct intentionally left untouched.

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = backtester_module.run_backtest(
                regime_detector=regime_detector,
                membership=membership,
                **kwargs,
            )
        s = res["summary"]

        row = {
            "multiplier":           f"{m}x",
            "slippage_pct":         kwargs["slippage_pct"],
            "commission_per_share": kwargs["commission_per_share"],
        }
        row.update({col: s.get(col) for col in _METRIC_COLS})
        rows.append(row)
        results_by_multiplier[m] = res

        print(f"✅ {m}x  slippage={kwargs['slippage_pct']*100:.3f}%  "
              f"commission=€{kwargs['commission_per_share']:.4f}/share  "
              f"→ total_return={s.get('total_return'):+.2f}%  "
              f"sharpe={s.get('sharpe_ratio'):.2f}")

    stress_df = pd.DataFrame(rows)
    return stress_df, results_by_multiplier


def print_stress_test_report(stress_df: pd.DataFrame) -> None:
    """
    Prints the comparison table plus a fragility read-out: how much
    total_return and Sharpe degrade from the 1x baseline to the worst
    (highest) multiplier tested, and whether the edge survives at all.
    """
    if stress_df.empty:
        print("⚠️  No stress test results to report.")
        return

    print(f"\n{'═'*70}")
    print(f"  TRANSACTION COST STRESS TEST")
    print(f"{'═'*70}\n")

    cols = ["multiplier", "slippage_pct", "commission_per_share",
            "total_return", "annual_return", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "win_rate", "profit_factor", "total_transaction_costs"]
    cols = [c for c in cols if c in stress_df.columns]
    print(stress_df[cols].to_string(index=False))

    baseline = stress_df.iloc[0]
    worst    = stress_df.iloc[-1]

    print(f"\n  📉 FRAGILITY  ({baseline['multiplier']} → {worst['multiplier']})")
    ret_drop = baseline["total_return"] - worst["total_return"]
    print(f"    Total return:  {baseline['total_return']:+.2f}% → {worst['total_return']:+.2f}%  "
          f"(-{ret_drop:.2f}pp)")
    sharpe_drop = baseline["sharpe_ratio"] - worst["sharpe_ratio"]
    print(f"    Sharpe ratio:  {baseline['sharpe_ratio']:.2f} → {worst['sharpe_ratio']:.2f}  "
          f"(-{sharpe_drop:.2f})")

    if worst["total_return"] <= 0:
        print(f"\n    ⚠️  Edge does NOT survive at {worst['multiplier']} costs — "
              f"total return turns non-positive. The strategy's edge is fragile "
              f"to execution quality.")
    elif worst["sharpe_ratio"] < 0.5 * baseline["sharpe_ratio"]:
        print(f"\n    ⚠️  Sharpe more than halves at {worst['multiplier']} costs — "
              f"edge is meaningfully cost-sensitive; worth checking whether it's "
              f"concentrated in low-liquidity names where real slippage could be worse.")
    else:
        print(f"\n    ✅ Edge holds up reasonably well through {worst['multiplier']} costs.")

    print(f"\n{'═'*70}\n")
