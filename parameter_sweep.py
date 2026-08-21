# parameter_sweep.py
"""
Parameter sweep / grid search — Phase 2.12 of the roadmap, the last item
of Phase 2 (backtesting validation).

Grid-searches any combination of run_backtest() kwargs (typically the
3-phase stop parameters: guard_days, hard_floor_atr, trail_trigger_atr,
atr_trail_mult — but works for signal_threshold, top_n, or anything else
run_backtest() accepts), and answers the actual question this phase is
for: are the CURRENT parameter values sitting near a genuine local
optimum, or are they arbitrary/fragile — would a small nudge in either
direction have given a meaningfully different result?

IMPORTANT — run this ONLY on the pre-out-of-sample tuning window. Pass
end_date through out_of_sample.clamp_to_pre_oos() (Phase 2.7) so the
sweep can never leak into the holdout — a parameter sweep is exactly the
kind of repeated, many-combination search that most easily overfits to
whatever it's run against.

COST WARNING: each grid cell is a full run_backtest() call. A 5x5 grid
= 25 backtests; 5x5x3 = 125. Keep grids small (3-6 values per axis) and
sweep 1-2 parameters at a time, not the whole parameter space at once.
"""

import contextlib
import io
import itertools

import numpy as np
import pandas as pd

_METRIC_COLS = [
    "total_return", "annual_return", "sharpe_ratio", "sortino_ratio",
    "calmar_ratio", "max_drawdown", "win_rate", "profit_factor",
    "expectancy_pct", "n_trades",
]


def run_parameter_sweep(
    backtester_module,
    base_kwargs:       dict,
    param_grid:        dict,     # {"atr_trail_mult": [1.5, 2.0, 2.5], "trail_trigger_atr": [0.5, 1.0, 1.5]}
    regime_detector    = None,
    membership:        pd.DataFrame | None = None,
    verbose:           bool = True,
) -> pd.DataFrame:
    """
    Full grid search: Cartesian product of every value list in
    `param_grid`. Each combination overrides those keys in `base_kwargs`
    and calls run_backtest() once.

    Returns
    -------
    Long-form DataFrame: one row per combination, swept-param columns +
    all standard metrics (from backtest_metrics via run_backtest()'s
    summary). Feed this into pivot_sweep_results() for a 2D heatmap.
    """
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    n = len(combos)

    if n > 60:
        print(f"⚠️  {n} combinations queued — this runs {n} full backtests and may take a while. "
              f"Consider narrowing the grid (fewer values per axis, or fewer parameters at once).")

    rows = []
    for i, combo in enumerate(combos):
        override = dict(zip(keys, combo))
        kwargs = dict(base_kwargs)
        kwargs.update(override)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = backtester_module.run_backtest(
                regime_detector=regime_detector,
                membership=membership,
                **kwargs,
            )
        s = res["summary"]

        row = dict(override)
        row.update({col: s.get(col) for col in _METRIC_COLS})
        rows.append(row)

        if verbose:
            desc = ", ".join(f"{k}={v}" for k, v in override.items())
            print(f"  [{i+1}/{n}] {desc}  →  total_return={s.get('total_return'):+.2f}%  "
                  f"sharpe={s.get('sharpe_ratio'):.2f}")

    return pd.DataFrame(rows)


def pivot_sweep_results(
    sweep_df:  pd.DataFrame,
    param_x:   str,
    param_y:   str,
    metric:    str = "sharpe_ratio",
) -> pd.DataFrame:
    """
    Reshapes the long-form sweep result into a 2D grid (param_y as rows,
    param_x as columns, `metric` as values) — ready for a heatmap. Only
    meaningful when the sweep covered exactly those two parameters (or
    when the other swept params are held constant per call).
    """
    return sweep_df.pivot_table(index=param_y, columns=param_x, values=metric, aggfunc="mean")


def plot_sweep_heatmap(
    pivot_df:  pd.DataFrame,
    metric:    str = "sharpe_ratio",
    title:     str | None = None,
    current_x                = None,
    current_y                = None,
) -> None:
    """
    Heatmap of `pivot_df` (from pivot_sweep_results). If current_x/
    current_y are given (your CURRENT production values for those two
    parameters), marks that cell with a box so you can see at a glance
    whether you're near the bright (good) region or an outlier.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.2 * len(pivot_df.columns) + 2, 1.0 * len(pivot_df.index) + 2))
    im = ax.imshow(pivot_df.values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    ax.set_xlabel(pivot_df.columns.name)
    ax.set_ylabel(pivot_df.index.name)
    ax.set_title(title or f"{metric} — parameter sweep")

    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")

    if current_x is not None and current_y is not None:
        try:
            xi = list(pivot_df.columns).index(current_x)
            yi = list(pivot_df.index).index(current_y)
            rect = plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor="blue", linewidth=3)
            ax.add_patch(rect)
            ax.text(xi, yi - 0.65, "current", ha="center", color="blue", fontsize=8, fontweight="bold")
        except ValueError:
            print(f"   (current_x/current_y = ({current_x}, {current_y}) not found in the swept grid — "
                  f"no marker drawn)")

    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    plt.show()


def analyze_local_stability(
    pivot_df:    pd.DataFrame,
    current_x,
    current_y,
    metric:      str = "sharpe_ratio",
) -> dict | None:
    """
    Answers the actual question Phase 2.12 exists for: is the CURRENT
    (current_x, current_y) combination near a genuine local optimum, or
    is it sitting in a noisy/fragile region where a small parameter
    change would have given a meaningfully different result?

    Compares the current cell's metric to:
    - the grid-wide best (how much is left on the table)
    - its immediate neighbors' mean/std (how fragile the neighborhood is)
    """
    if current_x not in pivot_df.columns or current_y not in pivot_df.index:
        print(f"⚠️  Current combination ({current_x}, {current_y}) is not in the swept grid — "
              f"can't assess. Include it explicitly in param_grid to check it directly.")
        return None

    xi = list(pivot_df.columns).index(current_x)
    yi = list(pivot_df.index).index(current_y)
    current_val = pivot_df.iloc[yi, xi]

    neighbor_vals = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < len(pivot_df.columns) and 0 <= ny < len(pivot_df.index):
                v = pivot_df.iloc[ny, nx]
                if pd.notna(v):
                    neighbor_vals.append(v)

    best_val = np.nanmax(pivot_df.values)
    best_pos = np.unravel_index(np.nanargmax(pivot_df.values), pivot_df.shape)
    best_x   = pivot_df.columns[best_pos[1]]
    best_y   = pivot_df.index[best_pos[0]]

    print(f"\n{'═'*60}")
    print(f"  LOCAL STABILITY CHECK — {metric}")
    print(f"{'═'*60}")
    print(f"  Current ({pivot_df.columns.name}={current_x}, {pivot_df.index.name}={current_y}): "
          f"{metric} = {current_val:.2f}")
    print(f"  Grid best ({pivot_df.columns.name}={best_x}, {pivot_df.index.name}={best_y}): "
          f"{metric} = {best_val:.2f}")

    result = {
        "current_value": current_val,
        "grid_best_value": best_val,
        "gap_to_best": best_val - current_val,
    }

    if neighbor_vals:
        neighbor_series = pd.Series(neighbor_vals)
        nmean, nstd = neighbor_series.mean(), neighbor_series.std()
        print(f"  Immediate neighbors ({len(neighbor_vals)} cells): mean={nmean:.2f}, std={nstd:.2f}")
        result.update({"neighbor_mean": nmean, "neighbor_std": nstd})

        if nstd > abs(nmean) * 0.5 and nstd > 0:
            print(f"\n  ⚠️  HIGH variance among neighbors relative to their mean — this region of the")
            print(f"     parameter space is noisy. The current value's result may be as much luck")
            print(f"     as edge; don't over-trust its exact position.")
            result["verdict"] = "fragile"
        else:
            z = abs(current_val - nmean) / nstd if nstd > 0 else 0.0
            if z > 3:
                # Current stands out sharply even against a LOW-variance
                # (otherwise flat/smooth) neighborhood — a genuine spike,
                # not just "being at the top of a smooth peak" (which will
                # always differ somewhat from the neighbor average, even
                # on a perfectly well-behaved surface with a coarse grid).
                print(f"\n  ℹ️  Current value stands out sharply (z≈{z:.1f}) even against an otherwise")
                print(f"     low-variance neighborhood — worth extra scrutiny. Could be a genuine")
                print(f"     sharp optimum, or a narrow overfit spike; a finer-grained grid around")
                print(f"     this point would help tell the two apart.")
                result["verdict"] = "spike_worth_checking"
            else:
                print(f"\n  ✅ Current value sits in a LOW-variance neighborhood — a stable region,")
                print(f"     not an isolated spike. Reasonable sign it's not overfit to one exact value.")
                result["verdict"] = "stable"
    else:
        print("  (Current combination is at the edge of the grid — no neighbors on all sides to compare.)")
        result["verdict"] = "edge_of_grid"

    if result["gap_to_best"] > 0.3:   # arbitrary but reasonable Sharpe-scale threshold
        print(f"\n  Note: the grid's best cell beats current by {result['gap_to_best']:.2f} in {metric} —")
        print(f"     worth checking if that best cell is ALSO stable (not just a lucky corner) before")
        print(f"     considering it a real candidate to switch to.")

    print(f"\n{'═'*60}\n")
    return result
