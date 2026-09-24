# statistical_validation.py
"""
Phase 6 — Statistical validation & survivorship-bias quantification.

Covers roadmap items 6.2, 6.3, 6.4 (6.1 report / 6.5 sensitivity report are separate).
Like monte_carlo.py / cost_stress_test.py, this module NEVER re-implements the
backtester: it consumes an existing `run_backtest()` result (trades + equity),
except `run_survivorship_arms()` / `run_placebo_universes()`, which call
`run_backtest()` with ONLY the `membership` argument changed (one variable per
comparison, same discipline as the Phase 3/4 A/B cells).

WHAT IS IN HERE
---------------
6.2  trade_significance()      trade-level test of "mean pnl_pct > 0": naive t-test,
                               cluster-robust t-test (trades opened in the same month are
                               NOT independent), cluster bootstrap CI, outlier dependence.
     portfolio_alpha_test()    portfolio-level: alpha/beta vs SPY with Newey-West (HAC)
                               standard errors, Sharpe CI (Mertens SE + stationary bootstrap).
6.3  sample_size_adequacy()    effective sample size, minimum detectable effect, trades /
                               years needed; required_track_record_months() for the demo
                               portfolio; rolling_window_independence() for the 26 windows.
6.4  data_aware_coverage()     how much of the point-in-time universe actually HAS price data
                               (survivors vs non-survivors, by year, by review date).
     survivor_trade_attribution()  did the strategy earn its edge on names that survived?
     selection_propensity()    how over-selected are (eventual) non-survivors vs their weight
                               in the universe? -> empirical multiplier k
     missing_data_bound()      bound the residual bias from tickers with NO data at all
     build_membership_variants()/run_survivorship_arms()/compare_arms()
                               A/B: PIT vs survivors-only vs today's-list-applied-backwards
     run_placebo_universes()   null distribution: same-size RANDOM universe thinning
     stop_gap_diagnostic()     related fill-optimism check: stop exits are filled AT the stop
                               price even when the day's Open gapped below it

OOS DISCIPLINE: every helper here works on whatever trades/equity you pass in. Pass
pre-OOS results only (clamp_to_pre_oos) — this module does not touch the holdout.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from scipy import stats


# ═════════════════════════════════════════════════════════════════════════════
# 0. SMALL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _z(p: float) -> float:
    return float(stats.norm.ppf(p))


def _cluster_ids(entry_dates, cluster_by: str = "entry_month") -> np.ndarray:
    """Integer cluster id per trade. Entries happen on monthly review dates, so
    'entry_month' = 'trades opened in the same market environment'."""
    d = pd.to_datetime(pd.Series(list(entry_dates)))
    if cluster_by == "entry_month":
        key = d.dt.strftime("%Y-%m")
    elif cluster_by == "entry_quarter":
        key = d.dt.year.astype(str) + "Q" + d.dt.quarter.astype(str)
    elif cluster_by == "entry_year":
        key = d.dt.year.astype(str)
    elif cluster_by == "entry_date":
        key = d.dt.strftime("%Y-%m-%d")
    else:
        raise ValueError("cluster_by must be entry_month | entry_quarter | entry_year | entry_date")
    return pd.factorize(key)[0]


def _cluster_mean_test(x: np.ndarray, cid: np.ndarray) -> dict:
    """One-sample test of mean(x) = 0: naive SE and cluster-robust SE (CR1)."""
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    se_naive = sd / np.sqrt(n) if n > 1 else float("nan")

    G_all = int(cid.max()) + 1
    counts = np.bincount(cid, minlength=G_all).astype(float)
    sums = np.bincount(cid, weights=x, minlength=G_all)
    G = int((counts > 0).sum())

    u = sums - mean * counts                          # cluster sums of deviations
    var_cl = (G / (G - 1)) * np.sum(u ** 2) / n ** 2 if G > 1 else float("nan")
    se_cl = float(np.sqrt(var_cl)) if G > 1 else float("nan")

    t_naive = mean / se_naive if se_naive and se_naive > 0 else float("nan")
    t_cl = mean / se_cl if se_cl and se_cl > 0 else float("nan")
    p_naive = float(2 * stats.t.sf(abs(t_naive), n - 1)) if np.isfinite(t_naive) else float("nan")
    p_cl = float(2 * stats.t.sf(abs(t_cl), G - 1)) if np.isfinite(t_cl) else float("nan")

    deff = (var_cl / se_naive ** 2) if (se_naive and se_naive > 0 and np.isfinite(var_cl)) else float("nan")
    return {
        "n": n, "n_clusters": G, "mean": mean, "sd": sd,
        "se_naive": se_naive, "t_naive": t_naive, "p_naive": p_naive,
        "se_cluster": se_cl, "t_cluster": t_cl, "p_cluster": p_cl,
        "design_effect": deff,
        "n_eff": (n / deff) if np.isfinite(deff) and deff > 0 else float("nan"),
        "_counts": counts, "_sums": sums,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6.2  SIGNIFICANCE — TRADE LEVEL
# ═════════════════════════════════════════════════════════════════════════════

def trade_significance(
    trades_df:  pd.DataFrame,
    value_col:  str = "pnl_pct",
    cluster_by: str = "entry_month",
    n_boot:     int = 10_000,
    seed:       int = 42,
    top_k:      tuple = (1, 3, 5, 10),
    trim:       float = 0.05,
) -> dict:
    """
    Is the average trade return distinguishable from zero?

    Three answers, from most naive to most honest:
      1. naive one-sample t-test (assumes ~700 independent trades — it is NOT true:
         up to 5 positions are opened on the same review date, and crash-recovery
         entries cluster in the same weeks),
      2. cluster-robust t-test (clusters = entry month, df = clusters-1),
      3. cluster bootstrap CI of the mean (resamples whole months).
    Plus outlier dependence: the mean with the top-k winners removed.
    """
    df = trades_df.dropna(subset=[value_col, "entry_date"]).copy()
    if len(df) < 5:
        raise ValueError("Need at least 5 trades.")
    x = df[value_col].to_numpy(float)
    cid = _cluster_ids(df["entry_date"], cluster_by)

    core = _cluster_mean_test(x, cid)
    counts, sums = core.pop("_counts"), core.pop("_sums")

    # cluster bootstrap of the pooled mean (ratio estimator: sum / count)
    rng = np.random.default_rng(seed)
    live = np.where(counts > 0)[0]
    idx = rng.integers(0, len(live), size=(n_boot, len(live)))
    s_l, c_l = sums[live], counts[live]
    boot = s_l[idx].sum(axis=1) / c_l[idx].sum(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p_boot = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))

    try:
        nz = x[x != 0]
        p_wilcoxon = float(stats.wilcoxon(nz).pvalue) if len(nz) > 10 else float("nan")
    except Exception:
        p_wilcoxon = float("nan")

    # outlier dependence
    order_desc = np.argsort(-x)
    outlier = {}
    pnl_total = df["pnl"].sum() if "pnl" in df.columns else None
    pnl_sorted_desc = np.sort(df["pnl"].to_numpy(float))[::-1] if "pnl" in df.columns else None
    for k in top_k:
        if k >= len(x) - 5:
            continue
        keep = np.ones(len(x), bool)
        keep[order_desc[:k]] = False
        sub = _cluster_mean_test(x[keep], cid[keep])
        outlier[k] = {
            "mean_excl_topk": sub["mean"],
            "t_cluster_excl_topk": sub["t_cluster"],
            "share_of_total_pnl_topk": (float(pnl_sorted_desc[:k].sum() / pnl_total)
                                        if pnl_total not in (None, 0) else float("nan")),
        }

    return {
        **core,
        "value_col": value_col, "cluster_by": cluster_by,
        "median": float(np.median(x)),
        "trimmed_mean": float(stats.trim_mean(x, trim)),
        "skew": float(stats.skew(x)), "excess_kurtosis": float(stats.kurtosis(x)),
        "win_rate": float((x > 0).mean() * 100),
        "boot_ci95": (float(ci_lo), float(ci_hi)),
        "p_bootstrap": p_boot,
        "prob_mean_le_0": float((boot <= 0).mean()),
        "p_wilcoxon": p_wilcoxon,
        "outliers": outlier,
    }


def print_trade_significance(r: dict) -> None:
    print(f"\n{'═'*64}\n  6.2  TRADE-LEVEL SIGNIFICANCE   ({r['value_col']}, clusters={r['cluster_by']})\n{'═'*64}")
    print(f"  Trades: {r['n']}   Clusters: {r['n_clusters']}   Win rate: {r['win_rate']:.1f}%")
    print(f"  Mean: {r['mean']:+.2f}%   Median: {r['median']:+.2f}%   Trimmed mean: {r['trimmed_mean']:+.2f}%")
    print(f"  Skew: {r['skew']:+.2f}   Excess kurtosis: {r['excess_kurtosis']:+.2f}"
          f"{'   (heavy right tail -> the mean is fragile, see outlier table)' if r['skew'] > 1 else ''}")
    print(f"\n  NAIVE t-test:            t = {r['t_naive']:+.2f}   p = {r['p_naive']:.4f}   (assumes independent trades)")
    print(f"  CLUSTER-ROBUST t-test:   t = {r['t_cluster']:+.2f}   p = {r['p_cluster']:.4f}   (df = {r['n_clusters']-1})")
    print(f"  Design effect: {r['design_effect']:.2f}  ->  effective independent trades ≈ {r['n_eff']:.0f} (of {r['n']})")
    print(f"  Cluster-bootstrap 95% CI of mean: [{r['boot_ci95'][0]:+.2f}%, {r['boot_ci95'][1]:+.2f}%]"
          f"   P(mean<=0) = {r['prob_mean_le_0']:.3f}")
    if np.isfinite(r["p_wilcoxon"]):
        print(f"  Wilcoxon signed-rank (median != 0): p = {r['p_wilcoxon']:.4f}   (ignores clustering)")
    if r["outliers"]:
        print(f"\n  OUTLIER DEPENDENCE (remove top-k winners by {r['value_col']}):")
        print(f"    {'k':>3} {'mean excl.':>11} {'cluster t':>10} {'share of total € pnl in top-k':>32}")
        for k, o in r["outliers"].items():
            print(f"    {k:>3} {o['mean_excl_topk']:>+10.2f}% {o['t_cluster_excl_topk']:>+10.2f} {o['share_of_total_pnl_topk']*100:>30.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# 6.2  SIGNIFICANCE — PORTFOLIO LEVEL (alpha vs SPY, Sharpe CI)
# ═════════════════════════════════════════════════════════════════════════════

def _load_benchmark_close(start, end, ticker: str = "SPY") -> pd.Series:
    import yfinance as yf
    raw = yf.download(ticker,
                      start=(pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                      end=(pd.Timestamp(end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.dropna()


def _ols_hac(y: np.ndarray, x: np.ndarray, lags: int | None = None):
    """OLS y = a + b x with Newey-West (Bartlett) HAC standard errors."""
    T = len(y)
    X = np.column_stack([np.ones(T), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = Xe[l:].T @ Xe[:-l]
        S = S + w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv * T / (T - 2)
    return b, np.sqrt(np.diag(V)), e, lags


def _stationary_bootstrap_idx(T: int, n_boot: int, mean_block: int, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix (n_boot x T)."""
    p = 1.0 / max(mean_block, 1)
    idx = np.empty((n_boot, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n_boot)
    jump = rng.random((n_boot, T)) < p
    new = rng.integers(0, T, (n_boot, T))
    for t in range(1, T):
        idx[:, t] = np.where(jump[:, t], new[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def portfolio_alpha_test(
    equity_df:        pd.DataFrame,
    benchmark_close:  pd.Series | None = None,
    benchmark_ticker: str = "SPY",
    periods_per_year: int = 12,
    risk_free_rate:   float = 0.0,
    hac_lags:         int | None = None,
    n_boot:           int = 5000,
    mean_block:       int | None = None,
    seed:             int = 42,
) -> dict:
    """
    Portfolio-level edge test on the EQUITY CURVE (same return series that
    backtest_metrics uses for Sharpe), regressed on the benchmark:

        r_strategy = alpha + beta * r_benchmark + eps

    * alpha t-stat with Newey-West HAC SEs (autocorrelation/heteroskedasticity robust)
    * paired stationary-bootstrap CI for annualized alpha and Sharpe
    * Sharpe SE adjusted for skew/kurtosis (Mertens 2002) — the naive SE
      sqrt((1+SR²/2)/T) is too small for fat-tailed, right-skewed returns.
    """
    eq = equity_df["portfolio_value"].astype(float).copy()
    eq.index = pd.to_datetime(eq.index)
    if benchmark_close is None:
        benchmark_close = _load_benchmark_close(eq.index.min(), eq.index.max(), benchmark_ticker)
    bench = benchmark_close.copy()
    bench.index = pd.to_datetime(bench.index)
    bench = bench.reindex(eq.index, method="ffill")

    both = pd.concat([eq.pct_change(), bench.pct_change()], axis=1, keys=["s", "b"]).dropna()
    T = len(both)
    if T < 24:
        raise ValueError(f"Only {T} return observations — too few for a meaningful alpha test.")
    rf_p = risk_free_rate / periods_per_year
    y = (both["s"] - rf_p).to_numpy()
    x = (both["b"] - rf_p).to_numpy()

    (a_p, beta), (se_a, se_b), e, L = _ols_hac(y, x, hac_lags)
    t_a = a_p / se_a
    p_a = float(2 * stats.t.sf(abs(t_a), T - 2))
    r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    alpha_ann = ((1 + a_p) ** periods_per_year - 1) * 100

    active = both["s"].to_numpy() - both["b"].to_numpy()
    te = active.std(ddof=1) * np.sqrt(periods_per_year)
    ir = active.mean() * periods_per_year / te if te > 0 else float("nan")

    sr_p = y.mean() / y.std(ddof=1)
    sr_ann = sr_p * np.sqrt(periods_per_year)
    g3 = float(stats.skew(y))
    g4 = float(stats.kurtosis(y, fisher=False))
    var_sr = (1 - g3 * sr_p + (g4 - 1) / 4 * sr_p ** 2) / (T - 1)
    se_sr_ann = float(np.sqrt(max(var_sr, 0)) * np.sqrt(periods_per_year))
    p_sr = float(stats.norm.sf(sr_ann / se_sr_ann)) if se_sr_ann > 0 else float("nan")

    # paired stationary bootstrap (alpha, Sharpe)
    rng = np.random.default_rng(seed)
    block = mean_block or max(2, int(round(T ** (1 / 3))))
    idx = _stationary_bootstrap_idx(T, n_boot, block, rng)
    yb, xb = y[idx], x[idx]
    xm, ym = xb.mean(1), yb.mean(1)
    beta_b = ((xb - xm[:, None]) * (yb - ym[:, None])).sum(1) / ((xb - xm[:, None]) ** 2).sum(1)
    alpha_b = ym - beta_b * xm
    alpha_b_ann = ((1 + alpha_b) ** periods_per_year - 1) * 100
    sr_b = ym / yb.std(1, ddof=1) * np.sqrt(periods_per_year)

    ac1 = float(pd.Series(y).autocorr(1))
    return {
        "T": T, "periods_per_year": periods_per_year, "hac_lags": L, "mean_block": block,
        "alpha_period_pct": a_p * 100, "alpha_annual_pct": alpha_ann,
        "t_alpha_hac": float(t_a), "p_alpha_hac": p_a,
        "beta": float(beta), "se_beta": float(se_b), "r2": float(r2),
        "alpha_annual_boot_ci95": tuple(np.percentile(alpha_b_ann, [2.5, 97.5])),
        "prob_alpha_le_0": float((alpha_b <= 0).mean()),
        "tracking_error_ann_pct": float(te * 100), "information_ratio": float(ir),
        "sharpe_annual": float(sr_ann), "sharpe_se_mertens": se_sr_ann,
        "sharpe_ci95_mertens": (float(sr_ann - 1.96 * se_sr_ann), float(sr_ann + 1.96 * se_sr_ann)),
        "p_sharpe_gt_0": p_sr,
        "sharpe_boot_ci95": tuple(np.percentile(sr_b, [2.5, 97.5])),
        "skew": g3, "kurtosis": g4, "lag1_autocorr": ac1,
        # inputs for 6.3 (per-period units, decimal)
        "alpha_period": float(a_p), "resid_sd_period": float(e.std(ddof=2)),
    }


def print_alpha_test(r: dict, benchmark: str = "SPY") -> None:
    print(f"\n{'═'*64}\n  6.2  PORTFOLIO-LEVEL EDGE vs {benchmark}   (T = {r['T']} periods)\n{'═'*64}")
    print(f"  Alpha (annualized): {r['alpha_annual_pct']:+.2f}%   HAC t = {r['t_alpha_hac']:+.2f}   p = {r['p_alpha_hac']:.4f}   (NW lags = {r['hac_lags']})")
    lo, hi = r["alpha_annual_boot_ci95"]
    print(f"  Alpha bootstrap 95% CI: [{lo:+.2f}%, {hi:+.2f}%]   P(alpha<=0) = {r['prob_alpha_le_0']:.3f}")
    print(f"  Beta: {r['beta']:.2f} (±{r['se_beta']:.2f})   R²: {r['r2']:.2f}   -> "
          f"{'mostly market exposure, alpha is the residual' if r['r2'] > 0.5 else 'low R²: returns are largely independent of the market'}")
    print(f"  Tracking error: {r['tracking_error_ann_pct']:.2f}%   Information ratio: {r['information_ratio']:.2f}")
    print(f"\n  Sharpe: {r['sharpe_annual']:.2f}   95% CI (Mertens SE): [{r['sharpe_ci95_mertens'][0]:.2f}, {r['sharpe_ci95_mertens'][1]:.2f}]"
          f"   P(SR>0) = {r['p_sharpe_gt_0']:.4f}")
    print(f"  Sharpe stationary-bootstrap 95% CI (block≈{r['mean_block']}): [{r['sharpe_boot_ci95'][0]:.2f}, {r['sharpe_boot_ci95'][1]:.2f}]")
    print(f"  Return skew {r['skew']:+.2f} | kurtosis {r['kurtosis']:.2f} | lag-1 autocorr {r['lag1_autocorr']:+.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# 6.3  SAMPLE-SIZE ADEQUACY
# ═════════════════════════════════════════════════════════════════════════════

def required_track_record_months(alpha_per_period: float, sd_per_period: float,
                                 alpha: float = 0.05, power: float = 0.80) -> float:
    """Periods (months if inputs are monthly) needed to detect `alpha_per_period` with
    residual volatility `sd_per_period` at the given size/power (iid approximation)."""
    if alpha_per_period is None or alpha_per_period <= 0 or sd_per_period <= 0:
        return float("inf")
    k = _z(1 - alpha / 2) + _z(power)
    return float((k * sd_per_period / alpha_per_period) ** 2)


def sample_size_adequacy(
    trade_result: dict,
    years_span:   float,
    alpha_result: dict | None = None,
    alpha:        float = 0.05,
    power:        float = 0.80,
) -> dict:
    """
    How many INDEPENDENT trades do you need to see the edge you measured, and do you have them?

    Uses the design-effect-adjusted n_eff from trade_significance(), not the raw trade count.
    Required n solves  mean = (z_{a/2} + z_b) * sd / sqrt(n).
    MDE = smallest true mean per-trade return this sample could reliably detect.
    """
    k = _z(1 - alpha / 2) + _z(power)
    mu, sd, n, n_eff = trade_result["mean"], trade_result["sd"], trade_result["n"], trade_result["n_eff"]
    req_n_eff = (k * sd / mu) ** 2 if mu > 0 else float("inf")
    mde = k * sd / np.sqrt(n_eff) if n_eff and n_eff > 0 else float("nan")
    per_year_eff = n_eff / years_span
    out = {
        "alpha": alpha, "power": power, "years_span": years_span,
        "n_raw": n, "n_eff": n_eff, "observed_mean": mu, "sd": sd,
        "required_n_eff": req_n_eff,
        "adequacy_ratio": (n_eff / req_n_eff) if np.isfinite(req_n_eff) and req_n_eff > 0 else 0.0,
        "mde_mean_pct": mde,
        "effective_trades_per_year": per_year_eff,
        "years_needed": (req_n_eff / per_year_eff) if per_year_eff > 0 else float("inf"),
    }
    if alpha_result is not None:
        m = required_track_record_months(alpha_result["alpha_period"], alpha_result["resid_sd_period"], alpha, power)
        out["portfolio_periods_needed"] = m
        out["portfolio_years_needed"] = m / alpha_result["periods_per_year"]
        out["portfolio_years_observed"] = alpha_result["T"] / alpha_result["periods_per_year"]
    return out


def rolling_window_independence(windows_start: str, windows_end: str,
                                window_years: float = 2, step_months: int = 6) -> dict:
    """The 26 rolling windows overlap heavily — how many independent 2-year samples are there really?"""
    total_years = (pd.Timestamp(windows_end) - pd.Timestamp(windows_start)).days / 365.25
    n_windows, s = 0, pd.Timestamp(windows_start)
    while s + pd.DateOffset(years=int(window_years)) <= pd.Timestamp(windows_end):
        n_windows += 1
        s = s + pd.DateOffset(months=step_months)
    return {
        "n_windows": n_windows, "total_years": total_years,
        "n_independent": int(np.floor(total_years / window_years)),
        "overlap_pct": (1 - step_months / (window_years * 12)) * 100,
        "windows_per_event": window_years * 12 / step_months,
    }


def print_sample_size(r: dict, indep: dict | None = None) -> None:
    print(f"\n{'═'*64}\n  6.3  SAMPLE-SIZE ADEQUACY   (size {r['alpha']:.2f}, power {r['power']:.0%})\n{'═'*64}")
    print(f"  Raw trades: {r['n_raw']}   Effective independent trades: {r['n_eff']:.0f}   Span: {r['years_span']:.1f}y "
          f"(≈{r['effective_trades_per_year']:.0f} effective trades/yr)")
    print(f"  Observed mean {r['observed_mean']:+.2f}% (sd {r['sd']:.2f}%) -> need n_eff ≈ {r['required_n_eff']:.0f}   "
          f"adequacy ratio = {r['adequacy_ratio']:.2f}  ({'ENOUGH' if r['adequacy_ratio'] >= 1 else 'NOT ENOUGH'})")
    print(f"  Minimum detectable mean per trade with this sample: ±{r['mde_mean_pct']:.2f}%   "
          f"(smaller true edges are indistinguishable from noise)")
    print(f"  Years of data needed at the current effective trade rate: {r['years_needed']:.1f}")
    if "portfolio_periods_needed" in r:
        print(f"  Portfolio-level: alpha of this size needs ≈ {r['portfolio_years_needed']:.1f} years to detect "
              f"(observed {r['portfolio_years_observed']:.1f}y)")
    if indep:
        print(f"\n  ROLLING WINDOWS: {indep['n_windows']} windows, but consecutive windows overlap {indep['overlap_pct']:.0f}% "
              f"-> ≈ {indep['n_independent']} independent 2y samples.")
        print(f"  Any single market event is counted in ~{indep['windows_per_event']:.0f} windows "
              f"(e.g. COVID = 1 event, not 4 wins/losses).")


# ═════════════════════════════════════════════════════════════════════════════
# 6.4  SURVIVORSHIP BIAS
# ═════════════════════════════════════════════════════════════════════════════
# Context: the MR backtest ALREADY uses (a) a point-in-time universe download that includes
# delisted names (download_universe_data) and (b) membership intervals restricting NEW
# entries per review date. So the question is NOT "PIT vs no PIT" but:
#   (1) how complete is the price data for the PIT universe — especially for the names that
#       left the index (non-survivors)?       -> data_aware_coverage()
#   (2) what did the strategy earn on survivors vs non-survivors?
#                                             -> survivor_trade_attribution()
#   (3) how big is the effect if the universe were the naive one?
#                                             -> run_survivorship_arms(), run_placebo_universes()
#   (4) how large could the bias from tickers with NO data at all be?
#                                             -> missing_data_bound()

def _norm_membership(membership: pd.DataFrame) -> pd.DataFrame:
    m = membership[["ticker", "date_added", "date_removed"]].copy()
    m["date_added"] = pd.to_datetime(m["date_added"])
    m["date_removed"] = pd.to_datetime(m["date_removed"])
    return m


def active_tickers(membership: pd.DataFrame, as_of=None) -> set:
    """Tickers that are index members on `as_of` (default: today) = what a scrape of today's
    constituent list would give you."""
    m = _norm_membership(membership)
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    mask = (m["date_added"] <= as_of) & (m["date_removed"].isna() | (m["date_removed"] > as_of))
    return set(m.loc[mask, "ticker"])


ARM_LABELS = {
    "pit":                  "A   PIT membership (current backtest)",
    "survivors_true_dates": "S1  survivors only, true add-dates  (pure survivorship)",
    "current_list_naive":   "S2  today's list applied backwards (naive scrape)",
    "no_membership":        "B   membership=None (ever-members, no date restriction)",
}


def build_membership_variants(membership: pd.DataFrame, as_of=None) -> dict:
    """
    Membership tables for the A/B arms. Only the candidate universe changes:
      pit                   unchanged
      survivors_true_dates  keep only tickers still in the index today, with their TRUE
                            add-dates  -> removes exactly the names that left/died
                            (A - S1 = pure survivorship effect)
      current_list_naive    same tickers, but member since 1900 -> what a scrape of today's
                            list gives (S2 - S1 = index-inclusion look-ahead: stocks that
                            were added AFTER a big run-up get traded before they joined)
      no_membership         None -> all tickers in `data`, no date restriction
    """
    m = _norm_membership(membership)
    act = active_tickers(m, as_of)
    surv = m[m["ticker"].isin(act)].copy()
    naive = pd.DataFrame({"ticker": sorted(act),
                          "date_added": pd.Timestamp("1900-01-01"),
                          "date_removed": pd.NaT})
    return {"pit": m, "survivors_true_dates": surv,
            "current_list_naive": naive, "no_membership": None}


def _first_last_valid(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    close = data.xs("Close", axis=1, level=1)
    return close.apply(pd.Series.first_valid_index), close.apply(pd.Series.last_valid_index)


def _d64(ts) -> np.datetime64:
    return np.datetime64(pd.Timestamp(ts).strftime("%Y-%m-%d"), "D")


def _overlap_days(beg, end, fv, lv):
    """calendar days of [beg,end] and of its overlap with [fv,lv]; all datetime64 arrays."""
    days = np.maximum((end - beg).astype("timedelta64[D]").astype(float), 0)
    ob = np.maximum(beg, fv)
    oe = np.minimum(end, lv)
    cov = np.maximum((oe - ob).astype("timedelta64[D]").astype(float), 0)
    return days, np.minimum(cov, days)


def data_aware_coverage(
    membership: pd.DataFrame,
    data:       pd.DataFrame,
    start,
    end,
    as_of=None,
    review_freq: str = "MS",
) -> dict:
    """
    Data-aware version of db.membership_coverage_gap(): for every membership interval,
    how many calendar days actually have a Close in `data` (first..last valid Close)?
    No DB and no hard-coded missing-ticker list needed. Catches partial coverage too
    (ticker present in `data` but history starts late / ends before delisting).

    The number that matters is the coverage of NON-SURVIVORS vs SURVIVORS: if survivors are
    ~100% covered and non-survivors are, say, 60%, the "point-in-time" backtest is still
    tilted toward survivors, just less than a naive scrape.
    """
    m = _norm_membership(membership)
    s0, e0 = pd.Timestamp(start), pd.Timestamp(end)
    act = active_tickers(m, as_of)

    first, last = _first_last_valid(data)
    FAR_F, FAR_P = pd.Timestamp("2200-01-01"), pd.Timestamp("1900-01-01")
    m["fv"] = m["ticker"].map(first).fillna(FAR_F)      # no data -> empty overlap
    m["lv"] = m["ticker"].map(last).fillna(FAR_P)
    m["beg"] = m["date_added"].clip(lower=s0)
    m["end"] = m["date_removed"].fillna(e0).clip(upper=e0)
    m = m[m["end"] > m["beg"]].reset_index(drop=True)
    m["survivor"] = m["ticker"].isin(act)

    # day-resolution arrays: avoids int64 overflow that datetime64[ns] has for the far-future /
    # far-past sentinels used for tickers without any data
    A = lambda col: m[col].to_numpy("datetime64[D]")
    days, cov = _overlap_days(A("beg"), A("end"), A("fv"), A("lv"))
    m["days"], m["covered"] = days, cov

    def _summ(sub: pd.DataFrame) -> dict:
        d, c = sub["days"].sum(), sub["covered"].sum()
        per_t = sub.groupby("ticker")[["days", "covered"]].sum()
        return {"n_tickers": int(per_t.shape[0]),
                "company_days": float(d), "covered_days": float(c),
                "coverage_pct": float(c / d * 100) if d else float("nan"),
                "n_fully_missing": int((per_t["covered"] == 0).sum()),
                "n_partial": int(((per_t["covered"] > 0) & (per_t["covered"] < per_t["days"] * 0.98)).sum())}

    overall, surv, nonsurv = _summ(m), _summ(m[m["survivor"]]), _summ(m[~m["survivor"]])

    # by calendar year
    rows = []
    for yr in range(s0.year, e0.year + 1):
        ys, ye = max(pd.Timestamp(f"{yr}-01-01"), s0), min(pd.Timestamp(f"{yr}-12-31"), e0)
        b = np.maximum(A("beg"), _d64(ys)); e = np.minimum(A("end"), _d64(ye))
        d, c = _overlap_days(b, e, A("fv"), A("lv"))
        sv = m["survivor"].to_numpy()
        rows.append({"year": yr,
                     "company_days": d.sum(),
                     "missing_pct_all": (1 - c.sum() / d.sum()) * 100 if d.sum() else np.nan,
                     "missing_pct_survivors": (1 - c[sv].sum() / d[sv].sum()) * 100 if d[sv].sum() else np.nan,
                     "missing_pct_non_survivors": (1 - c[~sv].sum() / d[~sv].sum()) * 100 if d[~sv].sum() else np.nan})
    by_year = pd.DataFrame(rows).set_index("year")

    # by review date (monthly): members vs members-with-data
    rd, recs = pd.date_range(s0, e0, freq=review_freq), []
    beg_raw, end_raw = A("beg"), A("end")
    fv_a, lv_a = A("fv"), A("lv")
    for d in rd:
        d64 = _d64(d)
        inm = (beg_raw <= d64) & (d64 <= end_raw)
        cov_m = inm & (fv_a <= d64) & (d64 <= lv_a)
        recs.append({"date": d, "n_members": int(inm.sum()), "n_covered": int(cov_m.sum())})
    by_date = pd.DataFrame(recs).set_index("date")
    by_date["missing_share"] = 1 - by_date["n_covered"] / by_date["n_members"].replace(0, np.nan)

    per_t = m.groupby("ticker")[["days", "covered"]].sum()
    missing_t = sorted(per_t.index[per_t["covered"] == 0])
    partial = per_t[(per_t["covered"] > 0) & (per_t["covered"] < per_t["days"] * 0.98)].copy()
    partial["missing_days"] = partial["days"] - partial["covered"]
    partial = partial.sort_values("missing_days", ascending=False).head(15)

    return {"window": (s0, e0), "overall": overall, "survivors": surv, "non_survivors": nonsurv,
            "by_year": by_year, "by_review_date": by_date,
            "missing_tickers": missing_t, "worst_partial": partial}


def print_coverage_report(c: dict) -> None:
    print(f"\n{'═'*64}\n  6.4  DATA-AWARE UNIVERSE COVERAGE  ({c['window'][0].date()} → {c['window'][1].date()})\n{'═'*64}")
    for label, k in (("ALL          ", "overall"), ("survivors    ", "survivors"), ("NON-survivors", "non_survivors")):
        s = c[k]
        print(f"  {label} tickers={s['n_tickers']:>4}  coverage={s['coverage_pct']:>6.2f}%   "
              f"fully missing={s['n_fully_missing']:>3}  partial={s['n_partial']:>3}")
    gap = c["survivors"]["coverage_pct"] - c["non_survivors"]["coverage_pct"]
    verdict = ("residual survivorship tilt EXISTS: survivors are better covered than the names that left"
               if gap > 2 else
               "non-survivors are at least as well covered as survivors: no survivorship tilt from data gaps"
               if gap < -2 else "negligible tilt")
    print(f"  -> coverage gap survivors − non-survivors: {gap:+.2f}pp  ({verdict})")
    print("\n  Missing company-days by year (% of that year's PIT company-days without price data):")
    print(c["by_year"][["missing_pct_all", "missing_pct_survivors", "missing_pct_non_survivors"]].round(2).to_string())
    print(f"\n  Tickers with NO data at all: {len(c['missing_tickers'])}")
    if len(c["worst_partial"]):
        print("  Worst partially-covered tickers (missing days):")
        print(c["worst_partial"][["days", "covered", "missing_days"]].astype(int).to_string())


# ── attribution: what did the strategy earn on survivors vs non-survivors? ──

def survivor_trade_attribution(trades_df: pd.DataFrame, membership: pd.DataFrame,
                               as_of=None, n_boot: int = 10_000, seed: int = 42) -> dict:
    """
    Split the EXISTING trades by whether the ticker is still an index member today.
    If non-survivor trades earn less, a survivors-only backtest would flatter the result by
    roughly  share_non_survivor * (mean_survivor - mean_non_survivor)  per trade.
    """
    act = active_tickers(membership, as_of)
    known = set(membership["ticker"])
    t = trades_df.copy()
    t["is_survivor"] = t["ticker"].isin(act)
    unknown = sorted(set(t["ticker"]) - known)

    def _grp(g: pd.DataFrame) -> dict:
        if len(g) == 0:
            return {"n": 0}
        return {"n": len(g), "mean_pnl_pct": g["pnl_pct"].mean(), "median_pnl_pct": g["pnl_pct"].median(),
                "win_rate": (g["pnl_pct"] > 0).mean() * 100, "sum_pnl": g["pnl"].sum(),
                "avg_hold_days": g["hold_days"].mean(),
                "n_delisted_exits": int((g["exit_reason"] == "delisted").sum()),
                "n_stop_guard_exits": int((g["exit_reason"] == "stop_guard").sum())}

    S, N = _grp(t[t["is_survivor"]]), _grp(t[~t["is_survivor"]])
    tot_pnl = t["pnl"].sum()
    out = {"survivors": S, "non_survivors": N, "unknown_tickers": unknown,
           "share_non_survivor_trades": N["n"] / len(t) * 100 if len(t) else float("nan"),
           "share_non_survivor_pnl": (N.get("sum_pnl", 0) / tot_pnl * 100) if tot_pnl else float("nan"),
           "mean_all": t["pnl_pct"].mean()}
    if S["n"] > 2 and N["n"] > 2:
        a = t.loc[t["is_survivor"], "pnl_pct"].to_numpy(float)
        b = t.loc[~t["is_survivor"], "pnl_pct"].to_numpy(float)
        w = stats.ttest_ind(a, b, equal_var=False)
        rng = np.random.default_rng(seed)
        da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
        db = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
        diff = da - db
        out.update({"diff_mean_surv_minus_non": float(a.mean() - b.mean()),
                    "welch_t": float(w.statistic), "welch_p": float(w.pvalue),
                    "diff_boot_ci95": tuple(np.percentile(diff, [2.5, 97.5])),
                    "first_order_bias_pp": float(a.mean() - t["pnl_pct"].mean())})
    return out


def print_attribution(r: dict) -> None:
    print(f"\n{'═'*64}\n  6.4  SURVIVOR vs NON-SURVIVOR TRADES (same backtest, split ex-post)\n{'═'*64}")
    for lab, k in (("survivors    ", "survivors"), ("NON-survivors", "non_survivors")):
        s = r[k]
        if s["n"] == 0:
            print(f"  {lab}: no trades"); continue
        print(f"  {lab}: n={s['n']:>4}  mean={s['mean_pnl_pct']:>+6.2f}%  median={s['median_pnl_pct']:>+6.2f}%  "
              f"win={s['win_rate']:>5.1f}%  Σpnl=€{s['sum_pnl']:>+9.0f}  hold={s['avg_hold_days']:.0f}d  "
              f"delisted-exits={s['n_delisted_exits']}  stop_guard={s['n_stop_guard_exits']}")
    print(f"  Non-survivors: {r['share_non_survivor_trades']:.1f}% of trades, {r['share_non_survivor_pnl']:.1f}% of total € pnl")
    if "welch_t" in r:
        lo, hi = r["diff_boot_ci95"]
        print(f"  Δ mean (surv − non): {r['diff_mean_surv_minus_non']:+.2f}pp   Welch t={r['welch_t']:+.2f} p={r['welch_p']:.3f}   "
              f"bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"  First-order bias if non-survivor trades were simply absent: {r['first_order_bias_pp']:+.2f}pp per trade")
    if r["unknown_tickers"]:
        print(f"  ⚠️  {len(r['unknown_tickers'])} traded tickers not in membership table (ticker rename?): {r['unknown_tickers'][:10]}")


def selection_propensity(trades_df: pd.DataFrame, membership: pd.DataFrame, coverage: dict, as_of=None) -> dict:
    """
    Odds ratio of (eventual) non-survivors being SELECTED vs their weight in the covered universe.
    k > 1 means the MR scorer over-picks names that later leave the index (oversold names are
    disproportionately distressed) — so tickers we could not download would have been picked
    MORE often than their company-day share suggests.
    """
    act = active_tickers(membership, as_of)
    t = trades_df[trades_df["ticker"].isin(set(membership["ticker"]))]
    s_tr = (~t["ticker"].isin(act)).mean() if len(t) else float("nan")
    cs, cn = coverage["survivors"]["covered_days"], coverage["non_survivors"]["covered_days"]
    s_days = cn / (cs + cn)
    k = (s_tr / (1 - s_tr)) / (s_days / (1 - s_days)) if 0 < s_tr < 1 and 0 < s_days < 1 else float("nan")
    return {"share_trades_non_survivor": float(s_tr), "share_covered_days_non_survivor": float(s_days),
            "k_empirical": float(k), "n_trades": int(len(t))}


def missing_data_bound(trades_df: pd.DataFrame, coverage: dict, k_empirical: float | None = None,
                       k_grid: tuple = (1.0, 2.0, 3.0), extra_loss_scenarios: dict | None = None,
                       value_col: str = "pnl_pct") -> dict:
    """
    Bound the residual bias from PIT members with NO price data.

    Expected 'ghost' trades = Σ_months  entries_in_month × missing_members / covered_members,
    scaled by k (over-selection of distressed names). Then the expectancy if each ghost trade
    had return x:   (n·mean + ghost·x) / (n + ghost).   First-order only: ignores that a ghost
    trade would also displace a real one (5 slots), which slightly understates the true effect.
    Also returns the break-even x* that would push expectancy to zero.
    """
    t = trades_df.dropna(subset=[value_col, "entry_date"])
    n, mu = len(t), float(t[value_col].mean())
    ent = pd.to_datetime(t["entry_date"]).dt.to_period("M").value_counts().sort_index()

    br = coverage["by_review_date"].copy()
    br.index = br.index.to_period("M")
    ratio = ((br["n_members"] - br["n_covered"]) / br["n_covered"].replace(0, np.nan)).fillna(0.0)
    ghost_prop = float((ent * ratio.reindex(ent.index).fillna(0.0)).sum())

    scen = {}
    guard = t.loc[t["exit_reason"] == "stop_guard", value_col] if "exit_reason" in t.columns else pd.Series(dtype=float)
    deli = t.loc[t["exit_reason"] == "delisted", value_col] if "exit_reason" in t.columns else pd.Series(dtype=float)
    if len(guard) >= 5:
        scen[f"avg stop_guard exit (empirical, n={len(guard)})"] = float(guard.mean())
    if len(deli) >= 3:
        scen[f"avg delisted exit (empirical, n={len(deli)})"] = float(deli.mean())
    scen.update({"−30% (gap through the stop)": -30.0, "−60% (distress)": -60.0, "−100% (bankruptcy)": -100.0})
    if extra_loss_scenarios:
        scen.update(extra_loss_scenarios)

    ks = list(k_grid) + ([k_empirical] if k_empirical and np.isfinite(k_empirical) and k_empirical not in k_grid else [])
    rows = []
    for k in ks:
        g = ghost_prop * k
        be = (-n * mu / g) if g > 0 and mu > 0 else float("nan")
        for name, x in scen.items():
            adj = (n * mu + g * x) / (n + g) if (n + g) > 0 else mu
            rows.append({"k": k, "ghost_trades": g, "scenario": name, "ghost_return_pct": x,
                         "adj_expectancy_pct": adj, "delta_pp": adj - mu, "breakeven_ghost_return_pct": be})
    return {"n_trades": n, "mean_pct": mu, "ghost_trades_proportional": ghost_prop,
            "table": pd.DataFrame(rows)}


def print_bound(b: dict, k_emp: float | None = None) -> None:
    print(f"\n{'═'*64}\n  6.4  RESIDUAL BIAS BOUND — PIT members with NO price data\n{'═'*64}")
    print(f"  Trades: {b['n_trades']}   Mean: {b['mean_pct']:+.2f}%   Ghost trades if proportional (k=1): {b['ghost_trades_proportional']:.1f}")
    if k_emp is not None and np.isfinite(k_emp):
        print(f"  Empirical over-selection of non-survivors: k = {k_emp:.2f}")
    tb = b["table"].copy()
    piv = tb.pivot_table(index="scenario", columns="k", values="adj_expectancy_pct", sort=False)
    piv.columns = [f"k={c:g}" for c in piv.columns]
    print(f"\n  Adjusted mean pnl_pct per trade (baseline {b['mean_pct']:+.2f}%):")
    print(piv.round(2).to_string())
    be = tb.drop_duplicates("k")[["k", "ghost_trades", "breakeven_ghost_return_pct"]]
    print("\n  Break-even: return each ghost trade would need for expectancy = 0:")
    for r in be.itertuples():
        print(f"    k={r.k:g}: {r.ghost_trades:.0f} ghost trades -> x* = {r.breakeven_ghost_return_pct:+.1f}%"
              f"{'  (a loss beyond −100% is impossible => edge survives)' if r.breakeven_ghost_return_pct < -100 else ''}")


# ── A/B arms (only `membership` changes) ────────────────────────────────────

def run_survivorship_arms(backtester_module, base_kwargs: dict, membership: pd.DataFrame,
                          regime_detector=None,
                          arms: tuple = ("pit", "survivors_true_dates", "current_list_naive"),
                          as_of=None, extra_run_kwargs: dict | None = None, quiet: bool = True) -> dict:
    """Runs run_backtest() once per arm. ONLY `membership` differs between arms."""
    variants = build_membership_variants(membership, as_of)
    kw = {k: v for k, v in base_kwargs.items() if k not in ("membership", "regime_detector")}
    kw.update(extra_run_kwargs or {})
    out = {}
    for arm in arms:
        buf = io.StringIO()
        with (contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()):
            res = backtester_module.run_backtest(**kw, membership=variants[arm], regime_detector=regime_detector)
        s = res["summary"]
        print(f"  ✔ {ARM_LABELS[arm]:<60} annual {s['annual_return']:+.2f}%  Sharpe {s['sharpe_ratio']:.2f}  trades {s['n_trades']}")
        out[arm] = res
    return out


_CMP_COLS = ["n_trades", "total_return", "annual_return", "sharpe_ratio", "max_drawdown",
             "calmar_ratio", "win_rate", "expectancy_pct"]


def compare_arms(results: dict, baseline: str = "pit", membership: pd.DataFrame | None = None, as_of=None) -> pd.DataFrame:
    base_t = results[baseline]["trades"]
    base_keys = set(zip(base_t["ticker"], base_t["entry_date"]))
    ns = None
    if membership is not None:
        ns = set(membership["ticker"]) - active_tickers(membership, as_of)
    rows = []
    for arm, res in results.items():
        s, t = res["summary"], res["trades"]
        keys = set(zip(t["ticker"], t["entry_date"]))
        row = {"arm": arm, **{c: s[c] for c in _CMP_COLS},
               "trade_overlap_vs_base_%": len(keys & base_keys) / len(keys | base_keys) * 100 if (keys | base_keys) else np.nan}
        if ns is not None:
            row["non_survivor_trades_%"] = t["ticker"].isin(ns).mean() * 100 if len(t) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("arm")
    for c in ("annual_return", "sharpe_ratio", "max_drawdown", "expectancy_pct"):
        df[f"Δ {c}"] = df[c] - df.loc[baseline, c]
    return df


def print_arm_comparison(df: pd.DataFrame) -> None:
    print(f"\n{'═'*64}\n  6.4  SURVIVORSHIP A/B  (only the candidate universe differs)\n{'═'*64}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.round(2).T.to_string())
    print("\n  Read: A − S1 = pure survivorship effect | S1 − S2 = index-inclusion look-ahead | A − S2 = total naive-scrape bias.")
    print("  Caveat: one deterministic path per arm — path-dependence (5 slots, compounding) adds noise; use the placebo test below to size it.")


def run_placebo_universes(backtester_module, base_kwargs: dict, membership: pd.DataFrame,
                          n_placebo: int = 10, regime_detector=None, seed: int = 42, as_of=None,
                          extra_run_kwargs: dict | None = None) -> pd.DataFrame:
    """
    Null distribution for the survivorship effect: remove the SAME NUMBER of tickers as there are
    non-survivors, but chosen at random from the whole universe. If the real survivors-only
    result falls inside this cloud, 'survivorship' is indistinguishable from universe-thinning
    noise (path dependence); if it sits in the tail, it is a systematic effect.
    """
    act = active_tickers(membership, as_of)
    all_t = sorted(set(membership["ticker"]))
    n_ns = len([t for t in all_t if t not in act])
    kw = {k: v for k, v in base_kwargs.items() if k not in ("membership", "regime_detector")}
    kw.update(extra_run_kwargs or {})
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_placebo):
        drop = set(rng.choice(all_t, size=n_ns, replace=False))
        mem_i = _norm_membership(membership)
        mem_i = mem_i[~mem_i["ticker"].isin(drop)]
        with contextlib.redirect_stdout(io.StringIO()):
            res = backtester_module.run_backtest(**kw, membership=mem_i, regime_detector=regime_detector)
        s = res["summary"]
        rows.append({"placebo": i, **{c: s[c] for c in _CMP_COLS}})
        print(f"  placebo {i+1}/{n_placebo}: annual {s['annual_return']:+.2f}%  Sharpe {s['sharpe_ratio']:.2f}")
    return pd.DataFrame(rows).set_index("placebo")


def placebo_verdict(baseline_summary: dict, s1_summary: dict, placebo_df: pd.DataFrame,
                    metrics: tuple = ("annual_return", "sharpe_ratio", "max_drawdown")) -> pd.DataFrame:
    """Where does the real (S1 − A) delta sit in the distribution of (placebo_i − A)?"""
    rows = []
    for m in metrics:
        real = s1_summary[m] - baseline_summary[m]
        null = (placebo_df[m] - baseline_summary[m]).to_numpy()
        pct = float((null <= real).mean() * 100)
        z = (real - null.mean()) / null.std(ddof=1) if len(null) > 2 and null.std(ddof=1) > 0 else float("nan")
        rows.append({"metric": m, "real_delta": real, "placebo_mean": null.mean(), "placebo_sd": null.std(ddof=1),
                     "placebo_p5": np.percentile(null, 5), "placebo_p95": np.percentile(null, 95),
                     "real_percentile_in_null": pct, "z": z})
    return pd.DataFrame(rows).set_index("metric")


# ── tie-break noise floor (needs the `tiebreak_seed` patch in backtester.py) ──

def run_tiebreak_noise(backtester_module, base_kwargs: dict, membership: pd.DataFrame | None,
                       n_runs: int = 10, regime_detector=None, extra_run_kwargs: dict | None = None) -> pd.DataFrame:
    """
    Same config, same data, only the ORDER in which equally-scored candidates are considered
    changes (tiebreak_seed = 1..n_runs). The spread of the headline metrics across these runs is
    the NOISE FLOOR for every A/B comparison in the project: a difference smaller than this
    (e.g. sector cap ON vs OFF = 0.7pp) cannot be attributed to the variable under test.
    """
    kw = {k: v for k, v in base_kwargs.items() if k not in ("membership", "regime_detector")}
    kw.update(extra_run_kwargs or {})
    rows = []
    for i in range(1, n_runs + 1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = backtester_module.run_backtest(**kw, membership=membership,
                                                     regime_detector=regime_detector, tiebreak_seed=i)
        except TypeError as e:
            raise TypeError("backtester.run_backtest() has no `tiebreak_seed` argument — apply "
                            "backtester_determinism.patch first.") from e
        s = res["summary"]
        rows.append({"tiebreak_seed": i, **{c: s[c] for c in _CMP_COLS}})
        print(f"  tie-break run {i}/{n_runs}: annual {s['annual_return']:+.2f}%  Sharpe {s['sharpe_ratio']:.2f}  "
              f"DD {s['max_drawdown']:.2f}%  trades {s['n_trades']}")
    return pd.DataFrame(rows).set_index("tiebreak_seed")


def print_tiebreak_noise(df: pd.DataFrame) -> None:
    print(f"\n{'═'*64}\n  NOISE FLOOR — same config, only the tie-break order differs ({len(df)} runs)\n{'═'*64}")
    tab = df[["annual_return", "sharpe_ratio", "max_drawdown", "n_trades", "expectancy_pct"]].agg(["mean", "std", "min", "max"]).T
    tab["range"] = tab["max"] - tab["min"]
    print(tab.round(2).to_string())
    print("\n  Rule of thumb: an A/B difference smaller than ~2×std (per metric) is NOT distinguishable from tie-break luck.")


# ── related fill-optimism check (not survivorship, but hits the same distressed names) ──

def stop_gap_diagnostic(trades_df: pd.DataFrame, data: pd.DataFrame, slippage_pct: float = 0.0005) -> dict:
    """
    backtester._check_exit_daily() fills stops at the STOP PRICE whenever Low <= stop, even if
    the day OPENED below the stop (gap-down through it). A real stop order fills at the open.
    This measures how often that happened among stop exits and what the fill would have cost.
    """
    t = trades_df[trades_df["exit_reason"].astype(str).str.startswith("stop_")].copy()
    extra = []
    for r in t.itertuples():
        try:
            op = float(data[(r.ticker, "Open")].loc[pd.Timestamp(r.exit_date)])
        except Exception:
            extra.append(np.nan); continue
        raw_exit = r.exit_price / (1 - slippage_pct)
        extra.append((op - raw_exit) / r.entry_price * 100 if op < raw_exit else 0.0)
    t["gap_extra_pct"] = extra
    ok = t["gap_extra_pct"].notna()
    gapped = ok & (t["gap_extra_pct"] < 0)
    n_all = len(trades_df)
    return {
        "n_stop_exits": int(len(t)), "n_checked": int(ok.sum()), "n_gapped_through": int(gapped.sum()),
        "share_gapped_pct": float(gapped.sum() / ok.sum() * 100) if ok.sum() else float("nan"),
        "mean_extra_loss_when_gapped_pct": float(t.loc[gapped, "gap_extra_pct"].mean()) if gapped.any() else 0.0,
        "expectancy_impact_pp": float(t["gap_extra_pct"].fillna(0).sum() / n_all) if n_all else float("nan"),
        "detail": t[["ticker", "entry_date", "exit_date", "exit_reason", "pnl_pct", "gap_extra_pct"]],
    }


def print_stop_gap(r: dict) -> None:
    print(f"\n{'═'*64}\n  FILL-OPTIMISM CHECK — stops filled at stop price vs gap-open\n{'═'*64}")
    print(f"  Stop exits: {r['n_stop_exits']} (checked {r['n_checked']})   gapped through the stop: {r['n_gapped_through']} ({r['share_gapped_pct']:.1f}%)")
    print(f"  Mean extra loss on those (in % of entry): {r['mean_extra_loss_when_gapped_pct']:+.2f}pp")
    print(f"  Estimated impact on mean pnl per trade (all trades): {r['expectancy_impact_pp']:+.3f}pp")


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TEST (synthetic data, no backtester needed):  python statistical_validation.py
# ═════════════════════════════════════════════════════════════════════════════

def _selftest() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2011-01-03", "2025-07-01")
    months = pd.date_range("2011-01-01", "2025-06-01", freq="MS")

    # trades: 5 per month, correlated within month
    rows = []
    for m in months:
        common = rng.normal(0.5, 4)
        for j in range(5):
            rows.append({"ticker": f"T{rng.integers(0, 300)}", "entry_date": (m + pd.Timedelta(days=1)).date(),
                         "pnl_pct": common + rng.normal(0, 8), "pnl": 0.0, "hold_days": 20,
                         "exit_reason": rng.choice(["stop_guard", "target_2", "delisted"], p=[0.6, 0.35, 0.05])})
    tr = pd.DataFrame(rows); tr["pnl"] = tr["pnl_pct"] * 20
    r = trade_significance(tr, n_boot=2000); print_trade_significance(r)
    span = (dates[-1] - dates[0]).days / 365.25
    ss = sample_size_adequacy(r, span)
    print_sample_size(ss, rolling_window_independence("2011-01-01", "2025-05-01"))

    spy = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.010, len(dates))), index=dates)
    m_spy = spy.reindex(months, method="bfill").pct_change().fillna(0)
    eq = pd.DataFrame({"portfolio_value": 10_000 * np.cumprod(1 + 0.004 + 0.7 * m_spy.to_numpy() + rng.normal(0, 0.03, len(months)))}, index=months)
    a = portfolio_alpha_test(eq, benchmark_close=spy, n_boot=1000); print_alpha_test(a)
    print("\n✅ self-test finished (numbers are synthetic).")


if __name__ == "__main__":
    _selftest()
