# equal_weight_benchmark.py
"""
Equal-weight buy & hold benchmark — Phase 2.10 of the roadmap.

Answers a different question than the SPY benchmark already in
run_backtest(): "is the outperformance from STOCK-PICKING/TIMING skill,
or simply from being exposed to the TYPE of stock the MR scanner tends
to select (high-dispersion, mean-reverting names)?"

Method: take the actual set of tickers the scanner selected at least once
during the backtest (trades_df['ticker'].unique()), buy them all
equal-weight at the start of the period (staggered to each ticker's first
available date, since you can't buy a stock before it has data), hold
with NO further trading/rebalancing through the end of the period, and
mark equity on the SAME review dates as the main backtest (so Sharpe/
Sortino/Calmar are computed the same way and are directly comparable via
backtest_metrics.compute_standard_metrics()).

If equal-weight-buy-&-hold-on-this-universe >> SPY, most of the strategy's
edge vs SPY is universe/exposure, not timing. If the STRATEGY also
meaningfully beats equal-weight-buy-&-hold-on-the-SAME-universe, that gap
is the actual stock-picking + entry/exit timing skill.
"""

import pandas as pd

try:
    from backtest_metrics import compute_standard_metrics
except ImportError:
    from trading.backtest_metrics import compute_standard_metrics


def _monthly_review_dates(start_date: str, end_date: str, data_index: pd.DatetimeIndex) -> list:
    """
    Standalone copy of backtester._get_monthly_dates()'s logic (first
    available trading day of each month) — duplicated here on purpose so
    this module has no import dependency on backtester.py (which pulls in
    yfinance just to fetch the SPY benchmark, unrelated to this module's job).
    """
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    trading_days  = data_index[(data_index >= start) & (data_index <= end)]
    monthly       = []
    current_month = None
    for day in trading_days:
        month_key = (day.year, day.month)
        if month_key != current_month:
            monthly.append(day)
            current_month = month_key
    return monthly


def compute_equal_weight_benchmark(
    data:             pd.DataFrame,
    tickers:          list[str],
    start_date:       str,
    end_date:         str,
    initial_capital:  float,
    review_dates:     list | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    data : the same multi-ticker OHLCV DataFrame used by run_backtest()
        (columns are a MultiIndex, level 0 = ticker).
    tickers : the universe to equal-weight — pass
        `trades_df['ticker'].unique().tolist()` for the "same universe
        the scanner picked" comparison.
    start_date, end_date : same window as the backtest being compared.
    initial_capital : same starting capital as the backtest being compared.
    review_dates : mark equity on these dates (should match the main
        backtest's monthly review dates for a fair Sharpe/Sortino
        comparison). If None, derives monthly dates itself via a
        standalone copy of the same "first trading day of each month"
        logic run_backtest() uses, so cadence matches by default.

    Returns
    -------
    equity_df indexed by date, column 'portfolio_value' — same shape as
    run_backtest()['equity'], so it drops straight into
    compute_standard_metrics().
    """
    if review_dates is None:
        review_dates = _monthly_review_dates(start_date, end_date, data.index)

    available = set(data.columns.get_level_values(0))
    tickers = [t for t in dict.fromkeys(tickers) if t in available]   # dedupe, preserve order
    if not tickers:
        raise ValueError("None of the given tickers are present in `data`.")

    start_ts = pd.Timestamp(start_date)
    slice_capital = initial_capital / len(tickers)

    entry_price = {}
    shares      = {}
    skipped     = []
    for t in tickers:
        tdf = data[t]["Close"].dropna()
        avail = tdf[tdf.index >= start_ts]
        if avail.empty:
            skipped.append(t)
            continue
        entry_price[t] = avail.iloc[0]
        shares[t]      = slice_capital / entry_price[t]

    if skipped:
        print(f"   ⚠️  Equal-weight benchmark: {len(skipped)} ticker(s) had no data "
              f">= {start_date}, excluded: {skipped}")

    if not shares:
        raise ValueError("No tickers had usable data on/after start_date.")

    # Re-split capital only across tickers that actually got a position,
    # so excluded tickers don't silently shrink total invested capital.
    if skipped:
        slice_capital = initial_capital / len(shares)
        for t in shares:
            shares[t] = slice_capital / entry_price[t]

    rows = []
    for d in review_dates:
        total_value = 0.0
        for t, sh in shares.items():
            tdf = data[t]["Close"]
            if d in tdf.index and pd.notna(tdf.loc[d]):
                price = tdf.loc[d]
            else:
                prior = tdf[tdf.index <= d].dropna()
                # Frozen at last known price if the ticker has since
                # stopped trading (delisted) — same convention run_backtest()
                # uses for marking open positions through data gaps.
                price = prior.iloc[-1] if not prior.empty else entry_price[t]
            total_value += sh * price
        rows.append({"date": d, "portfolio_value": total_value})

    return pd.DataFrame(rows).set_index("date")


def run_equal_weight_comparison(
    data:             pd.DataFrame,
    trades_df:        pd.DataFrame,
    strategy_summary: dict,
    start_date:       str,
    end_date:         str,
    initial_capital:  float,
) -> dict:
    """
    Convenience wrapper: builds the equal-weight benchmark from
    trades_df['ticker'].unique(), computes its standard metrics, and
    returns {'equity': ..., 'summary': ...} — same shape as a
    run_backtest() result, so print_equal_weight_report() (and your own
    code) can treat it uniformly alongside the strategy's own summary.
    """
    tickers = trades_df["ticker"].unique().tolist()
    equity_df = compute_equal_weight_benchmark(
        data=data, tickers=tickers, start_date=start_date, end_date=end_date,
        initial_capital=initial_capital,
    )
    metrics = compute_standard_metrics(
        trades_df=pd.DataFrame(),   # no discrete trades — buy & hold
        equity_df=equity_df,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
    )
    metrics["n_tickers"] = len(tickers)
    return {"equity": equity_df, "summary": metrics}


def print_equal_weight_report(
    strategy_summary:    dict,
    equal_weight_summary: dict,
    benchmark_return:    float,
    benchmark_ticker:    str = "SPY",
) -> None:
    """
    Three-way comparison: Strategy vs Equal-weight-same-universe vs SPY.
    The gap between columns 1 and 2 is picking/timing skill; the gap
    between columns 2 and 3 is universe/exposure effect.
    """
    s  = strategy_summary
    ew = equal_weight_summary

    print(f"\n{'═'*70}")
    print(f"  STRATEGY vs EQUAL-WEIGHT (same universe) vs {benchmark_ticker}")
    print(f"{'═'*70}")
    print(f"  Equal-weight universe: {ew.get('n_tickers', '?')} tickers "
          f"(every ticker the scanner selected at least once)\n")

    def row(label, s_val, ew_val, bm_val, pct=True, sign=True):
        fmt = f"{{:>+14.2f}}{'%' if pct else ''}" if sign else f"{{:>14.2f}}"
        print(f"    {label:<20}{fmt.format(s_val)}{fmt.format(ew_val)}{fmt.format(bm_val)}")

    print(f"    {'Metric':<20}{'Strategy':>15}{'Equal-weight':>15}{benchmark_ticker:>15}")
    row("Total return",  s["total_return"],  ew["total_return"],  benchmark_return)
    row("Annual return", s["annual_return"], ew["annual_return"], float("nan"))
    row("Sharpe",        s["sharpe_ratio"],  ew["sharpe_ratio"],  float("nan"), pct=False)
    row("Max drawdown",  s["max_drawdown"],  ew["max_drawdown"],  float("nan"))

    picking_skill_gap  = s["total_return"] - ew["total_return"]
    exposure_gap       = ew["total_return"] - benchmark_return

    print(f"\n  🔎 ATTRIBUTION")
    print(f"    Strategy vs equal-weight (same universe):  {picking_skill_gap:+.2f}pp  "
          f"→ picking/timing skill")
    print(f"    Equal-weight vs {benchmark_ticker}:{' '*(16-len(benchmark_ticker))}{exposure_gap:+.2f}pp  "
          f"→ universe/exposure effect")

    total_outperf = s["total_return"] - benchmark_return
    if total_outperf != 0:
        pct_from_exposure = exposure_gap / total_outperf * 100
        if 0 <= pct_from_exposure <= 100:
            print(f"\n    ~{pct_from_exposure:.0f}% of total outperformance vs {benchmark_ticker} "
                  f"is attributable to universe/exposure,")
            print(f"    ~{100 - pct_from_exposure:.0f}% to stock-picking/timing.")
        else:
            # Signs of the two components disagree (e.g. equal-weight
            # underperformed SPY but the strategy still beat SPY) — a
            # single "% attributable to X" split isn't meaningful here,
            # so report the two effects directly instead of a misleading ratio.
            print(f"\n    Note: the two effects point in different directions, so a clean")
            print(f"    percentage split isn't meaningful — read the two pp figures above directly.")
            if exposure_gap < 0:
                print(f"    (Equal-weight universe actually UNDERPERFORMED {benchmark_ticker} — the")
                print(f"    strategy's edge vs {benchmark_ticker} is coming entirely from picking/timing,")
                print(f"    which is also compensating for a weak underlying universe.)")

    print(f"\n{'═'*70}\n")
