# 06_backtester.py
"""
Walk-forward backtester, signal-based exits, 100% technical (no macro,
no fundamentals dependency).

Scoring reuses the SAME helper functions as the live scorer_mr.py, on a
point-in-time slice of the data (no look-ahead), so backtest scores match
what the live scanner would have produced that day.

A position closes only on: stop/trailing, target 2, score deterioration,
bear regime exit, or max hold days. New positions open only if capital
and slots are available.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

try:
    from trading.regime_detector import RegimeDetector, REGIME_CONFIGS
except ImportError:
    from regime_detector import RegimeDetector, REGIME_CONFIGS

try:
    from trading.backtest_metrics import compute_standard_metrics
except ImportError:
    from backtest_metrics import compute_standard_metrics

# ── MR scorer — reuses the SAME technical helpers as the live scanner ───────
try:
    from trading.scorer_mr import (
        _rsi, _bollinger, _atr,
        _score_rsi, _score_bollinger, _score_mean_reversion,
        _compute_stoch_rsi, _score_stoch_rsi,
        _compute_williams_r, _score_williams_r,
        RSI_PERIOD as MR_RSI_PERIOD, BB_PERIOD as MR_BB_PERIOD, BB_STD as MR_BB_STD,
        MR_PERIOD, ATR_PERIOD as MR_ATR_PERIOD, VOLUME_PERIOD as MR_VOLUME_PERIOD,
        STOCH_RSI_PERIOD, STOCH_SMOOTH_K, STOCH_SMOOTH_D, WILLIAMS_PERIOD,
        RSI_MAX, MIN_AVG_VOLUME as MR_MIN_AVG_VOLUME,
        W_RSI as MR_W_RSI, W_BB as MR_W_BB, W_MR as MR_W_MR,
        W_STOCHRSI as MR_W_STOCHRSI, W_WILLIAMS as MR_W_WILLIAMS,
        MAX_PER_SECTOR as MR_MAX_PER_SECTOR,
    )
except ImportError:
    from scorer_mr import (
        _rsi, _bollinger, _atr,
        _score_rsi, _score_bollinger, _score_mean_reversion,
        _compute_stoch_rsi, _score_stoch_rsi,
        _compute_williams_r, _score_williams_r,
        RSI_PERIOD as MR_RSI_PERIOD, BB_PERIOD as MR_BB_PERIOD, BB_STD as MR_BB_STD,
        MR_PERIOD, ATR_PERIOD as MR_ATR_PERIOD, VOLUME_PERIOD as MR_VOLUME_PERIOD,
        STOCH_RSI_PERIOD, STOCH_SMOOTH_K, STOCH_SMOOTH_D, WILLIAMS_PERIOD,
        RSI_MAX, MIN_AVG_VOLUME as MR_MIN_AVG_VOLUME,
        W_RSI as MR_W_RSI, W_BB as MR_W_BB, W_MR as MR_W_MR,
        W_STOCHRSI as MR_W_STOCHRSI, W_WILLIAMS as MR_W_WILLIAMS,
        MAX_PER_SECTOR as MR_MAX_PER_SECTOR,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_monthly_dates(
    start_date: str,
    end_date:   str,
    data_index: pd.DatetimeIndex,
) -> list[pd.Timestamp]:
    """First available trading day of each month — used for re-scoring."""
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


def _compute_mr_score_at_date(
    df:              pd.DataFrame,
    date:            pd.Timestamp,
    stop_atr_mult:   float = 1.5,
    target_atr_mult: float = 4.0,
    atr_trail_mult:  float = 2.5,
    weights:         dict | None = None,
) -> dict | None:
    """
    Point-in-time MR score — same formula as scorer_mr.compute_scores(),
    for one ticker/date with no look-ahead (df is sliced at date). 100%
    technical, no PCR (no reliable historical options data) — always uses
    the "no-PCR" redistributed weights.

    `weights` can override both scoring weights and any period/threshold
    used by scorer_mr (rsi_period, bb_period, mr_period, etc.) — same dict
    passed as `scorer_weights` to run_backtest().
    """
    w = weights or {}

    _rsi_period       = w.get("rsi_period",       MR_RSI_PERIOD)
    _bb_period        = w.get("bb_period",        MR_BB_PERIOD)
    _bb_std           = w.get("bb_std",           MR_BB_STD)
    _mr_period        = w.get("mr_period",        MR_PERIOD)
    _atr_period       = w.get("atr_period",       MR_ATR_PERIOD)
    _volume_period    = w.get("volume_period",    MR_VOLUME_PERIOD)
    _stoch_rsi_period = w.get("stoch_rsi_period", STOCH_RSI_PERIOD)
    _stoch_smooth_k   = w.get("stoch_smooth_k",   STOCH_SMOOTH_K)
    _stoch_smooth_d   = w.get("stoch_smooth_d",   STOCH_SMOOTH_D)
    _williams_period  = w.get("williams_period",  WILLIAMS_PERIOD)
    _rsi_max          = w.get("rsi_max",          RSI_MAX)
    _min_avg_volume   = w.get("min_avg_volume",   MR_MIN_AVG_VOLUME)

    df_slice = df[df.index <= date]
    warmup = max(_mr_period, _bb_period, _rsi_period, _volume_period) + 5
    if len(df_slice) < warmup:
        return None

    close  = df_slice["Close"]
    high   = df_slice["High"]
    low    = df_slice["Low"]
    volume = df_slice["Volume"]

    price = close.iloc[-1]
    if price <= 0 or pd.isna(price):
        return None

    avg_vol = volume.rolling(_volume_period).mean().iloc[-1]
    if pd.isna(avg_vol) or avg_vol < _min_avg_volume:
        return None

    rsi_val = _rsi(close, _rsi_period).iloc[-1]
    if rsi_val > _rsi_max:
        return None

    _, _, pct_b_series, _ = _bollinger(close, _bb_period, _bb_std)
    pct_b_val = pct_b_series.iloc[-1]

    atr_val = _atr(high, low, close, _atr_period).iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        return None

    mr_window    = min(_mr_period, len(close))
    rolling_mean = close.rolling(mr_window).mean()
    rolling_std  = close.rolling(mr_window).std()
    z_val        = ((close - rolling_mean) / rolling_std.replace(0, np.nan)).iloc[-1]

    stoch_k, _ = _compute_stoch_rsi(
        close, rsi_period=_rsi_period, stoch_period=_stoch_rsi_period,
        smooth_k=_stoch_smooth_k, smooth_d=_stoch_smooth_d,
    )
    stoch_k_val = stoch_k.iloc[-1]
    wr_val      = _compute_williams_r(high, low, close, _williams_period).iloc[-1]

    s_rsi      = _score_rsi(rsi_val)
    s_bb       = _score_bollinger(pct_b_val)
    s_mr       = _score_mean_reversion(z_val)
    s_stochrsi = _score_stoch_rsi(stoch_k_val)
    s_williams = _score_williams_r(wr_val)

    _w_rsi      = w.get("w_rsi",      MR_W_RSI)
    _w_bb       = w.get("w_bb",       MR_W_BB)
    _w_mr       = w.get("w_mr",       MR_W_MR)
    _w_stochrsi = w.get("w_stochrsi", MR_W_STOCHRSI)
    _w_williams = w.get("w_williams", MR_W_WILLIAMS)
    total = _w_rsi + _w_bb + _w_mr + _w_stochrsi + _w_williams
    if total <= 0:
        return None
    _w_rsi, _w_bb, _w_mr, _w_stochrsi, _w_williams = (
        _w_rsi / total, _w_bb / total, _w_mr / total,
        _w_stochrsi / total, _w_williams / total,
    )

    composite = round(
        s_rsi * _w_rsi + s_bb * _w_bb + s_mr * _w_mr +
        s_stochrsi * _w_stochrsi + s_williams * _w_williams,
        2
    )

    return {
        "signal_score":   composite,
        "price":          round(price, 2),
        "atr":            round(atr_val, 2),
        "avg_volume":     round(avg_vol, 0) if pd.notna(avg_vol) else None,  # shares/day — used by atr_risk_based liquidity cap
        "stop_loss":      round(price - stop_atr_mult  * atr_val, 2),
        "atr_trail_stop": round(price - atr_trail_mult * atr_val, 2),
        "target_2":       round(price + target_atr_mult * atr_val, 2),
    }


def _check_exit_daily(
    pos:            dict,
    period_data:    pd.DataFrame,
    max_hold_days:  int,
    atr_trail_mult: float = 2.5,
) -> tuple[str | None, float | None, pd.Timestamp | None]:
    """
    3-phase stop logic — eliminates premature stops:

    Phase 1 — Guard period (first guard_days):
        Only the hard floor = entry - hard_floor_atr x ATR is active.
        No trailing. Gives the position time to develop.

    Phase 2 — Breakeven (after guard, if the position hasn't won yet):
        Stop = entry price (lock breakeven, zero downside risk).
        Skipped if the position is already profitable.

    Phase 3 — ATR Trailing (only once close > entry + 1x ATR):
        Stop = peak - atr_trail_mult x ATR(entry).
        Lets winners run.

    Returns (exit_reason, exit_price, exit_date) or (None, None, None).
    """
    entry_price   = pos["entry_price"]
    atr_at_entry  = pos.get("atr_at_entry") or (entry_price * 0.02)  # fallback 2%
    entry_date    = pos["entry_date"]
    peak_price    = pos.get("peak_price", entry_price)

    # Phase parameters (could become args if tuning is needed)
    guard_days      = pos.get("guard_days", 15)          # phase 1: days without trail
    hard_floor_atr  = pos.get("hard_floor_atr", 3.0)     # phase 1: max loss = 3x ATR
    trail_trigger   = pos.get("trail_trigger_atr", 1.0)  # phase 3: activates only after +1x ATR profit

    hard_floor = entry_price - hard_floor_atr * atr_at_entry

    for day, row in period_data.iterrows():
        high_today  = row["High"]
        low_today   = row["Low"]
        close_today = row["Close"]
        days_held   = (day - entry_date).days

        # ── Actualise peak ────────────────────────────────────────────────────
        if high_today > peak_price:
            peak_price = high_today
            pos["peak_price"] = peak_price

        # ── Compute active stop ───────────────────────────────────────────────
        trail_stop = peak_price - atr_trail_mult * atr_at_entry
        profit_trigger_price = entry_price + trail_trigger * atr_at_entry

        if days_held < guard_days:
            # Phase 1: hard floor only
            current_stop = hard_floor
            stop_phase   = "guard"
        elif peak_price >= profit_trigger_price:
            # Phase 3: ATR trailing — the position has gained enough
            # Stop never drops below breakeven (lock-in)
            current_stop = max(trail_stop, entry_price)
            stop_phase   = "trail"
        else:
            # Phase 2: breakeven — haven't gained 1x ATR yet
            current_stop = entry_price
            stop_phase   = "breakeven"

        # ── Hit stop ──────────────────────────────────────────────────────────
        if low_today <= current_stop:
            reason = f"stop_{stop_phase}"
            return reason, round(current_stop, 2), day

        # ── Hit target ────────────────────────────────────────────────────────
        if high_today >= pos["target_2"]:
            return "target_2", pos["target_2"], day

        # ── Max hold ─────────────────────────────────────────────────────────
        if days_held >= max_hold_days:
            return "max_hold", close_today, day

    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL RELIABILITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_signal_reliability(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty or "regime" not in trades_df.columns:
        return {}

    analysis = {}

    for regime in ["bull", "neutral", "bear", "fixed"]:
        regime_trades = trades_df[trades_df["regime"] == regime]
        if len(regime_trades) == 0:
            continue
        wins     = (regime_trades["pnl_pct"] > 0).sum()
        total    = len(regime_trades)
        avg_pnl  = regime_trades["pnl_pct"].mean()
        avg_win  = regime_trades[regime_trades["pnl_pct"] > 0]["pnl_pct"].mean() if wins > 0 else 0
        avg_loss = regime_trades[regime_trades["pnl_pct"] <= 0]["pnl_pct"].mean() if (total - wins) > 0 else 0
        analysis[f"regime_{regime}"] = {
            "n_trades": total,
            "win_rate": round(wins / total * 100, 1),
            "avg_pnl":  round(avg_pnl, 2),
            "avg_win":  round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
        }

    if "signal_score" in trades_df.columns:
        bins   = [0, 6.0, 6.5, 7.0, 7.5, 10.0]
        labels = ["<6.0", "6.0-6.5", "6.5-7.0", "7.0-7.5", ">7.5"]
        trades_df = trades_df.copy()
        trades_df["score_bucket"] = pd.cut(
            trades_df["signal_score"], bins=bins, labels=labels, right=True
        )
        for bucket in labels:
            bt = trades_df[trades_df["score_bucket"] == bucket]
            if len(bt) == 0:
                continue
            wins  = (bt["pnl_pct"] > 0).sum()
            total = len(bt)
            analysis[f"score_{bucket}"] = {
                "n_trades": total,
                "win_rate": round(wins / total * 100, 1),
                "avg_pnl":  round(bt["pnl_pct"].mean(), 2),
            }

    if "exit_reason" in trades_df.columns:
        for reason in trades_df["exit_reason"].unique():
            rt = trades_df[trades_df["exit_reason"] == reason]
            analysis[f"exit_{reason}"] = {
                "n_trades": len(rt),
                "avg_pnl":  round(rt["pnl_pct"].mean(), 2),
                "avg_hold": round(rt["hold_days"].mean(), 1) if "hold_days" in rt.columns else None,
            }

    return analysis

# ─────────────────────────────────────────────────────────────────────────────
# NOTE: the earnings-avoidance filter (_has_upcoming_earnings) that used to
# live here relied on fundamentals.py's cached "next_earnings_date". Since
# the backtester became 100% fundamentals-independent, the filter was
# removed. It could be reintroduced later via a lightweight, standalone
# earnings-calendar cache (unrelated to fundamental scoring) if it proves
# important in practice.
# ─────────────────────────────────────────────────────────────────────────────
# POINT-IN-TIME MEMBERSHIP (restricts NEW-position candidates per review date)
# ─────────────────────────────────────────────────────────────────────────────

def _build_membership_intervals(membership: pd.DataFrame) -> dict:
    """
    membership: DataFrame [ticker, date_added, date_removed] (see
    db.get_membership_table()). Returns {ticker: [(start, end), ...]},
    end=pd.Timestamp.max for still-active memberships.
    """
    intervals = {}
    for row in membership.itertuples(index=False):
        start = pd.Timestamp(row.date_added)
        end = pd.Timestamp(row.date_removed) if pd.notna(row.date_removed) else pd.Timestamp.max
        intervals.setdefault(row.ticker, []).append((start, end))
    return intervals


def _tickers_active_on(review_date: pd.Timestamp, intervals: dict) -> set:
    """Tickers whose membership interval covers review_date."""
    return {
        ticker for ticker, ivs in intervals.items()
        if any(start <= review_date <= end for start, end in ivs)
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION — run_backtest
# ─────────────────────────────────────────────────────────────────────────────

def _ib_commission(
    shares:        float,
    price:         float,
    per_share:     float = 0.005,
    min_fee:       float = 1.00,
    max_pct:       float = 0.01,
) -> float:
    """
    IBKR Pro "Fixed" commission schedule for US stocks (as of mid-2026):
    $0.005/share, $1.00 minimum per order, capped at 1% of trade value.
    Applied identically to both the buy and the sell leg of a trade.

    NOTE: IBKR bills in USD; this backtest's capital is tracked in EUR.
    Treated 1:1 here as a simplification for order-of-magnitude cost
    realism — not a precise FX-adjusted figure.
    """
    trade_value = abs(shares) * price
    fee = abs(shares) * per_share
    fee = max(fee, min_fee)
    fee = min(fee, trade_value * max_pct)
    return fee


def run_backtest(
    data:              pd.DataFrame,
    start_date:        str   = "2021-01-01",
    end_date:          str   = None,
    initial_capital:   float = 10_000,

    # Regime — standalone RegimeDetector (VIX+SPY+breadth), NO macro
    regime_detector:   "RegimeDetector | None" = None,

    # Position sizing
    top_n:             int   = 5,       # down from 10 -> 5: concentration
    signal_threshold:  float = 6.5,
    max_hold_days:     int   = 365,     # safety valve only

    # ── Position sizing methodology (Phase 3.2) ────────────────────────────
    # "fixed_fractional" (default, backward-compatible): allocated =
    #   deployable / top_n, same for every new slot regardless of volatility.
    # "atr_risk_based": sizes each position so a stop-out loses exactly
    #   risk_per_trade_pct of capital — shares = (capital * risk_per_trade_pct)
    #   / (stop_atr_mult * ATR). Capped by max_pct_per_position (capital cap)
    #   and liquidity_pct_adv (% of average daily volume) so tight-ATR /
    #   thin-liquidity tickers can't blow past realistic fill sizes.
    sizing_method:         str   = "fixed_fractional",
    risk_per_trade_pct:    float = 0.01,   # 1% of capital risked per trade (atr_risk_based only)
    max_pct_per_position:  float = 0.20,   # capital cap: no position > 20% of capital (atr_risk_based only)
    liquidity_pct_adv:     float = 0.01,   # liquidity cap: no position > 1% of avg daily volume (atr_risk_based only)

    # Stop logic (3-phase)
    stop_atr_mult:     float = 1.5,    # legacy: unused in the 3-phase logic
    target_atr_mult:   float = 4.0,
    use_atr_trail:     bool  = True,   # compat flag (ignored: 3-phase is always active)
    atr_trail_mult:    float = 2.5,    # phase 3: peak - N x ATR
    use_trail_stop:    bool  = False,
    trail_pct:         float = 0.08,
    cash_pct:          float = 0.0,
    # 3-phase stop tuning
    guard_days:        int   = 15,     # phase 1: days without trailing
    hard_floor_atr:    float = 3.0,    # phase 1: max loss = N x ATR (catastrophe stop)
    trail_trigger_atr: float = 1.0,    # phase 3: activates only after +1x ATR profit

    # Signal-based exit
    exit_score_threshold:  float = 5.0,  # close if score < this
    bear_regime_exit:      bool  = True, # close if regime -> bear

    # Benchmark
    benchmark_ticker:  str   = "SPY",

    # Scorer weight overrides (dict) — passed through to the MR adapter
    scorer_weights:    dict  = None,

    # Point-in-time universe (see db.get_membership_table()) — if given,
    # new candidates are restricted to actual index members on the
    # review_date. None = old behavior (backward-compatible).
    membership:        pd.DataFrame | None = None,

    # ── Sector diversification (Phase 3.1) ──────────────────────────────────
    # Opt-in: pass {ticker: sector} (e.g. from sector_lookup.get_sectors_and_caps)
    # to enforce the SAME MAX_PER_SECTOR cap the live scanner already applies.
    # None (default) = old behavior, no sector cap in the backtest — matches
    # every Phase 2 result already on record.
    sectors:            dict | None = None,
    max_per_sector:     int   = MR_MAX_PER_SECTOR,

    # ── Correlation-aware position limits (Phase 3.4) ────────────────────────
    # Opt-in: rejects a candidate whose recent daily-return correlation with
    # any ALREADY-HELD position (or an already-accepted candidate this same
    # pass) exceeds max_pairwise_correlation. Catches same-factor exposure
    # that sector labels alone can miss (e.g. two different-sector mega-caps
    # that move together on rate expectations). None (default) = filter OFF,
    # backward-compatible with every Phase 2/3.1-3.3 result already on record.
    max_pairwise_correlation:  float | None = None,
    correlation_lookback_days: int   = 60,
    correlation_min_obs:       int   = 20,   # min overlapping return obs to trust the correlation

    # ── Execution realism ─────────────────────────────────────────────────
    # Adverse slippage applied to EVERY fill (entries pay more, exits get
    # less) — fixes the artificial "exactly 0.00%" clustering produced by
    # the breakeven/trail floor combined with a theoretical, frictionless
    # stop price.
    slippage_pct:       float = 0.0005,   # 5 bps
    # Score computed on the trading day BEFORE review_date; fill happens
    # at review_date's Open. Removes the same-bar look-ahead where the
    # backtest used to buy at the exact close that generated the signal.
    next_open_entry:    bool  = True,

    # ── Transaction costs (IBKR Pro "Fixed" US-stock schedule) ────────────
    apply_transaction_costs: bool = True,   # if False, no commission is deducted from `capital`
    show_cost_comparison:    bool = True,   # print with/without-cost equity side by side each review date
    commission_per_share:    float = 0.005,
    commission_min:          float = 1.00,
    commission_max_pct:      float = 0.01,
) -> dict:
    """
    Signal-based walk-forward backtest, 100% technical MR scorer.

    Each month: re-score every position. Close if score drops below
    exit_score_threshold, regime turns bear, or stop/target/max_hold is
    hit; otherwise hold. New positions open only if capital is free.
    """
    if end_date is None:
        end_date = data.index[-1].strftime("%Y-%m-%d")

    if sizing_method not in ("fixed_fractional", "atr_risk_based"):
        raise ValueError(f"sizing_method must be 'fixed_fractional' or 'atr_risk_based', got {sizing_method!r}")

    use_regime  = regime_detector is not None
    mode_label  = "v3-mr-signal-exit"
    if use_regime:
        mode_label += "-regime"

    print(f"\n🔄 Backtesting: {start_date} → {end_date}  [{mode_label}]")
    print(f"   Top {top_n} | Threshold: {signal_threshold} | Exit threshold: {exit_score_threshold}")
    if sizing_method == "atr_risk_based":
        print(f"   Sizing: atr_risk_based | risk/trade={risk_per_trade_pct*100:.2f}% | "
              f"max/position={max_pct_per_position*100:.0f}% | liquidity cap={liquidity_pct_adv*100:.2f}% ADV")
    else:
        print(f"   Sizing: fixed_fractional (equal-weight across {top_n} slots)")
    print(f"   3-phase stop: guard={guard_days}d hard={hard_floor_atr}×ATR trail={atr_trail_mult}×ATR trigger=+{trail_trigger_atr}×ATR")
    print(f"   Bear regime exit: {bear_regime_exit}")
    print(
        f"   Execution: slippage={slippage_pct*100:.2f}% | "
        f"entry={'next-day open' if next_open_entry else 'same-day close'} | "
        f"costs={'ON (IBKR-style)' if apply_transaction_costs else 'OFF'}"
    )

    cumulative_costs = 0.0

    available_tickers = data.columns.get_level_values(0).unique().tolist()
    monthly_dates     = _get_monthly_dates(start_date, end_date, data.index)

    # ── Real per-ticker "last traded" date ───────────────────────────────────
    # `available_tickers` is a static snapshot of columns present in `data` —
    # it never shrinks, so it CANNOT detect a ticker that stops trading mid-
    # backtest (delisting/bankruptcy/merger). The actual signal is that the
    # ticker's own row series simply stops. This dict is the dynamic
    # replacement used below to force-close positions exactly when (not long
    # after) the underlying stock stopped trading.
    last_valid_date = {}
    for t in available_tickers:
        t_close = data[t]["Close"].dropna()
        if not t_close.empty:
            last_valid_date[t] = t_close.index.max()

    membership_intervals = _build_membership_intervals(membership) if membership is not None else None
    if membership_intervals is not None:
        print(f"   Point-in-time universe: ACTIVE ({len(membership_intervals)} tickers in membership table)")
    else:
        print(f"   Point-in-time universe: INACTIVE (membership=None — all {len(available_tickers)} tickers are always considered available for new positions)")

    if sectors is not None:
        print(f"   Sector diversification: ACTIVE (max {max_per_sector} positions/sector, {len(sectors)} tickers mapped)")
    else:
        print(f"   Sector diversification: INACTIVE (sectors=None — no cap applied, same as all prior Phase 2 runs)")

    if max_pairwise_correlation is not None:
        print(f"   Correlation filter: ACTIVE (max |corr|={max_pairwise_correlation:.2f}, lookback={correlation_lookback_days}d)")
    else:
        print(f"   Correlation filter: INACTIVE (max_pairwise_correlation=None — no cap applied)")

    # ── Warm-up guard ─────────────────────────────────────────────────────────
    # 252 days (1 year) comfortably covers MR_PERIOD (z-score) and all
    # technical lookback windows.
    warmup_needed = 252
    valid_dates   = [d for d in monthly_dates if (data.index < d).sum() >= warmup_needed]
    skipped = len(monthly_dates) - len(valid_dates)
    if skipped:
        print(f"   ⚠️  {skipped} date(s) skipped (warm-up {warmup_needed} rows)")
    monthly_dates = valid_dates

    print(f"   Monthly review dates: {len(monthly_dates)}")
    print(f"   Universe: {len(available_tickers)} tickers\n")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    benchmark_return = 0.0
    try:
        start_ts = pd.Timestamp(start_date)
        end_ts   = pd.Timestamp(end_date)
        bm_raw = yf.download(
            benchmark_ticker,
            start       = (start_ts - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            end         = (end_ts   + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            progress    = False,
            auto_adjust = True,
        )
        if bm_raw is not None and not bm_raw.empty:
            if isinstance(bm_raw.columns, pd.MultiIndex):
                bm_raw.columns = bm_raw.columns.get_level_values(0)
            bm_raw.index = pd.to_datetime(bm_raw.index).tz_localize(None)
            bm_range = bm_raw[(bm_raw.index >= start_ts) & (bm_raw.index <= end_ts)]
            if len(bm_range) >= 2:
                bm_s = bm_range["Close"].iloc[0]
                bm_e = bm_range["Close"].iloc[-1]
                benchmark_return = round((bm_e - bm_s) / bm_s * 100, 2)
                print(f"   {benchmark_ticker}: {bm_s:.2f} → {bm_e:.2f}  ({benchmark_return:+.2f}%)\n")
    except Exception as e:
        print(f"   ⚠️  Benchmark ({benchmark_ticker}) failed: {e}\n")

    # ── State ─────────────────────────────────────────────────────────────────
    capital      = initial_capital
    portfolio    = {}          # ticker → position dict
    equity_curve = []
    all_trades   = []
    last_review  = None        # for intra-month stop tracking

    def _close_out(pos, ticker, exit_date, raw_exit_price, reason):
        """
        Centralised exit accounting: applies adverse slippage to the raw
        exit price, charges the IBKR-style commission, updates `capital`
        (only if apply_transaction_costs) and always tracks the true cost
        in `cumulative_costs` so the with/without-cost comparison stays
        accurate regardless of the toggle.
        """
        nonlocal capital, cumulative_costs
        exit_price  = raw_exit_price * (1 - slippage_pct)   # always worse than quoted
        shares      = pos["shares"]
        trade_value = shares * exit_price
        commission  = _ib_commission(shares, exit_price, commission_per_share, commission_min, commission_max_pct)
        cumulative_costs += commission

        capital += trade_value
        if apply_transaction_costs:
            capital -= commission

        pnl     = (exit_price - pos["entry_price"]) * shares
        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        return _make_trade_record(pos, ticker, exit_date, exit_price, pnl, pnl_pct, reason)

    # ── Monthly review loop ───────────────────────────────────────────────────
    for i, review_date in enumerate(monthly_dates):

        # ── Regime ───────────────────────────────────────────────────────────
        # Standalone RegimeDetector (VIX+SPY+breadth) — no macro dependency.
        # No circuit breaker anymore (that was macro-bridge-specific) —
        # entry_allowed is always True, regime only affects
        # top_n/threshold/stops/cash_pct.
        entry_allowed = True
        if use_regime:
            cfg             = regime_detector.get_config(review_date)
            cur_threshold   = cfg.signal_threshold
            cur_top_n       = cfg.top_n
            cur_stop_mult   = cfg.stop_atr_mult
            cur_target_mult = cfg.target_atr_mult
            cur_max_hold    = cfg.max_hold_days
            cur_cash_pct    = cfg.cash_pct
            regime_name     = cfg.name
            # if regime -> bear -> force exit if bear_regime_exit=True
            cur_bear_exit   = bear_regime_exit and (regime_name == "bear")
            cur_use_atr_trail = use_atr_trail
            cur_use_pct_trail = False
        else:
            cur_threshold   = signal_threshold
            cur_top_n       = top_n
            cur_stop_mult   = stop_atr_mult
            cur_target_mult = target_atr_mult
            cur_max_hold    = max_hold_days
            cur_cash_pct    = cash_pct
            regime_name     = "fixed"
            cur_bear_exit   = False
            cur_use_atr_trail = use_atr_trail
            cur_use_pct_trail = False

        # ── 1. Intra-month stop/target check ─────────────────────────────────
        # For each open position, check whether stop or target was hit
        # between the previous review and today.
        period_start = last_review if last_review else pd.Timestamp(start_date)
        to_close_stops = []

        for ticker, pos in portfolio.items():
            lvd = last_valid_date.get(ticker)

            # Ticker already stopped trading before this period even started
            # (previous period ended past its last valid date) — force-close
            # NOW at its true last price/date instead of silently "holding"
            # a dead position until end-of-backtest.
            if lvd is not None and lvd < period_start:
                ticker_df = data[ticker].dropna()
                exit_price = ticker_df["Close"].iloc[-1]
                to_close_stops.append((ticker, "delisted", lvd, exit_price))
                continue
            if lvd is None:
                # No valid Close data at all for this ticker — shouldn't
                # normally happen for an open position, but fail safe.
                to_close_stops.append((ticker, "delisted", review_date, pos["entry_price"]))
                continue

            ticker_df   = data[ticker].dropna()
            period_data = ticker_df[
                (ticker_df.index > period_start) &
                (ticker_df.index <= review_date)
            ]

            if period_data.empty:
                continue

            reason, exit_price, exit_date = _check_exit_daily(
                pos,
                period_data,
                max_hold_days = pos.get("max_hold_days", cur_max_hold),
                atr_trail_mult = atr_trail_mult,
            )

            if reason:
                to_close_stops.append((ticker, reason, exit_date, exit_price))
            elif lvd <= review_date and period_data.index.max() >= lvd:
                # This period_data window reaches all the way to the
                # ticker's last-ever trading day and no stop/target fired —
                # the stock stopped trading (delisting/merger) right here.
                # Force-close at the real last price instead of letting the
                # position sit "open" and frozen for the rest of the
                # backtest (artificial capital lock + hidden P&L).
                last_close = period_data["Close"].iloc[-1]
                to_close_stops.append((ticker, "delisted", period_data.index[-1], last_close))

        # Close from stop/target/max_hold
        for ticker, reason, exit_date, exit_price in to_close_stops:
            if ticker not in portfolio:
                continue
            pos = portfolio.pop(ticker)
            all_trades.append(_close_out(pos, ticker, exit_date, exit_price, reason))

        # ── 2. Signal deterioration check ────────────────────────────────────
        # Re-score existing positions. Close ONLY if score < threshold.
        # If a winner still has a good score -> HOLD.
        to_close_signal = []

        for ticker, pos in portfolio.items():
            if ticker not in available_tickers:
                continue

            # Bear regime exit: liquidate all
            if cur_bear_exit:
                ticker_df = data[ticker].dropna()
                avail = ticker_df[ticker_df.index <= review_date]
                exit_price = avail["Close"].iloc[-1] if not avail.empty else pos["entry_price"]
                to_close_signal.append((ticker, "bear_regime_exit", review_date, exit_price))
                continue

            # Score check
            ticker_df  = data[ticker].dropna()
            re_scored  = _compute_mr_score_at_date(
                ticker_df, review_date,
                stop_atr_mult   = cur_stop_mult,
                target_atr_mult = cur_target_mult,
                atr_trail_mult  = atr_trail_mult,
                weights         = scorer_weights,
            )

            if re_scored is None:
                continue

            current_score = re_scored["signal_score"]

            # Score OK -> hold, update stop if needed
            if current_score >= exit_score_threshold:
                # Update the ATR trailing stop based on the new ATR
                if cur_use_atr_trail and re_scored["atr"] > 0:
                    pos["atr_at_entry"] = re_scored["atr"]  # rolling ATR update
                continue

            # Score dropped -> close
            avail = ticker_df[ticker_df.index <= review_date]
            exit_price = avail["Close"].iloc[-1] if not avail.empty else pos["entry_price"]
            to_close_signal.append((ticker, "score_deterioration", review_date, exit_price))

        # Close for signal deterioration
        for ticker, reason, exit_date, exit_price in to_close_signal:
            if ticker not in portfolio:
                continue
            pos = portfolio.pop(ticker)
            all_trades.append(_close_out(pos, ticker, exit_date, exit_price, reason))

        # ── 3. Scan universe for new opportunities ───────────────────────────
        # New positions open ONLY if:
        #   a) fewer than top_n open positions exist
        #   b) free capital is available
        # (No circuit breaker anymore — that was macro-bridge-specific.)

        slots_available = cur_top_n - len(portfolio)
        deployable      = capital * (1 - cur_cash_pct)

        # Point-in-time candidates: with a membership table, only actual
        # index members on review_date are eligible for NEW positions.
        # Existing positions are unaffected (see steps 1-2) — leaving the
        # index doesn't force a sale, only delisting does.
        if membership_intervals is not None:
            candidate_universe = _tickers_active_on(review_date, membership_intervals) & set(available_tickers)
        else:
            candidate_universe = available_tickers

        n_new_opens = 0
        if slots_available > 0 and deployable > 100 and entry_allowed:
            scored = []
            for ticker in candidate_universe:
                if ticker in portfolio:
                    continue
                try:
                    ticker_df = data[ticker].dropna()

                    if next_open_entry:
                        # Score using the trading day BEFORE review_date —
                        # you can only know yesterday's close-based signal
                        # before today's session opens.
                        prior_dates = ticker_df.index[ticker_df.index < review_date]
                        if prior_dates.empty:
                            continue
                        score_date = prior_dates[-1]
                    else:
                        score_date = review_date

                    result = _compute_mr_score_at_date(
                        ticker_df, score_date,
                        stop_atr_mult   = cur_stop_mult,
                        target_atr_mult = cur_target_mult,
                        atr_trail_mult  = atr_trail_mult,
                        weights         = scorer_weights,
                    )
                    if result and result["signal_score"] >= cur_threshold:
                        scored.append((ticker, result))
                except Exception:
                    continue

            scored.sort(key=lambda x: x[1]["signal_score"], reverse=True)

            # ── Diversified selection (sector cap Phase 3.1 + correlation
            # filter Phase 3.4, applied together) ────────────────────────────
            correlation_active = max_pairwise_correlation is not None

            sector_counts: dict[str, int] = {}
            if sectors is not None:
                for pos in portfolio.values():
                    sec = pos.get("sector", "Unknown")
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1

            # Tickers whose correlation a new candidate must clear: currently
            # HELD positions, growing as candidates get accepted this pass
            # (so two highly-correlated NEW candidates can't both slip in).
            corr_check_tickers = list(portfolio.keys()) if correlation_active else []
            _returns_cache: dict = {}

            def _recent_returns(ticker: str):
                if ticker in _returns_cache:
                    return _returns_cache[ticker]
                try:
                    closes = data[ticker]["Close"].dropna()
                    closes = closes[closes.index <= review_date]
                    if len(closes) < correlation_lookback_days + 1:
                        ret = None
                    else:
                        ret = closes.iloc[-(correlation_lookback_days + 1):].pct_change().dropna()
                except Exception:
                    ret = None
                _returns_cache[ticker] = ret
                return ret

            candidates = []
            if sectors is None and not correlation_active:
                candidates = scored[:slots_available]
            else:
                for ticker, result in scored:
                    if len(candidates) >= slots_available:
                        break

                    if sectors is not None:
                        sec = sectors.get(ticker, "Unknown")
                        if sector_counts.get(sec, 0) >= max_per_sector:
                            continue

                    if correlation_active:
                        cand_ret = _recent_returns(ticker)
                        too_correlated = False
                        if cand_ret is not None:
                            for held_t in corr_check_tickers:
                                held_ret = _recent_returns(held_t)
                                if held_ret is None:
                                    continue
                                aligned = pd.concat([cand_ret, held_ret], axis=1).dropna()
                                if len(aligned) >= correlation_min_obs:
                                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                                    if pd.notna(corr) and abs(corr) > max_pairwise_correlation:
                                        too_correlated = True
                                        break
                        if too_correlated:
                            continue

                    candidates.append((ticker, result))
                    if sectors is not None:
                        sec = sectors.get(ticker, "Unknown")
                        sector_counts[sec] = sector_counts.get(sec, 0) + 1
                    if correlation_active:
                        corr_check_tickers.append(ticker)

            if candidates:
                per_slot_capital = deployable / cur_top_n
                for ticker, result in candidates:
                    ticker_df = data[ticker].dropna()

                    if next_open_entry:
                        # Earliest realistic fill: review_date's Open,
                        # one full session after the score was measured.
                        if review_date not in ticker_df.index or "Open" not in ticker_df.columns:
                            continue
                        raw_entry_price = ticker_df.loc[review_date, "Open"]
                    else:
                        raw_entry_price = result["price"]

                    if raw_entry_price <= 0 or pd.isna(raw_entry_price):
                        continue

                    # Adverse slippage: always pay a touch more than quoted.
                    entry_price = raw_entry_price * (1 + slippage_pct)
                    atr_val     = result["atr"]

                    if sizing_method == "atr_risk_based":
                        # 1) Risk-based: size so a stop-out costs exactly
                        #    risk_per_trade_pct of current capital.
                        stop_distance = cur_stop_mult * atr_val
                        if stop_distance <= 0:
                            continue
                        risk_dollars       = capital * risk_per_trade_pct
                        risk_based_shares  = risk_dollars / stop_distance

                        # 2) Capital cap: no single position > max_pct_per_position
                        #    of capital, however tight the stop is.
                        capital_cap_shares = (max_pct_per_position * capital) / entry_price

                        # 3) Liquidity cap: no position > liquidity_pct_adv of
                        #    the ticker's average daily volume (skip cap if
                        #    avg_volume unavailable rather than block the trade).
                        avg_vol = result.get("avg_volume")
                        liquidity_cap_shares = (
                            liquidity_pct_adv * avg_vol if avg_vol else float("inf")
                        )

                        shares = min(risk_based_shares, capital_cap_shares, liquidity_cap_shares)
                        # Still can't spend more than what's actually free this pass.
                        shares = min(shares, deployable / entry_price) if deployable > 0 else 0
                        if shares <= 0:
                            continue
                    else:
                        # fixed_fractional (default, backward-compatible):
                        # equal dollar allocation across the top_n slots.
                        allocated = min(deployable / cur_top_n, capital * (1 - cur_cash_pct))
                        if allocated <= 0:
                            continue
                        shares = allocated / entry_price

                    trade_value = shares * entry_price
                    commission  = _ib_commission(shares, entry_price, commission_per_share, commission_min, commission_max_pct)
                    cumulative_costs += commission

                    capital -= trade_value
                    if apply_transaction_costs:
                        capital -= commission
                    n_new_opens += 1

                    portfolio[ticker] = {
                        "shares":            shares,
                        "entry_price":       entry_price,
                        "sector":            sectors.get(ticker, "Unknown") if sectors is not None else "Unknown",
                        # Stop/target recentred on the ACTUAL fill price
                        # (result["price"] was yesterday's close, not what
                        # we paid) — ATR itself is still the point-in-time
                        # value from score_date.
                        "stop_loss":         round(entry_price - cur_stop_mult   * atr_val, 2),
                        "atr_trail_stop":    round(entry_price - atr_trail_mult  * atr_val, 2),
                        "target_2":          round(entry_price + cur_target_mult * atr_val, 2),
                        "peak_price":        entry_price,
                        "atr_at_entry":      atr_val,
                        "entry_date":        review_date,
                        "signal_score":      result["signal_score"],
                        "regime":            regime_name,
                        "use_trail_stop":    cur_use_pct_trail,
                        "use_atr_trail":     cur_use_atr_trail,
                        "trail_pct":         trail_pct,
                        "max_hold_days":     cur_max_hold,
                        # 3-phase stop params
                        "guard_days":        guard_days,
                        "hard_floor_atr":    hard_floor_atr,
                        "trail_trigger_atr": trail_trigger_atr,
                    }

        # ── 4. Equity snapshot ────────────────────────────────────────────────
        # Fallback for genuine data gaps (holiday quirks, not delisting —
        # true delistings are force-closed in step 1 before we get here)
        # uses the last KNOWN price, not the stale entry_price, so a gap
        # doesn't artificially erase/hide any P&L already accrued.
        def _mark_price(t, pos):
            tdf = data[t]["Close"]
            if review_date in tdf.index and pd.notna(tdf.loc[review_date]):
                return tdf.loc[review_date]
            prior = tdf[tdf.index <= review_date].dropna()
            return prior.iloc[-1] if not prior.empty else pos["entry_price"]

        port_value = capital + sum(
            pos["shares"] * _mark_price(t, pos)
            for t, pos in portfolio.items()
        )

        # With/without-cost shadow value: since the same shares/prices are
        # used regardless of the toggle, the "other" scenario's value is
        # just the official one adjusted by cumulative_costs so far.
        if apply_transaction_costs:
            port_value_with_costs    = port_value
            port_value_without_costs = port_value + cumulative_costs
        else:
            port_value_without_costs = port_value
            port_value_with_costs    = port_value - cumulative_costs

        equity_curve.append({
            "date":                     review_date,
            "portfolio_value":          port_value,
            "portfolio_value_with_costs":    port_value_with_costs,
            "portfolio_value_without_costs": port_value_without_costs,
            "regime":          regime_name,
            "n_positions":     len(portfolio),
            "n_new":           n_new_opens,
            # Phase 3.3 — exposure audit: how much of the portfolio is
            # actually invested right now vs sitting in cash. Not enforced
            # (cash_pct only gates NEW entries, doesn't trim winners back
            # to target), but visible here so drift is auditable.
            "pct_invested":    round((port_value - capital) / port_value * 100, 1) if port_value > 0 else 0.0,
            "cash_pct_target": round(cur_cash_pct * 100, 1),
        })

        freeze_tag = "" if entry_allowed else " 🔴CB"
        cost_note = ""
        if show_cost_comparison:
            other_label = "no_costs" if apply_transaction_costs else "with_costs"
            other_value = port_value_without_costs if apply_transaction_costs else port_value_with_costs
            cost_note = f" | {other_label}=€{other_value:,.0f}"
        print(
            f"  📅 {review_date.date()} [{regime_name:<12}] "
            f"open={len(portfolio)} new={n_new_opens}{freeze_tag} "
            f"capital=€{capital:,.0f} total=€{port_value:,.0f}{cost_note}"
        )

        last_review = review_date

    # ── 5. Close remaining positions at end_date ─────────────────────────────
    end_ts    = pd.Timestamp(end_date)
    last_date = data.index[data.index <= end_ts][-1] if (data.index <= end_ts).any() else data.index[-1]

    for ticker, pos in list(portfolio.items()):
        if ticker not in available_tickers:
            continue
        ticker_df  = data[ticker].dropna()
        avail      = ticker_df[ticker_df.index <= last_date]
        if avail.empty:
            continue
        exit_price = avail["Close"].iloc[-1]
        all_trades.append(_close_out(pos, ticker, last_date, exit_price, "end_of_backtest"))

    # ── Metrics — single source of truth via backtest_metrics.py (Phase 2.9) ──
    # Same function used by out_of_sample.py, monte_carlo.py, cost_stress_test.py,
    # parameter_sweep.py, equal_weight_benchmark.py — so every number here is
    # directly comparable across the whole roadmap, not just within this file.
    trades_df = pd.DataFrame(all_trades)
    equity_df = pd.DataFrame(equity_curve).set_index("date")

    metrics = compute_standard_metrics(
        trades_df       = trades_df,
        equity_df       = equity_df,
        initial_capital = initial_capital,
        final_capital   = capital,
        start_date      = start_date,
        end_date        = end_date,
        periods_per_year = 12,   # monthly review cadence
    )

    reliability = _analyze_signal_reliability(trades_df)

    if apply_transaction_costs:
        final_capital_with_costs    = capital
        final_capital_without_costs = capital + cumulative_costs
    else:
        final_capital_without_costs = capital
        final_capital_with_costs    = capital - cumulative_costs

    return {
        "summary": {
            "version":          "v2",
            "start_date":       start_date,
            "end_date":         end_date,
            "mode":             mode_label,
            "initial_capital":  initial_capital,
            "final_capital":    round(capital, 2),
            "benchmark_ticker": benchmark_ticker,
            "benchmark_return": round(benchmark_return, 2),
            "outperformance":   round(metrics["total_return"] - benchmark_return, 2),
            # Position sizing (Phase 3.2)
            "sizing_method":         sizing_method,
            "risk_per_trade_pct":    risk_per_trade_pct   if sizing_method == "atr_risk_based" else None,
            "max_pct_per_position":  max_pct_per_position if sizing_method == "atr_risk_based" else None,
            "liquidity_pct_adv":     liquidity_pct_adv    if sizing_method == "atr_risk_based" else None,
            # Sector diversification (Phase 3.1)
            "sector_cap_active":     sectors is not None,
            "max_per_sector":        max_per_sector if sectors is not None else None,
            # Correlation-aware limits (Phase 3.4)
            "correlation_filter_active":  max_pairwise_correlation is not None,
            "max_pairwise_correlation":   max_pairwise_correlation,
            "correlation_lookback_days":  correlation_lookback_days if max_pairwise_correlation is not None else None,
            # Standardized metrics (Phase 2.9) — total_return, annual_return,
            # max_drawdown, sharpe/sortino/calmar_ratio, win_rate, avg_win,
            # avg_loss, profit_factor, expectancy_pct, expectancy_abs, n_trades
            **metrics,
            # v2-specific
            "exit_score_threshold": exit_score_threshold,
            "bear_regime_exit":     bear_regime_exit,
            "use_atr_trail":        use_atr_trail,
            "atr_trail_mult":       atr_trail_mult,
            "guard_days":           guard_days,
            "hard_floor_atr":       hard_floor_atr,
            "trail_trigger_atr":    trail_trigger_atr,
            # Execution realism
            "slippage_pct":              slippage_pct,
            "next_open_entry":           next_open_entry,
            "apply_transaction_costs":   apply_transaction_costs,
            "total_transaction_costs":   round(cumulative_costs, 2),
            "final_capital_with_costs":    round(final_capital_with_costs, 2),
            "final_capital_without_costs": round(final_capital_without_costs, 2),
        },
        "trades":      trades_df,
        "equity":      equity_df,
        "reliability": reliability,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: trade record builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_trade_record(
    pos:        dict,
    ticker:     str,
    exit_date:  pd.Timestamp,
    exit_price: float,
    pnl:        float,
    pnl_pct:    float,
    reason:     str,
) -> dict:
    entry_date = pos["entry_date"]
    hold_days  = (exit_date - entry_date).days if hasattr(exit_date, "date") else 0
    return {
        "ticker":       ticker,
        "entry_date":   entry_date.date() if hasattr(entry_date, "date") else entry_date,
        "exit_date":    exit_date.date()  if hasattr(exit_date,  "date") else exit_date,
        "entry_price":  pos["entry_price"],
        "exit_price":   exit_price,
        "shares":       pos["shares"],
        "pnl":          round(pnl, 2),
        "pnl_pct":      round(pnl_pct, 2),
        "exit_reason":  reason,
        "signal_score": pos.get("signal_score", 0),
        "hold_days":    hold_days,
        "regime":       pos.get("regime", "fixed"),
        "use_trail":    pos.get("use_trail_stop", False) or pos.get("use_atr_trail", False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT
# ─────────────────────────────────────────────────────────────────────────────

def print_backtest_report(results: dict) -> None:
    s = results["summary"]
    t = results["trades"]
    r = results.get("reliability", {})

    bm  = s.get("benchmark_ticker", "Benchmark")
    pad = max(1, 10 - len(bm))

    print(f"\n{'═'*60}")
    print(f"  BACKTEST RESULTS v2  [{s.get('mode','').upper()}]")
    print(f"  {s['start_date']} → {s['end_date']}")
    print(f"  3-phase stop: guard={s.get('guard_days')}d | "
          f"hard={s.get('hard_floor_atr')}×ATR | "
          f"trail={s.get('atr_trail_mult')}×ATR | "
          f"trigger=+{s.get('trail_trigger_atr')}×ATR")
    print(f"  Bear exit: {s.get('bear_regime_exit')} | "
          f"Exit score threshold: {s.get('exit_score_threshold')}")
    if s.get("sizing_method") == "atr_risk_based":
        print(f"  Sizing: atr_risk_based | risk/trade={s.get('risk_per_trade_pct', 0)*100:.2f}% | "
              f"max/position={s.get('max_pct_per_position', 0)*100:.0f}% | "
              f"liquidity cap={s.get('liquidity_pct_adv', 0)*100:.2f}% ADV")
    else:
        print(f"  Sizing: {s.get('sizing_method', 'fixed_fractional')}")
    print(f"{'═'*60}")

    print(f"\n  💰 PERFORMANCE")
    print(f"    Initial capital:   €{s['initial_capital']:>10,.2f}")
    print(f"    Final capital:     €{s['final_capital']:>10,.2f}")
    print(f"    Total return:      {s['total_return']:>+10.2f}%")
    print(f"    Annual return:     {s['annual_return']:>+10.2f}%")
    print(f"    {bm} return:{' '*pad}{s['benchmark_return']:>+10.2f}%")
    print(f"    Outperformance:    {s['outperformance']:>+10.2f}%")

    print(f"\n  📊 TRADE STATISTICS")
    print(f"    Total trades:      {s['n_trades']:>10}")
    print(f"    Win rate:          {s['win_rate']:>10.1f}%")
    print(f"    Avg win:           {s['avg_win']:>+10.2f}%")
    print(f"    Avg loss:          {s['avg_loss']:>+10.2f}%")
    print(f"    Profit factor:     {s['profit_factor']:>10.2f}")

    print(f"\n  ⚠️  RISK")
    print(f"    Max drawdown:      {s['max_drawdown']:>+10.2f}%")
    print(f"    Sharpe ratio:      {s['sharpe_ratio']:>10.2f}")
    print(f"    Sortino ratio:     {s.get('sortino_ratio', 0):>10.2f}")
    print(f"    Calmar ratio:      {s.get('calmar_ratio', 0):>10.2f}")
    print(f"    Expectancy:        {s.get('expectancy_pct', 0):>+9.2f}%")

    if "total_transaction_costs" in s:
        print(f"\n  💸 EXECUTION & COSTS")
        print(f"    Slippage/fill:     {s.get('slippage_pct', 0)*100:>9.2f}%")
        print(f"    Entry timing:      {'next-day open' if s.get('next_open_entry') else 'same-day close':>10}")
        print(f"    Transaction costs: {'ON (IBKR-style)' if s.get('apply_transaction_costs') else 'OFF':>10}")
        print(f"    Total costs paid:  €{s['total_transaction_costs']:>10,.2f}")
        print(f"    Final (with costs):    €{s['final_capital_with_costs']:>10,.2f}")
        print(f"    Final (without costs): €{s['final_capital_without_costs']:>10,.2f}")

    if len(t) > 0:
        print(f"\n  🚪 EXIT REASONS")
        for reason, count in t["exit_reason"].value_counts().items():
            avg = t[t["exit_reason"] == reason]["pnl_pct"].mean()
            print(f"    {reason:<28} {count:>4}  ({count/len(t)*100:.1f}%)  avg: {avg:+.1f}%")

    # Avg hold time
    if "hold_days" in t.columns and len(t) > 0:
        print(f"\n  ⏱  HOLD TIME")
        print(f"    Avg hold:          {t['hold_days'].mean():>10.1f} days")
        print(f"    Median hold:       {t['hold_days'].median():>10.1f} days")
        print(f"    Max hold:          {t['hold_days'].max():>10.1f} days")

    if len(t) > 0 and "regime" in t.columns:
        print(f"\n  🌐 PERFORMANCE BY REGIME")
        print(f"    {'Regime':<12} {'Trades':>6} {'Win%':>6} {'AvgPnL':>8} {'AvgWin':>8} {'AvgLoss':>9}")
        print(f"    {'-'*55}")
        for regime in ["bull", "neutral", "bear", "fixed"]:
            key = f"regime_{regime}"
            if key not in r:
                continue
            d = r[key]
            print(
                f"    {regime:<12} {d['n_trades']:>6} {d['win_rate']:>5.1f}%"
                f" {d['avg_pnl']:>+8.2f}% {d['avg_win']:>+8.2f}% {d['avg_loss']:>+9.2f}%"
            )

    score_keys = [k for k in r if k.startswith("score_")]
    if score_keys:
        print(f"\n  🎯 SIGNAL SCORE RELIABILITY")
        print(f"    {'Score bucket':<12} {'Trades':>6} {'Win%':>6} {'AvgPnL':>8}")
        print(f"    {'-'*38}")
        for key in score_keys:
            bucket = key.replace("score_", "")
            d = r[key]
            print(f"    {bucket:<12} {d['n_trades']:>6} {d['win_rate']:>5.1f}% {d['avg_pnl']:>+8.2f}%")

    if len(t) > 0:
        cols = ["ticker", "entry_date", "exit_date", "pnl_pct", "hold_days", "exit_reason", "regime"]
        cols = [c for c in cols if c in t.columns]
        print(f"\n  🏆 BEST TRADES")
        print(t.nlargest(5, "pnl_pct")[cols].to_string(index=False))
        print(f"\n  💀 WORST TRADES")
        print(t.nsmallest(5, "pnl_pct")[cols].to_string(index=False))

    print(f"\n{'═'*60}\n")