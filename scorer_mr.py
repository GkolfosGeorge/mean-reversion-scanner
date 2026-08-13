# scorer_mr.py
"""
Mean-Reversion Scorer — Contrarian Style (100% Technical)
──────────────────────────────────────────────────────────
Replaces scorer.py with logic aligned to the contrarian mean-reversion style.

Architecture:
  HARD FILTERS   -> exclusion if any fails (RSI, volume)
  SCORING (100%) -> RSI 35% | Bollinger 25% | Mean Reversion 20% | PCR 20%
  MACRO SCORE    -> multiplier on position sizing (external, future)

  ⚠️ NO dependency on fundamentals data — not even as a hard filter.
  The fundamental-weighted scorer is a separate project (roadmap) until
  enough fundamentals history accumulates to be statistically reliable.
  `sector` (only for portfolio diversification) comes from the independent
  sector_lookup.py — it isn't fundamental analysis, just sector
  classification.

Usage:
    from scorer_mr import compute_scores, print_mr_report, to_dataframe
    from sector_lookup import get_sectors_and_caps

    sectors, market_caps = get_sectors_and_caps(tickers)

    scored = compute_scores(
        data        = data,           # OHLCV MultiIndex DataFrame
        sectors     = sectors,        # dict {ticker: sector} from sector_lookup.py
        options_df  = scanner.to_dataframe(),  # optional — from OptionsScanner
        market_caps = market_caps,    # dict {ticker: market_cap} — optional
        cap_tier    = 5,               # optional — 1=Small...5=Ultra Mega-cap (see CAP_TIERS)
        top_n       = 20,
    )
    print_mr_report(scored)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# ── Helpers from signals.py ───────────────────────────────────────────────────
# Importing new indicators from the central computation module.
try:
    # Relative import — when running as part of the trading package
    from .signals import (
        _compute_stoch_rsi,
        _compute_williams_r,
        _compute_atr_percentile,
    )
except ImportError:
    # Absolute import — when running standalone (testing, notebook)
    from signals import (
        _compute_stoch_rsi,
        _compute_williams_r,
        _compute_atr_percentile,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# ── Scoring weights ───────────────────────────────────────────────────────────
# Must sum to 1.0 — change only from CONFIG cell in notebook.
W_RSI      = 0.30   # RSI (plain) — primary oversold filter
W_BB       = 0.20   # Bollinger %B — price vs bands
W_MR       = 0.15   # Z-Score — statistical mean reversion
W_PCR      = 0.15   # Put/Call Ratio — contrarian sentiment
W_STOCHRSI = 0.10   # Stochastic RSI — precise entry timing
W_WILLIAMS = 0.10   # Williams %R — price position in range

# Weights without PCR (if no options data available)
# Redistributed proportionally when PCR unavailable
W_RSI_NO_PCR      = 0.355
W_BB_NO_PCR       = 0.235
W_MR_NO_PCR       = 0.175
W_STOCHRSI_NO_PCR = 0.117
W_WILLIAMS_NO_PCR = 0.118

# ── Hard filters ──────────────────────────────────────────────────────────────
RSI_MAX            = 50.0    # RSI > 50 -> excluded
MIN_AVG_VOLUME     = 500_000 # daily average volume
EARNINGS_DAYS      = 7       # days before earnings -> excluded

# ── Technical params ──────────────────────────────────────────────────────────
RSI_PERIOD    = 14
BB_PERIOD     = 20
BB_STD        = 2.0
MR_PERIOD     = 252          # 1 year for z-score mean reversion
ATR_PERIOD    = 14
VOLUME_PERIOD = 20

# ── New indicator params ───────────────────────────────────────────────────────
STOCH_RSI_PERIOD = 14        # Lookback for StochRSI
STOCH_SMOOTH_K   = 3         # Smoothing %K
STOCH_SMOOTH_D   = 3         # Smoothing %D (signal line)
WILLIAMS_PERIOD  = 14        # Lookback for Williams %R
ATR_PCT_LOOKBACK = 252       # History for ATR percentile (1 year)

# ── Options filters (PCR reliability) ────────────────────────────────────────
PCR_MIN_CALL_VOLUME  = 200   # minimum call volume for reliable PCR
PCR_MIN_TOTAL_VOLUME = 500   # minimum total options volume

# ── Output ────────────────────────────────────────────────────────────────────
TOP_N          = 20
MAX_PER_SECTOR = 4
MIN_SCORE      = 4.0         # below this, excluded

# ── Market Cap Tiers ──────────────────────────────────────────────────────────
# Configurable from the CONFIG cell. Bounds in USD. Calibrated to the S&P 500's
# own cap range (~$5-10B smallest to ~$3-4T largest), NOT the general US market
# (where "small cap" means something entirely different).
# Freely adjust in the notebook — these are just sensible defaults.
CAP_TIERS = {
    1: (0,                15_000_000_000),   # Small-cap   (relative to S&P500)
    2: (15_000_000_000,   50_000_000_000),   # Mid-cap
    3: (50_000_000_000,  150_000_000_000),   # Large-cap
    4: (150_000_000_000, 500_000_000_000),   # Mega-cap
    5: (500_000_000_000, float("inf")),      # Ultra Mega-cap (AAPL, MSFT, NVDA...)
}
CAP_TIER_LABELS = {
    1: "Small-cap",
    2: "Mid-cap",
    3: "Large-cap",
    4: "Mega-cap",
    5: "Ultra Mega-cap",
}


def _cap_tier_label(market_cap: Optional[float], cap_tiers: dict) -> str:
    """Returns the tier label for a given market cap, or 'Unknown' if missing."""
    if market_cap is None or pd.isna(market_cap):
        return "Unknown"
    for tier_num, (lo, hi) in cap_tiers.items():
        if lo <= market_cap < hi:
            return CAP_TIER_LABELS.get(tier_num, f"Tier {tier_num}")
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = avg_loss.replace(0, np.nan)
    rs       = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).fillna(50)


def _bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    sma    = close.rolling(period).mean()
    sigma  = close.rolling(period).std()
    upper  = sma + std * sigma
    lower  = sma - std * sigma
    pct_b  = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, lower, pct_b, sma


def _atr(high, low, close, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


# ─────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS (scaled, 0-10)
# ─────────────────────────────────────────────────────────────────────────────

def _score_rsi(rsi_val: float) -> float:
    """
    Scaled RSI score for mean-reversion.
    The more oversold, the higher the score.
    RSI > 50 -> hard filter (normally never reaches here).
    """
    if rsi_val <= 20:
        return 10.0   # extreme oversold
    elif rsi_val <= 25:
        return 9.0
    elif rsi_val <= 30:
        return 8.0
    elif rsi_val <= 35:
        return 7.0
    elif rsi_val <= 40:
        return 5.5
    elif rsi_val <= 45:
        return 3.5
    elif rsi_val <= 50:
        return 2.0
    else:
        return 0.0    # excluded by hard filter


def _score_bollinger(pct_b: float) -> float:
    """
    Scaled Bollinger %B score.
    pct_b = 0 -> at the lower band (oversold)
    pct_b = 1 -> at the upper band (overbought)
    Negative pct_b -> below the lower band (extreme oversold)
    """
    if pd.isna(pct_b):
        return 5.0   # neutral if missing

    if pct_b < -0.1:
        return 10.0  # below lower band — extreme
    elif pct_b < 0.0:
        return 9.0   # right at the lower band
    elif pct_b < 0.10:
        return 8.0   # very close to the lower band
    elif pct_b < 0.20:
        return 6.5
    elif pct_b < 0.35:
        return 5.0   # neutral zone
    elif pct_b < 0.65:
        return 3.0   # middle — not interesting
    elif pct_b < 0.80:
        return 2.0
    else:
        return 1.0   # near the upper band — overbought


def _score_mean_reversion(z_score: float) -> float:
    """
    Scaled Mean Reversion score based on z-score.
    z < 0 -> below the mean -> potential reversion up -> bullish
    The more negative, the more oversold.
    """
    if pd.isna(z_score):
        return 5.0   # neutral if missing

    if z_score < -2.5:
        return 10.0  # extremely oversold
    elif z_score < -2.0:
        return 9.0
    elif z_score < -1.5:
        return 7.5
    elif z_score < -1.0:
        return 6.0
    elif z_score < -0.5:
        return 4.0
    elif z_score < 0.0:
        return 2.5   # slightly below — a little interesting
    else:
        return 1.0   # above the mean — not interesting


def _score_pcr(pcr_volume: float, pcr_oi: Optional[float] = None) -> float:
    """
    Scaled PCR score for contrarian sentiment.
    High PCR = fear = contrarian bullish signal.
    Uses PCR_OI as confirmation.
    """
    if pcr_volume is None or pd.isna(pcr_volume):
        return None  # None -> redistributed to the other weights

    # Base score from PCR volume
    if pcr_volume >= 3.0:
        base = 10.0
    elif pcr_volume >= 2.0:
        base = 9.0
    elif pcr_volume >= 1.5:
        base = 8.0
    elif pcr_volume >= 1.2:
        base = 7.0
    elif pcr_volume >= 1.0:
        base = 6.0
    elif pcr_volume >= 0.7:
        base = 4.0
    elif pcr_volume >= 0.5:
        base = 2.5
    else:
        base = 1.0   # greed — avoid

    # Confirmation from PCR OI (+/- 0.5)
    if pcr_oi is not None and not pd.isna(pcr_oi):
        if pcr_oi >= 1.5 and base >= 7.0:
            base = min(10.0, base + 0.5)   # OI confirms
        elif pcr_oi < 0.5 and base >= 7.0:
            base = max(0.0, base - 0.5)    # OI contradicts

    return round(base, 1)


def _score_stoch_rsi(k_val: float) -> float:
    """
    Stochastic RSI %K score for mean-reversion entry timing.
    Values 0-1. Oversold < 0.20, Overbought > 0.80.
    More sensitive than plain RSI — gives more precise entry timing.
    """
    if pd.isna(k_val):
        return 5.0

    if k_val <= 0.05:
        return 10.0   # extreme oversold — very strong signal
    elif k_val <= 0.10:
        return 9.0
    elif k_val <= 0.20:
        return 8.0
    elif k_val <= 0.30:
        return 6.5
    elif k_val <= 0.50:
        return 4.0    # neutral zone
    elif k_val <= 0.70:
        return 2.5
    elif k_val <= 0.80:
        return 1.5
    else:
        return 0.5    # overbought — avoid


def _score_williams_r(wr_val: float) -> float:
    """
    Williams %R score for mean-reversion.
    Values 0 to -100. Oversold < -80, Overbought > -20.
    Confirms RSI — reacts directly to price, not a derivative.
    """
    if pd.isna(wr_val):
        return 5.0

    if wr_val <= -95:
        return 10.0   # extreme oversold
    elif wr_val <= -90:
        return 9.0
    elif wr_val <= -80:
        return 8.0
    elif wr_val <= -70:
        return 6.5
    elif wr_val <= -50:
        return 4.0    # neutral
    elif wr_val <= -30:
        return 2.5
    elif wr_val <= -20:
        return 1.5
    else:
        return 0.5    # overbought


def _score_atr_percentile(percentile: float, regime: str) -> float:
    """
    ATR Percentile score — NOT included in composite score.
    Used as FILTER and context for the setup label.

    Compression (< 25):  breakout opportunity — setup bonus
    Normal (25-75):      neutral
    Expansion (> 75):    capitulation potential — bonus if oversold
    """
    if regime == "compression":
        return "🟡 Compression"   # squeeze — ready to move
    elif regime == "expansion":
        return "🔴 Expansion"     # high vol — capitulation possible
    else:
        return "⚪ Normal"
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# EARNINGS FILTER
# ─────────────────────────────────────────────────────────────────────────────

def _near_earnings(ticker: str, days: int = EARNINGS_DAYS) -> bool:
    """
    Checks if earnings fall within N days.
    Uses the yfinance calendar.
    Returns True if NEAR earnings (-> excluded).
    """
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        if cal is None or cal.empty:
            return False

        # calendar returns a DataFrame with index=dates, or a dict
        if isinstance(cal, pd.DataFrame):
            earn_dates = cal.index.tolist()
        elif isinstance(cal, dict):
            earn_dates = cal.get("Earnings Date", [])
        else:
            return False

        today = pd.Timestamp.today().normalize()
        for d in earn_dates:
            d_ts = pd.Timestamp(d).normalize()
            if abs((d_ts - today).days) <= days:
                return True
        return False

    except Exception:
        return False   # if it fails, we don't exclude


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(
    data:           pd.DataFrame,
    sectors:        Optional[dict] = None,
    options_df:     Optional[pd.DataFrame] = None,
    market_caps:    Optional[dict] = None,   # {ticker: market_cap in USD} — from sector_lookup.get_market_caps()
    cap_tier:       Optional[int]  = None,   # 1-5 — if given, hard filter: only tickers in this tier pass
    cap_tiers:      Optional[dict] = None,   # override the CAP_TIERS module default
    top_n:          int   = TOP_N,
    max_per_sector: int   = MAX_PER_SECTOR,
    min_score:      float = MIN_SCORE,
    check_earnings: bool  = False,
    verbose:        bool  = False,
    # ── Scoring weights — override from CONFIG cell ───────────────────────────
    # If not passed, module-level defaults above are used.
    w_rsi:      float = None,
    w_bb:       float = None,
    w_mr:       float = None,
    w_pcr:      float = None,
    w_stochrsi: float = None,
    w_williams: float = None,
    # ── Technical periods/thresholds — override from CONFIG cell ─────────────
    rsi_period:        int   = None,
    bb_period:         int   = None,
    bb_std:            float = None,
    mr_period:         int   = None,
    atr_period:        int   = None,
    volume_period:     int   = None,
    stoch_rsi_period:  int   = None,
    stoch_smooth_k:    int   = None,
    stoch_smooth_d:    int   = None,
    williams_period:   int   = None,
    rsi_max:           float = None,
    min_avg_volume:    float = None,
) -> list[dict]:
    """
    Main scoring function — 100% technical, no dependency on fundamentals.

    Args:
        data:           OHLCV MultiIndex DataFrame (from download_sp500_data)
        sectors:        dict {ticker: sector} (from sector_lookup.py) — optional,
                         used ONLY for portfolio diversification
        options_df:     DataFrame from OptionsScanner.to_dataframe() — optional
        market_caps:    dict {ticker: market_cap} (from sector_lookup.get_market_caps())
                         — optional. Without it, cap_tier filter is ignored.
        cap_tier:       1-5 — if given, HARD FILTER: only tickers in the matching
                         market cap tier pass (see CAP_TIERS). Tickers with
                         unknown market cap are excluded when the filter is active.
        cap_tiers:      dict {tier_num: (lo, hi)} — override the module-level
                         CAP_TIERS from the CONFIG cell.
        top_n:          number of final results
        max_per_sector: diversification limit
        min_score:      minimum composite score
        check_earnings: if True, checks the earnings calendar (slow)
        verbose:        prints filtering details
        w_rsi/w_bb/...: weights from CONFIG cell — override module defaults
        rsi_period/bb_period/...: technical periods — override module defaults

    Returns:
        list[dict] ranked by composite score
    """

    # ── Resolve weights — CONFIG cell overrides module defaults ───────────────
    # Using `is not None` to avoid incorrect fallback on 0.0 values.
    _w_rsi      = w_rsi      if w_rsi      is not None else W_RSI
    _w_bb       = w_bb       if w_bb       is not None else W_BB
    _w_mr       = w_mr       if w_mr       is not None else W_MR
    _w_pcr      = w_pcr      if w_pcr      is not None else W_PCR
    _w_stochrsi = w_stochrsi if w_stochrsi is not None else W_STOCHRSI
    _w_williams = w_williams if w_williams is not None else W_WILLIAMS

    # ── Resolve technical periods/thresholds ───────────────────────────────────
    _rsi_period       = rsi_period       if rsi_period       is not None else RSI_PERIOD
    _bb_period        = bb_period        if bb_period        is not None else BB_PERIOD
    _bb_std           = bb_std           if bb_std           is not None else BB_STD
    _mr_period        = mr_period        if mr_period        is not None else MR_PERIOD
    _atr_period       = atr_period       if atr_period       is not None else ATR_PERIOD
    _volume_period    = volume_period    if volume_period    is not None else VOLUME_PERIOD
    _stoch_rsi_period = stoch_rsi_period if stoch_rsi_period is not None else STOCH_RSI_PERIOD
    _stoch_smooth_k   = stoch_smooth_k   if stoch_smooth_k   is not None else STOCH_SMOOTH_K
    _stoch_smooth_d   = stoch_smooth_d   if stoch_smooth_d   is not None else STOCH_SMOOTH_D
    _williams_period  = williams_period  if williams_period  is not None else WILLIAMS_PERIOD
    _rsi_max          = rsi_max          if rsi_max          is not None else RSI_MAX
    _min_avg_volume   = min_avg_volume   if min_avg_volume   is not None else MIN_AVG_VOLUME
    _cap_tiers        = cap_tiers        if cap_tiers        is not None else CAP_TIERS
    _market_caps      = market_caps or {}

    # No-PCR weights — proportional redistribution of PCR weight
    _total_no_pcr = _w_rsi + _w_bb + _w_mr + _w_stochrsi + _w_williams
    if _total_no_pcr > 0:
        _w_rsi_np      = _w_rsi      / _total_no_pcr
        _w_bb_np       = _w_bb       / _total_no_pcr
        _w_mr_np       = _w_mr       / _total_no_pcr
        _w_stochrsi_np = _w_stochrsi / _total_no_pcr
        _w_williams_np = _w_williams / _total_no_pcr
    else:
        _w_rsi_np = _w_bb_np = _w_mr_np = _w_stochrsi_np = _w_williams_np = 0.2
    options_index = {}
    if options_df is not None and not options_df.empty:
        for _, row in options_df.iterrows():
            ticker = row.get("ticker")
            if not ticker or row.get("error"):
                continue
            # Reliability check
            call_vol   = row.get("call_volume", 0) or 0
            total_vol  = (row.get("put_volume", 0) or 0) + call_vol
            if call_vol >= PCR_MIN_CALL_VOLUME and total_vol >= PCR_MIN_TOTAL_VOLUME:
                options_index[ticker] = {
                    "pcr_volume": row.get("pcr_volume"),
                    "pcr_oi":     row.get("pcr_oi"),
                    "signal":     row.get("signal"),
                    "unusual":    row.get("unusual_activity", False),
                }

    _sectors = sectors or {}

    # ── Tickers ───────────────────────────────────────────────────────────────
    tickers = data.columns.get_level_values(0).unique().tolist()

    results     = []
    filtered_out = {"no_data": 0, "volume": 0, "market_cap": 0, "rsi": 0, "earnings": 0, "low_score": 0}

    for ticker in tickers:

        # ── OHLCV data ────────────────────────────────────────────────────────
        try:
            df = data[ticker].dropna(subset=["Close"])
            if len(df) < max(_bb_period, _mr_period // 4, _rsi_period) + 5:
                filtered_out["no_data"] += 1
                continue
        except Exception:
            filtered_out["no_data"] += 1
            continue

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        price = close.iloc[-1]
        if price <= 0 or pd.isna(price):
            filtered_out["no_data"] += 1
            continue

        # ── Hard filter: Volume ───────────────────────────────────────────────
        avg_vol = volume.rolling(_volume_period).mean().iloc[-1]
        if pd.isna(avg_vol) or avg_vol < _min_avg_volume:
            filtered_out["volume"] += 1
            continue

        # ── Hard filter: Market Cap Tier ────────────────────────────────────────
        ticker_mcap = _market_caps.get(ticker)
        if cap_tier is not None:
            if ticker_mcap is None or pd.isna(ticker_mcap):
                filtered_out["market_cap"] += 1
                continue
            lo, hi = _cap_tiers.get(cap_tier, (0, float("inf")))
            if not (lo <= ticker_mcap < hi):
                filtered_out["market_cap"] += 1
                continue

        # ── Indicators ────────────────────────────────────────────────────────
        rsi_series              = _rsi(close, _rsi_period)
        upper, lower, pct_b_series, sma_bb = _bollinger(close, _bb_period, _bb_std)
        atr_val                 = _atr(high, low, close, _atr_period).iloc[-1]

        rsi_val   = rsi_series.iloc[-1]
        pct_b_val = pct_b_series.iloc[-1]

        # Mean Reversion z-score (rolling window)
        mr_window    = min(_mr_period, len(close))
        rolling_mean = close.rolling(mr_window).mean()
        rolling_std  = close.rolling(mr_window).std()
        z_series     = (close - rolling_mean) / rolling_std.replace(0, np.nan)
        z_val        = z_series.iloc[-1]

        # ── New indicators ─────────────────────────────────────────────────────
        # StochRSI — more sensitive RSI for precise entry timing
        stoch_k, stoch_d = _compute_stoch_rsi(
            close,
            rsi_period   = _rsi_period,
            stoch_period = _stoch_rsi_period,
            smooth_k     = _stoch_smooth_k,
            smooth_d     = _stoch_smooth_d,
        )
        stoch_k_val = stoch_k.iloc[-1]
        stoch_d_val = stoch_d.iloc[-1]

        # Williams %R — price position within the N-day range
        wr_series = _compute_williams_r(high, low, close, _williams_period)
        wr_val    = wr_series.iloc[-1]

        # ATR Percentile — volatility regime (used as context, not score)
        atr_pct, atr_regime = _compute_atr_percentile(
            high, low, close,
            atr_period = _atr_period,
            lookback   = ATR_PCT_LOOKBACK,
        )
        atr_pct_label = _score_atr_percentile(atr_pct, atr_regime)

        # Volume surge vs average
        vol_ratio = float(volume.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0

        # ── Hard filter: RSI ──────────────────────────────────────────────────
        if rsi_val > _rsi_max:
            filtered_out["rsi"] += 1
            continue

        # ── Hard filter: Earnings ─────────────────────────────────────────────
        if check_earnings and _near_earnings(ticker, EARNINGS_DAYS):
            filtered_out["earnings"] += 1
            if verbose:
                print(f"  {ticker}: near earnings -> excluded")
            continue

        # ── SCORING ───────────────────────────────────────────────────────────
        s_rsi      = _score_rsi(rsi_val)
        s_bb       = _score_bollinger(pct_b_val)
        s_mr       = _score_mean_reversion(z_val)
        s_stochrsi = _score_stoch_rsi(stoch_k_val)
        s_williams = _score_williams_r(wr_val)

        # Options PCR score
        opt_data  = options_index.get(ticker)
        s_pcr     = None
        pcr_value = None
        pcr_signal = "N/A"

        if opt_data:
            pcr_value  = opt_data.get("pcr_volume")
            pcr_oi     = opt_data.get("pcr_oi")
            pcr_signal = opt_data.get("signal", "N/A")
            s_pcr      = _score_pcr(pcr_value, pcr_oi)

        # Weights — readjusted if PCR is missing
        if s_pcr is not None:
            composite = round(
                s_rsi      * _w_rsi      +
                s_bb       * _w_bb       +
                s_mr       * _w_mr       +
                s_pcr      * _w_pcr      +
                s_stochrsi * _w_stochrsi +
                s_williams * _w_williams,
                2
            )
            weights_used = (
                f"RSI={_w_rsi} BB={_w_bb} MR={_w_mr} "
                f"PCR={_w_pcr} StochRSI={_w_stochrsi} Williams={_w_williams}"
            )
        else:
            composite = round(
                s_rsi      * _w_rsi_np      +
                s_bb       * _w_bb_np       +
                s_mr       * _w_mr_np       +
                s_stochrsi * _w_stochrsi_np +
                s_williams * _w_williams_np,
                2
            )
            weights_used = (
                f"RSI={_w_rsi_np:.3f} BB={_w_bb_np:.3f} MR={_w_mr_np:.3f} "
                f"StochRSI={_w_stochrsi_np:.3f} Williams={_w_williams_np:.3f} (no PCR)"
            )

        # ── Hard filter: min score ────────────────────────────────────────────
        if composite < min_score:
            filtered_out["low_score"] += 1
            continue

        # ── Setup label ───────────────────────────────────────────────────────
        if composite >= 7.5:
            setup = "🟢 Strong"
        elif composite >= 6.0:
            setup = "🟡 Watchlist"
        else:
            setup = "🔴 Monitor"

        # ── ATR-based levels ──────────────────────────────────────────────────
        stop_loss  = round(price - 1.5 * atr_val, 2)
        target_1   = round(price + 2.0 * atr_val, 2)
        target_2   = round(price + 4.0 * atr_val, 2)
        entry_low  = round(price - 0.5 * atr_val, 2)
        entry_high = round(price + 0.3 * atr_val, 2)
        risk       = price - stop_loss
        rr_ratio   = round((target_2 - price) / risk, 2) if risk > 0 else None

        results.append({
            "ticker":          ticker,
            "composite_score": composite,
            "setup":           setup,
            "sector":          _sectors.get(ticker, "Unknown"),
            "market_cap":      ticker_mcap,
            "cap_tier_label":  _cap_tier_label(ticker_mcap, _cap_tiers),
            "price":           round(price, 2),
            "atr":             round(atr_val, 2),
            # Scores breakdown
            "s_rsi":           round(s_rsi, 1),
            "s_bb":            round(s_bb,  1),
            "s_mr":            round(s_mr,  1),
            "s_pcr":           round(s_pcr, 1) if s_pcr is not None else None,
            "s_stochrsi":      round(s_stochrsi, 1),
            "s_williams":      round(s_williams, 1),
            # Raw values
            "rsi":             round(rsi_val, 1),
            "pct_b":           round(pct_b_val, 3) if pd.notna(pct_b_val) else None,
            "z_score":         round(z_val, 2) if pd.notna(z_val) else None,
            "stoch_k":         round(stoch_k_val, 3),
            "stoch_d":         round(stoch_d_val, 3),
            "williams_r":      round(wr_val, 1),
            "atr_percentile":  atr_pct,
            "atr_regime":      atr_regime,
            "atr_pct_label":   atr_pct_label,
            "vol_ratio":       round(vol_ratio, 2),
            "pcr_volume":      round(pcr_value, 2) if pcr_value else None,
            "pcr_signal":      pcr_signal,
            "avg_volume_m":    round(avg_vol / 1_000_000, 1),
            # Levels
            "entry_low":       entry_low,
            "entry_high":      entry_high,
            "stop_loss":       stop_loss,
            "target_1":        target_1,
            "target_2":        target_2,
            "rr_ratio":        rr_ratio,
            # Meta
            "has_pcr":         s_pcr is not None,
            "weights_used":    weights_used,
        })

    # ── Ranking ───────────────────────────────────────────────────────────────
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    # ── Diversification ───────────────────────────────────────────────────────
    sector_counts: dict[str, int] = {}
    diversified = []
    for r in results:
        sector = r["sector"]
        count  = sector_counts.get(sector, 0)
        if count < max_per_sector:
            r["rank"] = len(diversified) + 1
            diversified.append(r)
            sector_counts[sector] = count + 1
        if len(diversified) >= top_n:
            break

    # ── Filter summary ────────────────────────────────────────────────────────
    total = len(tickers)
    passed = len(results)
    print(f"\n  FILTER SUMMARY ({total} tickers)")
    print(f"  {'─'*35}")
    print(f"  No data          : {filtered_out['no_data']:>4}")
    print(f"  Volume < 500k    : {filtered_out['volume']:>4}")
    if cap_tier is not None:
        tier_label = CAP_TIER_LABELS.get(cap_tier, f"Tier {cap_tier}")
        print(f"  Cap tier != {cap_tier} ({tier_label}): {filtered_out['market_cap']:>4}")
    print(f"  RSI > {_rsi_max:.0f}         : {filtered_out['rsi']:>4}")
    if check_earnings:
        print(f"  Near earnings    : {filtered_out['earnings']:>4}")
    print(f"  Score < {min_score:.1f}       : {filtered_out['low_score']:>4}")
    print(f"  {'─'*35}")
    print(f"  Passed           : {passed:>4}")
    print(f"  Final (diversif.): {len(diversified):>4}")

    return diversified


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def to_dataframe(scored: list[dict]) -> pd.DataFrame:
    """Returns a DataFrame for display in the notebook."""
    rows = []
    for s in scored:
        rows.append({
            "Rank":        s["rank"],
            "Ticker":      s["ticker"],
            "Sector":      s["sector"],
            "Cap":         s["cap_tier_label"],
            "MarketCap($B)": round(s["market_cap"] / 1e9, 1) if s.get("market_cap") else None,
            "Score":       s["composite_score"],
            "Setup":       s["setup"],
            # Core signals
            "RSI":         s["rsi"],
            "s_RSI":       s["s_rsi"],
            "pct_B":       s["pct_b"],
            "s_BB":        s["s_bb"],
            "Z-Score":     s["z_score"],
            "s_MR":        s["s_mr"],
            "PCR":         s["pcr_volume"],
            "s_PCR":       s["s_pcr"],
            "PCR Signal":  s["pcr_signal"],
            # New indicators
            "StochRSI_K":  s["stoch_k"],
            "s_StochRSI":  s["s_stochrsi"],
            "Williams%R":  s["williams_r"],
            "s_Williams":  s["s_williams"],
            "ATR_Pct":     s["atr_percentile"],
            "Vol_Ratio":   s["vol_ratio"],
            "ATR_Regime":  s["atr_pct_label"],
            # Levels
            "Price":       s["price"],
            "ATR":         s["atr"],
            "Stop":        s["stop_loss"],
            "T2":          s["target_2"],
            "R/R":         s["rr_ratio"],
            "Vol(M)":      s["avg_volume_m"],
        })
    return pd.DataFrame(rows)


def print_mr_report(scored: list[dict]) -> None:
    """Terminal report — detailed table."""
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    # Get actual weights from the first result (already resolved from CONFIG)
    w_line = scored[0]["weights_used"] if scored else "N/A"

    print(f"\n{'═'*70}")
    print(f"  MEAN-REVERSION SCANNER — TOP {len(scored)}")
    print(f"  {now}")
    print(f"  Weights: {w_line}")
    print(f"{'═'*70}\n")

    for s in scored:
        pcr_str = (
            f"PCR={s['pcr_volume']:.1f}({s['s_pcr']:.0f})"
            if s["has_pcr"] else "PCR=N/A"
        )
        rr_str = f"{s['rr_ratio']:.1f}x" if s["rr_ratio"] else "N/A"

        # ATR regime emoji
        regime_str = s.get("atr_pct_label", "")

        mcap_str = (
            f"${s['market_cap']/1e9:,.0f}B ({s['cap_tier_label']})"
            if s.get("market_cap") else "Cap: N/A"
        )
        print(
            f"#{s['rank']:<3} {s['ticker']:<6} "
            f"Score: {s['composite_score']:.1f}  "
            f"{s['setup']}  {regime_str}  {mcap_str}"
        )
        # Line 1: core signals
        print(
            f"     RSI={s['rsi']:.0f}({s['s_rsi']:.0f})  "
            f"BB%={s['pct_b']:.2f}({s['s_bb']:.0f})  "
            f"Z={s['z_score']:+.2f}({s['s_mr']:.0f})  "
            f"{pcr_str}"
        )
        # Line 2: new signals
        print(
            f"     StochRSI_K={s['stoch_k']:.2f}({s['s_stochrsi']:.0f})  "
            f"Williams%R={s['williams_r']:.0f}({s['s_williams']:.0f})  "
            f"ATR_Pct={s['atr_percentile']:.0f}%  "
            f"VolRatio={s['vol_ratio']:.1f}x"
        )
        # Line 3: levels
        print(
            f"     ${s['price']:.2f}  ATR=${s['atr']:.2f}  "
            f"Entry: ${s['entry_low']:.2f}-${s['entry_high']:.2f}  "
            f"Stop: ${s['stop_loss']:.2f}  "
            f"T2: ${s['target_2']:.2f}  "
            f"R/R: {rr_str}  "
            f"Vol: {s['avg_volume_m']:.0f}M"
        )
        print(f"     {s['sector']}")
        print()

    # Sector distribution
    print(f"{'─'*70}")
    print(f"  SECTOR DISTRIBUTION:")
    sector_counts: dict[str, int] = {}
    for s in scored:
        sector_counts[s["sector"]] = sector_counts.get(s["sector"], 0) + 1
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {sector:<35} {count}  {bar}")
    print(f"{'═'*70}\n")
