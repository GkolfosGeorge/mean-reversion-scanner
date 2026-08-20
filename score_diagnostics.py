# score_diagnostics.py
"""
MR Scanner — Score Diagnostics (Roadmap 1.5 / 1.6 / 1.7)
──────────────────────────────────────────────────────────
Τρία εργαλεία πάνω στο ΙΔΙΟ point-in-time panel από raw sub-scores,
ώστε οι δείκτες να υπολογιστούν ΜΙΑ φορά ανά (ticker, date) και μετά
να τρέξουν πάνω του πολλαπλές αναλύσεις χωρίς re-computation:

  1.5  weight_sensitivity_analysis()  -> πόσο ευαίσθητο είναι το
       composite score / ranking σε μικρές μεταβολές βαρών
  1.6  correlation_audit()            -> correlation matrix των 5
       raw sub-scores (χωρίς PCR — δεν υπάρχει αξιόπιστο ιστορικό PCR,
       ίδια σύμβαση με το backtester.py)
  1.7  score_distribution_monitor()   -> rolling histogram / saturation
       metrics του composite score στον χρόνο

Reuses the EXACT same helper functions as scorer_mr.py / backtester.py
(καμία διπλή υλοποίηση δεικτών — single source of truth).

Usage (notebook):
    import score_diagnostics as sd
    panel = sd.build_score_panel(data, tickers=universe_tickers,
                                  start="2024-01-01", end="2025-12-31")
    sd.correlation_audit(panel)
    sd.weight_sensitivity_analysis(panel)
    sd.score_distribution_monitor(panel)
"""

import pandas as pd
import numpy as np
from typing import Optional

# ── Reuses the SAME technical helpers as scorer_mr.py / backtester.py ────────
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
    )

SUBSCORE_COLS = ["s_rsi", "s_bb", "s_mr", "s_stochrsi", "s_williams"]

# ── Default no-PCR baseline weights (ίδια convention με backtester.py) ───────
DEFAULT_WEIGHTS = {
    "w_rsi":      MR_W_RSI,
    "w_bb":       MR_W_BB,
    "w_mr":       MR_W_MR,
    "w_stochrsi": MR_W_STOCHRSI,
    "w_williams": MR_W_WILLIAMS,
}


# ─────────────────────────────────────────────────────────────────────────────
# PANEL BUILDER — υπολογίζει τα raw sub-scores μία φορά ανά (ticker, date)
# ─────────────────────────────────────────────────────────────────────────────

def _raw_subscores_at_date(df: pd.DataFrame, date: pd.Timestamp, params: dict | None = None) -> Optional[dict]:
    """
    Point-in-time raw sub-scores για ένα ticker/date — ίδια λογική με
    backtester._compute_mr_score_at_date(), αλλά επιστρέφει ΚΑΙ τα raw
    sub-scores (όχι μόνο το composite), απαραίτητο για correlation audit
    και weight sensitivity. No look-ahead: df κόβεται στο date.
    No PCR (δεν υπάρχει αξιόπιστο ιστορικό options data).
    """
    p = params or {}
    _rsi_period       = p.get("rsi_period",       MR_RSI_PERIOD)
    _bb_period        = p.get("bb_period",        MR_BB_PERIOD)
    _bb_std           = p.get("bb_std",           MR_BB_STD)
    _mr_period        = p.get("mr_period",        MR_PERIOD)
    _atr_period       = p.get("atr_period",        MR_ATR_PERIOD)
    _volume_period    = p.get("volume_period",    MR_VOLUME_PERIOD)
    _stoch_rsi_period = p.get("stoch_rsi_period", STOCH_RSI_PERIOD)
    _stoch_smooth_k   = p.get("stoch_smooth_k",   STOCH_SMOOTH_K)
    _stoch_smooth_d   = p.get("stoch_smooth_d",   STOCH_SMOOTH_D)
    _williams_period  = p.get("williams_period",  WILLIAMS_PERIOD)
    _rsi_max          = p.get("rsi_max",          RSI_MAX)
    _min_avg_volume   = p.get("min_avg_volume",   MR_MIN_AVG_VOLUME)

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
        return None   # ίδιο hard filter με τον ζωντανό scanner

    _, _, pct_b_series, _ = _bollinger(close, _bb_period, _bb_std)
    pct_b_val = pct_b_series.iloc[-1]

    atr_val = _atr(high, low, close, _atr_period).iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        return None

    mr_window    = min(_mr_period, len(close))
    rolling_mean = close.rolling(mr_window).mean()
    rolling_std  = close.rolling(mr_window).std()
    z_val = ((close - rolling_mean) / rolling_std.replace(0, np.nan)).iloc[-1]

    stoch_k, _ = _compute_stoch_rsi(
        close, rsi_period=_rsi_period, stoch_period=_stoch_rsi_period,
        smooth_k=_stoch_smooth_k, smooth_d=_stoch_smooth_d,
    )
    stoch_k_val = stoch_k.iloc[-1]
    wr_val      = _compute_williams_r(high, low, close, _williams_period).iloc[-1]

    if pd.isna(rsi_val) or pd.isna(pct_b_val) or pd.isna(stoch_k_val) or pd.isna(wr_val):
        return None

    return {
        "date":       date,
        "price":      round(float(price), 2),
        "rsi":        round(float(rsi_val), 2),
        "pct_b":      round(float(pct_b_val), 4),
        "z_score":    round(float(z_val), 4) if pd.notna(z_val) else None,
        "stoch_k":    round(float(stoch_k_val), 4),
        "williams_r": round(float(wr_val), 2),
        "s_rsi":      _score_rsi(rsi_val),
        "s_bb":       _score_bollinger(pct_b_val),
        "s_mr":       _score_mean_reversion(z_val),
        "s_stochrsi": _score_stoch_rsi(stoch_k_val),
        "s_williams": _score_williams_r(wr_val),
    }


def _get_review_dates(start: str, end: str, index: pd.DatetimeIndex, freq: str = "W-TUE") -> list[pd.Timestamp]:
    """
    Ημερομηνίες review — default: κάθε Τρίτη (ίδια σύμβαση με το trading
    cadence σου — θέσεις ανοίγουν Τρίτες, όχι Δευτέρες, λόγω gap opens).
    freq="M" -> πρώτη διαθέσιμη trading day κάθε μήνα (πιο γρήγορο, λιγότερα σημεία).
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    trading_days = index[(index >= start_ts) & (index <= end_ts)]
    if len(trading_days) == 0:
        return []

    if freq.upper().startswith("M"):
        dates, seen = [], None
        for d in trading_days:
            key = (d.year, d.month)
            if key != seen:
                dates.append(d)
                seen = key
        return dates

    # weekly: πρώτη trading day >= κάθε target weekday (default Τρίτη=1)
    target_dow = {"W-MON": 0, "W-TUE": 1, "W-WED": 2, "W-THU": 3, "W-FRI": 4}.get(freq.upper(), 1)
    candidates = pd.date_range(start_ts, end_ts, freq="W-" + ["MON","TUE","WED","THU","FRI"][target_dow])
    dates = []
    for c in candidates:
        avail = trading_days[trading_days >= c]
        if len(avail) > 0:
            dates.append(avail[0])
    return sorted(set(dates))


def build_score_panel(
    data:       pd.DataFrame,
    tickers:    Optional[list[str]] = None,
    start:      Optional[str] = None,
    end:        Optional[str] = None,
    freq:       str = "W-TUE",
    max_tickers: Optional[int] = None,
    params:     Optional[dict] = None,
    random_state: int = 42,
    verbose:    bool = True,
) -> pd.DataFrame:
    """
    Χτίζει long-format panel [date, ticker, raw values, s_rsi...s_williams]
    πάνω στο οποίο τρέχουν και τα 3 εργαλεία (1.5/1.6/1.7). Αυτό είναι το
    ακριβό βήμα (indicator computation) — γίνεται ΜΙΑ φορά.

    tickers=None      -> όλα τα tickers στο data
    max_tickers        -> τυχαίο υποσύνολο, για ταχύτητα σε πρώτο πέρασμα
    start/end=None     -> όλο το διαθέσιμο εύρος του data
    freq="W-TUE"        -> εβδομαδιαία review dates (ίδιο cadence με scanner)
                           "M" -> μηνιαία (γρηγορότερο, λιγότερα δείγματα)
    """
    all_tickers = data.columns.get_level_values(0).unique().tolist()
    universe = tickers if tickers else all_tickers

    if max_tickers is not None and len(universe) > max_tickers:
        rng = np.random.RandomState(random_state)
        universe = list(rng.choice(universe, size=max_tickers, replace=False))

    full_index = data.index
    _start = start or str(full_index.min().date())
    _end   = end   or str(full_index.max().date())
    review_dates = _get_review_dates(_start, _end, full_index, freq=freq)

    if verbose:
        print(f"📊 Building score panel: {len(universe)} tickers × {len(review_dates)} dates "
              f"({freq}) = up to {len(universe)*len(review_dates):,} snapshots")

    rows = []
    skipped_tickers = 0
    for ticker in universe:
        try:
            df = data[ticker].dropna(subset=["Close"])
        except Exception:
            skipped_tickers += 1
            continue
        if df.empty:
            skipped_tickers += 1
            continue

        for date in review_dates:
            rec = _raw_subscores_at_date(df, date, params)
            if rec is not None:
                rec["ticker"] = ticker
                rows.append(rec)

    panel = pd.DataFrame(rows)
    if panel.empty:
        print("⚠️ Άδειο panel — έλεγξε εύρος ημερομηνιών / hard filters (RSI, volume).")
        return panel

    panel = panel[["date", "ticker", "price", "rsi", "pct_b", "z_score",
                    "stoch_k", "williams_r"] + SUBSCORE_COLS]

    if verbose:
        n_dates = panel["date"].nunique()
        n_tick  = panel["ticker"].nunique()
        print(f"✅ Panel έτοιμο: {len(panel):,} γραμμές | {n_tick} tickers | {n_dates} ημερομηνίες "
              f"| {skipped_tickers} tickers παραλείφθηκαν (no data)")
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# 1.6 — CORRELATION AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def correlation_audit(
    panel: pd.DataFrame,
    method: str = "pearson",
    high_corr_threshold: float = 0.60,
) -> dict:
    """
    Correlation matrix των 5 raw sub-scores (χωρίς PCR). Αν RSI, StochRSI
    και Williams %R είναι σε μεγάλο βαθμό συσχετισμένα, το σύστημα δίνει
    de facto υπερβολικό βάρος στο "momentum oversold" concept 3 φορές
    αντί για 1 — flag ζευγών πάνω από το threshold.

    Επιστρέφει: {"corr_matrix": df, "flagged_pairs": list[dict]}
    """
    if panel.empty:
        print("⚠️ Άδειο panel.")
        return {}

    corr = panel[SUBSCORE_COLS].corr(method=method)

    flagged = []
    cols = SUBSCORE_COLS
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= high_corr_threshold:
                flagged.append({"pair": f"{cols[i]} ↔ {cols[j]}", "corr": round(float(r), 3)})
    flagged.sort(key=lambda x: -abs(x["corr"]))

    print(f"\n{'═'*66}")
    print(f"  1.6 — CORRELATION AUDIT ({method}, n={len(panel):,} snapshots)")
    print(f"{'═'*66}")
    print(corr.round(2).to_string())
    print(f"\n  Threshold για 'elevated overlap': |r| ≥ {high_corr_threshold}")
    if flagged:
        print(f"  🔴 {len(flagged)} ζεύγη πάνω από threshold:")
        for f in flagged:
            print(f"     {f['pair']:<28} r = {f['corr']:+.3f}")
        momentum_trio = {"s_rsi", "s_stochrsi", "s_williams"}
        trio_flagged = [f for f in flagged if set(f["pair"].split(" ↔ ")).issubset(momentum_trio)]
        if len(trio_flagged) >= 2:
            print(f"\n  ⚠️  Το RSI/StochRSI/Williams%R τρίπτυχο δείχνει υψηλή αλληλεπικάλυψη —")
            print(f"      de facto ~{(MR_W_RSI+MR_W_STOCHRSI+MR_W_WILLIAMS)*100:.0f}% του weight πάει στο")
            print(f"      ίδιο 'momentum oversold' concept. Σκέψου de-duplication:")
            print(f"      (α) merge σε ένα momentum-oversold score, ή")
            print(f"      (β) μείωσε συνολικό βάρος τριπτύχου, αύξησε BB/MR (διαφορετικό concept)")
    else:
        print(f"  🟢 Κανένα ζεύγος πάνω από το threshold — τα 5 layers φαίνονται σχετικά ανεξάρτητα.")
    print(f"{'═'*66}\n")

    return {"corr_matrix": corr, "flagged_pairs": flagged}


# ─────────────────────────────────────────────────────────────────────────────
# 1.5 — WEIGHT SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def weight_sensitivity_analysis(
    panel: pd.DataFrame,
    base_weights: Optional[dict] = None,
    bounds: Optional[dict] = None,
    n_samples: int = 200,
    top_n: int = 20,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Δειγματοληπτεί n_samples τυχαία weight vectors μέσα σε bounds (π.χ.
    RSI 25-35%, BB 15-25%), κανονικοποιεί σε άθροισμα 1.0, και για κάθε
    date στο panel συγκρίνει το ranking με τα perturbed weights έναντι
    του baseline ranking:

      - Spearman rank correlation (πόσο ίδια η σειρά)
      - Top-N overlap % (πόσα από τα σημερινά Top-N παραμένουν Top-N)
      - Mean |Δcomposite| ανά ticker

    Φτηνό υπολογιστικά: τα sub-scores είναι ήδη υπολογισμένα στο panel,
    το composite είναι απλά weighted sum — δεν ξαναϋπολογίζονται δείκτες.

    Ερμηνεία (rule of thumb):
      Spearman > 0.90 & overlap > 85%  -> weights "παγώνουν" με ασφάλεια
      Spearman < 0.75 ή overlap < 60%  -> το ranking είναι εύθραυστο,
                                            χρειάζεται περισσότερη δουλειά
                                            πριν freeze
    """
    if panel.empty:
        print("⚠️ Άδειο panel.")
        return {}

    _base = base_weights or DEFAULT_WEIGHTS
    _bounds = bounds or {
        "w_rsi":      (0.25, 0.35),
        "w_bb":       (0.15, 0.25),
        "w_mr":       (0.10, 0.20),
        "w_stochrsi": (0.05, 0.15),
        "w_williams": (0.05, 0.15),
    }
    weight_keys = ["w_rsi", "w_bb", "w_mr", "w_stochrsi", "w_williams"]
    subscore_map = dict(zip(weight_keys, SUBSCORE_COLS))

    def _composite(df_date: pd.DataFrame, w: dict) -> pd.Series:
        total = sum(w.values())
        wn = {k: v / total for k, v in w.items()}
        return sum(df_date[subscore_map[k]] * wn[k] for k in weight_keys)

    rng = np.random.RandomState(random_state)
    draws = []
    for _ in range(n_samples):
        w = {k: rng.uniform(lo, hi) for k, (lo, hi) in _bounds.items()}
        draws.append(w)

    dates = sorted(panel["date"].unique())
    spearman_rows, overlap_rows, delta_rows = [], [], []

    for date in dates:
        df_date = panel[panel["date"] == date].copy()
        if len(df_date) < max(5, top_n // 2):
            continue

        base_composite = _composite(df_date, _base)
        df_date["_base_composite"] = base_composite
        base_rank = df_date.sort_values("_base_composite", ascending=False)
        base_top = set(base_rank["ticker"].head(top_n))

        for w in draws:
            comp = _composite(df_date, w)
            spearman = base_composite.corr(comp, method="spearman")
            spearman_rows.append(spearman)

            ranked = df_date.assign(_c=comp).sort_values("_c", ascending=False)
            top_now = set(ranked["ticker"].head(top_n))
            overlap = len(base_top & top_now) / max(1, len(base_top))
            overlap_rows.append(overlap)

            delta_rows.append(float((comp - base_composite).abs().mean()))

    if not spearman_rows:
        print("⚠️ Δεν βρέθηκαν αρκετά tickers/date για ανάλυση.")
        return {}

    summary = {
        "n_dates":            len(dates),
        "n_weight_draws":     n_samples,
        "spearman_mean":      float(np.mean(spearman_rows)),
        "spearman_p10":       float(np.percentile(spearman_rows, 10)),
        "topN_overlap_mean":  float(np.mean(overlap_rows)),
        "topN_overlap_p10":   float(np.percentile(overlap_rows, 10)),
        "mean_abs_delta":     float(np.mean(delta_rows)),
    }

    if verbose:
        print(f"\n{'═'*66}")
        print(f"  1.5 — WEIGHT SENSITIVITY ANALYSIS")
        print(f"  {n_samples} τυχαία weight-draws × {summary['n_dates']} ημερομηνίες")
        print(f"{'═'*66}")
        print(f"  Bounds: " + ", ".join(f"{k}={v[0]:.2f}-{v[1]:.2f}" for k, v in _bounds.items()))
        print(f"\n  Spearman rank corr vs baseline:")
        print(f"    mean = {summary['spearman_mean']:.3f}   p10 = {summary['spearman_p10']:.3f}")
        print(f"  Top-{top_n} overlap vs baseline:")
        print(f"    mean = {summary['topN_overlap_mean']*100:.1f}%   p10 = {summary['topN_overlap_p10']*100:.1f}%")
        print(f"  Mean |Δcomposite| ανά ticker: {summary['mean_abs_delta']:.3f} pts (κλίμακα 0-10)")

        if summary["spearman_mean"] >= 0.90 and summary["topN_overlap_mean"] >= 0.85:
            verdict = "🟢 Σταθερό — τα βάρη μπορούν να 'παγώσουν' με σχετική ασφάλεια."
        elif summary["spearman_mean"] >= 0.75 and summary["topN_overlap_mean"] >= 0.60:
            verdict = "🟡 Μέτρια ευαισθησία — οριακά αποδεκτό, αλλά ελέγξε ποια layers κινούν τη διαφορά."
        else:
            verdict = "🔴 Εύθραυστο ranking — μικρές αλλαγές βαρών αλλάζουν σημαντικά τη λίστα. Μην παγώσεις ακόμα."
        print(f"\n  {verdict}")
        print(f"{'═'*66}\n")

    return {"summary": summary, "spearman_dist": spearman_rows,
            "overlap_dist": overlap_rows, "delta_dist": delta_rows}


# ─────────────────────────────────────────────────────────────────────────────
# 1.7 — SCORE DISTRIBUTION MONITOR
# ─────────────────────────────────────────────────────────────────────────────

def score_distribution_monitor(
    panel: pd.DataFrame,
    weights: Optional[dict] = None,
    top_bin_threshold: float = 8.5,
    bottom_bin_threshold: float = 2.0,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Rolling histogram / saturation metrics του composite score στον χρόνο.
    Saturation = μεγάλο ποσοστό scores μαζεμένο κοντά στο άκρο (>= top_bin_threshold)
    -> το σύστημα χάνει διαχωριστική ικανότητα (όλοι "strong", καμία πληροφορία
    στην κατάταξη). Αναφορά ανά ημερομηνία + optional plot.
    """
    if panel.empty:
        print("⚠️ Άδειο panel.")
        return pd.DataFrame()

    w = weights or DEFAULT_WEIGHTS
    total = sum(w.values())
    wn = {k: v / total for k, v in w.items()}
    subscore_map = dict(zip(["w_rsi", "w_bb", "w_mr", "w_stochrsi", "w_williams"], SUBSCORE_COLS))

    panel = panel.copy()
    panel["composite"] = sum(panel[subscore_map[k]] * wn[k] for k in wn)

    rows = []
    for date, g in panel.groupby("date"):
        c = g["composite"]
        rows.append({
            "date":            date,
            "n":               len(c),
            "mean":            round(float(c.mean()), 2),
            "std":             round(float(c.std()), 2),
            "skew":            round(float(c.skew()), 2),
            "pct_top_bin":     round(float((c >= top_bin_threshold).mean() * 100), 1),
            "pct_bottom_bin":  round(float((c <= bottom_bin_threshold).mean() * 100), 1),
            "p10":             round(float(c.quantile(0.10)), 2),
            "p50":             round(float(c.quantile(0.50)), 2),
            "p90":             round(float(c.quantile(0.90)), 2),
        })
    summary = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    print(f"\n{'═'*66}")
    print(f"  1.7 — SCORE DISTRIBUTION MONITOR ({len(summary)} ημερομηνίες)")
    print(f"{'═'*66}")
    print(summary.to_string(index=False))

    avg_top_bin = summary["pct_top_bin"].mean()
    print(f"\n  Μέσο % scores >= {top_bin_threshold}: {avg_top_bin:.1f}%")
    if avg_top_bin >= 15:
        print(f"  🔴 Πιθανό saturation — μεγάλο μέρος του universe συσσωρεύεται κοντά στο άκρο.")
        print(f"     Η κατάταξη Top-N χάνει νόημα αν πολλά tickers έχουν σχεδόν ίδιο score.")
    elif avg_top_bin >= 8:
        print(f"  🟡 Οριακό — παρακολούθησε αν αυξάνεται με τον χρόνο (ειδικά σε risk-off περιόδους).")
    else:
        print(f"  🟢 Καμία ένδειξη saturation στο top bin.")
    print(f"{'═'*66}\n")

    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

            axes[0].plot(summary["date"], summary["p10"], label="p10", color="#d62728")
            axes[0].plot(summary["date"], summary["p50"], label="median", color="#1f77b4")
            axes[0].plot(summary["date"], summary["p90"], label="p90", color="#2ca02c")
            axes[0].set_title("Composite score — rolling percentiles")
            axes[0].legend()
            axes[0].tick_params(axis="x", rotation=45)

            axes[1].plot(summary["date"], summary["pct_top_bin"], label=f"% ≥ {top_bin_threshold}", color="#d62728")
            axes[1].plot(summary["date"], summary["pct_bottom_bin"], label=f"% ≤ {bottom_bin_threshold}", color="#7f7f7f")
            axes[1].axhline(15, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
            axes[1].set_title("Saturation — % universe στα άκρα")
            axes[1].legend()
            axes[1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"(plot skipped: {e})")

    return summary
