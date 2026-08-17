# mega_cap_alert.py
"""
Automated Mega-Cap Alert + Watchlist Digest (single email)
─────────────────────────────────────────────────────────────────────────────
Standalone script — does NOT go through the notebook. Designed to run on a
GitHub Actions schedule (2x/day: market open + market close).

Sends ONE email per run with two sections:

  1. MEGA-CAP ALERTS — scans the market-cap tier(s) you configure
     (ALERT_CAP_TIERS, see scorer_mr.CAP_TIERS) and includes a ticker ONLY
     if its score >= ALERT_SCORE_THRESHOLD. Same compute_scores() logic as
     the manual weekly notebook scan.

  2. WATCHLIST — a fixed list of tickers you name yourself (WATCHLIST,
     can be anything yfinance recognizes — not limited to the S&P 500).
     These are ALWAYS included in the email, regardless of score, RSI, or
     volume — every hard filter is disabled for this section on purpose.
     Good for names you want to track daily no matter what (e.g. BABA,
     NBIS), not just when they happen to look like a technical opportunity.
     The only thing that can still exclude a ticker here is a genuine data
     problem (ticker not found, or not enough trading history yet).

The email is sent whenever there's anything to report — which, since the
watchlist is unconditional, is effectively every run (unless WATCHLIST is
empty AND no mega-cap ticker qualifies).

── CONFIG ─────────────────────────────────────────────────────────────────
Edit ALERT_SCORE_THRESHOLD, ALERT_CAP_TIERS, and WATCHLIST below.

Uses Yahoo SMTP (smtp.mail.yahoo.com). The env var names below still say
"GMAIL_*" for historical reasons — they hold Yahoo credentials now, the
names just weren't renamed to avoid touching the GitHub Secrets / workflow
config. Functionally it's Yahoo, not Gmail.

── SECRETS (set as GitHub repo secrets, read via env vars) ──────────────────
  GMAIL_USER          - sender Yahoo address (despite the name)
  GMAIL_APP_PASSWORD  - Yahoo App Password (NOT your regular password —
                         generate one from Yahoo Account Security ->
                         "Generate app password")
  ALERT_RECIPIENT     - where to send the alert (can be the same as GMAIL_USER)

Usage:
    python mega_cap_alert.py
"""

import os
import sys
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

from ticker_provider import get_tickers
from sector_lookup import get_sectors_and_caps
from data_engine import download_sp500_data
import scorer_mr

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit this
# ─────────────────────────────────────────────────────────────────────────────
ALERT_SCORE_THRESHOLD = 6.5     # <-- mega-cap section: minimum score to include
ALERT_CAP_TIERS       = [4,5]     # <-- mega-cap section: e.g. [4, 5] for Mega + Ultra Mega
                                 #     See scorer_mr.CAP_TIERS / CAP_TIER_LABELS.
WATCHLIST              = ["BABA", "V","UNH","GOOG","AMZN","NFLX","EBAY","AVGO","VT","COPX","IEMG"]   # <-- always-shown tickers, any yfinance symbol
UNIVERSE               = "sp500"
DATA_FOLDER            = "data"
# ─────────────────────────────────────────────────────────────────────────────


def _in_target_tiers(market_cap: float, tiers: list[int]) -> bool:
    """True if market_cap falls inside ANY of the requested tiers."""
    for tier in tiers:
        lo, hi = scorer_mr.CAP_TIERS.get(tier, (0, float("inf")))
        if lo <= market_cap < hi:
            return True
    return False


def _fmt(val, fmt: str, default: str = "N/A") -> str:
    """Safely formats a value that might be None (e.g. missing indicator,
    or not enough history yet for a newly-listed ticker)."""
    if val is None:
        return default
    return format(val, fmt)


def get_mega_cap_alerts() -> list[dict]:
    """Section 1: tier-filtered, threshold-gated. Returns [] if nothing qualifies."""
    all_tickers = get_tickers(UNIVERSE)
    sectors, market_caps = get_sectors_and_caps(
        all_tickers, folder_path=DATA_FOLDER, universe_name=UNIVERSE,
    )

    mega_tickers = [
        t for t in all_tickers
        if market_caps.get(t) is not None and _in_target_tiers(market_caps[t], ALERT_CAP_TIERS)
    ]
    print(f"   [Mega-cap] {len(mega_tickers)} tickers match tiers {ALERT_CAP_TIERS}")

    if not mega_tickers:
        return []

    universe_tag = "_".join(str(t) for t in sorted(ALERT_CAP_TIERS))
    data = download_sp500_data(
        mega_tickers,
        folder_path=DATA_FOLDER,
        universe_name=f"{UNIVERSE}_tiers{universe_tag}",
    )
    if data is None or data.empty:
        print("   [Mega-cap] ⚠️ Data download failed — skipping this section.")
        return []

    # NOTE: no cap_tier= passed to compute_scores() — the ticker list is
    # ALREADY filtered to the requested tier(s) above, since compute_scores()
    # only supports one tier at a time. Passing the pre-filtered `data` is enough.
    scored = scorer_mr.compute_scores(
        data           = data,
        sectors        = sectors,
        market_caps    = market_caps,
        top_n          = len(mega_tickers),
        max_per_sector = len(mega_tickers),
        min_score      = ALERT_SCORE_THRESHOLD,
        verbose        = False,
    )
    print(f"   [Mega-cap] {len(scored)} ticker(s) scored >= {ALERT_SCORE_THRESHOLD}")
    return scored


def _diagnose_exclusion(ticker: str, data) -> str:
    """
    With RSI/volume hard filters disabled for the watchlist (see
    get_watchlist_scores), the only way a ticker can still be missing is a
    genuine data problem: not found, or not enough history to compute
    anything (~63 trading days minimum — this floor can't be bypassed,
    without it the indicators would just be NaN).
    """
    try:
        df = data[ticker].dropna(subset=["Close"])
    except (KeyError, TypeError):
        return "no data (yfinance couldn't find this ticker)"

    min_needed = max(scorer_mr.BB_PERIOD, scorer_mr.MR_PERIOD // 4, scorer_mr.RSI_PERIOD) + 5
    if len(df) < min_needed:
        return f"insufficient history ({len(df)} days, need {min_needed}+)"

    price = df["Close"].iloc[-1]
    if price <= 0 or pd.isna(price):
        return "invalid price data"

    return "unknown (check manually — unexpected)"


def get_watchlist_scores() -> tuple[list[dict], list[tuple[str, str]]]:
    """Section 2: fixed tickers, ALWAYS returned regardless of score, RSI,
    or volume — the only thing that can still exclude a ticker here is a
    genuine data problem (see _diagnose_exclusion). Unlike the mega-cap
    section, this one bypasses every hard filter on purpose."""
    if not WATCHLIST:
        return [], []

    sectors, market_caps = get_sectors_and_caps(
        WATCHLIST, folder_path=DATA_FOLDER, universe_name="custom_watchlist",
    )
    data = download_sp500_data(
        WATCHLIST, folder_path=DATA_FOLDER, universe_name="custom_watchlist",
    )
    if data is None or data.empty:
        print("   [Watchlist] ⚠️ Data download failed — skipping this section.")
        return [], [(t, "data download failed") for t in WATCHLIST]

    scored = scorer_mr.compute_scores(
        data           = data,
        sectors        = sectors,
        market_caps    = market_caps,
        top_n          = len(WATCHLIST),
        max_per_sector = len(WATCHLIST),
        min_score      = -999,        # show every ticker, regardless of score
        rsi_max        = 999,         # disable RSI hard filter — show even overbought names
        min_avg_volume = 0,           # disable volume hard filter
        verbose        = False,
    )
    scored_tickers = {s["ticker"] for s in scored}
    missing = [(t, _diagnose_exclusion(t, data)) for t in WATCHLIST if t not in scored_tickers]
    print(f"   [Watchlist] {len(scored)}/{len(WATCHLIST)} tickers scored"
          + (f" — missing: {', '.join(f'{t} ({r})' for t, r in missing)}" if missing else ""))
    return scored, missing


def main():
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    print(f"🔍 Alert check — {now_str}")

    mega_scored = get_mega_cap_alerts()
    watchlist_scored, watchlist_missing = get_watchlist_scores()

    if not mega_scored and not watchlist_scored:
        print("   Nothing to report — no email sent.")
        return

    send_email(mega_scored, watchlist_scored, watchlist_missing, now_str)


def _format_ticker_block(s: dict) -> list[str]:
    """Formats one ticker's full indicator breakdown — shared by both sections."""
    pcr_str  = f"PCR={s['pcr_volume']:.1f}({s['s_pcr']:.0f})" if s.get("has_pcr") else "PCR=N/A"
    rr_str   = f"{s['rr_ratio']:.1f}x" if s.get("rr_ratio") else "N/A"
    # ETFs have no marketCap in yfinance (they use totalAssets instead) —
    # market_cap comes back None for them, so this must be handled safely
    # or the whole email crashes the moment an ETF makes it into either section.
    mcap_str = f"${s['market_cap']/1e9:,.0f}B" if s.get("market_cap") else "Cap: N/A"

    return [
        f"{s['ticker']}  Score: {s['composite_score']:.1f}  {s['setup']}  "
        f"{mcap_str} ({s['cap_tier_label']})  [{s['sector']}]",
        f"  RSI={s['rsi']:.0f}({s['s_rsi']:.0f})   "
        f"BB%={_fmt(s['pct_b'], '.2f')}({s['s_bb']:.0f})   "
        f"Z={_fmt(s['z_score'], '+.2f')}({s['s_mr']:.0f})   "
        f"{pcr_str}",
        f"  StochRSI_K={s['stoch_k']:.2f}({s['s_stochrsi']:.0f})   "
        f"Williams%R={s['williams_r']:.0f}({s['s_williams']:.0f})   "
        f"ATR_Pct={s['atr_percentile']:.0f}% {s['atr_pct_label']}   "
        f"VolRatio={s['vol_ratio']:.1f}x",
        f"  Price: ${s['price']}   ATR: ${s['atr']}   "
        f"Entry: ${s['entry_low']}-${s['entry_high']}   "
        f"Stop: ${s['stop_loss']}   T2: ${s['target_2']}   R/R: {rr_str}   "
        f"AvgVol: {s['avg_volume_m']}M",
        "",
    ]


def send_email(
    mega_scored:        list[dict],
    watchlist_scored:   list[dict],
    watchlist_missing:  list[tuple[str, str]],
    now_str:            str,
) -> None:
    lines = [f"Daily Alert — {now_str}", ""]

    lines.append("═" * 50)
    lines.append(f"MEGA-CAP ALERTS (score >= {ALERT_SCORE_THRESHOLD}, tiers {ALERT_CAP_TIERS})")
    lines.append("═" * 50)
    if mega_scored:
        for s in mega_scored:
            lines.extend(_format_ticker_block(s))
    else:
        lines.append("No qualifying mega-cap opportunities today.")
        lines.append("")

    if WATCHLIST:
        lines.append("═" * 50)
        lines.append("WATCHLIST (always shown, regardless of score)")
        lines.append("═" * 50)
        for s in sorted(watchlist_scored, key=lambda x: -x["composite_score"]):
            lines.extend(_format_ticker_block(s))
        if watchlist_missing:
            lines.append("Not scored (reason shown per ticker):")
            for ticker, reason in watchlist_missing:
                lines.append(f"  {ticker}: {reason}")
            lines.append("")

    lines.append("─" * 50)
    lines.append("Generated automatically by mega_cap_alert.py")

    body = "\n".join(lines)

    gmail_user  = os.environ["GMAIL_USER"]
    app_pw      = os.environ["GMAIL_APP_PASSWORD"]
    recipient   = os.environ.get("ALERT_RECIPIENT", gmail_user)

    subject_bits = []
    if mega_scored:
        subject_bits.append(f"{len(mega_scored)} mega-cap")
    if WATCHLIST:
        subject_bits.append(f"{len(watchlist_scored)} watchlist")
    subject = "📊 Daily Alert: " + ", ".join(subject_bits)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = gmail_user
    msg["To"]      = recipient

    with smtplib.SMTP_SSL("smtp.mail.yahoo.com", 465) as server:
        server.login(gmail_user, app_pw)
        server.send_message(msg)

    print("   ✅ Email sent.")


if __name__ == "__main__":
    main()
