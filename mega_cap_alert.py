# mega_cap_alert.py
"""
Automated Mega-Cap Mean-Reversion Alert
─────────────────────────────────────────────────────────────────────────────
Standalone script — does NOT go through the notebook. Designed to run on a
GitHub Actions schedule (2x/day: market open + market close).

Scans ONLY the market-cap tier(s) you configure below (ALERT_CAP_TIERS — see
scorer_mr.CAP_TIERS), using the exact same compute_scores() logic as the
manual weekly notebook scan. If any ticker scores >= ALERT_SCORE_THRESHOLD,
sends an email alert via Gmail SMTP.

Only downloads OHLCV for the filtered subset, not the full S&P 500 — keeps
the Actions run fast and light regardless of which tier(s) you pick.

── CONFIG ─────────────────────────────────────────────────────────────────
Edit ALERT_SCORE_THRESHOLD and ALERT_CAP_TIERS below to set your own bar.
ALERT_CAP_TIERS is a list — e.g. [5] for Ultra Mega-cap only, or [4, 5] for
Mega-cap + Ultra Mega-cap together.

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
# CONFIG — edit this to set your own alert threshold
# ─────────────────────────────────────────────────────────────────────────────
ALERT_SCORE_THRESHOLD = 5.0     # <-- SET YOUR OWN THRESHOLD HERE
ALERT_CAP_TIERS       = [4,5]     # <-- SET YOUR OWN TIER(S) HERE, e.g. [4, 5] for
                                 #     Mega-cap + Ultra Mega-cap together.
                                 #     See scorer_mr.CAP_TIERS / CAP_TIER_LABELS
                                 #     for what each number means.
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


def main():
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    tier_labels = ", ".join(
        f"{t} ({scorer_mr.CAP_TIER_LABELS.get(t, '?')})" for t in ALERT_CAP_TIERS
    )
    print(f"🔍 Mega-cap alert check — {now_str}")
    print(f"   Threshold: score >= {ALERT_SCORE_THRESHOLD}, tiers: {tier_labels}")

    # ── 1. Get full universe + market caps (cached ~3 days, cheap) ────────────
    all_tickers = get_tickers(UNIVERSE)
    sectors, market_caps = get_sectors_and_caps(
        all_tickers, folder_path=DATA_FOLDER, universe_name=UNIVERSE,
    )

    # ── 2. Filter down to the target tier(s) BEFORE downloading OHLCV ────────
    # This is the key efficiency trick — we only ever download price history
    # for the tickers in the requested tier(s), not the full 500.
    mega_tickers = [
        t for t in all_tickers
        if market_caps.get(t) is not None and _in_target_tiers(market_caps[t], ALERT_CAP_TIERS)
    ]
    print(f"   {len(mega_tickers)} tickers match the selected tier(s)")

    if not mega_tickers:
        print("   No tickers found in these tiers — nothing to scan.")
        return

    # ── 3. Download OHLCV for the filtered subset only ─────────────────────────
    universe_tag = "_".join(str(t) for t in sorted(ALERT_CAP_TIERS))
    data = download_sp500_data(
        mega_tickers,
        folder_path=DATA_FOLDER,
        universe_name=f"{UNIVERSE}_tiers{universe_tag}",
    )
    if data is None or data.empty:
        print("   ⚠️ Data download failed — aborting.")
        sys.exit(1)

    # ── 4. Score — same logic as the manual weekly scan ───────────────────────
    # NOTE: no cap_tier= passed here — the ticker list is ALREADY filtered to
    # the requested tier(s) above, since compute_scores() only supports one
    # tier at a time. Passing the pre-filtered `data` is enough.
    scored = scorer_mr.compute_scores(
        data           = data,
        sectors        = sectors,
        market_caps    = market_caps,
        top_n          = len(mega_tickers),   # don't truncate — we want all qualifiers
        max_per_sector = len(mega_tickers),   # no diversification cap for alerts
        min_score      = ALERT_SCORE_THRESHOLD,
        verbose        = False,
    )

    if not scored:
        print(f"   Nothing scored >= {ALERT_SCORE_THRESHOLD} today. No alert sent.")
        return

    print(f"   🚨 {len(scored)} mega-cap opportunity(ies) found — sending alert.")
    send_alert_email(scored, now_str)


def _fmt(val, fmt: str, default: str = "N/A") -> str:
    """Safely formats a value that might be None (e.g. missing indicator)."""
    if val is None:
        return default
    return format(val, fmt)


def send_alert_email(scored: list[dict], now_str: str) -> None:
    lines = [f"Mega-Cap Mean-Reversion Alert — {now_str}", ""]

    for s in scored:
        pcr_str = f"PCR={s['pcr_volume']:.1f}({s['s_pcr']:.0f})" if s.get("has_pcr") else "PCR=N/A"

        lines.append(
            f"{s['ticker']}  Score: {s['composite_score']:.1f}  {s['setup']}  "
            f"${s['market_cap']/1e9:,.0f}B ({s['cap_tier_label']})  [{s['sector']}]"
        )
        # Line 2: raw indicator values, with their sub-scores in parentheses —
        # same format as print_mr_report() in the notebook.
        lines.append(
            f"  RSI={s['rsi']:.0f}({s['s_rsi']:.0f})   "
            f"BB%={_fmt(s['pct_b'], '.2f')}({s['s_bb']:.0f})   "
            f"Z={_fmt(s['z_score'], '+.2f')}({s['s_mr']:.0f})   "
            f"{pcr_str}"
        )
        lines.append(
            f"  StochRSI_K={s['stoch_k']:.2f}({s['s_stochrsi']:.0f})   "
            f"Williams%R={s['williams_r']:.0f}({s['s_williams']:.0f})   "
            f"ATR_Pct={s['atr_percentile']:.0f}% {s['atr_pct_label']}   "
            f"VolRatio={s['vol_ratio']:.1f}x"
        )
        # Line 4: price / trade levels
        rr_str = f"{s['rr_ratio']:.1f}x" if s.get("rr_ratio") else "N/A"
        lines.append(
            f"  Price: ${s['price']}   ATR: ${s['atr']}   "
            f"Entry: ${s['entry_low']}-${s['entry_high']}   "
            f"Stop: ${s['stop_loss']}   T2: ${s['target_2']}   R/R: {rr_str}   "
            f"AvgVol: {s['avg_volume_m']}M"
        )
        lines.append("")

    lines.append("─" * 50)
    lines.append("Generated automatically by mega_cap_alert.py")

    body = "\n".join(lines)

    gmail_user  = os.environ["GMAIL_USER"]
    app_pw      = os.environ["GMAIL_APP_PASSWORD"]
    recipient   = os.environ.get("ALERT_RECIPIENT", gmail_user)

    msg = MIMEText(body)
    msg["Subject"] = f"🚨 Mega-Cap MR Alert: {len(scored)} opportunity(ies)"
    msg["From"]    = gmail_user
    msg["To"]      = recipient

    with smtplib.SMTP_SSL("smtp.mail.yahoo.com", 465) as server:
        server.login(gmail_user, app_pw)
        server.send_message(msg)

    print("   ✅ Email sent.")


if __name__ == "__main__":
    main()
