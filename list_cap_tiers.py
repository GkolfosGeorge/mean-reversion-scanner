# list_cap_tiers.py
"""
Lists all S&P 500 tickers grouped by market cap tier (see scorer_mr.CAP_TIERS).
Quick utility to see exactly which companies fall into each tier before
configuring cap_tier / ALERT_CAP_TIERS filters elsewhere.

Usage:
    python list_cap_tiers.py
"""

from ticker_provider import get_tickers
from sector_lookup import get_sectors_and_caps
import scorer_mr

UNIVERSE = "sp500"


def main():
    tickers = get_tickers(UNIVERSE)
    sectors, market_caps = get_sectors_and_caps(
        tickers, folder_path="data", universe_name=UNIVERSE,
    )

    by_tier = {t: [] for t in scorer_mr.CAP_TIERS}
    unknown = []

    for ticker in tickers:
        mcap = market_caps.get(ticker)
        if mcap is None:
            unknown.append(ticker)
            continue
        for tier, (lo, hi) in scorer_mr.CAP_TIERS.items():
            if lo <= mcap < hi:
                by_tier[tier].append((ticker, mcap))
                break

    for tier in sorted(by_tier, reverse=True):
        label = scorer_mr.CAP_TIER_LABELS[tier]
        lo, hi = scorer_mr.CAP_TIERS[tier]
        hi_str = f"${hi/1e9:,.0f}B" if hi != float("inf") else "no limit"
        names = sorted(by_tier[tier], key=lambda x: -x[1])   # biggest first

        print(f"\n=== Tier {tier}: {label} (${lo/1e9:,.0f}B - {hi_str}) — {len(names)} tickers ===")
        for ticker, mcap in names:
            print(f"  {ticker:<6} ${mcap/1e9:,.0f}B")

    if unknown:
        print(f"\n=== Unknown market cap ({len(unknown)} tickers) ===")
        print(", ".join(sorted(unknown)))


if __name__ == "__main__":
    main()
