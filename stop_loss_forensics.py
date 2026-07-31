# stop_loss_forensics.py
"""
Stop Loss Forensics — Post-mortem analysis of stopped-out trades.

For every trade that closed via a stop (guard / breakeven / trail),
this module looks forward N days and asks:
  "What did the stock do AFTER we were stopped out?"

This reveals whether stops are triggering prematurely and how much
opportunity cost each stop phase is generating.

Usage:
    from stop_loss_forensics import StopLossForensics

    forensics = StopLossForensics(data)
    forensics.run(trades_df)
    forensics.print_report()
    forensics.plot_distributions()
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ── Constants ─────────────────────────────────────────────────────────────────

# Stop reasons produced by the backtester
STOP_REASONS = ["stop_guard", "stop_breakeven", "stop_trail"]

# Forward-looking windows (trading days)
WINDOWS = [20, 40, 60]

# Colour palette per stop type
COLOURS = {
    "stop_guard":      "#e74c3c",   # red
    "stop_breakeven":  "#e67e22",   # orange
    "stop_trail":      "#3498db",   # blue
}

LABELS = {
    "stop_guard":      "Guard (Phase 1)",
    "stop_breakeven":  "Breakeven (Phase 2)",
    "stop_trail":      "Trailing (Phase 3)",
}


# ── Main class ────────────────────────────────────────────────────────────────

class StopLossForensics:
    """
    Analyses post-exit price behaviour for stopped-out trades.

    Parameters:
        data : pd.DataFrame
            Multi-level OHLCV DataFrame (ticker -> OHLCV columns),
            as produced by download_sp500_data().
    """

    def __init__(self, data: pd.DataFrame, output_path: str = "."):
        self.data        = data
        self.output_path = output_path
        self.trades_df   = None
        self.stops_df    = None
        self.summary     = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        trades_df:  pd.DataFrame,
        windows:    list[int] = WINDOWS,
        stop_types: list[str] = STOP_REASONS,
    ) -> "StopLossForensics":
        """
        Run the full forensics analysis.

        Parameters:
            trades_df  : backtest trades DataFrame (from results["trades"])
            windows    : forward-look windows in calendar days
            stop_types : which exit reasons to analyse
        """
        self.trades_df  = trades_df.copy()
        self.windows    = windows
        self.stop_types = stop_types

        print("🔬 Stop Loss Forensics — starting analysis...")
        print(f"   Total trades     : {len(trades_df)}")

        # Filter to stop trades only
        stops = trades_df[trades_df["exit_reason"].isin(stop_types)].copy()
        print(f"   Stop trades      : {len(stops)}")
        print(f"   Windows          : {windows} days\n")

        if stops.empty:
            print("⚠️  No stop trades found.")
            return self

        # Ensure date columns are datetime
        stops["exit_date"] = pd.to_datetime(stops["exit_date"])

        # ── Compute forward returns for each stopped trade ────────────────────
        records = []
        errors  = 0

        for _, row in stops.iterrows():
            ticker    = row["ticker"]
            exit_date = row["exit_date"]
            exit_px   = row["exit_price"]

            # Get ticker OHLCV
            if ticker not in self.data.columns.get_level_values(0):
                errors += 1
                continue

            ticker_df = self.data[ticker].dropna()
            future    = ticker_df[ticker_df.index > exit_date]

            if future.empty:
                errors += 1
                continue

            rec = {
                "ticker":       ticker,
                "entry_date":   row["entry_date"],
                "exit_date":    exit_date,
                "entry_price":  row["entry_price"],
                "exit_price":   exit_px,
                "exit_reason":  row["exit_reason"],
                "pnl_pct":      row["pnl_pct"],
                "hold_days":    row["hold_days"],
                "signal_score": row.get("signal_score", np.nan),
                "regime":       row.get("regime", "fixed"),
            }

            # Forward returns at each window
            for w in windows:
                col = f"fwd_{w}d"
                # Find the trading day closest to exit_date + w calendar days
                target_date = exit_date + pd.Timedelta(days=w)
                future_w    = future[future.index <= target_date]

                if future_w.empty:
                    rec[col] = np.nan
                else:
                    px_w      = future_w["Close"].iloc[-1]
                    rec[col]  = round((px_w - exit_px) / exit_px * 100, 2)

            # Max adverse excursion after stop (20d window)
            future_20 = future.iloc[:20] if len(future) >= 20 else future
            if not future_20.empty:
                worst_px        = future_20["Low"].min()
                rec["mae_post"] = round((worst_px - exit_px) / exit_px * 100, 2)
                # Also: how many days until it hit a new low vs exit price
                below_exit      = future_20[future_20["Low"] < exit_px]
                rec["days_to_new_low"] = (
                    (below_exit.index[0] - exit_date).days
                    if not below_exit.empty else None
                )
            else:
                rec["mae_post"]        = np.nan
                rec["days_to_new_low"] = None

            records.append(rec)

        self.stops_df = pd.DataFrame(records)

        if errors:
            print(f"   ⚠️  {errors} trades skipped (ticker not in data)\n")

        # ── Aggregate stats per stop type ─────────────────────────────────────
        self._compute_summary()

        print(f"✅ Analysis complete — {len(self.stops_df)} trades analysed\n")
        return self

    # ── Report ────────────────────────────────────────────────────────────────

    def print_report(self) -> None:
        """Print the full forensics report."""

        if self.stops_df is None or self.stops_df.empty:
            print("⚠️  Run .run(trades_df) first.")
            return

        print(f"\n{'═'*70}")
        print(f"  STOP LOSS FORENSICS REPORT")
        print(f"  Total stop trades analysed: {len(self.stops_df)}")
        print(f"{'═'*70}")

        # ── Section 1: Per stop type summary ──────────────────────────────────
        print(f"\n  📊 BREAKDOWN BY STOP TYPE")
        print(f"  {'─'*66}")

        for stop_type in self.stop_types:
            if stop_type not in self.summary:
                continue

            s     = self.summary[stop_type]
            label = LABELS.get(stop_type, stop_type)
            n     = s["n"]

            print(f"\n  {'▶ ' + label + ' (' + stop_type + ')'}")
            print(f"  {'─'*50}")
            print(f"    Trades          : {n}")
            print(f"    Avg PnL at exit : {s['avg_pnl_exit']:+.2f}%")
            print(f"    % trades that went up afterward:")

            for w in self.windows:
                col      = f"fwd_{w}d"
                pct_up   = s.get(f"pct_up_{w}d",   np.nan)
                med_ret  = s.get(f"median_fwd_{w}d", np.nan)
                avg_ret  = s.get(f"avg_fwd_{w}d",   np.nan)
                opp_cost = s.get(f"opp_cost_{w}d",  np.nan)

                if np.isnan(pct_up):
                    continue

                # Visual bar for % up
                bar_len = int(pct_up / 5)
                bar     = "█" * bar_len

                print(
                    f"      {w:>3}d: {pct_up:>5.1f}% up  "
                    f"median={med_ret:>+6.2f}%  avg={avg_ret:>+6.2f}%  "
                    f"opp.cost={opp_cost:>+6.2f}%  {bar}"
                )

            # MAE post-stop
            mae = s.get("avg_mae_post", np.nan)
            if not np.isnan(mae):
                print(f"    Avg max drop after stop (20d) : {mae:+.2f}%")

            # Interpretation
            self._print_interpretation(stop_type, s)

        # ── Section 2: Worst opportunity cost cases ────────────────────────────
        print(f"\n\n  🔥 TOP 10 MISSED OPPORTUNITIES (stop -> stock went up more)")
        print(f"  {'─'*66}")

        col_40 = "fwd_40d"
        if col_40 in self.stops_df.columns:
            top_opp = (
                self.stops_df
                .dropna(subset=[col_40])
                .nlargest(10, col_40)
                [["ticker", "exit_date", "exit_reason", "pnl_pct", col_40, "signal_score"]]
            )
            if not top_opp.empty:
                print(f"  {'Ticker':<8} {'Exit date':<12} {'Stop type':<18} "
                      f"{'PnL exit':>9} {'Fwd 40d':>8} {'Score':>6}")
                print(f"  {'─'*65}")
                for _, r in top_opp.iterrows():
                    print(
                        f"  {r['ticker']:<8} {str(r['exit_date'])[:10]:<12} "
                        f"{r['exit_reason']:<18} {r['pnl_pct']:>+8.2f}% "
                        f"{r[col_40]:>+7.2f}%  {r['signal_score']:>5.1f}"
                    )

        # ── Section 3: Trades where stop was correct (stock kept falling) ──────
        print(f"\n\n  ✅ TOP 10 CORRECT STOPS (stock kept falling)")
        print(f"  {'─'*66}")

        if col_40 in self.stops_df.columns:
            good_stops = (
                self.stops_df
                .dropna(subset=[col_40])
                .nsmallest(10, col_40)
                [["ticker", "exit_date", "exit_reason", "pnl_pct", col_40, "signal_score"]]
            )
            if not good_stops.empty:
                print(f"  {'Ticker':<8} {'Exit date':<12} {'Stop type':<18} "
                      f"{'PnL exit':>9} {'Fwd 40d':>8} {'Score':>6}")
                print(f"  {'─'*65}")
                for _, r in good_stops.iterrows():
                    print(
                        f"  {r['ticker']:<8} {str(r['exit_date'])[:10]:<12} "
                        f"{r['exit_reason']:<18} {r['pnl_pct']:>+8.2f}% "
                        f"{r[col_40]:>+7.2f}%  {r['signal_score']:>5.1f}"
                    )

        # ── Section 4: Recommendation ─────────────────────────────────────────
        self._print_recommendation()

        print(f"\n{'═'*70}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────

    def plot_distributions(self, figsize: tuple = (16, 10)) -> None:
        """
        Plot forward return distributions per stop type.
        """
        if self.stops_df is None or self.stops_df.empty:
            print("⚠️  Run .run(trades_df) first.")
            return

        n_windows = len(self.windows)
        n_types   = len([t for t in self.stop_types if t in self.summary])

        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            "Stop Loss Forensics — Forward Return Distributions",
            fontsize=14, fontweight="bold", y=1.01
        )

        gs   = gridspec.GridSpec(n_types, n_windows, figure=fig, hspace=0.5, wspace=0.35)
        row  = 0

        for stop_type in self.stop_types:
            if stop_type not in self.summary:
                continue

            colour = COLOURS.get(stop_type, "#95a5a6")
            label  = LABELS.get(stop_type, stop_type)
            subset = self.stops_df[self.stops_df["exit_reason"] == stop_type]

            for col_idx, w in enumerate(self.windows):
                col_name = f"fwd_{w}d"
                ax       = fig.add_subplot(gs[row, col_idx])
                vals     = subset[col_name].dropna()

                if vals.empty:
                    ax.set_visible(False)
                    continue

                # Histogram
                ax.hist(
                    vals, bins=25, color=colour, alpha=0.75,
                    edgecolor="white", linewidth=0.5
                )

                # Zero line
                ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

                # Median line
                med = vals.median()
                ax.axvline(
                    med, color=colour, linewidth=1.8, linestyle="-",
                    label=f"median={med:+.1f}%"
                )

                # % above zero annotation
                pct_up = (vals > 0).mean() * 100
                ax.text(
                    0.97, 0.95, f"{pct_up:.0f}% ↑",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, fontweight="bold",
                    color="#27ae60" if pct_up >= 50 else "#e74c3c"
                )

                ax.set_title(
                    f"{label}\n{w}d forward  (n={len(vals)})",
                    fontsize=8.5, pad=4
                )
                ax.set_xlabel("Return after stop exit (%)", fontsize=7.5)
                ax.set_ylabel("Frequency", fontsize=7.5)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7, loc="upper left")

            row += 1

        plt.tight_layout()
        save_path = os.path.join(self.output_path, "stop_loss_forensics.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"📊 Chart saved: {save_path}")

    def plot_opportunity_cost(self, figsize: tuple = (12, 5)) -> None:
        """
        Bar chart: average opportunity cost per stop type x window.
        """
        if not self.summary:
            print("⚠️  Run .run(trades_df) first.")
            return

        fig, ax = plt.subplots(figsize=figsize)

        x       = np.arange(len(self.windows))
        width   = 0.25
        offsets = np.linspace(-width, width, len(self.stop_types))

        for i, stop_type in enumerate(self.stop_types):
            if stop_type not in self.summary:
                continue
            s      = self.summary[stop_type]
            costs  = [s.get(f"opp_cost_{w}d", 0) for w in self.windows]
            colour = COLOURS.get(stop_type, "#95a5a6")
            label  = LABELS.get(stop_type, stop_type)

            bars = ax.bar(
                x + offsets[i], costs, width * 0.9,
                label=label, color=colour, alpha=0.8, edgecolor="white"
            )

            # Value labels on bars
            for bar, val in zip(bars, costs):
                if not np.isnan(val) and val != 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{val:+.1f}%",
                        ha="center", va="bottom", fontsize=8
                    )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{w}d forward" for w in self.windows])
        ax.set_ylabel("Avg return after stop exit (%)")
        ax.set_title(
            "Opportunity Cost per Stop Phase\n"
            "(positive = stock went up after we exited)",
            fontweight="bold"
        )
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_path, "stop_opportunity_cost.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"📊 Chart saved: {save_path}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_summary(self) -> None:
        """Aggregate stats per stop type."""

        df = self.stops_df

        for stop_type in self.stop_types:
            subset = df[df["exit_reason"] == stop_type]
            if subset.empty:
                continue

            s = {
                "n":            len(subset),
                "avg_pnl_exit": subset["pnl_pct"].mean(),
                "avg_mae_post": subset["mae_post"].mean() if "mae_post" in subset else np.nan,
            }

            for w in self.windows:
                col  = f"fwd_{w}d"
                vals = subset[col].dropna()
                if vals.empty:
                    continue
                s[f"pct_up_{w}d"]    = (vals > 0).mean() * 100
                s[f"median_fwd_{w}d"] = vals.median()
                s[f"avg_fwd_{w}d"]    = vals.mean()
                # Opportunity cost = avg return AFTER stop
                # (positive means we left money on the table)
                s[f"opp_cost_{w}d"]  = vals.mean()

            self.summary[stop_type] = s

    def _print_interpretation(self, stop_type: str, s: dict) -> None:
        """Print a plain-language interpretation for each stop type."""

        n       = s["n"]
        pct_40  = s.get("pct_up_40d",   np.nan)
        med_40  = s.get("median_fwd_40d", np.nan)

        if np.isnan(pct_40):
            return

        print(f"\n    💬 Interpretation:")

        if stop_type == "stop_breakeven":
            if pct_40 >= 65:
                print(
                    f"    ⚠️  ISSUE: {pct_40:.0f}% of breakeven trades went up "
                    f"(median +{med_40:.1f}%) 40 days later.\n"
                    f"    Phase 2 is closing too early. Try BREAKEVEN_BUFFER_ATR = 1.0-1.5."
                )
            elif pct_40 >= 50:
                print(
                    f"    🟡 BORDERLINE: {pct_40:.0f}% went up afterward. The breakeven stop "
                    f"prevents losses but misses some opportunities."
                )
            else:
                print(
                    f"    ✅ OK: only {pct_40:.0f}% went up afterward. "
                    f"The breakeven stop is working correctly."
                )

        elif stop_type == "stop_guard":
            if pct_40 >= 60:
                print(
                    f"    ⚠️  ISSUE: {pct_40:.0f}% of guard stops went up afterward "
                    f"(median +{med_40:.1f}%). HARD_FLOOR_ATR={s.get('hard_floor_note','?')} "
                    f"is too close. Try GUARD_DAYS ↑ or HARD_FLOOR_ATR ↑."
                )
            else:
                print(
                    f"    ✅ OK: {pct_40:.0f}% went up afterward — the guard stop "
                    f"is catching real losses."
                )

        elif stop_type == "stop_trail":
            if pct_40 >= 55:
                print(
                    f"    🟡 BORDERLINE: {pct_40:.0f}% of trailing stops went up afterward. "
                    f"ATR_TRAIL_MULT might be a bit tight."
                )
            else:
                print(
                    f"    ✅ OK: {pct_40:.0f}% went up afterward. "
                    f"The trailing stop is working correctly — catching the winners."
                )

    def _print_recommendation(self) -> None:
        """Print consolidated parameter recommendations."""

        print(f"\n\n  🎯 PARAMETER RECOMMENDATIONS")
        print(f"  {'─'*66}")

        recommendations = []

        for stop_type, s in self.summary.items():
            pct_40 = s.get("pct_up_40d",    np.nan)
            med_40 = s.get("median_fwd_40d", np.nan)
            opp_40 = s.get("opp_cost_40d",  np.nan)

            if np.isnan(pct_40):
                continue

            if stop_type == "stop_breakeven" and pct_40 >= 65:
                recommendations.append(
                    f"  • BREAKEVEN_BUFFER_ATR = 1.0-1.5\n"
                    f"    ({pct_40:.0f}% of breakeven trades went up +{med_40:.1f}% median)\n"
                    f"    Give some room below the entry instead of exactly at entry price."
                )

            elif stop_type == "stop_guard" and pct_40 >= 60:
                recommendations.append(
                    f"  • GUARD_DAYS ↑ (from {15} -> 20-25d) or HARD_FLOOR_ATR ↑\n"
                    f"    ({pct_40:.0f}% of guard stops went up afterward)"
                )

            elif stop_type == "stop_trail" and pct_40 >= 55:
                recommendations.append(
                    f"  • ATR_TRAIL_MULT ↑ (from 2.5 -> 3.0-3.5)\n"
                    f"    ({pct_40:.0f}% of trail stops went up afterward)"
                )

        if recommendations:
            for rec in recommendations:
                print(rec)
                print()
        else:
            print("  ✅ Stop loss parameters look optimized.")
            print("     No significant opportunity cost found.")

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_stops_df(self) -> pd.DataFrame:
        """Return the enriched stop trades DataFrame."""
        return self.stops_df.copy() if self.stops_df is not None else pd.DataFrame()

    def get_summary(self) -> dict:
        """Return the summary statistics dict."""
        return self.summary.copy()
