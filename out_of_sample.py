# out_of_sample.py
"""
Out-of-sample holdout enforcement — Phase 2.7 of the roadmap.

The idea: reserve a trailing slice of history that NEVER appears in any
tuning, rolling-window (2.6), or future parameter-sweep (2.12) run. Test
the final, locked configuration against it exactly once, at the very end.
Any tuning done AFTER looking at the holdout invalidates the test.

This module provides two things:

1. `OOS_START` — single source of truth for where the holdout begins.
   Rolling-windows (2.6) and any future parameter sweep (2.12) should
   clamp their date range through `clamp_to_pre_oos()` so they can never
   accidentally tune on holdout data.

2. `run_oos_test()` — the actual holdout run. Every call appends to an
   on-disk log (`oos_test_log.jsonl` inside DATA_FOLDER). If the log
   already has entries, it prints them loudly before running again, so
   you can never silently lose track of how many times you've "peeked" —
   repeated peeking turns an out-of-sample test back into curve-fitting.
"""

import hashlib
import json
import os
from datetime import datetime

import pandas as pd

# ── Single source of truth for where the holdout begins ────────────────────
# Change this ONE constant if you want to resize the holdout window.
# Everything else (rolling windows, future sweeps) reads from here.
OOS_START = "2025-08-01"   # adjust to reserve the last ~6-12 months you want


def clamp_to_pre_oos(end_date: str) -> str:
    """
    Utility for any tuning loop (rolling windows, parameter sweep): clamps
    a requested end_date so it can never cross into the OOS holdout,
    regardless of what the caller passes in.
    """
    end_ts = pd.Timestamp(end_date)
    oos_ts = pd.Timestamp(OOS_START)
    return min(end_ts, oos_ts).strftime("%Y-%m-%d")


def _log_path(data_folder: str) -> str:
    return os.path.join(data_folder, "oos_test_log.jsonl")


def _read_log(data_folder: str) -> list[dict]:
    path = _log_path(data_folder)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_oos_test(
    backtester_module,
    data_folder: str,
    confirm: bool = False,
    **backtest_kwargs,
) -> dict | None:
    """
    Runs run_backtest() ONCE over [OOS_START, end_of_data] and logs it.

    Parameters
    ----------
    backtester_module : the imported `backtester` module (pass it in so
        this module has no hard dependency on it — keeps it usable
        standalone/testable).
    data_folder : where to read/write oos_test_log.jsonl (same DATA_FOLDER
        used elsewhere in the notebook).
    confirm : must be explicitly True. Defaults to False on purpose — this
        is a deliberate, one-time action, not something that should fire
        from a stray re-run of the cell.
    **backtest_kwargs : everything run_backtest() needs EXCEPT start_date
        (fixed to OOS_START) — pass end_date, regime_detector,
        scorer_weights, top_n, signal_threshold, etc. as usual, with your
        FINAL, LOCKED configuration.

    Returns
    -------
    The same dict run_backtest() returns, or None if confirm=False.
    """
    if not confirm:
        print("⛔ run_oos_test() requires confirm=True — this is your one honest "
              "look at the holdout, don't fire it as a default.")
        return None

    prior_runs = _read_log(data_folder)
    if prior_runs:
        print(f"⚠️  WARNING: the OOS holdout has ALREADY been tested {len(prior_runs)} time(s):")
        for r in prior_runs:
            print(f"    {r['timestamp']}  mode={r.get('mode')}  params_hash={r.get('params_hash')}  "
                  f"total_return={r.get('total_return')}  sharpe={r.get('sharpe_ratio')}")
        print("    Running it again defeats the purpose of an out-of-sample test.")
        print("    Only proceed if this is genuinely a new, final, locked configuration —")
        print("    not a reaction to not liking the previous result.\n")

    end_date = backtest_kwargs.pop("end_date", None) or pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"🔒 OUT-OF-SAMPLE TEST — {OOS_START} → {end_date}  (single-shot, logged)")
    results = backtester_module.run_backtest(
        start_date=OOS_START,
        end_date=end_date,
        **backtest_kwargs,
    )

    # Coarse hash of the tunable params (not the full dict) — enough to spot
    # "did anything change since the last time I looked".
    relevant = {k: v for k, v in backtest_kwargs.items() if k not in ("data", "membership")}
    params_hash = hashlib.sha256(
        json.dumps(relevant, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]

    s = results["summary"]
    entry = {
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "oos_start":      OOS_START,
        "oos_end":        end_date,
        "mode":           s.get("mode"),
        "params_hash":    params_hash,
        "total_return":   s.get("total_return"),
        "annual_return":  s.get("annual_return"),
        "sharpe_ratio":   s.get("sharpe_ratio"),
        "sortino_ratio":  s.get("sortino_ratio"),
        "calmar_ratio":   s.get("calmar_ratio"),
        "max_drawdown":   s.get("max_drawdown"),
        "outperformance": s.get("outperformance"),
        "n_trades":       s.get("n_trades"),
    }
    os.makedirs(data_folder, exist_ok=True)
    with open(_log_path(data_folder), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\n📝 Logged to {_log_path(data_folder)} — this run is now permanently on record.")
    return results
