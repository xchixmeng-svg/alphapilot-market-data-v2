#!/usr/bin/env python3
"""Research-only persistence gate over the locked R10 MAX formal engine.

This does NOT modify the locked strategy. It reproduces the immutable formal baseline,
then reuses the exact stage-3 portfolio engine while adding one entry-only overlay:
a stock must be an otherwise-orderable R10 recommendation multiple times before the
BUY order is submitted.

Variants:
- P2_5: at least 2 recommendations within 5 trading sessions
- P3_5: at least 3 recommendations within 5 trading sessions
- P3_10: at least 3 recommendations within 10 trading sessions
- C3: 3 consecutive trading-session recommendations

All execution/accounting rules remain inherited from the locked engine: T+1, fixed
limit orders, adverse slippage, integer shares, common cash, fees/tax, corporate
actions, position/exposure caps, and the locked exits.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCKED_ENGINE = ROOT / "scripts" / "r10_max_formal.py"
LOCKED_ENGINE_SHA256 = "2fef3ba99b7c83b5db21e29c5f1c2abd8b3df77583840a025e6df937ff81c0d0"
LOCKED_END_NAV = 2403427.0678222505
LOCKED_TRADES = 150
LOCKED_WINS = 71
LOCKED_LOSSES = 79
LOCKED_MAX_DD = -0.19515438907459803
RUN_ROOT = ROOT / "persistence_results"
BASELINE_DIR = RUN_ROOT / "baseline"
DATA_DIR = ROOT / "data" / "history" / "2020-2025"
CORP = ROOT / "data" / "reference" / "official_corporate_actions_2020_2025.csv"

VARIANTS = [
    {"name": "P2_5", "min_count": 2, "window": 5, "consecutive": False},
    {"name": "P3_5", "min_count": 3, "window": 5, "consecutive": False},
    {"name": "P3_10", "min_count": 3, "window": 10, "consecutive": False},
    {"name": "C3", "min_count": 3, "window": 3, "consecutive": True},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def link_inputs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(DATA_DIR.glob("*.parquet")):
        dst = run_dir / path.name
        if not dst.exists():
            dst.symlink_to(path)
    dst = run_dir / CORP.name
    if not dst.exists():
        dst.symlink_to(CORP)


def run_python(script: Path, cwd: Path, log_name: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (cwd / log_name).write_text(proc.stdout, encoding="utf-8")
    print(f"[{cwd.name}] returncode={proc.returncode}")
    tail = "\n".join(proc.stdout.splitlines()[-20:])
    print(tail)
    if proc.returncode != 0:
        raise RuntimeError(f"run failed: {cwd.name}\n{tail}")


def verify_baseline() -> dict:
    result = json.loads((BASELINE_DIR / "r10max_formal_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((BASELINE_DIR / "contract_audit.json").read_text(encoding="utf-8"))
    s = result["strategy"]
    assert audit["all_pass"], audit
    assert abs(s["end_nav"] - LOCKED_END_NAV) < 1e-7, s
    assert s["completed_trades"] == LOCKED_TRADES, s
    assert s["wins"] == LOCKED_WINS and s["losses"] == LOCKED_LOSSES, s
    assert abs(s["max_drawdown"] - LOCKED_MAX_DD) < 1e-12, s
    return result


def build_variant_source(locked_source: str, cfg: dict) -> str:
    marker = (
        "# ============================================================\n"
        "# 步驟三：組合層回測引擎\n"
        "# ============================================================\n\n"
    )
    if marker not in locked_source:
        raise RuntimeError("stage-3 marker not found in locked engine")
    stage3 = locked_source.split(marker, 1)[1]
    old_load = "px = pd.read_pickle('r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)"
    new_load = "px = pd.read_pickle('../baseline/r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)"
    if old_load not in stage3:
        raise RuntimeError("stage-3 signal-load anchor not found")
    stage3 = stage3.replace(old_load, new_load, 1)

    state_anchor = "cash = INITIAL_CAPITAL\n"
    if state_anchor not in stage3:
        raise RuntimeError("cash-state anchor not found")
    injection = f'''cash = INITIAL_CAPITAL
# ---- Research-only recommendation persistence overlay ----
PERSISTENCE_MIN_COUNT = {int(cfg['min_count'])}
PERSISTENCE_WINDOW = {int(cfg['window'])}
PERSISTENCE_CONSECUTIVE = {bool(cfg['consecutive'])!r}
recommendation_history = {{}}
persistence_rows = []

def persistence_observe(di, idx, code, strat, limit, shares, nav, post_total, post_code):
    key = str(code)
    hist = recommendation_history.setdefault(key, [])
    if not hist or hist[-1] != idx:
        hist.append(idx)
    if PERSISTENCE_CONSECUTIVE:
        streak = 1
        for j in range(len(hist) - 1, 0, -1):
            if hist[j] - hist[j - 1] == 1:
                streak += 1
            else:
                break
        count = streak
        accepted = streak >= PERSISTENCE_MIN_COUNT
    else:
        hist[:] = [x for x in hist if idx - x < PERSISTENCE_WINDOW]
        count = len(hist)
        accepted = count >= PERSISTENCE_MIN_COUNT
    persistence_rows.append({{
        'signal_date': di, 'date_index': idx, 'code': key, 'strategy': strat,
        'count': count, 'accepted': bool(accepted), 'limit_price': limit,
        'shares_if_ordered': int(shares), 'pre_nav': nav,
        'post_total_exposure_if_ordered': post_total,
        'post_code_exposure_if_ordered': post_code,
    }})
    return bool(accepted)
'''
    stage3 = stage3.replace(state_anchor, injection, 1)

    gate_anchor = (
        "                if post_total > TOTAL_CAP + 1e-12 or post_code > SINGLE_CAP + 1e-12: continue\n"
        "                o = submit('BUY', di, exdate, code, strat, shares, 'ENTRY', limit,\n"
    )
    gate_replacement = (
        "                if post_total > TOTAL_CAP + 1e-12 or post_code > SINGLE_CAP + 1e-12: continue\n"
        "                if not persistence_observe(di, i, code, strat, limit, shares, nav, post_total, post_code):\n"
        "                    # A blocked recommendation consumes one recommendation slot for this day;\n"
        "                    # do not backfill with a lower-ranked stock that R10 would not have ordered.\n"
        "                    used.add(code)\n"
        "                    slots_free -= 1\n"
        "                    continue\n"
        "                o = submit('BUY', di, exdate, code, strat, shares, 'ENTRY', limit,\n"
    )
    if gate_anchor not in stage3:
        raise RuntimeError("BUY submit anchor not found")
    stage3 = stage3.replace(gate_anchor, gate_replacement, 1)

    fill_anchor = (
        "        complete_order(o, 'FILLED', fill_date=di, raw_fill_price=fill, gross_value=gross,\n"
        "                       fee=fee, tax=0.0, net_cash=-cost)\n"
    )
    fill_replacement = fill_anchor + "        recommendation_history.pop(str(code), None)\n"
    if fill_anchor not in stage3:
        raise RuntimeError("BUY fill anchor not found")
    stage3 = stage3.replace(fill_anchor, fill_replacement, 1)

    dump_anchor = "Path('contract_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')\n"
    dump_replacement = dump_anchor + "pd.DataFrame(persistence_rows).to_csv('persistence_observations.csv', index=False)\n"
    if dump_anchor not in stage3:
        raise RuntimeError("output anchor not found")
    stage3 = stage3.replace(dump_anchor, dump_replacement, 1)

    header = (
        "# Generated research runner from immutable R10 MAX stage-3 portfolio engine.\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import json\n"
        "from pathlib import Path\n\n"
    )
    return header + stage3


def compute_path_metrics(trades: pd.DataFrame, causal: pd.DataFrame) -> tuple[float, float, float, float]:
    if trades.empty:
        return np.nan, np.nan, np.nan, np.nan
    by_code = {str(code): sub.sort_values("date") for code, sub in causal.groupby("code")}
    maes, mfes = [], []
    for row in trades.itertuples(index=False):
        code = str(row.code).zfill(4)
        sub = by_code.get(code)
        if sub is None:
            continue
        path = sub[(sub.date >= int(row.entry_date)) & (sub.date <= int(row.exit_date))]
        entry = path[path.date == int(row.entry_date)]
        if path.empty or entry.empty:
            continue
        er = entry.iloc[0]
        if not np.isfinite(er.close) or er.close <= 0 or not np.isfinite(er.aclose):
            continue
        fill_index = float(row.entry_raw_price) * float(er.aclose) / float(er.close)
        if fill_index <= 0:
            continue
        low_col = "alow" if "alow" in path.columns else "aclose"
        high_col = "ahigh" if "ahigh" in path.columns else "aclose"
        maes.append(float(path[low_col].min() / fill_index - 1.0))
        mfes.append(float(path[high_col].max() / fill_index - 1.0))
    if not maes:
        return np.nan, np.nan, np.nan, np.nan
    return float(np.mean(maes)), float(np.median(maes)), float(np.mean(mfes)), float(np.median(mfes))


def collect_metrics(name: str, run_dir: Path, cfg: dict | None, causal: pd.DataFrame) -> dict:
    summary = json.loads((run_dir / "r10max_formal_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((run_dir / "contract_audit.json").read_text(encoding="utf-8"))
    if not audit["all_pass"]:
        raise RuntimeError(f"audit failed for {name}: {audit}")
    s = summary["strategy"]
    trades = pd.read_csv(run_dir / "r10max_formal_trades.csv", dtype={"code": str})
    wins = trades.loc[trades.pnl > 0, "return"] if len(trades) else pd.Series(dtype=float)
    losses = trades.loc[trades.pnl < 0, "return"] if len(trades) else pd.Series(dtype=float)
    avg_mae, median_mae, avg_mfe, median_mfe = compute_path_metrics(trades, causal)
    row = {
        "variant": name,
        "min_count": 1 if cfg is None else cfg["min_count"],
        "window_sessions": 1 if cfg is None else cfg["window"],
        "consecutive": False if cfg is None else cfg["consecutive"],
        "end_nav": s["end_nav"],
        "total_return": s["total_return"],
        "cagr": s["cagr"],
        "max_drawdown": s["max_drawdown"],
        "completed_trades": s["completed_trades"],
        "wins": s["wins"],
        "losses": s["losses"],
        "win_rate": s["win_rate"],
        "pnl_profit_factor": s["profit_factor"],
        "avg_win_return": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_return": float(losses.mean()) if len(losses) else np.nan,
        "avg_mae": avg_mae,
        "median_mae": median_mae,
        "avg_mfe": avg_mfe,
        "median_mfe": median_mfe,
        "forced_drawdown_events": summary.get("forced_drawdown_events", np.nan),
        "audit_all_pass": True,
    }
    for yr in summary.get("annual", []):
        row[f"return_{int(yr['year'])}"] = yr["strategy_return"]
    obs_path = run_dir / "persistence_observations.csv"
    if obs_path.exists():
        obs = pd.read_csv(obs_path, dtype={"code": str})
        row["recommendation_observations"] = int(len(obs))
        row["accepted_recommendation_observations"] = int(obs.accepted.astype(bool).sum()) if len(obs) else 0
        row["blocked_recommendation_observations"] = int((~obs.accepted.astype(bool)).sum()) if len(obs) else 0
        row["unique_recommended_codes"] = int(obs.code.nunique()) if len(obs) else 0
    else:
        row["recommendation_observations"] = np.nan
        row["accepted_recommendation_observations"] = np.nan
        row["blocked_recommendation_observations"] = np.nan
        row["unique_recommended_codes"] = np.nan
    return row


def main() -> None:
    actual = sha256(LOCKED_ENGINE)
    if actual != LOCKED_ENGINE_SHA256:
        raise RuntimeError(f"locked engine SHA mismatch: {actual}")
    if not CORP.exists():
        raise RuntimeError(f"missing corporate action snapshot: {CORP}")

    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True)

    # 1) Exact immutable baseline reproduction.
    link_inputs(BASELINE_DIR)
    baseline_script = BASELINE_DIR / "r10_max_formal_locked.py"
    baseline_script.write_text(LOCKED_ENGINE.read_text(encoding="utf-8"), encoding="utf-8")
    run_python(baseline_script, BASELINE_DIR, "execution.log")
    verify_baseline()

    locked_source = LOCKED_ENGINE.read_text(encoding="utf-8")

    # 2) Entry-only persistence variants, exact locked stage-3 engine otherwise unchanged.
    for cfg in VARIANTS:
        run_dir = RUN_ROOT / cfg["name"]
        run_dir.mkdir(parents=True)
        script = run_dir / "runner.py"
        script.write_text(build_variant_source(locked_source, cfg), encoding="utf-8")
        run_python(script, run_dir, "execution.log")

    # 3) Common comparison diagnostics.
    causal = pd.read_csv(
        BASELINE_DIR / "ohlcv_causal_2020_2025.csv.gz",
        dtype={"code": str},
        usecols=lambda c: c in {"date", "code", "close", "aclose", "alow", "ahigh"},
        low_memory=False,
    )
    causal["code"] = causal["code"].astype(str).str.zfill(4)
    causal["date"] = causal["date"].astype(int)

    rows = [collect_metrics("BASELINE", BASELINE_DIR, None, causal)]
    for cfg in VARIANTS:
        rows.append(collect_metrics(cfg["name"], RUN_ROOT / cfg["name"], cfg, causal))
    comp = pd.DataFrame(rows)
    base = comp.iloc[0]
    comp["cagr_delta_pp_vs_baseline"] = (comp.cagr - base.cagr) * 100
    comp["max_dd_improvement_pp_vs_baseline"] = (comp.max_drawdown - base.max_drawdown) * 100
    comp["pf_delta_vs_baseline"] = comp.pnl_profit_factor - base.pnl_profit_factor
    comp["win_rate_delta_pp_vs_baseline"] = (comp.win_rate - base.win_rate) * 100
    comp["trade_reduction_pct_vs_baseline"] = 1 - comp.completed_trades / base.completed_trades
    comp["beats_baseline_cagr"] = comp.cagr > base.cagr
    comp["improves_baseline_dd"] = comp.max_drawdown > base.max_drawdown
    comp["beats_baseline_pf"] = comp.pnl_profit_factor > base.pnl_profit_factor
    comp["beats_baseline_win_rate"] = comp.win_rate > base.win_rate
    comp.to_csv(RUN_ROOT / "R10_PERSISTENCE_COMPARISON.csv", index=False)

    records = json.loads(comp.replace({np.nan: None}).to_json(orient="records"))
    (RUN_ROOT / "R10_PERSISTENCE_COMPARISON.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cols = [
        "variant", "end_nav", "cagr", "max_drawdown", "pnl_profit_factor", "win_rate",
        "completed_trades", "avg_win_return", "avg_loss_return", "avg_mae", "avg_mfe",
        "cagr_delta_pp_vs_baseline", "max_dd_improvement_pp_vs_baseline",
        "pf_delta_vs_baseline", "win_rate_delta_pp_vs_baseline", "trade_reduction_pct_vs_baseline",
    ]
    md = [
        "# R10 MAX Recommendation Persistence Gate — 2021–2025",
        "",
        f"Locked engine SHA256: `{LOCKED_ENGINE_SHA256}`",
        "",
        "Persistence is observed only when the candidate has passed the locked R10 selection, sizing, cash, exposure and liquidity checks and would otherwise submit a BUY. A blocked recommendation consumes that day's recommendation slot, so lower-ranked stocks are not backfilled. The counter resets on an actual BUY fill.",
        "",
        comp[cols].to_markdown(index=False),
        "",
        "## Annual returns",
        "",
        comp[["variant", "return_2021", "return_2022", "return_2023", "return_2024", "return_2025"]].to_markdown(index=False),
    ]
    (RUN_ROOT / "R10_PERSISTENCE_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print("\n=== R10 PERSISTENCE COMPARISON ===")
    print(comp[cols].to_string(index=False))
    print(f"\nSaved to {RUN_ROOT}")


if __name__ == "__main__":
    main()
