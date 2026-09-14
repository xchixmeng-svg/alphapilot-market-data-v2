#!/usr/bin/env python3
"""Test the user's idea only on low-base, base-completed R10 candidates.

Locked R10 MAX remains untouched. Non-low-base candidates keep the original first-signal
entry. Only a structurally low-base candidate must be recommended three times before its
T+1 order can be submitted.

Pre-registered, causal low-base/base-completed definition (T-close only):
- adjusted close is no more than 90% of its trailing 120-session high (still low-base),
- adjusted close is at/above MA60,
- MA20 >= MA60 (short trend has turned up),
- adjusted close is at least 98% of MA120 (has reclaimed the long-term base area),
- MA60 <= 105% of MA120 (not already a mature extended trend).

Two robustness windows are run without selecting after seeing results:
- LB_P3_5: 3 recommendations within 5 sessions
- LB_P3_10: 3 recommendations within 10 sessions
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_r10_persistence_gate as base

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "lowbase_persistence_results"
BASELINE_DIR = RUN_ROOT / "baseline"
VARIANTS = [
    {"name": "LB_P3_5", "min_count": 3, "window": 5, "consecutive": False},
    {"name": "LB_P3_10", "min_count": 3, "window": 10, "consecutive": False},
]


def build_lowbase_variant_source(locked_source: str, cfg: dict) -> str:
    marker = (
        "# ============================================================\n"
        "# 步驟三：組合層回測引擎\n"
        "# ============================================================\n\n"
    )
    if marker not in locked_source:
        raise RuntimeError("stage-3 marker not found")
    stage3 = locked_source.split(marker, 1)[1]

    old_load = "px = pd.read_pickle('r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)"
    new_load = (
        "px = pd.read_pickle('../baseline/r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)\n"
        "px['research_high120'] = px.groupby('code')['aclose'].transform(lambda s: s.rolling(120, min_periods=120).max())"
    )
    if old_load not in stage3:
        raise RuntimeError("signal-load anchor not found")
    stage3 = stage3.replace(old_load, new_load, 1)

    state_anchor = "cash = INITIAL_CAPITAL\n"
    if state_anchor not in stage3:
        raise RuntimeError("cash-state anchor not found")
    injection = f'''cash = INITIAL_CAPITAL
# ---- Research-only low-base persistence overlay ----
PERSISTENCE_MIN_COUNT = {int(cfg['min_count'])}
PERSISTENCE_WINDOW = {int(cfg['window'])}
recommendation_history = {{}}
persistence_rows = []

def lowbase_completed(r):
    vals = [r.get('aclose', np.nan), r.get('ma20', np.nan), r.get('ma60', np.nan),
            r.get('ma120', np.nan), r.get('research_high120', np.nan)]
    if not all(np.isfinite(v) and v > 0 for v in vals):
        return False
    aclose, ma20, ma60, ma120, high120 = map(float, vals)
    return bool(
        aclose <= 0.90 * high120 and
        aclose >= ma60 and
        ma20 >= ma60 and
        aclose >= 0.98 * ma120 and
        ma60 <= 1.05 * ma120
    )

def lowbase_persistence_observe(di, idx, code, strat, r, limit, shares, nav, post_total, post_code):
    key = str(code)
    is_lb = lowbase_completed(r)
    if not is_lb:
        recommendation_history.pop(key, None)
        persistence_rows.append({{
            'signal_date': di, 'date_index': idx, 'code': key, 'strategy': strat,
            'is_lowbase_completed': False, 'count': 0, 'accepted': True,
            'limit_price': limit, 'shares_if_ordered': int(shares), 'pre_nav': nav,
            'post_total_exposure_if_ordered': post_total,
            'post_code_exposure_if_ordered': post_code,
        }})
        return True

    hist = recommendation_history.setdefault(key, [])
    if not hist or hist[-1] != idx:
        hist.append(idx)
    hist[:] = [x for x in hist if idx - x < PERSISTENCE_WINDOW]
    count = len(hist)
    accepted = count >= PERSISTENCE_MIN_COUNT
    persistence_rows.append({{
        'signal_date': di, 'date_index': idx, 'code': key, 'strategy': strat,
        'is_lowbase_completed': True, 'count': count, 'accepted': bool(accepted),
        'limit_price': limit, 'shares_if_ordered': int(shares), 'pre_nav': nav,
        'post_total_exposure_if_ordered': post_total,
        'post_code_exposure_if_ordered': post_code,
        'aclose': float(r.get('aclose', np.nan)), 'ma20': float(r.get('ma20', np.nan)),
        'ma60': float(r.get('ma60', np.nan)), 'ma120': float(r.get('ma120', np.nan)),
        'high120': float(r.get('research_high120', np.nan)),
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
        "                if not lowbase_persistence_observe(di, i, code, strat, r, limit, shares, nav, post_total, post_code):\n"
        "                    # Preserve the original rank/slot opportunity; do not backfill with lower-ranked names.\n"
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
    if fill_anchor not in stage3:
        raise RuntimeError("BUY fill anchor not found")
    stage3 = stage3.replace(fill_anchor, fill_anchor + "        recommendation_history.pop(str(code), None)\n", 1)

    dump_anchor = "Path('contract_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')\n"
    if dump_anchor not in stage3:
        raise RuntimeError("output anchor not found")
    stage3 = stage3.replace(
        dump_anchor,
        dump_anchor + "pd.DataFrame(persistence_rows).to_csv('persistence_observations.csv', index=False)\n",
        1,
    )

    header = (
        "# Generated research runner from immutable R10 MAX stage-3 portfolio engine.\n"
        "import pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n"
    )
    return header + stage3


def main() -> None:
    actual = base.sha256(base.LOCKED_ENGINE)
    if actual != base.LOCKED_ENGINE_SHA256:
        raise RuntimeError(f"locked engine SHA mismatch: {actual}")

    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True)

    # Repoint the shared helpers to this isolated research output tree.
    base.RUN_ROOT = RUN_ROOT
    base.BASELINE_DIR = BASELINE_DIR

    base.link_inputs(BASELINE_DIR)
    baseline_script = BASELINE_DIR / "r10_max_formal_locked.py"
    baseline_script.write_text(base.LOCKED_ENGINE.read_text(encoding="utf-8"), encoding="utf-8")
    base.run_python(baseline_script, BASELINE_DIR, "execution.log")
    base.verify_baseline()

    locked_source = base.LOCKED_ENGINE.read_text(encoding="utf-8")
    for cfg in VARIANTS:
        run_dir = RUN_ROOT / cfg["name"]
        run_dir.mkdir(parents=True)
        runner = run_dir / "runner.py"
        runner.write_text(build_lowbase_variant_source(locked_source, cfg), encoding="utf-8")
        base.run_python(runner, run_dir, "execution.log")

    causal = pd.read_csv(
        BASELINE_DIR / "ohlcv_causal_2020_2025.csv.gz",
        dtype={"code": str},
        usecols=lambda c: c in {"date", "code", "close", "aclose", "alow", "ahigh"},
        low_memory=False,
    )
    causal["code"] = causal["code"].astype(str).str.zfill(4)
    causal["date"] = causal["date"].astype(int)

    rows = [base.collect_metrics("BASELINE", BASELINE_DIR, None, causal)]
    for cfg in VARIANTS:
        row = base.collect_metrics(cfg["name"], RUN_ROOT / cfg["name"], cfg, causal)
        obs = pd.read_csv(RUN_ROOT / cfg["name"] / "persistence_observations.csv", dtype={"code": str})
        lb = obs[obs["is_lowbase_completed"].astype(bool)]
        row["lowbase_recommendation_observations"] = int(len(lb))
        row["lowbase_unique_codes"] = int(lb.code.nunique()) if len(lb) else 0
        row["lowbase_blocked_observations"] = int((~lb.accepted.astype(bool)).sum()) if len(lb) else 0
        row["lowbase_accepted_observations"] = int(lb.accepted.astype(bool).sum()) if len(lb) else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    b = df.iloc[0]
    df["cagr_delta_pp_vs_baseline"] = (df.cagr - b.cagr) * 100
    df["max_dd_improvement_pp_vs_baseline"] = (df.max_drawdown - b.max_drawdown) * 100
    df["pf_delta_vs_baseline"] = df.pnl_profit_factor - b.pnl_profit_factor
    df["win_rate_delta_pp_vs_baseline"] = (df.win_rate - b.win_rate) * 100
    df["trade_reduction_pct_vs_baseline"] = 1 - df.completed_trades / b.completed_trades

    csv_path = RUN_ROOT / "LOWBASE_PERSISTENCE_COMPARISON.csv"
    json_path = RUN_ROOT / "LOWBASE_PERSISTENCE_COMPARISON.json"
    report_path = RUN_ROOT / "LOWBASE_PERSISTENCE_REPORT.md"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(df.replace({np.nan: None}).to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")

    cols = ["variant", "end_nav", "cagr", "max_drawdown", "pnl_profit_factor", "win_rate", "completed_trades",
            "avg_mae", "avg_mfe", "lowbase_recommendation_observations", "lowbase_blocked_observations"]
    view = df.reindex(columns=cols)
    report = [
        "# R10 Low-base Persistence Gate Research",
        "",
        "Locked R10 is unchanged. The 3-recommendation gate is applied only when the candidate is causally classified as low-base + base-completed.",
        "",
        "Pre-registered structural definition: close <= 90% of trailing 120-session high; close >= MA60; MA20 >= MA60; close >= 98% MA120; MA60 <= 105% MA120.",
        "",
        view.to_markdown(index=False),
        "",
        "All variants inherit the locked T+1 execution, fixed-limit fills, adverse slippage, integer shares, common cash, fees/tax, corporate actions, caps, and exits.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("=== LOW-BASE PERSISTENCE COMPARISON ===")
    print(view.to_string(index=False))
    print(f"Saved to {RUN_ROOT}")


if __name__ == "__main__":
    main()
