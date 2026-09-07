#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


root = Path("all_results")
summaries = []
dirs = {}
for p in root.rglob("overlay_summary.json"):
    s = json.loads(p.read_text(encoding="utf-8"))
    summaries.append(s)
    dirs[s["policy"]] = p.parent
if len(summaries) != 11 or "BASE" not in dirs:
    raise RuntimeError(f"expected all 11 predefined policies, got {[s['policy'] for s in summaries]}")

base = pd.read_csv(dirs["BASE"] / "r10max_formal_trades.csv", dtype={"code": str})
base["entry_date"] = base.entry_date.astype(int)
base_key = base.set_index(["code", "entry_date"], drop=False)
impact_rows = []

for s in summaries:
    policy = s["policy"]
    if policy == "BASE":
        continue
    run = dirs[policy]
    if s["group"] == "entry":
        blocked = pd.read_csv(run / "margin_entry_blocks.csv", dtype={"code": str})
        for _, b in blocked.drop_duplicates(["code", "scheduled_date"]).iterrows():
            key = (b.code, int(b.scheduled_date))
            if key in base_key.index:
                t = base_key.loc[key]
                if isinstance(t, pd.DataFrame):
                    t = t.iloc[0]
                impact_rows.append({"policy": policy, "group": "entry", "code": b.code,
                                    "entry_date": int(t.entry_date), "baseline_return": float(t["return"]),
                                    "baseline_pnl": float(t.pnl), "effect": "direct_entry_block"})
    else:
        trades = pd.read_csv(run / "r10max_formal_trades.csv", dtype={"code": str})
        for _, t in trades[trades.reason == "MARGIN_EXIT"].iterrows():
            key = (t.code, int(t.entry_date))
            if key in base_key.index:
                b = base_key.loc[key]
                if isinstance(b, pd.DataFrame):
                    b = b.iloc[0]
                impact_rows.append({"policy": policy, "group": "holding", "code": t.code,
                                    "entry_date": int(t.entry_date), "baseline_return": float(b["return"]),
                                    "baseline_pnl": float(b.pnl), "overlay_return": float(t["return"]),
                                    "overlay_pnl": float(t.pnl), "effect": "early_t1_exit"})

impacts = pd.DataFrame(impact_rows)
out = Path("aggregate")
out.mkdir(exist_ok=True)
metrics = pd.DataFrame(summaries).sort_values(["group", "policy"])
metrics.to_csv(out / "all_threshold_results.csv", index=False)
impacts.to_csv(out / "direct_trade_impacts.csv", index=False)

detail = {}
for s in summaries:
    p = s["policy"]
    x = impacts[impacts.policy == p] if len(impacts) else impacts
    detail[p] = {
        **s,
        "directly_affected_baseline_trades": int(len(x)),
        "intercepted_baseline_loss_le_12": int((x.baseline_return <= -.12).sum()) if len(x) else 0,
        "intercepted_baseline_loss_le_15": int((x.baseline_return <= -.15).sum()) if len(x) else 0,
        "intercepted_baseline_loss_le_20": int((x.baseline_return <= -.20).sum()) if len(x) else 0,
        "intercepted_baseline_loss_le_25": int((x.baseline_return <= -.25).sum()) if len(x) else 0,
        "false_positive_profitable_trades": int((x.baseline_return > 0).sum()) if len(x) else 0,
    }
(out / "complete_margin_overlay_report.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
print(metrics[["policy","group","end_nav","cagr","max_drawdown","profit_factor","win_rate","trades","loss_le_12","loss_le_15","loss_le_20","loss_le_25","max_single_loss"]].to_string(index=False))
