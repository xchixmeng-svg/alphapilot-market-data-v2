#!/usr/bin/env python3
import json
from pathlib import Path

rows = []
for p in sorted(Path("all_results").glob("**/crash_overlay_summary.json")):
    rows.append(json.loads(p.read_text()))
if not rows:
    raise SystemExit("no summaries")
base = next(r for r in rows if r["policy"] == "BASE")
print("policy,end_nav,cagr,max_dd,pf,win_rate,trades,loss<=12,loss<=20,loss<=25,worst,vetoes,risk_exits")
for r in rows:
    print(f"{r['policy']},{r['end_nav']:.2f},{r['cagr']:.6%},{r['max_drawdown']:.6%},"
          f"{r['profit_factor']:.4f},{r['win_rate']:.4%},{r['trades']},{r['loss_le_12']},"
          f"{r['loss_le_20']},{r['loss_le_25']},{r['max_single_loss']:.6%},"
          f"{r['entry_vetoes']},{r['risk_exit_trades']}")
Path("aggregate").mkdir(exist_ok=True)
Path("aggregate/crash_overlay_matrix.json").write_text(json.dumps({"baseline": base, "policies": rows}, ensure_ascii=False, indent=2))
