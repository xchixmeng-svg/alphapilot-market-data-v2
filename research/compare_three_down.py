"""Paired R10 MAX baseline versus three consecutive lower closes, 2021-2025."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(".")
baseline = json.loads((ROOT / "baseline/r10max_formal_summary.json").read_text())
variant = json.loads((ROOT / "variant/r10max_formal_summary.json").read_text())
for dirname in ("baseline", "variant"):
    audit = json.loads((ROOT / dirname / "contract_audit.json").read_text())
    assert audit["all_pass"], (dirname, audit)
b, v = baseline["strategy"], variant["strategy"]
assert abs(b["end_nav"] - 2403427.0678222505) < 1e-7, b
assert b["completed_trades"] == 150 and b["wins"] == 71 and b["losses"] == 79, b
assert abs(b["max_drawdown"] - (-0.19515438907459803)) < 1e-12, b

read_trade = lambda folder: pd.read_csv(
    ROOT / folder / "r10max_formal_trades.csv",
    dtype={"code": str, "entry_date": str, "exit_date": str})
bt, vt = read_trade("baseline"), read_trade("variant")
three = vt[vt["reason"] == "THREE_DOWN"].copy()
keys = ["code", "strategy", "entry_date"]
base_match = bt[keys + ["exit_date", "return", "pnl"]].copy()
# The portfolio paths diverge after exits; matched entries give a descriptive
# recovery check, never a causal attribution of total portfolio P&L.
paired = three.merge(base_match, how="left", on=keys, suffixes=("_variant", "_baseline"))
matched = paired[paired["return_baseline"].notna()].copy()
delta = {key: v[key] - b[key] for key in
         ("end_nav", "cagr", "max_drawdown", "win_rate", "profit_factor", "completed_trades")}
report = {
    "definition": "After entry, 3 consecutive strictly lower adjusted closes on 3 successive trading sessions; T close SELL, T+1 open execution. Original exits take precedence. No new entry filter or other parameter change.",
    "baseline": b,
    "three_down_variant": v,
    "variant_minus_baseline": delta,
    "three_down_exits": int(len(three)),
    "three_down_winners": int((three["pnl"] > 0).sum()),
    "three_down_losers": int((three["pnl"] < 0).sum()),
    "matched_baseline_same_entry": int(len(matched)),
    "matched_baseline_recovered_more": int((matched["return_baseline"] > matched["return_variant"]).sum()),
    "matched_baseline_performed_worse": int((matched["return_baseline"] < matched["return_variant"]).sum()),
    "annual": {
        str(y): {"baseline_return": x["strategy_return"],
                 "variant_return": next(z["strategy_return"] for z in variant["annual"] if z["year"] == y),
                 "delta_pp": 100 * (next(z["strategy_return"] for z in variant["annual"] if z["year"] == y) - x["strategy_return"])}
        for x in baseline["annual"] for y in [x["year"]]
    },
}
(ROOT / "three_down_comparison.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
paired.to_csv(ROOT / "three_down_paired_exits.csv", index=False)
lines = [
    "# R10 MAX 三連跌出場簡易回測",
    "",
    "正式鎖定版保持原樣；2021-2025 同資料、同進場與交易成本。影子版只有新增三次連續下降的調整後收盤價，第三次收盤發出訊號，T+1 開盤依原賣出模型執行；原有賣出理由優先。",
    "",
    "| 指標 | 正式原版 | 三連跌影子版 | 差值 |",
    "|---|---:|---:|---:|",
]
for label, key, fmt in (
    ("期末 NAV (NT$)", "end_nav", ",.2f"),
    ("CAGR", "cagr", ".2%"),
    ("最大回撤", "max_drawdown", ".2%"),
    ("交易筆數", "completed_trades", ",.0f"),
    ("勝率", "win_rate", ".2%"),
    ("PF", "profit_factor", ".3f"),
):
    lines.append(f"| {label} | {format(b[key], fmt)} | {format(v[key], fmt)} | {format(delta[key], fmt)} |")
lines += [
    "",
    f"三連跌新增出場：{len(three)} 筆；與原版相同進場可配對 {len(matched)} 筆。",
    f"配對中原版後續報酬更好 {report['matched_baseline_recovered_more']} 筆，影子版較好 {report['matched_baseline_performed_worse']} 筆。配對數不代表總組合歸因。",
    "",
    "| 年度 | 原版 | 影子版 | 差值（百分點） |",
    "|---|---:|---:|---:|",
]
for year, values in report["annual"].items():
    lines.append(f"| {year} | {values['baseline_return']:.2%} | {values['variant_return']:.2%} | {values['delta_pp']:+.2f} |")
lines += ["", "這是單條件影子測試，沒有以此重新調參或更動正式 Forward 推報。"]
(ROOT / "three_down_comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print((ROOT / "three_down_comparison.md").read_text())
