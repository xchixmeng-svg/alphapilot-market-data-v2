"""Research-only check: allow original hard stops during the first two sessions."""
import json
from pathlib import Path
import pandas as pd

root=Path(".")
base=json.loads((root/"baseline/r10max_formal_summary.json").read_text())
variant=json.loads((root/"early/r10max_formal_summary.json").read_text())
for p in ("baseline","early"):
 assert json.loads((root/p/"contract_audit.json").read_text())["all_pass"]
b,v=base["strategy"],variant["strategy"]
assert abs(b["end_nav"]-2403427.0678222505)<1e-7 and b["completed_trades"]==150
bt=pd.read_csv(root/"baseline/r10max_formal_trades.csv")
vt=pd.read_csv(root/"early/r10max_formal_trades.csv")
early=vt[(vt.hold_days<=2)&(vt.reason.isin(["R7_HARD","R05_HARD"]))]
def tails(t):
 return {str(int(-100*x))+"pct":int((t["return"]<=x).sum()) for x in (-.10,-.15,-.20)}
report={
 "definition":"Existing R7 -12% and R0.5 -10% adjusted entry-return hard stops can signal at entry day 1/2 close, execute at next open; all other exits retain 3-session minimum.",
 "baseline":b,"early_hard":v,
 "delta":{k:v[k]-b[k] for k in ("end_nav","cagr","max_drawdown","completed_trades","win_rate","profit_factor")},
 "early_hard_exits":int(len(early)),
 "early_hard_exit_rows":early[["code","strategy","entry_date","exit_date","hold_days","return","pnl"]].to_dict("records"),
 "baseline_tail_counts":tails(bt),"early_tail_counts":tails(vt),
 "annual":{str(x["year"]):{"baseline_return":x["strategy_return"],"early_return":next(y["strategy_return"] for y in variant["annual"] if y["year"]==x["year"])} for x in base["annual"]}
}
(root/"early_hard_comparison.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
rows=["# R10 MAX 前兩日急跌硬停損例外測試","",
"原版 R7 −12%、R0.5 −10% 的硬停損門檻不變；只測持有第 1/2 日達門檻時，當天收盤決定、隔日開盤出場。其餘規則保持原版。","",
"| 指標 | 原版 | 提前硬停損 | 差值 |","|---|---:|---:|---:|"]
for label,k,fmt in (("期末 NAV (NT$)","end_nav",",.2f"),("CAGR","cagr",".2%"),("最大回撤","max_drawdown",".2%"),("交易筆數","completed_trades",",.0f"),("勝率","win_rate",".2%"),("PF","profit_factor",".3f")):
 rows.append(f"| {label} | {format(b[k],fmt)} | {format(v[k],fmt)} | {format(report['delta'][k],fmt)} |")
rows.extend(["",f"持有前兩日提早硬停損：{len(early)} 筆。","",
"| 尾部虧損 | 原版 | 提前硬停損 |","|---|---:|---:|"])
for k in ("10pct","15pct","20pct"):rows.append(f"| ≤−{k[:-3]}% | {report['baseline_tail_counts'][k]} | {report['early_tail_counts'][k]} |")
rows+=["","僅是單條件影子測試，2021–2025 歷史結果不能保證 2026 Forward；正式版未變。"]
(root/"early_hard_comparison.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
print((root/"early_hard_comparison.md").read_text())
