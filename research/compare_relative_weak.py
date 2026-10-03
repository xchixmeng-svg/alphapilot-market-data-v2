"""Research-only paired R7 relative-weakness exit comparison."""
import json
from pathlib import Path
import pandas as pd
r=Path(".")
base=json.loads((r/"baseline/r10max_formal_summary.json").read_text())
new=json.loads((r/"relative/r10max_formal_summary.json").read_text())
for folder in ("baseline","relative"):
 assert json.loads((r/folder/"contract_audit.json").read_text())["all_pass"]
b,v=base["strategy"],new["strategy"]
assert abs(b["end_nav"]-2403427.0678222505)<1e-7 and b["completed_trades"]==150
bt=pd.read_csv(r/"baseline/r10max_formal_trades.csv")
vt=pd.read_csv(r/"relative/r10max_formal_trades.csv")
new_exit=vt[vt.reason=="R7_RELATIVE_WEAK"]
def tails(t):return {str(int(-100*x))+"pct":int((t["return"]<=x).sum()) for x in (-.10,-.15,-.20)}
report={"definition":"R7 only: at fifth or later held session's close, own adjusted return since entry is negative and below 0050 adjusted return over the same period; T+1 sell. No arbitrary percentage parameter; existing exits take precedence.",
 "baseline":b,"relative_weak":v,"delta":{k:v[k]-b[k] for k in ("end_nav","cagr","max_drawdown","completed_trades","win_rate","profit_factor")},
 "new_exits":int(len(new_exit)),"new_exits_pnl_sum":float(new_exit.pnl.sum()),
 "baseline_tail_counts":tails(bt),"relative_tail_counts":tails(vt),
 "annual":{str(x["year"]):{"baseline":x["strategy_return"],"relative":next(y["strategy_return"] for y in new["annual"] if y["year"]==x["year"])} for x in base["annual"]}}
(r/"relative_weak_comparison.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
lines=["# R10 MAX 相對弱勢出場探索測試","",
"R7 持有第 5 個交易日以後：相對買進時調整後價格已虧損，而且同期落後 0050，則 T 收盤發出賣出訊號、T+1 開盤執行。原有出場規則優先；R0.5 不變。這是依同一歷史資料的探索性測試，沒有獨立樣本外驗證。","",
"| 指標 | 原版 | 相對弱勢版 | 差值 |","|---|---:|---:|---:|"]
for label,k,fmt in (("期末 NAV (NT$)","end_nav",",.2f"),("CAGR","cagr",".2%"),("最大回撤","max_drawdown",".2%"),("交易筆數","completed_trades",",.0f"),("勝率","win_rate",".2%"),("PF","profit_factor",".3f")):
 lines.append(f"| {label} | {format(b[k],fmt)} | {format(v[k],fmt)} | {format(report['delta'][k],fmt)} |")
lines+=["",f"新增相對弱勢出場：{len(new_exit)} 筆。","",
"| 尾部虧損 | 原版 | 相對弱勢版 |","|---|---:|---:|"]
for k in ("10pct","15pct","20pct"):lines.append(f"| ≤−{k[:-3]}% | {report['baseline_tail_counts'][k]} | {report['relative_tail_counts'][k]} |")
lines+=["","| 年度 | 原版 | 相對弱勢版 |","|---|---:|---:|"]
for year,x in report["annual"].items():lines.append(f"| {year} | {x['baseline']:.2%} | {x['relative']:.2%} |")
lines+=["","此測試只判斷方向是否值得進一步驗證，不據此更改正式版或每日指令。"]
(r/"relative_weak_comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print((r/"relative_weak_comparison.md").read_text())
