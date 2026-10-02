"""Simple second definition: three consecutive 0-3% adjusted-close declines."""
import json
from pathlib import Path
import pandas as pd

root = Path(".")
b = json.loads((root / "baseline/r10max_formal_summary.json").read_text())["strategy"]
v = json.loads((root / "gentle/r10max_formal_summary.json").read_text())["strategy"]
audit = json.loads((root / "gentle/contract_audit.json").read_text())
assert audit["all_pass"], audit
assert abs(b["end_nav"] - 2403427.0678222505) < 1e-7
bt = pd.read_csv(root / "baseline/r10max_formal_trades.csv", dtype={"code":str,"entry_date":str})
vt = pd.read_csv(root / "gentle/r10max_formal_trades.csv", dtype={"code":str,"entry_date":str})
gentle = vt[vt.reason == "THREE_DOWN"].copy()
paired = gentle.merge(bt[["code","strategy","entry_date","return"]], on=["code","strategy","entry_date"], how="left", suffixes=("_gentle","_baseline"))
matched = paired[paired["return_baseline"].notna()]
report = {
 "definition": "Three consecutive same-stock, post-entry, strictly negative adjusted close-to-close changes each no worse than -3%; original exit reasons take precedence; signal at T close, sell at T+1 open.",
 "baseline": b, "gentle": v,
 "delta": {k:v[k]-b[k] for k in ("end_nav","cagr","max_drawdown","completed_trades","win_rate","profit_factor")},
 "gentle_exit_count":int(len(gentle)),
 "same_entry_matched_count":int(len(matched)),
 "baseline_later_higher_return":int((matched.return_baseline > matched.return_gentle).sum()),
 "gentle_higher_return":int((matched.return_baseline < matched.return_gentle).sum()),
}
(root/"three_gentle_comparison.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
paired.to_csv(root/"three_gentle_paired_exits.csv",index=False)
rows=["# R10 MAX 三日緩跌出場簡易回測","",
"緩跌定義：持有後連續三個交易日，各日調整後收盤比前一日下跌、但跌幅不超過 3%。第三日收盤發出訊號，T+1 開盤模擬賣出。其他規則沿用鎖定版。","",
"| 指標 | 原版 | 三日緩跌 | 差值 |","|---|---:|---:|---:|"]
for label,k,fmt in (("期末 NAV (NT$)","end_nav",",.2f"),("CAGR","cagr",".2%"),("最大回撤","max_drawdown",".2%"),("交易筆數","completed_trades",",.0f"),("勝率","win_rate",".2%"),("PF","profit_factor",".3f")):
 rows.append(f"| {label} | {format(b[k],fmt)} | {format(v[k],fmt)} | {format(report['delta'][k],fmt)} |")
rows+=["",f"新增緩跌出場 {len(gentle)} 筆；相同進場可配對 {len(matched)} 筆；原版後續報酬較高 {report['baseline_later_higher_return']} 筆，緩跌版較高 {report['gentle_higher_return']} 筆。","",
"此單一門檻僅為影子測試；未調參、未修改正式 Forward 出場。"]
(root/"three_gentle_comparison.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
print((root/"three_gentle_comparison.md").read_text())
