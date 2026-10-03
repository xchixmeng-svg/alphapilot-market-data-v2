"""Descriptive, causal T-close paths for locked R10 completed trades."""
import json
from pathlib import Path
import pandas as pd

root=Path("baseline")
px=pd.read_pickle(root/"r10max_signals_final.pkl")
trades=pd.read_csv(root/"r10max_formal_trades.csv",dtype={"code":str,"entry_date":int,"exit_date":int})
codes=set(trades.code)
sub=px[px.code.isin(codes | {"0050"})].copy()
market=sub[sub.code=="0050"].drop_duplicates("date").set_index("date")["aclose"]
groups={code:frame.sort_values("date").set_index("date") for code,frame in sub[sub.code!="0050"].groupby("code")}
rows=[]
for t in trades.itertuples(index=False):
    if t.code not in groups:continue
    path=groups[t.code].loc[t.entry_date:t.exit_date-1]
    if len(path)==0:continue
    for day in (3,5,10):
        if len(path)<day:continue
        window=path.iloc[:day]
        row=window.iloc[-1]
        start=window.iloc[0]
        if t.entry_date not in market.index or int(row.name) not in market.index:continue
        own=float(row.aclose/start.aclose-1)
        mkt=float(market.loc[int(row.name)]/market.loc[t.entry_date]-1)
        returns=window.aclose.pct_change()
        recent=returns.tail(3)
        down3=(len(recent)==3 and recent.notna().all() and (recent<0).all())
        rows.append({"code":t.code,"strategy":t.strategy,"entry_date":t.entry_date,"exit_date":t.exit_date,
            "final_return":t._asdict()["return"],"final_pnl":t.pnl,"original_reason":t.reason,
            "day":day,"asof_date":int(row.name),"own_return":own,"relative_0050":own-mkt,
            "score_change":float(row.r7_score-start.r7_score) if t.strategy=="R7" and pd.notna(row.r7_score) and pd.notna(start.r7_score) else None,
            "three_lower_closes":bool(down3),"below_ma20":bool(row.aclose<row.ma20) if pd.notna(row.ma20) else None,
            "hard_candidate":bool(row.r7_hard) if t.strategy=="R7" else bool(row.r05_hard)})
out=pd.DataFrame(rows)
out.to_csv("exit_path_diagnostics.csv",index=False)
def cohort(v):
    return "tail_loss_15pct" if v<=-.15 else "winner_20pct" if v>=.20 else "other"
out["cohort"]=out.final_return.map(cohort)
summary=[]
for (day,cohort_name),d in out.groupby(["day","cohort"]):
    summary.append({"day":int(day),"cohort":cohort_name,"n":len(d),
        "median_own_return":float(d.own_return.median()),
        "median_relative_0050":float(d.relative_0050.median()),
        "median_score_change_R7":float(d.score_change.median()) if d.score_change.notna().any() else None,
        "three_lower_closes_pct":float(d.three_lower_closes.mean()),
        "below_ma20_pct":float(d.below_ma20.mean())})
Path("exit_path_diagnostics.json").write_text(json.dumps({"definition":"Descriptive trajectories at entry-relative trading days 3/5/10, conditional on remaining open to that day. Labels use eventual realized return and are retrospective, not live signals.","summary":summary},indent=2,ensure_ascii=False),encoding="utf-8")
lines=["# R10 MAX 出場路徑診斷","",
"正式鎖定版 150 筆交易；第 3/5/10 個持有日收盤，僅統計當天仍在持有的交易。尾部虧損＝最後虧超過15%；大贏家＝最後獲利超過20%。這是描述，不是已驗證的賣出訊號。","",
"| 持有日 | 類別 | 筆數 | 當時報酬中位數 | 相對0050中位數 | R7分數變化中位數 | 三連跌比率 | 跌破MA20比率 |",
"|---:|---|---:|---:|---:|---:|---:|---:|"]
for s in sorted(summary,key=lambda x:(x["day"],x["cohort"])):
 score="—" if s["median_score_change_R7"] is None else f"{s['median_score_change_R7']:+.3f}"
 lines.append(f"| {s['day']} | {s['cohort']} | {s['n']} | {s['median_own_return']:+.2%} | {s['median_relative_0050']:+.2%} | {score} | {s['three_lower_closes_pct']:.1%} | {s['below_ma20_pct']:.1%} |")
lines+=["","注意：只比較當天仍持有的交易會產生存活篩選；當時特徵與最後盈虧的關聯，不等於加入出場規則後組合會改善。"]
Path("exit_path_diagnostics.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(Path("exit_path_diagnostics.md").read_text())
