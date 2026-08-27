#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ORDER=["2008_2012","2013_2017","2018_2022","2023_NOW"]

def find(root:Path):
    out={}
    for p in root.rglob("summary.json"):
        try:q=json.loads(p.read_text(encoding="utf-8"))
        except Exception:continue
        c=q.get("cycle")
        if c in ORDER:out[c]=q
    return out

def pct(x):return f"{float(x):.2%}"
def money(x):return f"NT${float(x):,.0f}"

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--output",default="stress_results/five_year_compound/final");a=ap.parse_args()
    src=find(Path(a.input));missing=[c for c in ORDER if c not in src]
    if missing:raise SystemExit(f"missing cycle summaries: {missing}")
    cumulative_withdrawal=0.0;rows=[]
    for c in ORDER:
        q=src[c];w=float(q.get("withdrawal_50pct_profit",0));cumulative_withdrawal+=w
        if c!="2023_NOW":
            end=float(q["settled_nav_after_forced_T1_liquidation"]);ret=end/float(q["initial_capital"])-1;account=float(q["next_cycle_capital"])
        else:
            end=float(q["current_account_nav"]);ret=end/float(q["initial_capital"])-1;account=end
        rows.append({"cycle":c,"initial_capital":float(q["initial_capital"]),"settled_or_current_nav":end,"cycle_return":ret,"max_dd":float(q["max_dd"]),"withdrawal":w,"cumulative_withdrawal":cumulative_withdrawal,"capital_carried_or_current":account,"trades":int(q["completed_trades"]),"avg_exposure":float(q["avg_exposure"]),"max_exposure":float(q["max_exposure"]),"annual_returns":q.get("annual_returns",{})})
    current_account=rows[-1]["capital_carried_or_current"];total_wealth=current_account+cumulative_withdrawal
    start=1_000_000.0
    # Approx annualized investor-wealth growth from 2008 start through last repository trade date.
    last=date.fromisoformat(src["2023_NOW"].get("through",date.today().isoformat()));yrs=max((last-date(2008,1,2)).days/365.25,1/365.25)
    wealth_cagr=(total_wealth/start)**(1/yrs)-1
    worst=min(rows,key=lambda r:r["max_dd"]);best=max(rows,key=lambda r:r["cycle_return"]);worstret=min(rows,key=lambda r:r["cycle_return"])
    payload={"status":"PASS","method":"FULL_R10_5Y_50PCT_PROFIT_WITHDRAWAL","starting_capital":start,"through":src["2023_NOW"].get("through"),"completed_five_year_cycles":3,"current_incomplete_cycle":"2023_NOW","total_withdrawn_cash":cumulative_withdrawal,"current_account_nav":current_account,"combined_investor_wealth":total_wealth,"combined_wealth_multiple":total_wealth/start,"approx_combined_wealth_cagr":wealth_cagr,"worst_cycle_max_dd":{"cycle":worst["cycle"],"max_dd":worst["max_dd"]},"best_cycle_return":{"cycle":best["cycle"],"return":best["cycle_return"]},"worst_cycle_return":{"cycle":worstret["cycle"],"return":worstret["cycle_return"]},"cycles":rows,"important":"Withdrawals are external realized cash. Current incomplete 2023-NOW cycle has no profit withdrawal because five years have not elapsed."}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# AlphaPilot R10-MAX：2008→現在，每五年提領一半獲利","",f"起始本金：**{money(start)}**",f"截至：**{payload['through']}**","", "| 週期 | 起始本金 | 結算/目前NAV | 週期報酬 | Max DD | 提領50%獲利 | 下一期本金/目前帳戶 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['cycle']} | {money(r['initial_capital'])} | {money(r['settled_or_current_nav'])} | {pct(r['cycle_return'])} | {pct(r['max_dd'])} | {money(r['withdrawal'])} | {money(r['capital_carried_or_current'])} |")
    lines += ["", "## 最終資產", "", f"- 累積已提領現金：**{money(cumulative_withdrawal)}**", f"- 目前策略帳戶：**{money(current_account)}**", f"- 合計投資人財富：**{money(total_wealth)}**", f"- 相對最初100萬：**{total_wealth/start:.2f} 倍**", f"- 約略合併財富 CAGR：**{pct(wealth_cagr)}**", "", "## 風險", "", f"- 最深五年週期 Max DD：**{worst['cycle']} / {pct(worst['max_dd'])}**", f"- 最佳週期：**{best['cycle']} / {pct(best['cycle_return'])}**", f"- 最差週期：**{worstret['cycle']} / {pct(worstret['cycle_return'])}**", "", "註：前三個五年期都在期末依 R10 T+1 賣出規則實際清倉後才計算提領；2023→現在尚未滿五年，因此只標記市值、不提領。"]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__":main()
