#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-stock dynamic target prototype for 4772 台特化 using the frozen 2026-10-02 snapshot.

Research-only. No production Forward/R10 files are modified.

Method:
- Build the same 8 causal features on 2020-2025.
- Keep historically executable T+1-open cases with 30 future trading days.
- Robust-standardize features using historical median/IQR.
- Find the 1,000 nearest historical analogues to 4772's 2026-10-02 state.
- For target levels +5/+8/+10/+12/+15/+20/+25, simulate:
  * T+1 open entry
  * no intraday exit on T+1
  * stop from day 2 onward, stop priority on same-day collision
  * if +5% has never been touched by day 10 -> exit at day-10 close
  * otherwise target, stop, or day-30 close
- Report hit-before-stop probability, average/median return, PF, MFE/MAE and hit timing.

This is a one-stock prototype / analogue study, not a finalized production rule.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

from empirical_planv3 import (
    load_history, make_matrices, normalize_ohlcv, FEATURES,
    HOLD, GAP_ATR, LIMIT_UP, UNIV_MIN_AMOUNT20, UNIV_TOP
)

TARGETS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
K = 1000
STOP_PCT = 0.05

CAND = {
    "code":"4772","name":"台特化","decision_date":"2026-10-02",
    "ref_close":281.5,"atr20":9.18,"atr_pct":0.0325932504,
    "r5":0.1489795918,"r20":0.0996093750,"amount_ratio":6.7733236921,
    "ma20_gap":0.1275786101,"ma60_gap":0.1295392229,
    "inst5_ratio_pct":5.4949670935,"cost_dist_atr":3.38,
    "production_no_chase":290.68
}

def shift(a, k):
    out = np.full(a.shape, np.nan, dtype=float)
    if k < a.shape[0]:
        out[:-k] = a[k:]
    return out

def build_cases(dates, ids, O,H,L,C,V,A,IF,IT,ID):
    D=lambda x: pd.DataFrame(x)
    cdf,vdf,adf=D(C),D(V),D(A)
    prev=cdf.shift(1).to_numpy()
    tr=np.fmax(H,prev)-np.fmin(L,prev)
    tr=np.where(np.isnan(prev),H-L,tr)
    atr=D(tr).rolling(20,min_periods=15).mean().to_numpy()
    ma20=cdf.rolling(20,min_periods=15).mean().to_numpy()
    ma60=cdf.rolling(60,min_periods=45).mean().to_numpy()
    amt20=adf.rolling(20,min_periods=15).mean().to_numpy()
    feats={
      "r5":(cdf/cdf.shift(5)-1).to_numpy(),
      "r20":(cdf/cdf.shift(20)-1).to_numpy(),
      "atr_pct":atr/C,
      "amount_ratio":A/amt20,
      "ma20_gap":C/ma20-1,
      "ma60_gap":C/ma60-1,
    }
    total=IF+IT+ID
    tot5=D(total).rolling(5,min_periods=3).sum().to_numpy()
    vol5=vdf.rolling(5,min_periods=3).sum().to_numpy()
    feats["inst5_ratio_pct"]=tot5/vol5*100.0
    ft=IF+IT
    typ=(H+L+C)/3.0
    pos=np.clip(ft,0,None)
    den60=D(pos).rolling(60,min_periods=55).sum().to_numpy()
    num60=D(typ*pos).rolling(60,min_periods=55).sum().to_numpy()
    cost60=num60/np.where(den60>0,den60,np.nan)
    feats["cost_dist_atr"]=(C-cost60)/atr

    is_stock=np.array([bool(pd.Series([s]).str.fullmatch(r"[1-9]\d{3}").iloc[0]) for s in ids])
    rank=pd.DataFrame(np.where(is_stock[None,:],amt20,np.nan)).rank(axis=1,ascending=False).to_numpy()
    hist_ok=pd.DataFrame(~np.isnan(C)).rolling(60,min_periods=1).sum().to_numpy()>=55
    U=(is_stock[None,:]&(amt20>=UNIV_MIN_AMOUNT20)&(rank<=UNIV_TOP)&hist_ok&np.isfinite(C)&np.isfinite(atr))

    entry=shift(O,1); eL=shift(L,1); eV=shift(V,1)
    gap_cancel=entry >= (C + GAP_ATR*atr)
    locked=eL >= C*LIMIT_UP

    row_no=np.arange(C.shape[0])[:,None]
    future_ok=row_no <= (C.shape[0]-HOLD-2)
    prev_close=D(C).shift(1).to_numpy()
    jump=np.abs(C/prev_close-1.0)>0.105
    future_bad=np.zeros(C.shape,dtype=bool)
    jf=jump.astype(float)
    for kk in range(1,HOLD+1):
        future_bad |= np.nan_to_num(shift(jf,kk),nan=0.0)>0.5

    valid=U&future_ok&~future_bad&np.isfinite(entry)&(eV>0)&~gap_cancel&~locked
    ui,uj=np.nonzero(valid)
    X=np.column_stack([feats[n][ui,uj] for n in FEATURES]).astype(float)
    good=np.all(np.isfinite(X),axis=1)
    return {
      "ui":ui[good],"uj":uj[good],"X":X[good],"dates":dates,"ids":ids,
      "O":O,"H":H,"L":L,"C":C,"V":V,"entry":entry,"atr":atr
    }

def pf(rets):
    pos=rets[rets>0].sum()
    neg=-rets[rets<0].sum()
    return float(pos/neg) if neg>0 else None

def simulate(case, rows, target):
    H,L,C,entry=case["H"],case["L"],case["C"],case["entry"]
    ui,uj=case["ui"][rows],case["uj"][rows]
    out=[]
    hit_days=[]
    mfe30=[]; mae30=[]
    for i,j in zip(ui,uj):
        e=float(entry[i,j])
        stop=e*(1-STOP_PCT)
        tgt=e*(1+target)
        touched5=False
        ret=None; why=None
        highs=[]; lows=[]
        for k in range(2,31):
            h=float(H[i+k,j]); l=float(L[i+k,j]); cl=float(C[i+k,j])
            if not (math.isfinite(h) and math.isfinite(l) and math.isfinite(cl)):
                continue
            highs.append(h); lows.append(l)
            if l <= stop:
                ret=-STOP_PCT; why="stop"; break
            if h >= e*1.05:
                touched5=True
            if h >= tgt:
                ret=target; why="target"; hit_days.append(k); break
            if k==10 and not touched5:
                ret=cl/e-1; why="day10_no5"; break
            if k==30:
                ret=cl/e-1; why="day30"; break
        if ret is None:
            continue
        out.append((ret,why))
        if highs:
            mfe30.append(max(highs)/e-1)
            mae30.append(min(lows)/e-1)
    rets=np.array([x[0] for x in out],float)
    why=[x[1] for x in out]
    n=len(rets)
    return {
      "target_pct":target,
      "n":n,
      "target_hit_rate":why.count("target")/n if n else None,
      "stop_rate":why.count("stop")/n if n else None,
      "day10_no5_rate":why.count("day10_no5")/n if n else None,
      "day30_rate":why.count("day30")/n if n else None,
      "avg_return":float(rets.mean()) if n else None,
      "median_return":float(np.median(rets)) if n else None,
      "pf":pf(rets) if n else None,
      "positive_rate":float((rets>0).mean()) if n else None,
      "median_hit_day":float(np.median(hit_days)) if hit_days else None,
      "mfe30_median":float(np.median(mfe30)) if mfe30 else None,
      "mfe30_p75":float(np.quantile(mfe30,.75)) if mfe30 else None,
      "mae30_median":float(np.median(mae30)) if mae30 else None,
    }

def main():
    root=Path(".").resolve()
    px,inst,hashes=load_history(root)
    mats=make_matrices(px,inst)
    case=build_cases(*mats)

    x0=np.array([CAND[f] for f in FEATURES],float)
    X=case["X"]

    # Robust scaling; winsorize distances so one extreme variable cannot dominate.
    med=np.nanmedian(X,axis=0)
    q25=np.nanquantile(X,.25,axis=0); q75=np.nanquantile(X,.75,axis=0)
    scale=np.where((q75-q25)>1e-9,q75-q25,np.nanstd(X,axis=0))
    Z=(X-med)/scale; z0=(x0-med)/scale
    delta=np.clip(Z-z0,-6,6)
    dist=np.sqrt(np.nanmean(delta*delta,axis=1))

    order=np.argsort(dist)
    rows=order[:min(K,len(order))]
    nearest_dist=dist[rows]
    ui=case["ui"][rows]; uj=case["uj"][rows]
    yrs=pd.DatetimeIndex(case["dates"][ui]).year
    codes=case["ids"][uj]

    sims=[simulate(case,rows,t) for t in TARGETS]

    # Fail closed. A dynamic target is actionable only if the analogue policy has
    # positive gross expectancy and PF > 1; otherwise the prototype decision is NO_TRADE.
    eligible=[
        r for r in sims
        if r["target_hit_rate"] is not None
        and r["target_hit_rate"]>=.25
        and (r["pf"] or 0)>1
        and r["avg_return"]>0
    ]
    core=max(eligible,key=lambda r:r["avg_return"]) if eligible else None

    # Descriptive only: highest tested target with >=25% target-hit probability.
    feasible=[r for r in sims if r["target_hit_rate"] is not None and r["target_hit_rate"]>=.25]
    feasible_target=max(feasible,key=lambda r:r["target_pct"]) if feasible else None

    stretch=None
    if core is not None:
        for r in sims:
            if r["target_pct"]>core["target_pct"] and r["target_hit_rate"]>=.15 and (r["pf"] or 0)>1:
                stretch=r

    result={
      "schema":"AlphaPilot one-stock dynamic-target prototype v1",
      "stock":CAND,
      "history_hashes":hashes,
      "analogue_method":{
        "features":FEATURES,"k":int(len(rows)),
        "distance":"robust IQR-scaled Euclidean, per-feature delta clipped ±6 IQR",
        "years":{str(y):int((yrs==y).sum()) for y in sorted(set(yrs))},
        "unique_codes":int(len(set(codes.tolist()))),
        "median_distance":float(np.median(nearest_dist)),
        "p90_distance":float(np.quantile(nearest_dist,.9)),
      },
      "stop_pct":STOP_PCT,
      "simulations":sims,
      "prototype_decision":"TRADE" if core is not None else "NO_TRADE",
      "prototype_core_target_pct":core["target_pct"] if core else None,
      "descriptive_highest_target_with_25pct_hit":feasible_target["target_pct"] if feasible_target else None,
      "prototype_stretch_target_pct":stretch["target_pct"] if stretch else None,
      "price_examples_from_ref_close":{
        "reference_close":CAND["ref_close"],
        "no_chase_above":CAND["production_no_chase"],
        "stop_if_entry_at_ref":round(CAND["ref_close"]*(1-STOP_PCT),2),
        "core_if_entry_at_ref":round(CAND["ref_close"]*(1+core["target_pct"]),2) if core else None,
        "descriptive_feasible_if_entry_at_ref":round(CAND["ref_close"]*(1+feasible_target["target_pct"]),2) if feasible_target else None,
        "stretch_if_entry_at_ref":round(CAND["ref_close"]*(1+stretch["target_pct"]),2) if stretch else None,
      },
      "warning":"Prototype analogue study only. Actual prices must be recalculated from the real T+1 entry price; this does not alter production AlphaPilot."
    }
    out=root/"research"/"dynamic_target_4772_out"
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")

    lines=[
      "# 4772 台特化｜動態目標試跑",
      "",
      f"- 10/2 收盤：{CAND['ref_close']}",
      f"- 不追價：開盤 >= {CAND['production_no_chase']}",
      f"- 歷史相似案例：{len(rows):,} 筆",
      f"- 原型停損：買進價 -{STOP_PCT:.0%}",
      "",
      "| 目標 | 先碰目標 | 先停損 | 平均報酬 | 中位報酬 | PF | 命中目標中位天數 |",
      "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sims:
        lines.append(
          f"| +{r['target_pct']:.0%} | {r['target_hit_rate']:.1%} | {r['stop_rate']:.1%} | "
          f"{r['avg_return']:.2%} | {r['median_return']:.2%} | "
          f"{r['pf']:.2f} | {r['median_hit_day'] if r['median_hit_day'] is not None else '—'} |"
        )
    lines += ["", f"**原型結論：{result['prototype_decision']}**"]
    if core is not None:
        lines += [
          f"**核心目標：買進價 +{core['target_pct']:.0%}**",
          f"以 281.5 示意：約 {result['price_examples_from_ref_close']['core_if_entry_at_ref']}",
          f"**延伸目標：** " + (f"買進價 +{stretch['target_pct']:.0%}" if stretch else "無"),
        ]
    else:
        lines += [
          "所有測試目標的歷史類比平均報酬都為負、PF 都小於 1，因此不產生正式目標價。",
          (f"若只問『有至少25%機率碰到的最高測試目標』：+{feasible_target['target_pct']:.0%}，"
           f"以281.5示意約 {result['price_examples_from_ref_close']['descriptive_feasible_if_entry_at_ref']}；"
           "這不是買進建議。") if feasible_target else "沒有任何測試目標達25%命中率。",
        ]
    lines += ["", "> 真正成交後，目標價與停損價都應以 T+1 實際成交價重算。"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
