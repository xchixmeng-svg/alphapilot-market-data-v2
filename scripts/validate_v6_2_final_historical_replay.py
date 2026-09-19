#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

DECISIONS={"REJECT","WATCH","CANDIDATE","HIGH_CONVICTION"}

def validate(final_csv:Path, outcomes_csv:Path, outdir:Path):
    x=pd.read_csv(final_csv,dtype={"code":str})
    y=pd.read_csv(outcomes_csv,dtype={"code":str})
    req={"date","code","decision"}
    if not req<=set(x): raise RuntimeError(f"final outputs missing {sorted(req-set(x))}")
    if not {"date","code"}<=set(y): raise RuntimeError("outcomes missing date/code")
    x["code"]=x["code"].str.zfill(4); y["code"]=y["code"].str.zfill(4)
    z=x.merge(y,on=["date","code"],how="inner",validate="one_to_one")
    if z.empty: raise RuntimeError("empty replay merge")
    if not set(x["decision"].dropna().unique())<=DECISIONS: raise RuntimeError("invalid decision")

    cand=z[z["decision"].isin(["CANDIDATE","HIGH_CONVICTION"])].copy()
    daily=x.groupby("date")["decision"].apply(lambda s:s.isin(["CANDIDATE","HIGH_CONVICTION"]).sum())

    summary={
      "dates":int(daily.size),
      "candidate_days":int((daily>0).sum()),
      "no_candidate_days":int((daily==0).sum()),
      "total_candidates":int(len(cand)),
      "candidate_count_min":int(daily.min()),
      "candidate_count_median":float(daily.median()),
      "candidate_count_max":int(daily.max()),
      "fixed_positive_quota_signature":bool(len(daily)>=20 and (daily>0).all() and daily.nunique()==1),
    }
    if summary["fixed_positive_quota_signature"]:
        raise RuntimeError("forced quota signature")

    metric_rows=[]
    for pcol, acol in [
      ("p_up5_before_down3_h5","actual_up5_before_down3_h5"),
      ("p_up5_before_down3_h10","actual_up5_before_down3_h10"),
      ("p_up5_before_down3_h20","actual_up5_before_down3_h20"),
      ("p_up10_before_down5_h10","actual_up10_before_down5_h10"),
      ("p_up10_before_down5_h20","actual_up10_before_down5_h20"),
      ("p_up10_before_down5_h40","actual_up10_before_down5_h40"),
      ("p_up10_before_down5_h60","actual_up10_before_down5_h60"),
      ("p_up20_before_down10_h20","actual_up20_before_down10_h20"),
      ("p_up20_before_down10_h40","actual_up20_before_down10_h40"),
      ("p_up20_before_down10_h60","actual_up20_before_down10_h60"),
      ("p_up20_before_down10_h120","actual_up20_before_down10_h120"),
    ]:
        if pcol not in z or acol not in z: continue
        q=z[[pcol,acol,"decision"]].dropna()
        if q.empty: continue
        p=q[pcol].astype(float).clip(0,1).to_numpy()
        a=q[acol].astype(float).to_numpy()
        brier=float(np.mean((p-a)**2))
        bins=pd.cut(q[pcol],bins=np.linspace(0,1,11),include_lowest=True)
        cal=(q.assign(_bin=bins).groupby("_bin",observed=True)
             .agg(n=(acol,"size"),pred=(pcol,"mean"),actual=(acol,"mean")).reset_index())
        ece=float(np.sum(cal["n"]/cal["n"].sum()*np.abs(cal["pred"]-cal["actual"])))
        cq=q[q["decision"].isin(["CANDIDATE","HIGH_CONVICTION"])]
        metric_rows.append({
          "probability":pcol,"n":int(len(q)),"brier":brier,"ece":ece,
          "candidate_n":int(len(cq)),
          "candidate_realized_rate":float(cq[acol].mean()) if len(cq) else None,
          "all_realized_rate":float(q[acol].mean())
        })

    pd.DataFrame(metric_rows).to_csv(outdir/"probability_calibration.csv",index=False)

    outcome_pairs=[
      ("realized_mfe_20","base_low_pct","base_high_pct"),
      ("realized_mae_20","expected_mae_low_pct","expected_mae_high_pct"),
    ]
    range_rows=[]
    for actual,lo,hi in outcome_pairs:
        if {actual,lo,hi}<=set(z):
            q=z[[actual,lo,hi,"decision"]].dropna()
            if len(q):
                inside=((q[actual]>=q[lo])&(q[actual]<=q[hi]))
                range_rows.append({"actual":actual,"n":int(len(q)),"coverage":float(inside.mean())})
    pd.DataFrame(range_rows).to_csv(outdir/"range_accuracy.csv",index=False)

    hc=cand[cand["decision"]=="HIGH_CONVICTION"]
    cc=cand[cand["decision"]=="CANDIDATE"]
    if "actual_up10_before_down5_h20" in cand:
        summary["candidate_up10_before_down5_h20"]=float(cand["actual_up10_before_down5_h20"].mean()) if len(cand) else None
        summary["high_conviction_up10_before_down5_h20"]=float(hc["actual_up10_before_down5_h20"].mean()) if len(hc) else None
        summary["ordinary_candidate_up10_before_down5_h20"]=float(cc["actual_up10_before_down5_h20"].mean()) if len(cc) else None

    (outdir/"historical_validation_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--final",required=True); ap.add_argument("--outcomes",required=True); ap.add_argument("--outdir",default="v6_2_final_validation")
    ns=ap.parse_args(); out=Path(ns.outdir); out.mkdir(parents=True,exist_ok=True)
    validate(Path(ns.final),Path(ns.outcomes),out)
