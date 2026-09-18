#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"v6_corporate_action_contamination_audit"
OUT.mkdir(exist_ok=True)
HORIZONS=(5,10,20,40,60,120)

def load_panel(path: Path)->pd.DataFrame:
    x=pd.read_parquet(path, columns=["date","code","close"])
    x["code"]=x["code"].astype(str).str.zfill(4)
    x["date"]=pd.to_numeric(x["date"], errors="raise").astype(np.int64)
    x=x.sort_values(["code","date"]).reset_index(drop=True)
    x["ret1_raw"]=x.groupby("code")["close"].pct_change()
    x["extreme_gt20"]=x["ret1_raw"].abs()>0.20
    return x

def build_future_extreme_flags(x: pd.DataFrame)->pd.DataFrame:
    parts=[]
    for code,g in x.groupby("code",sort=False):
        g=g.sort_values("date").copy()
        e=g["extreme_gt20"].to_numpy(dtype=np.int8)
        rev=np.cumsum(e[::-1])[::-1]
        for h in HORIZONS:
            # count extreme events in next h sessions, excluding decision day
            arr=np.zeros(len(g),dtype=np.int32)
            for i in range(len(g)):
                j1=i+1; j2=min(i+1+h,len(g))
                if j1<j2:
                    arr[i]=int(e[j1:j2].sum())
            g[f"future_extreme_count_h{h}"]=arr
        parts.append(g)
    return pd.concat(parts,ignore_index=True)

def read_actionability_zip(path: Path)->pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        n=[n for n in z.namelist() if n.endswith("_TOP5_DETAIL.csv")][0]
        with z.open(n) as f:
            d=pd.read_csv(f,dtype={"code":str})
    d["code"]=d["code"].astype(str).str.zfill(4)
    d["date"]=pd.to_numeric(d["date"], errors="raise").astype(np.int64)
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--panel",required=True)
    ap.add_argument("--a2022",required=True)
    ap.add_argument("--a2023",required=True)
    ap.add_argument("--a2024",required=True)
    ns=ap.parse_args()
    x=build_future_extreme_flags(load_panel(Path(ns.panel)))

    daily=x[x["extreme_gt20"]].copy()
    daily.to_csv(OUT/"EXTREME_DAILY_RETURNS_GT20.csv",index=False,encoding="utf-8-sig")

    overall=[]
    for year in (2021,2022,2023,2024):
        y=x[(x["date"]>=year*10000+101)&(x["date"]<=year*10000+1231)].copy()
        rec={"year":year,"rows":int(len(y)),"daily_extreme_rows":int(y["extreme_gt20"].sum())}
        for h in HORIZONS:
            c=y[f"future_extreme_count_h{h}"]>0
            rec[f"label_contaminated_h{h}_rows"]=int(c.sum())
            rec[f"label_contaminated_h{h}_rate"]=float(c.mean()) if len(y) else None
        overall.append(rec)
    overall_df=pd.DataFrame(overall)
    overall_df.to_csv(OUT/"YEAR_HORIZON_LABEL_CONTAMINATION.csv",index=False,encoding="utf-8-sig")

    picks=[]
    for year,p in [(2022,ns.a2022),(2023,ns.a2023),(2024,ns.a2024)]:
        d=read_actionability_zip(Path(p))
        for h in HORIZONS:
            z=d[d["horizon"]==h].copy()
            flags=x[["date","code",f"future_extreme_count_h{h}"]]
            z=z.merge(flags,on=["date","code"],how="left",validate="many_to_one")
            z["corporate_action_suspect"]=z[f"future_extreme_count_h{h}"].fillna(0)>0
            for k in (1,3,5):
                q=z[z["rank_q50"]<=k]
                picks.append({
                    "year":year,"horizon":h,"top_k":k,"picks":int(len(q)),
                    "suspect_picks":int(q["corporate_action_suspect"].sum()),
                    "suspect_pick_rate":float(q["corporate_action_suspect"].mean()) if len(q) else None,
                    "mean_end_return_all":float(q["actual_end_return"].mean()) if len(q) else None,
                    "mean_end_return_clean":float(q.loc[~q["corporate_action_suspect"],"actual_end_return"].mean()) if (~q["corporate_action_suspect"]).any() else None,
                    "median_end_return_all":float(q["actual_end_return"].median()) if len(q) else None,
                    "median_end_return_clean":float(q.loc[~q["corporate_action_suspect"],"actual_end_return"].median()) if (~q["corporate_action_suspect"]).any() else None,
                })
    picks_df=pd.DataFrame(picks)
    picks_df.to_csv(OUT/"ACTIONABILITY_PICK_CONTAMINATION.csv",index=False,encoding="utf-8-sig")

    top=(daily.assign(abs_ret=daily["ret1_raw"].abs())
         .sort_values("abs_ret",ascending=False)
         [["date","code","close","ret1_raw","abs_ret"]].head(200))
    top.to_csv(OUT/"TOP_EXTREME_EVENTS.csv",index=False,encoding="utf-8-sig")

    summary={
        "status":"DIAGNOSTIC_ONLY",
        "rule":"A raw close-to-close move with abs(return)>20% is flagged as corporate-action/reference-price suspect. This does not prove every flagged event is corporate action, but it identifies labels requiring official-action reconciliation before performance claims.",
        "years":overall,
        "actionability_summary":picks,
        "top_extreme_events":top.head(50).to_dict(orient="records"),
    }
    (OUT/"V6_CORPORATE_ACTION_CONTAMINATION_AUDIT.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS_DIAGNOSTIC_CREATED","daily_extreme_rows":int(len(daily)),
                      "years":overall},ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
