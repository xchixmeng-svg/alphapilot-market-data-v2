#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

ROOT=Path("v6_2_core_numerical")
ROOT.mkdir(exist_ok=True)
V1=Path("inputs/v1/EVIDENCE_BUNDLE_V1.parquet")
FREG=Path("inputs/0f/0F_FINAL_ASSEMBLY/feature_registry.json")
EXPECTED_V1_SHA="5853b15ea69bfd3dd2eadb3214bd34894d589f5afce53cd5245c4079f8d1ab7e"
EXPECTED_0F_SHA="afd4b86fafed8d3ab576bf5fcc39e6a8ed5fbd0a015909534fe7ac070628c358"
SEED=926616
RESET_ABS_RAW_RETURN=0.20
MAX_TRAIN=400_000
MAX_CAL=180_000

TASKS={
  "u5_d3_h5": (0.05,-0.03,5),
  "u5_d3_h10": (0.05,-0.03,10),
  "u5_d3_h20": (0.05,-0.03,20),
  "u10_d5_h10": (0.10,-0.05,10),
  "u10_d5_h20": (0.10,-0.05,20),
  "u10_d5_h40": (0.10,-0.05,40),
  "u10_d5_h60": (0.10,-0.05,60),
  "u20_d10_h20": (0.20,-0.10,20),
  "u20_d10_h40": (0.20,-0.10,40),
  "u20_d10_h60": (0.20,-0.10,60),
  "u20_d10_h120": (0.20,-0.10,120),
}
CLASS_NAMES=np.array(["UP_FIRST","DOWN_FIRST","NEITHER"],dtype=object)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def date_int(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.strftime("%Y%m%d").astype(np.int64)
    n=pd.to_numeric(s,errors="coerce")
    if n.notna().all() and n.between(19000101,20991231).all():
        return n.astype(np.int64)
    d=pd.to_datetime(s,errors="raise")
    return d.dt.strftime("%Y%m%d").astype(np.int64)

def load():
    if sha(V1)!=EXPECTED_V1_SHA:
        raise RuntimeError(f"V1 hash mismatch {sha(V1)}")
    x=pd.read_parquet(V1)
    x["date"]=date_int(x["date"])
    if int(x["date"].max())>20241231:
        raise RuntimeError("2025 leaked into numerical engine")
    x["code"]=x["code"].astype(str).str.strip().str.replace(r"\\.0$","",regex=True).str.zfill(4)
    freg=json.loads(FREG.read_text())
    feats=list(freg["features"])
    miss=[c for c in feats if c not in x.columns]
    if miss: raise RuntimeError(f"missing frozen features {miss[:20]}")
    return x.sort_values(["code","date"]).reset_index(drop=True), feats

def safe_features(x,base_feats):
    z=x.copy()
    for c in base_feats:
        z[c]=pd.to_numeric(z[c],errors="coerce").astype(np.float32)

    raw_prev=z.groupby("code")["close"].shift(1)
    z["raw_ret1_for_reset"]=z["close"]/raw_prev-1.0
    z["price_reset"]=z["raw_ret1_for_reset"].abs()>RESET_ABS_RAW_RETURN
    z["price_segment_id"]=z.groupby("code")["price_reset"].cumsum().astype(np.int32)
    g=z.groupby(["code","price_segment_id"],group_keys=False)
    prev=g["close"].shift(1)
    amt=(z["close"]*z["volume"]).clip(lower=0)
    prev_amt=amt.groupby([z["code"],z["price_segment_id"]]).shift(1)
    z["ret1"]=z["close"]/prev-1
    z["gap1"]=z["open"]/prev-1
    z["range1"]=(z["high"]-z["low"])/prev.replace(0,np.nan)
    z["body1"]=z["close"]/z["open"].replace(0,np.nan)-1
    z["amount_logchg"]=np.log1p(amt)-np.log1p(prev_amt.clip(lower=0))
    feats=list(base_feats)
    channels=["ret1","gap1","range1","body1","amount_logchg"]
    for lag in (0,1,2,5,10,20,60,119):
        for c in channels:
            n=f"{c}_lag{lag}"
            z[n]=g[c].shift(lag).astype(np.float32)
            feats.append(n)

    z["ret20_obs"]=g["close"].pct_change(20)
    z["ret60_obs"]=g["close"].pct_change(60)
    z["vol20_obs"]=g["ret1"].transform(lambda s:s.rolling(20,min_periods=15).std())
    z["vol60_obs"]=g["ret1"].transform(lambda s:s.rolling(60,min_periods=40).std())
    hi20=g["high"].transform(lambda s:s.rolling(20,min_periods=15).max())
    lo20=g["low"].transform(lambda s:s.rolling(20,min_periods=15).min())
    hi60=g["high"].transform(lambda s:s.rolling(60,min_periods=40).max())
    lo60=g["low"].transform(lambda s:s.rolling(60,min_periods=40).min())
    z["range_width20"]=hi20/lo20.replace(0,np.nan)-1
    z["range_width60"]=hi60/lo60.replace(0,np.nan)-1
    z["close_pos20"]=(z["close"]-lo20)/(hi20-lo20).replace(0,np.nan)
    z["close_pos60"]=(z["close"]-lo60)/(hi60-lo60).replace(0,np.nan)
    z["season_ret20_1y"]=g["close"].shift(232)/g["close"].shift(252)-1
    z["season_ret60_1y"]=g["close"].shift(192)/g["close"].shift(252)-1
    z["season_ret20_2y"]=g["close"].shift(484)/g["close"].shift(504)-1
    extra=["ret20_obs","ret60_obs","vol20_obs","vol60_obs","range_width20","range_width60",
           "close_pos20","close_pos60","season_ret20_1y","season_ret60_1y","season_ret20_2y"]
    feats += extra

    dt=pd.to_datetime(z["date"].astype(str),format="%Y%m%d")
    doy=dt.dt.dayofyear.astype(float); mon=dt.dt.month.astype(float); dow=dt.dt.dayofweek.astype(float)
    for n,v in {
      "cal_doy_sin":np.sin(2*np.pi*doy/365.25),"cal_doy_cos":np.cos(2*np.pi*doy/365.25),
      "cal_month_sin":np.sin(2*np.pi*mon/12),"cal_month_cos":np.cos(2*np.pi*mon/12),
      "cal_dow_sin":np.sin(2*np.pi*dow/5),"cal_dow_cos":np.cos(2*np.pi*dow/5)
    }.items(): z[n]=v.astype(np.float32); feats.append(n)

    ind=[c for c in base_feats if c.startswith("industry_ret1_")]
    if ind:
        d=z[["date"]+ind].drop_duplicates("date").sort_values("date").set_index("date")
        d["sector_adv"]=(d[ind]>0).mean(axis=1)
        d["sector_median1"]=d[ind].median(axis=1)
        d["sector_dispersion1"]=d[ind].std(axis=1)
        sf=["sector_adv","sector_median1","sector_dispersion1"]
        z=z.merge(d[sf].reset_index(),on="date",how="left",validate="many_to_one")
        feats += sf

    # Admitted institutional evidence: use normalized ratios only, never raw shares.
    inst_base=["inst_foreign_net_ratio","inst_trust_net_ratio","inst_dealer_net_ratio"]
    for c in inst_base:
        if c in z.columns:
            z[c]=pd.to_numeric(z[c],errors="coerce").astype(np.float32)
            feats.append(c)
    g=z.groupby(["code","price_segment_id"],group_keys=False)
    for c in inst_base:
        if c not in z.columns: continue
        for w in (5,20):
            n=f"{c}_mean{w}"
            z[n]=g[c].transform(lambda s:s.rolling(w,min_periods=max(2,w//2)).mean()).astype(np.float32)
            feats.append(n)

    z=z.sort_values(["code","date"]).reset_index(drop=True)
    z["segment_hist_count"]=z.groupby(["code","price_segment_id"]).cumcount()+1
    z["universe_ok"]=(z["segment_hist_count"]>=120)&(z["close"]>0)
    feats=list(dict.fromkeys(feats))
    return z,feats

def label_one_segment(g,up,down,horizon):
    n=len(g); close=g["close"].to_numpy(float); high=g["high"].to_numpy(float); low=g["low"].to_numpy(float)
    state=np.full(n,2,dtype=np.int8)  # NEITHER
    t_hit=np.full(n,np.nan,dtype=np.float32)
    mfe=np.full(n,np.nan,dtype=np.float32); mae=np.full(n,np.nan,dtype=np.float32)
    unresolved=np.ones(n,dtype=bool)
    maxfav=np.full(n,-np.inf); minadv=np.full(n,np.inf)
    for step in range(1,horizon+1):
        m=n-step
        if m<=0: break
        origin=np.arange(m)
        fav=high[step:]/close[:m]-1.0
        adv=low[step:]/close[:m]-1.0
        maxfav[:m]=np.maximum(maxfav[:m],fav)
        minadv[:m]=np.minimum(minadv[:m],adv)
        u=unresolved[:m]
        hu=fav>=up; hd=adv<=down
        amb=u & hu & hd
        # ambiguous is coded -2; excluded from model/eval
        idx=np.flatnonzero(amb)
        if len(idx):
            state[idx]=-2; t_hit[idx]=step; unresolved[idx]=False
        upidx=np.flatnonzero(u & hu & ~hd)
        if len(upidx):
            state[upidx]=0; t_hit[upidx]=step; unresolved[upidx]=False
        dnidx=np.flatnonzero(u & hd & ~hu)
        if len(dnidx):
            state[dnidx]=1; t_hit[dnidx]=step; unresolved[dnidx]=False
    full=np.arange(n)+horizon<n
    cens=unresolved & ~full
    state[cens]=-1
    mfe[full]=maxfav[full].astype(np.float32)
    mae[full]=minadv[full].astype(np.float32)
    return state,t_hit,mfe,mae

def add_labels(z,up,down,horizon):
    state=np.full(len(z),-1,dtype=np.int8); th=np.full(len(z),np.nan,np.float32)
    mfe=np.full(len(z),np.nan,np.float32); mae=np.full(len(z),np.nan,np.float32)
    for _,idx in z.groupby(["code","price_segment_id"],sort=False).groups.items():
        ii=np.asarray(idx,dtype=np.int64)
        s,t,mf,ma=label_one_segment(z.loc[ii],up,down,horizon)
        state[ii]=s; th[ii]=t; mfe[ii]=mf; mae[ii]=ma
    z=z.copy(); z["y_state"]=state; z["time_to_first"]=th; z["mfe_h"]=mfe; z["mae_h"]=mae
    return z

def sample_idx(idx,n,seed):
    if len(idx)<=n: return idx
    rng=np.random.default_rng(seed)
    return np.sort(rng.choice(idx,size=n,replace=False))

def iso_calibrate(raw_cal,y_cal,raw_eval):
    out=[]
    for k in range(3):
        ir=IsotonicRegression(y_min=0,y_max=1,out_of_bounds="clip")
        ir.fit(raw_cal[:,k],(y_cal==k).astype(float))
        out.append(ir.predict(raw_eval[:,k]))
    p=np.column_stack(out)
    p=np.clip(p,1e-8,None)
    p=p/p.sum(axis=1,keepdims=True)
    return p

def brier_multiclass(y,p):
    yy=np.eye(3)[y]
    return float(np.mean(np.sum((p-yy)**2,axis=1)))

def ece_binary(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); e=0.0
    for i in range(bins):
        m=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if m.any(): e += m.mean()*abs(float(p[m].mean())-float(y[m].mean()))
    return float(e)

def run(task):
    up,down,horizon=TASKS[task]
    x,base_feats=load()
    z,feats=safe_features(x,base_feats)
    z=add_labels(z,up,down,horizon)

    dates=np.array(sorted(z["date"].unique()),dtype=np.int64)
    cal_start=int(dates[np.searchsorted(dates,20230101)])
    hold_start=int(dates[np.searchsorted(dates,20240101)])
    pcal=int(np.searchsorted(dates,cal_start)); phold=int(np.searchsorted(dates,hold_start))
    if pcal<horizon or phold<horizon: raise RuntimeError("insufficient purge")
    train_end=int(dates[pcal-horizon-1])
    cal_end=int(dates[phold-horizon-1])

    valid=z["universe_ok"] & z["y_state"].isin([0,1,2])
    tr=np.flatnonzero(valid & (z["date"]<=train_end))
    ca=np.flatnonzero(valid & (z["date"]>=cal_start) & (z["date"]<=cal_end))
    ev=np.flatnonzero(valid & (z["date"]>=hold_start) & (z["date"]<=20241231))
    tr=sample_idx(tr,MAX_TRAIN,SEED+horizon)
    ca=sample_idx(ca,MAX_CAL,SEED+1000+horizon)
    if min(len(tr),len(ca),len(ev))<5000: raise RuntimeError("split too small")

    X=z[feats].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=np.float32,copy=False)
    y=z["y_state"].to_numpy(np.int8)
    model=HistGradientBoostingClassifier(
        learning_rate=0.05,max_iter=110,max_leaf_nodes=31,min_samples_leaf=120,
        l2_regularization=4.0,random_state=SEED+horizon
    )
    model.fit(X[tr],y[tr])
    raw_cal=model.predict_proba(X[ca]); raw_ev=model.predict_proba(X[ev])
    # sklearn class order should be 0,1,2; guard it.
    if not np.array_equal(model.classes_,np.array([0,1,2])):
        raise RuntimeError(f"unexpected classes {model.classes_}")
    p=iso_calibrate(raw_cal,y[ca],raw_ev)
    yev=y[ev]
    train_freq=np.bincount(y[tr],minlength=3)/len(tr)
    base=np.tile(train_freq,(len(ev),1))

    p_up=p[:,0]
    q90=float(np.quantile(p_up,.90)); q95=float(np.quantile(p_up,.95))
    up_actual=(yev==0)
    hold_base=float(up_actual.mean())
    top10=p_up>=q90; top5=p_up>=q95

    result={
      "status":"PROTOTYPE_DIAGNOSTIC_ONLY",
      "task":task,"up":up,"down":down,"horizon":horizon,
      "split":{"train_max":train_end,"calibration_min":cal_start,"calibration_max":cal_end,
               "holdout_min":hold_start,"holdout_max":20241231,"purge_sessions":horizon},
      "rows":{"train":int(len(tr)),"calibration":int(len(ca)),"holdout":int(len(ev))},
      "class_rate_holdout":{CLASS_NAMES[k]:float(np.mean(yev==k)) for k in range(3)},
      "metrics":{
        "logloss_model":float(log_loss(yev,p,labels=[0,1,2])),
        "logloss_base":float(log_loss(yev,base,labels=[0,1,2])),
        "brier_model":brier_multiclass(yev,p),
        "brier_base":brier_multiclass(yev,base),
        "up_ece":ece_binary(up_actual.astype(int),p_up),
        "up_base_rate":hold_base,
        "top10_prob_threshold":q90,
        "top10_actual_up_rate":float(up_actual[top10].mean()),
        "top10_lift_vs_holdout_base":float(up_actual[top10].mean()/hold_base) if hold_base else None,
        "top5_prob_threshold":q95,
        "top5_actual_up_rate":float(up_actual[top5].mean()),
        "top5_lift_vs_holdout_base":float(up_actual[top5].mean()/hold_base) if hold_base else None,
      },
      "feature_count":len(feats),
      "reset_events":int(z["price_reset"].sum()),
      "ambiguous_rows_total":int((z["y_state"]==-2).sum()),
      "censored_rows_total":int((z["y_state"]==-1).sum()),
      "v1_sha256":sha(V1),
      "sealed_2025_rows_read":0,
    }

    # Fixed probability-decile diagnostics, not an admission rule.
    eval_df=z.iloc[ev][["date","code","time_to_first","mfe_h","mae_h"]].copy()
    eval_df["actual_state"]=CLASS_NAMES[yev]
    eval_df["p_up"]=p[:,0]; eval_df["p_down"]=p[:,1]; eval_df["p_neither"]=p[:,2]
    try:
        eval_df["p_up_decile"]=pd.qcut(eval_df["p_up"],10,labels=False,duplicates="drop")
    except Exception:
        eval_df["p_up_decile"]=0
    dec=(eval_df.groupby("p_up_decile",dropna=False)
      .agg(n=("code","size"),mean_p_up=("p_up","mean"),
           actual_up_rate=("actual_state",lambda s:float((s=="UP_FIRST").mean())),
           actual_down_rate=("actual_state",lambda s:float((s=="DOWN_FIRST").mean())),
           mean_mfe=("mfe_h","mean"),mean_mae=("mae_h","mean"),
           median_time_to_first=("time_to_first","median"))
      .reset_index())
    dec.to_csv(ROOT/f"{task}_PROBABILITY_DECILES.csv",index=False)
    (ROOT/f"{task}_SUMMARY.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    # Store a compact scored sample for audit, never a TopK admission list.
    eval_df.sort_values(["date","p_up","code"],ascending=[True,False,True]).groupby("date").head(25).to_csv(
        ROOT/f"{task}_AUDIT_SAMPLE.csv",index=False
    )
    print(json.dumps(result,ensure_ascii=False),flush=True)

def aggregate(inp):
    rows=[]
    for p in Path(inp).rglob("*_SUMMARY.json"):
        rows.append(json.loads(p.read_text()))
    if len(rows)!=11: raise RuntimeError(f"expected 11 summaries, got {len(rows)}")
    out={
      "layer":"V6.2 Core Numerical Opportunity Engine - Full 11-Task Matrix",
      "status":"PROTOTYPE_COMPLETE",
      "scientific_lock":False,
      "2025_opened":False,
      "tasks":rows,
      "interpretation_rule":"This is diagnostic only. No probability threshold, TopK rule, or candidate admission gate is chosen from these results.",
      "next_action":"Expand to all 11 fixed barrier tasks only if probability discrimination/calibration is nontrivial; otherwise revise architecture before any LLM integration."
    }
    (ROOT/"V6_2_CORE_NUMERICAL_FULL_MATRIX_SUMMARY.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    pd.DataFrame([{
      "task":r["task"],"horizon":r["horizon"],
      **r["metrics"],**{f"holdout_{k}":v for k,v in r["class_rate_holdout"].items()}
    } for r in rows]).to_csv(ROOT/"V6_2_CORE_NUMERICAL_FULL_MATRIX_METRICS.csv",index=False)
    print(json.dumps(out,ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--task",choices=list(TASKS)); ap.add_argument("--aggregate")
    ns=ap.parse_args()
    if bool(ns.task)==bool(ns.aggregate): raise SystemExit("choose exactly one")
    run(ns.task) if ns.task else aggregate(ns.aggregate)
