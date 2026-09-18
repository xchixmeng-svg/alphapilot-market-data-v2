#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "history" / "v6-layered" / "0F_FINAL_ASSEMBLY"
PANEL_PATH = FROZEN / "decision_ticker_panel.parquet"
FEATURE_REGISTRY_PATH = FROZEN / "feature_registry.json"
FINAL_MANIFEST_PATH = ROOT / "data" / "history" / "v6-layered" / "manifests" / "0F_FINAL_ASSEMBLY.json"
SOURCE_FREEZE_PATH = ROOT / "research" / "V6_SOURCE_FREEZE.json"
LOCK_PATH = ROOT / "research" / "V6_1_PREREG_LOCK.json"
OUT = ROOT / "ai_market_reasoning_v6_1_corp_safe_results"
OUT.mkdir(exist_ok=True)

RESET_ABS_RAW_RETURN = 0.20
MAX_HORIZON = 120
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
AUDIT_HORIZONS = (5, 10, 20, 40, 60, 120)
PURGE = 120
MAX_LONG_TRAIN_ROWS = 480_000
RNG_SEED = 926616
LAGS = (0, 1, 2, 5, 10, 20, 60, 119)
STOCK_CHANNELS = ("ret1", "gap1", "range1", "body1", "amount_logchg")
MODEL_PARAMS = dict(
    learning_rate=0.05,
    max_iter=160,
    max_leaf_nodes=31,
    min_samples_leaf=120,
    l2_regularization=4.0,
)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_MEAN_BLOCK = 20
BONFERRONI_ALPHA = 0.05 / len(AUDIT_HORIZONS)
PANEL_SHA256 = "afd4b86fafed8d3ab576bf5fcc39e6a8ed5fbd0a015909534fe7ac070628c358"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_lock_meta(require_lock: bool = True) -> dict:
    if not LOCK_PATH.exists():
        if require_lock:
            raise RuntimeError("formal V6.1 OOS prohibited: V6_1_PREREG_LOCK.json missing")
        return {}
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if require_lock and lock.get("lock_status") != "LOCKED":
        raise RuntimeError("formal V6.1 OOS prohibited: lock_status is not LOCKED")
    return lock


def load_frozen_panel(require_lock: bool = True) -> tuple[pd.DataFrame, list[str], dict]:
    lock = load_lock_meta(require_lock=require_lock)
    manifest = json.loads(FINAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    freg = json.loads(FEATURE_REGISTRY_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_PASS":
        raise RuntimeError("0F is not FROZEN_PASS")
    if manifest.get("pit_rules", {}).get("2025_model_evaluation_opened") is not False:
        raise RuntimeError("2025 seal is not intact")
    got = sha256(PANEL_PATH)
    if got != PANEL_SHA256:
        raise RuntimeError(f"0F panel hash mismatch: {got} != {PANEL_SHA256}")
    if manifest.get("file_sha256", {}).get("decision_ticker_panel.parquet") != got:
        raise RuntimeError("0F manifest panel hash mismatch")
    base_feats = list(freg.get("features") or [])
    if len(base_feats) != int(freg.get("feature_count", -1)):
        raise RuntimeError("feature registry count mismatch")
    x = pd.read_parquet(PANEL_PATH)
    if int(len(x)) != int(manifest.get("row_count", -1)):
        raise RuntimeError("0F row count mismatch")
    x["code"] = x["code"].astype(str).str.zfill(4)
    x["date"] = pd.to_numeric(x["date"], errors="raise").astype(np.int64)
    x = x.sort_values(["code", "date"]).reset_index(drop=True)
    return x, base_feats, lock


def add_safe_observable_transforms(
    x: pd.DataFrame, base_feats: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    z = x.copy()
    for c in base_feats:
        if c in z.columns:
            z[c] = pd.to_numeric(z[c], errors="coerce").astype(np.float32)

    by_code = z.groupby("code", group_keys=False)
    raw_prev = by_code["close"].shift(1)
    z["raw_ret1_for_reset"] = z["close"] / raw_prev - 1.0
    z["price_reset"] = z["raw_ret1_for_reset"].abs() > RESET_ABS_RAW_RETURN
    z["price_segment_id"] = z.groupby("code")["price_reset"].cumsum().astype(np.int32)

    seg_keys = [z["code"], z["price_segment_id"]]
    g = z.groupby(["code", "price_segment_id"], group_keys=False)
    prev_close = g["close"].shift(1)
    amt = (z["close"] * z["volume"]).clip(lower=0)
    prev_amount = amt.groupby(seg_keys).shift(1)

    z["ret1"] = z["close"] / prev_close - 1.0
    z["gap1"] = z["open"] / prev_close - 1.0
    z["range1"] = (z["high"] - z["low"]) / prev_close.replace(0, np.nan)
    z["body1"] = z["close"] / z["open"].replace(0, np.nan) - 1.0
    z["amount_logchg"] = np.log1p(amt) - np.log1p(prev_amount.clip(lower=0))
    for c in STOCK_CHANNELS:
        z[c] = pd.to_numeric(z[c], errors="coerce").astype(np.float32)

    feats = list(base_feats)
    g = z.groupby(["code", "price_segment_id"], group_keys=False)
    for lag in LAGS:
        for c in STOCK_CHANNELS:
            name = f"{c}_lag{lag}"
            z[name] = g[c].shift(lag).astype(np.float32)
            feats.append(name)

    z["ret20_obs"] = g["close"].pct_change(20)
    z["ret60_obs"] = g["close"].pct_change(60)
    z["vol20_obs"] = g["ret1"].transform(lambda s: s.rolling(20, min_periods=15).std())
    z["vol60_obs"] = g["ret1"].transform(lambda s: s.rolling(60, min_periods=40).std())
    hi20 = g["high"].transform(lambda s: s.rolling(20, min_periods=15).max())
    lo20 = g["low"].transform(lambda s: s.rolling(20, min_periods=15).min())
    hi60 = g["high"].transform(lambda s: s.rolling(60, min_periods=40).max())
    lo60 = g["low"].transform(lambda s: s.rolling(60, min_periods=40).min())
    z["range_width20"] = hi20 / lo20.replace(0, np.nan) - 1
    z["range_width60"] = hi60 / lo60.replace(0, np.nan) - 1
    z["close_pos20"] = (z["close"] - lo20) / (hi20 - lo20).replace(0, np.nan)
    z["close_pos60"] = (z["close"] - lo60) / (hi60 - lo60).replace(0, np.nan)
    z["season_ret20_1y"] = g["close"].shift(232) / g["close"].shift(252) - 1
    z["season_ret60_1y"] = g["close"].shift(192) / g["close"].shift(252) - 1
    z["season_ret20_2y"] = g["close"].shift(484) / g["close"].shift(504) - 1
    structure = [
        "ret20_obs", "ret60_obs", "vol20_obs", "vol60_obs",
        "range_width20", "range_width60", "close_pos20", "close_pos60",
        "season_ret20_1y", "season_ret60_1y", "season_ret20_2y",
    ]
    for c in structure:
        z[c] = pd.to_numeric(z[c], errors="coerce").astype(np.float32)
    feats.extend(structure)

    dt = pd.to_datetime(z["date"].astype(str), format="%Y%m%d", errors="coerce")
    doy, month, dow = dt.dt.dayofyear.astype(float), dt.dt.month.astype(float), dt.dt.dayofweek.astype(float)
    z["cal_doy_sin"] = np.sin(2*np.pi*doy/365.25).astype(np.float32)
    z["cal_doy_cos"] = np.cos(2*np.pi*doy/365.25).astype(np.float32)
    z["cal_month_sin"] = np.sin(2*np.pi*month/12.0).astype(np.float32)
    z["cal_month_cos"] = np.cos(2*np.pi*month/12.0).astype(np.float32)
    z["cal_dow_sin"] = np.sin(2*np.pi*dow/5.0).astype(np.float32)
    z["cal_dow_cos"] = np.cos(2*np.pi*dow/5.0).astype(np.float32)
    cal = ["cal_doy_sin","cal_doy_cos","cal_month_sin","cal_month_cos","cal_dow_sin","cal_dow_cos"]
    feats.extend(cal)

    daily = z[["date","macro_fed_funds","macro_us2y","macro_us10y"]].drop_duplicates("date").sort_values("date")
    macro_delta=[]
    for c in ("macro_fed_funds","macro_us2y","macro_us10y"):
        for h in (5,20,60):
            name=f"{c}_d{h}"
            daily[name]=(daily[c]-daily[c].shift(h)).astype(np.float32)
            macro_delta.append(name)
    z=z.merge(daily[["date"]+macro_delta],on="date",how="left",validate="many_to_one")
    feats.extend(macro_delta)

    ind_cols=[c for c in base_feats if c.startswith("industry_ret1_")]
    if ind_cols:
        d=z[["date"]+ind_cols].drop_duplicates("date").sort_values("date").set_index("date")
        d["sector_adv"]=(d[ind_cols]>0).mean(axis=1).astype(np.float32)
        d["sector_median1"]=d[ind_cols].median(axis=1).astype(np.float32)
        d["sector_dispersion1"]=d[ind_cols].std(axis=1).astype(np.float32)
        ret20=(1.0+d[ind_cols]).rolling(20,min_periods=15).apply(np.prod,raw=True)-1.0
        d["sector_top3_20"]=ret20.apply(lambda r:r.nlargest(min(3,r.notna().sum())).mean() if r.notna().any() else np.nan,axis=1).astype(np.float32)
        d["sector_bottom3_20"]=ret20.apply(lambda r:r.nsmallest(min(3,r.notna().sum())).mean() if r.notna().any() else np.nan,axis=1).astype(np.float32)
        sf=["sector_adv","sector_median1","sector_dispersion1","sector_top3_20","sector_bottom3_20"]
        z=z.merge(d[sf].reset_index(),on="date",how="left",validate="many_to_one")
        feats.extend(sf)

    z=z.sort_values(["code","date"]).reset_index(drop=True)
    z["segment_hist_count"]=z.groupby(["code","price_segment_id"]).cumcount()+1
    z["universe_ok"]=(z["segment_hist_count"]>=MAX_HORIZON)&(z["close"]>0)
    feats=list(dict.fromkeys(feats))
    missing=[c for c in feats if c not in z.columns]
    if missing:
        raise RuntimeError(f"V6.1 feature missing: {missing[:20]}")
    return z,feats


def model_frame(df: pd.DataFrame, feats: list[str], horizon: np.ndarray|float) -> np.ndarray:
    x=df[feats].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=np.float32,copy=False)
    h=np.asarray(horizon,dtype=np.float32)
    if h.ndim==0:
        h=np.full(len(df),float(h),dtype=np.float32)
    return np.concatenate([x,(h/MAX_HORIZON).reshape(-1,1),np.sqrt(h/MAX_HORIZON).reshape(-1,1)],axis=1)


def fwd_return(df: pd.DataFrame,h:int)->pd.Series:
    g=df.groupby(["code","price_segment_id"],group_keys=False)["close"]
    return g.shift(-h)/df["close"]-1.0


def make_long_training(train: pd.DataFrame, feats: list[str], seed:int):
    rng=np.random.default_rng(seed)
    per_h=max(500,MAX_LONG_TRAIN_ROWS//MAX_HORIZON)
    xs,ys=[],[]
    for h in range(1,MAX_HORIZON+1):
        y=fwd_return(train,h)
        valid=train["universe_ok"].to_numpy()&np.isfinite(y.to_numpy(dtype=float))
        idx=np.flatnonzero(valid)
        if len(idx)>per_h:
            idx=rng.choice(idx,size=per_h,replace=False)
        if len(idx)==0:
            continue
        part=train.iloc[idx]
        xs.append(model_frame(part,feats,np.full(len(part),h,dtype=np.float32)))
        ys.append(y.iloc[idx].to_numpy(dtype=np.float32))
    if not xs:
        raise RuntimeError("no usable V6.1 training rows")
    return np.concatenate(xs),np.concatenate(ys)


def fit_models(train,feats,seed):
    X,y=make_long_training(train,feats,seed)
    models={}
    for i,q in enumerate(QUANTILES):
        m=HistGradientBoostingRegressor(loss="quantile",quantile=q,random_state=seed+i,**MODEL_PARAMS)
        m.fit(X,y); models[q]=m
        print(f"[V6.1 FIT] q={q:.2f} rows={len(y):,}",flush=True)
    return models,len(y)


def predict_quantiles(models,df,feats,h):
    X=model_frame(df,feats,float(h))
    p=np.column_stack([models[q].predict(X) for q in QUANTILES]).astype(np.float32)
    p.sort(axis=1)
    return p


def pinball_rows(y,p):
    out=[]
    for j,q in enumerate(QUANTILES):
        e=y-p[:,j]; out.append(np.maximum(q*e,(q-1.0)*e))
    return np.column_stack(out)


def naive_quantiles(train,h):
    y=fwd_return(train,h).to_numpy(dtype=float)
    y=y[np.isfinite(y)]
    if len(y)<1000:
        raise RuntimeError(f"V6.1 naive reference support too small for h={h}: {len(y)}")
    return np.quantile(y,QUANTILES).astype(np.float32)


def year_context(ds,year):
    dates=np.array(sorted(ds["date"].unique()),dtype=np.int64)
    test=ds[(ds["date"]>=year*10000+101)&(ds["date"]<=year*10000+1231)&ds["universe_ok"]].copy()
    if test.empty: raise RuntimeError(f"empty test year {year}")
    first=int(test["date"].min()); pos=int(np.searchsorted(dates,first))
    if pos<PURGE: raise RuntimeError(f"insufficient purge history for {year}")
    cutoff=int(dates[pos-PURGE])
    train=ds[(ds["date"]<cutoff)&ds["universe_ok"]].copy()
    return train,test,cutoff


def run_year(year:int):
    if year not in (2022,2023,2024):
        raise RuntimeError("V6.1 repair-validation authorizes only 2022-2024; 2025 remains sealed")
    panel,base_feats,lock=load_frozen_panel(require_lock=True)
    ds,feats=add_safe_observable_transforms(panel,base_feats)
    train,test,cutoff=year_context(ds,year)
    models,nfit=fit_models(train,feats,RNG_SEED+year)
    summaries=[]; date_losses=[]
    for h in AUDIT_HORIZONS:
        actual=fwd_return(test,h)
        valid=np.isfinite(actual.to_numpy(dtype=float))
        eval_df=test.loc[valid].copy(); y=actual.to_numpy(dtype=float)[valid]
        p=predict_quantiles(models,eval_df,feats,h)
        nq=naive_quantiles(train,h); npred=np.tile(nq.reshape(1,-1),(len(y),1))
        row={"test_year":year,"horizon":h,"n":int(len(y)),"train_cutoff":cutoff,"fit_long_rows":nfit,"feature_count":len(feats)}
        for j,q in enumerate(QUANTILES):
            row[f"pinball_q{int(q*100):02d}"]=float(mean_pinball_loss(y,p[:,j],alpha=q))
            row[f"naive_pinball_q{int(q*100):02d}"]=float(mean_pinball_loss(y,npred[:,j],alpha=q))
            row[f"empirical_cdf_q{int(q*100):02d}"]=float(np.mean(y<=p[:,j]))
            row[f"naive_empirical_cdf_q{int(q*100):02d}"]=float(np.mean(y<=npred[:,j]))
        row["coverage_q10_q90"]=float(np.mean((y>=p[:,0])&(y<=p[:,-1])))
        row["naive_coverage_q10_q90"]=float(np.mean((y>=npred[:,0])&(y<=npred[:,-1])))
        row["median_abs_error"]=float(np.median(np.abs(y-p[:,2])))
        row["naive_median_abs_error"]=float(np.median(np.abs(y-npred[:,2])))
        ml=pinball_rows(y,p).mean(axis=1); nl=pinball_rows(y,npred).mean(axis=1)
        row["mean_pinball_all_q"]=float(ml.mean()); row["naive_mean_pinball_all_q"]=float(nl.mean())
        row["pinball_delta_vs_naive"]=float(nl.mean()-ml.mean())
        summaries.append(row)
        dl=pd.DataFrame({"date":eval_df["date"].to_numpy(dtype=np.int64),"horizon":h,"loss_diff_naive_minus_model":nl-ml})
        date_losses.append(dl.groupby(["date","horizon"],as_index=False)["loss_diff_naive_minus_model"].mean())
        print("[V6.1 YEAR]",json.dumps(row,ensure_ascii=False),flush=True)
    pd.DataFrame(summaries).to_csv(OUT/f"V6_1_STAGE_A_{year}_AUDIT.csv",index=False)
    pd.concat(date_losses,ignore_index=True).to_csv(OUT/f"V6_1_STAGE_A_{year}_DATE_LOSS.csv",index=False)
    meta={"year":year,"lock_id":lock.get("lock_id"),"panel_sha256":sha256(PANEL_PATH),"test_rows":int(len(test)),"train_cutoff":cutoff,"reset_events_total":int(ds["price_reset"].sum()),"2025_opened":False}
    (OUT/f"V6_1_STAGE_A_{year}_META.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")


def stationary_bootstrap_mean(x,seed):
    rng=np.random.default_rng(seed); n=len(x)
    if n<100: raise RuntimeError(f"date-level bootstrap support too small: {n}")
    out=np.empty(BOOTSTRAP_REPLICATES,dtype=float); p_restart=1.0/BOOTSTRAP_MEAN_BLOCK
    for b in range(BOOTSTRAP_REPLICATES):
        idx=np.empty(n,dtype=np.int64); idx[0]=rng.integers(0,n)
        for i in range(1,n):
            idx[i]=rng.integers(0,n) if rng.random()<p_restart else (idx[i-1]+1)%n
        out[b]=float(np.mean(x[idx]))
    return out


def aggregate(input_dir:Path):
    audits=[]; losses=[]
    for y in (2022,2023,2024):
        audits.append(pd.read_csv(input_dir/f"V6_1_STAGE_A_{y}_AUDIT.csv"))
        losses.append(pd.read_csv(input_dir/f"V6_1_STAGE_A_{y}_DATE_LOSS.csv"))
    a=pd.concat(audits,ignore_index=True); dl=pd.concat(losses,ignore_index=True)
    infer=[]
    for h in AUDIT_HORIZONS:
        x=dl.loc[dl["horizon"]==h].sort_values("date")["loss_diff_naive_minus_model"].to_numpy(dtype=float)
        boots=stationary_bootstrap_mean(x,RNG_SEED+50000+h); mean=float(np.mean(x))
        pim=float((1+np.sum(boots<=0))/(BOOTSTRAP_REPLICATES+1)); pdeg=float((1+np.sum(boots>=0))/(BOOTSTRAP_REPLICATES+1))
        infer.append({"horizon":h,"date_level_n":int(len(x)),"mean_pinball_improvement_naive_minus_model":mean,
        "bootstrap_ci95_low":float(np.quantile(boots,.025)),"bootstrap_ci95_high":float(np.quantile(boots,.975)),
        "one_sided_p_improve":pim,"one_sided_p_degrade":pdeg,"bonferroni_alpha":BONFERRONI_ALPHA,
        "significant_improvement":bool(mean>0 and pim<=BONFERRONI_ALPHA),
        "significant_degradation":bool(mean<0 and pdeg<=BONFERRONI_ALPHA)})
    inf=pd.DataFrame(infer)
    weighted=[]
    for h in AUDIT_HORIZONS:
        q=a[a["horizon"]==h].copy(); w=q["n"].to_numpy(dtype=float)
        wm=lambda c:float(np.average(q[c].to_numpy(dtype=float),weights=w))
        weighted.append({"horizon":h,"n":int(w.sum()),"model_pinball":wm("mean_pinball_all_q"),"naive_pinball":wm("naive_mean_pinball_all_q"),
        "model_coverage80":wm("coverage_q10_q90"),"naive_coverage80":wm("naive_coverage_q10_q90"),
        "model_median_abs_error":wm("median_abs_error"),"naive_median_abs_error":wm("naive_median_abs_error")})
    wdf=pd.DataFrame(weighted).merge(inf,on="horizon")
    wdf["mae_improved"]=wdf["model_median_abs_error"]<wdf["naive_median_abs_error"]
    wdf["coverage_not_materially_worse"]=(wdf["model_coverage80"]-.8).abs()<=(wdf["naive_coverage80"]-.8).abs()+.03
    obs={"significant_pinball_improvement_horizons":int(wdf["significant_improvement"].sum()),
    "significant_degradation_horizons":int(wdf["significant_degradation"].sum()),
    "mae_improved_horizons":int(wdf["mae_improved"].sum()),
    "coverage_not_materially_worse_horizons":int(wdf["coverage_not_materially_worse"].sum())}
    passed=obs["significant_pinball_improvement_horizons"]>=4 and obs["significant_degradation_horizons"]==0 and obs["mae_improved_horizons"]>=4 and obs["coverage_not_materially_worse_horizons"]>=5
    gate={"stage":"V6.1_REPAIR_VALIDATION","years":[2022,2023,2024],"passed":bool(passed),"observed":obs,"2025_opened":False,
    "next_action":"AUTHORIZE_SEALED_2025_CONFIRMATION" if passed else "STOP_V6_1_NO_RETUNE"}
    a.to_csv(OUT/"V6_1_REPAIR_VALIDATION_AUDIT.csv",index=False)
    wdf.to_csv(OUT/"V6_1_REPAIR_VALIDATION_HORIZON_GATE.csv",index=False)
    (OUT/"V6_1_REPAIR_VALIDATION_GATE.json").write_text(json.dumps(gate,ensure_ascii=False,indent=2),encoding="utf-8")
    print("[V6.1 FINAL GATE]",json.dumps(gate,ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--year",type=int); ap.add_argument("--aggregate-dir")
    ns=ap.parse_args()
    if bool(ns.year)==bool(ns.aggregate_dir): raise SystemExit("specify exactly one")
    run_year(ns.year) if ns.year else aggregate(Path(ns.aggregate_dir))

if __name__=="__main__": main()
