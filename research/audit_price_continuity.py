#!/usr/bin/env python3
import glob, os, json, re
import numpy as np
import pandas as pd

ROOT=os.environ.get("AUDIT_ROOT",".")
CORP=os.environ.get("CORP_FILE","data/reference/official_corporate_actions_2020_2025.csv")
OUT=os.environ.get("AUDIT_OUT","research/price_continuity_audit_out")
os.makedirs(OUT,exist_ok=True)

parts=[]
for p in sorted(glob.glob(os.path.join(ROOT,"data/history/2020-2025/ohlcv_*.parquet"))):
    d=pd.read_parquet(p)
    ren={}
    for c in d.columns:
        lc=str(c).strip().lower()
        if lc in ("stock_id","code","symbol","ticker"): ren[c]="code"
        elif lc in ("date","trade_date"): ren[c]="date"
        elif lc in ("open","open_price"): ren[c]="open"
        elif lc in ("high","high_price"): ren[c]="high"
        elif lc in ("low","low_price"): ren[c]="low"
        elif lc in ("close","close_price"): ren[c]="close"
        elif lc in ("volume","vol","trading_volume"): ren[c]="volume"
    d=d.rename(columns=ren)
    need=["date","code","open","high","low","close"]
    if not set(need)<=set(d.columns):
        raise RuntimeError(f"missing columns in {p}: {d.columns.tolist()}")
    d=d[need+([ "volume"] if "volume" in d.columns else [])].copy()
    d["date"]=pd.to_datetime(d["date"],errors="coerce")
    d["code"]=d["code"].astype(str).str.replace(r"\.0$","",regex=True).str.replace(r"\.(TW|TWO)$","",regex=True)
    for c in ("open","high","low","close"):
        d[c]=pd.to_numeric(d[c],errors="coerce")
    parts.append(d)

px=pd.concat(parts,ignore_index=True).dropna(subset=["date","code","close"]).sort_values(["code","date"])
px=px[px["code"].str.fullmatch(r"\d{4}")]
g=px.groupby("code",sort=False)
px["prev_date"]=g["date"].shift(1)
px["prev_close"]=g["close"].shift(1)
px["gap_days"]=(px["date"]-px["prev_date"]).dt.days
px["cc_ret"]=px["close"]/px["prev_close"]-1
px["open_ret"]=px["open"]/px["prev_close"]-1
px["ratio"]=px["close"]/px["prev_close"]

# Raw close-to-close discontinuity beyond ordinary ±10% price-limit tolerance.
sus=px[(px["prev_close"]>0)&np.isfinite(px["cc_ret"])&(px["cc_ret"].abs()>0.105)].copy()

ca=pd.read_csv(CORP,dtype={"code":str})
ca["date"]=pd.to_datetime(ca["date"].astype(str),errors="coerce")
ca["code"]=ca["code"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(4)
ca_key=set(zip(ca["date"].dt.normalize(),ca["code"]))
sus["corp_action_exact"]=[(pd.Timestamp(dt).normalize(),c) in ca_key for dt,c in zip(sus["date"],sus["code"])]

# Action may fall on first re-listing day after suspension; exact-date matching should catch correct records.
# Also mark nearby +/- 3 calendar days for diagnostics only.
bycode={c:np.array(sorted(x["date"].dropna().values),dtype="datetime64[ns]") for c,x in ca.groupby("code")}
def nearby(row):
    arr=bycode.get(row["code"])
    if arr is None or len(arr)==0:return False
    dt=np.datetime64(row["date"])
    delta=np.abs((arr-dt).astype("timedelta64[D]").astype(int))
    return bool((delta<=3).any())
sus["corp_action_nearby3d"]=sus.apply(nearby,axis=1)

# Heuristic split / par-value factors for triage only.
factors=np.array([0.05,0.1,0.2,0.25,0.5,2,4,5,10,20],float)
def nearest_factor(r):
    if not np.isfinite(r) or r<=0:return (np.nan,np.nan)
    i=int(np.argmin(np.abs(np.log(r)-np.log(factors))))
    f=factors[i]
    return f,abs(np.log(r/f))
vals=[nearest_factor(x) for x in sus["ratio"]]
sus["nearest_factor"]=[v[0] for v in vals]
sus["factor_log_error"]=[v[1] for v in vals]
sus["split_like"]=sus["factor_log_error"]<0.12

# Explicit flag for events missing from formal corporate-actions file.
sus["missing_formal_action"]=~sus["corp_action_exact"]

cols=["date","code","prev_date","prev_close","open","high","low","close","cc_ret","open_ret","gap_days",
      "corp_action_exact","corp_action_nearby3d","missing_formal_action","nearest_factor","factor_log_error","split_like"]
sus[cols].sort_values(["missing_formal_action","cc_ret"],ascending=[False,True]).to_csv(os.path.join(OUT,"price_discontinuities_gt_10p5.csv"),index=False,encoding="utf-8-sig")

missing=sus[sus["missing_formal_action"]].copy()
missing[cols].sort_values("cc_ret").to_csv(os.path.join(OUT,"missing_corporate_action_candidates.csv"),index=False,encoding="utf-8-sig")

summary={
 "rows_ohlcv":int(len(px)),
 "stocks":int(px["code"].nunique()),
 "date_min":str(px["date"].min().date()),
 "date_max":str(px["date"].max().date()),
 "discontinuities_gt_10p5":int(len(sus)),
 "exactly_covered_by_formal_corp_actions":int(sus["corp_action_exact"].sum()),
 "missing_exact_formal_action":int((~sus["corp_action_exact"]).sum()),
 "missing_and_split_like":int((sus["missing_formal_action"]&sus["split_like"]).sum()),
 "6919_rows":sus[sus["code"].eq("6919")][cols].astype(str).to_dict("records"),
}
json.dump(summary,open(os.path.join(OUT,"audit_summary.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)

print(json.dumps(summary,ensure_ascii=False,indent=2))
print("\nTop missing negative discontinuities:")
print(missing.nsmallest(30,"cc_ret")[cols].to_string(index=False))
print("\nTop missing positive discontinuities:")
print(missing.nlargest(30,"cc_ret")[cols].to_string(index=False))
