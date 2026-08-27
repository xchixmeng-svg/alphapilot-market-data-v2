#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended AlphaPilot R10-MAX historical stress battery.

No partial historical result is accepted as a completed R10 test. For eras
before TWSE T86 daily stock-level coverage, the engine reconstructs the same
Foreign3D / Foreign10D / Trust5D inputs from FinMind's 2005+ per-stock
institutional history. If the required institutional universe cannot be
retrieved with sufficient coverage, the scenario FAILS instead of silently
falling back to R7-only.

Important execution separation:
- Regression validation and five-year capital-path runs must reproduce the
  locked R10 execution profile, not the extra stress-only portfolio overlays.
- Event stress scenarios may keep the conservative DD-force-reduction and ADV
  shock assumptions as diagnostics, but those assumptions must never contaminate
  the locked benchmark gate or capital-path compounding.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import backtest_r10_stress as bt

# ----------------------- locked baseline execution profile ---------------------
# The stress engine contains additional diagnostic overlays (DD sizing throttle,
# forced de-risk/cooldown, and a 2% ADV shock cap). Those are useful in stress
# diagnostics but were not allowed to redefine the locked 2021-2025 benchmark.
# Keep the stress implementation intact and switch these knobs off only when a
# true benchmark-equivalent / five-year capital-path run is requested.
_BASE_VERSION = bt.VERSION
_BASE_ADV_CAP = bt.ADV_CAP
_BASE_FORCE_DD = bt.FORCE_DD
_BASE_DD_MULTIPLIER = bt.dd_multiplier
_BASELINE_PROFILE_ACTIVE = False

def apply_locked_baseline_execution_profile():
    global _BASELINE_PROFILE_ACTIVE
    if _BASELINE_PROFILE_ACTIVE:
        return
    # Effectively remove the stress-only liquidity haircut while retaining all
    # normal cash, position, single-name and total-exposure constraints.
    bt.ADV_CAP = 1.0
    # Disable stress-only forced cross-strategy liquidation/cooldown.
    bt.FORCE_DD = -999.0
    # Disable stress-only drawdown-dependent entry-size haircut.
    bt.dd_multiplier = lambda dd: 1.0
    bt.VERSION = _BASE_VERSION + "-LOCKED-BASELINE"
    _BASELINE_PROFILE_ACTIVE = True
    bt.log("[PROFILE] LOCKED_BASELINE: DD throttle/force-reduce and 2% ADV shock cap disabled")

def restore_stress_execution_profile():
    global _BASELINE_PROFILE_ACTIVE
    bt.ADV_CAP = _BASE_ADV_CAP
    bt.FORCE_DD = _BASE_FORCE_DD
    bt.dd_multiplier = _BASE_DD_MULTIPLIER
    bt.VERSION = _BASE_VERSION
    _BASELINE_PROFILE_ACTIVE = False

# IMPORTANT: five-year capital-path runs use the documented causal controls.
# Do not activate the legacy baseline profile at import time: it disables the
# locked 2% ADV cap, DD sizing throttle, and force-DD defense.
# apply_locked_baseline_execution_profile() is retained only for explicit legacy
# forensic experiments and must never be selected implicitly.

# --------------------------- historical OHLC hygiene ---------------------------
_ORIG_LOAD=bt.load_scenario_ohlcv
def _clean_load(cfg):
    q=_ORIG_LOAD(cfg)
    for c in ("open","high","low","close"):
        q=q[q[c].notna() & (q[c]>0)]
    return q.sort_values(["code","date"]).reset_index(drop=True)
bt.load_scenario_ohlcv=_clean_load

# --------------------------- causal 0050 total return --------------------------
DIV0050={
 20071024:2.5,20081024:2.0,20091023:1.0,20101025:2.2,20111026:1.95,
 20121024:1.85,20131024:1.35,20141024:1.55,20151026:2.0,20160728:.85,
 20170208:1.7,20170731:.7,20180129:2.2,20180723:.7,20190122:2.3,20190719:.7,
 20200131:2.9,20200721:.7,20210122:3.05,20210721:.35,20220121:3.2,
 20220718:1.8,20230130:2.6,20230718:1.9,20240117:3.0,20240716:1.0,
 20250117:2.7,20250721:.36,
}
_ORIG_BUILD=bt.build_features
def _build_features(raw):
    feat,events,_=_ORIG_BUILD(raw)
    etf=feat[feat.code.astype(str).eq("0050")][["date","close","aclose"]].copy().sort_values("date")
    etf["close"]=bt.pd.to_numeric(etf.close,errors="coerce");etf["aclose"]=bt.pd.to_numeric(etf.aclose,errors="coerce")
    etf=etf.dropna(subset=["date","close","aclose"]);etf=etf[(etf.close>0)&(etf.aclose>0)]
    tr=100.0;prev=None;rows=[]
    for r in etf.itertuples(index=False):
        di=int(r.date);adj=float(r.aclose);rawc=float(r.close);factor=adj/rawc if rawc>0 else 1.0
        if prev is not None:
            div=float(DIV0050.get(di,0.0))*factor
            tr*= (adj+div)/prev
        prev=adj;rows.append((di,float(tr)))
    bm=bt.pd.DataFrame(rows,columns=["date","mkt"])
    cal=bt.pd.DataFrame({"date":sorted(int(x) for x in raw.date.unique())})
    bm=cal.merge(bm,on="date",how="left").sort_values("date")
    bm["mkt"]=bt.pd.to_numeric(bm.mkt,errors="coerce").ffill();bm=bm.dropna(subset=["mkt"]).reset_index(drop=True)
    return feat,events,bm
bt.build_features=_build_features
if not _BASELINE_PROFILE_ACTIVE:
    bt.VERSION="AlphaPilot-R10-MAX-0p5-Stress-v1.2-FULL"
else:
    bt.VERSION="AlphaPilot-R10-MAX-0p5-Stress-v1.2-FULL-LOCKED-BASELINE"

# ----------------------- FinMind full-universe fallback ------------------------
FINMIND_URL="https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN=os.getenv("FINMIND_TOKEN","").strip()
FINMIND_CACHE=bt.CACHE_ROOT/"finmind_inst"
FINMIND_CACHE.mkdir(parents=True,exist_ok=True)
_ORIG_FETCH_INST=bt.fetch_institutional
_CURRENT_SCENARIO=""
_LAST_FINMIND_AUDIT={}
_QUOTA_START=time.monotonic();_QUOTA_USED=0

def _quota_gate():
    global _QUOTA_START,_QUOTA_USED
    limit=560 if FINMIND_TOKEN else 280
    elapsed=time.monotonic()-_QUOTA_START
    if elapsed>=3600:
        _QUOTA_START=time.monotonic();_QUOTA_USED=0;return
    if _QUOTA_USED>=limit:
        wait=max(5,3610-elapsed)
        bt.log(f"[FINMIND] hourly quota guard sleep {wait/60:.1f} min")
        time.sleep(wait);_QUOTA_START=time.monotonic();_QUOTA_USED=0

def _finmind_wide(code:str,start_s:str,end_s:str):
    global _QUOTA_USED,_QUOTA_START
    key=f"{start_s}_{end_s}".replace("-","")
    d=FINMIND_CACHE/key;d.mkdir(parents=True,exist_ok=True);p=d/f"{code}.csv"
    if p.exists():
        try:return bt.pd.read_csv(p,dtype={"stock_id":str})
        except Exception:p.unlink(missing_ok=True)
    params={"dataset":"TaiwanStockInstitutionalInvestorsBuySellWide","data_id":code,"start_date":start_s,"end_date":end_s}
    headers={"Accept":"application/json"}
    if FINMIND_TOKEN:headers["Authorization"]=f"Bearer {FINMIND_TOKEN}"
    for attempt in range(4):
        _quota_gate()
        r=bt.SESSION.get(FINMIND_URL,params=params,headers=headers,timeout=(20,120));_QUOTA_USED+=1
        if r.status_code==200:
            j=r.json();data=j.get("data",[]) if isinstance(j,dict) else []
            q=bt.pd.DataFrame(data)
            if not q.empty:q.to_csv(p,index=False,encoding="utf-8-sig")
            else:bt.pd.DataFrame(columns=["date","stock_id"]).to_csv(p,index=False,encoding="utf-8-sig")
            return q
        if r.status_code==402:
            bt.log("[FINMIND] quota response 402; wait for next window")
            time.sleep(3610);_QUOTA_START=time.monotonic();_QUOTA_USED=0;continue
        if r.status_code==403:
            bt.log("[FINMIND] 403/IP throttle; wait 31 minutes")
            time.sleep(1860);_QUOTA_START=time.monotonic();_QUOTA_USED=0;continue
        if r.status_code>=500:
            time.sleep(min(60,2**attempt*5));continue
        raise RuntimeError(f"FinMind {code} HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError(f"FinMind {code} failed after retries")

def _numcol(q,name):
    if name not in q.columns:return bt.pd.Series(0,index=q.index,dtype="float64")
    return bt.pd.to_numeric(q[name],errors="coerce").fillna(0)

def _finmind_full_institutional(feat,eval_start:int,eval_end:int):
    global _LAST_FINMIND_AUDIT
    start_dt=bt.dt_from_int(eval_start)-timedelta(days=55);end_dt=bt.dt_from_int(eval_end)
    start_s=start_dt.isoformat();end_s=end_dt.isoformat();lo=bt.intdate(start_dt);hi=bt.intdate(end_dt)
    sub=feat[(feat.date>=lo)&(feat.date<=hi)].copy();mask=bt.core.common(sub)
    codes=sorted(set(sub.loc[mask,"code"].astype(str)))
    if not codes:raise RuntimeError("FinMind fallback: empty R10 common universe")
    bt.log(f"[FINMIND] full R10 universe codes={len(codes)} {start_s}..{end_s}")
    rows=[];empty=[]
    for i,code in enumerate(codes,1):
        q=_finmind_wide(code,start_s,end_s)
        if q.empty:
            empty.append(code)
        else:
            q=q.copy();q["date_int"]=bt.pd.to_numeric(bt.pd.to_datetime(q["date"],errors="coerce").dt.strftime("%Y%m%d"),errors="coerce")
            q=q.dropna(subset=["date_int"]);q["date_int"]=q["date_int"].astype(int)
            foreign=_numcol(q,"Foreign_Investor_buy")-_numcol(q,"Foreign_Investor_sell")
            trust=_numcol(q,"Investment_Trust_buy")-_numcol(q,"Investment_Trust_sell")
            dealer=(_numcol(q,"Dealer_buy")-_numcol(q,"Dealer_sell")+_numcol(q,"Dealer_self_buy")-_numcol(q,"Dealer_self_sell")+_numcol(q,"Dealer_Hedging_buy")-_numcol(q,"Dealer_Hedging_sell"))
            rows.extend({"date":int(di),"market":"FINMIND","code":code,"foreign_net":float(f),"trust_net":float(t),"dealer_net":float(d)} for di,f,t,d in zip(q.date_int,foreign,trust,dealer))
        if i%50==0:bt.log(f"[FINMIND] {i}/{len(codes)} codes rows={len(rows)} empty={len(empty)}")
        time.sleep(.05)
    ins=bt.build_inst_features(rows)
    if ins.empty:raise RuntimeError("FinMind fallback returned no institutional rows")
    eval_common=feat[(feat.date>=eval_start)&(feat.date<=eval_end)].copy();eval_common=eval_common.loc[bt.core.common(eval_common),["date","code"]].drop_duplicates()
    have=ins[(ins.date>=eval_start)&(ins.date<=eval_end)][["date","code"]].drop_duplicates()
    matched=eval_common.merge(have,on=["date","code"],how="inner")
    pair_cov=len(matched)/len(eval_common) if len(eval_common) else 0.0
    day_cov=have.date.nunique()/eval_common.date.nunique() if eval_common.date.nunique() else 0.0
    _LAST_FINMIND_AUDIT={"source":"FinMind TaiwanStockInstitutionalInvestorsBuySellWide","documented_range":"2005-now","queried_codes":len(codes),"empty_codes":len(empty),"required_common_pairs":int(len(eval_common)),"matched_pairs":int(len(matched)),"pair_coverage":pair_cov,"day_coverage":day_cov,"token_used":bool(FINMIND_TOKEN)}
    bt.log(f"[FINMIND] pair coverage={pair_cov:.3%}; day coverage={day_cov:.3%}; empty codes={len(empty)}")
    if day_cov<1.0 or pair_cov<0.985:
        raise RuntimeError(f"FinMind institutional coverage insufficient for FULL R10: pair={pair_cov:.3%} day={day_cov:.3%}")
    return ins

def _fetch_inst_dispatch(feat,eval_start,eval_end):
    if _CURRENT_SCENARIO in {"gfc2008","euro2011"}:
        return _finmind_full_institutional(feat,eval_start,eval_end)
    return _ORIG_FETCH_INST(feat,eval_start,eval_end)
bt.fetch_institutional=_fetch_inst_dispatch

# ------------------------------- scenarios ------------------------------------
EXTRA={
 "gfc2008":{"label":"2008 Global Financial Crisis + 2009 recovery","warmup_start":"2007-01-01","eval_start":"2008-01-02","eval_end":"2009-12-31","years":[2007,2008,2009],"r05":True,"mode":"FULL_R10_RECONSTRUCTION_FINMIND","reason":"Full R7 + R0.5 + R10 portfolio layer. 2008/2009 stock-level institutional history reconstructed from FinMind's documented 2005+ dataset; no R7-only fallback allowed."},
 "euro2011":{"label":"2011 Euro-area / US downgrade selloff","warmup_start":"2010-01-01","eval_start":"2011-01-03","eval_end":"2011-12-30","years":[2010,2011],"r05":True,"mode":"FULL_R10_RECONSTRUCTION_FINMIND","reason":"Full R7 + R0.5 + R10 portfolio layer using FinMind 2005+ stock-level institutional history; no partial fallback."},
 "china2015":{"label":"2015 China devaluation / global equity shock","warmup_start":"2014-01-01","eval_start":"2015-01-05","eval_end":"2015-12-31","years":[2014,2015],"r05":True,"mode":"FULL_R10_RECONSTRUCTION","reason":"Full R7 + R0.5 reconstructed causally from official OHLCV and daily institutional history."},
 "tradewar2018":{"label":"2018 US-China trade-war / Q4 selloff","warmup_start":"2017-01-01","eval_start":"2018-01-02","eval_end":"2018-12-28","years":[2017,2018],"r05":True,"mode":"FULL_R10_RECONSTRUCTION","reason":"Full R7 + R0.5 reconstructed causally from official OHLCV and daily institutional history."},
 "crash2024":{"label":"2024 AI bull + 2024-08-05 crash","warmup_start":"2023-01-01","eval_start":"2024-01-02","eval_end":"2024-12-31","years":[2023,2024],"r05":True,"mode":"FULL_R10_IN_SAMPLE_STRESS","reason":"Full R10 event diagnostic; overlaps locked research sample, so not independent OOS evidence."},
 "tariff2025":{"label":"2025 tariff crash + recovery","warmup_start":"2024-01-01","eval_start":"2025-01-02","eval_end":"2025-12-31","years":[2024,2025],"r05":True,"mode":"FULL_R10_IN_SAMPLE_STRESS","reason":"Full R10 event diagnostic; overlaps locked research sample, so not independent OOS evidence."},
}
bt.SCENARIOS.update(EXTRA)
ALL=["validation2021_2025","gfc2008","euro2011","china2015","tradewar2018","covid2020","bear2022","crash2024","tariff2025"]

def validation_grade(result:dict)->dict:
    if result.get("status")!="PASS":return {"grade":"FAIL","reason":"validation run failed"}
    d=result.get("validation_delta") or {}
    checks={"end_nav_pct_abs_le_5pct":abs(float(d.get("end_nav_pct",99)))<=.05,"cagr_abs_le_2pp":abs(float(d.get("cagr",99)))<=.02,"max_dd_abs_le_2pp":abs(float(d.get("max_dd",99)))<=.02,"trades_abs_le_10pct":abs(float(d.get("trades",999)))<=max(10,round(bt.LOCKED_BENCHMARK["completed_trades"]*.10))}
    return {"grade":"PASS" if all(checks.values()) else "FAIL","checks":checks,"delta":d}

def run_one(name:str)->dict:
    global _CURRENT_SCENARIO
    if name not in bt.SCENARIOS:raise SystemExit(f"unknown scenario {name}")
    _CURRENT_SCENARIO=name
    # Start every executable path from the documented causal controls. The
    # legacy baseline helper disables ADV/DD defenses and is forensic-only.
    restore_stress_execution_profile()
    bt.VERSION="AlphaPilot-R10-MAX-0p5-Stress-v1.3-CAUSAL" if name=="validation2021_2025" else "AlphaPilot-R10-MAX-0p5-Stress-v1.3-FULL"
    result=bt.simulate(name,bt.SCENARIOS[name])
    result["execution_profile"]="DOCUMENTED_CAUSAL" if name=="validation2021_2025" else "STRESS_DIAGNOSTIC"
    if name in {"gfc2008","euro2011"}:
        result["institutional_audit"]=_LAST_FINMIND_AUDIT
        if not result.get("r05_enabled"):
            raise RuntimeError(f"{name} is not FULL R10: R0.5 disabled")
    if str(result.get("mode","")).startswith("PARTIAL"):
        raise RuntimeError(f"Partial result prohibited: {name}")
    if name=="validation2021_2025":result["regression_gate"]=validation_grade(result)
    out=bt.OUT_ROOT/"latest"/name/"summary.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--scenario",required=True,choices=ALL);a=ap.parse_args()
    try:print(json.dumps(run_one(a.scenario),ensure_ascii=False,indent=2,allow_nan=False))
    except Exception as exc:
        payload={"status":"ERROR","scenario":a.scenario,"engine_version":bt.VERSION,"generated_at":datetime.now().astimezone().isoformat(),"error":str(exc),"policy":"FULL_R10_ONLY_NO_PARTIAL_RESULTS"}
        out=bt.OUT_ROOT/"latest"/a.scenario;out.mkdir(parents=True,exist_ok=True);(out/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(payload,ensure_ascii=False,indent=2));raise
if __name__=="__main__":main()
