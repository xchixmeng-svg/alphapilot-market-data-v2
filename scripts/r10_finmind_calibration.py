#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibration gate for pre-T86 FinMind institutional fallback.

FinMind documents per-stock institutional history from 2005. Before using it
to reconstruct full R0.5 in 2008/2011, compare its Foreign_Investor and
Investment_Trust net flows against official TWSE/TPEx daily data in an overlap
period. A poor match fails the gate; it is never hand-waved away.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import r10_full_battery as full

bt=full.bt
OUT=bt.OUT_ROOT/"latest"/"finmind_calibration";OUT.mkdir(parents=True,exist_ok=True)
START=date(2015,1,5);END=date(2015,2,13)
# Broad TWSE + TPEx sample, deterministic and frozen.
CODES=["1101","1102","1216","1301","1303","2002","2303","2308","2317","2324","2330","2357","2382","2408","2454","2881","2882","2884","2891","3008","3105","4128","5347","5483","6147"]

def official_rows():
    rows=[];d=START
    while d<=END:
        if d.weekday()<5:
            a=b=[]
            try:a=bt.core.twse_inst(d)
            except Exception:pass
            try:b=bt.core.tpex_inst(d)
            except Exception:pass
            rows.extend(a);rows.extend(b)
        d+=timedelta(days=1)
    q=pd.DataFrame(rows)
    if q.empty:raise RuntimeError("official calibration data empty")
    q=q[q.code.astype(str).isin(CODES)].copy();q=q.drop_duplicates(["date","code"],keep="last")
    return q[["date","code","foreign_net","trust_net"]]

def finmind_rows():
    rows=[]
    for code in CODES:
        q=full._finmind_wide(code,START.isoformat(),END.isoformat())
        if q.empty:continue
        q=q.copy();q["di"]=pd.to_numeric(pd.to_datetime(q.date,errors="coerce").dt.strftime("%Y%m%d"),errors="coerce")
        q=q.dropna(subset=["di"]);q["di"]=q.di.astype(int)
        f=full._numcol(q,"Foreign_Investor_buy")-full._numcol(q,"Foreign_Investor_sell")
        t=full._numcol(q,"Investment_Trust_buy")-full._numcol(q,"Investment_Trust_sell")
        rows.extend({"date":int(di),"code":code,"fm_foreign":float(ff),"fm_trust":float(tt)} for di,ff,tt in zip(q.di,f,t))
    z=pd.DataFrame(rows)
    if z.empty:raise RuntimeError("FinMind calibration data empty")
    return z.drop_duplicates(["date","code"],keep="last")

def main():
    off=official_rows();fm=finmind_rows();m=off.merge(fm,on=["date","code"],how="inner")
    if len(m)<100:raise RuntimeError(f"too few calibration pairs: {len(m)}")
    for c in ["foreign_net","trust_net","fm_foreign","fm_trust"]:m[c]=pd.to_numeric(m[c],errors="coerce")
    m=m.dropna();m["foreign_diff"]=(m.foreign_net-m.fm_foreign).abs();m["trust_diff"]=(m.trust_net-m.fm_trust).abs()
    f_exact=float((m.foreign_diff==0).mean());t_exact=float((m.trust_diff==0).mean());both=float(((m.foreign_diff==0)&(m.trust_diff==0)).mean())
    # Historical sources can receive small same-day buy/sell top-ups; R0.5 uses
    # net flow, so require very high exact net agreement and tiny median error.
    gate=bool(f_exact>=0.97 and t_exact>=0.97 and both>=0.95 and float(m.foreign_diff.median())==0 and float(m.trust_diff.median())==0)
    result={"status":"PASS" if gate else "FAIL","type":"FINMIND_OFFICIAL_CALIBRATION","period":[START.isoformat(),END.isoformat()],"codes_requested":len(CODES),"matched_pairs":int(len(m)),"foreign_exact_rate":f_exact,"trust_exact_rate":t_exact,"both_exact_rate":both,"foreign_median_abs_diff":float(m.foreign_diff.median()),"trust_median_abs_diff":float(m.trust_diff.median()),"gate":gate,"rule":"2008/2011 FULL_R10 cannot be promoted unless this gate passes."}
    m.to_csv(OUT/"comparison.csv",index=False,encoding="utf-8-sig");(OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not gate:raise SystemExit(2)
if __name__=="__main__":main()
