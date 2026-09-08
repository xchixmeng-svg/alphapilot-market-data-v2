#!/usr/bin/env python3
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

START_YEAR, END_YEAR = 2007, 2025
OUT = Path("taiex_structure_2007_2025"); OUT.mkdir(exist_ok=True)
CACHE = OUT / "monthly_cache"; CACHE.mkdir(exist_ok=True)
URLS = [
    "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST",
    "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST",
]

def parse_obj(obj):
    if obj.get("stat") != "OK": return pd.DataFrame()
    fields, data = obj.get("fields", []), obj.get("data", [])
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data, columns=fields); ren = {}
    for c in df.columns:
        if "日期" in c: ren[c]="date"
        elif "開盤" in c: ren[c]="open"
        elif "最高" in c: ren[c]="high"
        elif "最低" in c: ren[c]="low"
        elif "收盤" in c: ren[c]="close"
    df=df.rename(columns=ren); need=["date","open","high","low","close"]
    if any(c not in df.columns for c in need): raise RuntimeError(f"unexpected fields: {fields}")
    parts=df.date.astype(str).str.split("/",expand=True); gy=parts[0].astype(int)+1911
    df.date=pd.to_datetime(gy.astype(str)+"-"+parts[1]+"-"+parts[2])
    for c in ["open","high","low","close"]: df[c]=pd.to_numeric(df[c].astype(str).str.replace(",","",regex=False),errors="coerce")
    return df[need].dropna()

def fetch_month(year, month):
    cache=CACHE/f"{year:04d}-{month:02d}.csv"
    if cache.exists(): return pd.read_csv(cache,parse_dates=["date"])
    params={"date":f"{year:04d}{month:02d}01","response":"json"}; last=None
    for attempt in range(10):
        for url in URLS:
            try:
                r=requests.get(url,params=params,timeout=45,headers={"User-Agent":"Mozilla/5.0","Accept":"application/json,text/plain,*/*"})
                r.raise_for_status()
                if not r.text.strip(): raise RuntimeError("empty response")
                m=parse_obj(r.json())
                if not m.empty: m.to_csv(cache,index=False)
                return m
            except Exception as e: last=f"{url}: {e}"
        wait=min(30,2*(attempt+1)); print(f"RETRY {year}-{month:02d} attempt={attempt+1}/10 wait={wait}s err={last}",flush=True); time.sleep(wait)
    raise RuntimeError(f"failed {year}-{month:02d}: {last}")

frames=[]; total=(END_YEAR-START_YEAR+1)*12; done=0
for year in range(START_YEAR,END_YEAR+1):
    print(f"YEAR_START {year}",flush=True)
    for month in range(1,13):
        m=fetch_month(year,month); done+=1
        if not m.empty: frames.append(m)
        print(f"PROGRESS {done}/{total} ({done/total:.1%}) month={year}-{month:02d} rows={len(m)}",flush=True)
        time.sleep(.35)
    print(f"YEAR_DONE {year}",flush=True)

if not frames: raise RuntimeError("no TAIEX data fetched")
df=pd.concat(frames,ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)
df=df[(df.date.dt.year>=START_YEAR)&(df.date.dt.year<=END_YEAR)].copy()
missing=sorted(set(range(START_YEAR,END_YEAR+1))-set(df.date.dt.year.unique()))
if missing: raise RuntimeError(f"missing years: {missing}")
df["ret1"]=df.close.pct_change(); df["prior60_high"]=df.close.shift(1).rolling(60,min_periods=60).max(); df["breakout60"]=df.close>df.prior60_high
df["fwd20_ret"]=df.close.shift(-20)/df.close-1; df["breakout_event"]=df.breakout60 & ~df.breakout60.shift(1,fill_value=False)
rows=[]
for year in range(START_YEAR,END_YEAR+1):
    y=df[df.date.dt.year==year].copy(); prev=df[df.date<y.date.iloc[0]].tail(1)
    curve=pd.concat([prev[["date","close"]],y[["date","close"]]],ignore_index=True) if len(prev) else y[["date","close"]].copy()
    dd=curve.close/curve.close.cummax()-1; daily=y.ret1.dropna(); up=int((daily>0).sum()); down=int((daily<0).sum()); flat=int((daily==0).sum())
    events=y[y.breakout_event & y.fwd20_ret.notna()].copy()
    rows.append({"year":year,"trading_days":len(y),"year_end_close":y.close.iloc[-1],"annual_return":y.close.iloc[-1]/(prev.close.iloc[-1] if len(prev) else y.close.iloc[0])-1,"max_drawdown":dd.min(),"annualized_daily_vol":daily.std(ddof=1)*math.sqrt(252),"up_days":up,"down_days":down,"flat_days":flat,"up_day_pct_nonflat":up/(up+down) if up+down else np.nan,"breakout60_events":len(events),"breakout20_success_rate":(events.fwd20_ret>0).mean() if len(events) else np.nan,"breakout20_avg_return":events.fwd20_ret.mean() if len(events) else np.nan})
annual=pd.DataFrame(rows); annual.to_csv(OUT/"taiex_annual_structure_2007_2025.csv",index=False); df.to_csv(OUT/"taiex_daily_2007_2025.csv",index=False)
summary={"source":URLS,"source_owner":"TWSE","date_start":str(df.date.min().date()),"date_end":str(df.date.max().date()),"rows":len(df)}
(OUT/"methodology.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print("FINAL_TABLE",flush=True); print(annual.to_string(index=False),flush=True); print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
