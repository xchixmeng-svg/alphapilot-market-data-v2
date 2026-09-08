#!/usr/bin/env python3
from __future__ import annotations
import json, random, re, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests

year=int(sys.argv[1])
if year not in range(2015,2020): raise SystemExit("year must be 2015..2019")
zp=Path(sys.argv[2])
out=Path(sys.argv[3]); out.mkdir(parents=True,exist_ok=True)
HEAD={"User-Agent":f"Mozilla/5.0 AlphaPilot-Institutional-{year}/2.0","Accept":"application/json,text/plain,*/*"}

def num(v):
    s=str(v if v is not None else "").strip().replace(",","").replace("+","")
    if s in ("","--","---","null","None"): return None
    try:return float(s)
    except:return None

def code4(v):
    s=str(v or "").strip().strip("=").strip('"')
    return s if re.fullmatch(r"\d{4}",s) else None

def get_json(url,params):
    last=None
    for i in range(7):
        try:
            r=requests.get(url,params=params,headers=HEAD,timeout=60); r.raise_for_status(); j=r.json()
            stat=str(j.get("stat","")) if isinstance(j,dict) else ""
            if isinstance(j,dict) and j and "過於頻繁" not in stat and "沒有符合" not in stat:return j
            last=RuntimeError(stat)
        except Exception as e:last=e
        time.sleep(min(45,2.5*(2**min(i,4)))+random.uniform(.1,.8))
    raise RuntimeError(f"{url} {params}: {last}")

with zipfile.ZipFile(zp) as z:
    m=next(n for n in z.namelist() if n.lower().endswith(".csv"))
    with z.open(m) as fh:d=pd.read_csv(fh,dtype={"code":str},usecols=["date","code"])
d["code"]=d.code.astype(str).str.replace(r"\.0$","",regex=True).str.zfill(4)
dates=sorted(set(pd.to_numeric(d.loc[d.code=="0050","date"],errors="coerce").dropna().astype(int).astype(str)))
dates=[datetime.strptime(x,"%Y%m%d").strftime("%Y-%m-%d") for x in dates if x.startswith(str(year))]
if len(dates)<230:raise RuntimeError(f"too few trade dates {year}: {len(dates)}")

def twse(ds):
    j=get_json("https://www.twse.com.tw/rwd/zh/fund/T86",{"response":"json","date":ds.replace("-",""),"selectType":"ALLBUT0999"})
    fields=j.get("fields") or []; out=[]
    for vals in j.get("data") or []:
        r=dict(zip(fields,vals)); c=code4(r.get("證券代號"))
        if c:out.append({"date":ds,"market":"TWSE","code":c,"name":r.get("證券名稱",""),
          "foreign_net":num(r.get("外陸資買賣超股數(不含外資自營商)")),"trust_net":num(r.get("投信買賣超股數"))})
    if not out:raise RuntimeError("TWSE empty "+ds)
    return out

def roc(ds):
    x=datetime.strptime(ds,"%Y-%m-%d");return f"{x.year-1911:03d}/{x.month:02d}/{x.day:02d}"

def tpex(ds):
    j=get_json("https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade",{"type":"Daily","sect":"EW","date":roc(ds),"id":"","response":"json"})
    tabs=j.get("tables") or []; rows=tabs[0].get("data") if tabs else []
    out=[]
    for r in rows or []:
        if isinstance(r,list) and len(r)>=24:
            c=code4(r[0])
            if c:out.append({"date":ds,"market":"TPEX","code":c,"name":str(r[1]).strip(),"foreign_net":num(r[10]),"trust_net":num(r[13])})
    if not out:raise RuntimeError("TPEX empty "+ds)
    return out

def collect(fn,market,workers):
    final={}; rows=[]
    pending=list(dates)
    for rnd in range(4):
        if rnd:time.sleep(20*rnd)
        nxt={}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fs={ex.submit(fn,ds):ds for ds in pending}
            for i,f in enumerate(as_completed(fs),1):
                ds=fs[f]
                try:rows.extend(f.result())
                except Exception as e:nxt[ds]=str(e)
                if i%40==0 or i==len(fs):print(f"[{year} {market}] round={rnd+1} {i}/{len(fs)} pending={len(nxt)}",flush=True)
        if not nxt:return rows,{}
        pending=sorted(nxt);final=nxt
    return rows,final

# Independent market streams. Annual jobs are the durable checkpoint.
tr,tf=collect(tpex,"TPEX",3)
wr,wf=collect(twse,"TWSE",1)
rows=wr+tr
coverage={}
for market,part,fail in [("TWSE",wr,wf),("TPEX",tr,tf)]:
    have={r["date"] for r in part}
    coverage[market]={"expected":len(dates),"present":len(have),"ratio":len(have)/len(dates),"missing":sorted(set(dates)-have),"failures":fail}
if min(x["ratio"] for x in coverage.values())<0.98:
    raise RuntimeError("coverage below 98% "+json.dumps({k:v["ratio"] for k,v in coverage.items()}))
df=pd.DataFrame(rows).sort_values(["date","market","code"]).drop_duplicates(["date","market","code"],keep="last")
df.to_csv(out/f"institutional_{year}.csv.gz",index=False,compression="gzip")
(out/f"manifest_{year}.json").write_text(json.dumps({"year":year,"rows":len(df),"coverage":coverage},ensure_ascii=False,indent=2),encoding="utf-8")
print("[PASS]",year,len(df),{k:v["ratio"] for k,v in coverage.items()},flush=True)
