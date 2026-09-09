#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, random, re, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent.parent
RAW=ROOT/"input"/"raw"
BASE2020=ROOT/"input"/"base2020"
OUT=ROOT/"validation_input"
OUT.mkdir(parents=True,exist_ok=True)
HEAD={"User-Agent":"Mozilla/5.0 AlphaPilot-Backward-Validation/1.0","Accept":"application/json,text/plain,*/*"}

def num(v):
    s=str(v if v is not None else "").strip().replace(",","").replace("+","")
    if s in ("","--","---","null","None"): return None
    try: return float(s)
    except: return None

def code4(v):
    s=str(v or "").strip().strip("=").strip('"')
    return s if re.fullmatch(r"\d{4}",s) else None

def parse_date(v):
    s=re.sub(r"[^0-9]","",str(v or ""))
    try:
        if len(s)==8: return datetime.strptime(s,"%Y%m%d").date()
        if len(s)==7: return datetime.strptime(str(int(s[:3])+1911)+s[3:],"%Y%m%d").date()
    except: pass
    return None

def get_json(url,params=None,attempts=4,base=1.5):
    last=None
    for i in range(attempts):
        try:
            r=requests.get(url,params=params,headers=HEAD,timeout=20)
            r.raise_for_status(); j=r.json()
            stat=str(j.get("stat","")).strip() if isinstance(j,dict) else ""
            if isinstance(j,dict) and j and "過於頻繁" not in stat and "沒有符合" not in stat:
                return j
            last=RuntimeError("empty/throttled "+stat)
        except Exception as e: last=e
        time.sleep(min(12,base*(2**min(i,3)))+random.uniform(.1,.5))
    raise RuntimeError(f"GET failed {url} {params}: {last}")

years=[]
trade_dates=set()
for y in range(2015,2020):
    zp=RAW/f"yearly_{y}.zip"
    if not zp.exists(): raise FileNotFoundError(zp)
    with zipfile.ZipFile(zp) as z:
        bad=z.testzip()
        if bad: raise RuntimeError(f"corrupt {zp}: {bad}")
        member=next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(member) as fh: d=pd.read_csv(fh,dtype={"code":str},low_memory=False)
    d.columns=[str(c).strip().lower() for c in d.columns]
    need={"date","code","name","volume","open","high","low","close"}
    if not need.issubset(d.columns): raise RuntimeError(f"{y} missing {need-set(d.columns)}")
    d=d[["date","code","name","volume","open","high","low","close"]].copy()
    d["date"]=pd.to_numeric(d["date"],errors="coerce").astype("Int64")
    d["code"]=d["code"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(4)
    for c in ["volume","open","high","low","close"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["date","code","open","high","low","close"])
    d["date"]=d["date"].astype(int)
    d=d[d["date"].astype(str).str.startswith(str(y))]
    d=d.sort_values(["code","date"]).drop_duplicates(["date","code"],keep="last")
    if d.empty or not (d["code"]=="0050").any(): raise RuntimeError(f"{y} empty or missing 0050")
    d.to_parquet(OUT/f"ohlcv_{y}.parquet",index=False)
    ds=sorted(set(d.loc[d.code=="0050","date"].astype(str)))
    trade_dates.update(ds)
    years.append({"year":y,"rows":len(d),"dates":len(ds),"min":int(d.date.min()),"max":int(d.date.max()),"sha256":hashlib.sha256((OUT/f"ohlcv_{y}.parquet").read_bytes()).hexdigest()})
    print("[OHLCV]",years[-1],flush=True)

p2020=BASE2020/"ohlcv_2020.parquet"
if not p2020.exists(): raise FileNotFoundError(p2020)
(OUT/"ohlcv_2020.parquet").write_bytes(p2020.read_bytes())
d20=pd.read_parquet(p2020,columns=["date","code"])
if pd.api.types.is_datetime64_any_dtype(d20.date): vals=d20.date.dt.strftime("%Y%m%d")
else: vals=d20.date.astype(str).str.replace("-","",regex=False)
trade_dates.update(vals[d20.code.astype(str).str.zfill(4)=="0050"].tolist())

hist_dates=sorted(datetime.strptime(x,"%Y%m%d").strftime("%Y-%m-%d") for x in trade_dates if x.startswith(("2015","2016","2017","2018","2019")))

def twse_t86(ds):
    j=get_json("https://www.twse.com.tw/rwd/zh/fund/T86",{"response":"json","date":ds.replace("-",""),"selectType":"ALLBUT0999"})
    fields=j.get("fields") or []; data=j.get("data") or []; out=[]
    for vals in data:
        r=dict(zip(fields,vals)); c=code4(r.get("證券代號"))
        if not c: continue
        out.append({"date":ds,"market":"TWSE","code":c,"name":r.get("證券名稱",""),"foreign_net":num(r.get("外陸資買賣超股數(不含外資自營商)")),"trust_net":num(r.get("投信買賣超股數")),"dealer_net":num(r.get("自營商買賣超股數(自行買賣)")),"total_net":num(r.get("三大法人買賣超股數"))})
    if not out: raise RuntimeError("TWSE T86 empty "+ds)
    return out

def roc(ds):
    d=datetime.strptime(ds,"%Y-%m-%d").date(); return f"{d.year-1911:03d}/{d.month:02d}/{d.day:02d}"

def tpex_daily(ds):
    j=get_json("https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade",{"type":"Daily","sect":"EW","date":roc(ds),"id":"","response":"json"})
    tables=j.get("tables") or []
    if not tables or not tables[0].get("data"): raise RuntimeError("TPEX empty "+ds)
    out=[]
    for r in tables[0]["data"]:
        if not isinstance(r,list) or len(r)<24: continue
        c=code4(r[0])
        if not c: continue
        out.append({"date":ds,"market":"TPEX","code":c,"name":str(r[1]).strip(),"foreign_net":num(r[10]),"trust_net":num(r[13]),"dealer_net":num(r[22]),"total_net":num(r[23])})
    if not out: raise RuntimeError("TPEX parsed zero "+ds)
    return out

def fetch_market(fn,market,workers):
    rows=[]; failed={}; pending=list(hist_dates)
    for rnd in range(2):
        if rnd: time.sleep(10)
        nxt={}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fs={ex.submit(fn,ds):ds for ds in pending}
            for i,f in enumerate(as_completed(fs),1):
                ds=fs[f]
                try: rows.extend(f.result())
                except Exception as e: nxt[ds]=str(e)
                if i%50==0 or i==len(fs): print(f"[{market}] round={rnd+1} {i}/{len(fs)} pending={len(nxt)}",flush=True)
        if not nxt: return rows,{}
        pending=sorted(nxt); failed=nxt
    return rows,failed

shard_dir=os.environ.get("INSTITUTIONAL_SHARD_DIR")
if shard_dir:
    parts=list(Path(shard_dir).glob("**/institutional_*.csv.gz"))
    if len(parts)!=5: raise RuntimeError(f"expected 5 annual institutional shards, got {len(parts)}")
    shard_df=pd.concat([pd.read_csv(p,dtype={"code":str}) for p in parts],ignore_index=True)
    shard_df["code"]=shard_df.code.astype(str).str.zfill(4)
    recs=shard_df.to_dict("records")
    twse_rows=[r for r in recs if r.get("market")=="TWSE"]; tpex_rows=[r for r in recs if r.get("market")=="TPEX"]
    twse_fail={}; tpex_fail={}
    print(f"[SHARDS] files={len(parts)} rows={len(recs)}",flush=True)
else:
    print(f"[FETCH_START] historical institutional dates={len(hist_dates)}",flush=True)
    tpex_rows,tpex_fail=fetch_market(tpex_daily,"TPEX",8)
    twse_rows,twse_fail=fetch_market(twse_t86,"TWSE",6)

inst_old=pd.DataFrame(twse_rows+tpex_rows)
if inst_old.empty: raise RuntimeError("no historical institutional data")
inst_old["date"]=pd.to_datetime(inst_old.date).dt.strftime("%Y%m%d").astype(int)
inst_old["code"]=inst_old.code.astype(str).str.zfill(4)
inst20=pd.read_parquet(BASE2020/"institutional_2020_2025.parquet")
if pd.api.types.is_datetime64_any_dtype(inst20.date): inst20["date"]=inst20.date.dt.strftime("%Y%m%d").astype(int)
else: inst20["date"]=pd.to_datetime(inst20.date).dt.strftime("%Y%m%d").astype(int)
inst20=inst20[(inst20.date>=20200101)&(inst20.date<=20201231)].copy()
for c in ["foreign_net","trust_net"]:
    if c not in inst_old: inst_old[c]=0.0
    if c not in inst20: inst20[c]=0.0
inst=pd.concat([inst_old[["date","code","foreign_net","trust_net"]],inst20[["date","code","foreign_net","trust_net"]]],ignore_index=True)
inst=inst.sort_values(["date","code"]).drop_duplicates(["date","code"],keep="last")
inst.to_parquet(OUT/"institutional_2015_2020.parquet",index=False)
coverage={}
for market,rows,fail in [("TWSE",twse_rows,twse_fail),("TPEX",tpex_rows,tpex_fail)]:
    have={r["date"] for r in rows}; coverage[market]={"expected":len(hist_dates),"present":len(have),"ratio":len(have)/len(hist_dates),"missing":sorted(set(hist_dates)-have),"failures":fail}
if min(v["ratio"] for v in coverage.values())<0.95: raise RuntimeError("institutional coverage below 95%: "+json.dumps({k:v["ratio"] for k,v in coverage.items()}))

# Reuse locked 2020 corporate actions only for this fast diagnostic build; pre-2020 corporate-action
# completeness is explicitly marked partial and the adapter must quarantine impacted rows.
official20=pd.read_csv(BASE2020/"official_corporate_actions_2020_2025.csv",dtype={"code":str})
official20=official20[(official20.date>=20200101)&(official20.date<=20201231)].copy()
official20.to_csv(OUT/"official_corporate_actions_2015_2020.csv",index=False)
manifest={"status":"PASS_PARTIAL_CORP_ACTIONS","generated_at_utc":datetime.now(timezone.utc).isoformat(),"ohlcv":years+[{"year":2020,"sha256":hashlib.sha256((OUT/"ohlcv_2020.parquet").read_bytes()).hexdigest()}],"institutional":{"rows":len(inst),"coverage":coverage,"sha256":hashlib.sha256((OUT/"institutional_2015_2020.parquet").read_bytes()).hexdigest()},"corporate_actions":{"rows":len(official20),"pre2020_rows":0,"coverage":"2020_locked_only_pre2020_quarantine_required","sha256":hashlib.sha256((OUT/"official_corporate_actions_2015_2020.csv").read_bytes()).hexdigest()},"rules":{"warmup":"2015","evaluation":"2016-2020","incomplete_data_quarantined":True}}
(OUT/"input_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"status":manifest["status"],"institutional_coverage":{k:v["ratio"] for k,v in coverage.items()},"institutional_rows":len(inst)},ensure_ascii=False),flush=True)
