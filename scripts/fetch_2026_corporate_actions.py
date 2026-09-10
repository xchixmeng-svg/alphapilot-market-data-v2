#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, re, time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests

S=requests.Session()
S.headers.update({"User-Agent":"Mozilla/5.0 AlphaPilot-R10-Forward/2026.09","Accept":"application/json,text/plain,*/*"})
COLS=["date","code","market","event_type","official_prev_close","reference_price","cash_dividend_per_share","stock_shares_per_1000","continuity_bridge","source"]

def get(url,params=None,timeout=90):
    last=None
    for i in range(5):
        try:
            r=S.get(url,params=params,timeout=timeout); r.raise_for_status(); return r
        except Exception as e:
            last=e
            if i<4: time.sleep(min(8,2**i))
    raise RuntimeError(f"GET failed {url}: {last}")

def num(v):
    if v is None:return math.nan
    s=str(v).strip().replace(",","").replace("+","")
    if s in {"","--","---","N/A","null","None"}:return math.nan
    try:return float(s)
    except:return math.nan

def code4(v):
    s=str(v or "").strip().strip('=\"')
    return s if re.fullmatch(r"[1-9]\d{3}",s) else None

def roc_to_int(v):
    s=str(v or "").strip(); digits=re.sub(r"\D","",s)
    if len(digits)==8:return int(digits)
    if len(digits)==7:return int(str(int(digits[:3])+1911)+digits[3:])
    m=re.search(r"(\d{2,4})\D+(\d{1,2})\D+(\d{1,2})",s)
    if m:
        y,mo,d=map(int,m.groups()); y=y+1911 if y<1911 else y
        return y*10000+mo*100+d
    return None

def etype(v):
    s=str(v or "").strip()
    if "權" in s and "息" in s:return "EX_RIGHT_DIVIDEND"
    if "權" in s:return "EX_RIGHT"
    if "息" in s:return "EX_DIVIDEND"
    if "減資" in s:return "REDUCTION"
    return "CORPORATE_ACTION"

def pick(r,*candidates):
    for cand in candidates:
        if cand in r:return r[cand]
    for k,v in r.items():
        kk=re.sub(r"\s+","",str(k))
        for cand in candidates:
            cc=re.sub(r"\s+","",cand)
            if cc in kk:return v
    return None

def fetch_twse(start,end):
    j=get("https://www.twse.com.tw/exchangeReport/TWT49U",{"response":"json","strDate":start,"endDate":end}).json()
    fields=j.get("fields") or []; data=j.get("data") or []
    if not fields or not isinstance(data,list):raise RuntimeError(f"TWSE schema {list(j)[:20]}")
    out=[]
    for vals in data:
        r=dict(zip(fields,vals)); c=code4(pick(r,"股票代號","證券代號")); d=roc_to_int(pick(r,"資料日期","除權息日期"))
        if not c or not d or not(int(start)<=d<=int(end)):continue
        prev=num(pick(r,"除權息前收盤價","前日收盤價")); ref=num(pick(r,"除權息參考價","開盤競價基準")); typ=pick(r,"權/息","權息","類別")
        cash=num(pick(r,"現金股利","息值"))
        if not math.isfinite(cash) and "息" in str(typ) and math.isfinite(prev) and math.isfinite(ref):cash=max(0.0,prev-ref)
        out.append({"date":d,"code":c,"market":"TWSE","event_type":etype(typ),"official_prev_close":prev,"reference_price":ref,"cash_dividend_per_share":cash,"stock_shares_per_1000":num(pick(r,"每仟股無償配股","無償配股")),"continuity_bridge":ref/prev if math.isfinite(prev) and prev>0 and math.isfinite(ref) else math.nan,"source":"TWSE:TWT49U"})
    return out,{"fields":fields,"count":len(out),"stat":j.get("stat")}

def fetch_tpex(start,end):
    sy,sm,sd=int(start[:4]),start[4:6],start[6:]; ey,em,ed=int(end[:4]),end[4:6],end[6:]
    params={"l":"zh-tw","d":f"{sy-1911}/{sm}/{sd}","ed":f"{ey-1911}/{em}/{ed}","response":"json"}
    j=get("https://www.tpex.org.tw/web/stock/exright/dailyquo/exDailyQ_result.php",params).json()
    tables=j.get("tables") or []
    if not tables:raise RuntimeError(f"TPEx schema {list(j)[:20]}")
    out=[]; schemas=[]
    for table in tables:
        fields=table.get("fields") or []; data=table.get("data") or []
        if not fields or not isinstance(data,list):continue
        schemas.append(fields)
        for vals in data:
            if not isinstance(vals,list):continue
            r=dict(zip(fields,vals)); c=code4(pick(r,"代號","股票代號","證券代號")); d=roc_to_int(pick(r,"除權息日期","資料日期","日期"))
            if not c or not d or not(int(start)<=d<=int(end)):continue
            prev=num(pick(r,"除權息前收盤價","前收盤價")); ref=num(pick(r,"除權息參考價","參考價","開始交易基準價")); typ=pick(r,"權/息","權息","類別")
            out.append({"date":d,"code":c,"market":"TPEX","event_type":etype(typ),"official_prev_close":prev,"reference_price":ref,"cash_dividend_per_share":num(pick(r,"現金股利","息值")),"stock_shares_per_1000":num(pick(r,"每仟股無償配股","無償配股")),"continuity_bridge":ref/prev if math.isfinite(prev) and prev>0 and math.isfinite(ref) else math.nan,"source":"TPEX:exDailyQ_result"})
    return out,{"count":len(out),"tables":len(tables),"fields":schemas[:2],"stat":j.get("stat")}

def next_day(ds):
    return (datetime.strptime(ds,"%Y%m%d")+timedelta(days=1)).strftime("%Y%m%d")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start"); ap.add_argument("--end",required=True); ap.add_argument("--out",default="data/reference/official_corporate_actions_2026.csv"); a=ap.parse_args()
    out=Path(a.out); meta_path=Path(str(out)+".meta.json")
    existing=pd.DataFrame(columns=COLS); last_checked=None
    if out.exists() and out.stat().st_size>0:
        existing=pd.read_csv(out,dtype={"code":str})
        for c in COLS:
            if c not in existing:existing[c]=math.nan
    if meta_path.exists():
        try:last_checked=json.loads(meta_path.read_text(encoding="utf-8")).get("last_checked_date")
        except Exception:last_checked=None
    start=a.start or (next_day(last_checked) if last_checked else "20260101")
    if int(start)>int(a.end):
        print(json.dumps({"status":"PASS","mode":"NOOP","last_checked_date":last_checked,"rows":len(existing)},ensure_ascii=False));return
    twse,tm=fetch_twse(start,a.end); tpex,pm=fetch_tpex(start,a.end)
    fresh=pd.DataFrame(twse+tpex,columns=COLS)
    df=pd.concat([existing[COLS],fresh],ignore_index=True)
    if not df.empty:
        df["code"]=df["code"].astype(str).str.zfill(4); df["date"]=pd.to_numeric(df["date"],errors="raise").astype(int)
        df=df.sort_values(["date","code","source"]).drop_duplicates(["date","code"],keep="last")
        if df["reference_price"].isna().any():raise RuntimeError("missing reference_price in corporate-action rows")
    out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False,columns=COLS)
    meta={"status":"PASS","generated_at_utc":datetime.utcnow().isoformat()+"Z","last_checked_date":a.end,"increment_range":[start,a.end],"rows":int(len(df)),"new_rows":int(len(fresh)),"twse":tm,"tpex":pm}
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
