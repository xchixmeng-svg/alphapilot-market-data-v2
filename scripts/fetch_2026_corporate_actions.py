#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, re, time
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests

S=requests.Session()
S.headers.update({"User-Agent":"Mozilla/5.0 AlphaPilot-R10-Forward/2026.09","Accept":"application/json,text/plain,*/*"})
TPEX_COLS=["除權息日期","代號","名稱","除權息前收盤價","除權息參考價","權值","息值","權值+息值","權/息","漲停價","跌停價","開始交易基準價","減除股利參考價","現金股利","每仟股無償配股","現金增資股數","現金增資認購價","公開承銷股數","員工認購股數","原股東認購股數","按持股比例仟股認購"]

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
    if v is None: return math.nan
    s=str(v).strip().replace(",","").replace("+","")
    if s in {"","--","---","N/A","null","None"}: return math.nan
    try:return float(s)
    except:return math.nan

def code4(v):
    s=str(v or "").strip()
    return s if re.fullmatch(r"[1-9]\d{3}",s) else None

def roc_to_int(v):
    s=str(v or "").strip()
    digits=re.sub(r"\D","",s)
    if len(digits)==8:return int(digits)
    if len(digits)==7:return int(str(int(digits[:3])+1911)+digits[3:])
    m=re.search(r"(\d{2,3})\D+(\d{1,2})\D+(\d{1,2})",s)
    if m:
        y,mo,d=map(int,m.groups())
        if y<1911:y+=1911
        return y*10000+mo*100+d
    return None

def etype(v):
    s=str(v or "").strip()
    if "權" in s and "息" in s:return "EX_RIGHT_DIVIDEND"
    if "權" in s:return "EX_RIGHT"
    if "息" in s:return "EX_DIVIDEND"
    return "CORPORATE_ACTION"

def fetch_twse(start,end):
    j=get("https://www.twse.com.tw/exchangeReport/TWT49U",{"response":"json","strDate":start,"endDate":end}).json()
    fields=j.get("fields") or []; data=j.get("data") or []
    if not fields or not isinstance(data,list): raise RuntimeError(f"TWSE schema {list(j)[:20]}")
    out=[]
    for vals in data:
        r=dict(zip(fields,vals)); c=code4(r.get("股票代號") or r.get("證券代號")); d=roc_to_int(r.get("資料日期") or r.get("除權息日期"))
        if not c or not d or not(int(start)<=d<=int(end)): continue
        prev=num(r.get("除權息前收盤價")); ref=num(r.get("除權息參考價")); typ=r.get("權/息") or r.get("權息")
        cash=math.nan
        if "息" in str(typ) and "權" not in str(typ) and math.isfinite(prev) and math.isfinite(ref): cash=max(0.0,prev-ref)
        out.append({"date":d,"code":c,"market":"TWSE","event_type":etype(typ),"official_prev_close":prev,"reference_price":ref,"cash_dividend_per_share":cash,"stock_shares_per_1000":math.nan,"continuity_bridge":ref/prev if math.isfinite(prev) and prev>0 and math.isfinite(ref) else math.nan,"source":"TWSE:TWT49U"})
    return out,{"fields":fields,"count":len(out),"stat":j.get("stat")}

def fetch_tpex(start,end):
    sy,sm,sd=int(start[:4]),start[4:6],start[6:]; ey,em,ed=int(end[:4]),end[4:6],end[6:]
    j=get("https://www.tpex.org.tw/web/stock/exright/dailyquo/exDailyQ_result.php",{"l":"zh-tw","d":f"{sy-1911}/{sm}/{sd}","ed":f"{ey-1911}/{em}/{ed}"}).json()
    data=j.get("aaData")
    if not isinstance(data,list): raise RuntimeError(f"TPEx schema {list(j)[:20]}")
    out=[]
    for vals in data:
        if not isinstance(vals,list): continue
        r=dict(zip(TPEX_COLS,vals)); c=code4(r.get("代號")); d=roc_to_int(r.get("除權息日期"))
        if not c or not d or not(int(start)<=d<=int(end)): continue
        prev=num(r.get("除權息前收盤價")); ref=num(r.get("除權息參考價"))
        out.append({"date":d,"code":c,"market":"TPEX","event_type":etype(r.get("權/息")),"official_prev_close":prev,"reference_price":ref,"cash_dividend_per_share":num(r.get("現金股利")),"stock_shares_per_1000":num(r.get("每仟股無償配股")),"continuity_bridge":ref/prev if math.isfinite(prev) and prev>0 and math.isfinite(ref) else math.nan,"source":"TPEX:exDailyQ_result"})
    return out,{"count":len(out)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",default="20260101"); ap.add_argument("--end",required=True); ap.add_argument("--out",default="forward_runtime/official_corporate_actions_2026.csv"); a=ap.parse_args()
    twse,tm=fetch_twse(a.start,a.end); tpex,pm=fetch_tpex(a.start,a.end)
    df=pd.DataFrame(twse+tpex)
    if df.empty: raise RuntimeError("corporate-action dataset empty")
    df=df.sort_values(["date","code","source"]).drop_duplicates(["date","code"],keep="last")
    if df["reference_price"].isna().any(): raise RuntimeError("missing reference_price")
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    meta={"generated_at_utc":datetime.utcnow().isoformat()+"Z","range":[a.start,a.end],"rows":int(len(df)),"twse":tm,"tpex":pm}
    Path(str(out)+".meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
