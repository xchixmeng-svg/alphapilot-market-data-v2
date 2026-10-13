#!/usr/bin/env python3
"""Backfill historical Securities and Exchange Act 43-1 disclosures from MOPS."""
from __future__ import annotations
import argparse, calendar, hashlib, html, http.cookiejar, json, re, time
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

MOPS_HOME="https://mops.twse.com.tw/"
MOPS_GATEWAY="https://mops.twse.com.tw/mops/api/redirect"
MOPSOV_FORM="https://mopsov.twse.com.tw/mops/web/t05st01"
MOPSOV_AJAX="https://mopsov.twse.com.tw/mops/web/ajax_t05st01"
UA="Mozilla/5.0 (compatible; AlphaPilot-WhaleRadar/1.3; +github-actions)"
PATTERNS=("43條之1","43 條之 1","第43條之1","第四十三條之一","43-1","43之1")
POSITIVE_CONTROL={"roc_year":115,"month":10,"stock_code":"4194","subject_token":"隆欣投資","date":"2026-10-12"}
MARKETS=("sii","otc","rotc","pub")

class TableParser(HTMLParser):
    def __init__(self): super().__init__(); self.rows=[]; self._row=None; self._cell=None
    def handle_starttag(self,tag,attrs):
        tag=tag.lower()
        if tag=="tr": self._row=[]
        elif tag in {"td","th"} and self._row is not None: self._cell=[]
        elif tag=="br" and self._cell is not None: self._cell.append(" ")
    def handle_data(self,data):
        if self._cell is not None: self._cell.append(data)
    def handle_endtag(self,tag):
        tag=tag.lower()
        if tag in {"td","th"} and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+"," ",html.unescape("".join(self._cell))).strip()); self._cell=None
        elif tag=="tr" and self._row is not None:
            if any(x.strip() for x in self._row): self.rows.append(self._row)
            self._row=None; self._cell=None

def clean(x): return re.sub(r"\s+"," ",str(x or "")).strip()
def contains_43(text):
    c=re.sub(r"\s+","",text); return any(re.sub(r"\s+","",p) in c for p in PATTERNS)
def extract_actor(subject):
    s=clean(subject)
    for p in (r"代取得人(.+?)依(?:證券交易法|證交法)",r"共同取得人(.+?)依(?:證券交易法|證交法)",r"取得人(.+?)依(?:證券交易法|證交法)",r"代(.+?)依(?:證券交易法|證交法)第?43"):
        m=re.search(p,s)
        if m:
            a=m.group(1).strip(" ：:，,。()（）")
            if a and len(a)<=120:return a
    return ""
def parse_roc_date(text):
    text=clean(text)
    for pat,roc in ((r"(?<!\d)(\d{2,3})/(\d{1,2})/(\d{1,2})(?!\d)",True),(r"(?<!\d)(20\d{2})/(\d{1,2})/(\d{1,2})(?!\d)",False)):
        m=re.search(pat,text)
        if m:
            y,mo,d=map(int,m.groups()); y=y+1911 if roc else y
            try:return date(y,mo,d).isoformat()
            except ValueError:pass
    return ""
def event_id(code,d,subject,actor): return hashlib.sha256("|".join((code,d,subject,actor)).encode()).hexdigest()[:24]
def _row_code(cells):
    for c in cells:
        m=re.fullmatch(r"\s*(\d{4,6})\s*",c)
        if m:return m.group(1)
    return ""
def _row_subject(cells):
    for c in cells:
        if contains_43(c):return clean(c)
    candidates=[clean(c) for c in cells if len(clean(c))>=8 and not parse_roc_date(c) and not re.fullmatch(r"[\d:./ -]+",clean(c)) and clean(c) not in {"詳細資料","查詢"}]
    return max(candidates,key=len) if candidates else ""
def parse_page(page,market=""):
    if "FOR SECURITY" in page or "查詢過於頻繁" in page: raise RuntimeError("MOPS security/throttle response")
    p=TableParser(); p.feed(page); events={}; generic=0
    for cells in p.rows:
        code=_row_code(cells); d=parse_roc_date(" | ".join(cells)); subject=_row_subject(cells)
        if not code or not d or not subject:continue
        generic+=1
        if not contains_43(subject):continue
        actor=extract_actor(subject); name=""
        for i,c in enumerate(cells):
            if clean(c)==code and i+1<len(cells):
                n=clean(cells[i+1])
                if n and not parse_roc_date(n):name=n[:80];break
        eid=event_id(code,d,subject,actor); events[eid]={"event_id":eid,"event_type":"sec43_1_disclosure","dataset":"mops_t05st01_historical","stock_code":code,"company_name":name,"actor":actor,"event_date":d,"available_at":d,"market":market,"signal_score":50,"direction":"bullish_watch","subject":subject,"raw_cells":cells,"source":"MOPS historical material information"}
    return list(events.values()),generic

def build_opener_with_session():
    jar=http.cookiejar.CookieJar(); op=build_opener(HTTPCookieProcessor(jar)); h={"User-Agent":UA,"Accept-Language":"zh-TW,zh;q=0.9"}
    for u in (MOPS_HOME,MOPSOV_FORM):
        try:op.open(Request(u,headers=h),timeout=15).read(2048)
        except Exception:pass
    return op

def direct_post(op,params,timeout=30):
    h={"User-Agent":UA,"Accept":"text/html,application/xhtml+xml","Accept-Language":"zh-TW,zh;q=0.9","Content-Type":"application/x-www-form-urlencoded","Referer":MOPSOV_FORM}
    with op.open(Request(MOPSOV_AJAX,data=urlencode(params).encode(),headers=h,method="POST"),timeout=timeout) as r:raw=r.read()
    text=raw.decode("utf-8","replace")
    if len(text)<200:raise RuntimeError(f"short direct MOPS response: {len(text)} bytes")
    return text

def gateway_fetch(op,params,timeout=30):
    payload=json.dumps({"apiName":"ajax_t05st01","parameters":params},ensure_ascii=False).encode()
    h={"User-Agent":UA,"Accept":"application/json, text/plain, */*","Content-Type":"application/json","Origin":"https://mops.twse.com.tw","Referer":MOPS_HOME}
    with op.open(Request(MOPS_GATEWAY,data=payload,headers=h,method="POST"),timeout=timeout) as r:data=json.loads(r.read().decode("utf-8","replace"))
    if data.get("code")!=200:raise RuntimeError(f"MOPS gateway code={data.get('code')}")
    url=(data.get("result") or {}).get("url")
    if not url:raise RuntimeError("MOPS gateway returned no legacy URL")
    with op.open(Request(url,headers={"User-Agent":UA}),timeout=timeout) as r:raw=r.read()
    text=raw.decode("utf-8","replace")
    if len(text)<200:raise RuntimeError(f"short gateway MOPS response: {len(text)} bytes")
    return text

def market_month_params(typek,ry,mo,firstin="true"):
    days=calendar.monthrange(ry+1911,mo)[1]
    return {"encodeURIComponent":"1","step":"1","firstin":firstin,"off":"1","TYPEK":typek,"year":str(ry),"month":f"{mo:02d}","b_date":"01","e_date":f"{days:02d}","key_word":"","queryName":"date","inpuType":"date","co_id":"","keyword4":"","code1":"","TYPEK2":"","checkbtn":""}
def fetch_with_transport(op,t,params): return gateway_fetch(op,params) if t=="gateway" else direct_post(op,params)

def probe_positive_control():
    diag=[]; c=POSITIVE_CONTROL
    # Critical: 4194 is an emerging-stock disclosure. Probe the same date/market query used by backfill,
    # not TYPEK=all company lookup, which returns a generic company table but omits the disclosure.
    for t in ("gateway","direct"):
      for market in MARKETS:
       for firstin in ("true","1","ture"):
        try:
            page=fetch_with_transport(build_opener_with_session(),t,market_month_params(market,c["roc_year"],c["month"],firstin)); events,generic=parse_page(page,market)
            hit=any(e["stock_code"]==c["stock_code"] and c["subject_token"] in e["subject"] and e["event_date"]==c["date"] for e in events)
            diag.append({"transport":t,"market":market,"firstin":firstin,"bytes":len(page),"generic_rows":generic,"events":len(events),"positive_control":hit})
            if hit:return {"transport":t,"market":market},diag
        except Exception as exc:diag.append({"transport":t,"market":market,"firstin":firstin,"positive_control":False,"error":str(exc)})
    return None,diag

def fetch_market_month(transport,typek,ry,mo):
    diag=[]; best=[]; bestg=0
    for firstin in ("true","1","ture"):
        try:
            page=fetch_with_transport(build_opener_with_session(),transport,market_month_params(typek,ry,mo,firstin)); events,g=parse_page(page,typek); diag.append({"firstin":firstin,"bytes":len(page),"generic_rows":g,"events":len(events),"ok":True})
            if g>bestg:bestg=g;best=events
            if g>0:return events,{"generic_rows":g,"attempts":diag}
            if "查無資料" in page:return [],{"generic_rows":0,"attempts":diag}
        except Exception as exc:diag.append({"firstin":firstin,"ok":False,"error":str(exc)})
    if bestg>0:return best,{"generic_rows":bestg,"attempts":diag}
    raise RuntimeError(f"no usable t05st01 monthly table; attempts={diag}")
def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
def self_test():
    sample='<table><tr><td>4194</td><td>禾生技</td><td>115/10/12</td><td>代取得人隆欣投資(股)公司依證券交易法第43條之1第1項規定辦理公告</td></tr></table>'; rows,g=parse_page(sample); assert g==1 and len(rows)==1 and "隆欣投資" in rows[0]["actor"]; print("sec43-backfill self-test: OK")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default="data/whale_radar");ap.add_argument("--start-year",type=int,default=113);ap.add_argument("--end-year",type=int,default=datetime.now().year-1911);ap.add_argument("--sleep",type=float,default=.2);ap.add_argument("--self-test",action="store_true");ap.add_argument("--allow-partial",action="store_true");a=ap.parse_args()
    if a.self_test:self_test();return 0
    root=Path(a.root);latest=root/"latest";latest.mkdir(parents=True,exist_ok=True);route,probe=probe_positive_control();(latest/"sec43_positive_control_probe.json").write_text(json.dumps({"control":POSITIVE_CONTROL,"route":route,"diagnostics":probe},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not route:
        (latest/"sec43_backfill_manifest.json").write_text(json.dumps({"generated_at":datetime.now().astimezone().isoformat(),"status":"BROKEN_SOURCE","reason":"t05st01 monthly market routes could not retrieve positive control","events":0,"named_actors":0,"positive_control":POSITIVE_CONTROL},ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print("historical sec43: BROKEN_SOURCE; positive control not retrieved");return 3
    transport=route["transport"];print(f"positive control OK via {route}");all_rows={};coverage=[];failures=0;generic_total=0;now=date.today()
    for ry in range(a.start_year,a.end_year+1):
      last=now.month if ry+1911==now.year else 12
      for mo in range(1,last+1):
       for market in MARKETS:
        try:
            rows,d=fetch_market_month(transport,market,ry,mo);generic_total+=d.get("generic_rows",0)
            for r in rows:all_rows[r["event_id"]]=r
            coverage.append({"roc_year":ry,"month":mo,"market":market,"status":"OK","events":len(rows),**d});print(f"MOPS {market} {ry}/{mo:02d}: generic={d.get('generic_rows',0)} 43-1={len(rows)}")
        except Exception as exc:
            failures+=1;coverage.append({"roc_year":ry,"month":mo,"market":market,"status":"FAILED","events":0,"error":str(exc)});print(f"MOPS {market} {ry}/{mo:02d}: FAILED {exc}")
            if not a.allow_partial:raise
        time.sleep(max(0,a.sleep))
    rows=sorted(all_rows.values(),key=lambda x:(x.get("event_date",""),x.get("stock_code","")));write_jsonl(root/"historical_sec43_events.jsonl",rows);manifest={"generated_at":datetime.now().astimezone().isoformat(),"source":"MOPS t05st01 historical material information","route":route,"start_roc_year":a.start_year,"end_roc_year":a.end_year,"queries_attempted":len(coverage),"queries_failed":failures,"generic_material_rows":generic_total,"status":"OK" if failures==0 else "PARTIAL","events":len(rows),"named_actors":len({x.get('actor') for x in rows if x.get('actor')}),"positive_control":POSITIVE_CONTROL,"coverage":coverage};(latest/"sec43_backfill_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(latest/"historical_sec43_events.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(f"historical sec43: {len(rows)} events; named actors={manifest['named_actors']}; failures={failures}/{len(coverage)}");return 0 if failures==0 or a.allow_partial else 2
if __name__=="__main__":raise SystemExit(main())
