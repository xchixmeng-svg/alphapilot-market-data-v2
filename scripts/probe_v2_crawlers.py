"""Probe public historical sources before the full V2 crawl.

The probe is intentionally small and polite. It stores source URLs, status,
form fields, detected report dates and row counts. No proxy data is produced.
"""
from __future__ import annotations
import io, json, re, time
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

OUT=Path("artifacts/v2_crawler_probe"); OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session()
S.headers.update({"User-Agent":"Mozilla/5.0 AlphaPilot research crawler; contact repository owner"})

def get(url, **params):
    r=S.get(url,params=params,timeout=60)
    r.raise_for_status(); time.sleep(1)
    return r

def mops_industry_probe():
    url="https://mops.twse.com.tw/nas/t21/sii/t21sc03_110_1_0.html"
    r=get(url)
    tables=pd.read_html(io.StringIO(r.content.decode("big5","ignore")))
    found=[]
    for i,t in enumerate(tables):
        text=" ".join(map(str,t.columns))
        if "公司代號" in text:
            found.append({"table":i,"rows":len(t),"columns":[str(c) for c in t.columns]})
    if not found: raise RuntimeError("MOPS 2021-01 industry tables not detected")
    return {"url":url,"status":r.status_code,"industry_tables":found,"bytes":len(r.content)}

def histock_probe():
    base="https://histock.tw/stock/branch.aspx"
    probes=[]
    for params in (
        {"no":"2330"},
        {"no":"2330","date":"2021-01-05"},
        {"no":"2330","dt":"20210105"},
        {"no":"2330","d":"2021/01/05"},
    ):
        r=get(base,**params)
        text=r.text
        soup=BeautifulSoup(text,"html.parser")
        forms=[]
        for f in soup.find_all("form"):
            forms.append({"action":f.get("action"),"method":f.get("method"),
                          "inputs":[{"name":x.get("name"),"value":x.get("value"),"type":x.get("type")} for x in f.find_all("input") if x.get("name")]})
        dates=sorted(set(re.findall(r"20\d{2}[/-]\d{1,2}[/-]\d{1,2}",text)))
        tables=pd.read_html(io.StringIO(text))
        broker_tables=[]
        for i,t in enumerate(tables):
            cols=" ".join(map(str,t.columns))
            if "券商" in cols and ("買" in cols or "賣" in cols):
                broker_tables.append({"table":i,"rows":len(t),"columns":[str(c) for c in t.columns]})
        probes.append({"requested_params":params,"final_url":r.url,"status":r.status_code,
                       "dates":dates[-10:],"broker_tables":broker_tables,"forms":forms,"bytes":len(r.content)})
    return probes

def main():
    report={"policy":"real observations only; no proxies",
            "mops_historical_industry":mops_industry_probe(),
            "histock_candidate_date_probe":histock_probe()}
    (OUT/"probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
