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

def post(url, data):
    r=S.post(url,data=data,timeout=60)
    r.raise_for_status(); time.sleep(1)
    return r

def decode_tw_text(content):
    for encoding in ("utf-8-sig","big5","cp950"):
        try: return content.decode(encoding)
        except UnicodeDecodeError: pass
    return content.decode("utf-8","replace")

def table_summary(html):
    tables=pd.read_html(io.StringIO(html))
    found=[]
    for i,t in enumerate(tables):
        cols=[str(c) for c in t.columns]
        joined=" ".join(cols)
        if "券商" in joined and ("買" in joined or "賣" in joined):
            found.append({"table":i,"rows":len(t),"columns":cols,
                          "sample":t.head(2).fillna("").astype(str).to_dict("records")})
    return found

def page_dates(text):
    patterns=(r"20\d{2}[/-]\d{1,2}[/-]\d{1,2}",r"20\d{6}")
    return sorted(set(x for p in patterns for x in re.findall(p,text)))

def mops_industry_probe():
    url="https://mops.twse.com.tw/server-java/FileDownLoad"
    results=[]
    for market,path in (("listed","/t21/sii/"),("otc","/t21/otc/")):
        filename="t21sc03_110_1.csv"
        r=post(url,{"step":"9","functionName":"show_file2",
                    "filePath":path,"fileName":filename})
        text=decode_tw_text(r.content)
        if "公司代號" not in text:
            raise RuntimeError(f"MOPS {market} CSV lacks 公司代號; first bytes={text[:120]!r}")
        df=pd.read_csv(io.StringIO(text))
        df.columns=[str(c).strip() for c in df.columns]
        code_col=next((c for c in df.columns if "公司代號" in c),None)
        industry_cols=[c for c in df.columns if "產業" in c]
        valid_codes=int(df[code_col].astype(str).str.strip().str.fullmatch(r"\d{4}").sum()) if code_col else 0
        if not code_col or valid_codes == 0:
            raise RuntimeError(f"MOPS {market} CSV company codes not detected")
        results.append({"market":market,"url":url,"file_path":path,"file_name":filename,
                        "status":r.status_code,"rows":len(df),"columns":df.columns.tolist(),
                        "industry_columns":industry_cols,"valid_4digit_codes":valid_codes,
                        "sample":df.head(2).fillna("").astype(str).to_dict("records"),
                        "bytes":len(r.content)})
    return results

def histock_probe():
    base="https://histock.tw/stock/branch.aspx"
    probes=[]
    for params in (
        {"no":"2330"},
        {"no":"2330","from":"20210105","to":"20210105"},
    ):
        r=get(base,**params)
        text=r.text
        probes.append({"requested_params":params,"final_url":r.url,"status":r.status_code,
                       "dates":page_dates(text)[-20:],"broker_tables":table_summary(text),
                       "bytes":len(r.content)})

    initial=get(base,no="2330")
    soup=BeautifulSoup(initial.text,"html.parser")
    form=soup.find("form")
    if not form: raise RuntimeError("HiStock ASP.NET form not found")
    payload={x.get("name"):x.get("value","") for x in form.find_all("input") if x.get("name") and x.get("type") in ("hidden",None)}
    payload.update({"ctl00$CPHB1$Branch1$tbxStartDate":"2021/01/05",
                    "ctl00$CPHB1$Branch1$tbxEndDate":"2021/01/05",
                    "ctl00$CPHB1$Branch1$btnSearch":"查詢"})
    posted=post(initial.url,payload)
    probes.append({"requested_post_dates":["2021/01/05","2021/01/05"],
                   "submitted_field_names":sorted(payload),"final_url":posted.url,
                   "status":posted.status_code,"dates":page_dates(posted.text)[-20:],
                   "broker_tables":table_summary(posted.text),"bytes":len(posted.content)})

    historical=[p for p in probes[1:] if any("2021" in d for d in p["dates"]) and p["broker_tables"]]
    if not historical:
        raise RuntimeError("HiStock did not return a dated 2021 broker table")
    return probes

def main():
    report={"policy":"real observations only; no proxies"}
    errors={}
    for key,fn in (("mops_historical_industry",mops_industry_probe),
                   ("histock_candidate_date_probe",histock_probe)):
        try: report[key]=fn()
        except Exception as exc: errors[key]=str(exc)
    report["errors"]=errors
    (OUT/"probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(f"crawler probe incomplete: {errors}")

if __name__=="__main__": main()
