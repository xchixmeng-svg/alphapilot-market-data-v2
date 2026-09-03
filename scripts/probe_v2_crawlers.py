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
    urls=("https://mops.twse.com.tw/mops/server-java/FileDownLoad",
          "https://mops.twse.com.tw/server-java/FileDownLoad",
          "https://mopsov.twse.com.tw/mops/server-java/FileDownLoad")
    results=[]
    for market,path in (("listed","/t21/sii/"),("otc","/t21/otc/")):
        filename="t21sc03_110_1.csv"
        attempts=[]; r=None; text=""; used_url=""; used_path=""
        for url in urls:
            for full_path in (path,"/home/html/nas"+path):
                try:
                    candidate=post(url,{"step":"9","functionName":"show_file",
                                        "filePath":full_path,"fileName":filename})
                    candidate_text=decode_tw_text(candidate.content)
                    attempts.append({"url":url,"file_path":full_path,"status":candidate.status_code,
                                     "bytes":len(candidate.content),"has_company_code":"公司代號" in candidate_text})
                    if "公司代號" in candidate_text:
                        r,text,used_url,used_path=candidate,candidate_text,url,full_path; break
                except Exception as exc:
                    attempts.append({"url":url,"file_path":full_path,"error":str(exc)})
            if r is not None: break
        if r is None:
            raise RuntimeError(f"MOPS {market} historical CSV variants failed: {attempts}")
        if "公司代號" not in text:
            raise RuntimeError(f"MOPS {market} CSV lacks 公司代號; first bytes={text[:120]!r}")
        df=pd.read_csv(io.StringIO(text))
        df.columns=[str(c).strip() for c in df.columns]
        code_col=next((c for c in df.columns if "公司代號" in c),None)
        industry_cols=[c for c in df.columns if "產業" in c]
        valid_codes=int(df[code_col].astype(str).str.strip().str.fullmatch(r"\d{4}").sum()) if code_col else 0
        if not code_col or valid_codes == 0:
            raise RuntimeError(f"MOPS {market} CSV company codes not detected")
        results.append({"market":market,"url":used_url,"file_path":used_path,"file_name":filename,
                        "status":r.status_code,"rows":len(df),"columns":df.columns.tolist(),
                        "industry_columns":industry_cols,"valid_4digit_codes":valid_codes,
                        "sample":df.head(2).fillna("").astype(str).to_dict("records"),
                        "bytes":len(r.content),"attempts":attempts})
    return results

def dj_branch_probe():
    url="https://fubon-ebrokerdj.fbs.com.tw/z/zc/zco/zco_2330.djhtm"
    r=get(url)
    soup=BeautifulSoup(r.text,"html.parser")
    forms=[]
    for form in soup.find_all("form"):
        fields=[]
        for element in form.find_all(["input","select"]):
            item={"tag":element.name,"name":element.get("name"),"type":element.get("type"),
                  "value":element.get("value")}
            if element.name == "select":
                values=[o.get("value") or o.get_text(strip=True) for o in element.find_all("option")]
                item["option_count"]=len(values); item["option_first_last"]=[values[:2],values[-2:]]
            fields.append(item)
        forms.append({"action":form.get("action"),"method":form.get("method"),"fields":fields})
    script_text="\n".join(s.get_text(" ",strip=True) for s in soup.find_all("script"))
    script_snippets=[]
    for line in script_text.splitlines():
        if any(k in line for k in ("bDate","eDate","zco_","location.href")):
            script_snippets.append(line.strip()[:1000])
    historical=[]
    for params in ({"bDate":"20210105","eDate":"20210105"},
                   {"bDate":"2021/01/05","eDate":"2021/01/05"}):
        test=get(url,**params)
        historical.append({"params":params,"final_url":test.url,"dates":page_dates(test.text)[-20:],
                           "broker_tables":table_summary(test.text),"bytes":len(test.content)})
    query_url="https://fubon-ebrokerdj.fbs.com.tw/z/zc/zco/zco.djhtm"
    for start_end in (("2021/1/5","2021/1/5"),("20210105","20210105")):
        params={"a":"2330","e":start_end[0],"f":start_end[1]}
        test=get(query_url,**params)
        all_tables=[]
        for i,t in enumerate(pd.read_html(io.StringIO(test.text))):
            all_tables.append({"table":i,"rows":len(t),"columns":[str(c) for c in t.columns],
                               "sample":t.head(2).fillna("").astype(str).to_dict("records")})
        historical.append({"params":params,"final_url":test.url,"dates":page_dates(test.text)[-20:],
                           "broker_tables":table_summary(test.text),"all_tables":all_tables,
                           "bytes":len(test.content)})
    return {"url":r.url,"status":r.status_code,"dates":page_dates(r.text)[-20:],
            "broker_tables":table_summary(r.text),"forms":forms,
            "script_snippets":script_snippets[:30],"historical_gets":historical,
            "bytes":len(r.content)}

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
                   ("histock_candidate_date_probe",histock_probe),
                   ("dj_branch_form_probe",dj_branch_probe)):
        try: report[key]=fn()
        except Exception as exc: errors[key]=str(exc)
    report["errors"]=errors
    (OUT/"probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(f"crawler probe incomplete: {errors}")

if __name__=="__main__": main()
