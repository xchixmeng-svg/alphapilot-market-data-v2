#!/usr/bin/env python3
from __future__ import annotations

import json, re, urllib.parse, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"data/history/2020-2025/institutional_2020_2025.parquet"
OUT=ROOT/"v6_2_institutional_admission_audit"
OUT.mkdir(exist_ok=True)

DATES=[20210802,20230703,20240628]
FIELDS=["foreign_net","trust_net","dealer_net"]

def norm(s):
    return re.sub(r"[\s_\-()/（）]+","",str(s or "")).lower()

def num(v):
    if v is None: return None
    s=str(v).strip().replace(",","").replace("+","")
    if s in {"","--","---","null","None"}: return None
    try: return int(round(float(s)))
    except: return None

def fetch_json(url, params=None):
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 AlphaPilot-V6.2-PIT-Audit"})
    with urllib.request.urlopen(req,timeout=15) as r:
        return json.loads(r.read().decode("utf-8-sig"))

def find_table(obj):
    # TWSE/TPEx commonly return tables:[{fields,data}] or direct fields/data.
    if isinstance(obj,dict):
        if isinstance(obj.get("fields"),list) and isinstance(obj.get("data"),list):
            return obj["fields"],obj["data"]
        for key in ("tables","result","data"):
            v=obj.get(key)
            if isinstance(v,list):
                for item in v:
                    try:
                        f,d=find_table(item)
                        if f and d: return f,d
                    except Exception: pass
        for v in obj.values():
            if isinstance(v,(dict,list)):
                try:
                    f,d=find_table(v)
                    if f and d: return f,d
                except Exception: pass
    if isinstance(obj,list) and obj and all(isinstance(x,dict) for x in obj):
        # OpenAPI list-of-dicts
        return list(obj[0].keys()), [[r.get(k) for k in obj[0].keys()] for r in obj]
    raise RuntimeError("table not found")

def dict_rows(fields,data):
    return [{str(k):v for k,v in zip(fields,row)} for row in data if isinstance(row,list)]

def code_of(r):
    for k,v in r.items():
        nk=norm(k)
        if nk in {norm("證券代號"),norm("SecuritiesCompanyCode"),norm("Code"),norm("stock_id")}:
            s=str(v).strip()
            if s: return s
    return None

def best(r, institution_tokens, action_tokens, reject=()):
    cand=[]
    for k,v in r.items():
        nk=norm(k)
        if any(nk.startswith(norm(x)) or norm(x) in nk for x in reject): continue
        if not any(norm(x) in nk for x in institution_tokens): continue
        if not any(norm(x) in nk for x in action_tokens): continue
        # prefer explicit total/main institution line, penalize subcategories
        score=-len(nk)
        if any(x in nk for x in [norm("自行買賣"),norm("避險"),norm("ForeignDealers")]): score-=10000
        cand.append((score,v,k))
    if not cand: return None
    cand.sort(reverse=True,key=lambda x:x[0])
    return num(cand[0][1])

def official_normalize(rows,market):
    out=[]
    for r in rows:
        code=code_of(r)
        if not code: continue
        foreign_tokens=["外陸資","外資及陸資","ForeignInvestorsIncludeMainlandAreaInvestors","Foreign"]
        trust_tokens=["投信","SecuritiesInvestmentTrustCompanies","InvestmentTrust","Trust"]
        dealer_tokens=["自營商","Dealers"]
        o={"market":market,"stock_id":str(code)}
        o["foreign_buy"]=best(r,foreign_tokens,["買進","Buy"],reject=["外資自營商","ForeignDealers"])
        o["foreign_sell"]=best(r,foreign_tokens,["賣出","Sell"],reject=["外資自營商","ForeignDealers"])
        o["foreign_net"]=best(r,foreign_tokens,["買賣超","差額","Difference","Net"],reject=["外資自營商","ForeignDealers"])
        o["trust_buy"]=best(r,trust_tokens,["買進","Buy"])
        o["trust_sell"]=best(r,trust_tokens,["賣出","Sell"])
        o["trust_net"]=best(r,trust_tokens,["買賣超","差額","Difference","Net"])
        o["dealer_buy"]=best(r,dealer_tokens,["買進","Buy"],reject=["ForeignDealers"])
        o["dealer_sell"]=best(r,dealer_tokens,["賣出","Sell"],reject=["ForeignDealers"])
        o["dealer_net"]=best(r,dealer_tokens,["買賣超","差額","Difference","Net"],reject=["ForeignDealers"])
        out.append(o)
    return pd.DataFrame(out)

def fetch_twse(d):
    obj=fetch_json("https://www.twse.com.tw/rwd/zh/fund/T86",{"date":str(d),"selectType":"ALLBUT0999","response":"json"})
    f,rows=find_table(obj)
    print("[V6.2 TWSE FIELDS]", json.dumps({"date":d,"fields":f},ensure_ascii=False),flush=True)
    dr=dict_rows(f,rows)
    out=[]
    for r in dr:
        code=code_of(r)
        if not code: continue
        nm={norm(k):v for k,v in r.items()}
        def exact(*names):
            for name in names:
                k=norm(name)
                if k in nm:
                    return num(nm[k])
            return None
        foreign=exact(
          "外陸資買賣超股數(不含外資自營商)",
          "外資及陸資買賣超股數(不含外資自營商)",
          "外資及陸資(不含外資自營商)買賣超股數"
        )
        trust=exact("投信買賣超股數")
        dealer=exact("自營商買賣超股數")
        # Fail-open only to a deterministic generic parser for historical naming variants,
        # but diagnostics/per-field compare counts below make missing fields visible.
        if foreign is None:
            foreign=best(r,["外陸資","外資及陸資"],["買賣超","差額"],reject=["外資自營商"])
        if trust is None:
            trust=best(r,["投信"],["買賣超","差額"])
        if dealer is None:
            dealer=best(r,["自營商"],["買賣超","差額"],reject=["自行買賣","避險"])
        out.append({
          "market":"TWSE","stock_id":str(code),
          "foreign_net":foreign,"trust_net":trust,"dealer_net":dealer
        })
    z=pd.DataFrame(out)
    if z.empty:
        raise RuntimeError(f"TWSE parsed zero rows for {d}")
    return z

def fetch_tpex(d):
    roc=str(int(str(d)[:4])-1911)+"/"+str(d)[4:6]+"/"+str(d)[6:8]
    url="https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
    obj=fetch_json(url,{"l":"zh-tw","o":"json","d":roc,"se":"EW","t":"D","s":"0,asc"})

    aa=obj.get("aaData") if isinstance(obj,dict) else None
    if not isinstance(aa,list) or not aa:
        aa=None
        tables=obj.get("tables") if isinstance(obj,dict) else None
        if isinstance(tables,list):
            for t in tables:
                if not isinstance(t,dict):
                    continue
                data=t.get("data") or t.get("aaData")
                if isinstance(data,list) and data and isinstance(data[0],list) and len(data[0])>=23:
                    aa=data
                    break
    if not isinstance(aa,list) or not aa:
        shapes=[]
        if isinstance(obj,dict) and isinstance(obj.get("tables"),list):
            for t in obj["tables"]:
                if isinstance(t,dict):
                    data=t.get("data") or t.get("aaData")
                    shapes.append({"keys":list(t.keys()),"rows":len(data) if isinstance(data,list) else None,
                                   "first_len":len(data[0]) if isinstance(data,list) and data and isinstance(data[0],list) else None})
        raise RuntimeError(f"TPEx detail table not found for {d}; top_keys={list(obj.keys()) if isinstance(obj,dict) else type(obj).__name__}; shapes={shapes}")

    first=aa[0] if aa else []
    print("[V6.2 TPEX RAW]", json.dumps({
      "date":d,
      "row_len":len(first) if isinstance(first,list) else None,
      "idx0":first[0] if isinstance(first,list) and len(first)>0 else None,
      "idx10":first[10] if isinstance(first,list) and len(first)>10 else None,
      "idx13":first[13] if isinstance(first,list) and len(first)>13 else None,
      "idx22":first[22] if isinstance(first,list) and len(first)>22 else None,
    },ensure_ascii=False),flush=True)
    rows=[]
    for raw in aa:
        if not isinstance(raw,list) or len(raw)<23:
            continue
        rows.append({
          "market":"TPEX",
          "stock_id":str(raw[0]).strip(),
          "foreign_net":num(raw[10]),
          "trust_net":num(raw[13]),
          "dealer_net":num(raw[22]),
        })
    z=pd.DataFrame(rows)
    if z.empty:
        raise RuntimeError(f"TPEx parsed zero rows for {d}")
    return z,url

def normalize_history(x):
    ren={}
    aliases={
      "trade_date":["trade_date","date"],
      "market":["market"],
      "stock_id":["stock_id","code","證券代號"],
      "foreign_buy":["foreign_buy"],"foreign_sell":["foreign_sell"],"foreign_net":["foreign_net"],
      "trust_buy":["trust_buy"],"trust_sell":["trust_sell"],"trust_net":["trust_net"],
      "dealer_buy":["dealer_buy"],"dealer_sell":["dealer_sell"],"dealer_net":["dealer_net"],
    }
    nmap={norm(c):c for c in x.columns}
    for target,als in aliases.items():
        for a in als:
            if norm(a) in nmap:
                ren[nmap[norm(a)]]=target; break
    z=x.rename(columns=ren).copy()
    required={"trade_date","market","stock_id"}|set(FIELDS)
    miss=required-set(z.columns)
    if miss: raise RuntimeError(f"historical parquet missing expected columns: {sorted(miss)}; columns={list(x.columns)}")
    ds=z["trade_date"].astype(str).str.strip().str.replace(r"\\.0$","",regex=True)
    d=pd.to_datetime(ds,format="%Y%m%d",errors="coerce")
    coverage=float(d.notna().mean())
    if coverage < 0.99:
        raise RuntimeError(f"historical YYYYMMDD parse coverage too low: {coverage}")
    z["date"]=(d.dt.year*10000+d.dt.month*100+d.dt.day).astype("Int64")
    z["market"]=z["market"].astype(str).str.upper()
    z["stock_id"]=z["stock_id"].astype(str).str.strip()
    for c in FIELDS: z[c]=pd.to_numeric(z[c],errors="coerce").astype("Int64")
    return z

def main():
    raw=pd.read_parquet(SRC)
    z=normalize_history(raw)
    z=z[z["date"].notna() & (z["date"]<=20241231)].copy()

    schema={
      "source":"data/history/2020-2025/institutional_2020_2025.parquet",
      "original_columns":list(raw.columns),
      "dtypes":{c:str(t) for c,t in raw.dtypes.items()},
      "rows_pre2025":int(len(z)),
      "dates_pre2025":int(z["date"].nunique()),
      "min_date":int(z["date"].min()),"max_date":int(z["date"].max()),
      "markets":z["market"].value_counts(dropna=False).to_dict(),
      "has_available_at_column":bool(any(norm(c)==norm("available_at") for c in raw.columns)),
      "admission_policy_required":"NEXT_TRADING_SESSION_ONLY; decision_date must be strictly greater than source trade_date",
      "raw_unit_interpretation":"shares, based on official TWSE/TPEx column definitions; do not treat as board lots",
    }
    dup=int(z.duplicated(["date","market","stock_id"],keep=False).sum())
    schema["duplicate_decision_source_keys"]=dup
    schema["nonnull_by_market"]={
      str(m):{fld:int(g[fld].notna().sum()) for fld in FIELDS}
      for m,g in z.groupby("market")
    }

    identities={"status":"NOT_TESTABLE_NET_ONLY_ARCHIVE","reason":"Historical parquet stores only foreign_net/trust_net/dealer_net; buy/sell legs are absent."}

    comparisons=[]; fetch_status=[]
    for d in DATES:
        for market in ("TWSE","TPEX"):
            try:
                if market=="TWSE":
                    off=fetch_twse(d); endpoint="TWSE T86"
                else:
                    off,endpoint=fetch_tpex(d)
                hist=z[(z["date"]==d)&(z["market"]==market)][["stock_id"]+FIELDS].copy()
                m=hist.merge(off,on="stock_id",suffixes=("_hist","_official"))
                field_mismatch={}; field_compared={}; compared_total=0
                for fld in FIELDS:
                    h=pd.to_numeric(m[f"{fld}_hist"],errors="coerce")
                    o=pd.to_numeric(m[f"{fld}_official"],errors="coerce")
                    valid=h.notna()&o.notna()
                    field_mismatch[fld]=int((h[valid]!=o[valid]).sum())
                    field_compared[fld]=int(valid.sum())
                    compared_total+=field_compared[fld]
                mism=sum(field_mismatch.values())
                comparisons.append({
                  "date":d,"market":market,"hist_rows":int(len(hist)),"official_rows":int(len(off)),
                  "matched_codes":int(len(m)),"field_values_compared":compared_total,
                  "field_value_mismatches":mism,
                  **{f"compared_{k}":v for k,v in field_compared.items()},
                  **{f"mismatch_{k}":v for k,v in field_mismatch.items()}
                })
                fetch_status.append({"date":d,"market":market,"status":"PASS_FETCH","endpoint":endpoint})
            except Exception as e:
                fetch_status.append({"date":d,"market":market,"status":"FAIL_FETCH","error":f"{type(e).__name__}: {e}"})

    comp=pd.DataFrame(comparisons)
    fs=pd.DataFrame(fetch_status)
    comp.to_csv(OUT/"OFFICIAL_SPOT_CHECK.csv",index=False,encoding="utf-8-sig")
    fs.to_csv(OUT/"OFFICIAL_FETCH_STATUS.csv",index=False,encoding="utf-8-sig")

    endpoint_ok=bool((fs["status"]=="PASS_FETCH").all()) if len(fs) else False
    duplicate_ok=(dup==0)
    arithmetic_ok=True  # net-only archive; buy/sell identity is not testable.

    market_admission={}
    for market in ("TWSE","TPEX"):
        q=comp[comp["market"]==market].copy() if len(comp) else pd.DataFrame()
        source_nonnull=schema["nonnull_by_market"].get(market,{})
        all_three_source_present=all(int(source_nonnull.get(fld,0))>0 for fld in FIELDS)
        all_dates_compared=bool(
          len(q)==len(DATES)
          and all((q.get(f"compared_{fld}",pd.Series(dtype=int))>0).all() for fld in FIELDS)
        ) if len(q) else False
        zero_mismatch=bool(int(q["field_value_mismatches"].sum())==0) if len(q) else False
        fetched=bool((fs.loc[fs["market"]==market,"status"]=="PASS_FETCH").all()) if "market" in fs.columns and len(fs.loc[fs["market"]==market]) else False
        raw_numbers_ok=bool(all_three_source_present and all_dates_compared and zero_mismatch and fetched and duplicate_ok)
        market_admission[market]={
          "source_nonnull_all_three_fields":all_three_source_present,
          "official_all_dates_compared_all_three_fields":all_dates_compared,
          "official_zero_mismatch":zero_mismatch,
          "official_fetch_ok":fetched,
          "raw_numbers_ok":raw_numbers_ok,
          "pit_ready_as_is":False,
          "status":"RAW_NUMBERS_VERIFIED_NEEDS_PIT_WRAPPER" if raw_numbers_ok else ("REBUILD_REQUIRED" if market=="TPEX" and not all_three_source_present else "NOT_ADMITTED")
        }

    pit_ok=False  # archive itself has no available_at; source admission requires next-session wrapper.
    # Current daily normalizer regression evidence: detect English Foreign Dealers collision risk in code.
    fetch_today=(ROOT/"scripts"/"fetch_today.py").read_text(encoding="utf-8")
    current_normalizer_risk=(
      'reject_prefixes=["外資自營商"]' in fetch_today and "ForeignDealers" not in fetch_today.split("def foreign_value",1)[1].split("def trust_value",1)[0]
    )

    result={
      "status":"PARTIAL_SOURCE_ADMISSION_AUDIT",
      "schema":schema,
      "official_spot_check_rows":int(len(comp)),
      "official_total_field_mismatches":int(comp["field_value_mismatches"].sum()) if len(comp) else None,
      "official_fetch_all_pass":endpoint_ok,
      "market_admission":market_admission,
      "duplicate_key_ok":duplicate_ok,
      "arithmetic_identity_ok":None,
      "arithmetic_identity_note":"Not testable: archive is net-only; numerical admission relies on official net-value spot checks.",
      "source_has_native_available_at":False,
      "pit_admission_ok_as_is":pit_ok,
      "required_pit_wrapper":"For decision session T, only institutional source rows with trade_date < T are eligible. available_session = next trading session after trade_date.",
      "corporate_action_rule":"Never roll raw share counts across V6.1 price_segment_id boundaries. Prefer net_share_ratio = net_shares / same-day traded_shares for rolling evidence.",
      "current_2026_normalizer_foreign_dealer_collision_risk":current_normalizer_risk,
      "final_admission":"DO_NOT_JOIN_AS_IS. TWSE may be admitted only if RAW_NUMBERS_VERIFIED_NEEDS_PIT_WRAPPER then next-session PIT wrapper + segment-safe/scale-safe transforms pass. TPEx historical null fields require official rebuild before any admission."
    }
    (OUT/"V6_2_INSTITUTIONAL_ADMISSION_AUDIT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"PARQUET_SCHEMA.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("[V6.2 INSTITUTIONAL AUDIT]",json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
