#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "history" / "v6-context"
OUT.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 AlphaPilot-V6-Context/1.0", "Accept": "application/json,text/csv,text/html,*/*"})

START_YEAR = 2016
END_DATE = date(2026, 9, 15)
FRED_SERIES = {
    "fed_funds": "DFF",
    "us2y": "DGS2",
    "us10y": "DGS10",
    "us10y2y": "T10Y2Y",
    "vix": "VIXCLS",
    "wti": "DCOILWTICO",
    "twd_per_usd": "DEXTAUS",
    "broad_usd": "DTWEXBGS",
    "nasdaq": "NASDAQCOM",
}


def get(url: str, params=None, timeout=90, tries=5) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if i < tries - 1:
                time.sleep(min(12, 2 ** i))
    raise RuntimeError(f"GET failed {url}: {last}")


def num(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", "").replace("%", "").replace("--", "")
    if s in ("", "-", ".", "nan", "None", "N/A"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def code4(x):
    s = str(x or "").strip().replace("=", "").replace('"', "")
    return s if re.fullmatch(r"[1-9]\d{3}", s) else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_gz(path: Path, rows, fields=None):
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"empty output {path}")
    if fields is None:
        fields, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fields.append(k)
    with gzip.open(path, "wt", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


# ---------- A) Macro state: timestamped public historical series ----------
def build_macro():
    frames = []
    failures = []
    for short, sid in FRED_SERIES.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            r = get(url, timeout=120)
            df = pd.read_csv(io.BytesIO(r.content))
            if len(df.columns) < 2:
                raise RuntimeError(f"bad FRED columns {df.columns.tolist()}")
            df = df.iloc[:, :2].copy()
            df.columns = ["date", short]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df[short] = pd.to_numeric(df[short], errors="coerce")
            df = df[(df["date"].dt.year >= START_YEAR) & (df["date"].dt.date <= END_DATE)]
            frames.append(df)
            print("[MACRO]", short, sid, len(df), flush=True)
        except Exception as e:
            failures.append({"series": sid, "name": short, "error": str(e)})
            print("[MACRO FAIL]", short, sid, e, flush=True)
    if len(frames) < 6:
        raise RuntimeError(f"macro coverage too low: {failures}")
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    out = out.sort_values("date")
    # Forward-fill only previously published observations; no backward fill.
    value_cols = [c for c in out.columns if c != "date"]
    out[value_cols] = out[value_cols].ffill()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    rows = out.to_dict("records")
    p = OUT / "macro_daily.csv.gz"
    write_gz(p, rows)
    return p, failures, len(rows)


# ---------- B) Official TWSE industry-index tape ----------
def parse_twse_industry(d: date):
    ds = d.strftime("%Y%m%d")
    j = get(
        "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
        {"response": "json", "date": ds, "type": "IND"},
        timeout=60,
    ).json()
    if str(j.get("stat", "")).upper() not in ("OK", ""):
        return []
    out = []
    for table in j.get("tables") or []:
        fields = [str(x) for x in (table.get("fields") or [])]
        data = table.get("data") or []
        if not fields or not data:
            continue
        name_i = next((i for i, x in enumerate(fields) if x in ("指數", "指數名稱") or "指數" in x), 0)
        close_i = next((i for i, x in enumerate(fields) if "收盤指數" in x or x == "收盤"), None)
        pct_i = next((i for i, x in enumerate(fields) if "漲跌百分比" in x or "漲跌幅" in x), None)
        if close_i is None and pct_i is None:
            continue
        for vals in data:
            if not isinstance(vals, list) or name_i >= len(vals):
                continue
            name = str(vals[name_i]).strip()
            if not name:
                continue
            close = num(vals[close_i]) if close_i is not None and close_i < len(vals) else None
            pct = num(vals[pct_i]) if pct_i is not None and pct_i < len(vals) else None
            if close is None and pct is None:
                continue
            out.append({"date": d.isoformat(), "index_name": name, "close_index": close, "change_pct": pct})
    return out


def build_industry_indices():
    dates = list(weekdays(date(START_YEAR, 1, 1), END_DATE))
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=10) as ex:
        fut = {ex.submit(parse_twse_industry, d): d for d in dates}
        for i, f in enumerate(as_completed(fut), 1):
            d = fut[f]
            try:
                got = f.result()
                if got:
                    rows.extend(got)
            except Exception as e:
                failures.append({"date": d.isoformat(), "error": str(e)})
            if i % 250 == 0 or i == len(fut):
                print("[IND]", i, "/", len(fut), "rows", len(rows), "hard_fail", len(failures), flush=True)
    rows = sorted({(r["date"], r["index_name"]): r for r in rows}.values(), key=lambda r: (r["date"], r["index_name"]))
    if len({r["date"] for r in rows}) < 1800:
        raise RuntimeError(f"industry index coverage too low: dates={len({r['date'] for r in rows})}")
    p = OUT / "twse_industry_index_daily.csv.gz"
    write_gz(p, rows)
    return p, failures, len(rows), len({r["date"] for r in rows}), len({r["index_name"] for r in rows})


# ---------- C) Current semantic industry snapshot for live explanations only ----------
def build_company_industry_snapshot():
    rows = []
    for suffix, market in (("L", "TWSE"), ("O", "TPEX")):
        url = f"https://mopsfin.twse.com.tw/opendata/t187ap03_{suffix}.csv"
        text = get(url, timeout=90).content.decode("utf-8-sig", "ignore")
        for r in csv.DictReader(io.StringIO(text)):
            c = code4(r.get("公司代號"))
            if not c:
                continue
            rows.append(
                {
                    "code": c,
                    "market": market,
                    "name": r.get("公司簡稱") or r.get("公司名稱") or "",
                    "industry_current": r.get("產業別") or r.get("產業類別") or "",
                    "snapshot_date": END_DATE.isoformat(),
                    "oos_feature_allowed": False,
                }
            )
    rows = sorted({r["code"]: r for r in rows}.values(), key=lambda r: r["code"])
    if len(rows) < 1000:
        raise RuntimeError(f"company snapshot too small: {len(rows)}")
    p = OUT / "company_industry_live_snapshot.csv.gz"
    write_gz(p, rows)
    return p, len(rows)


# ---------- D) Official historical monthly revenue ----------
def flatcols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["|".join(str(v) for v in c if str(v) != "nan") for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def revenue_available_date(y, m):
    if m == 12:
        return date(y + 1, 1, 11).isoformat()
    return date(y, m + 1, 11).isoformat()


def fetch_revenue_month(y, m, market):
    roc = y - 1911
    url = f"https://mops.twse.com.tw/nas/t21/{market}/t21sc03_{roc}_{m}_0.html"
    r = get(url, timeout=90)
    r.encoding = "big5"
    tables = pd.read_html(io.StringIO(r.text))
    out = []
    for t in tables:
        t = flatcols(t)
        cols = list(t.columns)
        code_col = next((c for c in cols if "公司代號" in c), None)
        name_col = next((c for c in cols if "公司名稱" in c), None)
        rev_col = next((c for c in cols if "當月營收" in c and "去年" not in c), None)
        prev_col = next((c for c in cols if "上月營收" in c), None)
        last_col = next((c for c in cols if "去年當月營收" in c), None)
        yoy_col = next((c for c in cols if "去年同月增減" in c), None)
        mom_col = next((c for c in cols if "上月比較增減" in c), None)
        if not code_col or not rev_col:
            continue
        for _, row in t.iterrows():
            c = code4(row.get(code_col))
            if not c:
                continue
            out.append(
                {
                    "period_year": y,
                    "period_month": m,
                    "available_date": revenue_available_date(y, m),
                    "market": "TWSE" if market == "sii" else "TPEX",
                    "code": c,
                    "name": str(row.get(name_col, "")) if name_col else "",
                    "revenue_thousand": num(row.get(rev_col)),
                    "prev_month_revenue_thousand": num(row.get(prev_col)) if prev_col else None,
                    "last_year_month_revenue_thousand": num(row.get(last_col)) if last_col else None,
                    "mops_mom_pct": num(row.get(mom_col)) if mom_col else None,
                    "mops_yoy_pct": num(row.get(yoy_col)) if yoy_col else None,
                }
            )
    return out


def build_revenue():
    jobs = []
    now_y, now_m = END_DATE.year, END_DATE.month
    for y in range(START_YEAR, now_y + 1):
        maxm = 12 if y < now_y else max(1, now_m - 1)
        for m in range(1, maxm + 1):
            for market in ("sii", "otc"):
                jobs.append((y, m, market))
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {ex.submit(fetch_revenue_month, *job): job for job in jobs}
        for i, f in enumerate(as_completed(fut), 1):
            job = fut[f]
            try:
                rows.extend(f.result())
            except Exception as e:
                failures.append({"year": job[0], "month": job[1], "market": job[2], "error": str(e)})
            if i % 30 == 0 or i == len(fut):
                print("[REV]", i, "/", len(fut), "rows", len(rows), "fails", len(failures), flush=True)
    rows = list({(r["period_year"], r["period_month"], r["market"], r["code"]): r for r in rows}.values())
    idx = {(r["code"], r["period_year"], r["period_month"]): r["revenue_thousand"] for r in rows if r["revenue_thousand"] is not None}
    for r in rows:
        y, m, c, v = r["period_year"], r["period_month"], r["code"], r["revenue_thousand"]
        py = idx.get((c, y - 1, m))
        pm = idx.get((c, y - 1, 12)) if m == 1 else idx.get((c, y, m - 1))
        r["yoy_pct"] = None if v is None or py in (None, 0) else (v / py - 1) * 100
        r["mom_pct"] = None if v is None or pm in (None, 0) else (v / pm - 1) * 100
    rows = sorted(rows, key=lambda r: (r["available_date"], r["code"]))
    if len(rows) < 100000:
        raise RuntimeError(f"revenue coverage too low: {len(rows)}")
    p = OUT / "monthly_revenue_point_in_time.csv.gz"
    write_gz(p, rows)
    return p, failures, len(rows)


# ---------- E) Official daily valuation history ----------
def twse_val(d: date):
    ds = d.strftime("%Y%m%d")
    j = get("https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d", {"response": "json", "date": ds, "selectType": "ALL"}, timeout=60).json()
    fields, data = j.get("fields") or [], j.get("data") or []
    out = []
    for vals in data:
        r = dict(zip(fields, vals))
        c = code4(r.get("證券代號"))
        if c:
            out.append({"date": d.isoformat(), "market": "TWSE", "code": c, "pe": num(r.get("本益比")), "pb": num(r.get("股價淨值比")), "dividend_yield_pct": num(r.get("殖利率(%)"))})
    return out


def roc(d: date):
    return f"{d.year-1911:03d}/{d.month:02d}/{d.day:02d}"


def tpex_val(d: date):
    j = get("https://www.tpex.org.tw/web/stock/aftertrading/peratio_analysis/pera_result.php", {"l": "zh-tw", "o": "json", "d": roc(d), "c": ""}, timeout=60).json()
    data = j.get("aaData") or (j.get("tables", [{}])[0].get("data") if j.get("tables") else []) or []
    out = []
    for row in data:
        if isinstance(row, dict):
            c = code4(row.get("SecuritiesCompanyCode") or row.get("股票代號") or row.get("代號"))
            if c:
                out.append({"date": d.isoformat(), "market": "TPEX", "code": c, "pe": num(row.get("PriceEarningRatio") or row.get("本益比")), "pb": num(row.get("PriceBookRatio") or row.get("股價淨值比")), "dividend_yield_pct": num(row.get("DividendYield") or row.get("殖利率(%)"))})
        elif isinstance(row, list) and len(row) >= 7:
            c = code4(row[0])
            if c:
                out.append({"date": d.isoformat(), "market": "TPEX", "code": c, "pe": num(row[2]), "pb": num(row[6]), "dividend_yield_pct": num(row[5])})
    return out


def build_valuation():
    dates = list(weekdays(date(2020, 1, 1), END_DATE))
    jobs = [("TWSE", d) for d in dates] + [("TPEX", d) for d in dates]
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=12) as ex:
        fut = {ex.submit(twse_val if m == "TWSE" else tpex_val, d): (m, d) for m, d in jobs}
        for i, f in enumerate(as_completed(fut), 1):
            m, d = fut[f]
            try:
                got = f.result()
                if got:
                    rows.extend(got)
            except Exception as e:
                failures.append({"market": m, "date": d.isoformat(), "error": str(e)})
            if i % 300 == 0 or i == len(fut):
                print("[VAL]", i, "/", len(fut), "rows", len(rows), "hard_fail", len(failures), flush=True)
    rows = sorted({(r["date"], r["market"], r["code"]): r for r in rows}.values(), key=lambda r: (r["date"], r["market"], r["code"]))
    if len({r["date"] for r in rows}) < 1200:
        raise RuntimeError(f"valuation coverage too low: {len({r['date'] for r in rows})} dates")
    p = OUT / "daily_valuation_point_in_time.csv.gz"
    write_gz(p, rows)
    return p, failures, len(rows), len({r["date"] for r in rows})


def main():
    files = []
    p, macro_fail, macro_rows = build_macro()
    files.append({"file": p.name, "rows": macro_rows, "bytes": p.stat().st_size, "sha256": sha256(p)})

    p, ind_fail, ind_rows, ind_dates, ind_names = build_industry_indices()
    files.append({"file": p.name, "rows": ind_rows, "bytes": p.stat().st_size, "sha256": sha256(p)})

    p, company_rows = build_company_industry_snapshot()
    files.append({"file": p.name, "rows": company_rows, "bytes": p.stat().st_size, "sha256": sha256(p)})

    p, rev_fail, rev_rows = build_revenue()
    files.append({"file": p.name, "rows": rev_rows, "bytes": p.stat().st_size, "sha256": sha256(p)})

    p, val_fail, val_rows, val_dates = build_valuation()
    files.append({"file": p.name, "rows": val_rows, "bytes": p.stat().st_size, "sha256": sha256(p)})

    manifest = {
        "dataset": "AlphaPilot V6 causal context layer",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "end_date": END_DATE.isoformat(),
        "sources": {
            "macro": "FRED timestamped historical series",
            "industry_indices": "Official TWSE MI_INDEX industry index tape",
            "company_industry_live_snapshot": "Official MOPS company open data; LIVE EXPLANATION ONLY",
            "monthly_revenue": "Official MOPS static monthly-revenue archive",
            "valuation": "Official TWSE BWIBBU_d + TPEx peratio_analysis",
        },
        "point_in_time_policy": {
            "monthly_revenue_available_date": "Conservative next-month day 11; no report used earlier than that date.",
            "current_industry_snapshot": "Explicitly prohibited as a historical OOS feature; semantic live explanation only.",
            "macro": "Forward-fill only after an observation date; never backward-filled.",
            "industry_index": "Historical daily observations by their own date.",
            "valuation": "Historical daily observations by their own date.",
        },
        "coverage": {
            "macro_rows": macro_rows,
            "industry_index_rows": ind_rows,
            "industry_index_dates": ind_dates,
            "industry_index_names": ind_names,
            "company_snapshot_rows": company_rows,
            "monthly_revenue_rows": rev_rows,
            "valuation_rows": val_rows,
            "valuation_dates": val_dates,
        },
        "failures": {
            "macro": macro_fail,
            "industry_hard_failures": ind_fail[:5000],
            "revenue": rev_fail,
            "valuation_hard_failures": val_fail[:5000],
        },
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[V6 CONTEXT DONE]", json.dumps(manifest["coverage"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
