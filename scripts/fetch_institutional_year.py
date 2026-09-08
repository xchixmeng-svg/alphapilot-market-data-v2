#!/usr/bin/env python3
from __future__ import annotations
import json, random, re, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests

year = int(sys.argv[1])
if year not in range(2015, 2020):
    raise SystemExit("year must be 2015..2019")
zp = Path(sys.argv[2])
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

TWSE_T86 = "https://www.twse.com.tw/rwd/zh/fund/T86"
TPEX_OLD = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
TPEX_NEW = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"
S = requests.Session()
S.headers.update({
    "User-Agent": f"Mozilla/5.0 AlphaPilot-Clean-Research-{year}/1.0",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.5",
})

def get_payload(url, params=None, tries=5):
    last = None
    for i in range(tries):
        try:
            r = S.get(url, params=params or {}, timeout=(20, 120))
            r.raise_for_status()
            if not r.content:
                raise RuntimeError("empty response")
            return r.json()
        except Exception as e:
            last = e
            if i + 1 < tries:
                time.sleep(min(20, 2 ** i) + random.uniform(0.05, 0.4))
    raise RuntimeError(f"download failed {url}: {last}")

def nk(x):
    return re.sub(r"[\s_\-()/（）％%]+", "", str(x or "")).lower()

def num(x):
    s = re.sub(r"<[^>]+>", "", str(x or "")).strip()
    s = s.replace(",", "").replace("+", "").replace("−", "-")
    if s in {"", "--", "---", "----", "null", "None"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def nint(x):
    z = num(x)
    return np.nan if not np.isfinite(z) else int(round(z))

def roc(d):
    return f"{d.year - 1911}/{d:%m/%d}"

# This is the same recursive table parser used by the successful 2020-2025 build.
# It handles fields/data, aaData, fieldsN/dataN, nested tables and list-of-dicts.
def tables(payload):
    found = []

    def walk(x):
        if isinstance(x, dict):
            f = x.get("fields")
            d = x.get("data")
            if isinstance(f, list) and isinstance(d, list) and d:
                found.append((f, d))
            aa = x.get("aaData")
            if isinstance(aa, list) and aa:
                found.append((f or [], aa))
            for k, v in x.items():
                if str(k).startswith("fields") and isinstance(v, list):
                    dd = x.get("data" + str(k)[6:])
                    if isinstance(dd, list) and dd:
                        found.append((v, dd))
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            if x and isinstance(x[0], dict):
                found.append((list(x[0].keys()), x))
            for v in x[:10]:
                if isinstance(v, (dict, list)):
                    walk(v)

    walk(payload)
    return found

def dicts(fields, rows):
    return [
        r if isinstance(r, dict)
        else {fields[i]: r[i] if i < len(r) else None for i in range(len(fields))}
        for r in rows
    ]

def pick(row, *names):
    m = {nk(k): v for k, v in row.items()}
    for name in names:
        if nk(name) in m:
            return m[nk(name)]
    return None

def stock_code(row):
    c = str(pick(row, "Code", "SecuritiesCompanyCode", "證券代號", "代號", "股票代號") or "")
    c = c.strip().replace('"', "").replace("=", "")
    return c if re.fullmatch(r"\d{4}", c) else None

def match(row, institutions, actions, reject=()):
    candidates = []
    for k, v in row.items():
        z = nk(k)
        if any(z.startswith(nk(x)) for x in reject):
            continue
        if any(nk(x) in z for x in institutions) and any(nk(x) in z for x in actions):
            candidates.append((-len(z), v))
    return sorted(candidates, reverse=True)[0][1] if candidates else None

def foreign_value(row, actions):
    v = match(row, ["外陸資", "外資及陸資"], actions)
    return v if v is not None else match(row, ["Foreign", "外資"], actions, ["外資自營商"])

def trust_value(row, actions):
    return match(row, ["InvestmentTrust", "Trust", "投信"], actions)

def dealer_value(row, actions):
    candidates = []
    for k, v in row.items():
        z = nk(k)
        if z.startswith(nk("外資自營商")):
            continue
        if not (z.startswith(nk("自營商")) or "dealer" in z):
            continue
        if not any(nk(x) in z for x in actions):
            continue
        score = -len(z) + (10000 if "自行買賣" not in z and "避險" not in z else 0)
        candidates.append((score, v))
    return sorted(candidates, reverse=True)[0][1] if candidates else None

def normalize(rows, ds, market):
    result = []
    d = datetime.strptime(ds, "%Y-%m-%d").date()
    for row in rows:
        c = stock_code(row)
        if not c:
            continue
        fb = nint(foreign_value(row, ["Buy", "買進"]))
        fs = nint(foreign_value(row, ["Sell", "賣出"]))
        fn = nint(foreign_value(row, ["Net", "Difference", "買賣超", "差額"]))
        tb = nint(trust_value(row, ["Buy", "買進"]))
        ts = nint(trust_value(row, ["Sell", "賣出"]))
        tn = nint(trust_value(row, ["Net", "Difference", "買賣超", "差額"]))
        db = nint(dealer_value(row, ["Buy", "買進"]))
        dsell = nint(dealer_value(row, ["Sell", "賣出"]))
        dn = nint(dealer_value(row, ["Net", "Difference", "買賣超", "差額"]))
        if not np.isfinite(fn) and np.isfinite(fb) and np.isfinite(fs):
            fn = int(fb - fs)
        if not np.isfinite(tn) and np.isfinite(tb) and np.isfinite(ts):
            tn = int(tb - ts)
        if not np.isfinite(dn) and np.isfinite(db) and np.isfinite(dsell):
            dn = int(db - dsell)
        result.append({
            "date": d.strftime("%Y-%m-%d"),
            "market": market,
            "code": c,
            "name": str(pick(row, "證券名稱", "名稱", "CompanyName", "Name") or "").strip(),
            "foreign_net": fn,
            "trust_net": tn,
            "dealer_net": dn,
        })
    return result

with zipfile.ZipFile(zp) as z:
    member = next(n for n in z.namelist() if n.lower().endswith(".csv"))
    with z.open(member) as fh:
        daily = pd.read_csv(fh, dtype={"code": str}, usecols=["date", "code"])
daily["code"] = daily.code.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
raw_dates = sorted(set(pd.to_numeric(
    daily.loc[daily.code == "0050", "date"], errors="coerce"
).dropna().astype(int).astype(str)))
dates = [
    datetime.strptime(x, "%Y%m%d").strftime("%Y-%m-%d")
    for x in raw_dates if x.startswith(str(year))
]
if len(dates) < 230:
    raise RuntimeError(f"too few trade dates {year}: {len(dates)}")

def best_table(payload, ds, market):
    best = []
    for fields, rows in tables(payload):
        parsed = normalize(dicts(fields, rows), ds, market)
        if len(parsed) > len(best):
            best = parsed
    return best

def twse(ds):
    payload = get_payload(TWSE_T86, {
        "date": ds.replace("-", ""),
        "selectType": "ALLBUT0999",
        "response": "json",
    })
    best = best_table(payload, ds, "TWSE")
    if not best:
        raise RuntimeError("TWSE empty " + ds)
    return best

def tpex(ds):
    d = datetime.strptime(ds, "%Y-%m-%d").date()
    # Exact successful 2020-2025 option order and date formats:
    # old endpoint uses ROC date; new endpoint uses Gregorian YYYY/MM/DD.
    options = [
        (TPEX_OLD, {"l": "zh-tw", "o": "json", "se": "EW", "t": "D", "d": roc(d)}),
        (TPEX_NEW, {"date": d.strftime("%Y/%m/%d"), "response": "json", "type": "Daily"}),
        (TPEX_NEW, {"date": d.strftime("%Y/%m/%d"), "response": "json"}),
    ]
    errors = []
    for url, params in options:
        try:
            best = best_table(get_payload(url, params, tries=3), ds, "TPEX")
            if len(best) >= 50:
                return best
            errors.append(f"{url} rows={len(best)}")
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError("TPEX empty " + ds + " | " + " | ".join(errors))

def collect(fn, market, workers):
    by_date = {}
    failures = {}
    pending = list(dates)
    for rnd in range(4):
        if rnd:
            time.sleep(20 * rnd)
        nxt = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fn, ds): ds for ds in pending}
            for i, future in enumerate(as_completed(futures), 1):
                ds = futures[future]
                try:
                    by_date[ds] = future.result()
                except Exception as e:
                    nxt[ds] = str(e)
                if i % 40 == 0 or i == len(futures):
                    print(
                        f"[{year} {market}] round={rnd + 1} "
                        f"{i}/{len(futures)} pending={len(nxt)}",
                        flush=True,
                    )
        if not nxt:
            failures = {}
            break
        pending = sorted(nxt)
        failures = nxt
    return [r for ds in sorted(by_date) for r in by_date[ds]], failures

# Keep the proven low-concurrency pattern to avoid official endpoint throttling.
tpex_rows, tpex_fail = collect(tpex, "TPEX", 3)
twse_rows, twse_fail = collect(twse, "TWSE", 1)
rows = twse_rows + tpex_rows

coverage = {}
for market, part, failures in [
    ("TWSE", twse_rows, twse_fail),
    ("TPEX", tpex_rows, tpex_fail),
]:
    present = {r["date"] for r in part}
    coverage[market] = {
        "expected": len(dates),
        "present": len(present),
        "ratio": len(present) / len(dates),
        "missing": sorted(set(dates) - present),
        "failures": failures,
    }

df = pd.DataFrame(rows)
if not df.empty:
    df = (
        df.sort_values(["date", "market", "code"])
        .drop_duplicates(["date", "market", "code"], keep="last")
    )
    df.to_csv(out / f"institutional_{year}.csv.gz", index=False, compression="gzip")

manifest = {
    "year": year,
    "rows": len(df),
    "coverage": coverage,
    "status": "PASS" if min(x["ratio"] for x in coverage.values()) >= 0.98 else "FAIL",
    "method": "clean_history_data.py compatible: TWSE T86 + TPEx OLD/NEW Gregorian fallback + recursive field parser",
}
(out / f"manifest_{year}.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
if manifest["status"] != "PASS":
    raise RuntimeError(
        "coverage below 98% "
        + json.dumps({k: v["ratio"] for k, v in coverage.items()})
    )
print("[PASS]", year, len(df), {k: v["ratio"] for k, v in coverage.items()}, flush=True)
