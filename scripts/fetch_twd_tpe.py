"""台北外匯經紀 16:00 新台幣對美元收盤價（央行開放資料）→ data/extra/macro/TWD_TPE.csv"""
import csv, io, json, re, sys, requests
URL = "https://cpx.cbc.gov.tw/api/OpenData/FTDOpenData_Day"
OUT = "data/extra/macro/TWD_TPE.csv"
r = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/csv,*/*"})
r.raise_for_status()
open("data/extra/macro/TWD_TPE.raw", "wb").write(r.content[:200000])
txt = r.content.decode("utf-8-sig", "replace")
rows = {}
def add(d, v):
    m = re.match(r"(\d{4})\D?(\d{1,2})\D?(\d{1,2})", str(d).strip())
    try: v = float(str(v).replace(",", ""))
    except: return
    if m and 20 < v < 50: rows[f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}"] = v
try:
    data = json.loads(txt)
    def walk(o):
        if isinstance(o, dict):
            ks = list(o.keys()); dk = next((k for k in ks if re.search("date|日期", k, re.I)), None)
            vk = next((k for k in ks if re.search("ntd|usd|匯率|rate|value", k, re.I) and k != dk), None)
            if dk and vk: add(o[dk], o[vk])
            for x in o.values(): walk(x)
        elif isinstance(o, list):
            for x in o: walk(x)
    walk(data)
except json.JSONDecodeError:
    for row in csv.reader(io.StringIO(txt)):
        if len(row) >= 2: add(row[0], row[1])
old = {}
try:
    for row in csv.reader(open(OUT)):
        if row and row[0][:2] == "20": old[row[0]] = float(row[1])
except FileNotFoundError: pass
old.update(rows)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["date", "twd_tpe"])
    for d in sorted(old): w.writerow([d, old[d]])
print("parsed", len(rows), "total", len(old), min(old) if old else None, max(old) if old else None)
if not rows: print(txt[:1500]); sys.exit(1)
