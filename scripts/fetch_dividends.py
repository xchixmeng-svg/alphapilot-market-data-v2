"""一次性：抓 2008–2025 除權息資料。TWSE TWT49U（逐月）、TPEx（逐月，兩種網址都試）、Yahoo 股利/分割（備援，只抓回測交易過的代號）。"""
import requests, json, time, csv, os, datetime as dt
OUT = "data/extra/dividends"; os.makedirs(OUT, exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0"
def months():
    for y in range(2008, 2026):
        for m in range(1, 13):
            a = dt.date(y, m, 1); b = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1))
            yield a, b
# TWSE
rows = []; fields = None
for a, b in months():
    u = f"https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate={a:%Y%m%d}&endDate={b:%Y%m%d}&response=json"
    for k in range(3):
        try:
            j = S.get(u, timeout=30).json(); break
        except Exception as e:
            print("twse retry", a, e); time.sleep(10); j = {}
    if j.get("data"):
        fields = j.get("fields"); rows += j["data"]
    print("TWSE", a, len(j.get("data") or []), flush=True); time.sleep(2.5)
with open(f"{OUT}/twse_twt49u.json", "w", encoding="utf-8") as f: json.dump({"fields": fields, "data": rows}, f, ensure_ascii=False)
# TPEx
tp = []
for a, b in months():
    ry = lambda d: f"{d.year-1911}/{d:%m/%d}"
    got = None
    for u, kind in [(f"https://www.tpex.org.tw/web/stock/exright/dailyquo/exDailyQ_result.php?l=zh-tw&d={ry(a)}&ed={ry(b)}", "get"),
                    ("https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ", "post")]:
        try:
            r = S.get(u, timeout=30) if kind == "get" else S.post(u, data={"startDate": f"{a:%Y/%m/%d}", "endDate": f"{b:%Y/%m/%d}", "response": "json"}, timeout=30)
            j = r.json()
            data = j.get("aaData") or (j.get("tables") or [{}])[0].get("data") or j.get("data")
            if data is not None:
                got = {"src": u, "fields": (j.get("tables") or [{}])[0].get("fields"), "data": data}; break
        except Exception as e:
            print("tpex fail", u[:60], e)
    tp.append({"month": f"{a:%Y-%m}", **(got or {"data": []})})
    print("TPEx", a, len((got or {}).get("data") or []), flush=True); time.sleep(2.5)
with open(f"{OUT}/tpex_exright.json", "w", encoding="utf-8") as f: json.dump(tp, f, ensure_ascii=False)
# Yahoo 備援
codes = open(f"{OUT}/codes.txt").read().split()
ev = {}
for c in codes:
    for suf in (".TW", ".TWO"):
        try:
            j = S.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{c}{suf}?period1=1167609600&period2=1767225600&interval=1mo&events=div%7Csplit", timeout=30).json()
            r = (j.get("chart", {}).get("result") or [None])[0]
            if r and r.get("events"):
                ev[c] = {"suf": suf, **r["events"]}; break
        except Exception as e:
            print("yahoo", c, suf, e)
        time.sleep(0.5)
    print("Y", c, c in ev, flush=True)
with open(f"{OUT}/yahoo_events.json", "w", encoding="utf-8") as f: json.dump(ev, f)
print("done", len(rows), sum(len(x["data"]) for x in tp), len(ev))
