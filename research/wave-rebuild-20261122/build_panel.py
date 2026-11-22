"""全市場日資料面板 2007→今：date, code, open, close, value（成交金額；2007-2025 用 close×volume 估）"""
import sys, glob, zipfile, pandas as pd
R = sys.argv[1]; out = []
for f in sorted(glob.glob(f"{R}/data/history/2007-2019/raw/yearly_*.zip")):
    z = zipfile.ZipFile(f); x = pd.read_csv(z.open(z.namelist()[0]), dtype={"code": str})
    out.append(x)
for f in sorted(glob.glob(f"{R}/data/history/2020-2025/ohlcv_*.parquet")):
    out.append(pd.read_parquet(f))
h = pd.concat(out)
h["code"] = h.code.astype(str)
h = h[h.code.str.fullmatch(r"\d{4}")]
h["date"] = pd.to_datetime(h.date.astype(str))
h["value"] = h.close * h.volume
h = h[["date", "code", "open", "close", "value", "volume"]]
p = []
for f in sorted(glob.glob(f"{R}/data/history/prices_2026/*.csv")):
    x = pd.read_csv(f, dtype={"stock_id": str}); p.append(x)
p = pd.concat(p).rename(columns={"trade_date": "date", "stock_id": "code", "trading_value": "value"})
p["date"] = pd.to_datetime(p.date); p["open"] = float("nan"); p["volume"] = float("nan")
d = []
for f in sorted(glob.glob(f"{R}/data/2026-*/normalized/*_ohlcv.csv")):
    x = pd.read_csv(f, dtype={"stock_id": str}, encoding="utf-8-sig"); d.append(x)
d = pd.concat(d).rename(columns={"trade_date": "date", "stock_id": "code", "trading_value": "value"})
d["date"] = pd.to_datetime(d.date)
p = p[p.date < d.date.min()]
a = pd.concat([h, p[["date", "code", "open", "close", "value", "volume"]], d[["date", "code", "open", "close", "value", "volume"]]])
a = a[a.code.str.fullmatch(r"\d{4}") & (a.close > 0)].drop_duplicates(["date", "code"], keep="last").sort_values(["date", "code"])
a.to_parquet("/home/claude/work/panel.parquet", index=False)
print(len(a), a.date.min(), a.date.max(), a.groupby(a.date.dt.year).code.nunique().to_dict())
