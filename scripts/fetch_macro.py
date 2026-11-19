#!/usr/bin/env python3
"""總經與匯率日資料（美國聯準會 FRED，官方、免金鑰），存 data/extra/macro/{series}.csv（date,value）。
  DEXTAUS  新台幣兌美元（1 美元可換多少台幣；數字變大 = 台幣貶值）
  VIXCLS   美國 VIX 恐慌指數
  DGS10    美國 10 年期公債殖利率
每次整份覆蓋（FRED 會修正舊值）。"""
import io, sys, urllib.request
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "extra" / "macro"
SERIES = ["DEXTAUS", "VIXCLS", "DGS10"]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for s in SERIES:
        try:
            req = urllib.request.Request(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=60).read().decode()
            df = pd.read_csv(io.StringIO(raw))
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            if len(df) < 1000:
                raise RuntimeError(f"只有 {len(df)} 筆")
            df.to_csv(OUT / f"{s}.csv", index=False)
            print(f"✅ {s}: {len(df)} 筆 {df.date.iloc[0]}～{df.date.iloc[-1]}")
        except Exception as e:  # noqa
            bad += 1
            print(f"❌ {s}: {e}")
    sys.exit(1 if bad == len(SERIES) else 0)

if __name__ == "__main__":
    main()
