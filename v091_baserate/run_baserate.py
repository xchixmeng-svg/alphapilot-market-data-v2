"""AlphaPilot v0.9.1 全體基準重算（GitHub Actions 版）。

計算邏輯不在這裡：ap/plan.py 與 ap/baserate.py 是 AlphaPilot 程式的原檔複製，執行時會印出 sha256，
必須和交接說明裡的指紋一致，才能確定跑的是同一套規則。這支程式只負責讀 parquet、呼叫核心函式、存結果。

用法：python v091_baserate/run_baserate.py --hist data/history/2020-2025 --out v091_baserate_out
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ap.baserate import compute_frames, report  # noqa: E402

# 與 AlphaPilot config.json 相同（判斷範圍：20 日均成交額 ≥ 3,000 萬、取前 500 名）
CFG = {"min_amount20_twd": 30000000, "universe_top_n": 500}
EXPECT = {"ap/plan.py": "f708376273b56808de143ac6d22881f036cdbfa293e4a4695fae68f7ff7dbc73",
          "ap/baserate.py": "9fe0ec44f93449445425a7e273f65532ae81d53a38141b3f6bec96730cb6f7c3"}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--hist", default="data/history/2020-2025", help="ohlcv_{年}.parquet 所在資料夾")
    a.add_argument("--out", default="v091_baserate_out")
    a.add_argument("--start", default="20200101")
    a.add_argument("--end", default="20241231")
    a.add_argument("--step", type=int, default=3)
    a = a.parse_args()
    ok = True
    for rel, want in EXPECT.items():
        got = hashlib.sha256((HERE / rel).read_bytes()).hexdigest()
        print(f"指紋 {rel}：{got}　{'✓ 一致' if got == want else '✗ 不一致'}")
        ok &= got == want
    if not ok:
        sys.exit("程式指紋不一致：核心檔案被改過，停止。請改回原檔。")
    hist = Path(a.hist)
    if not list(hist.glob("ohlcv_*.parquet")):  # 找不到就在整個 repo 搜尋（例如 route2_input/data/history/2020-2025）
        found = sorted(Path(".").rglob("ohlcv_2020.parquet"))
        if found:
            hist = found[0].parent
            print(f"自動找到歷史資料：{hist}")
    files = sorted(hist.glob("ohlcv_*.parquet"))
    if not files:
        sys.exit(f"{hist} 找不到 ohlcv_*.parquet")
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["date", "code", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"].astype(str)).dt.strftime("%Y%m%d")
        frames.append(df)
        print(f"讀取 {f.name}：{len(df):,} 列 {df['date'].min()}～{df['date'].max()}")
    px = pd.concat(frames, ignore_index=True)
    px["amount"] = px["close"] * px["volume"]  # 歷史封存沒有成交金額，與 AlphaPilot 本機讀法相同
    r = compute_frames(px, CFG, a.start, a.end, a.step)
    txt = report(r)
    print("\n" + txt)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baserate_v091.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.txt").write_text(txt + "\n\n" + "\n".join(f"{k} sha256 {v}" for k, v in EXPECT.items()), encoding="utf-8")
    print(f"\n已輸出 {out}/baserate_v091.json、report.txt")


if __name__ == "__main__":
    main()
