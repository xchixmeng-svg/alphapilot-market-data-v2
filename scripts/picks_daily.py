#!/usr/bin/env python3
"""每日選股 V2（2026-11-22 凍結版）。規則在 picks_core.py（和歷史回測共用、逐筆對帳 746/746 一致）。

每晚燈號算完後跑：
  1. 用今天的開盤價執行昨晚的單（N+1）
  2. 今天收盤後，照燈號事件、三燈、熊市指標決定明天的單
  3. 寫 data/extra/picks/：state.json（模擬帳戶，100 萬起）、latest.json（推播全文）、trades.csv（每筆成交）
股價還原：上市第 5 天以後單日漲跌超過 10.5% 當成減資／分割（報酬算 0），和回測一樣；不含股息。
開盤漲停判斷用原始價格（前一天收盤，除權息當天用證交所參考價），不用到當天收盤。
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import picks_core as K  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "extra" / "picks"
LIGHTS = DATA / "extra" / "lights"
# 正常範圍（V2 回測 2008～2025 的最差值）與警戒線
NORMAL = {"dd": (-28.5, -35), "year": (-22.4, -28), "win30": (17, 13), "streak": (19, 24)}


# ---------- 資料 ----------
def load_prices(day, days=480):
    """最近約 480 天全市場：date, code, open, close, value（2025 以前用收盤×成交量）"""
    t = pd.Timestamp(day); t0 = t - pd.Timedelta(days=days)
    rows = []
    for y in range(t0.year, min(t.year, 2025) + 1):
        f = DATA / "history" / "2020-2025" / f"ohlcv_{y}.parquet"
        if f.exists():
            x = pd.read_parquet(f, columns=["date", "code", "open", "close", "volume"])
            x["date"] = pd.to_datetime(x.date.astype(str)); x["value"] = x.close * x.volume
            rows.append(x[["date", "code", "open", "close", "value"]])
    for f in sorted(glob.glob(str(DATA / "history" / "prices_2026" / "*.csv"))):
        x = pd.read_csv(f, dtype={"stock_id": str}).rename(columns={"trade_date": "date", "stock_id": "code", "trading_value": "value"})
        x["date"] = pd.to_datetime(x.date); x["open"] = np.nan
        rows.append(x[["date", "code", "open", "close", "value"]])
    names = {}
    for f in sorted(glob.glob(str(DATA / "20??-??-??" / "normalized" / "*_ohlcv.csv"))):
        x = pd.read_csv(f, dtype={"stock_id": str}, encoding="utf-8-sig").rename(
            columns={"trade_date": "date", "stock_id": "code", "trading_value": "value"})
        x["date"] = pd.to_datetime(x.date)
        rows.append(x[["date", "code", "open", "close", "value"]])
        names.update(dict(zip(x.code, x.name)))
    a = pd.concat(rows)
    a["code"] = a.code.astype(str)
    a = a[a.code.str.fullmatch(r"[1-9]\d{3}") & (a.close > 0) & (a.date <= t) & (a.date >= t0)]
    return a.drop_duplicates(["date", "code"], keep="last").sort_values(["code", "date"]).reset_index(drop=True), names


def features(day):
    a, names = load_prices(day)
    t = pd.Timestamp(day)
    if t not in set(a.date):
        return None
    g = a.groupby("code")
    r = g.close.pct_change(); k = g.cumcount()
    jump = (r.abs() > 0.105) & (k >= 5)
    a["acl"] = (1 + r.where(~jump, 0.0).fillna(0.0)).groupby(a.code).cumprod()
    a["acl"] *= a.groupby("code").close.transform("last") / a.groupby("code").acl.transform("last")   # 最新一天＝原始價
    a["aop"] = a.open * a.acl / a.close
    g = a.groupby("code")
    a["ma60"] = g.acl.transform(lambda x: x.rolling(60).mean())
    a["h250"] = g.acl.transform(lambda x: x.rolling(250, min_periods=120).max())
    a["ret60"] = a.acl / g.acl.shift(60) - 1
    a["val20"] = g.value.transform(lambda x: x.rolling(20, min_periods=15).mean())
    a["val5"] = g.value.transform(lambda x: x.rolling(5, min_periods=3).mean())
    a["n"] = g.cumcount() + 1
    a["prev_raw"] = g.close.shift(1)
    today = a[a.date == t].set_index("code")
    lastrow = a.groupby("code").tail(1).set_index("code")
    F = pd.DataFrame({
        "close": today.acl, "open": today.aop, "raw_open": today.open, "prev_raw": today.prev_raw,
        "ma60": today.ma60, "toh": today.acl / today.h250 - 1, "ret60": today.ret60,
        "val20": today.val20, "val5": today.val5, "n": today.n})
    F["name"] = [names.get(c, "") for c in F.index]
    ti = {d: i for i, d in enumerate(sorted(a.date.unique()))}
    held_last = {c: (float(lastrow.at[c, "acl"]), ti[t] - ti[lastrow.at[c, "date"]]) for c in lastrow.index}
    return F, held_last


def revenue_yoy(day):
    f = DATA / "extra" / "revenue_history.csv"
    if not f.exists():
        return {}
    rv = pd.read_csv(f, dtype={"code": str, "available_date": str})
    rv["ad"] = pd.to_datetime(rv.available_date); rv["yoy"] = pd.to_numeric(rv.yoy_pct, errors="coerce")
    rv = rv[rv.ad <= pd.Timestamp(day)].sort_values(["ad", "period"])
    return rv.groupby("code").yoy.last().to_dict()


def market(day):
    tx = pd.read_csv(DATA / "extra" / "macro" / "TAIEX.csv", parse_dates=["date"]).set_index("date")
    tx = tx[~(tx[["open", "high", "low", "close"]] == tx[["open", "high", "low", "close"]].shift(1)).all(axis=1)]
    c = tx.close[tx.index <= pd.Timestamp(day)]
    ma = c.rolling(60).mean()
    bear = bool(c.iloc[-1] < ma.iloc[-1] and ma.iloc[-1] < ma.iloc[-21])
    return float(c.iloc[-1]), bear, float(ma.iloc[-1])


def lot(sh):
    sh = int(sh)
    return f"{sh // 1000} 張 {sh % 1000} 股" if sh % 1000 else f"{sh // 1000} 張"


# ---------- 主程式 ----------
def main():
    lj = json.loads((LIGHTS / "latest.json").read_text())
    ymd = lj["date"]; day = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    if lj.get("stage") not in ("data", "evening"):
        print("燈號還不是收盤後完整版（stage=%s），先不選股" % lj.get("stage")); return
    if not (DATA / day / "normalized" / "twse_ohlcv.csv").exists():
        print(f"{day} 完整價量（含開盤價）還沒進來，先不選股"); return
    res = features(day)
    if res is None:
        print(f"{day} 沒有價格資料"); return
    F, held_last = res
    yoy = revenue_yoy(day)
    taiex, bear, ma60x = market(day)
    OUT.mkdir(parents=True, exist_ok=True)
    sf = OUT / "state.json"
    lstate = (lj.get("state") or {}).get("state", "cash")
    st = json.loads(sf.read_text()) if sf.exists() else None
    if st is not None and st.get("version") != "v2":
        st = None
    if st is None:      # V2 起點：100 萬，照燈號現在的狀態
        st = K.new_state(); st["version"] = "v2"; st["eq"] = {}
        st["mode"] = {"in": "bull", "bottom": "cash", "wait": "cash", "cash": "cash"}.get(lstate, "cash")
    if st.get("last_date") == day and "base" in st:       # 同一天重跑：從當天開始前的狀態重算
        st = st["base"]
    if st.get("last_date", "") >= day:
        print("今天已經處理過"); return
    st["base"] = json.loads(json.dumps({k: v for k, v in st.items() if k != "base"}, default=str))
    # 1) 執行昨晚的單（今天開盤）
    done = []
    pend = st.get("pend") or {}
    if pend.get("sells") or pend.get("buys") or pend.get("sellall") or st.get("carry"):
        opn = F.open.dropna().to_dict()
        lim_open = F.raw_open.dropna().to_dict()
        lim_ref = F.prev_raw.dropna().to_dict()
        try:      # 除權息當天：漲停用證交所參考價判斷
            ca = pd.read_csv(DATA / "extra" / "corporate_actions.csv", dtype={"code": str, "date": str})
            for _, r in ca[ca.date == day].iterrows():
                if r.reference_price == r.reference_price and r.reference_price:
                    lim_ref[r.code] = float(r.reference_price)
        except Exception as e:  # noqa
            print("除權息參考價讀不到：", e)
        last_px = {c: v[0] for c, v in held_last.items()}
        since = {c: v[1] for c, v in held_last.items()}
        done = K.execute(st, day, opn, lim_open, lim_ref, 1.092, last_px, since, fractional=False)   # 實際股數：整數（整張＋零股）
    # 2) 收盤判斷
    act = (lj.get("key") or {}).get("action", "")
    act = act if act in ("exit", "bottom", "back") else ""
    g3 = int(lj.get("green", 0))
    close = {c: v for c, v in F.close.to_dict().items() if v == v}
    ma = F.ma60.to_dict()

    def candA():
        x = F[(F.val20 >= 3e7) & (F.n >= 120) & F.toh.notna()]
        return list(x.sort_values("toh").index[:40])

    def candM():
        x = F[(F.val20 >= 3e7) & (F.n >= 250) & (F.toh >= -0.001) & (F.val5 > F.val20 * K.VOLR)]
        x = x[[yoy.get(c, np.nan) > 20 for c in x.index]]
        return list(x.sort_values("ret60", ascending=False).index[:40])

    sells, buys, sellall, room, log = K.decide(st, day, close, ma, act, g3, taiex, bear, candA, candM)
    val = st["value"]
    st.setdefault("eq", {})[day] = round(val)
    pos = st["pos"]
    st["pend"]["px"] = {c: close.get(c) for c in pos if c in close}   # 只存持股價格
    sf.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str))
    if done:
        tf = OUT / "trades.csv"
        for d_ in done:
            d_["name"] = F.name.get(d_["code"], "")
        pd.DataFrame(done).to_csv(tf, mode="a", header=not tf.exists(), index=False)

    # ---------- 正常範圍 ----------
    eqs = pd.Series(st["eq"]).sort_index()
    dd = (eqs.iloc[-1] / eqs.max() - 1) * 100
    yr = (eqs.iloc[-1] / eqs.iloc[-250] - 1) * 100 if len(eqs) >= 250 else None
    win30 = streak = None
    tf = OUT / "trades.csv"
    if tf.exists():
        tr = pd.read_csv(tf, dtype={"code": str})
        b = tr[(tr.side == "賣") & (tr.kind == "多頭")]
        if len(b) >= 30:
            win30 = (b.ret.tail(30) > 0).mean() * 100
        s = 0
        for x in b.ret:
            s = s + 1 if x <= 0 else 0
        streak = s
    alarms, checks = [], []
    checks.append(f"{'⛔' if dd <= NORMAL['dd'][1] else '🟡'} 帳戶離高點 {dd:.0f}%（歷史最深 {NORMAL['dd'][0]:.0f}%，超過 {NORMAL['dd'][1]}% 才是警戒）")
    if dd <= NORMAL["dd"][1]: alarms.append("帳戶回落超過歷史範圍")
    if yr is not None:
        checks.append(f"{'⛔' if yr <= NORMAL['year'][1] else '🟡'} 近一年 {yr:+.0f}%（歷史最差 {NORMAL['year'][0]:.0f}%）")
        if yr <= NORMAL["year"][1]: alarms.append("一年虧損超過歷史範圍")
    if win30 is not None:
        checks.append(f"{'⛔' if win30 < NORMAL['win30'][1] else '🟡'} 最近 30 筆多頭股勝率 {win30:.0f}%（歷史最低 {NORMAL['win30'][0]}%）")
        if win30 < NORMAL["win30"][1]: alarms.append("勝率低於歷史範圍")
    if streak is not None:
        checks.append(f"{'⛔' if streak > NORMAL['streak'][1] else '🟡'} 多頭股連續虧損 {streak} 筆（歷史最長 {NORMAL['streak'][0]}）")
        if streak > NORMAL["streak"][1]: alarms.append("連虧超過歷史範圍")

    # ---------- 推播 ----------
    md = f"{int(day[5:7])}/{int(day[8:10])}"
    stage = {"bull": "多頭（追強勢股）", "bottom": "抓底股持有中", "cash": "空手（等抓底）", "cashds": "空手（災難停損後，等三燈亮兩盞）"}[st["mode"]]
    lines = [f"🛒 {md} 選股（模擬帳戶）", f"階段：{stage}"]
    if log:
        lines.append(f"今天：{log}")
    if bear and st["mode"] == "bull":
        lines.append(f"⛔ 熊市指標開啟：大盤 {taiex:,.0f} 在季線 {ma60x:,.0f} 下、季線往下彎，不買新股")
    if alarms:
        lines += ["", "⛔ 超出歷史正常範圍：" + "、".join(alarms), "👉 先停止加碼，策略需要重新檢查"]
    if sellall:
        lines += ["", "⛔ 大盤出場訊號：明天開盤，持股全部賣出"]
    if sells:
        lines += ["", "━━ 明天開盤賣出 ━━"]
        for c in dict.fromkeys(sells):
            p = pos[c]; x = close.get(c, p["epx"])
            lines.append(f"⛔ {F.name.get(c, '')} {c}　{st['why'].get(c, '')}（買 {p['epx']:.2f} → 收 {x:.2f}，{(x / p['epx'] - 1) * 100:+.0f}%）")
    nbuy = min(len(buys), max(room, 0)) if not sellall else 0
    per = val * K.SLOT
    if nbuy > 0:
        lines += ["", f"━━ 明天開盤買進（前 {nbuy} 檔，每檔約 {per / 1e4:.1f} 萬）━━"]
        for k, b in enumerate(buys[:nbuy + 3]):
            c = b["code"]; x = close.get(c); sh = math.floor(per / (x * (1 + K.CIN))) if x else 0
            tag = "🟡" if k < nbuy else "⚪ 備選"
            if b["kind"] == "抄底":
                why = f"離一年高點 {F.toh.get(c) * 100:.0f}%"; stop = "不設停損；漲 50% 停利；大盤再跌 20% 全賣"
            else:
                why = (f"創一年新高、60 天 {F.ret60.get(c) * 100:+.0f}%、量 {F.val5.get(c) / F.val20.get(c):.1f} 倍、"
                       f"營收年增 {yoy.get(c, float('nan')):+.0f}%")
                stop = f"停損：收盤跌破 {x * 0.9:.1f}（−10%）或跌破季線 {F.ma60.get(c):.1f}"
            lines.append(f"{tag} {F.name.get(c, '')} {c}　收 {x:.2f}　約 {lot(sh)}")
            lines.append(f"　　{why}｜{stop}")
        lines.append("開盤漲停就買不到，改買下一檔")
    if not sells and not nbuy and not sellall:
        lines.append("今天沒有要買賣的")
    if pos:
        lines += ["", f"━━ 持股 {len(pos)} 檔 ━━"]
        for c, p in sorted(pos.items(), key=lambda z: str(z[1]["ent"])):
            x = close.get(c, p["epx"])
            lines.append(f"{F.name.get(c, '')} {c}（{p['kind']}）{str(p['ent'])[5:10].replace('-', '/')} 買 {p['epx']:.2f}，"
                         f"現 {x:.2f}，{(x / p['epx'] - 1) * 100:+.0f}%")
    lines += ["", f"模擬帳戶 {val / 1e4:,.1f} 萬（100 萬起，{(val / K.CASH0 - 1) * 100:+.1f}%）", "━━ 正常範圍 ━━"] + checks
    text = "\n".join(lines)
    has_orders = bool(sells or sellall or nbuy > 0 or alarms)
    tsmc = F.close.get("2330")
    out = {"date": ymd, "version": "v2", "mode": st["mode"], "notify": has_orders, "text": text, "value": round(val), "alarms": alarms,
           "tsmc_close": (round(float(tsmc), 1) if tsmc == tsmc else None),
           "picks": [dict(code=b["code"], name=F.name.get(b["code"], ""), market="", kind=b["kind"],
                          buy_max=round(float(F.close.get(b["code"])) * 1.092, 2)) for b in buys[:nbuy]],
           "sells": [dict(code=c, why=st["why"].get(c, "")) for c in dict.fromkeys(sells)], "sellall": sellall,
           "holdings": [dict(code=c, name=F.name.get(c, ""), kind=p["kind"], ent=str(p["ent"])[:10], epx=round(p["epx"], 2)) for c, p in pos.items()]}
    (OUT / "latest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(text)


if __name__ == "__main__":
    main()
