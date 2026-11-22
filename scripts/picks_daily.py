#!/usr/bin/env python3
"""每日選股（2026-11-22 上線，規則＝11/21 定案的整套策略，研究重建在 research/wave-rebuild-20261122 分支）。

跟著燈號（data/extra/lights/latest.json 的 state）走：
  空手（cash／wait）  不買；燈號出場訊號 → 明天開盤全部賣出
  抓底（bottom）      抓底訊號當晚：選離一年高點跌最深的股票，明天開盤買 10 檔（每檔 9%）
                      抓底股漲 50% → 隔天開盤停利；不設停損
                      三燈亮兩盞（燈號轉回 in）→ 剩下的抓底股隔天開盤賣掉，換多頭股
  多頭（in）          創一年新高＋5 日均量 > 20 日均量 1.3 倍，選 60 天漲最多的，補滿 10 檔
                      收盤比買價跌 10% → 隔天開盤停損；收盤跌破季線（60 日均）→ 隔天開盤出場
                      三燈不到兩盞不買新股；帳戶從高點回落 20% → 多頭股全賣、暫停買進，直到三燈全亮
共同：資金最多用 90%（每檔 9%，最多 10 檔）；股票池 20 日均成交金額 ≥ 3,000 萬、上市滿一年（抓底 120 天）；
      開盤漲停買不到就換下一檔；隔天開盤成交；成本買 0.3%、賣 0.3%（含稅、手續費約略）。
股價用證交所除權息參考價往前還原（data/extra/corporate_actions.csv），停損、季線、一年高點都不會被除息誤觸。

輸出：data/extra/picks/state.json（模擬帳戶，100 萬起）、latest.json（推播全文）、trades.csv（每筆成交）。
"""
import glob
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "extra" / "picks"
LIGHTS = DATA / "extra" / "lights"
SLOT, MAXN, TP, STOP, BRAKE, VOLR = 0.09, 10, 0.50, -0.10, 0.20, 1.3
CIN, COUT = 0.003, 0.003
CASH0 = 1_000_000


# ---------- 資料 ----------
def load_prices(day):
    """最近約 420 天的全市場日資料：date, code, name, open, close, value。"""
    t = pd.Timestamp(day)
    rows = []
    f25 = DATA / "history" / "2020-2025" / "ohlcv_2025.parquet"
    if f25.exists():
        x = pd.read_parquet(f25)
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
    a = a[a.code.str.fullmatch(r"[1-9]\d{3}") & (a.close > 0) & (a.date <= t) & (a.date >= t - pd.Timedelta(days=430))]
    a = a.drop_duplicates(["date", "code"], keep="last")
    return a, names


def adjust(C, O, day):
    """除權息往前還原：事件日 d 之前的價格 × 參考價／前收。"""
    f = DATA / "extra" / "corporate_actions.csv"
    if not f.exists():
        return C, O, {}
    ca = pd.read_csv(f, dtype={"code": str, "date": str})
    ca = ca[(ca.official_prev_close > 0) & (ca.reference_price > 0)]
    today_f = {}
    for _, r in ca.iterrows():
        d = pd.Timestamp(r.date); c = r.code
        if c not in C.columns or d > pd.Timestamp(day) or d <= C.index[0]:
            continue
        k = float(r.reference_price) / float(r.official_prev_close)
        if not (0.2 < k <= 1.0):
            continue
        C.loc[C.index < d, c] *= k; O.loc[O.index < d, c] *= k
        today_f.setdefault(d.strftime("%Y-%m-%d"), {})[c] = k
    # 分割、減資、合併等證交所參考價檔沒列到的：漲跌幅限制 10%，前後兩天收盤差超過 ±11% 一定是股本變動，照實際比例往前還原
    r = C / C.ffill().shift(1)
    for d, c in zip(*np.where((r > 1.11) | (r < 0.89))):
        dd, cc = C.index[d], C.columns[c]
        if dd == C.index[0]:
            continue
        k = float(r.iat[d, c])
        C.loc[C.index < dd, cc] *= k; O.loc[O.index < dd, cc] *= k
        if dd.strftime("%Y-%m-%d") == str(day)[:10]:
            today_f.setdefault(str(day)[:10], {})[cc] = today_f.get(str(day)[:10], {}).get(cc, 1.0) * k
    return C, O, today_f


def features(day):
    a, names = load_prices(day)
    C = a.pivot_table(index="date", columns="code", values="close")
    O = a.pivot_table(index="date", columns="code", values="open").reindex_like(C)
    V = a.pivot_table(index="date", columns="code", values="value").reindex_like(C)
    C, O, evf = adjust(C, O, day)
    t = pd.Timestamp(day)
    if t not in C.index:
        return None
    F = pd.DataFrame({
        "close": C.loc[t], "open": O.loc[t], "prev": C.shift(1).loc[t],
        "ma60": C.rolling(60).mean().loc[t],
        "toh": (C / C.rolling(250, min_periods=120).max() - 1).loc[t],
        "ret60": (C / C.shift(60) - 1).loc[t],
        "val20": V.rolling(20, min_periods=15).mean().loc[t],
        "val5": V.rolling(5, min_periods=3).mean().loc[t],
        "n": C.notna().sum(),
    })
    F["name"] = [names.get(c, "") for c in F.index]
    return F, evf.get(day, {})


def candA(F):
    x = F[(F.val20 >= 3e7) & (F.n >= 120) & F.toh.notna()]
    return list(x.sort_values("toh").index[:40])


def candM(F):
    x = F[(F.val20 >= 3e7) & (F.n >= 250) & (F.toh >= -0.001) & (F.val5 > F.val20 * VOLR)]
    return list(x.sort_values("ret60", ascending=False).index[:40])


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
    F, evf = res
    OUT.mkdir(parents=True, exist_ok=True)
    sf = OUT / "state.json"
    st = json.loads(sf.read_text()) if sf.exists() else {"cash": CASH0, "pos": {}, "pending": None, "peak": CASH0, "last_date": ""}
    if st.get("last_date") == day and "base" in st:        # 同一天重跑：從當天開始前的狀態重算
        st = st["base"]
    if st.get("last_date", "") >= day:
        print("今天已經處理過"); return
    st["base"] = json.loads(json.dumps({k: v for k, v in st.items() if k != "base"}))
    cash = float(st["cash"]); pos = st["pos"]; done = []
    # 1) 除權息：持股的買價跟著還原
    for c, k in evf.items():
        if c in pos:
            pos[c]["epx"] *= k
    # 2) 執行昨晚決定的單（今天開盤價）
    pend = st.get("pending") or {}
    if pend:
        sells = list(pos) if pend.get("sellall") else [s["code"] for s in pend.get("sells", [])]
        why = {s["code"]: s["why"] for s in pend.get("sells", [])}
        for c in sells:
            if c not in pos:
                continue
            x = F.open.get(c, np.nan)
            if not (x == x and x > 0):
                x = F.close.get(c, np.nan)
            if not (x == x and x > 0):
                continue          # 停牌：留著明天再賣
            p = pos.pop(c); cash += p["sh"] * x * (1 - COUT)
            done.append(dict(date=day, side="賣", code=c, name=F.name.get(c, ""), kind=p["kind"], shares=p["sh"], px=round(x, 2),
                             ret=round((x * (1 - COUT) / (p["epx"] * (1 + CIN)) - 1) * 100, 1),
                             why=("大盤出場訊號" if pend.get("sellall") else why.get(c, ""))))
        if pend.get("buys"):
            val = cash + sum(p["sh"] * F.prev.get(c, p["epx"]) for c, p in pos.items())
            for b in pend["buys"]:
                c = b["code"]
                if len(pos) >= MAXN or cash - val * (1 - MAXN * SLOT) < val * 0.02:
                    break
                if c in pos:
                    continue
                x = F.open.get(c, np.nan); pc = F.prev.get(c, np.nan)
                if not (x == x and x > 0) or (pc == pc and x >= pc * 1.092):
                    continue        # 沒開盤或開盤漲停買不到
                amt = min(cash - val * (1 - MAXN * SLOT), val * SLOT)
                sh = math.floor(amt / (x * (1 + CIN)))
                if sh <= 0:
                    continue
                pos[c] = dict(sh=sh, kind=b["kind"], ent=day, epx=x); cash -= sh * x * (1 + CIN)
                done.append(dict(date=day, side="買", code=c, name=F.name.get(c, ""), kind=b["kind"], shares=sh, px=round(x, 2), ret="", why=""))
    # 3) 收盤判斷
    cl = F.close
    val = cash + sum(p["sh"] * (cl.get(c) if cl.get(c) == cl.get(c) else p["epx"]) for c, p in pos.items())
    lst = lj.get("state", {}).get("state", "cash"); act = (lj.get("key") or {}).get("action", ""); g = int(lj.get("green", 0))
    room = 0
    sells, buys, sellall, mode = [], [], False, {"in": "多頭", "bottom": "抓底", "wait": "空手", "cash": "空手"}.get(lst, "空手")
    if act == "exit":
        sellall = bool(pos)
    elif act == "bottom":
        buys = [dict(code=c, kind="抄底") for c in candA(F)]; room = MAXN - len(pos)
    else:
        for c, p in pos.items():
            x = cl.get(c)
            if x != x:
                continue
            if p["kind"] == "抄底" and x >= p["epx"] * (1 + TP):
                sells.append(dict(code=c, why="抓底股漲 50% 停利"))
            elif p["kind"] == "抄底" and lst == "in":
                sells.append(dict(code=c, why="三燈兩盞，抓底股換多頭股"))
            elif p["kind"] == "多頭" and x < p["epx"] * (1 + STOP):
                sells.append(dict(code=c, why="收盤虧 10% 停損"))
            elif p["kind"] == "多頭" and x < F.ma60.get(c, 0):
                sells.append(dict(code=c, why="跌破季線"))
        if lst == "in":
            st["peak"] = max(float(st.get("peak", val)), val) if not st.get("braked") else st.get("peak", val)
            room = MAXN - len([c for c in pos if c not in {s["code"] for s in sells}])
            if g < 2:
                room = 0
            if val < float(st["peak"]) * (1 - BRAKE):
                if g < 3:
                    st["braked"] = True
                    for c, p in pos.items():
                        if p["kind"] == "多頭" and c not in {s["code"] for s in sells}:
                            sells.append(dict(code=c, why="帳戶回落 20% 煞車"))
                    room = 0
                else:
                    st["peak"] = val; st["braked"] = False
            elif st.get("braked") and g >= 3:
                st["braked"] = False; st["peak"] = val
            if st.get("braked"):
                room = 0
            if room > 0:
                buys = [dict(code=c, kind="多頭") for c in candM(F) if c not in pos][:room + 10]
        else:
            st["peak"] = val
    st.update(cash=cash, pos=pos, pending=dict(sells=sells, buys=buys, sellall=sellall), last_date=day, value=round(val))
    sf.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    if done:
        tf = OUT / "trades.csv"
        pd.DataFrame(done).to_csv(tf, mode="a", header=not tf.exists(), index=False)

    # ---------- 推播 ----------
    md = f"{int(day[5:7])}/{int(day[8:10])}"
    nbuy = 0 if sellall else min(len(buys), max(room, 0))
    per = val * SLOT
    lines = [f"🛒 {md} 選股", f"燈號：{mode}"]
    if sellall:
        lines += ["⛔ 大盤出場訊號：明天開盤，持股全部賣出"]
    if sells:
        lines += ["", "━━ 明天開盤賣出 ━━"]
        for s in sells:
            p = pos[s["code"]]; x = cl.get(s["code"])
            lines.append(f"⛔ {F.name.get(s['code'], '')} {s['code']}　{s['why']}（買 {p['epx']:.2f} → 收 {x:.2f}，{(x / p['epx'] - 1) * 100:+.0f}%）")
    if buys and nbuy > 0:
        lines += ["", f"━━ 明天開盤買進（系統買前 {nbuy} 檔，每檔約 {per / 1e4:.1f} 萬）━━"]
        for k, b in enumerate(buys[:max(nbuy, 0) + 3]):
            c = b["code"]; x = cl.get(c); sh = math.floor(per / (x * (1 + CIN))) if x == x and x > 0 else 0
            tag = "🟡" if k < nbuy else "⚪ 備選"
            why = (f"離一年高點 {F.toh.get(c) * 100:.0f}%" if b["kind"] == "抄底" else
                   f"創一年新高、60 天 {F.ret60.get(c) * 100:+.0f}%、量 {F.val5.get(c) / F.val20.get(c):.1f} 倍")
            stop = "不設停損，漲 50% 停利" if b["kind"] == "抄底" else f"停損：買價 −10%（約 {x * 0.9:.1f}）或跌破季線 {F.ma60.get(c):.1f}"
            lines.append(f"{tag} {F.name.get(c, '')} {c}　收 {x:.2f}　約 {lot(sh)}")
            lines.append(f"　　{why}｜{stop}")
        lines.append("開盤漲停就買不到，系統會改買下一檔")
    if not sells and not buys and not sellall:
        lines.append("今天沒有要買賣的" + ("（空手，等抓底訊號）" if mode == "空手" else ""))
    if pos:
        lines += ["", f"━━ 模擬帳戶持股 {len(pos)} 檔 ━━"]
        for c, p in sorted(pos.items(), key=lambda z: z[1]["ent"]):
            x = cl.get(c)
            lines.append(f"{F.name.get(c, '')} {c}（{p['kind']}）{p['ent'][5:].replace('-', '/')} 買 {p['epx']:.2f}，現 {x:.2f}，{(x / p['epx'] - 1) * 100:+.0f}%")
    lines += ["", f"模擬帳戶 {val / 1e4:,.1f} 萬（100 萬起，{(val / CASH0 - 1) * 100:+.1f}%）"]
    text = "\n".join(lines)
    has_orders = bool(sells or sellall or (buys and nbuy > 0))
    tsmc = cl.get("2330")
    out = {"date": ymd, "mode": mode, "notify": has_orders, "text": text, "value": round(val),
           "tsmc_close": (round(float(tsmc), 1) if tsmc == tsmc else None),
           # 給 Cloudflare 早報／開盤核對用（相容舊格式）
           "picks": [dict(code=b["code"], name=F.name.get(b["code"], ""), market="", kind=b["kind"],
                          buy_max=round(float(cl.get(b["code"])) * 1.095, 2)) for b in buys[:max(nbuy, 0)]] if nbuy > 0 else [],
           "sells": sells, "sellall": sellall,
           "holdings": [dict(code=c, name=F.name.get(c, ""), kind=p["kind"], ent=p["ent"], epx=round(p["epx"], 2), sh=p["sh"]) for c, p in pos.items()]}
    (OUT / "latest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(text)


if __name__ == "__main__":
    main()
