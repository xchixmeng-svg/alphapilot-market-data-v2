#!/usr/bin/env python3
"""每日燈號：開關＋方向三燈＋底燈（主燈：融資、法人；位置：切割線；輔助：恐慌、個股／匯率／VIX 背離）
＋劇本機率＋v1 抓底系統的部位狀態。輸出 data/extra/lights/{YYYYMMDD}.json 與 latest.json（含推播全文 text）。
Cloudflare alphapilot-brief 晚上讀 latest.json 推到 AlphaPilot 通知。

規則（2026-11-19/20 與 Chi 定案，研究程式在 /home/claude/work：fullall.py、system.py）：
  開關      大盤距一年高點跌超過 10%
  錢燈      台幣 ≤ 60 天最弱點 ×0.985（升值 1.5%）
  人燈      站上 20 日線的股票（20 日均量 ≥3,000 萬）超過 50%
  價燈      大盤收盤 > 60 天收盤高低點的中間
  融資燈    上市融資從一年高點的減幅 > 大盤從一年高點的跌幅，且 00631L+00685L 融資 5 天不增加
  法人燈    近 10 天外資有「大賣」（低於一年 5% 分位），且今天轉買，或今天賣超不到 10 天最大賣超的一半
  恐慌      近 60 天站上 20 日線的比例曾 < 10%
  背離      今天收盤創 60 天新低時：個股（比例比前 60 天最低高 3 點以上）／VIX（低於前 60 天最高）／台幣（低於前 60 天最弱）
  切割線    10%/15%/20% 波段與今年這段的 0.382/0.5/0.618/0.786/0.809/1.0，最低價離線 1.5% 內或盤中穿線收在線上
  抓底訊號  開關 ＋ 主燈一盞以上 ＋ 在切割線附近 → 下個交易日開盤買，收盤跌破訊號日最低 10% 停損，停損後訊號再出現可再試
  出場      三燈全紅且跌超過 10% → 下個交易日開盤出場；兩燈綠 → 回場
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MACRO = DATA / "extra" / "macro"
CREDIT = DATA / "history" / "credit"
OUT = DATA / "extra" / "lights"
sys.path.insert(0, str(ROOT / "scripts"))


# ---------- 資料 ----------
def latest_day():
    ds = sorted(p.name for p in DATA.glob("20??-??-??") if (p / "manifest.json").exists())
    return ds[-1]


def macro(name):
    s = pd.read_csv(MACRO / f"{name}.csv", parse_dates=["date"]).set_index("date")
    return s


TODAY = OUT / "_today"


def fetch_today_prices(day):
    """收盤後直接向證交所／櫃買中心抓當天收盤價（不等晚上的完整流程）。存 data/extra/lights/_today/。"""
    import csv, io, requests
    ymd = day.replace("-", "")
    TODAY.mkdir(parents=True, exist_ok=True)
    f = TODAY / f"{ymd}.csv"
    if f.exists():
        return True
    H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"}
    rows = []
    try:
        j = requests.get("https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
                         params={"date": ymd, "type": "ALLBUT0999", "response": "json"}, headers=H, timeout=(15, 60)).json()
        for t in j.get("tables") or []:
            fl = t.get("fields") or []
            if "證券代號" in fl and "收盤價" in fl and "成交金額" in fl:
                a, b, v = fl.index("證券代號"), fl.index("收盤價"), fl.index("成交金額")
                for r in t.get("data") or []:
                    try:
                        rows.append((day, str(r[a]).strip(), float(str(r[b]).replace(",", "")), float(str(r[v]).replace(",", ""))))
                    except ValueError:
                        pass
        roc = f"{int(ymd[:4]) - 1911:03d}/{ymd[4:6]}/{ymd[6:]}"
        x = requests.get("https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
                         params={"date": roc, "response": "csv"}, headers=H, timeout=(15, 60)).content.decode("cp950")
        lines = x.splitlines()
        if len(lines) > 3 and roc in lines[1]:
            for r in csv.DictReader(io.StringIO("\n".join(lines[2:]))):
                try:
                    code = str(r.get("代號", "")).strip(); cl = float(str(r.get("收盤", "")).replace(",", ""))
                    val = float(str(r.get("成交金額(元)", r.get("成交金額", ""))).replace(",", ""))
                    rows.append((day, code, cl, val))
                except (ValueError, TypeError):
                    pass
    except Exception as e:  # noqa
        print("當天收盤價抓不到：", e)
    if len(rows) < 1500:
        print(f"當天收盤價不足（{len(rows)} 筆），先不用")
        return False
    pd.DataFrame(rows, columns=["trade_date", "stock_id", "close", "trading_value"]).to_csv(f, index=False)
    print(f"當天收盤價 {len(rows)} 筆")
    return True


def breadth(days=90):
    ds = sorted(p.name for p in DATA.glob("20??-??-??") if (p / "normalized" / "twse_ohlcv.csv").exists())[-days:]
    rows = []
    for d in ds:
        for m in ("twse", "tpex"):
            f = DATA / d / "normalized" / f"{m}_ohlcv.csv"
            if f.exists():
                x = pd.read_csv(f, dtype={"stock_id": str}, encoding="utf-8-sig",
                                usecols=["trade_date", "stock_id", "close", "trading_value"])
                rows.append(x)
    for f in sorted(TODAY.glob("*.csv")) if TODAY.exists() else []:
        d = f.stem[:4] + "-" + f.stem[4:6] + "-" + f.stem[6:]
        if d > (ds[-1] if ds else ""):
            rows.append(pd.read_csv(f, dtype={"stock_id": str}))
    z = pd.concat(rows)
    z = z[z.stock_id.str.fullmatch(r"\d{4}")]
    px = z.pivot_table(index="trade_date", columns="stock_id", values="close")
    tv = z.pivot_table(index="trade_date", columns="stock_id", values="trading_value")
    ma = px.rolling(20).mean()
    liq = tv.rolling(20).mean() >= 3e7
    valid = (ma.notna() & liq).sum(axis=1)
    br = ((px > ma) & ma.notna() & liq).sum(axis=1) / valid
    br = br[valid > 100]
    br.index = pd.to_datetime(br.index)
    return br


def credit_update(day_list):
    """把缺的日子補進 credit。法人金額約 15:00 公布、融資約 21:00 公布：融資還沒出來時先存法人，晚上再補融資。
    連不上就略過，用現有資料。回傳最後一天的融資是否已到。"""
    try:
        import backfill_credit as bc
    except Exception as e:  # noqa
        print("credit 模組載入失敗", e)
        return False
    ok_last = False
    for d in day_list:
        y = d[:4]
        fm, fs = CREDIT / f"market_{y}.csv", CREDIT / f"stock_margin_{y}.parquet"
        mk = pd.read_csv(fm, dtype={"date": str}) if fm.exists() else pd.DataFrame(columns=["date", "margin_bal_k"])
        have = mk[mk.date == d]
        if len(have) and have.margin_bal_k.notna().all():
            ok_last = True
            continue
        ok_last = False
        bal_k, short, stocks = float("nan"), float("nan"), []
        try:
            _, bal_k, short, stocks = bc.margin(d)
        except Exception as e:  # noqa
            print(f"融資 {d} 抓不到：{e}")
        try:
            _, f, t, dl = bc.inst(d)
        except Exception as e:  # noqa
            print(f"法人 {d} 抓不到：{e}")
            f = t = dl = float("nan")
        if len(stocks) < 500:
            bal_k, short, stocks = float("nan"), float("nan"), []
        if not (f == f) and not (bal_k == bal_k):
            print(f"credit {d} 還沒公布"); continue
        mk = pd.concat([mk[mk.date != d], pd.DataFrame([{"date": d, "margin_bal_k": bal_k, "short_bal": short,
                        "foreign_net": f, "trust_net": t, "dealer_net": dl}])], ignore_index=True).sort_values("date")
        mk.to_csv(fm, index=False)
        if stocks:
            sk = pd.read_parquet(fs) if fs.exists() else pd.DataFrame()
            if len(sk): sk = sk[sk.date != d]
            sk = pd.concat([sk, pd.DataFrame([{"date": d, "code": c, "margin_bal": a, "short_bal": b} for c, a, b in stocks])],
                           ignore_index=True)
            sk.to_parquet(fs, index=False)
            ok_last = True
        print(f"credit {d}：融資 {'已到' if stocks else '未公布'}、外資 {f/1e8:+,.0f} 億" if f == f else f"credit {d}：只有融資")
    return ok_last


def credit():
    m = pd.concat([pd.read_csv(f, dtype={"date": str}) for f in sorted(CREDIT.glob("market_*.csv"))]).drop_duplicates("date")
    m.index = pd.to_datetime(m.date)
    yrs = sorted(CREDIT.glob("stock_margin_*.parquet"))[-2:]
    s = pd.concat([pd.read_parquet(f) for f in yrs])
    lev = s[s.code.isin(["00631L", "00685L"])].assign(d=lambda q: pd.to_datetime(q.date)) \
        .pivot_table(index="d", columns="code", values="margin_bal")
    return m.sort_index(), lev.sort_index()


# ---------- 切割線 ----------
def zigzag(c, th):
    piv, mode, h, l = [], 0, 0, 0
    for i in range(1, len(c)):
        if mode >= 0 and c[i] > c[h]: h = i
        if mode <= 0 and c[i] < c[l]: l = i
        if mode >= 0 and c[i] <= c[h] * (1 - th): piv.append(("H", h, i)); mode = -1; l = i
        elif mode <= 0 and c[i] >= c[l] * (1 + th): piv.append(("L", l, i)); mode = 1; h = i
    return piv


def fib_levels(close):
    c = close.values
    out = []
    for th in (0.10, 0.15, 0.20):
        piv = zigzag(c, th)
        for k in range(len(piv) - 1, 0, -1):
            if piv[k][0] == "H" and piv[k - 1][0] == "L":
                A, H = c[piv[k - 1][1]], c[piv[k][1]]
                for r in (0.382, 0.5, 0.618, 0.786, 0.809, 1.0):
                    out.append((H - (H - A) * r, f"{int(th*100)}%波段 {r}"))
                break
    yr = close[close.index >= f"{close.index[-1].year}-01-01"]
    if len(yr) > 60:
        lo_i = yr.iloc[: int(yr.values.argmax()) + 1].idxmin()
        A, H = yr[lo_i], yr.max()
        if H / A > 1.15:
            for r in (0.382, 0.5, 0.618, 0.786, 0.809, 1.0):
                out.append((H - (H - A) * r, f"今年這段 {r}"))
    return out


# ---------- 主程式 ----------
def main():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    stage = sys.argv[1] if len(sys.argv) > 1 else ("evening" if now.hour >= 21 or now.hour < 6 else
                                                     "data" if now.hour >= 17 else "afternoon" if now.hour >= 16 else "early")
    day = latest_day()
    tx = macro("TAIEX")
    today = now.strftime("%Y-%m-%d")
    if today > day and pd.Timestamp(today) in tx.index and fetch_today_prices(today):
        day = today
    ymd = day.replace("-", "")
    T = pd.Timestamp(day)
    if T not in tx.index:
        print(f"TAIEX 還沒有 {day} 的資料，停止"); sys.exit(0)
    tw = macro("TWD").close
    vx = macro("VIX").close
    br = breadth()
    recent = [d.strftime("%Y%m%d") for d in tx.index[tx.index <= T][-8:]]
    margin_ok = credit_update(recent)
    m, lev = credit()

    D = tx[["open", "high", "low", "close"]].copy()
    D = D[D.index <= T]
    D["br"] = br.reindex(D.index)
    D["twd"] = tw.reindex(D.index).ffill()
    D["vix"] = vx.reindex(D.index).ffill()
    D = D.join(m[["margin_bal_k", "foreign_net"]]).join(lev)
    c = D.close
    D["dd"] = c / c.rolling(250, min_periods=60).max() - 1
    i = len(D) - 1
    r = D.iloc[i]
    prev60 = D.iloc[i - 60:i]

    sw = r.dd <= -0.10
    L_money = r.twd <= D.twd.iloc[i - 59:i + 1].max() * 0.985
    L_people = (r.br > 0.5) if r.br == r.br else False
    L_price = r.close > (c.iloc[i - 59:i + 1].max() + c.iloc[i - 59:i + 1].min()) / 2
    g = int(L_money) + int(L_people) + int(L_price)

    panic = D.br.iloc[i - 59:i + 1].min() < 0.10
    newlow = r.close < prev60.close.min()
    div_stock = newlow and (r.br > prev60.br.min() + 0.03)
    div_vix = newlow and (r.vix < prev60.vix.max())
    div_twd = newlow and (r.twd < prev60.twd.max())

    mb = D.margin_bal_k.ffill()
    m_dd = mb.iloc[i] / mb.iloc[max(0, i - 249):i + 1].max() - 1
    cols = [k for k in ("00631L", "00685L") if k in D]
    tot = D[cols].ffill().sum(axis=1) if cols else pd.Series(0.0, index=D.index)
    lev5 = tot.iloc[i] / tot.iloc[i - 5] - 1 if tot.iloc[i - 5] > 0 else 0.0
    if abs(lev5) > 0.6: lev5 = 0.0   # 分割等異常跳動
    L_margin = bool((m_dd < r.dd) and (lev5 <= 0))
    fn = D.foreign_net / 1e8
    heavy = fn.iloc[i - 9:i + 1].min() < fn.iloc[max(0, i - 249):i + 1].quantile(0.05)
    fmin10 = fn.iloc[i - 9:i + 1].min()
    L_inst = bool(heavy and ((fn.iloc[i] > 0 and fn.iloc[i - 1] < 0) or (fn.iloc[i] < 0 and fn.iloc[i] > fmin10 * 0.5)))

    lv_all = fib_levels(c)
    near = [(v, n) for v, n in lv_all if abs(r.low / v - 1) <= 0.015 or (r.low <= v <= r.close)]
    above = sorted([(v, n) for v, n in lv_all if v > r.close], key=lambda x: x[0])[:2]
    below = sorted([(v, n) for v, n in lv_all if v < r.close], key=lambda x: -x[0])[:3]
    signal = bool(sw and (L_margin or L_inst) and near)

    # ---- v1 系統部位（狀態機，每天只處理一次）----
    OUT.mkdir(parents=True, exist_ok=True)
    sf = OUT / "state.json"
    st = json.loads(sf.read_text()) if sf.exists() else {"state": "bottom", "stop": 37737, "entry_date": "2026-11-11",
                                                          "entry_px": 43135, "last_date": "2026-11-19"}
    action = ""
    if st.get("last_date") == day and "base" in st:      # 同一天第二次跑（例如融資資料晚到）：從當天開始前的狀態重算
        st = st["base"]
    if st.get("last_date", "") < day:
        st["base"] = {k: v for k, v in st.items() if k != "base"}
        exitF = (g == 0) and r.dd <= -0.10
        s = st["state"]
        if s == "in" and exitF:
            st["state"] = "cash"; action = "紅燈出場：下個交易日開盤賣出"
        elif s == "cash":
            if g >= 2:
                st["state"] = "in"; action = "兩燈轉綠：下個交易日開盤回場"
            elif signal:
                st.update(state="bottom", stop=round(r.low * 0.9), entry_date="下個交易日開盤", entry_px=None)
                action = f"抓底訊號：下個交易日開盤買進，停損 {r.low*0.9:,.0f}"
        elif s == "bottom":
            if g >= 2:
                st["state"] = "in"; action = "兩燈轉綠：抓底單轉為正常持有"
            elif r.close < st["stop"]:
                st["state"] = "cash"; action = f"抓底停損：收盤 {r.close:,.0f} 跌破 {st['stop']:,}，下個交易日開盤賣出"
        st["last_date"] = day
        sf.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    pos_txt = {"in": "持有中", "cash": "空手", "bottom": f"抓底單持有中（停損 {st.get('stop', 0):,}）"}[st["state"]]

    # ---- 劇本機率（依歷史次數的粗估，每天隨燈號調整，非保證）----
    p1, p2, p3 = 55, 35, 10
    if div_stock: p1 += 10; p2 -= 10
    if div_twd: p1 += 5; p2 -= 5
    if L_margin or L_inst: p1 += 10; p2 -= 5; p3 -= 5
    if newlow and not div_stock: p1 -= 5; p2 += 5
    hi_days = len(c.iloc[max(0, i - 249):i + 1]) - 1 - int(np.argmax(c.iloc[max(0, i - 249):i + 1].values))
    if hi_days > 60: p1 -= 10; p3 += 10
    if r.close < 33260: p1, p2, p3 = 10, 40, 50
    ps = np.clip(np.array([p1, p2, p3], dtype=float), 2, None); ps = np.round(ps / ps.sum() * 100).astype(int)
    names = ["① 急跌見底（35,500～36,000 一帶止跌）", "② 再殺一段到 33,260～33,731 共振區", "③ 轉成熊市（跌破 31,723）"]
    best = names[int(ps.argmax())].split("（")[0]

    on = lambda b: "🟢" if b else "🔴"
    dv = lambda b: ("🟢" if b else "🔴") if newlow else "⚪"
    lines = [
        f"📊 {day[5:].replace('-', '/')} 燈號　大盤 {r.close:,.0f}（{(r.close / c.iloc[i-1] - 1)*100:+.1f}%）距高點 {r.dd*100:.1f}%",
        "",
        f"開關：{'🟠 下跌中（找底模式）' if sw else '🟢 正常'}",
        f"方向三燈：錢 {on(L_money)}　人 {on(L_people)}　價 {on(L_price)}（{g} 綠）",
        "",
        "底燈",
        f"主燈：融資 {on(L_margin)}（融資減幅 {m_dd*100:.0f}% vs 大盤 {r.dd*100:.0f}%，正2 融資 5 日 {lev5*100:+.0f}%）",
        (f"　　　法人 {on(L_inst)}（外資 {fn.iloc[i]:+,.0f} 億，10 天最大賣超 {fmin10:,.0f} 億）" if fn.iloc[i] == fn.iloc[i] else "　　　法人 ⚪（今天外資數字約 15:00 公布）"),
        "位置：" + ("🟢 在切割線 " + "、".join(f"{v:,.0f}" for v, _ in near) if near else
                   "⚪ 不在切割線（上 " + "／".join(f"{v:,.0f}" for v, _ in above) + "　下 " + "／".join(f"{v:,.0f}" for v, _ in below) + "）"),
        f"輔助：恐慌 {on(panic)}　個股背離 {dv(div_stock)}　匯率背離 {dv(div_twd)}　VIX 背離 {dv(div_vix)}",
        f"　（站上 20 日線 {r.br*100:.0f}%｜台幣 {r.twd:.3f}｜VIX {r.vix:.1f}）" + ("" if newlow else "　今天沒創新低，背離不適用"),
        "",
        "燈號組合解讀",
    ]
    if newlow and not div_stock: lines.append("• 創新低且多數個股跟跌：全面下跌，斷頭潮可能還在")
    if newlow and div_stock: lines.append("• 指數破底但多數個股沒破：只剩權值股在殺，接近底部（歷史約 2/3 是真底）")
    if div_twd: lines.append("• 匯率背離：大盤破底但台幣沒創新弱，外資匯出放慢")
    if div_stock and div_twd: lines.append("• 個股＋匯率雙背離：歷史 7 次中 6 次是真底")
    if not L_margin: lines.append("• 融資燈沒亮：" + ("正2 槓桿還在增加" if lev5 > 0 else "融資減幅還不夠"))
    if L_inst: lines.append("• 法人燈亮：外資賣壓衰竭或轉買")
    if signal: lines.append("• ✅ 抓底訊號成立：主燈＋切割線")
    elif sw: lines.append("• 抓底訊號未出現（需要主燈＋切割線）")
    if not margin_ok:
        lines.append("• 今天的融資資料約 21:00 公布，融資燈先用昨天的數字，22:00 會再推一次完整版")
    lines += ["", "劇本機率（依歷史次數粗估，非保證）"] + [f"{n}：約 {p}%" for n, p in zip(names, ps)]
    lines += [f"最有機會：{best}", "", f"系統部位：{pos_txt}" + (f"\n⚠️ {action}" if action else "")]
    text = "\n".join(lines)

    prev = {}
    if (OUT / "latest.json").exists():
        try:
            prev = json.loads((OUT / "latest.json").read_text())
        except Exception:  # noqa
            prev = {}
    key = {"money": bool(L_money), "people": bool(L_people), "price": bool(L_price), "margin": L_margin, "inst": L_inst,
           "div_stock": bool(div_stock), "div_twd": bool(div_twd), "signal": signal, "action": action}
    changed = prev.get("date") != ymd or prev.get("key") != key
    notify = stage in ("early", "afternoon") or changed
    stamp = {"early": "14:40 初版", "afternoon": "16:10 加外資／台幣", "data": "晚間完整股價", "evening": "21:40 加融資"}.get(stage, stage)
    text = text.replace("\n\n開關", f"（{stamp}）\n\n開關", 1)
    res = {"date": ymd, "stage": stage, "notify": bool(notify), "key": key, "text": text, "margin_ok": bool(margin_ok), "signal": signal, "switch": bool(sw), "green": g,
           "lights": {"money": bool(L_money), "people": bool(L_people), "price": bool(L_price), "margin": L_margin,
                      "inst": L_inst, "panic": bool(panic), "div_stock": bool(div_stock), "div_twd": bool(div_twd),
                      "div_vix": bool(div_vix), "near_fib": bool(near)},
           "scenario": dict(zip(["fast", "deeper", "bear"], map(int, ps))), "state": st}
    (OUT / f"{ymd}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (OUT / "latest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(text)


if __name__ == "__main__":
    main()
