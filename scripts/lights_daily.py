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
  ── 2026-11-20 整合版（回測 2008→今 100 萬→約 930 萬（一直抱著 473 萬），研究程式 /home/claude/work：replay.py、exit3.py）──
  事件類型  從一年高點跌到 −12% 只花 35 個交易日內、且那個高點是一年新高 → 急跌型；否則慢跌型
  抓底訊號  急跌型：開關 ＋ 主燈一盞以上 ＋ 切割線附近 ＋ 恐慌過（20 天內比例 <10%）＋ 台幣守住（低於前 20 天最弱）＋ VIX≥25
            慢跌型：上面的條件再加「融資比一年高點少 25%」；或順序規則先成立（台幣守住→恐慌→大盤創 20 天新低但個股沒有→
                    15 天內站上 20 日線比例回到 30% 且外資 3 天合計買超），一樣要 VIX≥25
            VIX 用台股收盤時已知的美股前一晚收盤
            → 下個交易日開盤買，**不停損**，長期持有；三燈兩綠後轉為正常持有
  出場      三燈全紅 ＋ 跌超過 7% ＋ 外資近 10 天合計賣超在近兩年最兇的 10% 內 → 下個交易日開盤出場
  回場      三燈兩綠（判定假警報）→ 下個交易日開盤回場（+2% 快速回場在補齊 2020、2026 資料後沒有幫助，已拿掉）
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


def taifex_foreign():
    """外資台指期（TXF）多空未平倉口數淨額，讀 data/history/taifex/raw/fut_*.csv 最近幾個月。"""
    import io
    rows = []
    for f in sorted((DATA / "history" / "taifex" / "raw").glob("fut_*.csv"))[-3:]:
        t = f.read_bytes().decode("big5", "replace")
        if "DOCTYPE" in t[:80]:
            continue
        x = pd.read_csv(io.StringIO(t), index_col=False)
        x = x[x["身份別"].astype(str).str.contains("外資")]
        rows.append(pd.Series(x["多空未平倉口數淨額"].astype(float).values, index=pd.to_datetime(x["日期"])))
    return pd.concat(rows).sort_index() if rows else pd.Series(dtype=float)


def maint_ratio(m, days=30):
    """大盤融資維持率（估算）＝ Σ 個股融資張數×1000×收盤價 ÷ 上市融資金額。回傳最近 days 天。"""
    yrs = sorted(CREDIT.glob("stock_margin_*.parquet"))[-1:]
    sm = pd.concat([pd.read_parquet(f) for f in yrs])
    sm = sm[sm.margin_bal > 0]
    sm["d"] = pd.to_datetime(sm.date.astype(str))
    keep = sorted(sm.d.unique())[-days:]
    sm = sm[sm.d.isin(keep)]
    rows = []
    for d in keep:
        ds = pd.Timestamp(d).strftime("%Y-%m-%d")
        f = DATA / ds / "normalized" / "twse_ohlcv.csv"
        if f.exists():
            x = pd.read_csv(f, dtype={"stock_id": str}, encoding="utf-8-sig", usecols=["stock_id", "close"])
        elif (TODAY / f"{ds.replace('-', '')}.csv").exists():
            x = pd.read_csv(TODAY / f"{ds.replace('-', '')}.csv", dtype={"stock_id": str}, usecols=["stock_id", "close"])
        else:
            continue
        j = sm[sm.d == d].merge(x, left_on="code", right_on="stock_id")
        if len(j) < 500 or pd.Timestamp(d) not in m.index:
            continue
        rows.append((pd.Timestamp(d), (j.margin_bal * 1000 * j.close).sum() / (m.loc[pd.Timestamp(d), "margin_bal_k"] * 1000) * 100))
    return pd.Series(dict(rows)).sort_index()


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
    # 過了午夜或完整資料還沒進來：用已抓到的當天收盤價（_today）中最新、且 TAIEX 有資料的那天
    if TODAY.exists():
        for f in sorted(TODAY.glob("*.csv")):
            d2 = f"{f.stem[:4]}-{f.stem[4:6]}-{f.stem[6:]}"
            if d2 > day and pd.Timestamp(d2) in tx.index:
                day = d2
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
    try:
        fut = taifex_foreign()
    except Exception as e:  # noqa
        print("外資台指期讀不到：", e); fut = pd.Series(dtype=float)
    try:
        mrs = maint_ratio(m, days=150)
    except Exception as e:  # noqa
        print("維持率算不出來：", e); mrs = pd.Series(dtype=float)

    D = tx[["open", "high", "low", "close"]].copy()
    D = D[D.index <= T]
    D["br"] = br.reindex(D.index)
    D["twd"] = tw.reindex(D.index).ffill()
    D["vix"] = vx.reindex(D.index).ffill()
    vxu = vx.copy(); vxu.index = vxu.index + pd.Timedelta(days=1)       # 台股收盤時已知：美股前一晚
    D["vix_us"] = vxu.reindex(D.index.union(vxu.index)).ffill().reindex(D.index)
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
    # ---- 整合版抓底條件 ----
    cv = c.values
    a0 = max(0, i - 249); h = a0 + int(np.argmax(cv[a0:i + 1]))
    prevh = cv[max(0, h - 250):h]
    newhigh = len(prevh) == 0 or cv[h] >= prevh.max()
    s12 = next((k for k in range(h, i + 1) if cv[k] <= cv[h] * 0.88), None)
    ev_type = "未定" if s12 is None else ("急跌型" if (s12 - h <= 35 and newhigh) else "慢跌型")
    panic20 = bool(D.br.iloc[i - 19:i + 1].min() < 0.10)
    twd_ok = bool(r.twd < D.twd.iloc[i - 20:i].max())
    vix_ok = bool(D.vix_us.iloc[i] >= 25)
    margin25 = bool(m_dd <= -0.25)
    # 順序規則（慢跌型的另一條路）：近 15 天內曾「台幣守住＋恐慌＋大盤創 20 天新低但個股沒有」，今天人回來＋外資 3 天轉買
    brs = D.br; tws = D.twd; ddv = D.dd
    seq = False
    for k in range(max(21, i - 15), i):
        if ddv.iloc[k] > -0.10: continue
        nl = cv[k] < cv[k - 20:k].min()
        cd = nl and brs.iloc[k] > brs.iloc[k - 20:k].min() + 0.03
        if cd and tws.iloc[k] < tws.iloc[k - 20:k].max() and brs.iloc[k - 19:k + 1].min() < 0.10:
            seq = True
    seq = bool(seq and r.dd <= -0.10 and (r.br >= 0.30 if r.br == r.br else False) and fn.iloc[i - 2:i + 1].sum() > 0)
    base_sig = bool(sw and (L_margin or L_inst) and near and panic20 and twd_ok)
    if ev_type == "慢跌型":
        signal = bool(vix_ok and ((base_sig and margin25) or seq))
    else:
        signal = bool(vix_ok and base_sig)
    # ---- 預估真底：第一次跌破 20% 那天的年線（240 日均）× 0.764（往下 23.6%，黃金比例）----
    # 歷史 2011/2015/2020/2022/2025：4 次離真底 3% 內，平均差 1.8%；熊市（2008）不適用
    bottom_est = None
    s20 = next((k for k in range(h, i + 1) if cv[k] <= cv[h] * 0.80), None)
    ma240 = c.rolling(240).mean()
    if s20 is not None and ma240.iloc[s20] == ma240.iloc[s20]:
        e = float(ma240.iloc[s20]) * 0.764
        bottom_est = {"date": D.index[s20].strftime("%Y%m%d"), "est": round(e), "lo": round(e * 0.97), "hi": round(e * 1.03),
                      "ma240": round(float(ma240.iloc[s20]))}
    f10 = fn.rolling(10).sum()
    f10q = f10.iloc[max(0, i - 499):i + 1].quantile(0.10)
    foreign_heavy = bool(f10.iloc[i] <= f10q)

    # ---- 系統部位（整合版狀態機，每天只處理一次）----
    OUT.mkdir(parents=True, exist_ok=True)
    sf = OUT / "state.json"
    st = json.loads(sf.read_text()) if sf.exists() else {}
    if st.get("rule") != "v2":      # 2026-11-20 換成整合版：照新規則重放，10/30 出場、11/10 因 VIX 20 不買 → 空手
        st = {"rule": "v2", "state": "cash", "exit_date": "2026-10-30", "exit_px": 45212, "last_date": "2026-11-19"}
    action = ""
    if st.get("last_date") == day and "base" in st:      # 同一天第二次跑（例如融資資料晚到）：從當天開始前的狀態重算
        st = st["base"]
    if st.get("last_date", "") < day:
        st["base"] = {k: v for k, v in st.items() if k != "base"}
        exitE = (g == 0) and r.dd <= -0.07 and foreign_heavy
        s_ = st["state"]
        if s_ == "in" and exitE:
            st.update(state="cash", exit_date=day, exit_px=round(float(r.close)))
            action = "出場訊號（三燈全紅＋跌超過 7%＋外資大賣）：下個交易日開盤出場"
        elif s_ == "cash":
            if signal:
                st.update(state="bottom", entry_date="下個交易日開盤", signal_date=day)
                action = "抓底訊號：下個交易日開盤買進，不停損，長期持有"
            elif g >= 2:
                st["state"] = "in"; action = "三燈兩綠：下個交易日開盤回場"
        elif s_ == "bottom":
            if g >= 2:
                st["state"] = "in"; action = "三燈兩綠：抓底單轉為正常持有"
        st["last_date"] = day
        sf.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    pos_txt = {"in": "持有中", "cash": "空手（等抓底訊號）", "bottom": "抓底單持有中（不停損）"}[st["state"]]

    on = lambda b: "🟡" if b else "⚪"
    hi_lv = lambda xs: "／".join(f"{v:,.0f}" for v, _ in xs[:1])
    md = f"{day[5:7]}/{day[8:10]}"
    # 狀態行
    if signal:
        status = "🎯 抓底訊號成立　明天開盤進場，不停損"
    elif action.startswith("出場"):
        status = "⛔ 出場訊號　明天開盤出場"
    elif sw:
        status = f"⛔ 下跌中　{ev_type}　系統{pos_txt.replace('（等抓底訊號）', '')}"
    else:
        status = f"正常　系統{pos_txt.replace('（等抓底訊號）', '')}"
    lines = [f"📊 {md} 探底燈號",
             f"大盤 {r.close:,.0f}　{(r.close / c.iloc[i-1] - 1)*100:+.1f}%　距高點 {r.dd*100:.1f}%",
             status]
    if action and not signal and not action.startswith("出場"):
        lines.append(f"⚠️ {action}")
    br20 = D.br.iloc[i - 19:i + 1].min()
    tw20 = D.twd.iloc[i - 20:i].max()
    # 各燈原因
    if L_margin:
        why_m = f"融資少 {-m_dd*100:.0f}%（大盤 {r.dd*100:.0f}%），正2融資 5天沒增加"
    elif m_dd > r.dd:
        why_m = f"融資少 {-m_dd*100:.0f}%，還沒比大盤（{r.dd*100:.0f}%）多"
    else:
        why_m = f"融資少 {-m_dd*100:.0f}% 已夠，但正2融資 5天 {lev5*100:+.0f}%，散戶還在加碼"
    if fn.iloc[i] != fn.iloc[i]:
        why_i = "今天外資數字約 15:00 公布"
    elif L_inst:
        why_i = f"外資賣壓衰竭或轉買（今天 {fn.iloc[i]:+,.0f}億，10天最大賣超 {fmin10:,.0f}億）"
    else:
        why_i = f"外資{'還在賣' if fn.iloc[i] < 0 else '小買'} {fn.iloc[i]:+,.0f}億，賣壓還沒衰竭（10天最大賣超 {fmin10:,.0f}億）"
    if sw:
        lines += ["", "抓底條件",
                  f"{on(twd_ok)} 台幣守住　{r.twd:.3f}，" + ("沒有再創新弱" if twd_ok else f"還在創 20 天新弱（前高 {tw20:.3f}）"),
                  f"{on(panic20)} 恐慌過　站上月線 {r.br*100:.0f}%" + ("" if panic20 else f"，20 天內最低 {br20*100:.0f}%，還沒到 10%"),
                  f"{on(vix_ok)} 美股恐慌　VIX {D.vix_us.iloc[i]:.1f}，" + ("超過 25" if vix_ok else "還沒到 25"),
                  f"{on(L_margin)} 融資燈　{why_m}",
                  f"{on(L_inst)} 法人燈　{why_i}",
                  f"{on(bool(near))} 切割線　" + (f"在切割線 {hi_lv(near)}" if near else
                                                  f"上 {hi_lv(above)}　下 {hi_lv(below)}")]
        if ev_type == "慢跌型":
            lines += [f"{on(margin25)} 慢跌型：融資少 25%　目前少 {-m_dd*100:.0f}%",
                      f"{on(seq)} 慢跌型：順序規則（人回來＋外資 3 天轉買）"]
        miss = [n for n, ok in [("台幣守住", twd_ok), ("恐慌過", panic20), ("美股恐慌", vix_ok),
                                ("融資或法人燈", L_margin or L_inst), ("切割線", bool(near))] if not ok]
        if ev_type == "慢跌型" and not (margin25 or seq): miss.append("融資少 25% 或順序規則")
        if not signal and miss:
            lines += ["", "還差：" + "、".join(miss)]
    if st["state"] == "in":
        lines += ["", f"出場條件：三燈 錢 {on(not L_money)} 人 {on(not L_people)} 價 {on(not L_price)} 全紅　"
                      f"{on(r.dd <= -0.07)} 跌超過 7%　{on(foreign_heavy)} 外資大賣"]
    if bottom_est:
        lines += ["", f"預估真底　{bottom_est['est']:,}（{bottom_est['lo']:,}～{bottom_est['hi']:,}）",
                  f"　{bottom_est['date'][4:6]}/{bottom_est['date'][6:]} 跌破 20% 那天的年線 {bottom_est['ma240']:,} × 0.764"]
    obs = []
    if len(mrs) >= 6:
        obs.append(f"融資維持率　{mrs.iloc[-1]:.0f}%（5天前 {mrs.iloc[-6]:.0f}%）真底多在 120～131")
    if len(fut) >= 6:
        obs.append(f"外資期貨{'空' if fut.iloc[-1] < 0 else '多'}單　{abs(fut.iloc[-1]):,.0f}口（5天前 {abs(fut.iloc[-6]):,.0f}）見底時通常在回補")
    if obs:
        lines += ["", "觀察"] + obs
    if not margin_ok:
        lines += ["", "（今天融資資料還沒公布，融資燈先用昨天的數字）"]
    text = "\n".join(lines)

    prev = {}
    if (OUT / "latest.json").exists():
        try:
            prev = json.loads((OUT / "latest.json").read_text())
        except Exception:  # noqa
            prev = {}
    key = {"money": bool(L_money), "people": bool(L_people), "price": bool(L_price), "margin": L_margin, "inst": L_inst,
           "div_stock": bool(div_stock), "div_twd": bool(div_twd), "signal": signal, "action": action,
           "panic20": panic20, "twd_ok": twd_ok, "vix_ok": vix_ok}
    changed = prev.get("date") != ymd or prev.get("key") != key
    notify = stage in ("early", "afternoon") or changed
    res = {"date": ymd, "stage": stage, "notify": bool(notify), "key": key, "text": text, "margin_ok": bool(margin_ok), "signal": signal, "switch": bool(sw), "green": g,
           "lights": {"money": bool(L_money), "people": bool(L_people), "price": bool(L_price), "margin": L_margin,
                      "inst": L_inst, "panic": bool(panic), "div_stock": bool(div_stock), "div_twd": bool(div_twd),
                      "div_vix": bool(div_vix), "near_fib": bool(near), "panic20": panic20, "twd_ok": twd_ok,
                      "vix_ok": vix_ok, "margin25": margin25, "seq": seq, "foreign_heavy": foreign_heavy},
           "event_type": ev_type, "vix_us": round(float(D.vix_us.iloc[i]), 1),
           "maint_ratio": (round(float(mrs.iloc[-1]), 1) if len(mrs) else None),
           "bottom_est": bottom_est,
           "foreign_fut_net": (int(fut.iloc[-1]) if len(fut) else None),
           "maint_ratio_date": (mrs.index[-1].strftime("%Y%m%d") if len(mrs) else None),
           "state": st}
    (OUT / f"{ymd}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (OUT / "latest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(text)


if __name__ == "__main__":
    main()
