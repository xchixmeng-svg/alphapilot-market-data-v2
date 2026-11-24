#!/usr/bin/env python3
"""App（alphapilot-app.pages.dev）每天的資料：雲端產生，寫到 site_out/，由 workflow 推到 site-data 分支。

  v2.json              燈號＋今天的 V2 買賣單＋模擬帳戶＋正常範圍＋策略對照
  market/index.json    加權指數、漲跌家數、近 60 日走勢
  market/flows.json    三大法人（今日／5 日／20 日／連續）＋各產業資金、占比、階段（原 App 的資金動態，規則照 ap/flows.py、ap/opportunity.py）
  search/index.json    全台股代號、名稱、產業
  stocks/<代號>.json   個股 K 線（近 80 日）、法人（股數）、月營收年增、V2 狀態
法人金額＝個股法人買賣超股數 × 收盤價加總（估算，億元），和原 App 一樣。
"""
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "site_out"
STAGE_FILE = DATA / "extra" / "picks" / "industry_stages.json"
KINDS = (("foreign_net", "外資", "foreign"), ("trust_net", "投信", "trust"), ("dealer_net", "自營商", "dealer"))


def w(p, obj):
    p = OUT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=lambda x: None if x != x else float(x)))


def streak(vals):
    s = [x for x in vals if x == x and x != 0]
    if not s:
        return 0
    sign = 1 if s[-1] > 0 else -1
    n = 0
    for x in reversed(s):
        if (x > 0) == (sign > 0):
            n += 1
        else:
            break
    return n * sign


def fnum(x, d=4):
    return None if x is None or x != x else round(float(x), d)


# ---------- 資料 ----------
def daily_dirs():
    return sorted(p.parent.parent for p in DATA.glob("20??-??-??/normalized/twse_ohlcv.csv"))


def load_ohlcv():
    rows, names = [], {}
    for f in sorted(glob.glob(str(DATA / "history" / "prices_2026" / "*.csv"))):
        x = pd.read_csv(f, dtype={"stock_id": str}).rename(columns={"trade_date": "date", "stock_id": "code", "trading_value": "value"})
        for c in ("open", "high", "low"):
            x[c] = x["close"]
        rows.append(x[["date", "code", "open", "high", "low", "close", "value"]])
    for d in daily_dirs():
        for m in ("twse", "tpex"):
            f = d / "normalized" / f"{m}_ohlcv.csv"
            if f.exists():
                x = pd.read_csv(f, dtype={"stock_id": str}, encoding="utf-8-sig").rename(
                    columns={"trade_date": "date", "stock_id": "code", "trading_value": "value"})
                rows.append(x[["date", "code", "open", "high", "low", "close", "value"]])
                names.update({c: str(n).strip() for c, n in zip(x.code, x.name)})
    a = pd.concat(rows)
    a["date"] = pd.to_datetime(a.date)
    a = a[a.code.astype(str).str.fullmatch(r"\d{4}") & (a.close > 0)]
    return a.drop_duplicates(["date", "code"], keep="last").sort_values(["code", "date"]).reset_index(drop=True), names


def load_inst(days):
    rows = []
    for d in days:
        for m in ("twse", "tpex"):
            f = DATA / d.strftime("%Y-%m-%d") / "normalized" / f"{m}_institutional.csv"
            if f.exists():
                x = pd.read_csv(f, dtype={"stock_id": str}, encoding="utf-8-sig", usecols=["trade_date", "market", "stock_id", "foreign_net", "trust_net", "dealer_net"])
                rows.append(x)
    if not rows:
        return None
    a = pd.concat(rows).rename(columns={"trade_date": "date", "stock_id": "code"})
    a["date"] = pd.to_datetime(a.date)
    return a[a.code.str.fullmatch(r"\d{4}")]


# ---------- 大盤 ----------
def market_index(a, day):
    tx = pd.read_csv(DATA / "extra" / "macro" / "TAIEX.csv", parse_dates=["date"]).set_index("date")
    tx = tx[~(tx[["open", "high", "low", "close"]] == tx[["open", "high", "low", "close"]].shift(1)).all(axis=1)]
    c = tx.close[tx.index <= day]
    t = a[a.date == day]
    p = a[a.date == a.date[a.date < day].max()].set_index("code").close
    t = t.set_index("code")
    ch = t.close / p.reindex(t.index) - 1
    obj = {"status": "OK" if c.index[-1] == day else "STALE", "date": day.strftime("%Y%m%d"),
           "taiex": {"close": fnum(c.iloc[-1], 2), "change": fnum(c.iloc[-1] - c.iloc[-2], 2), "change_pct": fnum(c.iloc[-1] / c.iloc[-2] - 1)},
           "turnover": fnum(t.value.sum(), 0),
           "breadth": {"up": int((ch > 0).sum()), "down": int((ch < 0).sum()), "flat": int((ch == 0).sum()),
                       "up_limit": int((ch >= 0.095).sum()), "down_limit": int((ch <= -0.095).sum())},
           "series": {"close": [fnum(x, 2) for x in c.iloc[-60:]]}}
    w("market/index.json", obj)
    return obj


# ---------- 資金動態（規則照 ap/flows.py＋ap/opportunity.py） ----------
def flows(a, day, imap, window=20):
    days = sorted(a.date.unique()); days = [d for d in days if d <= day][-(window + 1):]
    ins = load_inst(days[-window:])
    if ins is None or ins.date.max() != day:
        return {"date": day.strftime("%Y%m%d"), "inst_ok": False, "notes": ["今天的法人資料還沒進來"], "market": {}, "industries": []}
    px = a[a.date.isin(days)]
    today = px[px.date == day].set_index("code")
    uni = list(today.index[(today.value > 0)])
    close = px.pivot(index="date", columns="code", values="close").reindex(columns=uni)
    amt = px.pivot(index="date", columns="code", values="value").reindex(columns=uni)
    inst_days = sorted(ins.date.unique())[-window:]
    out = {"date": day.strftime("%Y%m%d"), "inst_ok": True, "notes": [], "unit": "億元（個股法人買賣超股數 × 收盤價加總，估算）", "market": {}, "industries": []}
    val = {}
    for col, lab, key in KINDS:
        p = ins.pivot_table(index="date", columns="code", values=col, aggfunc="sum").reindex(index=inst_days, columns=uni)
        val[key] = p * close.reindex(inst_days) / 1e8
    tot = sum(v.fillna(0) for v in val.values())
    allk = dict(val, all3=tot)
    for key, v in allk.items():
        daily = v.sum(axis=1, min_count=1)
        out["market"][key] = {"today": fnum(daily.iloc[-1], 2), "d5": fnum(daily.iloc[-5:].sum(), 2), "d20": fnum(daily.sum(), 2),
                              "streak": streak(daily.tolist()), "days": len(daily)}
    ind = imap.reindex(uni)
    share_t = amt.iloc[-1] / amt.iloc[-1].sum()
    share_avg = (amt.iloc[:-1].div(amt.iloc[:-1].sum(axis=1), axis=0)).mean()
    # 價格廣度：產業中位數 5／20 日報酬、站上月線比例
    piv = a[a.date <= day].pivot(index="date", columns="code", values="close").iloc[-21:].reindex(columns=uni)
    r5 = piv.iloc[-1] / piv.iloc[-6] - 1; r20 = piv.iloc[-1] / piv.iloc[0] - 1; ma20 = piv.iloc[-20:].mean()
    rows = []
    for name, codes in ind.dropna().groupby(ind.dropna()).groups.items():
        codes = list(codes)
        f = {k: v[codes].sum(axis=1) for k, v in allk.items()}
        rec = {"industry": name, "n": len(codes)}
        for k, s in f.items():
            rec[k] = {"today": fnum(s.iloc[-1], 2), "d5": fnum(s.iloc[-5:].sum(), 2), "d20": fnum(s.sum(), 2), "streak": streak(s.tolist())}
        f5, f20 = rec["all3"]["d5"], rec["all3"]["d20"]
        prev = f20 - f5
        rec["turn"] = "轉買" if prev < 0 < f5 else ("轉賣" if prev > 0 > f5 else "")
        rec["share"] = fnum(share_t[codes].sum()); rec["share_chg"] = fnum(share_t[codes].sum() - share_avg[codes].sum())
        rr5, rr20 = r5[codes].dropna(), r20[codes].dropna()
        ab = (piv.iloc[-1][codes] > ma20[codes])[ma20[codes].notna()]
        rec["med_r5"] = fnum(rr5.median()) if len(rr5) else None
        rec["med_r20"] = fnum(rr20.median()) if len(rr20) else None
        rec["breadth"] = fnum(ab.mean()) if len(ab) else None
        # 各法人一句話＋5 日主要推動者
        ph = {}
        for k, lab in (("foreign", "外資"), ("trust", "投信"), ("dealer", "自營商"), ("all3", "合計")):
            x = rec[k]; s_ = x["streak"]
            ph[k] = f"{lab} 5 日{'買' if (x['d5'] or 0) >= 0 else '賣'}超 {abs(x['d5'] or 0):.1f} 億" + (f"、連{'買' if s_ > 0 else '賣'} {abs(s_)} 日" if abs(s_) >= 2 else "")
        rec["phrases"] = ph
        rec["driver_5d"] = max((("外資", rec["foreign"]["d5"] or 0), ("投信", rec["trust"]["d5"] or 0), ("自營商", rec["dealer"]["d5"] or 0)), key=lambda z: abs(z[1]))[0]
        rows.append(rec)
    # 產業階段（照 ap/opportunity.industry_stages）
    prevs = json.loads(STAGE_FILE.read_text()) if STAGE_FILE.exists() else {}
    prev_map = prevs.get("stages", {}) if prevs.get("date", "") < out["date"] else prevs.get("prev", {})
    mr = np.array([r["med_r20"] if r["med_r20"] is not None else np.nan for r in rows], dtype=float)
    q = np.full(len(rows), np.nan); ok = ~np.isnan(mr)
    if ok.sum():
        q[ok] = (mr[ok].argsort().argsort() + 1) / ok.sum()
    for r, qq in zip(rows, q):
        f = r["all3"]; d5, d20, today_ = f["d5"], f["d20"], f["today"]; sc, st_, turn = r["share_chg"], f["streak"], r["turn"]
        prev = prev_map.get(r["industry"], "DORMANT"); br, m20 = r["breadth"], r["med_r20"]
        why = [f"三大法人合計（估算）5 日 {d5:+.1f} 億、20 日 {d20:+.1f} 億" + (f"（{turn}）" if turn else "")]
        if sc is not None:
            why.append(f"成交占比 {r['share'] * 100:.1f}%（較 20 日平均 {sc * 100:+.1f} 個百分點）")
        if m20 is not None:
            why.append(f"產業中位數 20 日 {m20 * 100:+.1f}%" + (f"、站上月線 {br * 100:.0f}%" if br is not None else ""))
        if r["n"] < 3:
            state, reason = "DORMANT", "樣本不足"
        else:
            hot = qq == qq and qq >= 0.9 and (sc or 0) > 0.02 and (br or 0) > 0.8
            confirm = d5 > 0 and d20 > 0 and (m20 or 0) > 0 and (br or 0) >= 0.5
            emerging = (turn == "轉買") or (d5 > 0 >= d20) or (d5 > 0 and (sc or 0) > 0.005 and (m20 or 0) <= 0.02)
            weakening = (turn == "轉賣") or (d5 < 0 and today_ is not None and today_ < 0)
            if prev in ("CONFIRMING", "ADVANCING", "OVERHEATED") and weakening:
                state, reason = "FADING", "原本走強，法人轉賣／5 日轉負"
            elif prev in ("CONFIRMING", "ADVANCING", "OVERHEATED") and not confirm:
                state, reason = "CONFIRMING", "資金或廣度轉弱但尚未轉賣，降為確認階段觀察" if prev != "CONFIRMING" else "訊號減弱但未轉賣，維持確認"
            elif confirm and hot:
                state, reason = "OVERHEATED", "漲幅前 10%、成交占比急升、八成以上站上月線"
            elif confirm and qq == qq and qq >= 0.75 and st_ >= 3:
                state, reason = "ADVANCING", "資金 5／20 日同步流入、連續買超 ≥3 天、漲幅前 25%"
            elif confirm:
                state, reason = "CONFIRMING", "資金 5／20 日同步流入，產業中位數上漲、半數以上站上月線"
            elif emerging:
                state, reason = "EMERGING", "資金剛轉入（5 日轉正或占比上升），價格尚未確認"
            elif prev == "FADING" and (d5 or 0) < 0:
                state, reason = "FADING", "資金持續流出"
            else:
                state, reason = "DORMANT", "沒有一致的資金或價格訊號"
        risks = (["擁擠：漲幅與成交占比都在高檔，追價風險高"] if state == "OVERHEATED" else []) + (["資金撤離中"] if state == "FADING" else [])
        r.update(state=state, prev_state=prev, changed=state != prev, reason=reason, why_now=why, risks=risks)
    out["industries"] = sorted(rows, key=lambda r: -(r["all3"]["d5"] or 0))
    STAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STAGE_FILE.write_text(json.dumps({"date": out["date"], "prev": prev_map, "stages": {r["industry"]: r["state"] for r in rows}}, ensure_ascii=False))
    # 官方上市三大法人金額（證交所 買進金額－賣出金額），燈號程式每天存在 history/credit
    try:
        cm = pd.concat([pd.read_csv(f, dtype={"date": str}) for f in sorted((DATA / "history" / "credit").glob("market_*.csv"))]).drop_duplicates("date", keep="last")
        cm = cm[cm.date <= out["date"]].dropna(subset=["foreign_net"]).tail(20)
        if len(cm) and cm.date.iloc[-1] == out["date"]:
            out["official"] = {"scope": "上市", "unit": "億元（官方：買進金額－賣出金額）"}
            for k in ("foreign", "trust", "dealer"):
                v = (cm[k + "_net"] / 1e8).tolist()
                out["official"][k] = {"today": fnum(v[-1], 2), "d5": fnum(sum(v[-5:]), 2), "d20": fnum(sum(v), 2), "streak": streak(v)}
    except Exception as ex:  # noqa
        print("官方法人金額讀取失敗：", ex)
    w("market/flows.json", out)
    return out, ins


# ---------- 個股 ----------
def stocks(a, names, imap, ins, day, yoy, v2codes, yoy_prev):
    a = a[a.date <= day]
    g = a.groupby("code")
    a = a.assign(ma20=g.close.transform(lambda x: x.rolling(20).mean()), ma60=g.close.transform(lambda x: x.rolling(60).mean()))
    a = a.assign(h250=g.close.transform(lambda x: x.rolling(250, min_periods=120).max()), ret60=g.close.transform(lambda x: x / x.shift(60) - 1),
                 v5=g.value.transform(lambda x: x.rolling(5, min_periods=3).mean()), v20=g.value.transform(lambda x: x.rolling(20, min_periods=15).mean()),
                 nd=g.cumcount() + 1, l250=g.close.transform(lambda x: x.rolling(250, min_periods=120).min()),
                 r20=g.close.transform(lambda x: x / x.shift(20) - 1))
    tail = a.groupby("code").tail(80)
    inst = {}
    if ins is not None:
        idays = sorted(ins.date.unique())
        for code, x in ins.groupby("code"):
            x = x.set_index("date").reindex(idays)
            inst[code] = {k[0] + "_" + p: (fnum(x[c].iloc[-1] if p == "today" else x[c].iloc[-int(p):].sum(), 0) if len(x) else None)
                          for c, k in (("foreign_net", "f"), ("trust_net", "t"), ("dealer_net", "d")) for p in ("today", "5", "20")}
            inst[code].update({"f_streak": streak(x.foreign_net.tolist()), "t_streak": streak(x.trust_net.tolist())})
    idx = []
    live = set(a[a.date == day].code)
    for code, x in tail.groupby("code"):
        if code not in live:
            continue
        nm = names.get(code, ""); ind = imap.get(code, "")
        idx.append([code, nm, ind])
        b = {"t": x.date.dt.strftime("%Y%m%d").tolist(), "o": [fnum(v, 2) for v in x.open], "h": [fnum(v, 2) for v in x.high],
             "l": [fnum(v, 2) for v in x.low], "c": [fnum(v, 2) for v in x.close], "ma20": [fnum(v, 2) for v in x.ma20], "ma60": [fnum(v, 2) for v in x.ma60]}
        w(f"stocks/{code}.json", {"code": code, "name": nm, "industry": ind, "bars": b, "inst": inst.get(code),
                                  "rev_yoy": fnum(yoy.get(code), 1), "v2": v2codes.get(code),
                                  "chk": {"vs_h250": fnum(x.close.iloc[-1] / x.h250.iloc[-1] - 1), "vol_ratio": fnum(x.v5.iloc[-1] / x.v20.iloc[-1], 2),
                                          "val20": fnum(x.v20.iloc[-1], 0), "ret60": fnum(x.ret60.iloc[-1])},
                                  "hc": {"r20": fnum(x.r20.iloc[-1]), "dev20": fnum(x.close.iloc[-1] / x.ma20.iloc[-1] - 1), "dev60": fnum(x.close.iloc[-1] / x.ma60.iloc[-1] - 1),
                                         "vs_l250": fnum(x.close.iloc[-1] / x.l250.iloc[-1] - 1), "vol1": fnum(x.value.iloc[-1] / x.v20.iloc[-1], 2),
                                         "chg1": fnum(x.close.iloc[-1] / x.close.iloc[-2] - 1) if len(x) > 1 else None,
                                         "black": bool(x.close.iloc[-1] < x.open.iloc[-1]), "yoy_prev3": fnum(yoy_prev.get(code), 1)}})
    w("search/index.json", sorted(idx))
    return len(idx)


# ---------- V2 ----------
def v2(day, a):
    lj = json.loads((DATA / "extra" / "lights" / "latest.json").read_text())
    pf = DATA / "extra" / "picks" / "latest.json"
    pk = json.loads(pf.read_text()) if pf.exists() else {}
    sf = DATA / "extra" / "picks" / "state.json"
    st = json.loads(sf.read_text()) if sf.exists() else {}
    tf = DATA / "extra" / "picks" / "trades.csv"
    trades = pd.read_csv(tf, dtype={"code": str}).tail(60).to_dict("records") if tf.exists() else []
    eq = st.get("eq", {})
    # 對照：0050（從 V2 開始那天起，同樣 100 萬）
    cmp_ = []
    if eq:
        d0 = min(eq)
        e = pd.Series(eq).sort_index()
        cmp_.append({"name": "V2.1（正式）", "value": e.iloc[-1], "ret": e.iloc[-1] / 1e6 - 1, "dd": e.iloc[-1] / e.max() - 1, "since": d0})
        try:
            p50 = []
            for d in daily_dirs():
                if d.name >= d0:
                    x = pd.read_csv(d / "normalized" / "twse_ohlcv.csv", dtype={"stock_id": str}, encoding="utf-8-sig")
                    v = x[x.stock_id == "0050"].close
                    if len(v): p50.append((d.name, float(v.iloc[0])))
            if p50:
                s = pd.Series(dict(p50)).sort_index(); v = 1e6 * s / s.iloc[0]
                cmp_.append({"name": "0050（基準）", "value": v.iloc[-1], "ret": v.iloc[-1] / 1e6 - 1, "dd": v.iloc[-1] / v.max() - 1, "since": d0})
        except Exception as ex:  # noqa
            print("0050 對照失敗：", ex)
    # 持股現況：現價、未實現報酬、出場線
    last = a[a.date <= day].groupby("code").tail(60)
    ma60 = last.groupby("code").close.mean().where(last.groupby("code").close.count() >= 60)
    cl = last.groupby("code").close.last()
    hold = []
    for c, p in (st.get("pos") or {}).items():
        px = cl.get(c); ep = p.get("epx")
        h = {"code": c, "kind": p.get("kind"), "ent": str(p.get("ent"))[:10], "epx": ep, "sh": p.get("sh"), "close": fnum(px, 2),
             "ret": fnum(px / ep - 1 - 0.006) if px and ep else None, "mv": fnum(px * p.get("sh", 0), 0) if px else None}
        if p.get("kind") == "多頭":
            h["stop"] = fnum(ep * 0.9, 2); h["ma60"] = fnum(ma60.get(c), 2)
        else:
            h["tp"] = fnum(ep * 1.5, 2)
        hold.append(h)
    # 成交統計（每筆報酬是那 9% 部位自己的漲跌）
    stats = {}
    if tf.exists():
        tt = pd.read_csv(tf, dtype={"code": str}); tt = tt[tt.side == "賣"]
        for k, g_ in tt.groupby("kind"):
            wv, lv = g_[g_.ret > 0], g_[g_.ret <= 0]
            stats[k] = {"n": len(g_), "win": fnum(len(wv) / len(g_)), "avg_win": fnum(wv.ret.mean()), "avg_loss": fnum(lv.ret.mean()), "max_loss": fnum(g_.ret.min())}
    obj = {"date": lj.get("date"), "lights": {"text": lj.get("text"), "state": (lj.get("state") or {}).get("state"), "green": lj.get("green"),
                                               "lamps_on": lj.get("lamps_on"), "event_type": lj.get("event_type"), "key": lj.get("key"), "stage": lj.get("stage"),
                                               "vix": lj.get("vix_us"), "maint_ratio": lj.get("maint_ratio"), "bottom_est": lj.get("bottom_est"),
                                               "lamps": lj.get("lights"), "exit_date": (lj.get("state") or {}).get("exit_date"), "exit_px": (lj.get("state") or {}).get("exit_px")},
           "picks": pk, "account": {"mode": st.get("mode"), "value": st.get("value"), "cash": st.get("cash"), "eq": eq,
                                    "pos": hold, "stats": stats, "peak": max(eq.values()) if eq else None},
           "trades": trades, "compare": cmp_}
    w("v2.json", obj)
    v2codes = {h["code"]: f"V2 持股（{h.get('kind')}）" for h in pk.get("holdings", [])}
    v2codes.update({b["code"]: "V2 明天要買" for b in pk.get("picks", [])})
    v2codes.update({s["code"]: f"V2 明天要賣（{s.get('why', '')}）" for s in pk.get("sells", [])})
    return v2codes


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    a, names = load_ohlcv()
    day = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else a.date.max()
    a = a[a.date <= day]
    imap = pd.read_csv(DATA / "extra" / "industry_map.csv", dtype={"code": str}).drop_duplicates("code").set_index("code").industry
    mi = market_index(a, day)
    fl, ins = flows(a, day, imap)
    rf = DATA / "extra" / "revenue_history.csv"
    yoy, yoy_prev = {}, {}
    if rf.exists():
        rv = pd.read_csv(rf, dtype={"code": str, "available_date": str})
        rv["ad"] = pd.to_datetime(rv.available_date); rv["yoy"] = pd.to_numeric(rv.yoy_pct, errors="coerce")
        rr = rv[rv.ad <= day].sort_values(["ad", "period"])
        yoy = rr.groupby("code").yoy.last().to_dict()
        yoy_prev = rr.groupby("code").yoy.apply(lambda x: x.iloc[-4:-1].mean() if len(x) >= 4 else float("nan")).to_dict()
    v2codes = v2(day, a)
    bt = DATA / "extra" / "picks_backtest_v2.json"
    if bt.exists():
        (OUT / "backtest.json").write_text(bt.read_text())
    n = stocks(a, names, imap.to_dict(), ins, day, yoy, v2codes, yoy_prev)
    ht = DATA / "extra" / "health_table.json"
    if ht.exists():
        (OUT / "health.json").write_text(ht.read_text())
    w("build.json", {"date": day.strftime("%Y%m%d"), "built_at": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M"),
                     "stocks": n, "inst_ok": fl.get("inst_ok"), "market_status": mi["status"]})
    print("App 資料完成：", day.date(), "個股", n, "產業", len(fl.get("industries", [])))


if __name__ == "__main__":
    main()
