"""燈號整合版（lights_daily.py 11/21 版，含底價區＋等 120 天）逐日重跑 2008→今。
訊號日收盤後判斷，下個交易日開盤執行。標的：加權指數（價格指數，不含股息），來回成本 0.385%。"""
import sys, glob, json, numpy as np, pandas as pd
R = "/home/claude/xchixmeng-svg/alphapilot-market-data-v2"
sys.path.insert(0, f"{R}/scripts")
import lights_daily as L

MAC = f"{R}/data/extra/macro"
mac = lambda n: pd.read_csv(f"{MAC}/{n}.csv", parse_dates=["date"]).set_index("date")
tx = mac("TAIEX"); tx = tx[tx.close > 0]
tw = mac("TWD").close
tpe = pd.read_csv(f"{MAC}/TWD_TPE.csv", parse_dates=["date"]).set_index("date").twd_tpe
tw = tpe.reindex(tw.index.union(tpe.index)).combine_first(tw)
vx = mac("VIX").close
P = pd.read_parquet("/home/claude/work/panel_adj.parquet")
px = P.pivot_table(index="date", columns="code", values="close")
tv = P.pivot_table(index="date", columns="code", values="value")
ma = px.rolling(20).mean(); liq = tv.rolling(20).mean() >= 3e7
valid = (ma.notna() & liq).sum(axis=1)
br = ((px > ma) & ma.notna() & liq).sum(axis=1) / valid
br = br[valid > 100]
m = pd.concat([pd.read_csv(f, dtype={"date": str}) for f in sorted(glob.glob(f"{R}/data/history/credit/market_*.csv"))]).drop_duplicates("date")
m.index = pd.to_datetime(m.date); m = m.sort_index()
s = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{R}/data/history/credit/stock_margin_*.parquet"))])
lev = s[s.code.isin(["00631L", "00685L"])].assign(d=lambda q: pd.to_datetime(q.date)).pivot_table(index="d", columns="code", values="margin_bal")

D = tx[["open", "high", "low", "close"]].copy()
D = D[D.index >= "2006-06-01"]
_dup = (D[["open", "high", "low", "close"]] == D[["open", "high", "low", "close"]].shift(1)).all(axis=1)
D = D[~_dup]   # 11/22：颱風休市日（指數資料原封不動複製前一天）不算交易日
D["br"] = br.reindex(D.index)
D["twd"] = tw.reindex(D.index.union(tw.index)).ffill().reindex(D.index)
D["vix"] = vx.reindex(D.index.union(vx.index)).ffill().reindex(D.index)
vxu = vx.copy(); vxu.index = vxu.index + pd.Timedelta(days=1)
D["vix_us"] = vxu.reindex(D.index.union(vxu.index)).ffill().reindex(D.index)
D = D.join(m[["margin_bal_k", "foreign_net"]]).join(lev)
c = D.close; cv = c.values
D["dd"] = c / c.rolling(250, min_periods=60).max() - 1
mb = D.margin_bal_k.ffill()
cols = [k for k in ("00631L", "00685L") if k in D]
tot = D[cols].ffill().fillna(0).sum(axis=1)
fn = D.foreign_net / 1e8
f10 = fn.rolling(10).sum()
ma240 = c.rolling(240).mean()

def day_eval(i):
    r = D.iloc[i]; prev60 = D.iloc[i - 60:i]
    sw = r.dd <= -0.10
    L_money = r.twd <= D.twd.iloc[i - 59:i + 1].max() * 0.985
    L_people = bool(r.br > 0.5) if r.br == r.br else False
    L_price = r.close > (c.iloc[i - 59:i + 1].max() + c.iloc[i - 59:i + 1].min()) / 2
    g = int(L_money) + int(L_people) + int(L_price)
    m_dd = mb.iloc[i] / mb.iloc[max(0, i - 249):i + 1].max() - 1
    lev5 = tot.iloc[i] / tot.iloc[i - 5] - 1 if tot.iloc[i - 5] > 0 else 0.0
    if abs(lev5) > 0.6: lev5 = 0.0
    L_margin = bool((m_dd < r.dd) and (lev5 <= 0))
    heavy = fn.iloc[i - 9:i + 1].min() < fn.iloc[max(0, i - 249):i + 1].quantile(0.05)
    fmin10 = fn.iloc[i - 9:i + 1].min()
    L_inst = bool(heavy and ((fn.iloc[i] > 0 and fn.iloc[i - 1] < 0) or (fn.iloc[i] < 0 and fn.iloc[i] > fmin10 * 0.5)))
    out = dict(sw=bool(sw), g=g, L_margin=L_margin, L_inst=L_inst, m_dd=m_dd)
    a0 = max(0, i - 249); h = a0 + int(np.argmax(cv[a0:i + 1]))
    s20 = next((k for k in range(h, i + 1) if cv[k] <= cv[h] * 0.80), None)
    be = None
    if s20 is not None and ma240.iloc[s20] == ma240.iloc[s20]:
        e = float(ma240.iloc[s20]) * 0.764; be = dict(date=D.index[s20], est=e, hi=e * 1.03)
    out["bottom_est"] = be
    f10q = f10.iloc[max(0, i - 499):i + 1].quantile(0.10)
    out["foreign_heavy"] = bool(f10.iloc[i] <= f10q)
    out["signal"] = False; out["lamps"] = None
    if not sw:
        return out
    lv_all = L.fib_levels(c.iloc[:i + 1])
    near = [(v, n) for v, n in lv_all if abs(r.low / v - 1) <= 0.015 or (r.low <= v <= r.close)]
    prevh = cv[max(0, h - 250):h]
    newhigh = len(prevh) == 0 or cv[h] >= prevh.max()
    s12 = next((k for k in range(h, i + 1) if cv[k] <= cv[h] * 0.88), None)
    ev = "未定" if s12 is None else ("急跌型" if (s12 - h <= 35 and newhigh) else "慢跌型")
    panic20 = bool(D.br.iloc[i - 19:i + 1].min() < 0.10)
    twd_ok = bool(r.twd < D.twd.iloc[i - 20:i].max())
    vix_ok = bool(D.vix_us.iloc[i] >= 25)
    margin25 = bool(m_dd <= -0.25)
    brs, tws, ddv = D.br, D.twd, D.dd
    seq = False
    for k in range(max(21, i - 15), i):
        if ddv.iloc[k] > -0.10: continue
        nl = cv[k] < cv[k - 20:k].min()
        cd = nl and brs.iloc[k] > brs.iloc[k - 20:k].min() + 0.03
        if cd and tws.iloc[k] < tws.iloc[k - 20:k].max() and brs.iloc[k - 19:k + 1].min() < 0.10: seq = True
    seq = bool(seq and r.dd <= -0.10 and (r.br >= 0.30 if r.br == r.br else False) and fn.iloc[i - 2:i + 1].sum() > 0)
    base = bool(sw and (L_margin or L_inst) and near and panic20 and twd_ok)
    sig = bool(vix_ok and ((base and margin25) or seq)) if ev == "慢跌型" else bool(vix_ok and base)
    out.update(signal=sig, ev=ev, lamps=dict(twd=twd_ok, panic=panic20, vix=vix_ok, mi=bool(L_margin or L_inst), fib=bool(near)))
    return out

COST_BUY, COST_SELL = 0.001425, 0.001425 + 0.001
start = D.index.get_loc(D.index[D.index >= "2008-01-02"][0])
st = {"state": "in"}; cash, units = 1_000_000.0, 0.0
# 起始：2008/1/2 開盤買進
i0 = start; units = cash * (1 - COST_BUY) / D.open.iloc[i0]; cash = 0.0
trades = [dict(kind="起始買進", sig=None, date=D.index[i0], px=D.open.iloc[i0])]
pend = None; eq = []; EV = {}; G3 = {}
for i in range(start, len(D)):
    day = D.index[i]
    if pend and i > start:   # 執行昨天的訊號（今天開盤）
        kind, sigd = pend; op = D.open.iloc[i]
        if kind == "sell" and units > 0:
            cash = units * op * (1 - COST_SELL); units = 0; trades.append(dict(kind="出場", sig=sigd, date=day, px=op))
        elif kind in ("bottom", "back") and units == 0:
            units = cash * (1 - COST_BUY) / op; cash = 0; trades.append(dict(kind="抓底買進" if kind == "bottom" else "回場買進", sig=sigd, date=day, px=op))
        pend = None
    e = day_eval(i); r = D.iloc[i]
    be = e["bottom_est"]; hi_px = be["hi"] if be else None
    price_ok = bool(hi_px is None or r.close <= hi_px or e.get("ev") == "急跌型")
    s_ = st["state"]; act = None
    if s_ == "in" and e["g"] == 0 and r.dd <= -0.07 and e["foreign_heavy"]:
        st = dict(st, state="cash"); act = "sell"
    elif s_ == "cash":
        if e["signal"] and price_ok:
            st = dict(st, state="bottom", signal_date=day); act = "bottom"
        elif e["signal"] and st.get("expired_est") != (be or {}).get("date"):
            st = dict(st, state="wait", wait_since=day, wait_n=0, wait_hi=hi_px, wait_est=be["date"])
            trades.append(dict(kind="燈號湊齊，等底價", sig=day, date=None, px=r.close, extra=f"底價區上緣 {hi_px:,.0f}"))
        elif e["g"] >= 2:
            st = dict(st, state="in"); act = "back"
    elif s_ == "wait":
        st["wait_n"] += 1
        if r.close <= st["wait_hi"]:
            st = dict(st, state="bottom", signal_date=day); act = "bottom"
        elif st["wait_n"] >= 120:
            trades.append(dict(kind="等 120 天沒到底價，放棄抓底", sig=day, date=None, px=r.close))
            st = dict(st, state="cash", expired_est=st.get("wait_est"))
    elif s_ == "bottom" and e["g"] >= 2:
        st = dict(st, state="in")   # 持股不動，只轉狀態
        trades.append(dict(kind="三燈兩盞，抓底股轉一般持有", sig=day, date=None, px=r.close))
    if act: pend = (act, day); EV[day] = {'sell': '出場', 'bottom': '抓底進場', 'back': '假警報回場'}[act]
    G3[day] = e['g']
    eq.append((day, cash + units * r.close, s_))
    if e["signal"]:
        trades.append(dict(kind="抓底訊號（6 盞）" if price_ok else "抓底訊號（5 盞，未到底價）", sig=day, date=None, px=r.close, extra=json.dumps(e["lamps"], ensure_ascii=False) + " " + e.get("ev", "")))
E = pd.DataFrame(eq, columns=["date", "equity", "state"]).set_index("date")
E.to_csv("/home/claude/work/lights_equity_final.csv"); pd.to_pickle(EV, "/home/claude/work/sysevents_final.pkl"); pd.to_pickle(G3, "/home/claude/work/g3_final.pkl")
pd.DataFrame(trades).to_csv("/home/claude/work/lights_trades_raw_final.csv", index=False)
hold = 1_000_000 * (1 - COST_BUY) * D.close.iloc[start:] / D.open.iloc[start]
dd = E.equity / E.equity.cummax() - 1; hdd = hold / hold.cummax() - 1
print("系統", round(E.equity.iloc[-1]), "最大回落", round(dd.min() * 100, 1), "| 一直抱", round(hold.iloc[-1]), round(hdd.min() * 100, 1))
print("目前狀態", st)
