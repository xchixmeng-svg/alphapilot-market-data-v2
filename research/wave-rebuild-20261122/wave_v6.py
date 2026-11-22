"""昨天定案的整套策略（wave12）重建：燈號事件 + 抄底 10 檔 + 多頭強勢股。規則照 11/21 對話原文。
差異：股價用原始收盤價（沒有除權息還原）；燈號事件用 replay.py 重跑的結果。"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
P = pd.read_parquet("/home/claude/work/panel_adj.parquet")
O = P.pivot_table(index="date", columns="code", values="open")
C = P.pivot_table(index="date", columns="code", values="close")
V = P.pivot_table(index="date", columns="code", values="value")
C = C[[c for c in C.columns if c[0] != "0"]]; O = O.reindex_like(C); V = V.reindex_like(C)
dates = C.index
PC = C.shift(1)
# 11/22：照研究版，每一檔用「自己有交易的日子」rolling（停牌日跳過）
_L = P[P.code.str[0] != "0"].sort_values(["code", "date"]).copy()
_g = _L.groupby("code")
_L["m60"] = _g.close.transform(lambda x: x.rolling(60).mean())
_L["h250"] = _g.close.transform(lambda x: x.rolling(250, min_periods=120).max())
_L["r60"] = _L.close / _g.close.shift(60) - 1
_L["v20"] = _g.value.transform(lambda x: x.rolling(20, min_periods=15).mean())
_L["v5"] = _g.value.transform(lambda x: x.rolling(5, min_periods=3).mean())
_pv = lambda col: _L.pivot_table(index="date", columns="code", values=col).reindex(index=C.index, columns=C.columns)
M60 = _pv("m60"); H250 = _pv("h250"); TOH = C / H250 - 1; RET60 = _pv("r60"); VAL20 = _pv("v20"); VAL5 = _pv("v5")
first = C.apply(lambda s: s.first_valid_index())
N = C.notna().cumsum()
E = pd.read_pickle("/home/claude/work/sysevents_nw_adj.pkl"); G3 = pd.read_pickle("/home/claude/work/g3_nw_adj.pkl")
tx = pd.read_csv("/home/claude/xchixmeng-svg/alphapilot-market-data-v2/data/extra/macro/TAIEX.csv", parse_dates=["date"]).set_index("date")
lim = lambda d: 1.095 if d >= pd.Timestamp("2015-06-01") else 1.065

def candA(t):
    ok = (VAL20.loc[t] >= 3e7) & (N.loc[t] >= 120) & TOH.loc[t].notna()
    return list(TOH.loc[t][ok].sort_values().index[:40])
def candM(t, volr=1.3):
    ok = (VAL20.loc[t] >= 3e7) & (N.loc[t] >= 250) & (TOH.loc[t] >= -0.001) & (VAL5.loc[t] > VAL20.loc[t] * volr)
    return list(RET60.loc[t][ok].sort_values(ascending=False).index[:40])

def run(tp=0.5, stop=-0.10, buygate=True, ddbrake=0.20, slot=0.09, start="2008-01-02", cash0=1_000_000, cost_in=0.003, cost_out=0.003, end=None):
    D = dates[(dates >= start) & ((dates <= end) if end else True)]
    cash = cash0; pos = {}; mode = "bull"; T = []; sells = []; buys = []; sellall = False; eq = {}; log = []; RS = {}; peak = [0.0]
    for k, d in enumerate(D):
        def px_open(c):
            v = O.at[d, c]; return v if v == v else C[c][:d].dropna().iloc[-1]
        for c in (list(pos) if sellall else [c for c in dict.fromkeys(sells) if c in pos]):
            p = pos.pop(c); x = px_open(c); cash += p["sh"] * x * (1 - cost_out)
            T.append(dict(code=c, kind=p["kind"], sig=p["sig"], ent=p["ent"], epx=p["epx"], sh=p["sh"], out=d, opx=x,
                          ret=x * (1 - cost_out) / (p["epx"] * (1 + cost_in)) - 1, why=("大盤出場訊號" if sellall else RS.get(c, ""))))
        if buys:
            pd_ = D[k - 1]
            val = cash + sum(p["sh"] * (C.at[pd_, c] if C.at[pd_, c] == C.at[pd_, c] else p["epx"]) for c, p in pos.items())
            for c, kind in buys:
                if len(pos) >= 10 or cash - val * (1 - 10 * slot) < val * 0.02: break
                if c in pos: continue
                x = O.at[d, c]; pc = PC.at[d, c]
                if x != x or (pc == pc and x >= pc * (lim(d) - 0.003)): continue
                amt = min(cash - val * (1 - 10 * slot), val * slot)
                pos[c] = dict(sh=amt / (x * (1 + cost_in)), kind=kind, ent=d, epx=x, sig=D[k - 1]); cash -= amt
        sells = []; buys = []; sellall = False
        val = cash + sum(p["sh"] * (C.at[d, c] if C.at[d, c] == C.at[d, c] else p["epx"]) for c, p in pos.items()); eq[d] = val
        ev = E.get(d)
        if ev == "出場":
            sellall = True; mode = "cash"; log.append((d, "出場")); continue
        if mode == "cash":
            if ev == "抓底進場": mode = "bottom"; buys = [(c, "抄底") for c in candA(d)]; log.append((d, "抓底買進"))
            elif ev == "假警報回場": mode = "bull"; peak[0] = val; log.append((d, "回場"))
            continue
        if mode == "bottom":
            for c, p in pos.items():
                cl = C.at[d, c]
                if p["kind"] == "抄底" and tp and cl == cl and cl >= p["epx"] * (1 + tp): sells.append(c); RS[c] = "抓底股漲50%停利"
            left = [c for c, p in pos.items() if p["kind"] == "抄底" and c not in sells]
            if G3.get(d, 0) >= 2 or not left:
                sells += left
                for c in left: RS[c] = "三燈兩綠，抓底股換多頭"
                mode = "bull"; peak[0] = val; log.append((d, "轉多頭"))
            else:
                continue
        for c, p in pos.items():
            cl = C.at[d, c]
            if p["kind"] == "多頭" and cl == cl and (cl < M60.at[d, c] or (stop and cl < p["epx"] * (1 + stop))):
                sells.append(c); RS[c] = "虧10%停損" if (stop and cl < p["epx"] * (1 + stop)) else "跌破季線"
        room = 10 - len([c for c in pos if c not in sells])
        if buygate and G3.get(d, 0) < 2: room = 0
        peak[0] = max(peak[0], val)
        if ddbrake and val < peak[0] * (1 - ddbrake):
            if G3.get(d, 0) < 3:
                br = [c for c, p in pos.items() if p["kind"] == "多頭" and c not in sells]; sells += br; room = 0
                for c in br: RS[c] = "帳戶回落20%煞車"
            else:
                peak[0] = val
        if room > 0: buys = [(c, "多頭") for c in candM(d) if c not in pos][:room + 10]
    return pd.Series(eq), pd.DataFrame(T), log, pos, cash

if __name__ == "__main__":
    import sys
    Q, T, L, pos, cash = run(end=sys.argv[1] if len(sys.argv) > 1 else None)
    Y = Q.index.year; yl = Q.groupby(Y).last(); y0 = np.r_[1e6, yl.values[:-1]]
    print("期末", round(Q.iloc[-1] / 1e4), "萬 回落", round((Q / Q.cummax() - 1).min() * 100), "% 交易", len(T))
    print("逐年", " ".join(f"{a%100:02d}:{round((b/c-1)*100)}" for a, b, c in zip(yl.index, yl.values, y0)))
    Q.to_pickle("/home/claude/work/wave_eq_v6.pkl"); T.to_pickle("/home/claude/work/wave_trades_v6.pkl"); pd.to_pickle((L, pos, cash), "/home/claude/work/wave_state_v6.pkl")
