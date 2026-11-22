import pandas as pd, numpy as np, grid as g
src = open('/home/claude/work/wave_v6.py').read()
s = src[src.index("def run("):src.index("if __name__")]
s = s.replace('cash0=1_000_000, cost_in=0.003', 'cash0=1_000_000, bear=None, ds=None, start_mode="bull", cost_in=0.003')
s = s.replace('pos = {}; mode = "bull";', 'pos = {}; mode = start_mode; ix0 = [None];')
s = s.replace('''            if ev == "抓底進場": mode = "bottom"; buys = [(c, "抄底") for c in candA(d)]; log.append((d, "抓底買進"))''','''            if ev == "抓底進場": mode = "bottom"; ix0[0] = IX.get(d); buys = [(c, "抄底") for c in candA(d)]; log.append((d, "抓底買進"))''')
s = s.replace('''        if mode == "bottom":
            for c, p in pos.items():''','''        if mode == "cashds":
            if G3.get(d, 0) >= 2: mode = "bull"; peak[0] = val; log.append((d, "災難停損後回場"))
            else: continue
        if mode == "bottom" and ds and IX.get(d, 1e18) < ix0[0] * (1 - ds):
            sellall = True; mode = "cashds"; log.append((d, f"災難停損（大盤比抓底日再跌 {int(ds*100)}%）")); continue
        if mode == "bottom":
            for c, p in pos.items():''')
s = s.replace('''        if buygate and G3.get(d, 0) < 2: room = 0''', '''        if buygate and G3.get(d, 0) < 2: room = 0
        if bear is not None and BEAR.get(d, False): room = 0''')
exec(s, g.__dict__)
ix = g.tx.close; ma = ix.rolling(60).mean()
g.BEAR = ((ix < ma) & (ma < ma.shift(20))).to_dict(); g.IX = ix.to_dict()
e = pd.read_pickle('sysevents_mix.pkl'); G3 = pd.read_pickle('g3_mix.pkl')
def go(dsv, E=e, start='2008-01-02', end='2025-12-31', start_mode='bull'):
    g.E, g.G3 = E, G3; g.M60 = g.M60b; g.candM = lambda t, volr=2.0: g.candM2(t, volr)
    return g.run(tp=0.5, stop=-0.15, ddbrake=0.2, bear=True, ds=dsv, start=start, end=end, start_mode=start_mode)
if __name__ == "__main__":
    print("== 歷史（2008～2025）")
    for dsv in [None, 0.10, 0.15, 0.20, 0.25]:
        Q, T, L, pos, cash = go(dsv)
        hits = [(d.date(), a) for d, a in L if a.startswith("災難")]
        q2 = Q["2020-01-02":]
        print(dsv, round(Q.iloc[-1] / 1e4), "萬 回落", round((Q / Q.cummax() - 1).min() * 100), "% 2020起", round(q2.iloc[-1] / q2.iloc[0] * 100), "觸發", hits, flush=True)
    print("== 假設 2008/1/9 空手時抓底（急跌型 6 盞），之後變熊市")
    E2 = dict(e); E2 = {k: v for k, v in E2.items() if not (pd.Timestamp("2008-01-01") <= k <= pd.Timestamp("2008-12-31"))}
    E2[pd.Timestamp("2008-01-09")] = "抓底進場"
    for dsv in [None, 0.10, 0.15, 0.20, 0.25]:
        Q, T, L, pos, cash = go(dsv, E=E2, start="2008-01-02", end="2009-12-31", start_mode="cash")
        lo = Q["2008"].min()
        print(dsv, "2008 最低帳戶", round(lo / 1e4), "萬，2008 底", round(Q["2008"].iloc[-1] / 1e4), "萬，2009 底", round(Q.iloc[-1] / 1e4), "萬", [(d.date(), a) for d, a in L][:6], flush=True)
