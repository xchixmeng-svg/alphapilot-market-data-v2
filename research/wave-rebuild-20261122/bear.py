import pandas as pd, numpy as np
import grid as g
src = open('/home/claude/work/wave_v6.py').read()
s = src[src.index("def run("):src.index("if __name__")]
s = s.replace('cash0=1_000_000, cost_in=0.003', 'cash0=1_000_000, bear=None, cost_in=0.003')
s = s.replace('''        if buygate and G3.get(d, 0) < 2: room = 0''', '''        if buygate and G3.get(d, 0) < 2: room = 0
        if bear is not None and BEAR[bear].get(d, False):
            room = 0
            if bear.endswith("全賣"):
                for c, p in pos.items():
                    if p["kind"] == "多頭" and c not in sells: sells.append(c); RS[c] = "熊市指標"''')
exec(s, g.__dict__)
ix = g.tx.close
ma240 = ix.rolling(240).mean(); ma60 = ix.rolling(60).mean()
B = {}
B["年線下"] = (ix < ma240)
B["年線下彎"] = (ma240 < ma240.shift(20))
B["季線下且下彎"] = (ix < ma60) & (ma60 < ma60.shift(20))
B["年線下且下彎"] = (ix < ma240) & (ma240 < ma240.shift(20))
for k in list(B):
    B[k + "全賣"] = B[k]
g.BEAR = {k: v.to_dict() for k, v in B.items()}
def go(bear, ev="不等", tp=0.5, stop=-0.15, volr=2.0, brake=0.2):
    g.E, g.G3 = g.EV[ev]; g.M60 = g.M60b; g.candM = lambda t, volr=volr: g.candM2(t, volr)
    Q, T, L, pos, cash = g.run(tp=tp, stop=stop, ddbrake=brake, bear=bear, end="2025-12-31")
    return Q, T
if __name__ == "__main__":
    rows = []
    for ev, volr, stop in [("不等", 2.0, -0.15), ("A等底價", 1.3, -0.10)]:
        for b in [None, "年線下", "年線下彎", "季線下且下彎", "年線下且下彎", "年線下全賣", "年線下彎全賣", "季線下且下彎全賣", "年線下且下彎全賣"]:
            Q, T = go(b, ev=ev, volr=volr, stop=stop)
            Y = Q.index.year; yl = Q.groupby(Y).last(); y0 = [1e6] + list(yl.values[:-1]); yr = dict(zip(yl.index, [(a / c - 1) * 100 for a, c in zip(yl.values, y0)]))
            r1 = Q[:"2016-12-31"]
            rows.append(dict(設定=("最好組合" if ev == "不等" else "現行"), 熊市=b or "無", 期末=round(Q.iloc[-1] / 1e4), 回落=round((Q / Q.cummax() - 1).min() * 100),
                             研究段=round(r1.iloc[-1] / 1e4), 驗證倍數=round(Q.iloc[-1] / r1.iloc[-1], 1), y08=round(yr[2008]), y11=round(yr[2011]), y15=round(yr[2015]), y18=round(yr[2018]), y22=round(yr[2022]), y25=round(yr[2025])))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_pickle("bear.pkl")
