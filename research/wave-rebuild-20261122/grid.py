"""最優策略搜尋：研究段 2008～2016 挑，驗證段 2017～2025 驗。季線用正確算法（每檔自己的交易日）。"""
import itertools, pickle, sys
import pandas as pd, numpy as np
src = open('/home/claude/work/wave_v6.py').read().split('if __name__')[0]
src = src.replace('E = pd.read_pickle("/home/claude/work/sysevents_nw_adj.pkl"); G3 = pd.read_pickle("/home/claude/work/g3_nw_adj.pkl")', 'E = {}; G3 = {}')
exec(src)
_g2 = _L.groupby("code"); _L["m20"] = _g2.close.transform(lambda x: x.rolling(20).mean())
M20 = _pv("m20"); M60b = M60
EV = {"A等底價": (pd.read_pickle("sysevents_w_adj.pkl"), pd.read_pickle("g3_w_adj.pkl")),
      "不等": (pd.read_pickle("sysevents_nw_adj.pkl"), pd.read_pickle("g3_nw_adj.pkl"))}
def candM2(t, volr):
    ok = (VAL20.loc[t] >= 3e7) & (N.loc[t] >= 250) & (TOH.loc[t] >= -0.001) & (VAL5.loc[t] > VAL20.loc[t] * volr)
    return list(RET60.loc[t][ok].sort_values(ascending=False).index[:40])
def one(p):
    global E, G3, M60, candM
    ev, tp, stop, ma, volr, brake = p
    E, G3 = EV[ev]; M60 = M20 if ma == "月線" else M60b
    candM = lambda t, volr=volr: candM2(t, volr)
    Q, T, L, pos, cash = run(tp=tp, stop=stop, ddbrake=brake, end="2025-12-31")
    r1 = Q[:"2016-12-31"]; r2 = Q["2017-01-01":]
    dd = lambda q: (q / q.cummax() - 1).min()
    return dict(事件=ev, 抓底停利=tp, 停損=stop, 出場線=ma, 量增=volr, 煞車=brake, 期末=Q.iloc[-1] / 1e4,
                研究段=r1.iloc[-1] / 1e4, 研究回落=dd(r1) * 100, 驗證倍數=r2.iloc[-1] / r1.iloc[-1], 驗證回落=dd(Q["2016-12-31":] / Q["2016-12-31":].iloc[0]) * 100,
                全期回落=dd(Q) * 100, 交易=len(T))
if __name__ == "__main__":
    from multiprocessing import Pool
    grid = list(itertools.product(["A等底價", "不等"], [0.3, 0.5, 1.0, None], [-0.07, -0.10, -0.15], ["季線", "月線"], [1.3, 2.0], [0.20, None]))
    out = []
    with Pool(2) as pool:
        for k, r in enumerate(pool.imap_unordered(one, grid)):
            out.append(r)
            if k % 12 == 11:
                pd.DataFrame(out).to_pickle("grid_partial.pkl"); print(k + 1, "/", len(grid), flush=True)
    pd.DataFrame(out).to_pickle("grid.pkl"); print("done")
