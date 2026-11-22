"""對帳：用 picks_core 一天一天跑 2008～2025，跟 final.py 回測逐筆比。特徵用回測同一份。"""
import sys, pandas as pd, numpy as np
sys.path.insert(0, '/home/claude/alphapilot-market-data-v2/scripts')
import importlib, picks_core as K; importlib.reload(K)
import final as F, grid as g
F.go(end='2008-01-03')   # 設定 g 的全域（E、G3、candM…）
E, G3 = F.E, F.G3
C, O, PC, M60 = g.C, g.O, g.PC, g.M60b
BEAR, IX = g.BEAR, g.IX
D = C.index[(C.index >= '2008-01-02') & (C.index <= '2025-12-31')]
act_map = {'出場': 'exit', '抓底進場': 'bottom', '假警報回場': 'back'}
st = K.new_state(); trades = []; eq = {}
lastpos = {}
for k, d in enumerate(D):
    if k > 0:
        opn = O.loc[d].dropna().to_dict(); ref = PC.loc[d].dropna().to_dict()
        lim = (1.095 if d >= pd.Timestamp('2015-06-01') else 1.065) - 0.003
        held = list(st['pos'])
        lp = {}; ds_ = {}
        for c in held:
            s_ = C[c][:d].dropna()
            if len(s_): lp[c] = s_.iloc[-1]; ds_[c] = D.get_loc(d) - D.get_loc(s_.index[-1]) if s_.index[-1] in D else 0
        trades += K.execute(st, d, opn, opn, ref, lim, lp, ds_)
    close = C.loc[d].to_dict(); ma = M60.loc[d].to_dict()
    r = K.decide(st, d, close, ma, act_map.get(E.get(d), ''), G3.get(d, 0), IX.get(d), bool(BEAR.get(d, False)),
                 lambda: g.candA(d), lambda: g.candM(d))
    eq[d] = st["value"]
    if eq[d] != eq[d]: print("NaN at", d, st["cash"], [(c, p) for c, p in st["pos"].items() if p["epx"] != p["epx"] or p["sh"] != p["sh"]][:3]); break
Q = pd.Series(eq); print("NaN 第一天", Q[Q.isna()].index[:1].tolist(), "cash", st["cash"]); T = pd.DataFrame([t for t in trades if t['side'] == '賣'])
print('核心程式：期末', round(Q.iloc[-1] / 1e4), '萬 回落', round((Q / Q.cummax() - 1).min() * 100), '% 已平倉', len(T), '持有', len(st['pos']))
B = pd.read_pickle('/home/claude/work/final_trades.pkl')
a = set(zip(B.code, B.ent.dt.strftime('%Y-%m-%d'), B.out.dt.strftime('%Y-%m-%d')))
b = set(zip(T.code, pd.to_datetime(T.ent).dt.strftime('%Y-%m-%d'), pd.to_datetime(T.date).dt.strftime('%Y-%m-%d')))
print('回測', len(a), '核心', len(b), '完全一樣', len(a & b))
print('只在回測', sorted(a - b, key=lambda x: x[1])[:5]); print('只在核心', sorted(b - a, key=lambda x: x[1])[:5])
