"""上線版回測（11/22）：所有規則 N+1；寫法和給 GPT 的條件一字一句對應。"""
import pandas as pd, numpy as np, grid as g, ds
E = pd.read_pickle('/home/claude/work/sysevents_final.pkl'); G3 = pd.read_pickle('/home/claude/work/g3_final.pkl')
rv = pd.read_csv('/home/claude/xchixmeng-svg/alphapilot-market-data-v2/data/extra/revenue_history.csv', dtype={'code': str})
rv['ad'] = pd.to_datetime(rv.available_date.astype(str)); rv['yoy'] = pd.to_numeric(rv.yoy_pct, errors='coerce'); rv = rv.sort_values(['ad', 'period'])
_cache = {}
def yoy_at(t):
    k = t.strftime('%Y-%m-%d')
    if k not in _cache: _cache[k] = rv[rv.ad <= t].groupby('code').yoy.last()
    return _cache[k]
def candM(t, volr=2.0):
    ok = (g.VAL20.loc[t] >= 3e7) & (g.N.loc[t] >= 250) & (g.TOH.loc[t] >= -0.001) & (g.VAL5.loc[t] > g.VAL20.loc[t] * volr)
    s = g.RET60.loc[t][ok]
    if t >= pd.Timestamp('2013-06-01'):
        y = yoy_at(t); s = s[s.index.map(lambda c: y.get(c, np.nan)) > 20]
    return list(s.sort_values(ascending=False).index[:40])
src = open('/home/claude/work/wave_v6.py').read(); s = src[src.index("def run("):src.index("if __name__")]
s = s.replace('cash0=1_000_000, cost_in=0.003', 'cash0=1_000_000, bear=None, ds=None, start_mode="bull", cost_in=0.003')
s = s.replace('pos = {}; mode = "bull";', 'pos = {}; mode = start_mode; ix0 = [None]; carry = [];')
# 賣出：沒有開盤價（停牌）就延到下一個有開盤的日子；連續 20 個交易日沒有價格（下市）就用最後收盤價出清
old_sell = '''        for c in (list(pos) if sellall else [c for c in dict.fromkeys(sells) if c in pos]):
            p = pos.pop(c); x = px_open(c); cash += p["sh"] * x * (1 - cost_out)'''
assert old_sell in s
new_sell = '''        todo = list(pos) if sellall else [c for c in dict.fromkeys(carry + sells) if c in pos]
        carry = []
        for c in todo:
            x = O.at[d, c]
            if not (x == x):
                last = C[c][:d].dropna()
                if len(last) and (D.get_loc(d) - D.get_loc(last.index[-1]) if last.index[-1] in D else 0) >= 20:
                    x = last.iloc[-1]; RS[c] = RS.get(c, "") + "（停牌20天，用最後收盤價出清）"
                else:
                    carry.append(c); continue
            p = pos.pop(c); cash += p["sh"] * x * (1 - cost_out)'''
s = s.replace(old_sell, new_sell)
s = s.replace('why=("大盤出場訊號" if sellall else RS.get(c, ""))', 'why=("大盤出場訊號" if sellall and c not in RS else RS.get(c, "大盤出場訊號"))')
s = s.replace('if ev == "出場":\n            sellall = True;', 'if ev == "出場":\n            RS.clear(); sellall = True;')
s = s.replace('''            if ev == "抓底進場": mode = "bottom"; buys = [(c, "抄底") for c in candA(d)]; log.append((d, "抓底買進"))''', '''            if ev == "抓底進場": mode = "bottom"; ix0[0] = IX.get(d); buys = [(c, "抄底") for c in candA(d)]; log.append((d, "抓底買進"))''')
s = s.replace('''        if mode == "bottom":
            for c, p in pos.items():''', '''        if mode == "cashds":
            if G3.get(d, 0) >= 2: mode = "bull"; peak[0] = val; log.append((d, "災難停損後回場"))
            else: continue
        if mode == "bottom" and ds and IX.get(d, 1e18) < ix0[0] * (1 - ds):
            for c in pos: RS[c] = "抓底災難停損"
            sells = list(pos); mode = "cashds"; log.append((d, "抓底災難停損")); continue
        if mode == "bottom":
            for c, p in pos.items():''')
s = s.replace('''        if buygate and G3.get(d, 0) < 2: room = 0''', '''        if buygate and G3.get(d, 0) < 2: room = 0
        if bear is not None and BEAR.get(d, False): room = 0''')
s = s.replace('RS[c] = "虧10%停損" if (stop and cl < p["epx"] * (1 + stop)) else "跌破季線"', 'RS[c] = f"收盤虧{int(-stop*100)}%停損" if (stop and cl < p["epx"] * (1 + stop)) else "跌破季線"')
exec(s, g.__dict__)
g.D = g.dates
g.BEAR = ds.g.BEAR; g.IX = ds.g.IX
def go(end='2025-12-31', start='2008-01-02', stop=-0.10):
    g.E, g.G3 = E, G3; g.M60 = g.M60b; g.candM = candM
    return g.run(tp=0.5, stop=stop, ddbrake=0.2, bear=True, ds=0.2, start=start, end=end)
if __name__ == '__main__':
    Q, T, L, pos, cash = go()
    Y = Q.index.year; yl = Q.groupby(Y).last(); y0 = [1e6] + list(yl.values[:-1])
    print('期末', round(Q.iloc[-1] / 1e4), '萬 回落', round((Q / Q.cummax() - 1).min() * 100), '% 交易', len(T), '未平倉', len(pos))
    print('逐年', ' '.join(f"{a%100:02d}:{(b/c-1)*100:+.0f}" for a, b, c in zip(yl.index, yl.values, y0)))
    print('出場原因', T.why.value_counts().to_dict())
    print('抓底', [(str(d.date()), a) for d, a in L if '抓底' in a])
    Q.to_pickle('final_eq.pkl'); T.to_pickle('final_trades.pkl'); pd.to_pickle((L, pos, cash), 'final_state.pkl')
