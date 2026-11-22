import pandas as pd, numpy as np, grid as g, ds
def build(buy_on_back, reset_daily):
    s=open('/home/claude/work/wave_v6.py').read(); s=s[s.index("def run("):s.index("if __name__")]
    s=s.replace('cash0=1_000_000, cost_in=0.003','cash0=1_000_000, bear=None, ds=None, start_mode="bull", cost_in=0.003')
    s=s.replace('pos = {}; mode = "bull";','pos = {}; mode = start_mode; ix0 = [None];')
    old='''        if mode == "cash":
            if ev == "抓底進場": mode = "bottom"; buys = [(c, "抄底") for c in candA(d)]; log.append((d, "抓底買進"))
            elif ev == "假警報回場": mode = "bull"; peak[0] = val; log.append((d, "回場"))
            continue'''
    assert old in s
    new='''        if mode == "cash":
            if ev == "抓底進場": mode = "bottom"; ix0[0] = IX.get(d); buys = [(c, "抄底") for c in candA(d)]; log.append((d, "抓底買進")); continue
            elif ev == "假警報回場": mode = "bull"; peak[0] = val; log.append((d, "回場"))''' + ('' if buy_on_back else '; continue') + '''
            else: continue'''
    s=s.replace(old,new)
    if reset_daily:
        s=s.replace('''        peak[0] = max(peak[0], val)''','''        peak[0] = max(peak[0], val)
        if G3.get(d, 0) >= 3: peak[0] = val''')
    s=s.replace('''        if mode == "bottom":
            for c, p in pos.items():''','''        if mode == "cashds":
            if G3.get(d, 0) >= 2: mode = "bull"; peak[0] = val
            else: continue
        if mode == "bottom" and ds and IX.get(d, 1e18) < ix0[0] * (1 - ds):
            sellall = True; mode = "cashds"; continue
        if mode == "bottom":
            for c, p in pos.items():''')
    s=s.replace('''        if buygate and G3.get(d, 0) < 2: room = 0''','''        if buygate and G3.get(d, 0) < 2: room = 0
        if bear is not None and BEAR.get(d, False): room = 0''')
    exec(s, g.__dict__)
if __name__ == "__main__":
    for bb,rd in [(False,False),(True,False),(False,True),(True,True)]:
        build(bb,rd)
        Q,T,L,pos,cash=ds.go(0.2)
        print('回場當天就買' if bb else '回場隔天才買', '|', '三燈全亮每天重設高點' if rd else '只在煞車時重設', '→', round(Q.iloc[-1]/1e4),'萬 回落',round((Q/Q.cummax()-1).min()*100),'% 交易',len(T),'煞車',(T.why=='帳戶回落20%煞車').sum(), flush=True)
