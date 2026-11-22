import pandas as pd, numpy as np, bear as b, grid as g
e = pd.read_pickle('sysevents_mix.pkl'); g.EV['混合'] = (e, pd.read_pickle('g3_mix.pkl'))
src = open('/home/claude/work/bear.py').read()
Q, T = b.go('季線下且下彎', ev='混合', volr=2.0, stop=-0.15)
T = T.copy()
T['pnl'] = T.sh * T.opx * (1 - 0.003) - T.sh * T.epx * (1 + 0.003)
# 1) 帳
_, _, _, pos, cash = (None,)*5
