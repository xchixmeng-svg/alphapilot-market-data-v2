from pathlib import Path
import subprocess

# Start from the v10 structural hypotheses, then repair research-only portfolio accounting.
subprocess.run(['python','scripts/build_repricing_v10_runtime.py'], check=True)
p = Path('scripts/research_12_stock_repricing_v10_runtime.py')
src = p.read_text()

# 1) A scheduled exit must remain due until the first later tradable quote.
old = "            if p['exit_date']!=d: continue"
new = "            if d < p['exit_date']: continue"
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new, 1)

# 2) Refresh marks on quoted days, but carry forward the last valid mark through suspensions.
needle = "        for code,p in list(pos.items()):\n            if d < p['exit_date']: continue"
insert = "        for code,p in list(pos.items()):\n            rr=rowmap.get((d,code))\n            if rr is not None and np.isfinite(rr.close) and rr.close>0:\n                p['last_mark']=float(rr.close)\n        for code,p in list(pos.items()):\n            if d < p['exit_date']: continue"
assert src.count(needle) == 1, src.count(needle)
src = src.replace(needle, insert, 1)

# 3) Every new position receives an explicit executable-day mark.
old = "            pos[code]={'shares':shares,'entry_price':ep,'entry_date':d,'signal_date':item['signal_date'],'exit_date':calendar[j]}"
new = "            pos[code]={'shares':shares,'entry_price':ep,'entry_date':d,'signal_date':item['signal_date'],'exit_date':calendar[j],'last_mark':float(rr.close) if np.isfinite(rr.close) and rr.close>0 else ep}"
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new, 1)

# 4) NAV must never zero-out a live holding merely because that stock has no row that day.
old = "        nav=cash+sum(p['shares']*float(rowmap[(d,c)].close) for c,p in pos.items() if (d,c) in rowmap and np.isfinite(rowmap[(d,c)].close)); navrows.append({'date':d,'nav':nav,'cash':cash,'positions':len(pos)})"
new = "        nav=cash+sum(p['shares']*float(p.get('last_mark',p['entry_price'])) for c,p in pos.items());\n        if cash < -0.01 or nav < -0.01: raise RuntimeError('portfolio accounting invariant failed')\n        navrows.append({'date':d,'nav':nav,'cash':cash,'positions':len(pos)})"
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new, 1)

# 5) At the research-window boundary, only actually quoted positions are executable exits.
# Suspended positions remain marked, not fictitiously converted to cash or dropped from NAV.
old = "    nav=pd.DataFrame(navrows); nav.loc[nav.index[-1],['nav','cash','positions']]=[cash,cash,0]; t=pd.DataFrame(trades)\n    years=max((pd.to_datetime(str(end))-pd.to_datetime(str(start))).days/365.25,1/252); cagr=(cash/INIT)**(1/years)-1; dd=float((nav.nav/nav.nav.cummax()-1).min())"
new = "    final_nav=cash+sum(p['shares']*float(p.get('last_mark',p['entry_price'])) for p in pos.values())\n    nav=pd.DataFrame(navrows); nav.loc[nav.index[-1],['nav','cash','positions']]=[final_nav,cash,len(pos)]; t=pd.DataFrame(trades)\n    years=max((pd.to_datetime(str(end))-pd.to_datetime(str(start))).days/365.25,1/252); cagr=(final_nav/INIT)**(1/years)-1; dd=float((nav.nav/nav.nav.cummax()-1).min())"
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new, 1)

# Keep all v10 hypotheses unchanged for this repair-validation batch: their prior economics were contaminated.
out = Path('scripts/research_12_stock_repricing_v11_runtime.py')
compile(src, str(out), 'exec')
out.write_text(src)
