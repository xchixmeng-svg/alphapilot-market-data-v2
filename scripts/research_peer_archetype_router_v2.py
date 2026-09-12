from __future__ import annotations
import json, runpy
from pathlib import Path
import numpy as np
import pandas as pd

# Reuse the audited research machinery, then replace only the degenerate selector.
# V1 selected nearest CALENDAR days, while trades are sparse; alternate archetypes therefore
# rarely reached minimum history and the router collapsed to base_peer. V2 finds nearest
# PAST COMPLETED ORDER CONTEXTS for each archetype instead. 2025 remains evaluation only.
ns = runpy.run_path('scripts/research_peer_archetype_router.py')
OUT = Path('research_out_peer_archetype_router_v2'); OUT.mkdir(exist_ok=True)
variants = ns['variants']; ctx = ns['ctx']; FEATURES = ns['FEATURES']; calendar = ns['calendar']
simulate = ns['simulate']; START = ns['START']; DEV_END = ns['DEV_END']; HOLDOUT_START = ns['HOLDOUT_START']; END = ns['END']
ARCH = ns['ARCH']

# A signal-day context bank. Every row is known by the signal close. Outcome eligibility is
# separately gated by exit_date < fit_date, so no unfinished/future trade can affect routing.
context_bank = ctx[FEATURES].copy()
context_bank.index = context_bank.index.astype(int)
month_first = {}
for d in calendar:
    month_first.setdefault(str(d)[:6], int(d))

choices=[]
for month, fit_date in month_first.items():
    past_ctx = context_bank.loc[context_bank.index < fit_date].copy()
    current_days = [d for d in calendar if str(d)[:6] == month]
    if len(past_ctx) < 80 or not current_days:
        choices.append({'month':month,'fit_date':fit_date,'archetype':'base_peer','reason':'warmup'}); continue
    mu = past_ctx.tail(252).mean(); sd = past_ctx.tail(252).std().replace(0,1).fillna(1)
    x = ((context_bank.loc[current_days[0]]-mu)/sd).fillna(0).clip(-5,5)
    scores=[]
    for a in ARCH:
        q = variants[a]
        q = q[q.exit_date < fit_date].copy()
        q = q[q.signal_date.isin(past_ctx.index)]
        if len(q) < 18:
            continue
        sig = q.signal_date.astype(int).to_numpy()
        X = ((context_bank.loc[sig]-mu)/sd).fillna(0).clip(-5,5)
        dist = ((X-x)**2).mean(axis=1).pow(.5).to_numpy()
        q = q.assign(_dist=dist).sort_values('_dist').head(min(60, len(q)))
        r = q.realized_return_net.astype(float)
        n=len(r)
        if n < 18: continue
        pos=float(r[r>0].sum()); neg=float(-r[r<0].sum()); pf=pos/neg if neg>0 else (99. if pos>0 else 0.)
        wr=float((r>0).mean()); mean=float(r.mean()); med=float(r.median())
        # Shrink toward neutral so a smaller archetype cannot win solely on noisy samples.
        shrink=n/(n+30.0)
        score=shrink*(10*mean + 2*med + .35*(wr-.5) + .08*min(pf,3))
        scores.append((score,a,n,mean,wr,pf,float(q._dist.median())))
    if not scores:
        choices.append({'month':month,'fit_date':fit_date,'archetype':'base_peer','reason':'insufficient_completed_order_contexts'})
    else:
        best=max(scores,key=lambda t:t[0])
        choices.append({'month':month,'fit_date':fit_date,'archetype':best[1],'score':best[0],'hist_n':best[2],
                        'hist_mean':best[3],'hist_wr':best[4],'hist_pf':best[5],'median_context_distance':best[6],
                        'reason':'nearest_completed_order_contexts'})

choices=pd.DataFrame(choices); choices.to_csv(OUT/'monthly_archetype_choices.csv',index=False)
choice_map=dict(zip(choices.month,choices.archetype))
selected=[]
for d in calendar:
    a=choice_map.get(str(d)[:6],'base_peer')
    z=variants[a][variants[a].signal_date==d].copy()
    if z.empty: continue
    z['archetype']=a; selected.append(z)
orders=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame()
orders.to_csv(OUT/'selected_orders.csv',index=False)

# IMPORTANT: runpy.run_path returns a mapping, but the inherited function resolves globals
# from its own __globals__ dictionary. Updating only ns['orders'] can leave simulate() bound
# to the original V1 order table. Patch the function-global explicitly and assert that the
# routed order mix is visible before any portfolio result is accepted.
simulate.__globals__['orders'] = orders
bound_orders = simulate.__globals__['orders']
if len(bound_orders) != len(orders):
    raise AssertionError(('router_order_binding_failed', len(orders), len(bound_orders)))
if not orders.empty and not bound_orders['archetype'].equals(orders['archetype']):
    raise AssertionError('router_archetype_binding_failed')
if (choices.archetype != 'base_peer').any() and (orders.archetype != 'base_peer').sum() == 0:
    raise AssertionError('router_selected_nonbase_months_but_no_nonbase_orders')

rows=[]
for a,b,l in [(START,DEV_END,'dev'),(HOLDOUT_START,END,'holdout_2025'),(START,END,'full')]:
    s,td,nd=simulate(a,b,l); rows.append(s); td.to_csv(OUT/f'trades_{l}.csv',index=False); nd.to_csv(OUT/f'nav_{l}.csv',index=False)
for y in [2023,2024,2025]:
    a=max(START,y*10000+101); b=min(END,y*10000+1231)
    if a<=b: rows.append(simulate(a,b,str(y))[0])
summary=pd.DataFrame(rows); summary.to_csv(OUT/'summary.csv',index=False)
dev=summary[summary.label=='dev'].iloc[0]; h=summary[summary.label=='holdout_2025'].iloc[0]
# Deliberately stricter than V1: minute work stays closed unless both dev and OOS are positive,
# OOS has enough trades and materially positive PF/WR. This gate is predeclared, not fit to 2025.
stable=bool(dev['return']>0 and dev.pf>1.10 and h['return']>0 and h.pf>=1.25 and h.win_rate>=.55 and h.trades>=25)
decision={'architecture':'monthly_frozen_causal_peer_archetype_router_v2','semantic_bull_bear_gate':False,
          'holdout_not_used_for_fit':True,'quality_is_fundamental':False,'minute_stage_allowed':stable,
          'primary_target_met':bool(h.cagr>=.50 and h.win_rate>=.70 and h.trades>=25),'formal_r10_modified':False,
          'repair':'nearest completed order contexts plus explicit simulate global-order binding'}
json.dump(decision,open(OUT/'decision.json','w'),indent=2)
print(choices.to_string(index=False)); print(summary.to_string(index=False)); print(json.dumps(decision,indent=2))
