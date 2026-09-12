from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

START, DEV_END, HOLDOUT_START, END = 20230523, 20241231, 20250101, 20251231
INIT, FEE, TAX, MAX_POS, SLOT = 1_300_000.0, 0.000855, 0.003, 4, 0.24
parser = argparse.ArgumentParser(); parser.add_argument('--hypothesis', default='all'); args = parser.parse_args()
OUT = Path('research_out_12_stock_setups') / args.hypothesis; OUT.mkdir(parents=True, exist_ok=True)

px = pd.read_csv('formal_run/ohlcv_causal_2020_2025.csv.gz', dtype={'code':str}, low_memory=False)
px['code']=px.code.astype(str).str.zfill(4); px=px.sort_values(['code','date']).reset_index(drop=True)
inst=pd.read_parquet('formal_run/institutional_2020_2025.parquet'); inst['code']=inst.code.astype(str).str.zfill(4)
inst['date']=pd.to_datetime(inst.date).dt.strftime('%Y%m%d').astype(int) if not np.issubdtype(inst.date.dtype, np.integer) else inst.date.astype(int)
inst=inst.sort_values(['code','date']).copy()

px['amount']=px.close*px.volume; g=px.groupby('code',group_keys=False)
for w in (1,2,3,5,10,20,60,120): px[f'r{w}']=g.aclose.transform(lambda s,w=w:s.pct_change(w))
for w in (5,10,20,60,120): px[f'ma{w}']=g.aclose.transform(lambda s,w=w:s.rolling(w,min_periods=w).mean())
px['amount20']=g.amount.transform(lambda s:s.rolling(20,min_periods=20).mean())
px['amount5']=g.amount.transform(lambda s:s.rolling(5,min_periods=5).mean())
px['amount_ratio']=px.amount/px.amount20.replace(0,np.nan)
px['amount5_ratio']=px.amount5/px.amount20.replace(0,np.nan)
px['prior_high20']=g.aclose.transform(lambda s:s.shift(1).rolling(20,min_periods=20).max())
px['prior_high60']=g.aclose.transform(lambda s:s.shift(1).rolling(60,min_periods=60).max())
px['prior_low20']=g.aclose.transform(lambda s:s.shift(1).rolling(20,min_periods=20).min())
px['high20']=g.aclose.transform(lambda s:s.rolling(20,min_periods=20).max())
px['draw20']=px.aclose/px.high20-1
px['vol10']=g.aclose.transform(lambda s:s.pct_change().rolling(10,min_periods=10).std())
px['vol20']=g.aclose.transform(lambda s:s.pct_change().rolling(20,min_periods=20).std())
px['vol60']=g.aclose.transform(lambda s:s.pct_change().rolling(60,min_periods=60).std())
px['vol_compress']=px.vol10/px.vol60.replace(0,np.nan)
px['dist_ma20']=px.aclose/px.ma20-1; px['dist_ma60']=px.aclose/px.ma60-1
px['prev_close']=g.aclose.shift(1); px['prev_ma20']=g.ma20.shift(1)
px['upday']=px.r1>0
px['up_ratio10']=g.upday.transform(lambda s:s.rolling(10,min_periods=10).mean())

gi=inst.groupby('code',group_keys=False)
for w in (3,5,10,20):
    inst[f'foreign{w}']=gi.foreign_net.transform(lambda s,w=w:s.rolling(w,min_periods=w).sum())
    inst[f'trust{w}']=gi.trust_net.transform(lambda s,w=w:s.rolling(w,min_periods=w).sum())
    inst[f'inst{w}']=inst[f'foreign{w}']+inst[f'trust{w}']
inst['inst_prev5']=gi[['foreign_net','trust_net']].transform(lambda s:s.shift(5).rolling(5,min_periods=5).sum()).sum(axis=1)
inst['inst_accel']=inst.inst5-inst.inst_prev5
inst['foreign_trust_agree']=((inst.foreign5>0)&(inst.trust5>0)).astype(int)
px=px.merge(inst[['date','code','inst3','inst5','inst10','inst20','inst_prev5','inst_accel','foreign_trust_agree']],on=['date','code'],how='left')
px[['inst3','inst5','inst10','inst20','inst_prev5','inst_accel','foreign_trust_agree']]=px[['inst3','inst5','inst10','inst20','inst_prev5','inst_accel','foreign_trust_agree']].fillna(0)
for w in (3,5,10,20): px[f'flow{w}_to_amount']=(px[f'inst{w}']*px.close)/(px.amount20*w).replace(0,np.nan)
px['flow_accel_value']=(px.inst_accel*px.close)/(px.amount20*5).replace(0,np.nan)

valid=(px.code.str.fullmatch(r'[1-9]\d{3}') & (~px.name.astype(str).str.contains('KY',case=False,na=False)) & (px.close>=5) & (px.amount20>=30_000_000))
stocks=px[valid].copy()
rank_cols=['r2','r3','r5','r10','r20','r60','r120','amount20','amount_ratio','amount5_ratio','flow3_to_amount','flow5_to_amount','flow10_to_amount','flow20_to_amount','flow_accel_value','vol20','vol_compress','up_ratio10']
for c in rank_cols: stocks[c+'_pr']=stocks.groupby('date')[c].rank(pct=True)

ctx=stocks.groupby('date').agg(
    breadth20=('dist_ma20',lambda s:float((s>0).mean())),
    breadth60=('dist_ma60',lambda s:float((s>0).mean())),
    med_r20=('r20','median'), med_r60=('r60','median'), dispersion20=('r20','std'),
    flow_breadth=('flow5_to_amount',lambda s:float((s>0).mean())), flow_dispersion=('flow5_to_amount','std'),
    liquidity_expansion=('amount_ratio',lambda s:float((s>1.2).mean()))
).reset_index()
stocks=stocks.merge(ctx,on='date',how='left')

def quality(d):
    return (d.close>=10)&(d.amount20>=100_000_000)&(d.vol20_pr<=0.97)&(d.r120_pr>=0.15)

SETUPS={
'breakout_flow_confirmation':dict(hold=8, cond=lambda d:(d.aclose>d.prior_high20)&(d.flow5_to_amount>0)&(d.amount_ratio>1.05), score=lambda d:.30*d.r20_pr+.30*d.flow5_to_amount_pr+.25*d.amount_ratio_pr+.15*d.r60_pr),
'prebreakout_flow_acceleration':dict(hold=8, cond=lambda d:(d.aclose<d.prior_high20)&(d.aclose>d.ma20)&(d.flow_accel_value>0)&(d.r20>-0.03), score=lambda d:.40*d.flow_accel_value_pr+.25*d.r20_pr+.20*d.amount_ratio_pr+.15*d.r60_pr),
'persistent_sponsorship_trend':dict(hold=12, cond=lambda d:(d.ma20>d.ma60)&(d.inst10>0)&(d.inst20>0)&(d.dist_ma20<.12), score=lambda d:.35*d.flow20_to_amount_pr+.25*d.flow10_to_amount_pr+.25*d.r60_pr+.15*d.amount20_pr),
'fresh_sponsorship_turn':dict(hold=7, cond=lambda d:(d.inst_prev5<=0)&(d.inst5>0)&(d.aclose>d.ma20)&(d.prev_close<=d.prev_ma20), score=lambda d:.45*d.flow_accel_value_pr+.25*d.amount_ratio_pr+.20*d.r5_pr+.10*d.r20_pr),
'foreign_trust_agreement':dict(hold=10, cond=lambda d:(d.foreign_trust_agree>0)&(d.aclose>d.ma20)&(d.r20>-0.05), score=lambda d:.35*d.flow5_to_amount_pr+.30*d.flow20_to_amount_pr+.20*d.r20_pr+.15*d.amount20_pr),
'compression_release':dict(hold=8, cond=lambda d:(d.vol_compress<.75)&(d.aclose>d.ma20)&(d.r3>0)&(d.amount_ratio>1), score=lambda d:.30*(1-d.vol_compress_pr)+.30*d.amount_ratio_pr+.25*d.r5_pr+.15*d.flow5_to_amount_pr),
'leader_pullback_absorption':dict(hold=7, cond=lambda d:(d.r60_pr>.70)&(d.draw20<-.02)&(d.draw20>-.12)&(d.aclose>d.ma60)&(d.r2>-.05), score=lambda d:.35*d.r60_pr+.25*d.flow5_to_amount_pr+.20*(1-d.vol20_pr)+.20*d.amount20_pr),
'leader_reacceleration':dict(hold=8, cond=lambda d:(d.r60_pr>.70)&(d.r10>0)&(d.r3>d.r10/4)&(d.aclose>d.ma20), score=lambda d:.30*d.r60_pr+.30*d.r3_pr+.20*d.amount_ratio_pr+.20*d.flow5_to_amount_pr),
'liquidity_repricing':dict(hold=6, cond=lambda d:(d.amount5_ratio>1.15)&(d.r5>0)&(d.aclose>d.ma20), score=lambda d:.35*d.amount5_ratio_pr+.25*d.r5_pr+.20*d.flow5_to_amount_pr+.20*d.r20_pr),
'quiet_accumulation':dict(hold=12, cond=lambda d:(d.flow20_to_amount>0)&(d.amount_ratio<1.3)&(d.vol20_pr<.65)&(d.aclose>d.ma20), score=lambda d:.40*d.flow20_to_amount_pr+.25*(1-d.vol20_pr)+.20*d.r20_pr+.15*d.amount20_pr),
'oversold_leader_reclaim':dict(hold=6, cond=lambda d:(d.r120_pr>.60)&(d.draw20<-.08)&(d.aclose>d.ma20)&(d.prev_close<=d.prev_ma20), score=lambda d:.35*d.r120_pr+.30*d.flow_accel_value_pr+.20*d.r3_pr+.15*d.amount_ratio_pr),
'flow_divergence_reversal':dict(hold=7, cond=lambda d:(d.r10<0)&(d.flow5_to_amount>0)&(d.flow_accel_value>0)&(d.aclose>d.ma60), score=lambda d:.40*d.flow_accel_value_pr+.25*d.flow5_to_amount_pr+.20*(1-d.r10_pr)+.15*d.amount20_pr),
}
if args.hypothesis!='all':
    if args.hypothesis not in SETUPS: raise SystemExit(f'unknown hypothesis {args.hypothesis}')
    SETUPS={args.hypothesis:SETUPS[args.hypothesis]}

def tick(p):
    return .01 if p<10 else .05 if p<50 else .1 if p<100 else .5 if p<500 else 1.0 if p<1000 else 5.0
def ceil_tick(p): t=tick(p); return math.ceil((p-1e-12)/t)*t
def floor_tick(p): t=tick(p); return math.floor((p+1e-12)/t)*t

stocks=stocks.sort_values(['date','code']).copy(); bydate={int(d):q.copy() for d,q in stocks.groupby('date')}
calendar=sorted(int(x) for x in stocks[(stocks.date>=START)&(stocks.date<=END)].date.unique()); cal_i={d:i for i,d in enumerate(calendar)}
rowmap={(int(r.date),r.code):r for r in stocks.itertuples(index=False)}
orders={k:{} for k in SETUPS}; sig=[]
for d in calendar[:-1]:
    day=bydate[d]; entry=calendar[cal_i[d]+1]
    for name,spec in SETUPS.items():
        raw=day[spec['cond'](day).fillna(False)].copy()
        if raw.empty: continue
        raw['score']=spec['score'](raw)
        q=raw[quality(raw).fillna(False)].sort_values(['score','amount20'],ascending=False).head(8)
        for r in q.itertuples(index=False):
            item={'code':r.code,'signal_date':d,'entry_date':entry,'hold':spec['hold'],'score':float(r.score)}
            orders[name].setdefault(entry,[]).append(item); sig.append({'setup':name,**item})
pd.DataFrame(sig).to_csv(OUT/'signal_ledger.csv',index=False)
signal_counts={k:sum(len(v) for v in orders[k].values()) for k in SETUPS}
(OUT/'signal_counts.json').write_text(json.dumps(signal_counts,indent=2),encoding='utf-8')

def simulate(setup,start,end,label):
    cash=INIT; pos={}; trades=[]; navrows=[]; dates=[d for d in calendar if start<=d<=end]
    for d in dates:
        for code,p in list(pos.items()):
            rr=rowmap.get((d,code))
            if rr is not None and bool(getattr(rr,'is_official_event',False)):
                sf=float(getattr(rr,'share_factor',1.0))
                if np.isfinite(sf) and sf>0 and abs(sf-1)>1e-12: p['shares']=int(round(p['shares']*sf))
        for code,p in list(pos.items()):
            if p['exit_date']!=d: continue
            rr=rowmap.get((d,code))
            if rr is None or not np.isfinite(rr.open) or rr.open<=0: continue
            xp=floor_tick(float(rr.open)*.995); cash+=p['shares']*xp*(1-FEE-TAX)
            ret=(xp*(1-FEE-TAX))/(p['entry_price']*(1+FEE))-1
            trades.append({'setup':setup,'code':code,'signal_date':p['signal_date'],'entry_date':p['entry_date'],'exit_date':d,'entry_price':p['entry_price'],'exit_price':xp,'shares':p['shares'],'return_net':ret}); del pos[code]
        for item in sorted(orders[setup].get(d,[]),key=lambda x:x['score'],reverse=True):
            if len(pos)>=MAX_POS: break
            code=item['code']
            if code in pos: continue
            rr=rowmap.get((d,code))
            if rr is None or not np.isfinite(rr.open) or rr.open<=0 or bool(getattr(rr,'is_official_event',False)): continue
            j=cal_i[d]+item['hold']
            if j>=len(calendar): continue
            ep=ceil_tick(float(rr.open)*1.005); mv=sum(p['shares']*float(rowmap[(d,c)].close) for c,p in pos.items() if (d,c) in rowmap and np.isfinite(rowmap[(d,c)].close))
            budget=min(cash,(cash+mv)*SLOT); shares=int(budget//(ep*(1+FEE)))
            if shares<=0: continue
            cost=shares*ep*(1+FEE)
            if cost>cash+1e-9: continue
            cash-=cost
            if cash<-.01: raise RuntimeError('negative cash')
            pos[code]={'shares':shares,'entry_price':ep,'entry_date':d,'signal_date':item['signal_date'],'exit_date':calendar[j]}
        nav=cash+sum(p['shares']*float(rowmap[(d,c)].close) for c,p in pos.items() if (d,c) in rowmap and np.isfinite(rowmap[(d,c)].close)); navrows.append({'date':d,'nav':nav,'cash':cash,'positions':len(pos)})
    last=dates[-1]
    for code,p in list(pos.items()):
        rr=rowmap.get((last,code))
        if rr is None or not np.isfinite(rr.close) or rr.close<=0: continue
        xp=floor_tick(float(rr.close)*.995); cash+=p['shares']*xp*(1-FEE-TAX); ret=(xp*(1-FEE-TAX))/(p['entry_price']*(1+FEE))-1
        trades.append({'setup':setup,'code':code,'signal_date':p['signal_date'],'entry_date':p['entry_date'],'exit_date':last,'entry_price':p['entry_price'],'exit_price':xp,'shares':p['shares'],'return_net':ret,'forced_exit':True}); del pos[code]
    nav=pd.DataFrame(navrows); nav.loc[nav.index[-1],['nav','cash','positions']]=[cash,cash,0]; t=pd.DataFrame(trades)
    years=max((pd.to_datetime(str(end))-pd.to_datetime(str(start))).days/365.25,1/252); cagr=(cash/INIT)**(1/years)-1; dd=float((nav.nav/nav.nav.cummax()-1).min())
    wr=float((t.return_net>0).mean()) if len(t) else 0.; gp=float(t.loc[t.return_net>0,'return_net'].sum()) if len(t) else 0.; gl=float(-t.loc[t.return_net<0,'return_net'].sum()) if len(t) else 0.; pf=gp/gl if gl>0 else float('inf')
    if cash<-.01: raise RuntimeError('negative final cash')
    nav.to_csv(OUT/f'nav_{setup}_{label}.csv',index=False); t.to_csv(OUT/f'trades_{setup}_{label}.csv',index=False)
    return {'setup':setup,'label':label,'start':start,'end':end,'end_nav':float(cash),'cagr':float(cagr),'max_drawdown':dd,'trades':len(t),'wins':int((t.return_net>0).sum()) if len(t) else 0,'win_rate':wr,'profit_factor':pf,'signals':signal_counts[setup]}

results=[]; yearly=[]
for setup in SETUPS:
    for s,e,l in [(START,DEV_END,'dev'),(HOLDOUT_START,END,'holdout_2025'),(START,END,'full')]: results.append(simulate(setup,s,e,l))
    for y,s,e in [(2023,20230523,20231229),(2024,20240102,20241231),(2025,20250102,20251231)]:
        r=simulate(setup,s,e,f'year_{y}'); r['year']=y; yearly.append(r)
res=pd.DataFrame(results); yr=pd.DataFrame(yearly); res.to_csv(OUT/'portfolio_summary.csv',index=False); yr.to_csv(OUT/'yearly_summary.csv',index=False)
dev=res[res.label=='dev'].copy(); dev['valid_sample']=dev.trades>=25; dev['dev_score']=np.where(dev.valid_sample,2*dev.win_rate+1.2*np.clip(dev.cagr,-1,1)+.3*np.clip(dev.profit_factor,0,3)+.3*np.clip(dev.max_drawdown,-1,0),-999)
dev_rank=dev.sort_values(['dev_score','trades'],ascending=False)[['setup','dev_score','trades','win_rate','cagr','profit_factor','max_drawdown']]
dev_rank.to_csv(OUT/'dev_rank.csv',index=False)
minute=False
for setup in SETUPS:
    d=res[(res.setup==setup)&(res.label=='dev')].iloc[0]; h=res[(res.setup==setup)&(res.label=='holdout_2025')].iloc[0]
    if d.trades>=25 and h.trades>=25 and d.cagr>0 and h.cagr>0 and d.profit_factor>1.1 and h.profit_factor>1.2 and h.win_rate>=.55: minute=True
(OUT/'decision.json').write_text(json.dumps({'hypothesis':args.hypothesis,'signal_counts':signal_counts,'minute_optimization_allowed':minute,'fundamental_filter_present':False,'formal_r10_untouched':True,'selection_uses_2025':False},indent=2),encoding='utf-8')
print(res.to_string(index=False)); print(dev_rank.to_string(index=False)); print(json.dumps(signal_counts,indent=2))
