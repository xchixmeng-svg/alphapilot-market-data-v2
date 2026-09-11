from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path('research_out_market_playbook'); OUT.mkdir(exist_ok=True)
START=20230523; DEV_END=20241231; HOLDOUT_START=20250101; END=20251231
INIT=1_300_000.0; FEE=0.000855; TAX=0.003; MAX_POS=4; SLOT=0.24

# ---------- load frozen causal daily layer ----------
px=pd.read_csv('formal_run/ohlcv_causal_2020_2025.csv.gz',dtype={'code':str},low_memory=False)
px['code']=px.code.astype(str).str.zfill(4)
px=px.sort_values(['code','date']).reset_index(drop=True)
inst=pd.read_parquet('formal_run/institutional_2020_2025.parquet')
inst['code']=inst.code.astype(str).str.zfill(4)
if pd.api.types.is_datetime64_any_dtype(inst.date):
    inst['date']=inst.date.dt.strftime('%Y%m%d').astype(int)
else:
    inst['date']=pd.to_datetime(inst.date).dt.strftime('%Y%m%d').astype(int)
inst=inst.sort_values(['code','date']).copy()

# ---------- stock features, all known by T close ----------
px['amount']=px['close']*px['volume']
g=px.groupby('code',group_keys=False)
for w in (1,5,10,20,60,120):
    px[f'r{w}']=g['aclose'].transform(lambda s,w=w:s.pct_change(w))
for w in (20,60,120):
    px[f'ma{w}']=g['aclose'].transform(lambda s,w=w:s.rolling(w,min_periods=w).mean())
px['amount20']=g['amount'].transform(lambda s:s.rolling(20,min_periods=20).mean())
px['prior_high20']=g['aclose'].transform(lambda s:s.shift(1).rolling(20,min_periods=20).max())
px['prior_high60']=g['aclose'].transform(lambda s:s.shift(1).rolling(60,min_periods=60).max())
px['high20']=g['aclose'].transform(lambda s:s.rolling(20,min_periods=20).max())
px['draw20']=px.aclose/px.high20-1.0
px['ret_vol20']=g['aclose'].transform(lambda s:s.pct_change().rolling(20,min_periods=20).std())
px['range_pct']=(px.high-px.low)/px.close.replace(0,np.nan)
px['range20']=g['range_pct'].transform(lambda s:s.rolling(20,min_periods=20).mean())
px['prev_aclose']=g.aclose.shift(1); px['prev_ma20']=g.ma20.shift(1)

gi=inst.groupby('code',group_keys=False)
for w in (5,20):
    inst[f'foreign{w}']=gi.foreign_net.transform(lambda s,w=w:s.rolling(w,min_periods=w).sum())
    inst[f'trust{w}']=gi.trust_net.transform(lambda s,w=w:s.rolling(w,min_periods=w).sum())
inst['inst5']=inst.foreign5+inst.trust5
inst['inst20']=inst.foreign20+inst.trust20
px=px.merge(inst[['date','code','inst5','inst20']],on=['date','code'],how='left')
px[['inst5','inst20']]=px[['inst5','inst20']].fillna(0.0)
px['flow_value20']=px.inst20*px.close
px['flow_to_amount']=px.flow_value20/(px.amount20*20).replace(0,np.nan)

# ---------- market state built BEFORE stock universe filter ----------
m=px[px.code=='0050'][['date','aclose']].drop_duplicates('date').sort_values('date').copy()
for w in (20,60,120): m[f'ma{w}']=m.aclose.rolling(w,min_periods=w).mean()
m['r20']=m.aclose.pct_change(20);m['r60']=m.aclose.pct_change(60)
m['vol20']=m.aclose.pct_change().rolling(20,min_periods=20).std()*np.sqrt(252)
m['high60']=m.aclose.rolling(60,min_periods=60).max();m['dd60']=m.aclose/m.high60-1

base_valid=(px.code.str.fullmatch(r'[1-9]\d{3}')) & (~px.name.astype(str).str.contains('KY',case=False,na=False)) & (px.amount20>=30_000_000)
stocks=px[base_valid].copy()
stocks['above60']=stocks.aclose>stocks.ma60
breadth=stocks.groupby('date').above60.mean().rename('breadth60')
m=m.merge(breadth,on='date',how='left');m['breadth_chg10']=m.breadth60-m.breadth60.shift(10)

def state_row(r):
    if not np.isfinite(r.ma120) or not np.isfinite(r.breadth60): return 'UNKNOWN'
    if r.aclose<r.ma120 and r.r20<0 and r.r60<0 and r.breadth60<0.35: return 'BEAR'
    if r.aclose>r.ma60 and r.ma60>r.ma120 and r.r20>0.05 and r.r60>0.10 and r.breadth60>=0.60: return 'STRONG_BULL'
    if r.aclose>r.ma60 and r.r20>0 and r.r60>0 and r.breadth60>=0.48: return 'BULL'
    if r.r20>0 and r.breadth_chg10>0.08 and (r.dd60<=-0.05 or r.r60<=0): return 'REPAIR'
    if abs(r.r20)<=0.05 and 0.38<=r.breadth60<=0.62: return 'RANGE'
    if r.aclose<r.ma60 or r.breadth60<0.40: return 'WEAK'
    return 'RANGE'
m['state']=m.apply(state_row,axis=1)
state_map=dict(zip(m.date,m.state))

# ---------- cross-sectional stock ranks ----------
for c in ['r5','r20','r60','r120','flow_to_amount','ret_vol20','amount20']:
    stocks[c+'_pr']=stocks.groupby('date')[c].rank(pct=True)

# ---------- playbooks: market chooses method FIRST ----------
# Each playbook has its own candidate screen + fixed holding horizon for stage-1 discovery.
PLAYBOOKS={
 'breakout': {
   'hold':8,
   'cond':lambda d:(d.aclose>d.prior_high20)&(d.r20_pr>=.72)&(d.r5>0)&(d.aclose>d.ma60),
   'score':lambda d:.40*d.r20_pr+.25*d.r5_pr+.20*d.flow_to_amount_pr+.15*d.amount20_pr},
 'trend_pullback': {
   'hold':10,
   'cond':lambda d:(d.ma20>d.ma60)&(d.aclose>d.ma60)&(d.r60_pr>=.65)&(d.draw20<=-.02)&(d.draw20>=-.10),
   'score':lambda d:.40*d.r60_pr+.25*d.flow_to_amount_pr+.20*(1-d.ret_vol20_pr)+.15*d.amount20_pr},
 'relative_strength': {
   'hold':10,
   'cond':lambda d:(d.r20_pr>=.82)&(d.r60_pr>=.70)&(d.aclose>d.ma20),
   'score':lambda d:.40*d.r20_pr+.30*d.r60_pr+.20*d.flow_to_amount_pr+.10*d.amount20_pr},
 'mean_reversion': {
   'hold':6,
   'cond':lambda d:(d.aclose>d.ma120)&(d.draw20<=-.05)&(d.draw20>=-.14)&(d.r5<0)&(d.r120_pr>=.45),
   'score':lambda d:.30*d.r120_pr+.25*d.flow_to_amount_pr+.25*(1-d.ret_vol20_pr)+.20*d.amount20_pr},
 'repair_reversal': {
   'hold':8,
   'cond':lambda d:(d.aclose>d.ma20)&(d.prev_aclose<=d.prev_ma20)&(d.r20>-0.10)&(d.flow_to_amount_pr>=.55),
   'score':lambda d:.35*d.flow_to_amount_pr+.25*d.r20_pr+.20*d.r60_pr+.20*d.amount20_pr},
}

# Second-stage quality filter: candidates are created by playbook first; only then quality removes weak names.
def quality_mask(d,tier):
    common=(d.close>=10)&(d.amount20>=50_000_000)&(d.ret_vol20_pr<=.95)
    if tier=='loose': return common
    if tier=='balanced':
        return common&(d.amount20>=100_000_000)&(d.r120_pr>=.35)&(d.ret_vol20_pr<=.85)&((d.inst20>0)|(d.flow_to_amount_pr>=.50))
    if tier=='strict':
        return common&(d.amount20>=200_000_000)&(d.aclose>d.ma120)&(d.r120_pr>=.50)&(d.ret_vol20_pr<=.72)&(d.inst20>0)
    raise ValueError(tier)
QUALITY=['loose','balanced','strict']

# Legal tick helpers.
def tick(p):
    if p<10:return .01
    if p<50:return .05
    if p<100:return .1
    if p<500:return .5
    if p<1000:return 1.0
    return 5.0
def ceil_tick(p):
    t=tick(p);return math.ceil((p-1e-12)/t)*t
def floor_tick(p):
    t=tick(p);return math.floor((p+1e-12)/t)*t

stocks=stocks.sort_values(['date','code']).copy()
bydate={int(d):q.copy() for d,q in stocks.groupby('date')}
calendar=sorted(int(x) for x in m[(m.date>=START)&(m.date<=END)].date)
cal_i={d:i for i,d in enumerate(calendar)}
rowmap={(int(r.date),r.code):r for r in stocks.itertuples(index=False)}

# Build independent candidate-event returns. T-close signal -> T+1 open entry; no same-day hindsight.
events=[]
for d in calendar:
    if d not in bydate: continue
    state=state_map.get(d,'UNKNOWN')
    if state=='UNKNOWN': continue
    i=cal_i[d]
    if i+1>=len(calendar): continue
    entry_date=calendar[i+1]
    day=bydate[d]
    for pb,spec in PLAYBOOKS.items():
        raw=day[spec['cond'](day).fillna(False)].copy()
        if raw.empty: continue
        raw['score']=spec['score'](raw)
        for tier in QUALITY:
            q=raw[quality_mask(raw,tier).fillna(False)].sort_values('score',ascending=False).head(6)
            for r in q.itertuples(index=False):
                er=rowmap.get((entry_date,r.code))
                if er is None or not np.isfinite(er.open) or er.open<=0: continue
                if bool(getattr(er,'is_official_event',False)): continue
                ei=cal_i[entry_date]+int(spec['hold'])
                if ei>=len(calendar): continue
                exit_date=calendar[ei]; xr=rowmap.get((exit_date,r.code))
                if xr is None or not np.isfinite(xr.open) or xr.open<=0: continue
                ep=ceil_tick(float(er.open)*1.005); xp=floor_tick(float(xr.open)*0.995)
                net=(xp*(1-FEE-TAX))/(ep*(1+FEE))-1
                events.append({'signal_date':d,'entry_date':entry_date,'exit_date':exit_date,'state':state,'playbook':pb,'quality':tier,'code':r.code,'score':float(r.score),'hold':int(spec['hold']),'entry_price':ep,'exit_price':xp,'return_net':net})
ev=pd.DataFrame(events)
if ev.empty: raise RuntimeError('no candidate events generated')
ev.to_csv(OUT/'candidate_events.csv',index=False)

# Development selection: choose playbook+quality independently for every market state.
# No R10 trade-count target. CASH is allowed if no method is credible.
summary=[]
for (st,pb,qt),q in ev[ev.entry_date<=DEV_END].groupby(['state','playbook','quality']):
    n=len(q); wr=float((q.return_net>0).mean()); mean=float(q.return_net.mean()); med=float(q.return_net.median())
    p10=float(q.return_net.quantile(.10)); score=wr*2.0+mean*8.0+med*2.0+min(n,60)/300.0
    summary.append({'state':st,'playbook':pb,'quality':qt,'n':n,'win_rate':wr,'mean_return':mean,'median_return':med,'p10':p10,'selection_score':score})
ss=pd.DataFrame(summary)
ss.to_csv(OUT/'dev_state_method_grid.csv',index=False)
chosen={}
for st in ['STRONG_BULL','BULL','REPAIR','RANGE','WEAK','BEAR']:
    q=ss[(ss.state==st)&(ss.n>=12)&(ss.mean_return>0)].copy()
    if q.empty:
        chosen[st]={'playbook':'CASH','quality':'none','reason':'no_dev_method_with_n12_positive_mean'}
    else:
        # Prefer >=60% dev win-rate first, then best composite; otherwise best positive method.
        good=q[q.win_rate>=.60]
        pick=(good if len(good) else q).sort_values(['selection_score','n'],ascending=False).iloc[0]
        chosen[st]={'playbook':pick.playbook,'quality':pick.quality,'dev_n':int(pick.n),'dev_win_rate':float(pick.win_rate),'dev_mean_return':float(pick.mean_return),'dev_median_return':float(pick.median_return),'selection_score':float(pick.selection_score)}
(OUT/'chosen_market_policy.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2),encoding='utf-8')

# ---------- full portfolio simulator using ONLY the frozen dev-selected mapping ----------
def run_portfolio(start,end,label):
    cash=INIT; pos={}; trades=[]; navrows=[]; pending={}
    dates=[d for d in calendar if start<=d<=end]
    if not dates:return None
    for d in dates:
        # official share-factor adjustment before any execution/valuation
        for code,p in list(pos.items()):
            rr=rowmap.get((d,code))
            if rr is not None and bool(getattr(rr,'is_official_event',False)) and np.isfinite(rr.share_factor) and float(rr.share_factor)>0 and abs(float(rr.share_factor)-1)>1e-12:
                p['shares']=int(round(p['shares']*float(rr.share_factor)))
        # exits
        for code,p in list(pos.items()):
            if p['exit_date']!=d: continue
            rr=rowmap.get((d,code))
            if rr is None or not np.isfinite(rr.open): continue
            xp=floor_tick(float(rr.open)*0.995); gross=p['shares']*xp; cash+=gross*(1-FEE-TAX)
            ret=(xp*(1-FEE-TAX))/(p['entry_price']*(1+FEE))-1
            trades.append({'label':label,'state':p['state'],'playbook':p['playbook'],'quality':p['quality'],'code':code,'entry_date':p['entry_date'],'exit_date':d,'entry_price':p['entry_price'],'exit_price':xp,'shares':p['shares'],'return_net':ret})
            del pos[code]
        # entries locked from prior close
        for item in pending.pop(d,[]):
            if len(pos)>=MAX_POS: break
            code=item['code']
            if code in pos: continue
            rr=rowmap.get((d,code))
            if rr is None or not np.isfinite(rr.open) or rr.open<=0 or bool(getattr(rr,'is_official_event',False)): continue
            ci=calendar.index(d)+item['hold']
            if ci>=len(calendar): continue
            exit_date=calendar[ci]
            ep=ceil_tick(float(rr.open)*1.005)
            mv=sum(p['shares']*float(rowmap[(d,c)].open) for c,p in pos.items() if (d,c) in rowmap and np.isfinite(rowmap[(d,c)].open))
            nav_open=cash+mv; budget=min(cash,nav_open*SLOT); sh=int(budget//(ep*(1+FEE)))
            if sh<=0: continue
            cost=sh*ep*(1+FEE)
            if cost>cash+1e-9: continue
            cash-=cost
            pos[code]={'shares':sh,'entry_price':ep,'entry_date':d,'exit_date':exit_date,'state':item['state'],'playbook':item['playbook'],'quality':item['quality']}
        # mark close nav
        mv=0.0
        for code,p in pos.items():
            rr=rowmap.get((d,code))
            if rr is not None and np.isfinite(rr.close): mv+=p['shares']*float(rr.close)
        navrows.append({'date':d,'nav':cash+mv})
        # signal after close: market state -> playbook -> screen -> quality -> top candidates
        st=state_map.get(d,'UNKNOWN'); policy=chosen.get(st,{'playbook':'CASH'})
        if policy.get('playbook')=='CASH': continue
        if d not in bydate: continue
        i=calendar.index(d)
        if i+1>=len(calendar): continue
        nd=calendar[i+1]
        pb=policy['playbook']; qt=policy['quality']; spec=PLAYBOOKS[pb]; day=bydate[d]
        raw=day[spec['cond'](day).fillna(False)].copy()
        if raw.empty: continue
        raw['score']=spec['score'](raw)
        q=raw[quality_mask(raw,qt).fillna(False)].sort_values('score',ascending=False).head(6)
        pending[nd]=[{'code':r.code,'score':float(r.score),'hold':int(spec['hold']),'state':st,'playbook':pb,'quality':qt} for r in q.itertuples(index=False)]
    # residual positions liquidated at final close, preserving cash accounting
    last=dates[-1]
    for code,p in list(pos.items()):
        rr=rowmap.get((last,code))
        if rr is None or not np.isfinite(rr.close): continue
        xp=floor_tick(float(rr.close)*0.995);cash+=p['shares']*xp*(1-FEE-TAX)
        ret=(xp*(1-FEE-TAX))/(p['entry_price']*(1+FEE))-1
        trades.append({'label':label,'state':p['state'],'playbook':p['playbook'],'quality':p['quality'],'code':code,'entry_date':p['entry_date'],'exit_date':last,'entry_price':p['entry_price'],'exit_price':xp,'shares':p['shares'],'return_net':ret})
    nav=pd.DataFrame(navrows); t=pd.DataFrame(trades)
    if len(nav):
        nav.loc[nav.index[-1],'nav']=cash
        dd=float((nav.nav/nav.nav.cummax()-1).min())
    else: dd=0.0
    yrs=max((pd.to_datetime(str(end))-pd.to_datetime(str(start))).days/365.25,1/365.25)
    cagr=(cash/INIT)**(1/yrs)-1
    wr=float((t.return_net>0).mean()) if len(t) else 0.0
    gp=float(t.loc[t.return_net>0,'return_net'].sum()) if len(t) else 0.0; gl=float(-t.loc[t.return_net<0,'return_net'].sum()) if len(t) else 0.0
    pf=gp/gl if gl>0 else float('inf')
    # accounting invariant: final cash equals reported final nav after forced liquidation
    if abs(float(nav.iloc[-1].nav)-cash)>0.01: raise RuntimeError(f'cash/nav mismatch {label}')
    nav.to_csv(OUT/f'nav_{label}.csv',index=False); t.to_csv(OUT/f'trades_{label}.csv',index=False)
    return {'label':label,'start':start,'end':end,'end_nav':float(cash),'cagr':float(cagr),'max_drawdown':dd,'trades':int(len(t)),'wins':int((t.return_net>0).sum()) if len(t) else 0,'win_rate':wr,'profit_factor':pf}

res=[run_portfolio(START,DEV_END,'dev_2023m5_2024'),run_portfolio(HOLDOUT_START,END,'holdout_2025'),run_portfolio(START,END,'full_2023m5_2025')]
res=[x for x in res if x]
pd.DataFrame(res).to_csv(OUT/'portfolio_summary.csv',index=False)
full=next(x for x in res if x['label']=='full_2023m5_2025')
hold=next(x for x in res if x['label']=='holdout_2025')
decision={'status':'PASS','architecture':'market_state -> entry_playbook -> candidate_screen -> quality_filter -> T+1 execution','research_window':[START,END],'dev_window':[START,DEV_END],'holdout_window':[HOLDOUT_START,END],'target':{'cagr':0.50,'win_rate':0.70},'chosen_policy':chosen,'full':full,'holdout':hold,'target_hit_full':bool(full['cagr']>=.50 and full['win_rate']>=.70),'target_hit_holdout':bool(hold['cagr']>=.50 and hold['win_rate']>=.70),'next_stage':'minute-bar entry refinement only after this architecture produces valid state/playbook candidates','formal_r10_untouched':True}
(OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print('CHOSEN_POLICY',json.dumps(chosen,ensure_ascii=False))
print(pd.DataFrame(res).to_string(index=False))
print(json.dumps(decision,ensure_ascii=False,indent=2,default=str))
