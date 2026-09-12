from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path('research_out_continuous_context'); OUT.mkdir(exist_ok=True)
START=20230523; DEV_END=20241231; HOLDOUT_START=20250101; END=20251231
INIT=1_300_000.0; FEE=0.000855; TAX=0.003; MAX_POS=4; SLOT=0.24
MIN_HISTORY_DAYS=80; K_NEIGHBORS=45; MIN_EVENTS=24

# ---------- frozen causal daily layer ----------
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

# ---------- stock features: every field known by T close ----------
px['amount']=px['close']*px['volume']
g=px.groupby('code',group_keys=False)
for w in (1,5,10,20,60,120):
    px[f'r{w}']=g['aclose'].transform(lambda s,w=w:s.pct_change(w))
for w in (20,60,120):
    px[f'ma{w}']=g['aclose'].transform(lambda s,w=w:s.rolling(w,min_periods=w).mean())
px['amount20']=g['amount'].transform(lambda s:s.rolling(20,min_periods=20).mean())
px['amount_ratio']=px['amount']/px['amount20'].replace(0,np.nan)
px['prior_high20']=g['aclose'].transform(lambda s:s.shift(1).rolling(20,min_periods=20).max())
px['prior_low20']=g['aclose'].transform(lambda s:s.shift(1).rolling(20,min_periods=20).min())
px['high20']=g['aclose'].transform(lambda s:s.rolling(20,min_periods=20).max())
px['draw20']=px.aclose/px.high20-1.0
px['ret_vol20']=g['aclose'].transform(lambda s:s.pct_change().rolling(20,min_periods=20).std())
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

base_valid=(px.code.str.fullmatch(r'[1-9]\d{3}')) & (~px.name.astype(str).str.contains('KY',case=False,na=False)) & (px.amount20>=30_000_000)
stocks=px[base_valid].copy()
for c in ['r5','r20','r60','r120','flow_to_amount','ret_vol20','amount20']:
    stocks[c+'_pr']=stocks.groupby('date')[c].rank(pct=True)

# ---------- continuous market context; NO bull/bear semantic labels ----------
etf=px[px.code=='0050'][['date','aclose','amount']].drop_duplicates('date').sort_values('date').copy()
for w in (20,60,120): etf[f'ma{w}']=etf.aclose.rolling(w,min_periods=w).mean()
etf['etf_r5']=etf.aclose.pct_change(5); etf['etf_r20']=etf.aclose.pct_change(20); etf['etf_r60']=etf.aclose.pct_change(60)
etf['etf_ma20_gap']=etf.aclose/etf.ma20-1; etf['etf_ma60_gap']=etf.aclose/etf.ma60-1; etf['etf_ma120_gap']=etf.aclose/etf.ma120-1
etf['etf_vol20']=etf.aclose.pct_change().rolling(20,min_periods=20).std()*np.sqrt(252)
etf['etf_high60']=etf.aclose.rolling(60,min_periods=60).max(); etf['etf_dd60']=etf.aclose/etf.etf_high60-1
etf['etf_turn20']=etf.amount/etf.amount.rolling(20,min_periods=20).mean()-1

# broad-market aggregates are computed before candidate-specific filters beyond basic liquidity/code validity
agg=[]
for d,q in stocks.groupby('date'):
    q=q[np.isfinite(q.r1)&np.isfinite(q.r5)&np.isfinite(q.r20)]
    if len(q)<100: continue
    amount=q.amount.clip(lower=0)
    total_amt=float(amount.sum())
    top_amt=float(amount.nlargest(max(1,int(len(q)*.05))).sum()) if total_amt>0 else np.nan
    agg.append({
        'date':int(d),
        'median_r1':float(q.r1.median()),
        'median_r5':float(q.r5.median()),
        'median_r20':float(q.r20.median()),
        'adv_frac':float((q.r1>0).mean()),
        'ad_balance':float(np.sign(q.r1).mean()),
        'breadth20':float((q.aclose>q.ma20).mean()),
        'breadth60':float((q.aclose>q.ma60).mean()),
        'newhigh20':float((q.aclose>=q.prior_high20).mean()),
        'newlow20':float((q.aclose<=q.prior_low20).mean()),
        'dispersion1':float(q.r1.std(ddof=0)),
        'turnover_expansion':float(q.amount_ratio.replace([np.inf,-np.inf],np.nan).median()),
        'leadership_concentration':top_amt/total_amt if total_amt>0 else np.nan,
        'inst_flow_breadth':float((q.inst5>0).mean()),
        'inst_flow_median':float(q.flow_to_amount.replace([np.inf,-np.inf],np.nan).median()),
    })
ctx=etf.merge(pd.DataFrame(agg),on='date',how='inner').sort_values('date').reset_index(drop=True)
CTX_FEATURES=['etf_r5','etf_r20','etf_r60','etf_ma20_gap','etf_ma60_gap','etf_ma120_gap','etf_vol20','etf_dd60','etf_turn20','median_r1','median_r5','median_r20','adv_frac','ad_balance','breadth20','breadth60','newhigh20','newlow20','dispersion1','turnover_expansion','leadership_concentration','inst_flow_breadth','inst_flow_median']
# each day's standardization uses only PRIOR observations, never future/current distribution information
for c in CTX_FEATURES:
    mu=ctx[c].shift(1).rolling(252,min_periods=60).mean()
    sd=ctx[c].shift(1).rolling(252,min_periods=60).std().replace(0,np.nan)
    ctx[c+'_z']=((ctx[c]-mu)/sd).clip(-5,5)
Z=[c+'_z' for c in CTX_FEATURES]
ctx['context_ready']=ctx[Z].notna().sum(axis=1)>=int(len(Z)*.80)
ctx.to_csv(OUT/'daily_continuous_context.csv',index=False)
ctx_by_date={int(r.date):np.array([getattr(r,c) for c in Z],dtype=float) for r in ctx.itertuples(index=False) if bool(r.context_ready)}
ctx_dates=sorted(ctx_by_date)

# ---------- playbooks; context chooses one BEFORE stock screening ----------
PLAYBOOKS={
 'breakout': {'hold':8,'cond':lambda d:(d.aclose>d.prior_high20)&(d.r20_pr>=.72)&(d.r5>0)&(d.aclose>d.ma60),'score':lambda d:.40*d.r20_pr+.25*d.r5_pr+.20*d.flow_to_amount_pr+.15*d.amount20_pr},
 'trend_pullback': {'hold':10,'cond':lambda d:(d.ma20>d.ma60)&(d.aclose>d.ma60)&(d.r60_pr>=.65)&(d.draw20<=-.02)&(d.draw20>=-.10),'score':lambda d:.40*d.r60_pr+.25*d.flow_to_amount_pr+.20*(1-d.ret_vol20_pr)+.15*d.amount20_pr},
 'relative_strength': {'hold':10,'cond':lambda d:(d.r20_pr>=.82)&(d.r60_pr>=.70)&(d.aclose>d.ma20),'score':lambda d:.40*d.r20_pr+.30*d.r60_pr+.20*d.flow_to_amount_pr+.10*d.amount20_pr},
 'mean_reversion': {'hold':6,'cond':lambda d:(d.aclose>d.ma120)&(d.draw20<=-.05)&(d.draw20>=-.14)&(d.r5<0)&(d.r120_pr>=.45),'score':lambda d:.30*d.r120_pr+.25*d.flow_to_amount_pr+.25*(1-d.ret_vol20_pr)+.20*d.amount20_pr},
 'repair_reversal': {'hold':8,'cond':lambda d:(d.aclose>d.ma20)&(d.prev_aclose<=d.prev_ma20)&(d.r20>-0.10)&(d.flow_to_amount_pr>=.55),'score':lambda d:.35*d.flow_to_amount_pr+.25*d.r20_pr+.20*d.r60_pr+.20*d.amount20_pr},
}
QUALITY=['loose','balanced','strict']
def quality_mask(d,tier):
    common=(d.close>=10)&(d.amount20>=50_000_000)&(d.ret_vol20_pr<=.95)
    if tier=='loose': return common
    if tier=='balanced': return common&(d.amount20>=100_000_000)&(d.r120_pr>=.35)&(d.ret_vol20_pr<=.85)&((d.inst20>0)|(d.flow_to_amount_pr>=.50))
    if tier=='strict': return common&(d.amount20>=200_000_000)&(d.aclose>d.ma120)&(d.r120_pr>=.50)&(d.ret_vol20_pr<=.72)&(d.inst20>0)
    raise ValueError(tier)

def tick(p):
    if p<10:return .01
    if p<50:return .05
    if p<100:return .1
    if p<500:return .5
    if p<1000:return 1.0
    return 5.0
def ceil_tick(p): t=tick(p); return math.ceil((p-1e-12)/t)*t
def floor_tick(p): t=tick(p); return math.floor((p+1e-12)/t)*t

stocks=stocks.sort_values(['date','code']).copy()
bydate={int(d):q.copy() for d,q in stocks.groupby('date')}
calendar=sorted(int(x) for x in etf[(etf.date>=START)&(etf.date<=END)].date)
cal_i={d:i for i,d in enumerate(calendar)}
rowmap={(int(r.date),r.code):r for r in stocks.itertuples(index=False)}

# ---------- candidate events independent of context choice ----------
# These are research outcomes for playbook selection. They are not used until their exit is already in the past.
events=[]
for d in calendar:
    if d not in bydate or d not in ctx_by_date: continue
    i=cal_i[d]
    if i+1>=len(calendar): continue
    entry_date=calendar[i+1]; day=bydate[d]
    for pb,spec in PLAYBOOKS.items():
        raw=day[spec['cond'](day).fillna(False)].copy()
        if raw.empty: continue
        raw['score']=spec['score'](raw)
        for tier in QUALITY:
            q=raw[quality_mask(raw,tier).fillna(False)].sort_values('score',ascending=False).head(6)
            for r in q.itertuples(index=False):
                er=rowmap.get((entry_date,r.code))
                if er is None or not np.isfinite(er.open) or er.open<=0 or bool(getattr(er,'is_official_event',False)): continue
                ei=cal_i[entry_date]+int(spec['hold'])
                if ei>=len(calendar): continue
                exit_date=calendar[ei]; xr=rowmap.get((exit_date,r.code))
                if xr is None or not np.isfinite(xr.open) or xr.open<=0: continue
                ep=ceil_tick(float(er.open)*1.005); xp=floor_tick(float(xr.open)*0.995)
                net=(xp*(1-FEE-TAX))/(ep*(1+FEE))-1
                events.append({'signal_date':d,'entry_date':entry_date,'exit_date':exit_date,'playbook':pb,'quality':tier,'code':r.code,'score':float(r.score),'hold':int(spec['hold']),'entry_price':ep,'exit_price':xp,'return_net':net})
ev=pd.DataFrame(events)
if ev.empty: raise RuntimeError('no candidate events generated')
ev.to_csv(OUT/'candidate_events.csv',index=False)

# ---------- causal continuous-context routing ----------
def distance(a,b):
    mask=np.isfinite(a)&np.isfinite(b)
    if mask.sum()<int(len(a)*.70): return np.inf
    return float(np.sqrt(np.mean((a[mask]-b[mask])**2)))

def choose_method(d):
    if d not in ctx_by_date: return {'playbook':'CASH','quality':'none','reason':'context_not_ready'}
    prior=[x for x in ctx_dates if x<d]
    if len(prior)<MIN_HISTORY_DAYS: return {'playbook':'CASH','quality':'none','reason':'insufficient_context_history'}
    cur=ctx_by_date[d]
    ds=sorted(((distance(cur,ctx_by_date[x]),x) for x in prior), key=lambda t:t[0])
    neighbors=[x for dist,x in ds if np.isfinite(dist)][:K_NEIGHBORS]
    if len(neighbors)<20:return {'playbook':'CASH','quality':'none','reason':'insufficient_neighbors'}
    # only outcomes fully known before current T close are eligible
    hist=ev[(ev.signal_date.isin(neighbors))&(ev.exit_date<d)].copy()
    rows=[]
    for (pb,qt),q in hist.groupby(['playbook','quality']):
        n=len(q)
        if n<MIN_EVENTS: continue
        r=q.return_net.astype(float)
        wins=r[r>0].sum(); losses=-r[r<0].sum(); pf=float(wins/losses) if losses>0 else 99.0
        wr=float((r>0).mean()); mean=float(r.mean()); med=float(r.median()); p10=float(r.quantile(.10))
        # economic score rewards hit-rate + payoff but penalizes unstable downside; no target forcing
        score=2.2*wr+7.0*mean+1.5*med+0.15*min(pf,3.0)+1.5*min(p10,0)
        rows.append({'playbook':pb,'quality':qt,'n':n,'win_rate':wr,'mean_return':mean,'median_return':med,'p10':p10,'pf':pf,'route_score':score})
    if not rows:return {'playbook':'CASH','quality':'none','reason':'no_completed_neighbor_sample'}
    rr=pd.DataFrame(rows)
    # Require positive historical edge in similar contexts; otherwise cash.
    viable=rr[(rr.mean_return>0)&(rr.median_return>-0.02)&(rr.pf>1.05)].copy()
    if viable.empty:return {'playbook':'CASH','quality':'none','reason':'no_positive_neighbor_edge','neighbors':len(neighbors)}
    pick=viable.sort_values(['route_score','n'],ascending=False).iloc[0]
    return {'playbook':str(pick.playbook),'quality':str(pick.quality),'n':int(pick.n),'win_rate':float(pick.win_rate),'mean_return':float(pick.mean_return),'median_return':float(pick.median_return),'pf':float(pick.pf),'route_score':float(pick.route_score),'neighbors':len(neighbors)}

route_cache={d:choose_method(d) for d in calendar}
with open(OUT/'daily_routes.jsonl','w',encoding='utf-8') as f:
    for d in calendar:f.write(json.dumps({'date':d,**route_cache[d]},ensure_ascii=False)+'\n')

# ---------- portfolio: route first -> screen names -> quality filter -> T+1 ----------
def run_portfolio(start,end,label):
    cash=INIT; pos={}; trades=[]; navrows=[]; pending={}; route_rows=[]
    dates=[d for d in calendar if start<=d<=end]
    if not dates:return None
    for d in dates:
        for code,p in list(pos.items()):
            rr=rowmap.get((d,code))
            if rr is not None and bool(getattr(rr,'is_official_event',False)) and np.isfinite(rr.share_factor) and float(rr.share_factor)>0 and abs(float(rr.share_factor)-1)>1e-12:
                p['shares']=int(round(p['shares']*float(rr.share_factor)))
        # exits first
        for code,p in list(pos.items()):
            if p['exit_date']!=d: continue
            rr=rowmap.get((d,code))
            if rr is None or not np.isfinite(rr.open): continue
            xp=floor_tick(float(rr.open)*0.995); gross=p['shares']*xp; cash+=gross*(1-FEE-TAX)
            ret=(xp*(1-FEE-TAX))/(p['entry_price']*(1+FEE))-1
            trades.append({'label':label,'playbook':p['playbook'],'quality':p['quality'],'code':code,'signal_date':p['signal_date'],'entry_date':p['entry_date'],'exit_date':d,'entry_price':p['entry_price'],'exit_price':xp,'shares':p['shares'],'return_net':ret})
            del pos[code]
        # execute pending T+1 entries
        for item in pending.pop(d,[]):
            if len(pos)>=MAX_POS: break
            code=item['code']
            if code in pos: continue
            rr=rowmap.get((d,code))
            if rr is None or not np.isfinite(rr.open) or rr.open<=0 or bool(getattr(rr,'is_official_event',False)): continue
            ci=cal_i[d]+item['hold']
            if ci>=len(calendar): continue
            exit_date=calendar[ci]
            ep=ceil_tick(float(rr.open)*1.005)
            mv=0.0
            for c,p in pos.items():
                vr=rowmap.get((d,c))
                if vr is not None and np.isfinite(vr.open): mv+=p['shares']*float(vr.open)
            nav_open=cash+mv; budget=min(cash,nav_open*SLOT); sh=int(budget//(ep*(1+FEE)))
            if sh<=0: continue
            cost=sh*ep*(1+FEE)
            if cost>cash+1e-9: continue
            cash-=cost; pos[code]={'shares':sh,'entry_price':ep,'signal_date':item['signal_date'],'entry_date':d,'exit_date':exit_date,'playbook':item['playbook'],'quality':item['quality']}
        # choose today's method from past similar contexts ONLY, then screen names, then quality filter
        route=route_cache.get(d,{'playbook':'CASH','quality':'none','reason':'missing'})
        route_rows.append({'date':d,**route})
        if route.get('playbook')!='CASH' and d in bydate:
            pb=route['playbook']; qt=route['quality']; spec=PLAYBOOKS[pb]; day=bydate[d]
            raw=day[spec['cond'](day).fillna(False)].copy()
            if len(raw):
                raw['score']=spec['score'](raw)
                q=raw[quality_mask(raw,qt).fillna(False)].sort_values('score',ascending=False)
                # do not queue more than available slots; routing is independent of R10 trade count
                slots=max(0,MAX_POS-len(pos))
                picks=[]
                for r in q.itertuples(index=False):
                    if r.code in pos: continue
                    picks.append(r)
                    if len(picks)>=slots: break
                if picks:
                    i=cal_i[d]
                    if i+1<len(calendar):
                        ed=calendar[i+1]
                        pending.setdefault(ed,[]).extend([{'signal_date':d,'code':r.code,'hold':int(spec['hold']),'playbook':pb,'quality':qt} for r in picks])
        # close valuation
        mv=0.0
        for c,p in pos.items():
            rr=rowmap.get((d,c))
            if rr is not None and np.isfinite(rr.close):mv+=p['shares']*float(rr.close)
        navrows.append({'date':d,'cash':cash,'market_value':mv,'nav':cash+mv,'positions':len(pos),'route_playbook':route.get('playbook','CASH'),'route_quality':route.get('quality','none')})
    nav=pd.DataFrame(navrows); tr=pd.DataFrame(trades); routes=pd.DataFrame(route_rows)
    end_nav=float(nav.nav.iloc[-1]); years=max((pd.to_datetime(str(end))-pd.to_datetime(str(start))).days/365.25,1/252)
    cagr=(end_nav/INIT)**(1/years)-1; dd=float((nav.nav/nav.nav.cummax()-1).min())
    if len(tr):
        wins=int((tr.return_net>0).sum()); wr=wins/len(tr); gp=float(tr.loc[tr.return_net>0,'return_net'].sum()); gl=float(-tr.loc[tr.return_net<0,'return_net'].sum()); pf=gp/gl if gl>0 else 99.0
    else:wins=0;wr=np.nan;pf=np.nan
    return {'label':label,'start':start,'end':end,'end_nav':end_nav,'cagr':float(cagr),'max_drawdown':dd,'trades':int(len(tr)),'wins':wins,'win_rate':float(wr) if np.isfinite(wr) else None,'profit_factor':float(pf) if np.isfinite(pf) else None},nav,tr,routes

results=[]
for start,end,label in [(START,DEV_END,'dev_2023m5_2024'),(HOLDOUT_START,END,'holdout_2025'),(START,END,'full_2023m5_2025')]:
    r,nav,tr,routes=run_portfolio(start,end,label);results.append(r)
    nav.to_csv(OUT/f'nav_{label}.csv',index=False);tr.to_csv(OUT/f'trades_{label}.csv',index=False);routes.to_csv(OUT/f'routes_{label}.csv',index=False)

# year-by-year from full trade ledger using separate capital restart per year for transparency
for y in (2024,2025):
    ys=int(f'{y}0101'); ye=int(f'{y}1231')
    if ys<START:ys=START
    r,nav,tr,routes=run_portfolio(ys,ye,f'year_{y}');results.append(r)

pd.DataFrame(results).to_csv(OUT/'portfolio_summary.csv',index=False)
hold=[r for r in results if r['label']=='holdout_2025'][0]
full=[r for r in results if r['label']=='full_2023m5_2025'][0]
decision={
 'status':'PASS',
 'architecture':'continuous causal context -> past-similar-context route -> playbook screen -> separate quality filter -> T+1 daily execution',
 'semantic_market_labels_used':False,
 'context_features':CTX_FEATURES,
 'neighbor_policy':{'min_history_days':MIN_HISTORY_DAYS,'k_neighbors':K_NEIGHBORS,'min_completed_events_per_method':MIN_EVENTS},
 'research_window':[START,END],'dev_window':[START,DEV_END],'holdout_window':[HOLDOUT_START,END],
 'target':{'cagr':0.50,'win_rate':0.70},'full':full,'holdout':hold,
 'target_hit_full':bool(full['cagr']>=.50 and (full['win_rate'] or 0)>=.70),
 'target_hit_holdout':bool(hold['cagr']>=.50 and (hold['win_rate'] or 0)>=.70),
 'minute_stage_allowed':bool(hold['trades']>=20 and hold['cagr']>0 and (hold['profit_factor'] or 0)>1.10),
 'formal_r10_untouched':True,
}
(OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(decision,ensure_ascii=False,indent=2))
