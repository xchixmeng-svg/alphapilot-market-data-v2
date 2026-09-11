from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path('research_out_distinct_flow'); OUT.mkdir(exist_ok=True)
START=20210101; END=20251231; DEV_END=20231231
FEE=0.000855; TAX=0.003; INIT=1_300_000.0; MAX_POS=4; SLOT=0.24

px=pd.read_csv('formal_run/ohlcv_causal_2020_2025.csv.gz',dtype={'code':str},low_memory=False)
px['code']=px.code.astype(str).str.zfill(4)
px=px.sort_values(['code','date']).reset_index(drop=True)
px=px[(px.date>=20200101)&(px.date<=END)].copy()
px['amt']=px['close']*px['volume']
g=px.groupby('code',group_keys=False)
for w in (5,10,20,60,120):
    px[f'r{w}']=g['aclose'].transform(lambda s:s.pct_change(w))
for w in (20,60,120):
    px[f'ma{w}']=g['aclose'].transform(lambda s:s.rolling(w,min_periods=w).mean())
px['amt20']=g['amt'].transform(lambda s:s.rolling(20,min_periods=20).mean())
px['prior_high20']=g['aclose'].transform(lambda s:s.shift(1).rolling(20,min_periods=20).max())
px['prior_high60']=g['aclose'].transform(lambda s:s.shift(1).rolling(60,min_periods=60).max())
px['draw20']=px['aclose']/g['aclose'].transform(lambda s:s.rolling(20,min_periods=20).max())-1
px['vol20ret']=g['aclose'].transform(lambda s:s.pct_change().rolling(20,min_periods=20).std())
inst=pd.read_parquet('formal_run/institutional_2020_2025.parquet')
inst['code']=inst.code.astype(str).str.zfill(4)
if pd.api.types.is_datetime64_any_dtype(inst.date): inst['date']=inst.date.dt.strftime('%Y%m%d').astype(int)
else: inst['date']=pd.to_datetime(inst.date).dt.strftime('%Y%m%d').astype(int)
inst=inst.sort_values(['code','date'])
gi=inst.groupby('code',group_keys=False)
inst['foreign5']=gi.foreign_net.transform(lambda s:s.rolling(5,min_periods=5).sum())
inst['foreign20']=gi.foreign_net.transform(lambda s:s.rolling(20,min_periods=20).sum())
inst['trust5']=gi.trust_net.transform(lambda s:s.rolling(5,min_periods=5).sum())
px=px.merge(inst[['date','code','foreign5','foreign20','trust5']],on=['date','code'],how='left')
px[['foreign5','foreign20','trust5']]=px[['foreign5','foreign20','trust5']].fillna(0)
px['flow_value5']=(px.foreign5+px.trust5)*px.close
px['flow_to_amt']=px.flow_value5/(px.amt20*5).replace(0,np.nan)
valid=px.code.str.fullmatch(r'[1-9]\d{3}') & ~px.name.astype(str).str.contains('KY',case=False,na=False) & (px.amt20>=30_000_000)
px=px[valid].copy()
# market regime: 0050 only, causal and simple
m=px[px.code=='0050'][['date','aclose']].drop_duplicates('date').sort_values('date')
m['ma60']=m.aclose.rolling(60,min_periods=60).mean();m['r20']=m.aclose.pct_change(20);m['r60']=m.aclose.pct_change(60)
m['risk_on']=(m.aclose>m.ma60)&(m.r20>0)&(m.r60>0)
reg=dict(zip(m.date,m.risk_on.fillna(False)))
# cross-sectional ranks by date, all based on T-close information
for c in ['r10','r20','r60','flow_to_amt','vol20ret']:
    px[c+'_pr']=px.groupby('date')[c].rank(pct=True)

families={
 'flow_breakout_10': dict(hold=10, cond=lambda d:(d.aclose>d.prior_high20)&(d.flow_to_amt_pr>=.70)&(d.r20_pr>=.60)&(d.aclose>d.ma60), score=lambda d:.40*d.flow_to_amt_pr+.35*d.r20_pr+.25*d.r10_pr),
 'flow_breakout_20': dict(hold=20, cond=lambda d:(d.aclose>d.prior_high60)&(d.flow_to_amt_pr>=.65)&(d.r60_pr>=.60)&(d.aclose>d.ma60), score=lambda d:.35*d.flow_to_amt_pr+.35*d.r60_pr+.30*d.r20_pr),
 'flow_pullback_10': dict(hold=10, cond=lambda d:(d.aclose>d.ma60)&(d.r60_pr>=.70)&(d.draw20<=-.03)&(d.draw20>=-.10)&(d.flow_to_amt_pr>=.65), score=lambda d:.35*d.flow_to_amt_pr+.35*d.r60_pr+.30*(1-d.vol20ret_pr)),
 'lowvol_momo_20': dict(hold=20, cond=lambda d:(d.aclose>d.ma60)&(d.ma60>d.ma120)&(d.r60_pr>=.75)&(d.vol20ret_pr<=.45)&(d.flow_to_amt_pr>=.50), score=lambda d:.35*d.r60_pr+.25*d.r20_pr+.25*(1-d.vol20ret_pr)+.15*d.flow_to_amt_pr),
}

# Legal tick helpers for modeled adverse opening fills.
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

bydate={int(k):v.copy() for k,v in px.groupby('date')}
dates=sorted(d for d in bydate if START<=d<=END)
nextdate={dates[i]:dates[i+1] for i in range(len(dates)-1)}
# per-code date ordering for holding-day exits
code_dates={c:list(g.date.astype(int)) for c,g in px.groupby('code')}
rowmap={(int(r.date),r.code):r for r in px.itertuples(index=False)}

all_summary=[]; all_trades=[]; annual=[]
for fname,spec in families.items():
    cash=INIT; pos={}; trades=[]; nav_hist=[]; pending={}
    for d in dates:
        day=bydate[d]
        # corporate-action share adjustment before valuation/execution
        for code,p in list(pos.items()):
            rr=rowmap.get((d,code))
            if rr is not None and bool(rr.is_official_event) and float(rr.share_factor)>0 and abs(float(rr.share_factor)-1)>1e-12:
                p['shares']=int(round(p['shares']*float(rr.share_factor)))
        # exits first at today's open if scheduled
        for code,p in list(pos.items()):
            if p['exit_date']!=d: continue
            rr=rowmap.get((d,code))
            if rr is None or not np.isfinite(rr.open): continue
            fp=floor_tick(float(rr.open)*0.995); gross=p['shares']*fp; cash+=gross*(1-FEE-TAX)
            ret=(fp*(1-FEE-TAX))/(p['entry_price']*(1+FEE))-1
            trades.append({'family':fname,'code':code,'entry_date':p['entry_date'],'exit_date':d,'entry_price':p['entry_price'],'exit_price':fp,'shares':p['shares'],'return_net':ret})
            del pos[code]
        # buys from yesterday's locked candidates
        cand=pending.pop(d,[])
        if cand:
            # opening NAV before buys
            mv=sum(p['shares']*float(rowmap[(d,c)].open) for c,p in pos.items() if (d,c) in rowmap and np.isfinite(rowmap[(d,c)].open))
            nav_open=cash+mv
            for code,score in cand:
                if len(pos)>=MAX_POS or code in pos: break
                rr=rowmap.get((d,code))
                if rr is None or not np.isfinite(rr.open) or rr.open<=0: continue
                fp=ceil_tick(float(rr.open)*1.005)
                budget=min(cash,nav_open*SLOT)
                sh=int(budget//(fp*(1+FEE)))
                if sh<=0: continue
                cost=sh*fp*(1+FEE); cash-=cost
                cds=code_dates[code]; idx=cds.index(d); ei=min(idx+int(spec['hold']),len(cds)-1); ed=int(cds[ei])
                if ed<=d: continue
                pos[code]={'shares':sh,'entry_price':fp,'entry_date':d,'exit_date':ed,'score':score}
        # close NAV
        mv=0.0
        for code,p in pos.items():
            rr=rowmap.get((d,code))
            if rr is not None and np.isfinite(rr.close):mv+=p['shares']*float(rr.close)
        nav=cash+mv;nav_hist.append((d,nav))
        # create T-close signal for T+1, immutable at d close
        nd=nextdate.get(d)
        if nd is not None and reg.get(d,False):
            dd=day.copy(); mask=spec['cond'](dd).fillna(False); dd=dd[mask].copy()
            if len(dd):
                dd['score']=spec['score'](dd); dd=dd.sort_values('score',ascending=False)
                pending[nd]=[(r.code,float(r.score)) for r in dd.head(8).itertuples(index=False)]
    # liquidate residual at final close, conservative sell costs
    last=dates[-1]
    for code,p in list(pos.items()):
        rr=rowmap.get((last,code));
        if rr is None: continue
        fp=floor_tick(float(rr.close)*0.995);cash+=p['shares']*fp*(1-FEE-TAX)
        ret=(fp*(1-FEE-TAX))/(p['entry_price']*(1+FEE))-1
        trades.append({'family':fname,'code':code,'entry_date':p['entry_date'],'exit_date':last,'entry_price':p['entry_price'],'exit_price':fp,'shares':p['shares'],'return_net':ret})
    t=pd.DataFrame(trades); nh=pd.DataFrame(nav_hist,columns=['date','nav'])
    end=float(cash); yrs=(pd.to_datetime(str(END))-pd.to_datetime(str(START))).days/365.25; cagr=(end/INIT)**(1/yrs)-1
    peak=nh.nav.cummax(); dd=(nh.nav/peak-1); mdd=float(dd.min()) if len(dd) else 0
    wins=int((t.return_net>0).sum()) if len(t) else 0; n=len(t); wr=wins/n if n else 0
    gp=float(t.loc[t.return_net>0,'return_net'].sum()) if n else 0; gl=float(-t.loc[t.return_net<0,'return_net'].sum()) if n else 0; pf=gp/gl if gl>0 else np.inf
    dev=t[t.entry_date<=DEV_END]; val=t[t.entry_date>DEV_END]
    rec={'family':fname,'end_nav':end,'cagr':cagr,'max_drawdown':mdd,'trades':n,'wins':wins,'win_rate':wr,'profit_factor':pf,'dev_n':len(dev),'dev_win_rate':float((dev.return_net>0).mean()) if len(dev) else np.nan,'dev_mean':float(dev.return_net.mean()) if len(dev) else np.nan,'holdout_n':len(val),'holdout_win_rate':float((val.return_net>0).mean()) if len(val) else np.nan,'holdout_mean':float(val.return_net.mean()) if len(val) else np.nan}
    all_summary.append(rec);all_trades.extend(trades)
    if len(t):
        t['year']=(t.entry_date//10000).astype(int)
        for y,gg in t.groupby('year'):annual.append({'family':fname,'year':int(y),'n':len(gg),'win_rate':float((gg.return_net>0).mean()),'mean_return':float(gg.return_net.mean()),'sum_return':float(gg.return_net.sum())})

s=pd.DataFrame(all_summary).sort_values(['cagr','win_rate'],ascending=False);s.to_csv(OUT/'summary.csv',index=False)
pd.DataFrame(all_trades).to_csv(OUT/'trades.csv',index=False);pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False)
# Strict promotion: target + adequate samples + both dev/holdout positive and >=60% WR before any minute refinement.
s['target']= (s.cagr>=.50)&(s.win_rate>=.70)&(s.trades>=30)&(s.dev_n>=15)&(s.holdout_n>=15)&(s.dev_mean>0)&(s.holdout_mean>0)
payload={'status':'PASS','objective':'distinct daily selection families before minute refinement','dev_end':DEV_END,'formal_branch_untouched':True,'best':s.iloc[0].to_dict() if len(s) else None,'promotable':s[s.target].to_dict('records'),'note':'Research only. No formal R10 files modified.'}
(OUT/'decision.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=lambda o:bool(o) if isinstance(o,(np.bool_,)) else float(o) if isinstance(o,(np.floating,)) else int(o)),encoding='utf-8')
print(s.to_string(index=False));print(json.dumps(payload,ensure_ascii=False,indent=2,default=str))
