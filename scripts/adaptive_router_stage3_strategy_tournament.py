#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, math, zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
REF=ROOT/'reference_bundle'
S1=ROOT/'adaptive_router_stage1'
OUT=ROOT/'adaptive_router_stage3'; OUT.mkdir(exist_ok=True)

START_CAPITAL=1_300_000.0
FEE=0.000855
SELL_TAX=0.003
BUY_SLIP=0.005
SELL_SLIP=0.005
MAX_POS=3
MAX_STOCK=0.25
TOTAL_EXPOSURE=0.75
STATES=['TREND_EXPANSION','ROTATION_DISPERSION','RISK_OFF','CHOP_FALSE_BREAKOUT','NEUTRAL_MIXED']
FAMILIES=['TREND_BREAKOUT','ROTATION_PULLBACK','MEAN_REVERSION','OVERSOLD_REBOUND','SELECTIVE_NEUTRAL']


def norm_code(s):
    return s.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(4)

def load_ohlcv():
    parts=[]
    for y in range(2015,2020):
        zp=REF/'ohlcv_raw'/f'yearly_{y}.zip'
        with zipfile.ZipFile(zp) as z:
            member=next(n for n in z.namelist() if n.lower().endswith('.csv'))
            with z.open(member) as f: d=pd.read_csv(f,dtype={'code':str},low_memory=False)
        parts.append(d)
    parts.append(pd.read_parquet(REF/'formal_2020'/'ohlcv_2020.parquet'))
    d=pd.concat(parts,ignore_index=True)
    d.columns=[str(c).strip().lower() for c in d.columns]
    d['code']=norm_code(d.code)
    if pd.api.types.is_datetime64_any_dtype(d.date): d['date']=d.date.dt.strftime('%Y%m%d').astype(int)
    else: d['date']=pd.to_numeric(d.date.astype(str).str.replace('-','',regex=False),errors='coerce').astype('Int64')
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['date','code','open','high','low','close','volume']).copy(); d['date']=d.date.astype(int)
    # common-stock-like four-digit universe; excludes 00xx ETFs and malformed symbols.
    d=d[d.code.str.fullmatch(r'[1-9]\d{3}')].copy()
    d=d.sort_values(['code','date']).drop_duplicates(['code','date'],keep='last').reset_index(drop=True)
    return d

def load_inst():
    parts=[]
    for y in range(2015,2020):
        p=REF/'institutional_2015_2019'/f'institutional_{y}.csv.gz'
        x=pd.read_csv(p,dtype={'code':str},low_memory=False); parts.append(x)
    x=pd.read_parquet(REF/'formal_2020'/'institutional_2020_2025.parquet')
    if pd.api.types.is_datetime64_any_dtype(x.date): yr=x.date.dt.year
    else: yr=pd.to_datetime(x.date).dt.year
    parts.append(x[yr==2020].copy())
    d=pd.concat(parts,ignore_index=True)
    d.columns=[str(c).strip().lower() for c in d.columns]
    d['code']=norm_code(d.code)
    if pd.api.types.is_datetime64_any_dtype(d.date): d['date']=d.date.dt.strftime('%Y%m%d').astype(int)
    else: d['date']=pd.to_datetime(d.date).dt.strftime('%Y%m%d').astype(int)
    for c in ['foreign_net','trust_net','dealer_net','total_net']:
        if c not in d: d[c]=0.0
        d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0.0)
    return d[['date','code','foreign_net','trust_net','dealer_net','total_net']].sort_values(['code','date']).drop_duplicates(['date','code'],keep='last')

def prepare():
    d=load_ohlcv(); inst=load_inst()
    d=d.merge(inst,on=['date','code'],how='left')
    for c in ['foreign_net','trust_net','dealer_net','total_net']: d[c]=d[c].fillna(0.0)
    g=d.groupby('code',group_keys=False)
    d['prev_raw_close']=g.close.shift(1)
    d['bridge_mult']=d.prev_raw_close/d.open
    # Taiwan daily-price limits make >18% overnight discontinuities overwhelmingly corporate-action/data-continuity events.
    # Infer only a continuity bridge; no external market fetch and no future row is used.
    d['action_flag']=d.prev_raw_close.notna() & ((d.bridge_mult<0.82)|(d.bridge_mult>1.18))
    d.loc[~d.action_flag,'bridge_mult']=1.0
    d.loc[(d.bridge_mult<0.10)|(d.bridge_mult>10.0),'bridge_mult']=1.0
    d['adj_factor']=g.bridge_mult.cumprod()
    for c in ['open','high','low','close']: d['adj_'+c]=d[c]*d.adj_factor
    d['amount']=d.close*d.volume
    g=d.groupby('code',group_keys=False)
    d['amt20']=g.amount.transform(lambda s:s.rolling(20,min_periods=20).mean())
    d['vma20']=g.volume.transform(lambda s:s.rolling(20,min_periods=20).mean())
    d['vol_ratio']=d.volume/d.vma20.replace(0,np.nan)
    for n in [10,20,60,120]: d[f'ma{n}']=g.adj_close.transform(lambda s,n=n:s.rolling(n,min_periods=n).mean())
    for n in [5,20,60]: d[f'r{n}']=g.adj_close.transform(lambda s,n=n:s.pct_change(n,fill_method=None))
    d['prior20']=g.adj_close.transform(lambda s:s.shift(1).rolling(20,min_periods=20).max())
    d['prior60']=g.adj_close.transform(lambda s:s.shift(1).rolling(60,min_periods=60).max())
    d['inst_net']=d.foreign_net+d.trust_net
    d['inst5']=g.inst_net.transform(lambda s:s.rolling(5,min_periods=3).sum())
    d['inst5_ratio']=d.inst5/d.vma20.replace(0,np.nan)
    d['liquid']=d.amt20>=30_000_000
    # Previous suspicious bridge count is causal and used to quarantine recently discontinuous symbols.
    d['action_recent']=g.action_flag.transform(lambda s:s.shift(1).rolling(5,min_periods=1).max()).fillna(False).astype(bool)
    states=pd.read_csv(S1/'stage1_daily_states.csv')
    states['date']=pd.to_numeric(states.date,errors='coerce').astype('Int64')
    d=d.merge(states[['date','state']],on='date',how='left')
    return d.sort_values(['date','code']).reset_index(drop=True)

def signal_mask(x,fam):
    base=x.liquid & ~x.action_recent & x.ma120.notna() & (x.close>=3.0)
    if fam=='TREND_BREAKOUT':
        m=base&(x.adj_close>x.prior60)&(x.ma20>x.ma60)&(x.ma60>x.ma120)&(x.r20>0.08)&(x.r60>0.10)&(x.vol_ratio>1.10)
        score=2*x.r20+x.r60+0.03*np.log1p(x.vol_ratio.clip(lower=0)) + 0.05*x.inst5_ratio.clip(-1,1)
    elif fam=='ROTATION_PULLBACK':
        pull=x.adj_close/x.ma20
        m=base&(x.adj_close>x.ma120)&(x.ma60>x.ma120)&(x.r60>0.08)&(x.r20>0)&(x.r5>=-0.08)&(x.r5<=0.01)&(pull>=0.96)&(pull<=1.02)
        score=x.r60+0.5*x.r20-abs(pull-1)*2+0.05*x.inst5_ratio.clip(-1,1)
    elif fam=='MEAN_REVERSION':
        m=base&(x.adj_close>x.ma120*0.95)&(x.r5<-0.06)&(x.r20>-0.20)&(x.adj_close/x.ma20<0.95)&(x.vol_ratio>0.80)
        score=-x.r5-0.5*abs(x.r20)+0.02*x.inst5_ratio.clip(-1,1)
    elif fam=='OVERSOLD_REBOUND':
        m=base&(x.r5<-0.10)&(x.r20<-0.08)&(x.adj_close/x.ma20<0.92)&(x.vol_ratio>1.10)
        score=-1.5*x.r5-0.5*x.r20+0.02*x.inst5_ratio.clip(-1,1)
    else:
        near=x.adj_close/x.prior20
        m=base&(x.adj_close>x.ma60)&(x.ma60>x.ma120)&(x.r20>=0.03)&(x.r20<=0.15)&(near>=0.94)&(x.r5>-0.04)&(x.vol_ratio>=0.75)&(x.vol_ratio<=2.0)
        score=x.r20+0.4*x.r60+0.04*x.inst5_ratio.clip(-1,1)-abs(near-0.98)*0.5
    return m,score

def exit_signal(row,pos,fam):
    held=pos['held_days']; c=row.adj_close; entry=pos['entry_adj']
    stop=c <= entry*0.88
    if fam=='TREND_BREAKOUT': return stop or held>=25 or (pd.notna(row.ma20) and c<row.ma20) or (pd.notna(row.r5) and row.r5<-0.08)
    if fam=='ROTATION_PULLBACK': return stop or held>=15 or (pd.notna(row.ma60) and c<row.ma60) or (pd.notna(row.ma20) and c>row.ma20*1.07)
    if fam=='MEAN_REVERSION': return stop or held>=10 or (pd.notna(row.ma20) and c>=row.ma20)
    if fam=='OVERSOLD_REBOUND': return stop or held>=7 or (pd.notna(row.ma10) and c>=row.ma10)
    return stop or held>=18 or (pd.notna(row.ma20) and c<row.ma20)

def simulate(d,fam,allowed_state):
    dd=d[(d.date>=20160101)&(d.date<=20201231)].copy()
    dates=sorted(dd.date.unique())
    bydate={dt:x.set_index('code',drop=False) for dt,x in dd.groupby('date')}
    cash=START_CAPITAL; positions={}; pending_buys=[]; pending_sells=set(); trades=[]; nav_rows=[]
    for dt in dates:
        x=bydate[dt]
        # Apply inferred corporate-action share bridges before any T+1 open execution.
        for code,pos in list(positions.items()):
            if code in x.index:
                r=x.loc[code]
                if bool(r.action_flag):
                    new_sh=int(math.floor(pos['shares']*float(r.bridge_mult)))
                    pos['shares']=max(new_sh,0)
        # Sell orders committed at previous close.
        for code in list(pending_sells):
            if code not in positions or code not in x.index: continue
            r=x.loc[code]; pos=positions.pop(code); sh=int(pos['shares'])
            px=float(r.open)*(1-SELL_SLIP); gross=sh*px; proceeds=gross*(1-FEE-SELL_TAX); cash+=proceeds
            pnl=proceeds-pos['cash_cost']
            trades.append({'family':fam,'state':allowed_state,'code':code,'entry_date':pos['entry_date'],'exit_date':dt,'shares':sh,'entry_price':pos['entry_price'],'exit_price':px,'pnl':pnl,'return':pnl/pos['cash_cost'] if pos['cash_cost'] else np.nan,'hold_days':pos['held_days'],'exit_reason':pos.get('exit_reason','RULE')})
        pending_sells=set()
        # Buy orders committed at previous close, fixed limit known before today.
        still=[]
        for o in pending_buys:
            code=o['code']
            if code in positions or code not in x.index or len(positions)>=MAX_POS: continue
            r=x.loc[code]; limit=float(o['limit'])
            if float(r.low)>limit: continue
            px=min(float(r.open)*(1+BUY_SLIP),limit) if float(r.open)<=limit else limit
            budget=min(o['budget'],cash/(1+FEE))
            sh=int(math.floor(budget/px))
            if sh<=0: continue
            gross=sh*px; cost=gross*(1+FEE)
            if cost>cash: sh=int(math.floor(cash/(px*(1+FEE)))); gross=sh*px; cost=gross*(1+FEE)
            if sh<=0: continue
            cash-=cost
            positions[code]={'shares':sh,'entry_date':dt,'entry_price':px,'entry_adj':px*float(r.adj_factor),'cash_cost':cost,'held_days':0}
        pending_buys=still
        # Close mark.
        market_value=0.0
        for code,pos in positions.items():
            if code in x.index:
                market_value += int(pos['shares'])*float(x.loc[code].close)
                pos['held_days']+=1
        nav=cash+market_value
        st=x.state.dropna().iloc[0] if x.state.notna().any() else None
        nav_rows.append({'date':dt,'nav':nav,'cash':cash,'positions':len(positions),'state':st})
        # Commit exits using only today's close information. Regime change exits next open.
        for code,pos in positions.items():
            if code not in x.index: continue
            r=x.loc[code]
            if st!=allowed_state:
                pos['exit_reason']='STATE_CHANGE'; pending_sells.add(code)
            elif exit_signal(r,pos,fam):
                pos['exit_reason']='RULE'; pending_sells.add(code)
        # New orders only when currently in the tested state.
        if st==allowed_state:
            free_slots=MAX_POS-len(positions)-len(pending_buys)
            if free_slots>0:
                m,score=signal_mask(x,fam); cand=x.loc[m].copy(); cand['score']=score.loc[m]
                cand=cand[~cand.code.isin(positions) & ~cand.code.isin([o['code'] for o in pending_buys])]
                cand=cand.sort_values(['score','amount'],ascending=False).head(free_slots)
                budget_each=min(nav*MAX_STOCK, nav*TOTAL_EXPOSURE/max(MAX_POS,1))
                for _,r in cand.iterrows():
                    pending_buys.append({'code':r.code,'limit':float(r.close)*1.01,'budget':budget_each,'signal_date':dt})
    # Liquidate final positions at final available close with conservative sell friction (research boundary only).
    if dates:
        dt=dates[-1]; x=bydate[dt]
        for code,pos in list(positions.items()):
            if code not in x.index: continue
            px=float(x.loc[code].close)*(1-SELL_SLIP); sh=int(pos['shares']); proceeds=sh*px*(1-FEE-SELL_TAX); cash+=proceeds
            pnl=proceeds-pos['cash_cost']; trades.append({'family':fam,'state':allowed_state,'code':code,'entry_date':pos['entry_date'],'exit_date':dt,'shares':sh,'entry_price':pos['entry_price'],'exit_price':px,'pnl':pnl,'return':pnl/pos['cash_cost'] if pos['cash_cost'] else np.nan,'hold_days':pos['held_days'],'exit_reason':'END'})
            positions.pop(code,None)
        nav_rows[-1]['nav']=cash; nav_rows[-1]['cash']=cash; nav_rows[-1]['positions']=0
    navdf=pd.DataFrame(nav_rows); tdf=pd.DataFrame(trades)
    if navdf.empty: return {},tdf,navdf
    peak=navdf.nav.cummax(); navdf['drawdown']=navdf.nav/peak-1
    years=(pd.to_datetime(str(int(navdf.date.iloc[-1])))-pd.to_datetime(str(int(navdf.date.iloc[0])))).days/365.25
    end=float(navdf.nav.iloc[-1]); cagr=(end/START_CAPITAL)**(1/years)-1 if years>0 else np.nan
    if len(tdf):
        gp=float(tdf.loc[tdf.pnl>0,'pnl'].sum()); gl=float(-tdf.loc[tdf.pnl<0,'pnl'].sum()); pf=gp/gl if gl>0 else np.nan; wr=float((tdf.pnl>0).mean())
    else: pf=np.nan; wr=np.nan
    summ={'family':fam,'state':allowed_state,'end_nav':end,'cagr':float(cagr),'max_dd':float(navdf.drawdown.min()),'trades':int(len(tdf)),'win_rate':wr,'pf':pf,'net_pnl':end-START_CAPITAL}
    return summ,tdf,navdf

def annual(navdf,fam,state):
    z=navdf.copy(); z['year']=z.date.astype(str).str[:4].astype(int); rows=[]
    for y,g in z.groupby('year'):
        start=float(g.nav.iloc[0]); end=float(g.nav.iloc[-1]); rows.append({'family':fam,'state':state,'year':int(y),'return':end/start-1,'max_dd':float((g.nav/g.nav.cummax()-1).min())})
    return rows

def main():
    d=prepare()
    action_stats=d[d.action_flag].groupby(d.date.astype(str).str[:4]).size().to_dict()
    summaries=[]; alltr=[]; ally=[]
    for state in STATES:
        for fam in FAMILIES:
            s,t,n=simulate(d,fam,state); summaries.append(s)
            if len(t): alltr.append(t)
            if len(n): ally += annual(n,fam,state)
            print('PAIR',state,fam,json.dumps(s,ensure_ascii=False),flush=True)
    sm=pd.DataFrame(summaries); tr=pd.concat(alltr,ignore_index=True) if alltr else pd.DataFrame(); yr=pd.DataFrame(ally)
    sm.to_csv(OUT/'strategy_state_matrix_2016_2020.csv',index=False); tr.to_csv(OUT/'all_trades_2016_2020.csv',index=False); yr.to_csv(OUT/'annual_pair_stability.csv',index=False)
    # Fixed pre-2021 tournament summary: rank by risk-adjusted evidence, not by holdout.
    q=sm.copy(); q['score']=q.cagr-0.75*(-q.max_dd).clip(lower=0)+0.015*np.log1p(q.trades.clip(lower=0))+0.02*np.log(q.pf.clip(lower=0.05,upper=10))
    q['eligible']=(q.trades>=8)&(q.pf>=1.10)&(q.max_dd>=-0.25)
    best=q[q.eligible].sort_values(['state','score'],ascending=[True,False]).groupby('state').head(3)
    best.to_csv(OUT/'pre2021_family_shortlist.csv',index=False)
    manifest={'design':'stage3_fixed-family_transaction_tournament','period':'2016-2020 discovery only','warmup':2015,'market_refetch':False,'source_artifact_run':34313660140,'formal_r10_modified':False,'uses_2021_2025_for_selection':False,'execution':'T-close fixed buy limit 1% above close; T+1 fill only if touched; adverse buy 0.5%; sells committed T-close and T+1 open adverse 0.5%; fees/tax applied; integer shares; shared cash','corporate_action_handling':'causal inferred continuity bridge for >18% overnight discontinuities; share multiplier applied when held; recent events quarantined from new signals','action_flags_by_year':{str(k):int(v) for k,v in action_stats.items()},'warning':'pre-2020 official corporate-action completeness is unavailable in cached bundle; inferred bridges are conservative research handling, not a claim of exact official action reconstruction','next':'rolling walk-forward router using only prior realized pair evidence; holdout 2021-2025 remains untouched'}
    (OUT/'stage3_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('ADAPTIVE_ROUTER_STAGE3_COMPLETE')
    print(best.to_string(index=False))

if __name__=='__main__': main()
