from pathlib import Path
import json, zipfile, math
import numpy as np, pandas as pd

ROOT=Path('reference_bundle'); OUT=Path('adaptive_router_stage15'); OUT.mkdir(exist_ok=True)
INIT=1_300_000.0; FEE=.000855; TAX=.003; SLIP=.005; MAX_POS=4; MAX_W=.25; MAX_HOLD=120
DISC=[2016,2017,2018]; OOS1=2019; OOS2=2020

def load_year(y):
    if y==2020:return pd.read_parquet(ROOT/'formal_2020'/'ohlcv_2020.parquet')
    with zipfile.ZipFile(ROOT/'ohlcv_raw'/f'yearly_{y}.zip') as z:
        n=next(n for n in z.namelist() if n.lower().endswith('.csv'))
        with z.open(n) as f:return pd.read_csv(f,dtype={'code':str},low_memory=False)

def norm(x):
    x=x.copy(); x.columns=[str(c).lower() for c in x.columns]; ren={}
    for c in x.columns:
        if c in ('date','trade_date'):ren[c]='date'
        elif c in ('code','stock_id','symbol'):ren[c]='code'
        elif c in ('open','opening_price'):ren[c]='open'
        elif c in ('high','highest_price'):ren[c]='high'
        elif c in ('low','lowest_price'):ren[c]='low'
        elif c in ('close','closing_price'):ren[c]='close'
        elif c in ('volume','trade_volume'):ren[c]='volume'
        elif c in ('amount','trade_value','turnover'):ren[c]='amount'
    x=x.rename(columns=ren); s=x.date
    if pd.api.types.is_datetime64_any_dtype(s):x['date']=pd.to_datetime(s,errors='coerce')
    else:
        ss=s.astype(str).str.strip().str.replace(r'\.0$','',regex=True);m=ss.str.fullmatch(r'\d{8}');d=pd.Series(pd.NaT,index=x.index,dtype='datetime64[ns]');d.loc[m]=pd.to_datetime(ss.loc[m],format='%Y%m%d',errors='coerce');d.loc[~m]=pd.to_datetime(ss.loc[~m],errors='coerce');x['date']=d
    x['code']=x.code.astype(str).str.replace(r'\.0$','',regex=True).str.extract(r'(\d+)')[0].str.zfill(4)
    for c in ['open','high','low','close','volume']+[c for c in ['amount'] if c in x]:x[c]=pd.to_numeric(x[c],errors='coerce')
    if 'amount' not in x:x['amount']=x.close*x.volume
    return x.dropna(subset=['date','code','open','high','low','close','volume'])

raw=pd.concat([norm(load_year(y)) for y in range(2015,2021)],ignore_index=True).sort_values(['code','date']).drop_duplicates(['code','date'],keep='last').reset_index(drop=True)
g=raw.groupby('code',group_keys=False);raw['prev_close']=g.close.shift(1);raw['bridge']=raw.prev_close/raw.open;raw['action_flag']=raw.prev_close.notna()&((raw.bridge<.82)|(raw.bridge>1.18));raw.loc[~raw.action_flag,'bridge']=1.;raw.loc[(raw.bridge<.10)|(raw.bridge>10),'bridge']=1.;raw['adj_factor']=raw.groupby('code').bridge.cumprod()
for c in ['open','high','low','close']:raw['adj_'+c]=raw[c]*raw.adj_factor
raw['action_recent']=raw.groupby('code').action_flag.transform(lambda s:s.shift(1).rolling(5,min_periods=1).max()).fillna(False).astype(bool);raw['amount']=raw.close*raw.volume;g=raw.groupby('code',group_keys=False)
for n in [10,20,60,120]:raw[f'r{n}']=g.adj_close.transform(lambda s,n=n:s.pct_change(n,fill_method=None))
for n in [20,60,120]:raw[f'ma{n}']=g.adj_close.transform(lambda s,n=n:s.rolling(n,min_periods=n).mean())
raw['prior120']=g.adj_high.transform(lambda s:s.shift(1).rolling(120,min_periods=120).max());raw['break120']=raw.adj_close/raw.prior120-1;raw['amt20']=g.amount.transform(lambda s:s.rolling(20,min_periods=20).mean());raw['liq_pct']=raw.groupby('date').amt20.rank(pct=True)

b=raw[raw.code=='0050'][['date','adj_close','r60','r120','ma120']].drop_duplicates('date').sort_values('date').rename(columns={c:'b_'+c for c in ['adj_close','r60','r120','ma120']});raw=raw.merge(b,on='date',how='left');raw['rel60']=raw.r60-raw.b_r60;raw['rel120']=raw.r120-raw.b_r120;raw['market_on']=(raw.b_adj_close>raw.b_ma120)&(raw.b_r60>0)
base=raw.code.str.fullmatch(r'[1-9]\d{3}')&(raw.liq_pct>=.35)&(~raw.action_recent)&raw.market_on&(raw.adj_close>raw.ma120)&(raw.ma60>raw.ma120)&(raw.r60>0)

VARIANTS=['RS60_MA20','RS60_MA60','RS60_120_MA20','BREAK120_MA20']
def entry_mask_score(d,name):
    if name=='RS60_MA20':m=d.rel60>0;s=d.rel60
    elif name=='RS60_MA60':m=d.rel60>0;s=d.rel60
    elif name=='RS60_120_MA20':m=(d.rel60>0)&(d.rel120>0);s=d.rel60+d.rel120
    else:m=(d.rel60>0)&(d.break120>=0);s=d.rel60+2*d.break120
    return m,s

def should_exit(r,p,name):
    if not bool(r.market_on):return True
    if p['held']>=MAX_HOLD:return True
    if r.adj_close<=p['entry_adj']*.88:return True
    if name=='RS60_MA60':return (pd.notna(r.ma60) and r.adj_close<r.ma60) or (pd.notna(r.rel60) and r.rel60<=0)
    if name=='RS60_120_MA20':return (pd.notna(r.ma20) and r.adj_close<r.ma20) or (pd.notna(r.rel60) and r.rel60<=0)
    if name=='BREAK120_MA20':return (pd.notna(r.ma20) and r.adj_close<r.ma20)
    return (pd.notna(r.ma20) and r.adj_close<r.ma20) or (pd.notna(r.rel60) and r.rel60<=0)

def bench(y):
    q=raw[(raw.code=='0050')&(raw.date.dt.year==y)].sort_values('date');return float(q.adj_close.iloc[-1]/q.adj_close.iloc[0]-1)

def sim(y,name):
    d=raw[raw.date.dt.year==y].copy();dates=sorted(d.date.unique());by={dt:x.set_index('code',drop=False) for dt,x in d.groupby('date')};m,s=entry_mask_score(d,name);c=d[base.loc[d.index]&m].copy();c['score']=s.loc[c.index];c['score_pct']=c.groupby('date').score.rank(pct=True);c=c[c.score_pct>=.80];sig={dt:x.sort_values(['score_pct','score','liq_pct'],ascending=False) for dt,x in c.groupby('date')}
    cash=INIT;pos={};pend_buy=[];pend_sell=set();pnls=[];navs=[]
    for i,dt in enumerate(dates):
        day=by[dt]
        for code,p in list(pos.items()):
            if code in day.index and bool(day.loc[code].action_flag):p['shares']=max(int(math.floor(p['shares']*float(day.loc[code].bridge))),0)
        for code in list(pend_sell):
            if code not in pos or code not in day.index:continue
            p=pos.pop(code);px=float(day.loc[code].open)*(1-SLIP);pro=p['shares']*px*(1-FEE-TAX);cash+=pro;pnls.append(pro-p['cost'])
        pend_sell=set()
        for o in pend_buy:
            code=o['code']
            if code in pos or len(pos)>=MAX_POS or code not in day.index:continue
            r=day.loc[code]
            if bool(r.action_flag) or float(r.low)>o['limit']:continue
            px=min(float(r.open)*(1+SLIP),o['limit']);budget=min(INIT*MAX_W,cash/(1+FEE));sh=int(budget//px)
            if sh<=0:continue
            cost=sh*px*(1+FEE)
            if cost<=cash:cash-=cost;pos[code]={'shares':sh,'cost':cost,'entry_adj':px*float(r.adj_factor),'held':0}
        pend_buy=[]
        mv=0.
        for code,p in pos.items():
            if code in day.index:mv+=p['shares']*float(day.loc[code].close);p['held']+=1
        nav=cash+mv;navs.append(nav)
        for code,p in pos.items():
            if code in day.index and should_exit(day.loc[code],p,name):pend_sell.add(code)
        slots=MAX_POS-len(pos)
        if slots>0:
            for _,r in sig.get(dt,pd.DataFrame()).head(slots).iterrows():
                if r.code not in pos:pend_buy.append({'code':r.code,'limit':float(r.close)*(1+SLIP)})
    last=by[dates[-1]]
    for code,p in list(pos.items()):
        if code in last.index:
            px=float(last.loc[code].close)*(1-SLIP);pro=p['shares']*px*(1-FEE-TAX);cash+=pro;pnls.append(pro-p['cost'])
    n=np.array(navs or [INIT]);dd=float(np.min(n/np.maximum.accumulate(n)-1));t=np.array(pnls);gp=t[t>0].sum();gl=-t[t<0].sum();pf=float(gp/gl) if gl>0 else (999. if gp>0 else 0.);rr=float(cash/INIT-1);bb=bench(y)
    return {'variant':name,'year':y,'return':rr,'benchmark_return':bb,'alpha':rr-bb,'max_dd':dd,'trades':len(t),'win_rate':float((t>0).mean()) if len(t) else 0.,'pf':pf}

rows=[sim(y,v) for v in VARIANTS for y in DISC+[OOS1,OOS2]];r=pd.DataFrame(rows);r.to_csv(OUT/'yearly.csv',index=False);disc=r[r.year.isin(DISC)].groupby('variant').agg(alpha_mean=('alpha','mean'),alpha_years=('alpha',lambda s:int((s>0).sum())),ret_mean=('return','mean'),dd_min=('max_dd','min'),trades=('trades','sum'),pf_mean=('pf','mean')).reset_index();disc['discovery_pass']=(disc.alpha_mean>0)&(disc.alpha_years>=2)&(disc.ret_mean>0)&(disc.dd_min>=-.25)&(disc.trades>=12)&(disc.pf_mean>1.05);disc.to_csv(OUT/'discovery.csv',index=False);ps=disc[disc.discovery_pass].sort_values(['alpha_mean','pf_mean'],ascending=False);selected=None if ps.empty else str(ps.iloc[0].variant)
if selected:
    a=r[(r.variant==selected)&(r.year==OOS1)].iloc[0];z=r[(r.variant==selected)&(r.year==OOS2)].iloc[0];o1=bool(a.alpha>0 and a['return']>0 and a.pf>1 and a.max_dd>=-.25 and a.trades>=3);o2=bool(o1 and z.alpha>0 and z['return']>0 and z.pf>1 and z.max_dd>=-.25 and z.trades>=3)
else:o1=o2=False
manifest={'stage':'15_winner_persistence','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'future_labels_used':False,'benchmark':'0050','benchmark_gate_required':True,'discovery_years':DISC,'oos1':OOS1,'oos2':OOS2,'oos1_used_for_selection':False,'oos2_used_for_selection':False,'variants':VARIANTS,'selected':selected,'oos1_pass':o1,'oos2_pass':o2,'entry_family':'relative_strength_plus_absolute_uptrend','exit_family':'causal_trend_break_or_rel_strength_break_or_market_off','max_hold_days':MAX_HOLD,'hard_stop':-.12,'corporate_action_method':'inferred_bridge_0p82_1p18_same_as_stage7','t_plus_1':True,'integer_shares':True,'shared_cash':True};(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('STAGE15_COMPLETE');print(json.dumps(manifest,indent=2));print('\nDISCOVERY');print(disc.to_string(index=False));print('\nYEARLY');print(r.to_string(index=False))