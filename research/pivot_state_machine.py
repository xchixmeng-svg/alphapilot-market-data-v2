import json
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
DOLD = R/'data/history/2007-2019/raw'
DNEW = R/'data/history/2020-2025'
OUT = R/'research/results/pivot_state_machine_20261025'
OUT.mkdir(parents=True, exist_ok=True)


def pdate(s):
    z=s.astype(str).str.replace(r'\.0$','',regex=True).str.strip()
    if z.str.fullmatch(r'\d{8}').mean()>.8:
        return pd.to_datetime(z,format='%Y%m%d',errors='coerce')
    return pd.to_datetime(s,errors='coerce')


def load_px():
    parts=[]
    for y in range(2007,2020):
        p=DOLD/f'yearly_{y}.zip'
        x=pd.read_csv(p,compression='zip',usecols=['date','code','open','high','low','close','volume'],low_memory=False)
        x['date']=pdate(x.date); x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
        parts.append(x)
    for y in range(2020,2026):
        x=pd.read_parquet(DNEW/f'ohlcv_{y}.parquet')[['date','code','open','high','low','close','volume']]
        x['date']=pdate(x.date); x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
        parts.append(x)
    x=pd.concat(parts,ignore_index=True)
    for c in ['open','high','low','close','volume']:
        x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x.dropna().sort_values(['code','date']).drop_duplicates(['code','date'],keep='last').reset_index(drop=True)
    return x


def load_inst():
    x=pd.read_parquet(DNEW/'institutional_2020_2025.parquet')
    x['date']=pdate(x.date); x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
    for c in ['foreign_net','trust_net','dealer_net']:
        x[c]=pd.to_numeric(x[c],errors='coerce').fillna(0)
    x=x.sort_values(['code','date']); g=x.groupby('code',group_keys=False)
    x['t5']=g.trust_net.transform(lambda s:s.rolling(5,min_periods=5).sum())
    x['t20']=g.trust_net.transform(lambda s:s.rolling(20,min_periods=20).sum())
    return x[['date','code','t5','t20']]


def kd(b):
    lo=b.low.rolling(9,min_periods=9).min(); hi=b.high.rolling(9,min_periods=9).max()
    rsv=(b.close-lo)/(hi-lo).replace(0,np.nan)*100
    K=[]; D=[]; pk=pdv=50.0
    for v in rsv:
        if np.isfinite(v):
            pk=(2*pk+v)/3; pdv=(2*pdv+pk)/3
            K.append(pk); D.append(pdv)
        else:
            K.append(np.nan); D.append(np.nan)
    q=b.copy(); q['K']=K; q['D']=D
    return q


def completed_period_kd(g,freq):
    q=g.set_index('date').resample(freq).agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')).dropna()
    q=kd(q)
    # Strict PIT: completed weekly/monthly value is available only in the next period.
    q[['K','D']]=q[['K','D']].shift(1)
    return q[['K','D']]


def enrich(g):
    g=g.copy().reset_index(drop=True); c,o,h,l,v=g.close,g.open,g.high,g.low,g.volume
    g['ret1']=c/c.shift(1)-1
    for n in [5,10,20,60,120,240,1200,2400]:
        mp=n if n<=20 else max(20,int(n*.8))
        g[f'ma{n}']=c.rolling(n,min_periods=mp).mean()
    g['v20']=v.rolling(20,min_periods=15).mean(); g['v5']=v.rolling(5,min_periods=4).mean()
    spread=(pd.concat([g.ma5,g.ma10,g.ma20],axis=1).max(axis=1)-pd.concat([g.ma5,g.ma10,g.ma20],axis=1).min(axis=1))/c
    g['ma_spread']=spread
    min5=c.rolling(1200,min_periods=900).min(); max5=c.rolling(1200,min_periods=900).max()
    min10=c.rolling(2400,min_periods=1800).min(); max10=c.rolling(2400,min_periods=1800).max()
    g['pos5']=(c-min5)/(max5-min5).replace(0,np.nan)*100
    g['pos10']=(c-min10)/(max10-min10).replace(0,np.nan)*100
    # bottom-structure proxies, all lagged/current only
    l1=l.shift(45).rolling(20,min_periods=15).min(); l2=l.shift(8).rolling(25,min_periods=18).min(); midh=h.shift(25).rolling(20,min_periods=15).max()
    g['w_setup']=(np.abs(l1/l2-1)<=0.08)&(midh/np.minimum(l1,l2)-1>=0.07)&(c>g.ma20)
    g['round_turn']=(g.ma60.shift(20)<g.ma60.shift(40))&(g.ma60>=g.ma60.shift(10))&(c>g.ma20)&(g.ma20>g.ma20.shift(5))
    ls=l.shift(50).rolling(18,min_periods=12).min(); hd=l.shift(28).rolling(18,min_periods=12).min(); rs=l.shift(8).rolling(16,min_periods=10).min()
    neck=h.shift(8).rolling(42,min_periods=25).max()
    g['ihs_setup']=(np.abs(ls/rs-1)<=0.10)&(hd<np.minimum(ls,rs)*0.97)&(hd>np.minimum(ls,rs)*0.78)&(c>g.ma20)&(c<=neck*1.05)
    g['bottom_struct']=g.w_setup|g.round_turn|g.ihs_setup
    # Completed weekly/monthly KD mapped to daily.
    wk=completed_period_kd(g,'W-FRI'); mo=completed_period_kd(g,'ME')
    wkmap={d:(r.K,r.D) for d,r in wk.iterrows()}; momap={d:(r.K,r.D) for d,r in mo.iterrows()}
    wv=[]; mv=[]
    for d in g.date:
        wv.append(wkmap.get(d.to_period('W-FRI').end_time.normalize(),(np.nan,np.nan)))
        mv.append(momap.get(d.to_period('M').end_time.normalize(),(np.nan,np.nan)))
    g[['wK','wD']]=pd.DataFrame(wv,index=g.index); g[['mK','mD']]=pd.DataFrame(mv,index=g.index)
    g['base_long']=(g.pos5<=30)&((g.wK<=40)|(g.mK<=40))
    g['base_struct']=(g.pos5<=30)&g.bottom_struct
    g['base_full']=(g.pos5<=30)&g.bottom_struct&((g.wK<=40)|(g.mK<=40))
    # Entry A: first volume-price ignition while still low and short MAs not stretched.
    raw_ign=(v>=1.5*g.v20)&(c>o)&(c>g.ma20)&(g.ma_spread<0.05)&(g.pos5<=35)&((c/g.ma20-1).abs()<0.08)
    prior_ign=raw_ign.shift(1).rolling(10,min_periods=1).max().fillna(False).astype(bool)
    g['ignite']=raw_ign & ~prior_ign
    # Activation B: volume-backed first cross above MA60 from below.
    cross60=(c>g.ma60)&(c.shift(1)<=g.ma60.shift(1))
    g['ma60_break']=cross60&(v>=1.3*g.v20)&(c>o)&(g.pos5<=45)
    # Daily reversal confirmation for a future retest day.
    body=(c-o).abs(); rng=(h-l).replace(0,np.nan); lower=np.minimum(c,o)-l
    g['reversal_day']=((c>o)&(c>=l+0.60*rng)) | ((lower>=1.5*np.maximum(body,0.002*c))&(c>=l+0.60*rng))
    return g


def chip_ok(inst, date, code, mode):
    t5,t20=inst.get((date,code),(np.nan,np.nan))
    if mode=='NONE': return True
    if mode=='T5': return np.isfinite(t5) and t5>0
    if mode=='T5T20': return np.isfinite(t5) and np.isfinite(t20) and t5>0 and t20>0
    return False


def outcomes(g, signal_i, entry_i, meta):
    e=float(g.loc[entry_i,'open'])
    if not np.isfinite(e) or e<=0: return None
    fw=g.iloc[entry_i:min(entry_i+30,len(g))]
    if len(fw)==0 or (fw.ret1.abs()>0.15).any(): return None
    r=dict(meta); r.update(signal_date=g.loc[signal_i,'date'],entry_date=g.loc[entry_i,'date'],entry=e)
    for n in [5,10,20,30]:
        w=g.iloc[entry_i:min(entry_i+n,len(g))]
        if len(w):
            r[f'r{n}']=float(w.iloc[-1].close/e-1); r[f'mfe{n}']=float(w.high.max()/e-1); r[f'mae{n}']=float(w.low.min()/e-1)
    w=g.iloc[entry_i:min(entry_i+20,len(g))]
    up=np.where(w.high.to_numpy()>=e*1.10)[0]; dn=np.where(w.low.to_numpy()<=e*.95)[0]
    ui=up[0] if len(up) else 999; di=dn[0] if len(dn) else 999
    r['ft']=1 if ui<di else (-1 if di<ui else 0)
    return r


def build_events(by, inst, base_col, entry_type, chip):
    rows=[]
    for code,g in by.items():
        # Only evaluated signals 2021-2025; older history is warmup.
        candidate=np.where((g.date.dt.year>=2021)&(g.date.dt.year<=2025)&g[base_col].fillna(False).to_numpy())[0]
        last_entry=-999
        for i in candidate:
            if i-last_entry<15: continue
            # Public descriptions imply a low-stage watch can precede activation; cap watch window at 20 sessions.
            stop=min(i+20,len(g)-2); act=None
            for j in range(i,stop+1):
                if entry_type=='IGNITION':
                    if bool(g.loc[j,'ignite']) and chip_ok(inst,g.loc[j,'date'],code,chip): act=j; break
                elif entry_type=='RETEST':
                    if bool(g.loc[j,'ma60_break']) and chip_ok(inst,g.loc[j,'date'],code,chip): act=j; break
            if act is None: continue
            if entry_type=='IGNITION':
                sig=act; ei=sig+1
                r=outcomes(g,sig,ei,{'code':code,'base_date':g.loc[i,'date'],'activation_date':g.loc[act,'date'],'base':base_col,'entry_type':entry_type,'chip':chip,'delay':act-i})
                if r: rows.append(r); last_entry=ei
                continue
            # RETEST: after volume-backed MA60 break, wait 3-20 sessions for support retest + contraction + reversal.
            act_close=float(g.loc[act,'close'])
            prelow=float(g.loc[max(0,act-60):act,'low'].min()) if act>0 else np.nan
            swing=max(act_close-prelow,0) if np.isfinite(prelow) else 0
            fibs=[act_close-0.382*swing,act_close-0.5*swing,act_close-0.618*swing] if swing>0 else []
            sig=None
            for j in range(act+3,min(act+20,len(g)-2)+1):
                low=float(g.loc[j,'low']); close=float(g.loc[j,'close'])
                ma20=float(g.loc[j,'ma20']) if np.isfinite(g.loc[j,'ma20']) else np.nan
                ma60=float(g.loc[j,'ma60']) if np.isfinite(g.loc[j,'ma60']) else np.nan
                tests=[]
                for lvl in [ma20,ma60,act_close]+fibs:
                    if np.isfinite(lvl) and lvl>0: tests.append(abs(low/lvl-1)<=0.025 or abs(close/lvl-1)<=0.025)
                support=any(tests)
                vol_contract=np.isfinite(g.loc[j,'v20']) and g.loc[j,'volume']<=0.9*g.loc[j,'v20']
                ma60_not_bad=np.isfinite(ma60) and np.isfinite(g.loc[max(0,j-10),'ma60']) and ma60>=g.loc[max(0,j-10),'ma60']*0.99
                if support and vol_contract and ma60_not_bad and bool(g.loc[j,'reversal_day']):
                    sig=j; break
            if sig is None: continue
            ei=sig+1
            r=outcomes(g,sig,ei,{'code':code,'base_date':g.loc[i,'date'],'activation_date':g.loc[act,'date'],'base':base_col,'entry_type':entry_type,'chip':chip,'delay':act-i,'retest_delay':sig-act})
            if r: rows.append(r); last_entry=ei
    return pd.DataFrame(rows)


def add_stats(row,prefix,z):
    row[f'n_{prefix}']=len(z)
    for n in [5,10,20,30]:
        if len(z):
            row[f'r{n}_{prefix}']=z[f'r{n}'].mean(); row[f'win{n}_{prefix}']=(z[f'r{n}']>0).mean(); row[f'mae{n}_{prefix}']=z[f'mae{n}'].median(); row[f'mfe{n}_{prefix}']=z[f'mfe{n}'].median()
    if len(z):
        row[f'up10_{prefix}']=(z.ft==1).mean(); row[f'dn5_{prefix}']=(z.ft==-1).mean()


def main():
    px=load_px(); ins=load_inst()
    by={code:enrich(g) for code,g in px.groupby('code',sort=False) if len(g)>=1000}
    im=ins.set_index(['date','code'])[['t5','t20']]; inst={k:tuple(v) for k,v in zip(im.index,im.to_numpy())}
    bases=['base_long','base_struct','base_full']; entries=['IGNITION','RETEST']; chips=['NONE','T5','T5T20']
    summary=[]; holdouts=[]
    for b in bases:
      for e in entries:
       for ch in chips:
        ev=build_events(by,inst,b,e,ch)
        tr=ev[(ev.signal_date.dt.year>=2021)&(ev.signal_date.dt.year<=2023)] if len(ev) else ev
        va=ev[ev.signal_date.dt.year==2024] if len(ev) else ev
        ho=ev[ev.signal_date.dt.year==2025] if len(ev) else ev
        row={'base':b,'entry_type':e,'chip':ch}; add_stats(row,'tr',tr); add_stats(row,'va',va); add_stats(row,'ho',ho); summary.append(row)
        # pre-holdout gate only; 2025 never participates in selection
        if len(tr)>=25 and row.get('r20_tr',-9)>0.005 and row.get('win20_tr',0)>=.53 and len(va)>=8 and row.get('r20_va',-9)>0.005 and row.get('win20_va',0)>=.52 and len(ho):
            hh=ho.copy(); hh['rule']=f'{b}|{e}|{ch}'; holdouts.append(hh)
    S=pd.DataFrame(summary)
    S['pre_holdout_pass']=(S.n_tr>=25)&(S.r20_tr>.005)&(S.win20_tr>=.53)&(S.n_va>=8)&(S.r20_va>.005)&(S.win20_va>=.52)
    S=S.sort_values(['pre_holdout_pass','r20_va','r20_tr'],ascending=False)
    S.to_csv(OUT/'summary.csv',index=False)
    S[S.pre_holdout_pass].sort_values('r20_ho',ascending=False).to_csv(OUT/'selected_holdout.csv',index=False)
    if holdouts: pd.concat(holdouts,ignore_index=True).to_csv(OUT/'holdout_events.csv',index=False)
    meta={
      'rows_total':int(len(px)),'stocks_tested':int(len(by)),'history':'2007-2025','evaluation':'2021-2025',
      'rules_tested':int(len(S)),'split':{'train':'2021-2023','validation':'2024','holdout':'2025'},
      'base_defs':{'base_long':'5y range position <=30 and completed weekly K<=40 or monthly K<=40','base_struct':'5y position<=30 plus W/round/inverse-HS proxy','base_full':'both long-stage oscillator and bottom structure'},
      'entry_A':'first 10-session volume-price ignition: volume>=1.5x V20, bullish close above MA20, MA5/10/20 spread<5%, 5y position<=35%; T+1 open',
      'entry_B':'volume-backed MA60 cross (>=1.3x V20), then 3-20 sessions later support retest at MA20/MA60/break level/Fib 38.2-61.8, volume<=0.9x V20, MA60 not materially down, bullish/lower-wick reversal; T+1 open',
      'chips':['none','trust 5d net>0','trust 5d and 20d net>0'],
      'pit':'weekly/monthly KD shifted one full completed period; all signal features use data through signal close only; entry next open; 2025 excluded from selection.'
    }
    (OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(json.dumps(meta,ensure_ascii=False,indent=2)); print(S.to_string(index=False))

if __name__=='__main__': main()
