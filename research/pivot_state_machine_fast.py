import json
from pathlib import Path
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]
DOLD=R/'data/history/2007-2019/raw'; DNEW=R/'data/history/2020-2025'
OUT=R/'research/results/pivot_state_machine_fast_20261025'; OUT.mkdir(parents=True,exist_ok=True)

def pdate(s):
    z=s.astype(str).str.replace(r'\.0$','',regex=True).str.strip()
    if z.str.fullmatch(r'\d{8}').mean()>.8: return pd.to_datetime(z,format='%Y%m%d',errors='coerce')
    return pd.to_datetime(s,errors='coerce')

def load_px():
    a=[]
    for y in range(2007,2020):
        x=pd.read_csv(DOLD/f'yearly_{y}.zip',compression='zip',usecols=['date','code','open','high','low','close','volume'],low_memory=False)
        x['date']=pdate(x.date); x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False); a.append(x)
    for y in range(2020,2026):
        x=pd.read_parquet(DNEW/f'ohlcv_{y}.parquet')[['date','code','open','high','low','close','volume']]
        x['date']=pdate(x.date); x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False); a.append(x)
    x=pd.concat(a,ignore_index=True)
    for c in ['open','high','low','close','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna().sort_values(['code','date']).drop_duplicates(['code','date'],keep='last').reset_index(drop=True)

def load_inst():
    x=pd.read_parquet(DNEW/'institutional_2020_2025.parquet')
    x['date']=pdate(x.date); x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
    x['trust_net']=pd.to_numeric(x.trust_net,errors='coerce').fillna(0); x=x.sort_values(['code','date']); g=x.groupby('code')
    x['t5']=g.trust_net.transform(lambda s:s.rolling(5,min_periods=5).sum()); x['t20']=g.trust_net.transform(lambda s:s.rolling(20,min_periods=20).sum())
    return {(r.date,r.code):(r.t5,r.t20) for r in x[['date','code','t5','t20']].itertuples(index=False)}

def kd(b):
    lo=b.low.rolling(9,min_periods=9).min(); hi=b.high.rolling(9,min_periods=9).max(); rsv=(b.close-lo)/(hi-lo).replace(0,np.nan)*100
    ks=[]; ds=[]; k=d=50.0
    for v in rsv:
        if np.isfinite(v): k=(2*k+v)/3; d=(2*d+k)/3; ks.append(k); ds.append(d)
        else: ks.append(np.nan); ds.append(np.nan)
    q=b.copy(); q['K']=ks; q['D']=ds; return q

def period_values(g,freq):
    q=g.set_index('date').resample(freq).agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')).dropna(); q=kd(q)
    q[['K','D']]=q[['K','D']].shift(1); return {d:(r.K,r.D) for d,r in q[['K','D']].iterrows()}

def enrich(g):
    g=g.copy().reset_index(drop=True); c,o,h,l,v=g.close,g.open,g.high,g.low,g.volume
    g['ret1']=c/c.shift(1)-1
    for n in [5,10,20,60,120,240]: g[f'ma{n}']=c.rolling(n,min_periods=n).mean()
    g['v20']=v.rolling(20,min_periods=15).mean()
    mn=c.rolling(1200,min_periods=900).min(); mx=c.rolling(1200,min_periods=900).max(); g['pos5']=(c-mn)/(mx-mn).replace(0,np.nan)*100
    spread=(pd.concat([g.ma5,g.ma10,g.ma20],axis=1).max(axis=1)-pd.concat([g.ma5,g.ma10,g.ma20],axis=1).min(axis=1))/c; g['spread']=spread
    l1=l.shift(45).rolling(20,min_periods=15).min(); l2=l.shift(8).rolling(25,min_periods=18).min(); midh=h.shift(25).rolling(20,min_periods=15).max()
    w=(np.abs(l1/l2-1)<=.08)&(midh/np.minimum(l1,l2)-1>=.07)&(c>g.ma20)
    rnd=(g.ma60.shift(20)<g.ma60.shift(40))&(g.ma60>=g.ma60.shift(10))&(c>g.ma20)&(g.ma20>g.ma20.shift(5))
    ls=l.shift(50).rolling(18,min_periods=12).min(); hd=l.shift(28).rolling(18,min_periods=12).min(); rs=l.shift(8).rolling(16,min_periods=10).min(); neck=h.shift(8).rolling(42,min_periods=25).max()
    ihs=(np.abs(ls/rs-1)<=.10)&(hd<np.minimum(ls,rs)*.97)&(hd>np.minimum(ls,rs)*.78)&(c>g.ma20)&(c<=neck*1.05)
    g['bottom']=w|rnd|ihs
    wm=period_values(g,'W-FRI'); mm=period_values(g,'ME'); wk=[]; mo=[]
    for d in g.date:
        wk.append(wm.get(d.to_period('W-FRI').end_time.normalize(),(np.nan,np.nan))); mo.append(mm.get(d.to_period('M').end_time.normalize(),(np.nan,np.nan)))
    g[['wK','wD']]=pd.DataFrame(wk,index=g.index); g[['mK','mD']]=pd.DataFrame(mo,index=g.index)
    g['base_long']=(g.pos5<=30)&((g.wK<=40)|(g.mK<=40)); g['base_struct']=(g.pos5<=30)&g.bottom; g['base_full']=g.base_long&g.bottom
    raw=(v>=1.5*g.v20)&(c>o)&(c>g.ma20)&(g.spread<.05)&(g.pos5<=35)&((c/g.ma20-1).abs()<.08)
    g['ignite']=raw & ~raw.shift(1).rolling(10,min_periods=1).max().fillna(False).astype(bool)
    g['break60']=(c>g.ma60)&(c.shift(1)<=g.ma60.shift(1))&(v>=1.3*g.v20)&(c>o)&(g.pos5<=45)
    body=(c-o).abs(); rng=(h-l).replace(0,np.nan); lower=np.minimum(c,o)-l
    g['rev']=((c>o)&(c>=l+.60*rng))|((lower>=1.5*np.maximum(body,.002*c))&(c>=l+.60*rng))
    return g

def out(g,sig,ei):
    if ei>=len(g): return None
    e=float(g.loc[ei,'open'])
    if not np.isfinite(e) or e<=0:return None
    fw=g.iloc[ei:min(ei+30,len(g))]
    if len(fw)==0 or (fw.ret1.abs()>.15).any():return None
    r={'signal_date':g.loc[sig,'date'],'entry_date':g.loc[ei,'date'],'entry':e}
    for n in [5,10,20,30]:
        w=g.iloc[ei:min(ei+n,len(g))]
        if len(w):r[f'r{n}']=w.iloc[-1].close/e-1;r[f'mfe{n}']=w.high.max()/e-1;r[f'mae{n}']=w.low.min()/e-1
    w=g.iloc[ei:min(ei+20,len(g))];u=np.where(w.high.to_numpy()>=e*1.10)[0];d=np.where(w.low.to_numpy()<=e*.95)[0];ui=u[0] if len(u) else 999;di=d[0] if len(d) else 999;r['ft']=1 if ui<di else(-1 if di<ui else 0)
    return r

def chip_modes(inst,d,code):
    t5,t20=inst.get((d,code),(np.nan,np.nan)); z=['NONE']
    if np.isfinite(t5) and t5>0:z.append('T5')
    if np.isfinite(t5) and np.isfinite(t20) and t5>0 and t20>0:z.append('T5T20')
    return z

def main():
    px=load_px();inst=load_inst(); rows=[]; bases=['base_long','base_struct','base_full']; last={}
    for code,rawg in px.groupby('code',sort=False):
        if len(rawg)<1000:continue
        g=enrich(rawg); years=g.date.dt.year
        acts=np.where(((years>=2021)&(years<=2025)&(g.ignite|g.break60)).to_numpy())[0]
        for a in acts:
            if a+1>=len(g): continue
            bactive={b:bool(g.loc[max(0,a-20):a,b].fillna(False).any()) for b in bases}
            if not any(bactive.values()):continue
            modes=chip_modes(inst,g.loc[a,'date'],code)
            if bool(g.loc[a,'ignite']):
                sig=a;ei=a+1;baseout=out(g,sig,ei)
                if baseout:
                    for b,on in bactive.items():
                        if not on:continue
                        for ch in modes:
                            key=(code,b,'IGNITION',ch)
                            if ei-last.get(key,-999)<15:continue
                            r=dict(baseout);r.update(code=code,base=b,entry_type='IGNITION',chip=ch,activation_date=g.loc[a,'date']);rows.append(r);last[key]=ei
            if bool(g.loc[a,'break60']):
                act_close=float(g.loc[a,'close']);prelow=float(g.loc[max(0,a-60):a,'low'].min());sw=max(act_close-prelow,0);fibs=[act_close-.382*sw,act_close-.5*sw,act_close-.618*sw] if sw>0 else[];sig=None
                for j in range(a+3,min(a+20,len(g)-2)+1):
                    lvls=[g.loc[j,'ma20'],g.loc[j,'ma60'],act_close]+fibs;support=any(np.isfinite(x) and x>0 and (abs(g.loc[j,'low']/x-1)<=.025 or abs(g.loc[j,'close']/x-1)<=.025) for x in lvls)
                    vol=np.isfinite(g.loc[j,'v20']) and g.loc[j,'volume']<=.9*g.loc[j,'v20']; maok=np.isfinite(g.loc[j,'ma60']) and np.isfinite(g.loc[max(0,j-10),'ma60']) and g.loc[j,'ma60']>=g.loc[max(0,j-10),'ma60']*.99
                    if support and vol and maok and bool(g.loc[j,'rev']):sig=j;break
                if sig is None:continue
                ei=sig+1;baseout=out(g,sig,ei)
                if baseout:
                    for b,on in bactive.items():
                        if not on:continue
                        for ch in modes:
                            key=(code,b,'RETEST',ch)
                            if ei-last.get(key,-999)<15:continue
                            r=dict(baseout);r.update(code=code,base=b,entry_type='RETEST',chip=ch,activation_date=g.loc[a,'date'],retest_delay=sig-a);rows.append(r);last[key]=ei
    E=pd.DataFrame(rows);E.to_csv(OUT/'events.csv',index=False)
    ss=[]
    for (b,e,ch),z in E.groupby(['base','entry_type','chip']):
        row={'base':b,'entry_type':e,'chip':ch}
        for p,mask in [('tr',(z.signal_date.dt.year>=2021)&(z.signal_date.dt.year<=2023)),('va',z.signal_date.dt.year==2024),('ho',z.signal_date.dt.year==2025)]:
            q=z[mask];row[f'n_{p}']=len(q)
            for n in [5,10,20,30]:
                if len(q):row[f'r{n}_{p}']=q[f'r{n}'].mean();row[f'win{n}_{p}']=(q[f'r{n}']>0).mean();row[f'mae{n}_{p}']=q[f'mae{n}'].median();row[f'mfe{n}_{p}']=q[f'mfe{n}'].median()
            if len(q):row[f'up10_{p}']=(q.ft==1).mean();row[f'dn5_{p}']=(q.ft==-1).mean()
        ss.append(row)
    S=pd.DataFrame(ss);S['pre_holdout_pass']=(S.n_tr>=25)&(S.r20_tr>.005)&(S.win20_tr>=.53)&(S.n_va>=8)&(S.r20_va>.005)&(S.win20_va>=.52);S=S.sort_values(['pre_holdout_pass','r20_va','r20_tr'],ascending=False);S.to_csv(OUT/'summary.csv',index=False);S[S.pre_holdout_pass].sort_values('r20_ho',ascending=False).to_csv(OUT/'selected_holdout.csv',index=False)
    meta={'rows_total':len(px),'stocks_tested':px.code.nunique(),'history':'2007-2025','split':'2021-23 train / 2024 validation / 2025 holdout','rules':18,'selection_does_not_use_2025':True,'entry_A':'low-stage watch within 20d -> first 1.5xV20 bullish MA20 ignition with <5% MA5/10/20 spread -> T+1','entry_B':'low-stage watch within 20d -> 1.3xV20 MA60 break -> 3-20d support retest (MA20/MA60/break/Fib) + <=0.9xV20 + MA60 not down + reversal -> T+1'};(OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));print(json.dumps(meta,ensure_ascii=False,indent=2));print(S.to_string(index=False))
if __name__=='__main__':main()
