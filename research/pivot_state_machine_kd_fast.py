import json
from pathlib import Path
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];D0=R/'data/history/2007-2019/raw';D1=R/'data/history/2020-2025';O=R/'research/results/pivot_state_machine_kd_fast_20261025';O.mkdir(parents=True,exist_ok=True)
def pdate(s):
 z=s.astype(str).str.replace(r'\.0$','',regex=True).str.strip();return pd.to_datetime(z,format='%Y%m%d',errors='coerce') if z.str.fullmatch(r'\d{8}').mean()>.8 else pd.to_datetime(s,errors='coerce')
def load():
 a=[]
 for y in range(2016,2020):
  x=pd.read_csv(D0/f'yearly_{y}.zip',compression='zip',usecols=['date','code','open','high','low','close','volume'],low_memory=False);x.date=pdate(x.date);x.code=x.code.astype(str).str.extract(r'(\d{4})',expand=False);a.append(x)
 for y in range(2020,2026):
  x=pd.read_parquet(D1/f'ohlcv_{y}.parquet')[['date','code','open','high','low','close','volume']];x.date=pdate(x.date);x.code=x.code.astype(str).str.extract(r'(\d{4})',expand=False);a.append(x)
 x=pd.concat(a,ignore_index=True)
 for c in ['open','high','low','close','volume']:x[c]=pd.to_numeric(x[c],errors='coerce')
 return x.dropna().sort_values(['code','date']).drop_duplicates(['code','date'],keep='last').reset_index(drop=True)
def inst():
 x=pd.read_parquet(D1/'institutional_2020_2025.parquet');x.date=pdate(x.date);x.code=x.code.astype(str).str.extract(r'(\d{4})',expand=False);x.trust_net=pd.to_numeric(x.trust_net,errors='coerce').fillna(0);x=x.sort_values(['code','date']);g=x.groupby('code');x['t5']=g.trust_net.transform(lambda s:s.rolling(5,min_periods=5).sum());x['t20']=g.trust_net.transform(lambda s:s.rolling(20,min_periods=20).sum());return {(r.date,r.code):(r.t5,r.t20) for r in x[['date','code','t5','t20']].itertuples(index=False)}
def kd(q):
 lo=q.low.rolling(9,min_periods=9).min();hi=q.high.rolling(9,min_periods=9).max();r=(q.close-lo)/(hi-lo).replace(0,np.nan)*100;ks=[];ds=[];k=d=50.
 for v in r:
  if np.isfinite(v):k=(2*k+v)/3;d=(2*d+k)/3;ks.append(k);ds.append(d)
  else:ks.append(np.nan);ds.append(np.nan)
 q=q.copy();q['K']=ks;q['D']=ds;return q
def pvals(g,freq):
 q=g.set_index('date').resample(freq).agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')).dropna();q=kd(q);q[['K','D']]=q[['K','D']].shift(1);return q[['K','D']]
def enrich(g):
 g=g.reset_index(drop=True).copy();c,o,h,l,v=g.close,g.open,g.high,g.low,g.volume;g['ret1']=c/c.shift(1)-1
 for n in [5,10,20,60]:g[f'ma{n}']=c.rolling(n,min_periods=n).mean()
 g['v20']=v.rolling(20,min_periods=15).mean();mn=c.rolling(1200,min_periods=900).min();mx=c.rolling(1200,min_periods=900).max();g['pos5']=(c-mn)/(mx-mn).replace(0,np.nan)*100;g['spread']=(pd.concat([g.ma5,g.ma10,g.ma20],axis=1).max(axis=1)-pd.concat([g.ma5,g.ma10,g.ma20],axis=1).min(axis=1))/c
 wq=pvals(g,'W-FRI');mq=pvals(g,'ME');wi=g.date.dt.to_period('W-FRI').dt.end_time.dt.normalize();mi=g.date.dt.to_period('M').dt.end_time.dt.normalize();g['wK']=wq.K.reindex(wi).to_numpy();g['mK']=mq.K.reindex(mi).to_numpy()
 l1=l.shift(45).rolling(20,min_periods=15).min();l2=l.shift(8).rolling(25,min_periods=18).min();midh=h.shift(25).rolling(20,min_periods=15).max();w=(np.abs(l1/l2-1)<=.08)&(midh/np.minimum(l1,l2)-1>=.07)&(c>g.ma20);rnd=(g.ma60.shift(20)<g.ma60.shift(40))&(g.ma60>=g.ma60.shift(10))&(c>g.ma20)&(g.ma20>g.ma20.shift(5));ls=l.shift(50).rolling(18,min_periods=12).min();hd=l.shift(28).rolling(18,min_periods=12).min();rs=l.shift(8).rolling(16,min_periods=10).min();neck=h.shift(8).rolling(42,min_periods=25).max();ihs=(np.abs(ls/rs-1)<=.10)&(hd<np.minimum(ls,rs)*.97)&(hd>np.minimum(ls,rs)*.78)&(c>g.ma20)&(c<=neck*1.05);g['bottom']=w|rnd|ihs
 g['base_kd']=(g.pos5<=30)&((g.wK<=40)|(g.mK<=40));g['base_full']=g.base_kd&g.bottom
 raw=(v>=1.5*g.v20)&(c>o)&(c>g.ma20)&(g.spread<.05)&(g.pos5<=35)&((c/g.ma20-1).abs()<.08);g['ignite']=raw&~raw.shift(1).rolling(10,min_periods=1).max().fillna(False).astype(bool);g['break60']=(c>g.ma60)&(c.shift(1)<=g.ma60.shift(1))&(v>=1.3*g.v20)&(c>o)&(g.pos5<=45);body=(c-o).abs();rng=(h-l).replace(0,np.nan);lower=np.minimum(c,o)-l;g['rev']=((c>o)&(c>=l+.6*rng))|((lower>=1.5*np.maximum(body,.002*c))&(c>=l+.6*rng));return g
def outcome(g,s,e):
 if e>=len(g):return None
 p=float(g.loc[e,'open']);fw=g.iloc[e:min(e+30,len(g))]
 if not np.isfinite(p) or p<=0 or len(fw)==0 or (fw.ret1.abs()>.15).any():return None
 r={'signal_date':g.loc[s,'date'],'entry_date':g.loc[e,'date'],'entry':p}
 for n in [5,10,20,30]:
  w=g.iloc[e:min(e+n,len(g))];r[f'r{n}']=w.iloc[-1].close/p-1;r[f'mfe{n}']=w.high.max()/p-1;r[f'mae{n}']=w.low.min()/p-1
 w=g.iloc[e:min(e+20,len(g))];u=np.where(w.high.to_numpy()>=p*1.10)[0];d=np.where(w.low.to_numpy()<=p*.95)[0];ui=u[0] if len(u) else 999;di=d[0] if len(d) else 999;r['ft']=1 if ui<di else(-1 if di<ui else 0);return r
def modes(I,d,c):
 a,b=I.get((d,c),(np.nan,np.nan));z=['NONE']
 if np.isfinite(a) and a>0:z.append('T5')
 if np.isfinite(a) and np.isfinite(b) and a>0 and b>0:z.append('T5T20')
 return z
def main():
 X=load();I=inst();rows=[];last={};bases=['base_kd','base_full']
 for code,rg in X.groupby('code',sort=False):
  if len(rg)<900:continue
  g=enrich(rg);Y=g.date.dt.year;A=np.where(((Y>=2021)&(Y<=2025)&(g.ignite|g.break60)).to_numpy())[0]
  for a in A:
   if a+1>=len(g):continue
   ba={b:bool(g.loc[max(0,a-20):a,b].fillna(False).any()) for b in bases}
   if not any(ba.values()):continue
   M=modes(I,g.loc[a,'date'],code)
   if g.loc[a,'ignite']:
    q=outcome(g,a,a+1)
    if q:
     for b,on in ba.items():
      if on:
       for ch in M:
        k=(code,b,'IGNITION',ch)
        if a+1-last.get(k,-999)>=15:rows.append(dict(q,code=code,base=b,entry_type='IGNITION',chip=ch));last[k]=a+1
   if g.loc[a,'break60']:
    ac=float(g.loc[a,'close']);pl=float(g.loc[max(0,a-60):a,'low'].min());sw=max(ac-pl,0);fs=[ac-.382*sw,ac-.5*sw,ac-.618*sw];s=None
    for j in range(a+3,min(a+20,len(g)-2)+1):
     lv=[g.loc[j,'ma20'],g.loc[j,'ma60'],ac]+fs;sup=any(np.isfinite(x) and x>0 and (abs(g.loc[j,'low']/x-1)<=.025 or abs(g.loc[j,'close']/x-1)<=.025) for x in lv);vc=np.isfinite(g.loc[j,'v20']) and g.loc[j,'volume']<=.9*g.loc[j,'v20'];mu=np.isfinite(g.loc[j,'ma60']) and np.isfinite(g.loc[max(0,j-10),'ma60']) and g.loc[j,'ma60']>=g.loc[max(0,j-10),'ma60']*.99
     if sup and vc and mu and g.loc[j,'rev']:s=j;break
    if s is not None:
     q=outcome(g,s,s+1)
     if q:
      for b,on in ba.items():
       if on:
        for ch in M:
         k=(code,b,'RETEST',ch)
         if s+1-last.get(k,-999)>=15:rows.append(dict(q,code=code,base=b,entry_type='RETEST',chip=ch,retest_delay=s-a));last[k]=s+1
 E=pd.DataFrame(rows);E.to_csv(O/'events.csv',index=False);ss=[]
 for (b,e,ch),z in E.groupby(['base','entry_type','chip']):
  r={'base':b,'entry_type':e,'chip':ch}
  for p,m in [('tr',(z.signal_date.dt.year>=2021)&(z.signal_date.dt.year<=2023)),('va',z.signal_date.dt.year==2024),('ho',z.signal_date.dt.year==2025)]:
   q=z[m];r[f'n_{p}']=len(q)
   for n in [5,10,20,30]:
    if len(q):r[f'r{n}_{p}']=q[f'r{n}'].mean();r[f'win{n}_{p}']=(q[f'r{n}']>0).mean();r[f'mae{n}_{p}']=q[f'mae{n}'].median();r[f'mfe{n}_{p}']=q[f'mfe{n}'].median()
   if len(q):r[f'up10_{p}']=(q.ft==1).mean();r[f'dn5_{p}']=(q.ft==-1).mean()
  ss.append(r)
 S=pd.DataFrame(ss);S['pass']=(S.n_tr>=25)&(S.r20_tr>.005)&(S.win20_tr>=.53)&(S.n_va>=8)&(S.r20_va>.005)&(S.win20_va>=.52);S.to_csv(O/'summary.csv',index=False);S[S['pass']].sort_values('r20_ho',ascending=False).to_csv(O/'selected_holdout.csv',index=False);(O/'meta.json').write_text(json.dumps({'history':'2016-2025','5y warmup':True,'weekly_monthly_KD':'completed-period values shifted one full period','rules':12,'2025_used_for_selection':False},indent=2));print(S.sort_values(['pass','r20_va'],ascending=False).to_string(index=False))
if __name__=='__main__':main()
