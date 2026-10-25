import json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]; D=R/'data/history/2020-2025'; B=R/'research/results/pivot5y_20261025'; O=R/'research/results/pivot_trust_strength_20261025'; O.mkdir(parents=True,exist_ok=True)
def pdate(s):
 z=s.astype(str).str.replace(r'\.0$','',regex=True).str.strip(); return pd.to_datetime(z,format='%Y%m%d',errors='coerce') if z.str.fullmatch(r'\d{8}').mean()>.8 else pd.to_datetime(s,errors='coerce')
def load():
 a=[]
 for y in range(2020,2026):
  x=pd.read_parquet(D/f'ohlcv_{y}.parquet'); x.date=pdate(x.date); x.code=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
  for c in ['open','high','low','close','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
  a.append(x[['date','code','open','high','low','close','volume']])
 x=pd.concat(a).dropna().sort_values(['code','date']).reset_index(drop=True); g=x.groupby('code',group_keys=False); x['prev']=g.close.shift(1); x['ret1']=x.close/x.prev-1; x['bull']=x.close>x.open; x['ma5']=g.close.transform(lambda s:s.rolling(5,min_periods=5).mean()); x['ma10']=g.close.transform(lambda s:s.rolling(10,min_periods=10).mean()); x['ma5_up']=x.ma5>g.ma5.shift(2); x['vol5']=g.volume.transform(lambda s:s.rolling(5,min_periods=3).sum()); x['anom']=x.ret1.abs()>.18
 ins=pd.read_parquet(D/'institutional_2020_2025.parquet'); ins.date=pdate(ins.date); ins.code=ins.code.astype(str).str.extract(r'(\d{4})',expand=False); 
 for c in ['trust_net','foreign_net','dealer_net']: ins[c]=pd.to_numeric(ins[c],errors='coerce').fillna(0)
 ins=ins.sort_values(['code','date']); ig=ins.groupby('code',group_keys=False); ins['t1']=ins.trust_net; ins['t3']=ig.trust_net.transform(lambda s:s.rolling(3,min_periods=1).sum()); ins['t5']=ig.trust_net.transform(lambda s:s.rolling(5,min_periods=1).sum()); ins['t5_prev']=ig.t5.shift(1); ins['f5']=ig.foreign_net.transform(lambda s:s.rolling(5,min_periods=1).sum())
 x=x.merge(ins[['date','code','t1','t3','t5','t5_prev','f5']],on=['date','code'],how='left').fillna({'t1':0,'t3':0,'t5':0,'t5_prev':0,'f5':0}); x['tratio']=x.t5/x.vol5.replace(0,np.nan); return x
def stat(z,label,base,flt):
 o={'label':label,'base':base,'filter':flt,'n':len(z),'dates':z.sig_date.nunique() if len(z) else 0,'stocks':z.code.nunique() if len(z) else 0}
 if len(z):
  for n in [5,10,20,30]: o[f'r{n}']=z[f'r{n}'].mean(); o[f'med{n}']=z[f'r{n}'].median(); o[f'win{n}']=(z[f'r{n}']>0).mean(); o[f'mae{n}']=z[f'mae{n}'].median()
  o['up10_first']=(z.ft==1).mean(); o['dn5_first']=(z.ft==-1).mean(); o['tratio_med']=z.tratio.median()
 return o
def main():
 px=load(); by={k:g.reset_index(drop=True) for k,g in px.groupby('code')}; bases={'PRE':'signals_PRE_POS15_K20_S5_REV.csv','P20':'signals_P20.csv'}
 filters={
  'T5POS':lambda r:r.t5>0,
  'T1_T5':lambda r:r.t1>0 and r.t5>0,
  'T3_T5':lambda r:r.t3>0 and r.t5>0,
  'T5_ACCEL':lambda r:r.t5>0 and r.t5>r.t5_prev,
  'T1_T3_T5':lambda r:r.t1>0 and r.t3>0 and r.t5>0,
  'TR001':lambda r:pd.notna(r.tratio) and r.tratio>=.001,
  'TR0025':lambda r:pd.notna(r.tratio) and r.tratio>=.0025,
  'TR005':lambda r:pd.notna(r.tratio) and r.tratio>=.005,
  'TR010':lambda r:pd.notna(r.tratio) and r.tratio>=.01,
  'TR0025_T1':lambda r:pd.notna(r.tratio) and r.tratio>=.0025 and r.t1>0,
  'TR005_T1':lambda r:pd.notna(r.tratio) and r.tratio>=.005 and r.t1>0,
  'T5_F5POS':lambda r:r.t5>0 and r.f5>0,
 }
 rows=[]
 for bn,fn in bases.items():
  sig=pd.read_csv(B/fn,dtype={'code':str}); sig.date=pd.to_datetime(sig.date); sig.code=sig.code.str.zfill(4)
  for flt,ff in filters.items():
   E=[]
   for s in sig.itertuples(index=False):
    g=by.get(s.code)
    if g is None: continue
    ii=g.index[g.date==s.date]
    if not len(ii): continue
    i=int(ii[0])
    for j in range(i+1,min(i+6,len(g)-1)):
     r=g.iloc[j]; reclaim=(pd.notna(r.ma5) and r.close>r.ma5 and r.ma5_up) or (pd.notna(r.ma10) and r.close>r.ma10 and r.bull)
     if bool(r.anom) or not reclaim or not ff(r): continue
     ei=j+1
     if ei>=len(g): break
     e=float(g.loc[ei,'open']); z={'sig_date':s.date,'code':s.code,'entry_date':g.loc[ei,'date'],'entry':e,'tratio':r.tratio}
     for n in [5,10,20,30]:
      w=g.iloc[ei:min(ei+n,len(g))]
      if len(w)<3 or w.anom.any(): z[f'r{n}']=np.nan; z[f'mae{n}']=np.nan
      else: z[f'r{n}']=float(w.iloc[-1].close/e-1); z[f'mae{n}']=float(w.low.min()/e-1)
     w=g.iloc[ei:min(ei+20,len(g))]; up=np.where(w.high.to_numpy()>=e*1.10)[0] if len(w) else []; dn=np.where(w.low.to_numpy()<=e*.95)[0] if len(w) else []; ui=up[0] if len(up) else 999; di=dn[0] if len(dn) else 999; z['ft']=1 if ui<di else (-1 if di<ui else 0); E.append(z); break
   z=pd.DataFrame(E)
   for period,yrs in [('TRAIN',[2021,2022,2023]),('VAL',[2024]),('HOLD',[2025])]:
    q=z[z.sig_date.dt.year.isin(yrs)] if len(z) else pd.DataFrame(); s=stat(q,f'{bn}_{flt}',bn,flt); s['period']=period; rows.append(s)
 S=pd.DataFrame(rows); S.to_csv(O/'periods.csv',index=False)
 tr=S[S.period=='TRAIN']; va=S[S.period=='VAL']; ho=S[S.period=='HOLD']; cand=tr[(tr.n>=20)&(tr.r20>0)&(tr.r30>0)&(tr.win20>=.5)].merge(va,on=['label','base','filter'],suffixes=('_tr','_va')); gate=cand[(cand.n_va>=5)&(cand.r20_va>0)&(cand.r30_va>0)&(cand.win20_va>=.5)].merge(ho,on=['label','base','filter'],suffixes=('','_ho')); gate.to_csv(O/'gated_holdout.csv',index=False)
 (O/'meta.json').write_text(json.dumps({'selection':'TRAIN 2021-23 then VAL 2024; HOLD 2025 reported only','note':'2025 already seen in earlier family work, so this is robustness not pristine discovery'},indent=2)); print('GATED'); print(gate.to_string(index=False) if len(gate) else 'NONE'); print('\nALL'); print(S.to_string(index=False))
if __name__=='__main__': main()
