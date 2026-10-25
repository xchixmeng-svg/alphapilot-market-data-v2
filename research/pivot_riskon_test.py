import json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]; D=R/'data/history/2020-2025'; B=R/'research/results/pivot5y_20261025'; O=R/'research/results/pivot_riskon_20261025'; O.mkdir(parents=True,exist_ok=True)
def pdate(s):
 z=s.astype(str).str.replace(r'\.0$','',regex=True).str.strip(); return pd.to_datetime(z,format='%Y%m%d',errors='coerce') if z.str.fullmatch(r'\d{8}').mean()>.8 else pd.to_datetime(s,errors='coerce')
def load():
 a=[]
 for y in range(2020,2026):
  x=pd.read_parquet(D/f'ohlcv_{y}.parquet'); x.date=pdate(x.date); x.code=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
  for c in ['open','high','low','close','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
  a.append(x[['date','code','open','high','low','close','volume']])
 x=pd.concat(a).dropna().sort_values(['code','date']).reset_index(drop=True); g=x.groupby('code',group_keys=False); x['prev']=g.close.shift(1); x['ret1']=x.close/x.prev-1; x['bull']=x.close>x.open; x['ma5']=g.close.transform(lambda s:s.rolling(5,min_periods=5).mean()); x['ma10']=g.close.transform(lambda s:s.rolling(10,min_periods=10).mean()); x['ma20']=g.close.transform(lambda s:s.rolling(20,min_periods=20).mean()); x['ma5_up']=x.ma5>g.ma5.shift(2); x['ret20']=x.close/g.close.shift(20)-1; x['anom']=x.ret1.abs()>.18
 ins=pd.read_parquet(D/'institutional_2020_2025.parquet'); ins.date=pdate(ins.date); ins.code=ins.code.astype(str).str.extract(r'(\d{4})',expand=False); ins.trust_net=pd.to_numeric(ins.trust_net,errors='coerce').fillna(0); ins=ins.sort_values(['code','date']); ig=ins.groupby('code',group_keys=False); ins['t5']=ig.trust_net.transform(lambda s:s.rolling(5,min_periods=1).sum()); x=x.merge(ins[['date','code','t5']],on=['date','code'],how='left').fillna({'t5':0})
 # broad-market regime from equities only; no future data
 eq=x[~x.code.str.startswith('00')].copy(); md=eq.groupby('date').agg(breadth20=('ma20',lambda s:np.nan), med20=('ret20','median'))
 # compute fraction close>MA20 directly
 b=eq.assign(above20=eq.close>eq.ma20).groupby('date').above20.mean(); md['breadth20']=b; md['riskon']=(md.breadth20>=.50)&(md.med20>0)
 x=x.merge(md[['breadth20','med20','riskon']],left_on='date',right_index=True,how='left'); return x
def stat(z,label):
 o={'label':label,'n':len(z),'dates':z.sig_date.nunique() if len(z) else 0,'stocks':z.code.nunique() if len(z) else 0}
 if len(z):
  for n in [5,10,20,30]: o[f'r{n}']=z[f'r{n}'].mean(); o[f'win{n}']=(z[f'r{n}']>0).mean(); o[f'mae{n}']=z[f'mae{n}'].median()
  o['up10_first']=(z.ft==1).mean(); o['dn5_first']=(z.ft==-1).mean()
 return o
def main():
 px=load(); by={k:g.reset_index(drop=True) for k,g in px.groupby('code')}; bases={'PRE':'signals_PRE_POS15_K20_S5_REV.csv','P20':'signals_P20.csv'}; rows=[]; years=[]
 for bn,fn in bases.items():
  sig=pd.read_csv(B/fn,dtype={'code':str}); sig.date=pd.to_datetime(sig.date); sig.code=sig.code.str.zfill(4)
  for mode in ['ALL','RISKON']:
   E=[]
   for s in sig.itertuples(index=False):
    g=by.get(s.code); 
    if g is None: continue
    ii=g.index[g.date==s.date]
    if not len(ii): continue
    i=int(ii[0]); hit=False
    for j in range(i+1,min(i+6,len(g)-1)):
     r=g.iloc[j]; reclaim=(pd.notna(r.ma5) and r.close>r.ma5 and r.ma5_up) or (pd.notna(r.ma10) and r.close>r.ma10 and r.bull)
     if bool(r.anom) or r.t5<=0 or not reclaim or (mode=='RISKON' and not bool(r.riskon)): continue
     ei=j+1
     if ei>=len(g): break
     e=float(g.loc[ei,'open']); z={'sig_date':s.date,'code':s.code,'entry_date':g.loc[ei,'date'],'entry':e}
     for n in [5,10,20,30]:
      w=g.iloc[ei:min(ei+n,len(g))]
      if len(w)<3 or w.anom.any(): z[f'r{n}']=np.nan; z[f'mae{n}']=np.nan
      else: z[f'r{n}']=float(w.iloc[-1].close/e-1); z[f'mae{n}']=float(w.low.min()/e-1)
     w=g.iloc[ei:min(ei+20,len(g))]; up=np.where(w.high.to_numpy()>=e*1.10)[0] if len(w) else []; dn=np.where(w.low.to_numpy()<=e*.95)[0] if len(w) else []; ui=up[0] if len(up) else 999; di=dn[0] if len(dn) else 999; z['ft']=1 if ui<di else (-1 if di<ui else 0); E.append(z); break
   z=pd.DataFrame(E); rows.append(stat(z,f'{bn}_{mode}'))
   if len(z):
    for y in range(2021,2026): q=z[z.sig_date.dt.year==y]; yy=stat(q,f'{bn}_{mode}'); yy['year']=y; years.append(yy)
 pd.DataFrame(rows).to_csv(O/'summary.csv',index=False); pd.DataFrame(years).to_csv(O/'yearly.csv',index=False); (O/'meta.json').write_text(json.dumps({'riskon':'cross-sectional fraction close>MA20 >=50% AND median 20d stock return >0; evaluated on trigger close, enter next open','posthoc':True},indent=2)); print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(years).to_string(index=False))
if __name__=='__main__': main()
