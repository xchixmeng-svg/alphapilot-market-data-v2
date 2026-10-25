import json
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]; D=R/'data/history/2020-2025'; B=R/'research/results/pivot5y_20261025'; O=R/'research/results/pivot_family_robustness_20261025'; O.mkdir(parents=True,exist_ok=True)

def pdate(s):
 z=s.astype(str).str.replace(r'\.0$','',regex=True).str.strip(); return pd.to_datetime(z,format='%Y%m%d',errors='coerce') if z.str.fullmatch(r'\d{8}').mean()>.8 else pd.to_datetime(s,errors='coerce')

def load():
 a=[]
 for y in range(2020,2026):
  x=pd.read_parquet(D/f'ohlcv_{y}.parquet'); x.date=pdate(x.date); x.code=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
  for c in ['open','high','low','close','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
  a.append(x[['date','code','open','high','low','close','volume']])
 x=pd.concat(a).dropna().sort_values(['code','date']).reset_index(drop=True); g=x.groupby('code',group_keys=False)
 x['prev']=g.close.shift(1); x['ret1']=x.close/x.prev-1; x['bull']=x.close>x.open
 for n in [5,10]: x[f'ma{n}']=g.close.transform(lambda s:s.rolling(n,min_periods=n).mean())
 x['ma5_up']=x.ma5>g.ma5.shift(2)
 lo=g.low.transform(lambda s:s.rolling(9,min_periods=9).min()); hi=g.high.transform(lambda s:s.rolling(9,min_periods=9).max()); x['k9']=100*(x.close-lo)/(hi-lo).replace(0,np.nan)
 x['anom']=x.ret1.abs()>.18
 ins=pd.read_parquet(D/'institutional_2020_2025.parquet'); ins.date=pdate(ins.date); ins.code=ins.code.astype(str).str.extract(r'(\d{4})',expand=False); ins.trust_net=pd.to_numeric(ins.trust_net,errors='coerce').fillna(0); ig=ins.sort_values(['code','date']).groupby('code',group_keys=False); ins=ins.sort_values(['code','date']); ins['t5']=ig.trust_net.transform(lambda s:s.rolling(5,min_periods=1).sum()); ins['t3']=ig.trust_net.transform(lambda s:s.rolling(3,min_periods=1).sum()); ins['t1']=ins.trust_net
 x=x.merge(ins[['date','code','t1','t3','t5']],on=['date','code'],how='left').fillna({'t1':0,'t3':0,'t5':0}); return x

def st(z,label,base,trig):
 o={'label':label,'base':base,'trigger':trig,'n':len(z),'stocks':z.code.nunique() if len(z) else 0,'dates':z.sig_date.nunique() if len(z) else 0}
 if not len(z): return o
 for n in [5,10,20,30]:
  s=z[f'r{n}'].dropna(); o[f'r{n}']=s.mean(); o[f'med{n}']=s.median(); o[f'win{n}']=(s>0).mean(); o[f'mae{n}']=z[f'mae{n}'].median(); o[f'mfe{n}']=z[f'mfe{n}'].median()
 o['up10_first']=(z.ft==1).mean(); o['dn5_first']=(z.ft==-1).mean(); o['delay_med']=z.delay.median(); return o

def main():
 px=load(); by={k:g.reset_index(drop=True) for k,g in px.groupby('code')}
 bases=['PRE_POS15_K20_S5_REV','P10','P20','K15','K25','S4','S6','NOREV']
 files={'PRE_POS15_K20_S5_REV':'signals_PRE_POS15_K20_S5_REV.csv',**{x:f'signals_{x}.csv' for x in bases if x!='PRE_POS15_K20_S5_REV'}}
 triggers=['MA5_T5','MA10_T5','K20_MA5_T5','ANY_RECLAIM_T5']; allrows=[]; yearly=[]
 for base in bases:
  fp=B/files[base]
  if not fp.exists(): continue
  sig=pd.read_csv(fp,dtype={'code':str}); sig.date=pd.to_datetime(sig.date); sig.code=sig.code.str.zfill(4); E={t:[] for t in triggers}
  for s in sig.itertuples(index=False):
   g=by.get(s.code)
   if g is None: continue
   ii=g.index[g.date==s.date]
   if not len(ii): continue
   i=int(ii[0])
   hit={t:False for t in triggers}
   for j in range(i+1,min(i+6,len(g)-1)):
    r=g.iloc[j]
    if bool(r.anom) or r.t5<=0: continue
    cond={'MA5_T5': bool(pd.notna(r.ma5) and r.close>r.ma5 and r.ma5_up), 'MA10_T5': bool(pd.notna(r.ma10) and r.close>r.ma10 and r.bull), 'K20_MA5_T5': bool(pd.notna(r.k9) and r.k9>=20 and pd.notna(r.ma5) and r.close>r.ma5 and r.ma5_up)}
    cond['ANY_RECLAIM_T5']=cond['MA5_T5'] or cond['MA10_T5'] or cond['K20_MA5_T5']
    for t in triggers:
     if hit[t] or not cond[t]: continue
     ei=j+1
     if ei>=len(g): continue
     e=float(g.loc[ei,'open']); z={'sig_date':s.date,'code':s.code,'entry_date':g.loc[ei,'date'],'delay':j-i,'entry':e}
     for n in [5,10,20,30]:
      w=g.iloc[ei:min(ei+n,len(g))]
      if len(w)<3 or w.anom.any(): z[f'r{n}']=np.nan; z[f'mae{n}']=np.nan; z[f'mfe{n}']=np.nan
      else: z[f'r{n}']=float(w.iloc[-1].close/e-1); z[f'mae{n}']=float(w.low.min()/e-1); z[f'mfe{n}']=float(w.high.max()/e-1)
     w=g.iloc[ei:min(ei+20,len(g))]; up=np.where(w.high.to_numpy()>=e*1.10)[0] if len(w) else []; dn=np.where(w.low.to_numpy()<=e*.95)[0] if len(w) else []; ui=up[0] if len(up) else 999; di=dn[0] if len(dn) else 999; z['ft']=1 if ui<di else (-1 if di<ui else 0); E[t].append(z); hit[t]=True
  for t in triggers:
   z=pd.DataFrame(E[t]); allrows.append(st(z,f'{base}__{t}',base,t))
   if len(z):
    for y in range(2021,2026):
     q=z[z.sig_date.dt.year==y]; yy=st(q,f'{base}__{t}',base,t); yy['year']=y; yearly.append(yy)
 pd.DataFrame(allrows).to_csv(O/'summary.csv',index=False); Y=pd.DataFrame(yearly); Y.to_csv(O/'yearly.csv',index=False)
 # fixed family robustness: no re-selection by 2025; show all bases and years
 pivot=[]
 for (base,t),g in Y.groupby(['base','trigger']):
  d={'base':base,'trigger':t}
  for y in range(2021,2026):
   q=g[g.year==y]
   if len(q): d|={f'n{y}':int(q.iloc[0].n),f'r20_{y}':q.iloc[0].r20,f'win20_{y}':q.iloc[0].win20}
  pivot.append(d)
 pd.DataFrame(pivot).to_csv(O/'crossyear.csv',index=False)
 meta={'bases':bases,'fixed_trigger_family':'reclaim MA5/MA10 within 5 sessions AND trust 5-day net >0; enter next open','purpose':'post-hoc neighborhood robustness after strict-base family emerged; 2025 is no longer pristine for family discovery','rows':len(px)}; (O/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); print(pd.DataFrame(allrows).sort_values('r20',ascending=False).to_string(index=False)); print('\nCROSSYEAR\n',pd.DataFrame(pivot).to_string(index=False))
if __name__=='__main__': main()
