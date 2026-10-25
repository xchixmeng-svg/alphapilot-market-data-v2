import json
from pathlib import Path
import numpy as np, pandas as pd
R=Path(__file__).resolve().parents[1]; D=R/'data/history/2020-2025'; O=R/'research/results/pivot_rescue_20261025'; O.mkdir(parents=True,exist_ok=True)

def pdate(s):
    z=s.astype(str).str.replace(r'\.0$','',regex=True).str.strip(); return pd.to_datetime(z,format='%Y%m%d',errors='coerce') if z.str.fullmatch(r'\d{8}').mean()>.8 else pd.to_datetime(s,errors='coerce')

def load_px():
    a=[]
    for y in range(2020,2026):
        x=pd.read_parquet(D/f'ohlcv_{y}.parquet'); x['date']=pdate(x.date); x.code=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
        for c in ['open','high','low','close','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
        a.append(x[['date','code','open','high','low','close','volume']])
    return pd.concat(a).dropna().sort_values(['code','date'])

def load_inst():
    x=pd.read_parquet(D/'institutional_2020_2025.parquet'); x['date']=pdate(x.date); x.code=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
    for c in ['foreign_net','trust_net','dealer_net']: x[c]=pd.to_numeric(x[c],errors='coerce').fillna(0)
    x=x.sort_values(['code','date']); g=x.groupby('code',group_keys=False)
    x['f3']=g.foreign_net.transform(lambda s:s.rolling(3,min_periods=1).sum()); x['f5']=g.foreign_net.transform(lambda s:s.rolling(5,min_periods=1).sum()); x['t5']=g.trust_net.transform(lambda s:s.rolling(5,min_periods=1).sum()); x['d5']=g.dealer_net.transform(lambda s:s.rolling(5,min_periods=1).sum()); x['all5']=x.f5+x.t5+x.d5
    return x[['date','code','f3','f5','t5','d5','all5']]

def event_stats(events, px, label):
    rows=[]
    by={k:g.reset_index(drop=True) for k,g in px.groupby('code')}
    for r in events.itertuples(index=False):
        g=by.get(str(r.code));
        if g is None: continue
        ii=g.index[g.date==r.date]
        if len(ii)==0: continue
        i=int(ii[0]); entry_i=getattr(r,'entry_i',None)
        if entry_i is None or (isinstance(entry_i,float) and np.isnan(entry_i)): entry_i=i+1
        entry_i=int(entry_i)
        if entry_i>=len(g): continue
        e=float(g.loc[entry_i,'open']);
        if not np.isfinite(e) or e<=0: continue
        z={'date':r.date,'code':str(r.code),'entry_date':g.loc[entry_i,'date'],'entry':e}
        for n in [5,10,20,30]:
            j=min(entry_i+n-1,len(g)-1); w=g.iloc[entry_i:j+1]
            if len(w)==0: continue
            z[f'r{n}']=float(w.iloc[-1].close/e-1); z[f'mfe{n}']=float(w.high.max()/e-1); z[f'mae{n}']=float(w.low.min()/e-1)
        w=g.iloc[entry_i:min(entry_i+20,len(g))]; up=np.where(w.high.to_numpy()>=e*1.10)[0]; dn=np.where(w.low.to_numpy()<=e*.95)[0]; ui=up[0] if len(up) else 999; di=dn[0] if len(dn) else 999; z['ft10_5']=1 if ui<di else (-1 if di<ui else 0)
        rows.append(z)
    z=pd.DataFrame(rows)
    out={'label':label,'n':len(z),'dates':z.date.nunique() if len(z) else 0,'stocks':z.code.nunique() if len(z) else 0}
    for n in [5,10,20,30]:
        if len(z) and f'r{n}' in z: out|={f'r{n}_mean':z[f'r{n}'].mean(),f'r{n}_med':z[f'r{n}'].median(),f'r{n}_win':(z[f'r{n}']>0).mean(),f'mfe{n}_med':z[f'mfe{n}'].median(),f'mae{n}_med':z[f'mae{n}'].median()}
    if len(z): out|={'ft10_5_up':(z.ft10_5==1).mean(),'ft10_5_dn':(z.ft10_5==-1).mean()}
    return out,z

def main():
    px=load_px(); inst=load_inst(); sig=pd.read_csv(R/'research/results/pivot5y_20261025/signals_PRE_POS15_K20_S5_REV.csv',dtype={'code':str}); sig['date']=pd.to_datetime(sig.date); sig.code=sig.code.str.zfill(4); sig=sig.merge(inst,on=['date','code'],how='left').fillna({'f3':0,'f5':0,'t5':0,'d5':0,'all5':0})
    by={k:g.reset_index(drop=True) for k,g in px.groupby('code')}; ev=[]
    for r in sig.itertuples(index=False):
        g=by.get(r.code)
        if g is None: continue
        ii=g.index[g.date==r.date]
        if not len(ii): continue
        i=int(ii[0]); base=dict(r._asdict()); base['base_i']=i
        # immediate T+1
        z=base.copy(); z['entry_i']=i+1; z['mode']='IMM'; ev.append(z)
        # T+1 bullish confirmation, enter T+2
        if i+2<len(g) and g.loc[i+1,'close']>g.loc[i,'close'] and g.loc[i+1,'close']>g.loc[i+1,'open']:
            z=base.copy(); z['entry_i']=i+2; z['mode']='C1'; ev.append(z)
        # T+1 close above signal high, enter T+2
        if i+2<len(g) and g.loc[i+1,'close']>g.loc[i,'high']:
            z=base.copy(); z['entry_i']=i+2; z['mode']='HIGH1'; ev.append(z)
        # first close above signal high in next 3 sessions, enter next open
        for j in range(i+1,min(i+4,len(g)-1)):
            if g.loc[j,'close']>g.loc[i,'high']:
                z=base.copy(); z['entry_i']=j+1; z['mode']='HIGH3'; ev.append(z); break
        # two consecutive higher closes, enter following open
        if i+3<len(g) and g.loc[i+1,'close']>g.loc[i,'close'] and g.loc[i+2,'close']>g.loc[i+1,'close']:
            z=base.copy(); z['entry_i']=i+3; z['mode']='2UP'; ev.append(z)
    E=pd.DataFrame(ev); summaries=[]; yearly=[]
    filters={
      'ALL':lambda x:np.ones(len(x),dtype=bool),
      'F3_POS':lambda x:x.f3>0,
      'T5_POS':lambda x:x.t5>0,
      'ALL5_POS':lambda x:x.all5>0,
      'F3_T5':lambda x:(x.f3>0)&(x.t5>=0),
      'F3_OR_T5':lambda x:(x.f3>0)|(x.t5>0)
    }
    for mode in ['IMM','C1','HIGH1','HIGH3','2UP']:
      q=E[E['mode']==mode]
      for fn,ff in filters.items():
        s=q[ff(q)].copy(); stat,det=event_stats(s,px,f'{mode}_{fn}'); summaries.append(stat)
        for y in range(2021,2026): st,_=event_stats(s[s.date.dt.year==y],px,f'{mode}_{fn}'); st['year']=y; yearly.append(st)
    pd.DataFrame(summaries).to_csv(O/'summary.csv',index=False); pd.DataFrame(yearly).to_csv(O/'yearly.csv',index=False)
    # 2025 ranking
    Y=pd.DataFrame(yearly); y25=Y[Y.year==2025].sort_values(['r10_mean','r20_mean'],ascending=False); y25.to_csv(O/'holdout_2025_rank.csv',index=False)
    meta={'candidate_signals':len(sig),'institutional_coverage':float(sig.f3.notna().mean()),'modes':['IMM','C1','HIGH1','HIGH3','2UP'],'filters':list(filters),'note':'All institutional sums use data available through signal date T. Confirmation uses post-signal closes; entry occurs next open, preserving causality.'}; (O/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); print(pd.DataFrame(summaries).sort_values('r10_mean',ascending=False).head(20).to_string(index=False)); print('\n2025\n',y25.head(20).to_string(index=False))
if __name__=='__main__': main()
