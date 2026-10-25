import json, math
from pathlib import Path
import numpy as np, pandas as pd
R=Path(__file__).resolve().parents[1]; D=R/'data/history/2020-2025'; O=R/'research/results/pivot5y_20261025'; O.mkdir(parents=True,exist_ok=True)

def col(cs,*xs):
    m={str(c).lower():c for c in cs}
    for x in xs:
        if x.lower() in m:return m[x.lower()]
    for c in cs:
        if any(x.lower() in str(c).lower() for x in xs):return c

def norm(x):
    cs=x.columns; mp={'date':col(cs,'date','日期'),'code':col(cs,'code','stock_id','代號'),'name':col(cs,'name','名稱'),'open':col(cs,'open','開盤'),'high':col(cs,'high','最高','max'),'low':col(cs,'low','最低','min'),'close':col(cs,'close','收盤'),'volume':col(cs,'volume','成交量')}
    miss=[k for k,v in mp.items() if k!='name' and v is None]
    if miss: raise ValueError((miss,list(cs)))
    z=pd.DataFrame({k:(x[v] if v is not None else '') for k,v in mp.items()}); z.date=pd.to_datetime(z.date,errors='coerce'); z.code=z.code.astype(str).str.extract(r'(\d{4})',expand=False)
    for c in 'open high low close volume'.split(): z[c]=pd.to_numeric(z[c],errors='coerce')
    z=z.dropna(subset=['date','code','open','high','low','close','volume']); z=z[(z.open>0)&(z.high>0)&(z.low>0)&(z.close>0)&~z.code.str.startswith('00')]
    return z

def load():
    a=[]; schema={}
    for y in range(2020,2026):
        p=D/f'ohlcv_{y}.parquet'; q=pd.read_parquet(p); schema[str(y)]=list(map(str,q.columns)); q=norm(q); q['srcyear']=y; a.append(q)
    (O/'schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)); return pd.concat(a).sort_values(['code','date']).reset_index(drop=True)

def feat(g):
    g=g.sort_values('date').copy(); c,h,l,o,v=[g[x] for x in ['close','high','low','open','volume']]; pc=c.shift(); ph=h.shift(); g['seq']=np.arange(len(g))
    for w in [9,20,40,60,120]: g[f'lo{w}']=l.rolling(w,min_periods=max(5,w//2)).min(); g[f'hi{w}']=h.rolling(w,min_periods=max(5,w//2)).max(); g[f'ma{w}']=c.rolling(w,min_periods=max(5,w//2)).mean()
    g['vmed20']=v.rolling(20,min_periods=10).median(); g['vmean20']=v.rolling(20,min_periods=10).mean(); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); g['atrp']=tr.rolling(20,min_periods=10).mean()/c
    g['pos']=100*(c-g.lo120)/(g.hi120-g.lo120).replace(0,np.nan); g['k']=100*(c-g.lo9)/(g.hi9-g.lo9).replace(0,np.nan)
    plo20=l.shift().rolling(20,min_periods=10).min(); plo60=l.shift().rolling(60,min_periods=20).min(); phi20=h.shift().rolling(20,min_periods=10).max(); g['f1']=((c/plo20-1).between(-.01,.035)|(c/plo60-1).between(-.01,.035)|((pc<phi20.shift())&(c>=phi20.shift()))).fillna(False)
    step=pd.Series(np.where(c<50,1,np.where(c<100,5,np.where(c<500,10,50))),index=g.index); g['f2']=(((c/step).round()*step-c).abs()/c<=.012).fillna(False)
    g['f3']=((c/g.ma20-1).between(-.01,.025)|(c/g.ma60-1).between(-.01,.025)).fillna(False)
    lo40=l.shift().rolling(40,min_periods=20).min(); hi40=h.shift().rolling(40,min_periods=20).max(); lower=lo40+.18*(hi40-lo40); g['f4']=((c/lower-1).between(-.01,.04)).fillna(False)
    peak=h.shift().rolling(60,min_periods=20).max(); dd=1-c/peak; common=pd.concat([(dd-x).abs() for x in [.1,.2,.3,.4]],axis=1).min(axis=1)<=.025; a=l.shift().rolling(20,min_periods=10).min(); b=l.shift(21).rolling(20,min_periods=10).min(); g['f5']=(common|(((a/b-1).abs()<=.035)&(c>=a*.99))).fillna(False)
    lo=l.shift().rolling(60,min_periods=20).min(); hi=h.shift().rolling(60,min_periods=20).max(); rr=hi-lo; fib=[hi-x*rr for x in [.382,.5,.618]]; g['f6']=(pd.concat([((c-x).abs()/c) for x in fib],axis=1).min(axis=1)<=.025).fillna(False)
    gap=ph.where(l>ph*1.01).rolling(20,min_periods=1).max(); g['f7']=((c>=gap*.985)&(c<=gap*1.04)).fillna(False)
    tp=(h+l+c)/3; vw=(tp*v).rolling(20,min_periods=10).sum()/v.rolling(20,min_periods=10).sum(); g['f8']=((c/vw-1).between(-.015,.035)).fillna(False)
    sd=c.rolling(20,min_periods=10).std(ddof=0); bl=g.ma20-2*sd; g['f9']=((c/bl-1).between(-.02,.05)).fillna(False)
    g['sc']=g[[f'f{i}' for i in range(1,10)]].sum(axis=1); loc=(c-l)/(h-l).replace(0,np.nan); g['rev']=((c>o)&(c>pc)&(loc>=.6)).fillna(False)
    p20=h.shift().rolling(20,min_periods=10).max(); g['bo']=(c>p20).fillna(False); g['volx']=(v>g.vmean20*1.5).fillna(False); g['eligible']=(c>=5)&(g.vmed20>=200000)&g.pos.notna()&g.k.notna()&(g.atrp<.2)
    e=o.shift(-1); g['entry']=e
    for n in [5,10,20,30]:
        g[f'r{n}']=c.shift(-n)/e-1; hh=pd.concat([h.shift(-j) for j in range(1,n+1)],axis=1); ll=pd.concat([l.shift(-j) for j in range(1,n+1)],axis=1); g[f'mfe{n}']=hh.max(axis=1)/e-1; g[f'mae{n}']=ll.min(axis=1)/e-1
    H=np.column_stack([h.shift(-j) for j in range(1,21)]); L=np.column_stack([l.shift(-j) for j in range(1,21)])
    for up in [.1,.15]:
      for dn in [.05,.065,.08]:
        U=H>=e.to_numpy()[:,None]*(1+up); Q=L<=e.to_numpy()[:,None]*(1-dn); ui=np.where(U.any(1),U.argmax(1)+1,999); di=np.where(Q.any(1),Q.argmax(1)+1,999); g[f'ft{int(up*100)}_{int(dn*1000)}']=np.where(ui<di,1,np.where(di<ui,-1,0))
    return g

def cool(x,b=10):
    keep=[]
    for _,g in x.sort_values(['code','date']).groupby('code'):
      last=-9999
      for i,r in g.iterrows():
        if r.seq-last>=b: keep.append(i); last=r.seq
    return x.loc[keep].copy() if keep else x.iloc[:0].copy()

def sumr(x,label):
    z={'label':label,'n':len(x),'dates':x.date.nunique(),'stocks':x.code.nunique()}
    for n in [5,10,20,30]:
      s=x[f'r{n}'].dropna(); z|={f'r{n}_mean':s.mean(),f'r{n}_med':s.median(),f'r{n}_win':(s>0).mean(),f'mfe{n}_med':x[f'mfe{n}'].median(),f'mae{n}_med':x[f'mae{n}'].median(),f'a{n}_mean':x[f'a{n}'].mean(),f'a{n}_win':(x[f'a{n}']>0).mean()}
    for up in [10,15]:
      for dn in [50,65,80]:
        s=x[f'ft{up}_{dn}']; z[f'ft{up}_{dn}_up']=(s==1).mean(); z[f'ft{up}_{dn}_dn']=(s==-1).mean()
    return z

def main():
    x=load(); print('raw',len(x),x.code.nunique(),x.date.nunique()); x=pd.concat([feat(g) for _,g in x.groupby('code')],ignore_index=True).sort_values(['date','code'])
    x['pb']=(x.pos//10).clip(0,9); x['kb']=(x.k//10).clip(0,9)
    for n in [5,10,20,30]:
      v=x.eligible&x[f'r{n}'].notna(); b=x[v].groupby(['date','pb','kb'])[f'r{n}'].median().rename(f'b{n}'); x=x.join(b,on=['date','pb','kb']); x[f'a{n}']=x[f'r{n}']-x[f'b{n}']
    q=x[(x.date.dt.year>=2021)&(x.date.dt.year<=2025)&x.eligible].copy(); base=(q.pos<=15)&(q.k<=20)&(q.sc>=5)&q.rev
    rules={'PRE_POS15_K20_S5_REV':base,'P10':(q.pos<=10)&(q.k<=20)&(q.sc>=5)&q.rev,'P20':(q.pos<=20)&(q.k<=20)&(q.sc>=5)&q.rev,'K15':(q.pos<=15)&(q.k<=15)&(q.sc>=5)&q.rev,'K25':(q.pos<=15)&(q.k<=25)&(q.sc>=5)&q.rev,'S4':(q.pos<=15)&(q.k<=20)&(q.sc>=4)&q.rev,'S6':(q.pos<=15)&(q.k<=20)&(q.sc>=6)&q.rev,'NOREV':(q.pos<=15)&(q.k<=20)&(q.sc>=5),'BREAKOUT':q.bo&q.volx}
    S=[]; Y=[]
    for nm,m in rules.items():
      s=cool(q[m]); S.append(sumr(s,nm));
      for y in range(2021,2026): d=sumr(s[s.date.dt.year==y],nm); d['year']=y; Y.append(d)
      s[['date','code','name','entry','pos','k','sc','rev','r5','r10','r20','r30','a10','a20','mfe20','mae20']].to_csv(O/f'signals_{nm}.csv',index=False)
    pd.DataFrame(S).to_csv(O/'summary.csv',index=False); pd.DataFrame(Y).to_csv(O/'yearly.csv',index=False)
    A=[]; st=(q.pos<=15)&(q.k<=20)&q.rev
    for i in range(1,10): A.append(sumr(cool(q[st&q[f'f{i}']]),f'f{i}'))
    pd.DataFrame(A).to_csv(O/'ablation.csv',index=False)
    G=[]
    for p in [10,15,20]:
      for k in [15,20,25,30]:
       for sc in [4,5,6]:
        s=cool(q[(q.pos<=p)&(q.k<=k)&(q.sc>=sc)&q.rev]); d=sumr(s,f'p{p}_k{k}_s{sc}'); d|={'p':p,'kcut':k,'scut':sc}; G.append(d)
    pd.DataFrame(G).to_csv(O/'robust_grid.csv',index=False)
    b=cool(q[base]); hold=b[b.date.dt.year==2025]; rep={'rows':len(x),'stocks':x.code.nunique(),'dates':x.date.nunique(),'rule':'pos<=15,k<=20,support_count>=5,reversal,T+1 open,10-bar cooldown','all':sumr(b,'PRE'),'holdout_2025':sumr(hold,'HOLDOUT_2025'),'note':'2024Q3 influenced rule design; 2025 is cleanest holdout. alpha=return minus same-date same position/K decile median. 9 factors are independent proxies.'}; (O/'report.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2,default=lambda z:None if pd.isna(z) else float(z))); print(json.dumps(rep,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__': main()
