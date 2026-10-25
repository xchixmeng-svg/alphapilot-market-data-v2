import json
from pathlib import Path
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]
D=R/'data/history/2020-2025'
O=R/'research/results/pivot_structure_engine_20261025'
O.mkdir(parents=True,exist_ok=True)

def pdate(s):
    z=s.astype(str).str.replace(r'\.0$','',regex=True).str.strip()
    if z.str.fullmatch(r'\d{8}').mean()>.8:
        return pd.to_datetime(z,format='%Y%m%d',errors='coerce')
    return pd.to_datetime(s,errors='coerce')

def load_px():
    a=[]
    for y in range(2020,2026):
        x=pd.read_parquet(D/f'ohlcv_{y}.parquet')
        x['date']=pdate(x.date)
        x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
        for c in ['open','high','low','close','volume']:
            x[c]=pd.to_numeric(x[c],errors='coerce')
        a.append(x[['date','code','open','high','low','close','volume']])
    x=pd.concat(a,ignore_index=True).dropna().sort_values(['code','date']).reset_index(drop=True)
    x['ret1_raw']=x.groupby('code').close.pct_change()
    market=x.groupby('date').ret1_raw.median().rename('mkt_ret1')
    x=x.merge(market,on='date',how='left')
    return x

def load_inst():
    x=pd.read_parquet(D/'institutional_2020_2025.parquet')
    x['date']=pdate(x.date)
    x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
    for c in ['foreign_net','trust_net','dealer_net']:
        x[c]=pd.to_numeric(x[c],errors='coerce').fillna(0)
    x=x.sort_values(['code','date'])
    x['all_net']=x.foreign_net+x.trust_net+x.dealer_net
    g=x.groupby('code',group_keys=False)
    for n in [5,10,20]:
        x[f'f{n}']=g.foreign_net.transform(lambda s,n=n:s.rolling(n,min_periods=1).sum())
        x[f't{n}']=g.trust_net.transform(lambda s,n=n:s.rolling(n,min_periods=1).sum())
        x[f'all{n}']=g.all_net.transform(lambda s,n=n:s.rolling(n,min_periods=1).sum())
    return x[['date','code','f5','f20','t5','t20','all5','all20']]

def enrich(g):
    g=g.copy().reset_index(drop=True)
    c,o,h,l,v=g.close,g.open,g.high,g.low,g.volume
    prev=c.shift(1)
    g['ret1']=c/prev-1
    g['ret5']=c/c.shift(5)-1
    g['ret10']=c/c.shift(10)-1
    g['ret20']=c/c.shift(20)-1
    for n in [5,10,20,60]: g[f'ma{n}']=c.rolling(n,min_periods=n).mean()
    g['ma5_up']=g.ma5>g.ma5.shift(1)
    g['ma20_up']=g.ma20>g.ma20.shift(1)
    g['ma60_up']=g.ma60>g.ma60.shift(1)
    g['ma60_s5']=g.ma60/g.ma60.shift(5)-1
    g['v5']=v.rolling(5,min_periods=3).mean(); g['v20']=v.rolling(20,min_periods=10).mean()
    g['h5p']=h.shift(1).rolling(5,min_periods=5).max(); g['h10p']=h.shift(1).rolling(10,min_periods=10).max(); g['h20p']=h.shift(1).rolling(20,min_periods=20).max()
    g['l20p']=l.shift(1).rolling(20,min_periods=20).min()
    g['range40']=h.shift(1).rolling(40,min_periods=30).max()/l.shift(1).rolling(40,min_periods=30).min()-1
    g['range60']=h.shift(1).rolling(60,min_periods=45).max()/l.shift(1).rolling(60,min_periods=45).min()-1
    # public-clue structure proxies. These are independent research definitions.
    g['S_MA60_OK']=(g.ma60_s5>=0) & (c>=g.ma20) & (c>o)
    g['S_MA60_TURN']=g.ma60_up & (~g.ma60_up.shift(1).fillna(False)) & (c>g.ma20)
    # moving-average deduction proxy: current close overtakes the close rolling out of MA60
    g['S_DEDUCT_FLIP']=(c>c.shift(60)) & (c.shift(1)<=c.shift(61)) & (c>g.ma20)
    # pullback to rising season line and close back above it
    g['S_MA60_RETEST']=(g.ma60_s5>=0) & (l<=g.ma60*1.02) & (c>=g.ma60) & (c>o)
    # volume leads price: volume improves while price has not yet run, then price confirms
    quiet=(g.ret10>-0.08)&(g.ret10<0.05)
    g['S_VOL_LEAD']=quiet.shift(1).fillna(False) & (g.v5.shift(1)>1.25*g.v20.shift(1)) & (c>g.ma5) & g.ma5_up
    # accumulation proxy: high recent volume but muted 10d price movement, then bullish close
    g['S_ABSORB']=((g.v5/g.v20)>1.20) & (g.ret10.abs()<0.05) & (c>o) & (c>=g.ma5)
    # shakeout recovery: a 20d-low undercut occurred recently, but entry waits for MA10 recovery
    undercut=(l<g.l20p)
    had_undercut=undercut.shift(1).rolling(8,min_periods=1).max().fillna(False).astype(bool)
    g['S_SHAKE_RECOVER']=had_undercut & (c>g.ma10) & (c>o) & (c>prev)
    # relative strength / inverse expression on weak market days
    g['S_RS_DAY']=(g.mkt_ret1<=-0.01) & (g.ret1>=0.0) & (c>=g.ma5)
    # 5d relative strength uses rolling market median return compounded approximately by sum
    g['mkt5']=(1+g.mkt_ret1.fillna(0)).rolling(5,min_periods=5).apply(np.prod,raw=True)-1
    g['S_RS5']=(g.ret5-g.mkt5>=0.05) & (c>g.ma5) & g.ma5_up
    # long base then initial breakout, deliberately tighter than generic breakout
    g['S_BASE40_BREAK']=(g.range40<0.16) & (c>g.h10p) & (v<=2.0*g.v20)
    g['S_BASE60_BREAK']=(g.range60<0.22) & (c>g.h20p) & (v<=2.0*g.v20)
    # reference from prior best family: MA compression then expansion
    spread=(pd.concat([g.ma5,g.ma10,g.ma20],axis=1).max(axis=1)-pd.concat([g.ma5,g.ma10,g.ma20],axis=1).min(axis=1))/c
    g['REF_MA_SQUEEZE']=(spread.shift(1)<0.025) & (c>g.ma5)&(c>g.ma10)&(c>g.ma20)&g.ma5_up
    return g

def load_base(path,label):
    x=pd.read_csv(path,dtype={'code':str}); x['date']=pd.to_datetime(x.date); x['code']=x.code.str.zfill(4); x['base']=label
    return x[['date','code','name','base']]

def build_events(base,by,inst,pattern,modifier,window):
    rows=[]
    for r in base.itertuples(index=False):
        g=by.get(r.code)
        if g is None: continue
        ix=g.index[g.date==r.date]
        if len(ix)==0: continue
        i=int(ix[0]); end=min(i+window,len(g)-2); found=None
        for j in range(i+1,end+1):
            if not bool(g.loc[j,pattern]): continue
            q=inst.get((g.loc[j,'date'],r.code),(0,0,0,0,0,0))
            f5,f20,t5,t20,all5,all20=q
            ok={
              'NONE':True,'T5':t5>0,'T20':t20>0,'T5T20':(t5>0 and t20>0),'F20':f20>0,'ALL20':all20>0
            }[modifier]
            if ok: found=j; break
        if found is None: continue
        j=found; ei=j+1; e=float(g.loc[ei,'open'])
        if not np.isfinite(e) or e<=0: continue
        fw=g.iloc[ei:min(ei+30,len(g))]
        if (fw.ret1.abs()>0.15).any(): continue
        z={'base_date':r.date,'signal_date':g.loc[j,'date'],'entry_date':g.loc[ei,'date'],'code':r.code,'name':r.name,'base':r.base,'pattern':pattern,'modifier':modifier,'window':window,'delay':j-i,'entry':e}
        for n in [5,10,20,30]:
            w=g.iloc[ei:min(ei+n,len(g))]
            if len(w): z[f'r{n}']=float(w.iloc[-1].close/e-1); z[f'mfe{n}']=float(w.high.max()/e-1); z[f'mae{n}']=float(w.low.min()/e-1)
        w=g.iloc[ei:min(ei+20,len(g))]
        up=np.where(w.high.to_numpy()>=e*1.10)[0]; dn=np.where(w.low.to_numpy()<=e*.95)[0]
        ui=up[0] if len(up) else 999; di=dn[0] if len(dn) else 999; z['ft']=1 if ui<di else (-1 if di<ui else 0)
        rows.append(z)
    return pd.DataFrame(rows)

def add_stats(row,pref,z):
    row[f'n_{pref}']=len(z)
    for n in [5,10,20,30]:
        if len(z): row[f'r{n}_{pref}']=z[f'r{n}'].mean(); row[f'win{n}_{pref}']=(z[f'r{n}']>0).mean(); row[f'mae{n}_{pref}']=z[f'mae{n}'].median(); row[f'mfe{n}_{pref}']=z[f'mfe{n}'].median()
    if len(z): row[f'up10_first_{pref}']=(z.ft==1).mean(); row[f'dn5_first_{pref}']=(z.ft==-1).mean()

def main():
    px=load_px(); ins=load_inst(); by={code:enrich(g) for code,g in px.groupby('code',sort=False)}
    im=ins.set_index(['date','code'])[['f5','f20','t5','t20','all5','all20']]
    inst={k:tuple(v) for k,v in zip(im.index,im.to_numpy())}
    bases=pd.concat([
      load_base(R/'research/results/pivot5y_20261025/signals_PRE_POS15_K20_S5_REV.csv','PRE15'),
      load_base(R/'research/results/pivot5y_20261025/signals_P20.csv','P20')
    ],ignore_index=True).drop_duplicates(['base','date','code'])
    patterns=['S_MA60_OK','S_MA60_TURN','S_DEDUCT_FLIP','S_MA60_RETEST','S_VOL_LEAD','S_ABSORB','S_SHAKE_RECOVER','S_RS_DAY','S_RS5','S_BASE40_BREAK','S_BASE60_BREAK','REF_MA_SQUEEZE']
    mods=['NONE','T5','T20','T5T20','F20','ALL20']; windows=[3,5,10]
    rows=[]; hold_events=[]
    for base_name in ['PRE15','P20']:
      b=bases[bases.base==base_name]
      for p in patterns:
       for m in mods:
        for w in windows:
          ev=build_events(b,by,inst,p,m,w)
          tr=ev[(ev.signal_date.dt.year>=2021)&(ev.signal_date.dt.year<=2023)] if len(ev) else ev
          va=ev[ev.signal_date.dt.year==2024] if len(ev) else ev
          ho=ev[ev.signal_date.dt.year==2025] if len(ev) else ev
          row={'base':base_name,'pattern':p,'modifier':m,'window':w}; add_stats(row,'tr',tr); add_stats(row,'va',va); add_stats(row,'ho',ho); rows.append(row)
          if len(tr)>=15 and row.get('r20_tr',-9)>0.005 and row.get('win20_tr',0)>=0.52 and len(va)>=5 and row.get('r20_va',-9)>0.005 and row.get('win20_va',0)>=0.50 and len(ho):
              hh=ho.copy(); hh['rule']=f'{base_name}|{p}|{m}|W{w}'; hold_events.append(hh)
    A=pd.DataFrame(rows); A.to_csv(O/'all_rules.csv',index=False)
    S=A[(A.n_tr>=15)&(A.r20_tr>0.005)&(A.win20_tr>=0.52)&(A.n_va>=5)&(A.r20_va>0.005)&(A.win20_va>=0.50)].copy()
    S.sort_values(['r20_ho','r20_va'],ascending=False).to_csv(O/'selected_holdout.csv',index=False)
    if hold_events: pd.concat(hold_events,ignore_index=True).to_csv(O/'holdout_events.csv',index=False)
    # concise family summary on P20 W5 with no modifier and T5/T20 for diagnostics
    A[(A.base=='P20')&(A.window==5)&(A.modifier.isin(['NONE','T5','T20','T5T20']))].sort_values(['pattern','modifier']).to_csv(O/'family_diagnostics.csv',index=False)
    meta={'base_signals':{k:int((bases.base==k).sum()) for k in bases.base.unique()},'patterns':patterns,'modifiers':mods,'windows':windows,'rules_tested':len(A),'selected_pre_holdout':len(S),'split':{'train':'2021-2023','validation':'2024','holdout':'2025'},'notes':'Independent quantitative proxies inspired only by public descriptions: volume-price, accumulation/shakeout, MA60 direction/deduction, low-stage rotation/relative-strength. Not a proprietary-formula reconstruction.'}
    (O/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); print(json.dumps(meta,ensure_ascii=False,indent=2)); print(S.sort_values('r20_ho',ascending=False).head(20).to_string(index=False))
if __name__=='__main__': main()
