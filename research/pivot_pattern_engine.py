import json
from pathlib import Path
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]
D=R/'data/history/2020-2025'
O=R/'research/results/pivot_pattern_engine_20261025'
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
    return x


def load_inst():
    x=pd.read_parquet(D/'institutional_2020_2025.parquet')
    x['date']=pdate(x.date)
    x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
    for c in ['foreign_net','trust_net','dealer_net']:
        x[c]=pd.to_numeric(x[c],errors='coerce').fillna(0)
    x=x.sort_values(['code','date'])
    g=x.groupby('code',group_keys=False)
    x['f5']=g.foreign_net.transform(lambda s:s.rolling(5,min_periods=1).sum())
    x['t5']=g.trust_net.transform(lambda s:s.rolling(5,min_periods=1).sum())
    x['all5']=g.apply(lambda q:(q.foreign_net+q.trust_net+q.dealer_net).rolling(5,min_periods=1).sum(),include_groups=False).reset_index(level=0,drop=True)
    return x[['date','code','f5','t5','all5']]


def enrich(g):
    g=g.copy().reset_index(drop=True)
    c,o,h,l,v=g.close,g.open,g.high,g.low,g.volume
    prev=c.shift(1)
    g['ret1']=c/prev-1
    for n in [5,10,20,60]:
        g[f'ma{n}']=c.rolling(n,min_periods=n).mean()
    g['ma5_up']=g.ma5>g.ma5.shift(1)
    g['ma10_up']=g.ma10>g.ma10.shift(1)
    g['ma20_up']=g.ma20>g.ma20.shift(1)
    g['v5']=v.rolling(5,min_periods=3).mean()
    g['v20']=v.rolling(20,min_periods=10).mean()
    g['h5p']=h.shift(1).rolling(5,min_periods=5).max()
    g['h10p']=h.shift(1).rolling(10,min_periods=10).max()
    g['h20p']=h.shift(1).rolling(20,min_periods=20).max()
    g['l20p']=l.shift(1).rolling(20,min_periods=20).min()
    g['range20']=(h.shift(1).rolling(20,min_periods=20).max()/l.shift(1).rolling(20,min_periods=20).min()-1)
    g['range60']=(h.shift(1).rolling(60,min_periods=40).max()/l.shift(1).rolling(60,min_periods=40).min()-1)
    tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    g['atr14']=tr.rolling(14,min_periods=10).mean()
    g['atrp']=g.atr14/c
    body=(c-o).abs(); rng=(h-l).replace(0,np.nan)
    upper=h-np.maximum(c,o); lower=np.minimum(c,o)-l
    g['body']=body; g['rng']=rng; g['upper']=upper; g['lower']=lower
    # objective candlestick patterns
    g['P_XIANREN_RAW']=(upper>=2*np.maximum(body,0.002*c)) & (upper/rng>=0.40) & (body/rng<=0.40)
    g['P_XIANREN_RISE']=g.P_XIANREN_RAW & (c>g.ma5) & g.ma5_up
    g['P_HAMMER']=(lower>=2*np.maximum(body,0.002*c)) & (lower/rng>=0.40) & (c>=l+0.65*rng)
    g['P_ENGULF']=(c>o) & (g.open.shift(1)>g.close.shift(1)) & (o<=g.close.shift(1)) & (c>=g.open.shift(1))
    g['P_FALSE_BREAK']=(l<g.l20p) & (c>g.l20p) & (c>o)
    g['P_INSIDE_BREAK']=(h.shift(1)<h.shift(2)) & (l.shift(1)>l.shift(2)) & (c>h.shift(2))
    # compression / base structures
    g['P_BOX_BREAK']=(g.range20<0.12) & (c>g.h10p)
    ma_spread=(pd.concat([g.ma5,g.ma10,g.ma20],axis=1).max(axis=1)-pd.concat([g.ma5,g.ma10,g.ma20],axis=1).min(axis=1))/c
    g['P_MA_SQUEEZE']=(ma_spread.shift(1)<0.025) & (c>g.ma5) & (c>g.ma10) & (c>g.ma20) & g.ma5_up
    g['P_VOL_DRY_REV']=(g.v5.shift(1)<0.70*g.v20.shift(1)) & (c>o) & (v>1.30*g.v5.shift(1))
    g['P_SQUEEZE_BREAK']=(g.atrp.shift(1)<g.atrp.shift(1).rolling(60,min_periods=30).quantile(.25)) & (c>g.h5p)
    # N-like continuation: recent impulse, shallow pullback, new short high
    imp=(c.shift(3)/c.shift(8)-1)>0.05
    hold=l.shift(1).rolling(3,min_periods=3).min()>c.shift(8)*0.98
    g['P_N_BREAK']=imp & hold & (c>g.h5p)
    # W proxy: two 3-bar local-low zones within 3% over previous 20 bars, neckline break today
    lowA=l.shift(3).rolling(5,min_periods=5).min()
    lowB=l.shift(8).rolling(8,min_periods=5).min()
    dbl=((lowA/lowB-1).abs()<=0.03)
    neckline=h.shift(1).rolling(10,min_periods=8).max()
    g['P_W_BREAK']=dbl & (c>neckline) & (neckline/np.minimum(lowA,lowB)-1>=0.04)
    # retest: a 20d breakout happened in prior 5 bars, current low tests breakout level and closes back above
    prior_level=g.h20p
    broke=(c>prior_level)
    recent_broke=broke.shift(1).rolling(5,min_periods=1).max().fillna(False).astype(bool)
    lvl=prior_level.shift(1).rolling(5,min_periods=1).max()
    g['P_RETEST_HOLD']=recent_broke & (l<=lvl*1.015) & (c>=lvl) & (c>o)
    # Xianren confirmation: a long-upper-wick pattern occurred 1-3 sessions ago and today closes above that pattern high
    xr_high=h.where(g.P_XIANREN_RAW)
    last_xr_high=xr_high.shift(1).rolling(3,min_periods=1).max()
    g['P_XIANREN_CONFIRM']=last_xr_high.notna() & (c>last_xr_high)
    # basic bullish reclaim (control, not a chart pattern)
    g['P_MA_RECLAIM']=(c>g.ma5) & g.ma5_up & (c.shift(1)<=g.ma5.shift(1))
    return g


def load_base(path,label):
    x=pd.read_csv(path,dtype={'code':str})
    x['date']=pd.to_datetime(x.date); x['code']=x.code.str.zfill(4); x['base']=label
    return x[['date','code','name','base']]


def stats(z,label):
    out={'label':label,'n':len(z),'dates':z.signal_date.nunique() if len(z) else 0,'stocks':z.code.nunique() if len(z) else 0}
    for n in [5,10,20,30]:
        if len(z):
            out[f'r{n}']=z[f'r{n}'].mean(); out[f'med{n}']=z[f'r{n}'].median(); out[f'win{n}']=(z[f'r{n}']>0).mean(); out[f'mae{n}']=z[f'mae{n}'].median(); out[f'mfe{n}']=z[f'mfe{n}'].median()
    if len(z):
        out['up10_first']=(z.ft==1).mean(); out['dn5_first']=(z.ft==-1).mean(); out['delay_med']=z.delay.median()
    return out


def build_events(base, by, inst, pattern, modifier, window):
    rows=[]
    for r in base.itertuples(index=False):
        g=by.get(r.code)
        if g is None: continue
        ix=g.index[g.date==r.date]
        if len(ix)==0: continue
        i=int(ix[0]); end=min(i+window,len(g)-2)
        found=None
        for j in range(i+1,end+1):
            if not bool(g.loc[j,pattern]): continue
            d=g.loc[j,'date']; key=(d,r.code)
            q=inst.get(key,(0.0,0.0,0.0))
            f5,t5,all5=q
            ok=(modifier=='NONE') or (modifier=='T5' and t5>0) or (modifier=='F5' and f5>0) or (modifier=='ALL5' and all5>0)
            if ok:
                found=j; break
        if found is None: continue
        j=found; ei=j+1
        e=float(g.loc[ei,'open'])
        if not np.isfinite(e) or e<=0: continue
        # reject corporate-action-like jumps in forward evaluation window
        fw=g.iloc[ei:min(ei+30,len(g))]
        if (fw.ret1.abs()>0.15).any(): continue
        z={'base_date':r.date,'signal_date':g.loc[j,'date'],'entry_date':g.loc[ei,'date'],'code':r.code,'name':r.name,'base':r.base,'pattern':pattern,'modifier':modifier,'window':window,'delay':j-i,'entry':e}
        for n in [5,10,20,30]:
            w=g.iloc[ei:min(ei+n,len(g))]
            if len(w)==0: continue
            z[f'r{n}']=float(w.iloc[-1].close/e-1); z[f'mfe{n}']=float(w.high.max()/e-1); z[f'mae{n}']=float(w.low.min()/e-1)
        w=g.iloc[ei:min(ei+20,len(g))]
        up=np.where(w.high.to_numpy()>=e*1.10)[0]; dn=np.where(w.low.to_numpy()<=e*.95)[0]
        ui=up[0] if len(up) else 999; di=dn[0] if len(dn) else 999
        z['ft']=1 if ui<di else (-1 if di<ui else 0)
        rows.append(z)
    return pd.DataFrame(rows)


def main():
    px=load_px(); ins=load_inst();
    by={}
    for code,g in px.groupby('code',sort=False): by[code]=enrich(g)
    im=ins.set_index(['date','code'])[['f5','t5','all5']]
    inst={k:tuple(v) for k,v in zip(im.index,im.to_numpy())}
    bases=pd.concat([
        load_base(R/'research/results/pivot5y_20261025/signals_PRE_POS15_K20_S5_REV.csv','PRE15'),
        load_base(R/'research/results/pivot5y_20261025/signals_P20.csv','P20')
    ],ignore_index=True).drop_duplicates(['base','date','code'])
    patterns=['P_XIANREN_RAW','P_XIANREN_RISE','P_XIANREN_CONFIRM','P_HAMMER','P_ENGULF','P_FALSE_BREAK','P_INSIDE_BREAK','P_BOX_BREAK','P_MA_SQUEEZE','P_VOL_DRY_REV','P_SQUEEZE_BREAK','P_N_BREAK','P_W_BREAK','P_RETEST_HOLD','P_MA_RECLAIM']
    modifiers=['NONE','T5','F5','ALL5']; windows=[3,5,10]
    allsum=[]; selected=[]; hold=[]
    for base_name in ['PRE15','P20']:
      b=bases[bases.base==base_name]
      for p in patterns:
       for m in modifiers:
        for w in windows:
            ev=build_events(b,by,inst,p,m,w)
            tr=ev[(ev.signal_date.dt.year>=2021)&(ev.signal_date.dt.year<=2023)] if len(ev) else ev
            va=ev[ev.signal_date.dt.year==2024] if len(ev) else ev
            ho=ev[ev.signal_date.dt.year==2025] if len(ev) else ev
            row={'base':base_name,'pattern':p,'modifier':m,'window':w}
            for pref,z in [('tr',tr),('va',va),('ho',ho)]:
                s=stats(z,'x');
                for k,v in s.items():
                    if k!='label': row[f'{k}_{pref}']=v
            allsum.append(row)
            # select without holdout: reasonable sample + positive train and 2024 validation
            if len(tr)>=15 and row.get('r20_tr',-9)>0.005 and row.get('win20_tr',0)>=0.52 and len(va)>=5 and row.get('r20_va',-9)>0.005 and row.get('win20_va',0)>=0.50:
                selected.append(row)
                if len(ho):
                    hh=ho.copy(); hh['rule']=f'{base_name}|{p}|{m}|W{w}'; hold.append(hh)
    A=pd.DataFrame(allsum); S=pd.DataFrame(selected)
    A.to_csv(O/'all_rules.csv',index=False)
    if len(S):
        S=S.sort_values(['r20_va','r20_tr'],ascending=False); S.to_csv(O/'selected_pre_holdout.csv',index=False)
        # now reveal holdout only for preselected rules
        cols=['base','pattern','modifier','window','n_tr','r20_tr','win20_tr','n_va','r20_va','win20_va','n_ho','r5_ho','r10_ho','r20_ho','r30_ho','win20_ho','up10_first_ho','dn5_first_ho']
        S[[c for c in cols if c in S.columns]].sort_values('r20_ho',ascending=False).to_csv(O/'holdout_results.csv',index=False)
    if hold:
        pd.concat(hold,ignore_index=True).to_csv(O/'holdout_events.csv',index=False)
    # pattern-only ablation over P20, W5, no modifier
    abl=A[(A.base=='P20')&(A.modifier=='NONE')&(A.window==5)].copy().sort_values('r20_va',ascending=False)
    abl.to_csv(O/'pattern_ablation.csv',index=False)
    meta={'base_signals':{k:int((bases.base==k).sum()) for k in bases.base.unique()},'patterns':patterns,'modifiers':modifiers,'windows':windows,'rules_tested':len(A),'selected_before_holdout':len(S),'split':{'train':'2021-2023','validation':'2024','holdout':'2025'},'selection_note':'2025 holdout is not used to select rules. Pattern at T is evaluated using data through T only; entry is T+1 open. Forward windows containing >15% one-day jumps are excluded as corporate-action-like anomalies.','xianren_note':'XIANREN variants are independent quantitative proxies for long-upper-wick / probe behavior; they are not claimed to reproduce any proprietary definition.'}
    (O/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(json.dumps(meta,ensure_ascii=False,indent=2))
    if len(S): print('\nSELECTED / HOLDOUT\n',S[[c for c in cols if c in S.columns]].sort_values('r20_ho',ascending=False).head(30).to_string(index=False))
    print('\nABLATION\n',abl[['pattern','n_tr','r20_tr','win20_tr','n_va','r20_va','win20_va','n_ho','r20_ho','win20_ho']].head(20).to_string(index=False))

if __name__=='__main__': main()
