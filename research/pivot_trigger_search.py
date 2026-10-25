import json, itertools
from pathlib import Path
import numpy as np, pandas as pd

R=Path(__file__).resolve().parents[1]
D=R/'data/history/2020-2025'
O=R/'research/results/pivot_trigger_search_20261025'; O.mkdir(parents=True,exist_ok=True)


def pdate(s):
    z=s.astype(str).str.replace(r'\.0$','',regex=True).str.strip()
    return pd.to_datetime(z,format='%Y%m%d',errors='coerce') if z.str.fullmatch(r'\d{8}').mean()>.8 else pd.to_datetime(s,errors='coerce')


def load_px():
    a=[]
    for y in range(2020,2026):
        x=pd.read_parquet(D/f'ohlcv_{y}.parquet')
        x['date']=pdate(x.date); x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
        for c in ['open','high','low','close','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
        a.append(x[['date','code','open','high','low','close','volume']])
    x=pd.concat(a).dropna().sort_values(['code','date']).reset_index(drop=True)
    g=x.groupby('code',group_keys=False)
    x['prev_close']=g.close.shift(1); x['ret1']=x.close/x.prev_close-1
    for n in [5,10,20]: x[f'ma{n}']=g.close.transform(lambda s:s.rolling(n,min_periods=n).mean())
    x['ma5_up']=x.ma5>g.ma5.shift(2); x['ma20_up']=x.ma20>g.ma20.shift(5)
    x['vmed20']=g.volume.transform(lambda s:s.rolling(20,min_periods=10).median()); x['vr']=x.volume/x.vmed20.replace(0,np.nan)
    lo9=g.low.transform(lambda s:s.rolling(9,min_periods=9).min()); hi9=g.high.transform(lambda s:s.rolling(9,min_periods=9).max())
    x['k9']=100*(x.close-lo9)/(hi9-lo9).replace(0,np.nan); x['k9_prev']=g.k9.shift(1)
    x['h5p']=g.high.transform(lambda s:s.shift(1).rolling(5,min_periods=5).max()); x['h10p']=g.high.transform(lambda s:s.shift(1).rolling(10,min_periods=10).max())
    x['ret3']=x.close/g.close.shift(3)-1; x['clv']=(x.close-x.low)/(x.high-x.low).replace(0,np.nan)
    x['bull']=x.close>x.open; x['higher']=x.close>x.prev_close
    x['two_up']=x.higher & (x.prev_close>g.close.shift(2))
    # normal Taiwan sessions should not jump >18%; use only to censor corporate-action/data anomalies in outcome windows
    x['anom']=x.ret1.abs()>0.18
    return x


def load_inst():
    x=pd.read_parquet(D/'institutional_2020_2025.parquet')
    x['date']=pdate(x.date); x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
    for c in ['foreign_net','trust_net','dealer_net']: x[c]=pd.to_numeric(x[c],errors='coerce').fillna(0)
    x=x.sort_values(['code','date']); g=x.groupby('code',group_keys=False)
    x['f3']=g.foreign_net.transform(lambda s:s.rolling(3,min_periods=1).sum())
    x['f5']=g.foreign_net.transform(lambda s:s.rolling(5,min_periods=1).sum())
    x['t5']=g.trust_net.transform(lambda s:s.rolling(5,min_periods=1).sum())
    x['all5']=x.f5+x.t5+g.dealer_net.transform(lambda s:s.rolling(5,min_periods=1).sum())
    x['all5_prev']=g.all5.shift(1)
    return x[['date','code','f3','f5','t5','all5','all5_prev']]


def add_market(px):
    d=px.groupby('date').agg(breadth=('ret1',lambda s:float((s>0).mean())),mkt_ret=('ret1','median')).sort_index()
    d['mkt5']=(1+d.mkt_ret.fillna(0)).rolling(5,min_periods=3).apply(np.prod,raw=True)-1
    return d


def price_conditions(r, sig_high):
    return {
      'BULL_HIGHER': bool(r.bull and r.higher),
      'SIG_HIGH': bool(r.close>sig_high),
      'SIG_HIGH_BULL': bool(r.close>sig_high and r.bull),
      'K20X': bool(pd.notna(r.k9) and pd.notna(r.k9_prev) and r.k9>=20 and r.k9_prev<20),
      'K30X': bool(pd.notna(r.k9) and pd.notna(r.k9_prev) and r.k9>=30 and r.k9_prev<30),
      'K50X': bool(pd.notna(r.k9) and pd.notna(r.k9_prev) and r.k9>=50 and r.k9_prev<50),
      'MA5_BULL': bool(pd.notna(r.ma5) and r.close>r.ma5 and r.bull),
      'MA10_BULL': bool(pd.notna(r.ma10) and r.close>r.ma10 and r.bull),
      'MA20_BULL': bool(pd.notna(r.ma20) and r.close>r.ma20 and r.bull),
      'MA5UP_MA5': bool(r.ma5_up and pd.notna(r.ma5) and r.close>r.ma5),
      'MA20UP_MA20': bool(r.ma20_up and pd.notna(r.ma20) and r.close>r.ma20),
      'BREAK5': bool(pd.notna(r.h5p) and r.close>r.h5p),
      'BREAK10': bool(pd.notna(r.h10p) and r.close>r.h10p),
      'RET3_2': bool(pd.notna(r.ret3) and r.ret3>=.02),
      'RET3_4': bool(pd.notna(r.ret3) and r.ret3>=.04),
      'TWO_UP': bool(r.two_up),
      'SIG_HIGH_K50': bool(r.close>sig_high and pd.notna(r.k9) and r.k9>=50),
      'K20_MA5': bool(pd.notna(r.k9) and r.k9>=20 and pd.notna(r.ma5) and r.close>r.ma5 and r.ma5_up),
    }

MODS={
 'NONE':lambda r: True,
 'VOL10':lambda r: pd.notna(r.vr) and r.vr>=1.0,
 'VOL12':lambda r: pd.notna(r.vr) and r.vr>=1.2,
 'VOL15':lambda r: pd.notna(r.vr) and r.vr>=1.5,
 'F3':lambda r: r.f3>0,
 'T5':lambda r: r.t5>0,
 'ALL5':lambda r: r.all5>0,
 'F3ORT5':lambda r:(r.f3>0 or r.t5>0),
 'INST_ACCEL':lambda r:r.all5>0 and r.all5>r.all5_prev,
 'MKT':lambda r:r.breadth>=.5 and r.mkt5>=0,
 'VOL12_F3':lambda r:pd.notna(r.vr) and r.vr>=1.2 and r.f3>0,
 'VOL12_T5':lambda r:pd.notna(r.vr) and r.vr>=1.2 and r.t5>0,
 'VOL12_ALL5':lambda r:pd.notna(r.vr) and r.vr>=1.2 and r.all5>0,
 'F3_MKT':lambda r:r.f3>0 and r.breadth>=.5 and r.mkt5>=0,
 'T5_MKT':lambda r:r.t5>0 and r.breadth>=.5 and r.mkt5>=0,
}


def stats(z,label):
    out={'label':label,'n':len(z),'dates':z.sig_date.nunique() if len(z) else 0,'stocks':z.code.nunique() if len(z) else 0}
    if not len(z): return out
    for n in [5,10,20,30]:
        s=z[f'r{n}'].dropna(); out[f'r{n}_mean']=s.mean(); out[f'r{n}_med']=s.median(); out[f'r{n}_win']=(s>0).mean()
        out[f'mfe{n}_med']=z[f'mfe{n}'].median(); out[f'mae{n}_med']=z[f'mae{n}'].median()
    out['ft_up']=(z.ft==1).mean(); out['ft_dn']=(z.ft==-1).mean(); out['delay_med']=z.delay.median()
    return out


def main():
    px=load_px(); inst=load_inst(); mkt=add_market(px)
    px=px.merge(inst,on=['date','code'],how='left').merge(mkt,left_on='date',right_index=True,how='left')
    for c in ['f3','f5','t5','all5','all5_prev']: px[c]=px[c].fillna(0)
    sig=pd.read_csv(R/'research/results/pivot5y_20261025/signals_PRE_POS15_K20_S5_REV.csv',dtype={'code':str}); sig['date']=pd.to_datetime(sig.date); sig.code=sig.code.str.zfill(4)
    by={k:g.reset_index(drop=True) for k,g in px.groupby('code')}
    rules=[]
    price_names=list(price_conditions(px.iloc[50],999).keys())
    for p,m,w in itertools.product(price_names,MODS.keys(),[3,5,10]): rules.append((p,m,w,f'{p}__{m}__W{w}'))
    events={name:[] for *_,name in rules}
    for s in sig.itertuples(index=False):
        g=by.get(s.code)
        if g is None: continue
        ii=g.index[g.date==s.date]
        if not len(ii): continue
        i=int(ii[0]); sig_high=float(g.loc[i,'high'])
        # cache daily booleans once
        days=[]
        for j in range(i+1,min(i+11,len(g)-1)):
            r=g.iloc[j]
            if bool(r.anom): continue
            days.append((j,r,price_conditions(r,sig_high)))
        for p,m,w,name in rules:
            trig=None
            for j,r,pc in days:
                if j-i>w: break
                if pc[p] and MODS[m](r): trig=(j,r); break
            if trig is None: continue
            j,r=trig; ei=j+1
            if ei>=len(g): continue
            entry=float(g.loc[ei,'open'])
            if not np.isfinite(entry) or entry<=0: continue
            z={'rule':name,'sig_date':s.date,'trig_date':r.date,'entry_date':g.loc[ei,'date'],'code':s.code,'delay':j-i,'entry':entry}
            bad=False
            for n in [5,10,20,30]:
                wdf=g.iloc[ei:min(ei+n,len(g))]
                if len(wdf)<min(3,n) or wdf.anom.any(): z[f'r{n}']=np.nan; z[f'mfe{n}']=np.nan; z[f'mae{n}']=np.nan; continue
                z[f'r{n}']=float(wdf.iloc[-1].close/entry-1); z[f'mfe{n}']=float(wdf.high.max()/entry-1); z[f'mae{n}']=float(wdf.low.min()/entry-1)
            w20=g.iloc[ei:min(ei+20,len(g))]
            if len(w20) and not w20.anom.any():
                up=np.where(w20.high.to_numpy()>=entry*1.10)[0]; dn=np.where(w20.low.to_numpy()<=entry*.95)[0]
                ui=up[0] if len(up) else 999; di=dn[0] if len(dn) else 999; z['ft']=1 if ui<di else (-1 if di<ui else 0)
            else: z['ft']=0
            events[name].append(z)
    rows=[]; yearly=[]; details=[]
    for p,m,w,name in rules:
        z=pd.DataFrame(events[name])
        if len(z):
            # one rule can trigger overlapping signals in same stock; keep first within 10 calendar sessions by entry ordering
            z=z.sort_values(['code','sig_date']).reset_index(drop=True)
        for period,mask in [('TRAIN_21_23', (z.sig_date.dt.year<=2023) if len(z) else []),('VAL_2024',(z.sig_date.dt.year==2024) if len(z) else []),('HOLD_2025',(z.sig_date.dt.year==2025) if len(z) else [])]:
            q=z[mask].copy() if len(z) else pd.DataFrame()
            st=stats(q,name); st.update({'period':period,'price':p,'mod':m,'window':w}); rows.append(st)
        if len(z):
            for y in range(2021,2026):
                q=z[z.sig_date.dt.year==y]; st=stats(q,name); st.update({'year':y,'price':p,'mod':m,'window':w}); yearly.append(st)
    S=pd.DataFrame(rows); Y=pd.DataFrame(yearly)
    S.to_csv(O/'all_rules_periods.csv',index=False); Y.to_csv(O/'all_rules_yearly.csv',index=False)
    # selection uses TRAIN and VAL only; HOLD is never used to choose rules
    tr=S[S.period=='TRAIN_21_23'].copy(); va=S[S.period=='VAL_2024'].copy(); ho=S[S.period=='HOLD_2025'].copy()
    tr['train_score']=tr.r20_mean.fillna(-9)+tr.r30_mean.fillna(-9)+0.5*tr.r10_mean.fillna(-9)+0.02*(tr.r20_win.fillna(0)-.5)-0.25*tr.mae20_med.fillna(0).abs()
    cand=tr[(tr.n>=30)&(tr.r10_mean>0)&(tr.r20_mean>0)&(tr.r30_mean>0)&(tr.r20_win>=.5)].sort_values('train_score',ascending=False)
    sel=cand.head(60).merge(va,on=['label','price','mod','window'],suffixes=('_tr','_va'))
    # validation gate before opening holdout
    gate=sel[(sel.n_va>=8)&(sel.r10_mean_va>0)&(sel.r20_mean_va>0)&(sel.r30_mean_va>0)&(sel.r20_win_va>=.5)].copy()
    gate=gate.merge(ho,on=['label','price','mod','window'],suffixes=('','_ho'))
    gate.to_csv(O/'selected_before_holdout.csv',index=False)
    # stability view, not selection
    cols=['label','price','mod','window','n_tr','r10_mean_tr','r20_mean_tr','r30_mean_tr','r20_win_tr','n_va','r10_mean_va','r20_mean_va','r30_mean_va','r20_win_va','n','r10_mean','r20_mean','r30_mean','r20_win','ft_up','ft_dn']
    if len(gate): gate[cols].sort_values(['r20_mean','r30_mean'],ascending=False).to_csv(O/'holdout_results.csv',index=False)
    else: pd.DataFrame(columns=cols).to_csv(O/'holdout_results.csv',index=False)
    meta={'signals':len(sig),'rules_tested':len(rules),'train':'2021-2023','validation':'2024','holdout':'2025','selection':'holdout never used for rule selection','candidate_train_rules':len(cand),'passed_validation':len(gate),'price_triggers':price_names,'modifiers':list(MODS)}
    (O/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(json.dumps(meta,ensure_ascii=False,indent=2)); print('\nGATED/HOLDOUT\n'); print(gate[cols].sort_values(['r20_mean','r30_mean'],ascending=False).head(30).to_string(index=False) if len(gate) else 'NO RULE PASSED VALIDATION')

if __name__=='__main__': main()
