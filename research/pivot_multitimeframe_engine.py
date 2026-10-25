import json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]
DOLD=R/'data/history/2007-2019/raw'
DNEW=R/'data/history/2020-2025'
O=R/'research/results/pivot_multitimeframe_20261025'; O.mkdir(parents=True,exist_ok=True)

def pdate(s):
    z=s.astype(str).str.replace(r'\.0$','',regex=True).str.strip()
    if z.str.fullmatch(r'\d{8}').mean()>.8: return pd.to_datetime(z,format='%Y%m%d',errors='coerce')
    return pd.to_datetime(s,errors='coerce')

def load_px():
    a=[]
    for y in range(2007,2020):
        p=DOLD/f'yearly_{y}.zip'
        x=pd.read_csv(p,compression='zip',usecols=['date','code','open','high','low','close','volume'])
        x['date']=pdate(x.date); x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
        a.append(x)
    for y in range(2020,2026):
        x=pd.read_parquet(DNEW/f'ohlcv_{y}.parquet')[['date','code','open','high','low','close','volume']]
        x['date']=pdate(x.date); x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
        a.append(x)
    x=pd.concat(a,ignore_index=True)
    for c in ['open','high','low','close','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna().sort_values(['code','date']).drop_duplicates(['code','date'],keep='last').reset_index(drop=True)

def load_inst():
    x=pd.read_parquet(DNEW/'institutional_2020_2025.parquet')
    x['date']=pdate(x.date); x['code']=x.code.astype(str).str.extract(r'(\d{4})',expand=False)
    for c in ['foreign_net','trust_net','dealer_net']: x[c]=pd.to_numeric(x[c],errors='coerce').fillna(0)
    x=x.sort_values(['code','date']); x['all_net']=x.foreign_net+x.trust_net+x.dealer_net
    g=x.groupby('code',group_keys=False)
    for n in [5,20]:
        x[f'f{n}']=g.foreign_net.transform(lambda s,n=n:s.rolling(n,min_periods=n).sum())
        x[f't{n}']=g.trust_net.transform(lambda s,n=n:s.rolling(n,min_periods=n).sum())
        x[f'all{n}']=g.all_net.transform(lambda s,n=n:s.rolling(n,min_periods=n).sum())
    return x[['date','code','f5','f20','t5','t20','all5','all20']]

def kd_from_bars(b):
    lo=b.low.rolling(9,min_periods=9).min(); hi=b.high.rolling(9,min_periods=9).max()
    rsv=(b.close-lo)/(hi-lo).replace(0,np.nan)*100
    k=[]; d=[]; pk=pdv=50.0
    for v in rsv:
        if np.isfinite(v): pk=(2*pk+v)/3; pdv=(2*pdv+pk)/3
        k.append(pk if np.isfinite(v) else np.nan); d.append(pdv if np.isfinite(v) else np.nan)
    out=b.copy(); out['K']=k; out['D']=d; out['K_prev']=out.K.shift(1); out['D_prev']=out.D.shift(1)
    out['gold']=(out.K>out.D)&(out.K_prev<=out.D_prev)
    return out

def period_kd(g,freq):
    q=g.set_index('date').resample(freq).agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')).dropna()
    q=kd_from_bars(q)
    q[['K','D','gold']]=q[['K','D','gold']].shift(1)
    return q[['K','D','gold']]

def enrich(g):
    g=g.copy().reset_index(drop=True); c,o,h,l,v=g.close,g.open,g.high,g.low,g.volume
    prev=c.shift(1); g['ret1']=c/prev-1
    for n in [5,10,20,60,120,240,1200,2400]:
        mp=n if n<60 else max(20,int(n*.8))
        g[f'ma{n}']=c.rolling(n,min_periods=mp).mean()
    g['ma20_up']=g.ma20>g.ma20.shift(5); g['ma60_up']=g.ma60>g.ma60.shift(10); g['ma240_up']=g.ma240>g.ma240.shift(20)
    min5=c.rolling(1200,min_periods=900).min(); max5=c.rolling(1200,min_periods=900).max()
    min10=c.rolling(2400,min_periods=1800).min(); max10=c.rolling(2400,min_periods=1800).max()
    g['pos5']=(c-min5)/(max5-min5).replace(0,np.nan)*100
    g['pos10']=(c-min10)/(max10-min10).replace(0,np.nan)*100
    wk=period_kd(g,'W-FRI'); mo=period_kd(g,'ME')
    wk_map={p:(r.K,r.D,bool(r.gold) if pd.notna(r.gold) else False) for p,r in wk.iterrows()}
    mo_map={p:(r.K,r.D,bool(r.gold) if pd.notna(r.gold) else False) for p,r in mo.iterrows()}
    wvals=[]; mvals=[]
    for dte in g.date:
        wp=dte.to_period('W-FRI').end_time.normalize(); mp=dte.to_period('M').end_time.normalize()
        wvals.append(wk_map.get(wp,(np.nan,np.nan,False))); mvals.append(mo_map.get(mp,(np.nan,np.nan,False)))
    g[['wK','wD','wGold']]=pd.DataFrame(wvals,index=g.index); g[['mK','mD','mGold']]=pd.DataFrame(mvals,index=g.index)
    l1=l.shift(45).rolling(20,min_periods=15).min(); l2=l.shift(8).rolling(25,min_periods=18).min()
    midh=h.shift(25).rolling(20,min_periods=15).max()
    g['W_SETUP']=(np.abs(l1/l2-1)<=0.08) & (midh/np.minimum(l1,l2)-1>=0.07) & (c>g.ma20)
    g['W_BREAK']=g.W_SETUP & (c>midh)
    ls=l.shift(50).rolling(18,min_periods=12).min(); hd=l.shift(28).rolling(18,min_periods=12).min(); rs=l.shift(8).rolling(16,min_periods=10).min()
    shoulders=(np.abs(ls/rs-1)<=0.10); head_ok=(hd<np.minimum(ls,rs)*0.97)&(hd>np.minimum(ls,rs)*0.78)
    neck=h.shift(8).rolling(42,min_periods=25).max()
    g['IHS_SETUP']=shoulders&head_ok&(c>g.ma20)
    g['IHS_BREAK']=g.IHS_SETUP&(c>neck)
    g['ROUND_TURN']=(g.ma60.shift(20)<g.ma60.shift(40)) & (g.ma60>=g.ma60.shift(10)) & (c>g.ma20) & g.ma20_up
    spread=(pd.concat([g.ma5,g.ma10,g.ma20],axis=1).max(axis=1)-pd.concat([g.ma5,g.ma10,g.ma20],axis=1).min(axis=1))/c
    g['MA_SQUEEZE']=(spread.shift(1)<0.025)&(c>g.ma5)&(c>g.ma10)&(c>g.ma20)&(g.ma5>g.ma5.shift(1))
    g['LT5_MA']=c<=g.ma1200
    g['LT10_MA']=c<=g.ma2400
    g['LT5_POS30']=g.pos5<=30
    g['LT10_POS30']=g.pos10<=30
    g['WK_LOW']=g.wK<=30
    g['WK_VLOW']=g.wK<=20
    g['WK_GOLD_LOW']=g.wGold & (g.wK<=40)
    g['MO_LOW']=g.mK<=30
    g['MO_VLOW']=g.mK<=20
    g['MO_GOLD_LOW']=g.mGold & (g.mK<=40)
    g['KD_MULTI_LOW']=(g.wK<=30)&(g.mK<=35)
    g['KD_MLOW_WGOLD']=(g.mK<=35)&g.wGold&(g.wK<=45)
    g['MA_SUPPORT']=(c>=g.ma20)&(g.ma60>=g.ma60.shift(10)*0.99)
    g['MA_LONG_OK']=g.ma240_up | (c>=g.ma240)
    return g

def ev_for_rule(by,inst,stage,pat,chip,wait):
    rows=[]
    for code,g in by.items():
        z=g[(g.date.dt.year>=2021)&(g.date.dt.year<=2025)]
        baseidx=z.index[z[stage].fillna(False)]
        last_entry=-999
        for i in baseidx:
            if i-last_entry<15: continue
            end=min(i+wait,len(g)-2); jfound=None
            for j in range(i,end+1):
                if not bool(g.loc[j,pat]): continue
                q=inst.get((g.loc[j,'date'],code),(0,0,0,0,0,0)); f5,f20,t5,t20,a5,a20=q
                ok={'NONE':True,'T5':t5>0,'T5T20':(t5>0 and t20>0),'F20':f20>0,'ALL20':a20>0,'ALL5ALL20':(a5>0 and a20>0)}[chip]
                if ok: jfound=j; break
            if jfound is None: continue
            j=jfound; ei=j+1; e=float(g.loc[ei,'open'])
            if not np.isfinite(e) or e<=0: continue
            fw=g.iloc[ei:min(ei+30,len(g))]
            if (fw.ret1.abs()>0.15).any(): continue
            r={'code':code,'base_date':g.loc[i,'date'],'signal_date':g.loc[j,'date'],'entry_date':g.loc[ei,'date'],'entry':e,'stage':stage,'pattern':pat,'chip':chip,'wait':wait,'wK':g.loc[j,'wK'],'wD':g.loc[j,'wD'],'mK':g.loc[j,'mK'],'mD':g.loc[j,'mD'],'pos5':g.loc[j,'pos5'],'pos10':g.loc[j,'pos10']}
            for n in [5,10,20,30]:
                w=g.iloc[ei:min(ei+n,len(g))]
                if len(w): r[f'r{n}']=w.iloc[-1].close/e-1; r[f'mfe{n}']=w.high.max()/e-1; r[f'mae{n}']=w.low.min()/e-1
            w=g.iloc[ei:min(ei+20,len(g))]; up=np.where(w.high.to_numpy()>=e*1.10)[0]; dn=np.where(w.low.to_numpy()<=e*.95)[0]
            ui=up[0] if len(up) else 999; di=dn[0] if len(dn) else 999; r['ft']=1 if ui<di else (-1 if di<ui else 0)
            rows.append(r); last_entry=ei
    return pd.DataFrame(rows)

def stats(row,pref,z):
    row[f'n_{pref}']=len(z)
    for n in [5,10,20,30]:
        if len(z): row[f'r{n}_{pref}']=z[f'r{n}'].mean(); row[f'win{n}_{pref}']=(z[f'r{n}']>0).mean(); row[f'mae{n}_{pref}']=z[f'mae{n}'].median(); row[f'mfe{n}_{pref}']=z[f'mfe{n}'].median()
    if len(z): row[f'up10_{pref}']=(z.ft==1).mean(); row[f'dn5_{pref}']=(z.ft==-1).mean()

def main():
    px=load_px(); ins=load_inst(); by={code:enrich(g) for code,g in px.groupby('code',sort=False) if len(g)>=1000}
    im=ins.set_index(['date','code'])[['f5','f20','t5','t20','all5','all20']]; inst={k:tuple(v) for k,v in zip(im.index,im.to_numpy())}
    stages=['LT5_MA','LT10_MA','LT5_POS30','LT10_POS30','WK_LOW','MO_LOW','KD_MULTI_LOW','KD_MLOW_WGOLD']
    pats=['W_SETUP','W_BREAK','IHS_SETUP','IHS_BREAK','ROUND_TURN','MA_SQUEEZE','MA_SUPPORT']
    chips=['NONE','T5','T5T20','F20','ALL20','ALL5ALL20']; waits=[0,5,10]
    rows=[]; hold=[]
    for stage in stages:
      for pat in pats:
       for chip in chips:
        for wait in waits:
          ev=ev_for_rule(by,inst,stage,pat,chip,wait)
          tr=ev[(ev.signal_date.dt.year>=2021)&(ev.signal_date.dt.year<=2023)] if len(ev) else ev
          va=ev[ev.signal_date.dt.year==2024] if len(ev) else ev
          ho=ev[ev.signal_date.dt.year==2025] if len(ev) else ev
          row={'stage':stage,'pattern':pat,'chip':chip,'wait':wait}; stats(row,'tr',tr); stats(row,'va',va); stats(row,'ho',ho); rows.append(row)
          if len(tr)>=25 and row.get('r20_tr',-9)>0.005 and row.get('win20_tr',0)>=.53 and len(va)>=8 and row.get('r20_va',-9)>0.005 and row.get('win20_va',0)>=.52 and len(ho):
              hh=ho.copy(); hh['rule']=f'{stage}|{pat}|{chip}|W{wait}'; hold.append(hh)
    A=pd.DataFrame(rows); A.to_csv(O/'all_rules.csv',index=False)
    S=A[(A.n_tr>=25)&(A.r20_tr>.005)&(A.win20_tr>=.53)&(A.n_va>=8)&(A.r20_va>.005)&(A.win20_va>=.52)].copy().sort_values(['r20_ho','r20_va'],ascending=False)
    S.to_csv(O/'selected_holdout.csv',index=False)
    if hold: pd.concat(hold,ignore_index=True).to_csv(O/'holdout_events.csv',index=False)
    meta={'rows_total':len(px),'stocks_tested':len(by),'years':'2007-2025','rules_tested':len(A),'selected_pre_holdout':len(S),'split':{'train':'2021-2023','validation':'2024','holdout':'2025'},'kd':'9-period stochastic K/D, recursive 1/3 smoothing from 50; completed weekly/monthly bars shifted one full period before use (strict no-lookahead)','long_stage':'1200/2400 trading-day moving-average and range-position independent proxies for public 5y/10y low-stage descriptions','patterns':pats,'notes':'Clean-room independent proxies based only on public descriptions; institutional flows are proxies and are not claimed to equal proprietary main-force data.'}
    (O/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)); print(json.dumps(meta,ensure_ascii=False,indent=2)); print(S.head(30).to_string(index=False))
if __name__=='__main__': main()
