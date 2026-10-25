import json
from pathlib import Path
import pandas as pd, numpy as np
R=Path(__file__).resolve().parents[1]; D=R/'data/history/2020-2025'; O=R/'research/results/pivot_debug_20261025'; O.mkdir(parents=True,exist_ok=True)
a=[]
for y in range(2020,2026):
 x=pd.read_parquet(D/f'ohlcv_{y}.parquet'); x['date']=pd.to_datetime(x['date']); x['code']=x['code'].astype(str).str.extract(r'(\d{4})',expand=False)
 for c in ['volume','open','high','low','close']: x[c]=pd.to_numeric(x[c],errors='coerce')
 x=x.dropna(subset=['date','code','volume','open','high','low','close']); x=x[(x.close>0)&~x.code.str.startswith('00')]; a.append(x)
x=pd.concat(a).sort_values(['code','date']).reset_index(drop=True); g=x.groupby('code',group_keys=False)
x['vmed20']=g.volume.transform(lambda s:s.rolling(20,min_periods=10).median()); x['lo120']=g.low.transform(lambda s:s.rolling(120,min_periods=60).min()); x['hi120']=g.high.transform(lambda s:s.rolling(120,min_periods=60).max()); x['lo9']=g.low.transform(lambda s:s.rolling(9,min_periods=5).min()); x['hi9']=g.high.transform(lambda s:s.rolling(9,min_periods=5).max()); x['pos']=100*(x.close-x.lo120)/(x.hi120-x.lo120).replace(0,np.nan); x['k']=100*(x.close-x.lo9)/(x.hi9-x.lo9).replace(0,np.nan)
pc=g.close.shift(); tr=pd.concat([x.high-x.low,(x.high-pc).abs(),(x.low-pc).abs()],axis=1).max(axis=1); x['atrp']=tr.groupby(x.code).transform(lambda s:s.rolling(20,min_periods=10).mean())/x.close
m=(x.date.dt.year>=2021)&(x.date.dt.year<=2025); z=x[m]
q=lambda s:[float(v) for v in s.dropna().quantile([0,.1,.25,.5,.75,.9,.99,1]).values]
res={'rows':len(z),'stocks':z.code.nunique(),'volume_q':q(z.volume),'vmed20_q':q(z.vmed20),'close_q':q(z.close),'pos_nonnull':int(z.pos.notna().sum()),'k_nonnull':int(z.k.notna().sum()),'atr_nonnull':int(z.atrp.notna().sum()),'price5':int((z.close>=5).sum()),'vmed_ge_200k':int((z.vmed20>=200000).sum()),'vmed_ge_200':int((z.vmed20>=200).sum()),'atr_lt_20pct':int((z.atrp<.2).sum()),'all_200k':int(((z.close>=5)&(z.vmed20>=200000)&z.pos.notna()&z.k.notna()&(z.atrp<.2)).sum()),'all_200':int(((z.close>=5)&(z.vmed20>=200)&z.pos.notna()&z.k.notna()&(z.atrp<.2)).sum())}
(O/'debug.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
