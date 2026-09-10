from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib, json, os, re
import numpy as np
import pandas as pd

MODEL='FutureAlpha-v0.1'
TZ=ZoneInfo('Asia/Taipei')
TODAY=datetime.now(TZ).date()
DATA=Path(os.getenv('FUTURE_ALPHA_DATA_ROOT','data'))
OUT=Path('future_alpha'); OUT.mkdir(exist_ok=True)
LEDGER=OUT/'predictions.csv'
STATUS=OUT/'status.json'


def date_dirs():
    out=[]
    if not DATA.exists(): return out
    for p in DATA.iterdir():
        if p.is_dir() and re.fullmatch(r'\d{4}-\d{2}-\d{2}',p.name):
            try: out.append((pd.Timestamp(p.name),p))
            except: pass
    return sorted(out)


def read_day(dt,p):
    frames=[]; insts=[]
    n=p/'normalized'
    for market in ('twse','tpex'):
        op=n/f'{market}_ohlcv.csv'; ip=n/f'{market}_institutional.csv'
        if op.exists():
            x=pd.read_csv(op,dtype={'stock_id':str}); x['market']=market.upper(); frames.append(x)
        if ip.exists():
            z=pd.read_csv(ip,dtype={'stock_id':str}); z['market']=market.upper(); insts.append(z)
    if not frames: return None
    o=pd.concat(frames,ignore_index=True)
    o['stock_id']=o.stock_id.astype(str).str.zfill(4); o['date']=dt
    for c in ['open','high','low','close','volume','trading_value']:
        if c in o: o[c]=pd.to_numeric(o[c],errors='coerce')
    if insts:
        z=pd.concat(insts,ignore_index=True); z['stock_id']=z.stock_id.astype(str).str.zfill(4)
        cols=['stock_id','foreign_net','trust_net','dealer_net']
        z=z[[c for c in cols if c in z.columns]].copy()
        for c in ['foreign_net','trust_net','dealer_net']:
            if c in z: z[c]=pd.to_numeric(z[c],errors='coerce').fillna(0)
        o=o.merge(z,on='stock_id',how='left')
    for c in ['foreign_net','trust_net','dealer_net']:
        if c not in o: o[c]=0.0
        o[c]=o[c].fillna(0)
    return o


dirs=date_dirs()
if not dirs:
    raise RuntimeError('no dated market snapshots')
rows=[]
for dt,p in dirs:
    x=read_day(dt,p)
    if x is not None: rows.append(x)
df=pd.concat(rows,ignore_index=True).sort_values(['stock_id','date'])
latest=pd.Timestamp(df.date.max()).date()

# No backdating: a prediction can only be created after the official snapshot for the current Taiwan date exists.
if latest < TODAY:
    status={'model':MODEL,'state':'WAIT_DATA','today':str(TODAY),'latest_official_snapshot':str(latest),'prediction_created':False,'reason':'current Taiwan EOD snapshot not yet available; refusing to backdate'}
    STATUS.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2)); raise SystemExit(0)

# Only normal 4-digit equities; remove obvious ETF family codes beginning with 0.
df=df[df.stock_id.str.fullmatch(r'\d{4}') & ~df.stock_id.str.startswith('0')].copy()
g=df.groupby('stock_id',group_keys=False)
df['r1']=g.close.pct_change(1); df['r5']=g.close.pct_change(5)
df['tv5']=g.trading_value.transform(lambda s:s.rolling(5,min_periods=3).mean())
df['tv_prev5']=g.trading_value.transform(lambda s:s.shift(5).rolling(5,min_periods=3).mean())
df['tv_accel']=df.tv5/df.tv_prev5.replace(0,np.nan)-1
# Flow intensity uses shares bought net of sold divided by recent volume; persistence measures repeated positive-flow days.
df['inst_net']=df.foreign_net+df.trust_net
df['flow5']=g.inst_net.transform(lambda s:s.rolling(5,min_periods=3).sum())
df['vol5']=g.volume.transform(lambda s:s.rolling(5,min_periods=3).sum())
df['flow_intensity']=df.flow5/df.vol5.replace(0,np.nan)
df['flow_pos_days']=g.inst_net.transform(lambda s:(s>0).rolling(5,min_periods=3).sum())
# Absorption: positive institutional flow with limited downside; this intentionally rewards capital absorption rather than raw chase.
df['downside5']=(-df.r5).clip(lower=0)
df['absorb_raw']=df.flow_intensity-(0.5*df.downside5)

cur=df[df.date==pd.Timestamp(latest)].copy()
cur=cur[(cur.close>0)&(cur.volume>0)&cur.trading_value.notna()]
# Liquidity gate is cross-sectional and scale invariant.
cur['liq_pct']=cur.trading_value.rank(pct=True)
cur=cur[cur.liq_pct>=0.35].copy()

def pct(s): return s.rank(pct=True).fillna(0.5)
cur['institution_score']=0.60*pct(cur.flow_intensity)+0.40*pct(cur.flow_pos_days)
cur['absorption_score']=pct(cur.absorb_raw)
cur['relative_strength_score']=pct(cur.r5)
cur['participation_score']=0.60*pct(cur.tv_accel)+0.40*cur.liq_pct
cur['score']=0.35*cur.institution_score+0.25*cur.absorption_score+0.20*cur.relative_strength_score+0.20*cur.participation_score
cur=cur.sort_values(['score','trading_value'],ascending=False).head(10).copy()

hist_days=int(df.date.nunique())
confidence='LOW_BOOTSTRAP' if hist_days<20 else ('MEDIUM' if hist_days<60 else 'FULL')
# Fingerprint the exact model + as-of input rows used for publication.
finger_cols=['date','stock_id','close','volume','trading_value','foreign_net','trust_net']
fingerprint=hashlib.sha256((MODEL+'\n'+cur[finger_cols].to_csv(index=False)).encode()).hexdigest()

pub=pd.DataFrame({
    'model_version':MODEL,'prediction_date':str(latest),'t_plus_1_required':True,
    'rank':range(1,len(cur)+1),'stock_id':cur.stock_id.values,'name':cur.get('name',pd.Series(['']*len(cur))).fillna('').values,
    'market':cur.market.values,'close':cur.close.values,
    'institution_score':cur.institution_score.values,'absorption_score':cur.absorption_score.values,
    'relative_strength_score':cur.relative_strength_score.values,'participation_score':cur.participation_score.values,
    'total_score':cur.score.values,'history_days':hist_days,'confidence':confidence,'input_fingerprint':fingerprint,
})

if LEDGER.exists():
    old=pd.read_csv(LEDGER,dtype={'stock_id':str})
    if ((old.model_version==MODEL)&(old.prediction_date.astype(str)==str(latest))).any():
        # Never overwrite/re-rank a published prediction date.
        status={'model':MODEL,'state':'ALREADY_PUBLISHED','prediction_date':str(latest),'prediction_created':False,'rows':int(len(old))}
        STATUS.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(status,ensure_ascii=False,indent=2)); raise SystemExit(0)
    pub=pd.concat([old,pub],ignore_index=True)

pub.to_csv(LEDGER,index=False)
latest_out=OUT/f'prediction_{latest}.csv'; cur_out=pub[pub.prediction_date.astype(str)==str(latest)]
cur_out.to_csv(latest_out,index=False)
status={'model':MODEL,'state':'PUBLISHED','prediction_date':str(latest),'prediction_created':True,'candidate_count':int(len(cur_out)),'history_days':hist_days,'confidence':confidence,'input_fingerprint':fingerprint,'formal_r10_modified':False,'historical_parameter_fit':False,'t_plus_1':True}
STATUS.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(status,ensure_ascii=False,indent=2)); print(cur_out.to_string(index=False))
