#!/usr/bin/env python3
# exact replay trigger v1
from __future__ import annotations

import base64, gzip, io, json, math
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'.clean_cache'
REF=ROOT/'exact_reference'/'r10_exact_reference_bundle.b64.gzjson'
OUT=ROOT/'clean_results'
OUT.mkdir(exist_ok=True)
BUY_FEE=0.000855
SELL_FEE=0.000855
SELL_TAX=0.003
INITIAL=1_300_000.0


def tick(p: float)->float:
    if p<10:return .01
    if p<50:return .05
    if p<100:return .1
    if p<500:return .5
    if p<1000:return 1.0
    return 5.0

def ceil_tick(p: float)->float:
    t=tick(p); q=math.floor((p+1e-12)/t); b=q*t
    return round(b if abs(b-p)<1e-10 else (q+1)*t,8)

def load_bundle():
    enc=REF.read_text().strip()
    return json.loads(gzip.decompress(base64.b64decode(enc)).decode('utf-8'))

def csvdf(text:str)->pd.DataFrame:
    return pd.read_csv(io.StringIO(text),dtype={'code':str,'代碼':str})

def load_raw():
    parts=[]
    for y in range(2020,2026):
        p=CACHE/f'ohlcv_{y}.parquet'
        if not p.exists(): raise RuntimeError(f'missing {p}')
        parts.append(pd.read_parquet(p))
    q=pd.concat(parts,ignore_index=True)
    q['code']=q.code.astype(str).str.strip().str.zfill(4)
    q['date']=q.date.astype(int)
    return q.drop_duplicates(['date','code'],keep='last').sort_values(['date','code']).reset_index(drop=True)

def d8(s): return int(str(s).replace('-','')[:8])

def validate_orders(raw, orders):
    bars=raw.set_index(['date','code'])
    rows=[]
    for r in orders.itertuples(index=False):
        code=str(r.code).zfill(4); od=d8(r.order_date); limit=float(r.t1_limit); expected=str(r.status)
        key=(od,code)
        if key not in bars.index:
            rows.append({'strategy':getattr(r,'strategy',getattr(r,'version','')),'order_date':od,'code':code,'expected':expected,'actual':'NO_BAR','match':False,'reason':'NO_BAR'})
            continue
        b=bars.loc[key]
        if isinstance(b,pd.DataFrame): b=b.iloc[-1]
        op=float(b.open); lo=float(b.low)
        touched=(op<=limit+1e-9) or (lo<=limit+1e-9)
        actual='FILLED' if touched else 'MISSED'
        fill=None
        if touched:
            fill=min(op,limit) if op<=limit else limit
            fill=ceil_tick(fill); fill=min(fill,limit)
        exp_fill=getattr(r,'fill_price',float('nan'))
        fill_match=True
        if expected=='FILLED':
            fill_match=pd.notna(exp_fill) and abs(float(exp_fill)-float(fill))<1e-6
        status_match=(expected==actual)
        rows.append({'strategy':getattr(r,'strategy',getattr(r,'version','')),'order_date':od,'code':code,'limit':limit,'open':op,'low':lo,'expected':expected,'actual':actual,'expected_fill':None if pd.isna(exp_fill) else float(exp_fill),'actual_fill':fill,'status_match':status_match,'fill_match':fill_match,'match':status_match and fill_match})
    return pd.DataFrame(rows)

def validate_trades(raw,trades):
    bars=raw.set_index(['date','code'])
    rows=[]
    for r in trades.itertuples(index=False):
        code=str(getattr(r,'代碼')).zfill(4); bd=d8(getattr(r,'T_1買進日')); sd=d8(getattr(r,'賣出日'))
        bp=float(getattr(r,'實際買進價')); limit=float(getattr(r,'T_1預掛價')); sell_open=float(getattr(r,'T_1開盤價')); sp=float(getattr(r,'實際賣出價'))
        bk=(bd,code); sk=(sd,code)
        buy_bar_ok=bk in bars.index; sell_bar_ok=sk in bars.index
        if buy_bar_ok:
            bb=bars.loc[bk]; bb=bb.iloc[-1] if isinstance(bb,pd.DataFrame) else bb
            buy_touch=(float(bb.open)<=limit+1e-9) or (float(bb.low)<=limit+1e-9)
            buy_price_ok=bp<=limit+1e-9 and buy_touch
        else: buy_touch=False; buy_price_ok=False
        if sell_bar_ok:
            sb=bars.loc[sk]; sb=sb.iloc[-1] if isinstance(sb,pd.DataFrame) else sb
            open_ok=abs(float(sb.open)-sell_open)<1e-6
            slip_ok=sp<=sell_open+1e-9 and sp>=sell_open*0.99-0.11
        else: open_ok=False; slip_ok=False
        rows.append({'code':code,'buy_date':bd,'sell_date':sd,'buy_bar_ok':buy_bar_ok,'buy_touch':buy_touch,'buy_price_ok':buy_price_ok,'sell_bar_ok':sell_bar_ok,'sell_open_ok':open_ok,'sell_price_plausible':slip_ok,'match':buy_bar_ok and buy_price_ok and sell_bar_ok and open_ok and slip_ok})
    return pd.DataFrame(rows)

def replay_nav(raw,trades):
    t=trades.copy()
    for c in ['T+1買進日','賣出日']: t[c]=t[c].map(d8)
    t['代碼']=t['代碼'].astype(str).str.zfill(4)
    buys={}; sells={}
    for r in t.to_dict('records'):
        buys.setdefault(int(r['T+1買進日']),[]).append(r); sells.setdefault(int(r['賣出日']),[]).append(r)
    dates=sorted(int(x) for x in raw.loc[(raw.date>=20210104)&(raw.date<=20251231),'date'].unique())
    close_by_date={int(d):z.set_index('code')['close'].to_dict() for d,z in raw.groupby('date')}
    cash=INITIAL; pos={}; navrows=[]; peak=INITIAL; min_cash=INITIAL; max_pos=0
    for d in dates:
        for r in sells.get(d,[]):
            c=r['代碼']; proceeds=float(r['賣出淨收入'])
            if c not in pos: raise RuntimeError(f'sell without position {d} {c}')
            cash+=proceeds; del pos[c]
        for r in buys.get(d,[]):
            c=r['代碼']; sh=int(r['買進股數']); cost=float(r['實際投入總額'])
            if c in pos: raise RuntimeError(f'duplicate position {d} {c}')
            if cost>cash+1e-6: raise RuntimeError(f'cash overdraft {d} {c}: cost={cost} cash={cash}')
            cash-=cost; pos[c]=sh
        marks=close_by_date.get(d,{})
        mv=0.0; missing=[]
        for c,sh in pos.items():
            if c not in marks: missing.append(c)
            else: mv+=sh*float(marks[c])
        if missing: raise RuntimeError(f'missing close {d}: {missing[:5]}')
        nav=cash+mv; peak=max(peak,nav); dd=nav/peak-1
        min_cash=min(min_cash,cash); max_pos=max(max_pos,len(pos))
        navrows.append({'date':d,'nav':nav,'cash':cash,'market_value':mv,'positions':len(pos),'peak':peak,'drawdown':dd})
    return pd.DataFrame(navrows),min_cash,max_pos

def annual_returns(nav):
    z=nav.copy(); z['year']=z.date.astype(str).str[:4]
    return {str(y):float(g.iloc[-1].nav/g.iloc[0].nav-1) for y,g in z.groupby('year')}

def main():
    b=load_bundle(); raw=load_raw()
    r7=csvdf(b['r7_orders_csv']); r05=csvdf(b['r05_orders_csv']); trades=csvdf(b['trades_csv'])
    r7=r7.rename(columns={'version':'strategy'}); orders=pd.concat([r7,r05],ignore_index=True)
    oa=validate_orders(raw,orders); ta=validate_trades(raw,trades); nav,min_cash,max_pos=replay_nav(raw,trades)
    end=float(nav.iloc[-1].nav); maxdd=float(nav.drawdown.min()); yrs=annual_returns(nav); target=b['target']
    summary={
      'strategy':'AlphaPilot R10-MAX Exact Replay','reference_orders':int(len(orders)),'reference_trades':int(len(trades)),
      'order_exact_matches':int(oa['match'].sum()),'order_mismatches':int((~oa['match']).sum()),
      'trade_market_checks_passed':int(ta['match'].sum()),'trade_market_check_failures':int((~ta['match']).sum()),
      'initial_nav':INITIAL,'ending_nav_replayed':end,'max_drawdown_replayed':maxdd,'annual_returns_replayed':yrs,
      'min_cash_replayed':float(min_cash),'max_positions_replayed':int(max_pos),'target_ending_nav':float(target['ending_nav']),
      'target_max_drawdown':float(target['max_drawdown']),'ending_nav_diff':end-float(target['ending_nav']),
      'orders_410_gate':len(orders)==410 and int(oa['match'].sum())==410,'trades_241_gate':len(trades)==241 and int(ta['match'].sum())==241,
      'cash_nonnegative_gate':min_cash>=-1e-6,
    }
    summary['exact_replay_pass']=bool(summary['orders_410_gate'] and summary['trades_241_gate'] and summary['cash_nonnegative_gate'] and abs(summary['ending_nav_diff'])<1.0)
    oa.to_csv(OUT/'r10_exact_order_audit.csv',index=False); ta.to_csv(OUT/'r10_exact_trade_audit.csv',index=False); nav.to_csv(OUT/'r10_exact_nav.csv',index=False)
    (OUT/'r10_exact_replay_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not summary['exact_replay_pass']: raise SystemExit(2)

if __name__=='__main__': main()
