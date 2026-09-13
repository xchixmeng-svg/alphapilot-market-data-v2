#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import research_fundamental_repricing_v9 as base

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'v14_prescreen_out'; OUT.mkdir(parents=True,exist_ok=True)
R10_CAGR_FLOOR=0.131103
HOLDS=(5,10,20,40,60)
SLOTS=(1,2,3,5)
INITIAL=1_300_000.0


def cagr(nav0, nav1, years=4.0):
    if nav0<=0 or nav1<=0: return -1.0
    return (nav1/nav0)**(1/years)-1


def families(d):
    # Deliberately broad and structurally different. No user-preference gates.
    return {
      'rs_acceleration':((d.r20_pr>=.70)&(d.r60_pr>=.55)&(d.r20_pr>d.r60_pr)&(d.aclose>d.ma20), .45*d.r20_pr+.25*d.r60_pr+.15*d.flow_accel_pr+.15*d.amount20_pr),
      'quiet_breakout':((d.r20_pr>=.75)&(d.vol20_pr<=.40)&(d.aclose>d.ma20)&(d.flow5>=0), .40*d.r20_pr+.25*(1-d.vol20_pr)+.20*d.flow5_pr+.15*d.amount20_pr),
      'sponsor_accumulation':((d.flow20_pr>=.75)&(d.flow5_pr>=.70)&(d.r20_pr<=.75), .35*d.flow20_pr+.30*d.flow5_pr+.20*d.flow_accel_pr+.15*d.amount20_pr),
      'sponsor_reversal':((d.r20_pr<=.35)&(d.flow5_pr>=.75)&(d.flow_accel_pr>=.70), .35*(1-d.r20_pr)+.30*d.flow5_pr+.25*d.flow_accel_pr+.10*d.amount20_pr),
      'leader_pullback':((d.r60_pr>=.75)&d.r20.between(-.08,.04)&(d.aclose>d.ma60), .40*d.r60_pr+.25*(1-d.r20_pr)+.20*d.flow5_pr+.15*(1-d.vol20_pr)),
      'leader_persistence':((d.r20_pr>=.80)&(d.r60_pr>=.80)&(d.vol20_pr<=.65), .40*d.r20_pr+.35*d.r60_pr+.15*(1-d.vol20_pr)+.10*d.amount20_pr),
      'lowvol_strength':((d.r60_pr>=.65)&(d.vol20_pr<=.30)&(d.aclose>d.ma60), .40*d.r60_pr+.30*(1-d.vol20_pr)+.15*d.flow5_pr+.15*d.amount20_pr),
      'flow_price_divergence':((d.r60_pr<=.55)&(d.flow20_pr>=.75)&(d.flow_accel_pr>=.65), .35*(1-d.r60_pr)+.30*d.flow20_pr+.25*d.flow_accel_pr+.10*d.amount20_pr),
      'liquid_mean_reversion':((d.r20_pr<=.20)&(d.amount20_pr>=.70)&(d.vol20_pr<=.75), .45*(1-d.r20_pr)+.25*d.amount20_pr+.15*(1-d.vol20_pr)+.15*d.flow_accel_pr),
      'deep_repair':((d.r20<=-.10)&(d.r60<=-.15)&(d.flow_accel>0)&(d.flow5>0), .35*(1-d.r20_pr)+.30*(1-d.r60_pr)+.25*d.flow_accel_pr+.10*d.flow5_pr),
      'trend_reclaim':((d.r60_pr>=.60)&(d.r20_pr>=.45)&(d.aclose>d.ma20)&(d.aclose>d.ma60)&(d.flow5_pr>=.50), .30*d.r60_pr+.25*d.r20_pr+.20*d.flow5_pr+.15*d.amount20_pr+.10*(1-d.vol20_pr)),
      'broad_leader':((d.r20_pr>=.75)&(d.r60_pr>=.70), .32*d.r20_pr+.28*d.r60_pr+.15*d.flow5_pr+.10*d.flow_accel_pr+.10*(1-d.vol20_pr)+.05*d.amount20_pr),
      'anti_crowded_momentum':((d.r20_pr>=.70)&(d.r60_pr>=.65)&(d.flow5_pr<=.65)&(d.vol20_pr<=.55), .40*d.r20_pr+.30*d.r60_pr+.20*(1-d.vol20_pr)+.10*(1-d.flow5_pr)),
      'early_flow_impulse':((d.flow_accel_pr>=.90)&(d.r20_pr.between(.35,.70))&(d.aclose>d.ma20), .45*d.flow_accel_pr+.20*d.flow5_pr+.20*d.r20_pr+.15*d.amount20_pr),
      'stable_compounder_proxy':((d.r60_pr>=.60)&(d.r20_pr>=.55)&(d.vol20_pr<=.35)&(d.amount20_pr>=.55), .35*d.r60_pr+.25*d.r20_pr+.25*(1-d.vol20_pr)+.15*d.amount20_pr),
      'high_liquidity_momentum':((d.amount20_pr>=.80)&(d.r20_pr>=.70)&(d.r60_pr>=.60), .35*d.r20_pr+.25*d.r60_pr+.25*d.amount20_pr+.15*d.flow5_pr),
      'breadth_independent_leader':((d.r20_pr>=.85)&(d.r60_pr>=.75), .45*d.r20_pr+.35*d.r60_pr+.10*d.flow5_pr+.10*(1-d.vol20_pr)),
      'balanced_flow_strength':(d.amount20.notna(), .23*d.r20_pr+.18*d.r60_pr+.20*d.flow5_pr+.17*d.flow_accel_pr+.12*(1-d.vol20_pr)+.10*d.amount20_pr),
    }


def simulate_optimistic(sig, hold, slots, px):
    # Optimistic by design: zero fees, zero tax, zero slippage, fractional shares, T+1 open.
    # It is a prescreen only. Failure here means the realistic version is not worth running.
    dates=sorted(px.date.unique())
    nextd={dates[i]:dates[i+1] for i in range(len(dates)-1)}
    byd={d:g.set_index('code') for d,g in px.groupby('date')}
    sigd={d:g.sort_values('score',ascending=False) for d,g in sig.groupby('signal_date')}
    cash=INITIAL; pos=[]; nav_rows=[]; trades=[]
    for d in [x for x in dates if 20210104<=int(x)<=20241231]:
        sub=byd.get(d)
        # exits at today's open after fixed trading-day hold count
        keep=[]
        for p in pos:
            p['age']+=1
            if p['age']>=hold and sub is not None and p['code'] in sub.index:
                r=sub.loc[p['code']]; xp=float(r['open'])
                if np.isfinite(xp) and xp>0:
                    proceeds=p['shares']*xp; cash+=proceeds
                    trades.append(proceeds/p['cost']-1.0)
                    continue
            keep.append(p)
        pos=keep
        # mark nav at close
        mv=0.0
        if sub is not None:
            for p in pos:
                if p['code'] in sub.index:
                    cp=float(sub.loc[p['code']]['close'])
                    if np.isfinite(cp) and cp>0: mv+=p['shares']*cp
        nav=cash+mv; nav_rows.append((d,nav))
        # today's signals schedule for T+1 open
        nd=nextd.get(d)
        if nd is None or nd not in byd: continue
        free=max(0,slots-len(pos))
        if free<=0 or d not in sigd: continue
        nsub=byd[nd]
        held={p['code'] for p in pos}
        picks=sigd[d]
        for r in picks.itertuples(index=False):
            if free<=0: break
            if r.code in held or r.code not in nsub.index: continue
            op=float(nsub.loc[r.code]['open'])
            if not np.isfinite(op) or op<=0: continue
            # equal allocation over total slot budget; fractional shares = optimistic upper version
            budget=nav/slots
            if budget<=0 or cash<=0: break
            budget=min(budget,cash)
            sh=budget/op
            if sh<=0: continue
            cash-=sh*op
            pos.append({'code':r.code,'shares':sh,'cost':sh*op,'age':0})
            held.add(r.code); free-=1
    if not nav_rows: return None
    nav=pd.Series([v for _,v in nav_rows],index=[d for d,_ in nav_rows],dtype=float)
    eq=nav/nav.iloc[0]; dd=eq/eq.cummax()-1
    yr={}
    for y in (2021,2022,2023,2024):
        z=nav[(nav.index>=y*10000+101)&(nav.index<=y*10000+1231)]
        yr[y]=float(z.iloc[-1]/z.iloc[0]-1) if len(z)>1 else np.nan
    return {
      'end_nav':float(nav.iloc[-1]),'total_return':float(nav.iloc[-1]/INITIAL-1),
      'cagr':float(cagr(INITIAL,nav.iloc[-1],4.0)),'max_dd':float(dd.min()),
      'trade_count':int(len(trades)),'win_rate':float((np.array(trades)>0).mean()) if trades else 0.0,
      'mean_trade':float(np.mean(trades)) if trades else 0.0,
      'positive_years':int(sum(1 for v in yr.values() if np.isfinite(v) and v>0)),
      'worst_year':float(min(v for v in yr.values() if np.isfinite(v))) if any(np.isfinite(v) for v in yr.values()) else -9.0,
      **{f'ret_{y}':yr[y] for y in yr}
    }


def main():
    px_all,d=base.build_daily()
    d=d[(d.date>=20210101)&(d.date<=20241231)].copy(); d['signal_date']=d.date.astype(int)
    rows=[]
    for fam,(mask,score) in families(d).items():
        raw=d[mask.fillna(False)].copy()
        raw['score']=score.loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
        for hold in HOLDS:
            for slots in SLOTS:
                cfg=f'{fam}__h{hold}__s{slots}'
                sig=raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(slots)
                z=simulate_optimistic(sig,hold,slots,px_all)
                if z is None: continue
                # Necessary-condition screen. The optimistic version must clear the real R10 CAGR floor
                # and show non-trivial deployment. Otherwise the realistic version is not worth full audit.
                pass_gate=bool(z['cagr']>R10_CAGR_FLOOR and z['trade_count']>=40 and z['mean_trade']>0 and z['positive_years']>=3)
                reason=[]
                if z['cagr']<=R10_CAGR_FLOOR: reason.append('optimistic_cagr_not_above_r10')
                if z['trade_count']<40: reason.append('too_few_trades')
                if z['mean_trade']<=0: reason.append('nonpositive_gross_edge')
                if z['positive_years']<3: reason.append('weak_multiyear_consistency')
                rows.append({'config':cfg,'family':fam,'hold_days':hold,'slots':slots,'pass_prescreen':pass_gate,'reject_reason':';'.join(reason),**z})
    out=pd.DataFrame(rows)
    out=out.sort_values(['pass_prescreen','cagr','max_dd'],ascending=[False,False,False])
    out.to_csv(OUT/'prescreen_all.csv',index=False)
    surv=out[out.pass_prescreen].copy()
    surv.to_csv(OUT/'survivors.csv',index=False)
    audit={'version':'v14-optimistic-prescreen','purpose':'reject historically dominated candidates before expensive common-cash audit','hard_reject_logic':'Candidate gets zero fees/tax/slippage and fractional shares. If even this optimistic version cannot clear R10 CAGR floor, it is not promoted.','r10_cagr_floor':R10_CAGR_FLOOR,'development_period':'2021-2024 only','2025_used':False,'family_count':len(families(d)),'config_count':int(len(out)),'survivor_count':int(len(surv)),'warning':'Prescreen is a necessary-condition filter, not proof of future performance. Survivors still require full common-cash T+1 realistic audit.'}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out.head(40).to_string(index=False))
    print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__': main()
