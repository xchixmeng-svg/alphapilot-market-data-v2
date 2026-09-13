#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import research_open_tournament_technical_v12 as v12
import research_v14_optimistic_prescreen as v14
import research_v16_friction_prescreen as v16
import validate_v12_champion_portfolio_v13 as v13

ROOT=Path(__file__).resolve().parent.parent
INITIAL=1_300_000.0
DEV_START=20230523
DEV_END=20241231
BLIND_START=20250101
BLIND_END=20251231

POLICIES={
 'adaptive_flow_vs_strength':('early_flow_impulse','rs_acceleration',20,3,'flow_accel_minus_dispersion'),
 'breakout_vs_pullback':('quiet_breakout','leader_pullback',40,2,'breadth_minus_dispersion'),
 'accumulation_vs_reversal':('sponsor_accumulation','sponsor_reversal',20,3,'trend_plus_flow'),
 'residual_momentum_vs_meanrev':('residual_momentum','liquid_mean_reversion',20,3,'dispersion_plus_breadth'),
 'leader_vs_anticrowded':('leader_persistence','anti_crowded_momentum',40,2,'flow_spread_plus_breadth'),
 'compression_vs_reclaim':('lowvol_strength','trend_reclaim',40,2,'calm_plus_breadth_change'),
 'divergence_vs_liquid_momentum':('flow_price_divergence','high_liquidity_momentum',20,3,'flow_minus_trend'),
 'early_flow_vs_pullback':('early_flow_impulse','leader_pullback',40,2,'flow_accel_plus_trend'),
 'lowvol_vs_highliq':('lowvol_strength','high_liquidity_momentum',60,2,'calm_context'),
 'repair_vs_trend':('deep_repair','trend_reclaim',40,2,'weak_trend_plus_flow_accel'),
 'base_spring_vs_persistence':('base_spring','leader_persistence',40,2,'low_dispersion'),
 'adaptive_multifactor_ensemble':('adaptive_ensemble','adaptive_ensemble',40,3,'ensemble'),
}


def sigmoid(x):
    x=np.clip(pd.Series(x,dtype=float),-6,6)
    return 1/(1+np.exp(-x))


def lagged_z(s: pd.Series, window=126, minp=40):
    mu=s.rolling(window,min_periods=minp).mean().shift(1)
    sd=s.rolling(window,min_periods=minp).std(ddof=0).shift(1).replace(0,np.nan)
    return ((s-mu)/sd).clip(-4,4).fillna(0.0)


def build_context(d: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for dt,g in d.groupby('date',sort=True):
        f5=g.flow5.replace([np.inf,-np.inf],np.nan).dropna()
        rows.append({
            'date':int(dt),
            'mkt_r20':float(g.r20.median()),
            'breadth60':float((g.aclose>g.ma60).mean()),
            'dispersion20':float(g.r20.std(ddof=0)),
            'flow_breadth':float((g.flow5>0).mean()),
            'flow_accel_breadth':float((g.flow_accel>0).mean()),
            'median_vol20':float(g.vol20.median()),
            'flow_spread':float(f5.quantile(.90)-f5.median()) if len(f5) else 0.0,
        })
    c=pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    for x in ['mkt_r20','breadth60','dispersion20','flow_breadth','flow_accel_breadth','median_vol20','flow_spread']:
        c[x+'_z']=lagged_z(c[x])
    c['breadth_change']=c.breadth60-c.breadth60.shift(5)
    c['breadth_change_z']=lagged_z(c.breadth_change.fillna(0))
    return c


def setup_map(d: pd.DataFrame):
    base=v14.families(d)
    out={k:(m,s) for k,(m,s) in base.items()}
    res20=d.r20-d.mkt_r20
    res60=d.r60-d.mkt_r60
    res20_pr=res20.groupby(d.date).rank(pct=True)
    res60_pr=res60.groupby(d.date).rank(pct=True)
    out['residual_momentum']=((res20_pr>=.70)&(res60_pr>=.60)&(d.aclose>d.ma20),
                              .45*res20_pr+.25*res60_pr+.15*d.flow5_pr+.15*d.amount20_pr)
    out['base_spring']=((d.vol20_pr<=.35)&(d.dist_ma20.abs()<=.06)&(d.flow20>0)&(d.flow_accel_pr>=.55),
                        .30*(1-d.vol20_pr)+.25*d.flow20_pr+.25*d.flow_accel_pr+.20*d.amount20_pr)
    return out


def context_weight(d: pd.DataFrame, key: str) -> pd.Series:
    if key=='flow_accel_minus_dispersion': x=d.flow_accel_breadth_z-d.dispersion20_z
    elif key=='breadth_minus_dispersion': x=d.breadth60_z-d.dispersion20_z
    elif key=='trend_plus_flow': x=d.mkt_r20_z+d.flow_breadth_z
    elif key=='dispersion_plus_breadth': x=d.dispersion20_z+d.breadth60_z
    elif key=='flow_spread_plus_breadth': x=d.flow_spread_z+d.breadth60_z
    elif key=='calm_plus_breadth_change': x=-d.median_vol20_z+d.breadth_change_z
    elif key=='flow_minus_trend': x=d.flow_breadth_z-d.mkt_r20_z
    elif key=='flow_accel_plus_trend': x=d.flow_accel_breadth_z+d.mkt_r20_z
    elif key=='calm_context': x=-d.median_vol20_z
    elif key=='weak_trend_plus_flow_accel': x=-d.mkt_r20_z+d.flow_accel_breadth_z
    elif key=='low_dispersion': x=-d.dispersion20_z
    else: x=.45*d.flow_breadth_z+.35*d.breadth60_z-.20*d.dispersion20_z
    return pd.Series(sigmoid(x).to_numpy(),index=d.index)


def make_signals(d: pd.DataFrame, policy: str):
    a,b,hold,slots,key=POLICIES[policy]
    sm=setup_map(d)
    w=context_weight(d,key)
    if policy=='adaptive_multifactor_ensemble':
        trend_w=.25+.35*w
        flow_w=.45-.20*w
        lowvol_w=.15+.10*(1-w)
        liq_w=1-trend_w-flow_w-lowvol_w
        score=(trend_w*(.55*d.r20_pr+.45*d.r60_pr)+flow_w*(.55*d.flow5_pr+.45*d.flow_accel_pr)
               +lowvol_w*(1-d.vol20_pr)+liq_w*d.amount20_pr)
        mask=((d.r20_pr>=.55)|(d.flow5_pr>=.65)|(d.flow_accel_pr>=.70))&(d.aclose>d.ma60*.92)
    else:
        ma,sa=sm[a]; mb,sb=sm[b]
        mask=ma.fillna(False)|mb.fillna(False)
        score=w*sa.fillna(0)+(1-w)*sb.fillna(0)+.12*w*ma.astype(float)+.12*(1-w)*mb.astype(float)
    raw=d[mask.fillna(False)].copy()
    if raw.empty:
        return pd.DataFrame(),hold,slots
    raw['score']=score.loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    q=(raw.amount20_pr>=.35)&(raw.vol20_pr<=.95)&(raw.close>=10)&raw.amount20.notna()
    raw=raw[q].copy()
    sig=(raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False])
            .groupby('signal_date',as_index=False).head(slots))
    return sig,hold,slots


def yearly(nav):
    out={}
    if nav.empty:return out
    n=nav.set_index('date').nav.astype(float)
    for y in (2023,2024,2025):
        lo=max(y*10000+101,DEV_START) if y==2023 else y*10000+101
        z=n[(n.index>=lo)&(n.index<=y*10000+1231)]
        out[str(y)]=float(z.iloc[-1]/z.iloc[0]-1) if len(z)>1 else None
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(POLICIES)); args=ap.parse_args()
    policy=args.hypothesis
    out=ROOT/'v18_context_adaptive_out'/policy; out.mkdir(parents=True,exist_ok=True)
    px_all,daily=v12.base.build_daily()
    d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index()
    c=build_context(d)
    d=d.merge(m,on='date',how='left').merge(c,on='date',how='left',suffixes=('','_ctx'))
    sig,hold,slots=make_signals(d,policy)
    if sig.empty: raise RuntimeError(f'no signals for {policy}')
    bycode=v16.build_price_index(px_all)
    schedule=v16.simulate_fast(policy,sig,hold,bycode)
    schedule.to_csv(out/'signal_schedule.csv',index=False)
    navdev,trdev,corpdev=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,DEV_END,INITIAL)
    nav25,tr25,corp25=v13.simulate_portfolio(px_all,schedule,slots,BLIND_START,BLIND_END,INITIAL)
    navfull,trfull,corpfull=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,BLIND_END,INITIAL)
    years_dev=(pd.Timestamp(str(DEV_END))-pd.Timestamp(str(DEV_START))).days/365.2425
    years_full=(pd.Timestamp(str(BLIND_END))-pd.Timestamp(str(DEV_START))).days/365.2425
    md=v13.metrics(navdev,trdev,years_dev); m25=v13.metrics(nav25,tr25,1.0); mf=v13.metrics(navfull,trfull,years_full)
    yr=yearly(navfull)
    navdev.to_csv(out/'dev_nav.csv',index=False); pd.DataFrame(trdev).to_csv(out/'dev_trades.csv',index=False)
    nav25.to_csv(out/'blind_2025_nav.csv',index=False); pd.DataFrame(tr25).to_csv(out/'blind_2025_trades.csv',index=False)
    navfull.to_csv(out/'full_nav.csv',index=False); pd.DataFrame(trfull).to_csv(out/'full_trades.csv',index=False)
    if not corpfull.empty: corpfull.to_csv(out/'corporate_actions.csv',index=False)
    audit={
      'version':'v18-context-adaptive-stock-setups','hypothesis':policy,'archetypes':POLICIES[policy][:2],
      'hold_days':hold,'slots':slots,'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],
      'selection_uses_2025':False,'context':'continuous T-close cross-sectional context standardized only against lagged trailing 126 trading days; no bull/bear hard gate',
      'quality_layer':'market/liquidity quality only; no claim of fundamental quality',
      'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick',
                   'integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,
                   'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
      'dev':md,'blind_2025':m25,'full':mf,'year_returns':yr,
      'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},
      'minute_gate_open':False,
    }
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__': main()
