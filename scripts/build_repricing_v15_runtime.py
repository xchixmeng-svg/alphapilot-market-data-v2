from pathlib import Path
import re

src = Path('scripts/research_12_stock_setups.py').read_text()
ctx = """# causal continuous context: current T-close values; normalization uses T-1 and earlier only
ctx_cols=['breadth20','breadth60','med_r20','med_r60','dispersion20','flow_breadth','flow_dispersion','liquidity_expansion']
daily=ctx[['date']+ctx_cols].drop_duplicates('date').sort_values('date').copy()
for c in ctx_cols:
    mu=daily[c].shift(1).rolling(120,min_periods=40).mean()
    sd=daily[c].shift(1).rolling(120,min_periods=40).std().replace(0,np.nan)
    daily[c+'_ctxz']=((daily[c]-mu)/sd).clip(-4,4)
    daily[c+'_d5']=daily[c]-daily[c].shift(5)
stocks=stocks.drop(columns=ctx_cols,errors='ignore').merge(daily,on='date',how='left')

def quality(d):"""
src,n=re.subn(r'def quality\(d\):',ctx,src,count=1)
assert n==1,n
setups="""SETUPS={
'fresh_sponsor_first_breakout':dict(hold=7, cond=lambda d:(d.inst_prev5<=0)&(d.inst5>0)&(d.flow_accel_value>0)&(d.aclose>d.prior_high20)&(d.amount_ratio>1.10)&(d.r20_pr<.88), score=lambda d:.35*d.flow_accel_value_pr+.25*d.amount_ratio_pr+.20*d.r20_pr+.20*d.r60_pr),
'fresh_sponsor_pullback_reclaim':dict(hold=8, cond=lambda d:(d.inst_prev5<=0)&(d.inst5>0)&(d.r60_pr>.55)&(d.draw20<-.03)&(d.draw20>-.13)&(d.prev_close<=d.prev_ma20)&(d.aclose>d.ma20)&(d.r3>0), score=lambda d:.40*d.flow_accel_value_pr+.20*d.r60_pr+.20*d.r3_pr+.20*(1-d.vol20_pr)),
'sponsor_compression_escape':dict(hold=9, cond=lambda d:(d.inst10>0)&(d.inst20>0)&(d.flow20_to_amount>0)&(d.vol_compress<.72)&(d.r20_pr>.35)&(d.r20_pr<.78)&(d.aclose>d.ma20)&(d.r3>0), score=lambda d:.30*d.flow20_to_amount_pr+.20*d.flow10_to_amount_pr+.20*(1-d.vol_compress_pr)+.15*d.r3_pr+.15*d.r60_pr),
'quiet_sponsor_liquidity_release':dict(hold=10, cond=lambda d:(d.flow20_to_amount>0)&(d.vol20_pr<.55)&(d.amount5_ratio<1.10)&(d.amount_ratio>1.15)&(d.r5>0)&(d.aclose>d.ma20), score=lambda d:.30*d.flow20_to_amount_pr+.25*(1-d.vol20_pr)+.20*d.amount_ratio_pr+.15*d.r5_pr+.10*d.r60_pr),
'leader_sponsored_reclaim':dict(hold=7, cond=lambda d:(d.r120_pr>.78)&(d.r60_pr>.65)&(d.draw20<-.03)&(d.draw20>-.11)&(d.flow10_to_amount>0)&(d.prev_close<=d.prev_ma20)&(d.aclose>d.ma20), score=lambda d:.30*d.r120_pr+.25*d.flow10_to_amount_pr+.20*d.r3_pr+.15*(1-d.vol20_pr)+.10*d.amount20_pr),
'second_line_sponsor_catchup':dict(hold=9, cond=lambda d:(d.r120_pr>.72)&(d.r60_pr>.58)&(d.r20_pr>.32)&(d.r20_pr<.64)&(d.flow5_to_amount>0)&(d.flow_accel_value>0)&(d.r5>0)&(d.aclose>d.ma20), score=lambda d:.25*d.r120_pr+.25*d.flow_accel_value_pr+.20*d.flow5_to_amount_pr+.15*d.r5_pr+.15*d.r20_pr),
'participation_sponsor_midrank':dict(hold=8, cond=lambda d:(d.breadth20_d5>0)&(d.flow_breadth_d5>0)&(d.r20_pr>.40)&(d.r20_pr<.72)&(d.r60_pr>.45)&(d.r60_pr<.82)&(d.flow10_to_amount>0)&(d.dist_ma20<.08), score=lambda d:.25*d.flow10_to_amount_pr+.20*d.flow_accel_value_pr+.20*d.r60_pr+.15*d.r20_pr+.10*((d.breadth20_d5.clip(-.25,.25)+.25)/.5)+.10*((d.flow_breadth_d5.clip(-.25,.25)+.25)/.5)),
'dispersion_sponsor_reclaim':dict(hold=8, cond=lambda d:(d.dispersion20_ctxz>.50)&(d.flow_dispersion_ctxz>.25)&(d.r120_pr>.60)&(d.draw20<-.04)&(d.draw20>-.14)&(d.flow10_to_amount>0)&(d.r3>0)&(d.aclose>d.ma20), score=lambda d:.25*d.flow10_to_amount_pr+.20*d.flow_accel_value_pr+.20*d.r120_pr+.15*d.r3_pr+.10*(d.dispersion20_ctxz.clip(0,3)/3)+.10*(d.flow_dispersion_ctxz.clip(0,3)/3)),
'liquidity_sponsor_rotation':dict(hold=9, cond=lambda d:(d.liquidity_expansion_d5>0)&(d.r20_pr>.42)&(d.r20_pr<.70)&(d.r60_pr>.50)&(d.r60_pr<.80)&(d.flow20_to_amount>0)&(d.flow5_to_amount>0)&(d.amount_ratio>1.0)&(d.amount_ratio<1.55), score=lambda d:.25*d.flow20_to_amount_pr+.20*d.flow5_to_amount_pr+.20*d.r60_pr+.15*d.amount_ratio_pr+.10*d.r5_pr+.10*((d.liquidity_expansion_d5.clip(-.25,.25)+.25)/.5)),
'multihorizon_flow_breakout':dict(hold=8, cond=lambda d:(d.aclose>d.prior_high20)&(d.flow5_to_amount>0)&(d.flow10_to_amount>0)&(d.flow20_to_amount>0)&(d.amount_ratio>1.05)&(d.r20_pr<.90), score=lambda d:.25*d.flow5_to_amount_pr+.20*d.flow10_to_amount_pr+.20*d.flow20_to_amount_pr+.20*d.amount_ratio_pr+.15*d.r60_pr),
'flow_price_response_gap':dict(hold=10, cond=lambda d:(d.flow_accel_value_pr>.72)&(d.flow10_to_amount>0)&(d.r20_pr>.30)&(d.r20_pr<.62)&(d.r3_pr<.70)&(d.aclose>d.ma20)&(d.dist_ma20<.07), score=lambda d:.35*d.flow_accel_value_pr+.25*d.flow10_to_amount_pr+.15*d.r60_pr+.15*(1-d.r3_pr)+.10*d.amount20_pr),
'sponsor_trend_continuous_control':dict(hold=9, cond=lambda d:(d.ma20>d.ma60)&(d.flow10_to_amount>0)&(d.r20>0)&(d.dist_ma20<.10), score=lambda d:.22*d.flow20_to_amount_pr+.20*d.flow10_to_amount_pr+.18*d.r60_pr+.15*d.r20_pr+.10*d.amount20_pr+.05*((d.breadth20_ctxz.clip(-3,3)+3)/6)+.05*((d.flow_breadth_ctxz.clip(-3,3)+3)/6)+.05*((d.liquidity_expansion_ctxz.clip(-3,3)+3)/6)),
}"""
src,n=re.subn(r"SETUPS=\{.*?\n\}\nif args\.hypothesis",setups+"\nif args.hypothesis",src,flags=re.S)
assert n==1,n
ca_pat=re.compile(r"(?m)^(\s*)if rr is not None and bool\(getattr\(rr,'is_official_event',False\)\):\n\1    sf=float\(getattr\(rr,'share_factor',1\.0\)\)\n\1    if np\.isfinite\(sf\) and sf>0 and abs\(sf-1\)>1e-12: p\['shares'\]=int\(round\(p\['shares'\]\*sf\)\)")
def repl(m):
    i=m.group(1)
    return '\n'.join([i+"if rr is not None and bool(getattr(rr,'is_official_event',False)):",i+"    old_shares=int(p['shares'])",i+"    factor_raw=getattr(rr,'share_factor',1.0)",i+"    factor=float(factor_raw) if np.isfinite(factor_raw) and float(factor_raw)>0 else 1.0",i+"    exact_new=old_shares*factor",i+"    new_shares=max(0,int(np.floor(exact_new+1e-10)))",i+"    cash_div_raw=getattr(rr,'cash_dividend_per_share',0.0)",i+"    cash_div_ps=float(cash_div_raw) if np.isfinite(cash_div_raw) else 0.0",i+"    ref_raw=getattr(rr,'reference_price',np.nan)",i+"    ref=float(ref_raw) if np.isfinite(ref_raw) and float(ref_raw)>0 else float(rr.close)",i+"    cash += old_shares*cash_div_ps + max(0.0,exact_new-new_shares)*ref",i+"    p['shares']=new_shares",i+"    if cash < -0.01: raise RuntimeError('negative cash after corporate action')"])
src,n=ca_pat.subn(repl,src,count=1)
assert n==1,n
compile(src,'research_12_stock_repricing_v15_runtime.py','exec')
Path('scripts/research_12_stock_repricing_v15_runtime.py').write_text(src)
