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
'breadth_flow_joint_recovery':dict(hold=9, cond=lambda d:(d.breadth20_d5>0)&(d.flow_breadth_d5>0)&(d.r20_pr>.40)&(d.r20_pr<.76)&(d.r60_pr>.45)&(d.flow5_to_amount>0)&(d.aclose>d.ma20)&(d.dist_ma20<.10), score=lambda d:.25*d.flow5_to_amount_pr+.20*d.r20_pr+.20*d.r60_pr+.15*d.flow_accel_value_pr+.10*((d.breadth20_d5.clip(-.25,.25)+.25)/.5)+.10*((d.flow_breadth_d5.clip(-.25,.25)+.25)/.5)),
'breadth_recovery_second_line_breakout':dict(hold=10, cond=lambda d:(d.breadth20_d5>0)&(d.breadth60_d5>=0)&(d.r20_pr>.45)&(d.r20_pr<.78)&(d.r60_pr>.45)&(d.r60_pr<.84)&(d.aclose>d.prior_high20)&(d.flow5_to_amount>0)&(d.dist_ma20<.11), score=lambda d:.25*d.flow5_to_amount_pr+.20*d.r20_pr+.20*d.r60_pr+.20*d.amount_ratio_pr+.15*((d.breadth20_d5.clip(-.25,.25)+.25)/.5)),
'flow_concentration_price_reclaim':dict(hold=8, cond=lambda d:(d.flow_dispersion_ctxz>.30)&(d.draw20<-.03)&(d.draw20>-.14)&(d.prev_close<=d.prev_ma20)&(d.aclose>d.ma20)&(d.flow10_to_amount>0)&(d.r3>0), score=lambda d:.30*d.flow10_to_amount_pr+.20*d.flow_accel_value_pr+.20*d.r120_pr+.15*d.r3_pr+.15*(d.flow_dispersion_ctxz.clip(0,3)/3)),
'dispersion_recovery_sponsor':dict(hold=8, cond=lambda d:(d.dispersion20_ctxz>.30)&(d.breadth20_d5>0)&(d.flow_accel_value>0)&(d.r120_pr>.55)&(d.draw20<-.04)&(d.draw20>-.15)&(d.r3>0)&(d.aclose>d.ma20), score=lambda d:.25*d.flow_accel_value_pr+.20*d.r120_pr+.20*d.r3_pr+.15*d.r60_pr+.10*(d.dispersion20_ctxz.clip(0,3)/3)+.10*((d.breadth20_d5.clip(-.25,.25)+.25)/.5)),
'participation_escape_confirmed':dict(hold=8, cond=lambda d:(d.liquidity_expansion_d5>0)&(d.breadth20_d5>0)&(d.r60_pr>.75)&(d.r20_pr>.58)&(d.r20_pr<.90)&(d.r5>0)&(d.flow5_to_amount>0)&(d.amount_ratio>1.0)&(d.amount_ratio<1.75), score=lambda d:.25*d.r60_pr+.20*d.r20_pr+.20*d.flow5_to_amount_pr+.15*d.amount_ratio_pr+.10*((d.breadth20_d5.clip(-.25,.25)+.25)/.5)+.10*((d.liquidity_expansion_d5.clip(-.25,.25)+.25)/.5)),
'liquidity_rotation_moderate_rs':dict(hold=9, cond=lambda d:(d.liquidity_expansion_d5>0)&(d.r20_pr>.45)&(d.r20_pr<.76)&(d.r60_pr>.50)&(d.r60_pr<.86)&(d.flow10_to_amount>0)&(d.aclose>d.ma20)&(d.amount_ratio>1.0)&(d.amount_ratio<1.60), score=lambda d:.25*d.flow10_to_amount_pr+.20*d.r60_pr+.20*d.r20_pr+.15*d.amount_ratio_pr+.10*d.r5_pr+.10*((d.liquidity_expansion_d5.clip(-.25,.25)+.25)/.5)),
'sponsor_reclaim_control':dict(hold=8, cond=lambda d:(d.r120_pr>.55)&(d.draw20<-.06)&(d.draw20>-.18)&(d.prev_close<=d.prev_ma20)&(d.aclose>d.ma20)&(d.flow10_to_amount>0)&(d.r3>0), score=lambda d:.30*d.flow10_to_amount_pr+.25*d.r120_pr+.20*d.r3_pr+.15*d.flow_accel_value_pr+.10*d.amount_ratio_pr),
'sponsor_reclaim_breadth_support':dict(hold=8, cond=lambda d:(d.breadth20_d5>0)&(d.breadth60_d5>=0)&(d.r120_pr>.55)&(d.draw20<-.06)&(d.draw20>-.18)&(d.prev_close<=d.prev_ma20)&(d.aclose>d.ma20)&(d.flow10_to_amount>0)&(d.r3>0), score=lambda d:.25*d.flow10_to_amount_pr+.20*d.r120_pr+.20*d.r3_pr+.15*d.flow_accel_value_pr+.10*d.amount_ratio_pr+.10*((d.breadth20_d5.clip(-.25,.25)+.25)/.5)),
'quiet_accumulation_context_release':dict(hold=11, cond=lambda d:(d.vol_compress<.80)&(d.flow20_to_amount>0)&(d.amount5_ratio<1.02)&(d.breadth20_d5>0)&(d.r20_pr>.35)&(d.r20_pr<.72)&(d.r3>0)&(d.aclose>d.ma20), score=lambda d:.30*d.flow20_to_amount_pr+.20*(1-d.vol20_pr)+.15*d.r3_pr+.15*d.r60_pr+.10*d.amount20_pr+.10*((d.breadth20_d5.clip(-.25,.25)+.25)/.5)),
'leader_lag_catchup':dict(hold=9, cond=lambda d:(d.r120_pr>.75)&(d.r60_pr>.65)&(d.r20_pr>.35)&(d.r20_pr<.66)&(d.r5>0)&(d.breadth20_d5>0)&(d.flow10_to_amount>0)&(d.aclose>d.ma20), score=lambda d:.25*d.r120_pr+.20*d.r60_pr+.20*d.flow10_to_amount_pr+.15*d.r5_pr+.10*d.r20_pr+.10*((d.breadth20_d5.clip(-.25,.25)+.25)/.5)),
'flow_breadth_inflection_second_line':dict(hold=9, cond=lambda d:(d.flow_breadth_d5>0)&(d.r20_pr>.40)&(d.r20_pr<.74)&(d.r60_pr>.45)&(d.flow20_to_amount>0)&(d.flow5_to_amount>0)&(d.aclose>d.ma20)&(d.dist_ma20<.09), score=lambda d:.25*d.flow20_to_amount_pr+.20*d.flow5_to_amount_pr+.20*d.r60_pr+.15*d.r20_pr+.10*d.amount20_pr+.10*((d.flow_breadth_d5.clip(-.25,.25)+.25)/.5)),
'continuous_context_rank_control':dict(hold=10, cond=lambda d:(d.ma20>d.ma60)&(d.r20>0)&(d.flow5_to_amount>0)&(d.dist_ma20<.10), score=lambda d:.20*d.r60_pr+.20*d.r20_pr+.20*d.flow5_to_amount_pr+.10*d.flow20_to_amount_pr+.10*((d.breadth20_ctxz.clip(-3,3)+3)/6)+.10*((d.flow_breadth_ctxz.clip(-3,3)+3)/6)+.10*((d.liquidity_expansion_ctxz.clip(-3,3)+3)/6)),
}"""
src,n=re.subn(r"SETUPS=\{.*?\n\}\nif args\.hypothesis",setups+"\nif args.hypothesis",src,flags=re.S)
assert n==1,n

# Preserve integer-share corporate-action accounting and cash compensation for fractional entitlements.
ca_pat=re.compile(r"(?m)^(\s*)if rr is not None and bool\(getattr\(rr,'is_official_event',False\)\):\n\1    sf=float\(getattr\(rr,'share_factor',1\.0\)\)\n\1    if np\.isfinite\(sf\) and sf>0 and abs\(sf-1\)>1e-12: p\['shares'\]=int\(round\(p\['shares'\]\*sf\)\)")
def repl(m):
    i=m.group(1)
    return '\n'.join([
        i+"if rr is not None and bool(getattr(rr,'is_official_event',False)):",
        i+"    old_shares=int(p['shares'])",
        i+"    factor_raw=getattr(rr,'share_factor',1.0)",
        i+"    factor=float(factor_raw) if np.isfinite(factor_raw) and float(factor_raw)>0 else 1.0",
        i+"    exact_new=old_shares*factor",
        i+"    new_shares=max(0,int(np.floor(exact_new+1e-10)))",
        i+"    cash_div_raw=getattr(rr,'cash_dividend_per_share',0.0)",
        i+"    cash_div_ps=float(cash_div_raw) if np.isfinite(cash_div_raw) else 0.0",
        i+"    ref_raw=getattr(rr,'reference_price',np.nan)",
        i+"    ref=float(ref_raw) if np.isfinite(ref_raw) and float(ref_raw)>0 else float(rr.close)",
        i+"    cash += old_shares*cash_div_ps + max(0.0,exact_new-new_shares)*ref",
        i+"    p['shares']=new_shares",
        i+"    if cash < -0.01: raise RuntimeError('negative cash after corporate action')",
    ])
src,n=ca_pat.subn(repl,src,count=1)
assert n==1,n
compile(src,'research_12_stock_repricing_v14_runtime.py','exec')
Path('scripts/research_12_stock_repricing_v14_runtime.py').write_text(src)
