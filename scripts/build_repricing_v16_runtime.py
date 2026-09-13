from pathlib import Path
import runpy,re

# Start from v15, which already injects causal context and repaired corporate actions.
runpy.run_path('scripts/build_repricing_v15_runtime.py', run_name='__main__')
p=Path('scripts/research_12_stock_repricing_v15_runtime.py')
src=p.read_text()

# V16: genuine structural variants, not threshold nudges.  Each hypothesis changes
# the causal setup archetype and/or holding horizon while keeping 2025 untouched.
setups="""SETUPS={
'flow_underreaction_early':dict(hold=6, cond=lambda d:(d.flow_accel_value>0)&(d.flow10_to_amount>0)&(d.r20_pr>.25)&(d.r20_pr<.60)&(d.dist_ma20<.06)&(d.aclose>d.ma20), score=lambda d:.40*d.flow_accel_value_pr+.25*d.flow10_to_amount_pr+.15*d.r60_pr+.10*(1-d.r3_pr)+.10*d.amount20_pr),
'flow_underreaction_patient':dict(hold=14, cond=lambda d:(d.flow20_to_amount>0)&(d.flow10_to_amount>0)&(d.r20_pr>.25)&(d.r20_pr<.62)&(d.vol20_pr<.75)&(d.aclose>d.ma20), score=lambda d:.35*d.flow20_to_amount_pr+.25*d.flow10_to_amount_pr+.15*(1-d.vol20_pr)+.15*d.r60_pr+.10*d.amount20_pr),
'fresh_sponsor_reversal':dict(hold=7, cond=lambda d:(d.inst_prev5<=0)&(d.inst5>0)&(d.draw20<-.03)&(d.draw20>-.14)&(d.aclose>d.ma20)&(d.prev_close<=d.prev_ma20), score=lambda d:.45*d.flow_accel_value_pr+.20*d.r3_pr+.15*d.amount_ratio_pr+.10*d.r120_pr+.10*(1-d.vol20_pr)),
'fresh_sponsor_breakout':dict(hold=7, cond=lambda d:(d.inst_prev5<=0)&(d.inst5>0)&(d.aclose>d.prior_high20)&(d.amount_ratio>1.05)&(d.r20_pr<.88), score=lambda d:.40*d.flow_accel_value_pr+.25*d.amount_ratio_pr+.20*d.r60_pr+.15*d.r20_pr),
'leader_absorption_then_reclaim':dict(hold=8, cond=lambda d:(d.r120_pr>.72)&(d.r60_pr>.60)&(d.draw20<-.03)&(d.draw20>-.12)&(d.flow10_to_amount>0)&(d.aclose>d.ma20)&(d.prev_close<=d.prev_ma20), score=lambda d:.30*d.r120_pr+.25*d.flow10_to_amount_pr+.20*d.r3_pr+.15*(1-d.vol20_pr)+.10*d.amount20_pr),
'leader_continuation_sponsor':dict(hold=10, cond=lambda d:(d.r120_pr>.72)&(d.r60_pr>.62)&(d.r20_pr>.45)&(d.r20_pr<.82)&(d.flow20_to_amount>0)&(d.flow10_to_amount>0)&(d.dist_ma20<.09), score=lambda d:.30*d.r120_pr+.25*d.flow20_to_amount_pr+.20*d.flow10_to_amount_pr+.15*d.r20_pr+.10*d.amount20_pr),
'midrank_rotation_sponsor':dict(hold=9, cond=lambda d:(d.r120_pr>.55)&(d.r120_pr<.82)&(d.r20_pr>.35)&(d.r20_pr<.68)&(d.flow5_to_amount>0)&(d.flow_accel_value>0)&(d.r5>0), score=lambda d:.30*d.flow_accel_value_pr+.20*d.flow5_to_amount_pr+.20*d.r120_pr+.15*d.r20_pr+.15*d.r5_pr),
'quiet_accumulation_release':dict(hold=10, cond=lambda d:(d.flow20_to_amount>0)&(d.flow10_to_amount>0)&(d.amount5_ratio<1.08)&(d.amount_ratio>1.12)&(d.vol20_pr<.60)&(d.r5>0)&(d.aclose>d.ma20), score=lambda d:.30*d.flow20_to_amount_pr+.25*(1-d.vol20_pr)+.20*d.amount_ratio_pr+.15*d.r5_pr+.10*d.r60_pr),
'compression_sponsored_escape':dict(hold=8, cond=lambda d:(d.vol_compress<.70)&(d.flow10_to_amount>0)&(d.flow20_to_amount>0)&(d.aclose>d.ma20)&(d.r3>0)&(d.amount_ratio>1.05), score=lambda d:.30*(1-d.vol_compress_pr)+.25*d.flow10_to_amount_pr+.20*d.amount_ratio_pr+.15*d.r5_pr+.10*d.r60_pr),
'breadth_expansion_stock_leader':dict(hold=8, cond=lambda d:(d.breadth20_d5>0)&(d.flow_breadth_d5>0)&(d.r60_pr>.68)&(d.r20_pr>.50)&(d.flow10_to_amount>0)&(d.dist_ma20<.10), score=lambda d:.30*d.r60_pr+.20*d.flow10_to_amount_pr+.15*d.r20_pr+.15*d.amount20_pr+.10*((d.breadth20_d5.clip(-.25,.25)+.25)/.5)+.10*((d.flow_breadth_d5.clip(-.25,.25)+.25)/.5)),
'dispersion_selective_reclaim':dict(hold=7, cond=lambda d:(d.dispersion20_ctxz>.35)&(d.flow_dispersion_ctxz>.15)&(d.r120_pr>.65)&(d.draw20<-.04)&(d.draw20>-.13)&(d.flow10_to_amount>0)&(d.r3>0)&(d.aclose>d.ma20), score=lambda d:.30*d.flow10_to_amount_pr+.20*d.r120_pr+.20*d.r3_pr+.10*d.flow_accel_value_pr+.10*(d.dispersion20_ctxz.clip(0,3)/3)+.10*(d.flow_dispersion_ctxz.clip(0,3)/3)),
'flow_agreement_quality_control':dict(hold=10, cond=lambda d:(d.foreign_trust_agree>0)&(d.flow20_to_amount>0)&(d.r60_pr>.50)&(d.r20_pr>.30)&(d.r20_pr<.75)&(d.vol20_pr<.75)&(d.aclose>d.ma20), score=lambda d:.30*d.flow20_to_amount_pr+.20*d.flow10_to_amount_pr+.20*d.r60_pr+.15*(1-d.vol20_pr)+.15*d.amount20_pr),
}"""
src,n=re.subn(r"SETUPS=\{.*?\n\}\nif args\.hypothesis",setups+"\nif args.hypothesis",src,flags=re.S)
assert n==1,n

# Accounting repair: never mark an existing holding at zero merely because that
# symbol has no row on a market-wide calendar date.  Carry only the last known
# close known by T close; entries/exits still require a real tradable row.
insert="""
last_mark={}
def mark_value(d,code,shares):
    rr=rowmap.get((d,code))
    if rr is not None and np.isfinite(rr.close) and rr.close>0:
        last_mark[code]=float(rr.close)
    pxm=last_mark.get(code,np.nan)
    return shares*pxm if np.isfinite(pxm) and pxm>0 else 0.0
"""
src,n=re.subn(r"(orders=\{k:\{\} for k in SETUPS\}; sig=\[\])",r"\1\n"+insert,src,count=1)
assert n==1,n
src=src.replace("mv=sum(p['shares']*float(rowmap[(d,c)].close) for c,p in pos.items() if (d,c) in rowmap and np.isfinite(rowmap[(d,c)].close))","mv=sum(mark_value(d,c,p['shares']) for c,p in pos.items())")
src=src.replace("nav=cash+sum(p['shares']*float(rowmap[(d,c)].close) for c,p in pos.items() if (d,c) in rowmap and np.isfinite(rowmap[(d,c)].close)); navrows.append({'date':d,'nav':nav,'cash':cash,'positions':len(pos)})","nav=cash+sum(mark_value(d,c,p['shares']) for c,p in pos.items()); navrows.append({'date':d,'nav':nav,'cash':cash,'positions':len(pos)})")

compile(src,'research_12_stock_repricing_v16_runtime.py','exec')
Path('scripts/research_12_stock_repricing_v16_runtime.py').write_text(src)
