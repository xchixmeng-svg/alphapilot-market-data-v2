from pathlib import Path
import re, subprocess

# Build on the repaired v11 shared-capital/accounting engine only.
subprocess.run(['python','scripts/build_repricing_v11_runtime.py'], check=True)
p = Path('scripts/research_12_stock_repricing_v11_runtime.py')
src = p.read_text()

# Twelve genuinely different stock-level repricing playbooks. Market context is a continuous
# conditioning feature, never a hard bull/bear gate. All fields are known at T close; execution is T+1.
setups = r'''SETUPS={
'early_flow_inflection':dict(hold=7, cond=lambda d:(d.inst_prev5<=0)&(d.inst5>0)&(d.flow_accel_value>0)&(d.r20_pr>.30)&(d.r20_pr<.80)&(d.aclose>d.ma20), score=lambda d:.40*d.flow_accel_value_pr+.20*d.flow5_to_amount_pr+.15*d.r20_pr+.15*d.amount5_ratio_pr+.10*d.r60_pr),
'quiet_sponsored_base':dict(hold=13, cond=lambda d:(d.flow20_to_amount>0)&(d.flow10_to_amount>0)&(d.vol_compress<.75)&(d.amount_ratio<1.20)&(d.amount5_ratio<1.10)&(d.aclose>d.ma20)&(d.dist_ma20<.06), score=lambda d:.35*d.flow20_to_amount_pr+.20*d.flow10_to_amount_pr+.20*(1-d.vol20_pr)+.15*d.r60_pr+.10*d.amount20_pr),
'breakout_with_sponsorship':dict(hold=8, cond=lambda d:(d.aclose>d.prior_high20)&(d.flow5_to_amount>0)&(d.flow10_to_amount>0)&(d.amount_ratio>1.05)&(d.r20_pr>.55)&(d.dist_ma20<.14), score=lambda d:.30*d.flow10_to_amount_pr+.25*d.r60_pr+.20*d.amount_ratio_pr+.15*d.r20_pr+.10*d.flow_accel_value_pr),
'rs_reset_reclaim':dict(hold=8, cond=lambda d:(d.r120_pr>.75)&(d.r60_pr>.65)&(d.draw20<-.04)&(d.draw20>-.15)&(d.prev_close<=d.prev_ma20)&(d.aclose>d.ma20)&(d.r3>0), score=lambda d:.30*d.r120_pr+.20*d.r60_pr+.20*d.r3_pr+.15*d.flow_accel_value_pr+.15*(1-d.vol20_pr)),
'flow_price_divergence':dict(hold=7, cond=lambda d:(d.r10<0)&(d.r5<=0)&(d.flow5_to_amount>0)&(d.flow_accel_value>0)&(d.aclose>d.ma60)&(d.r3>0), score=lambda d:.35*d.flow_accel_value_pr+.25*d.flow5_to_amount_pr+.15*(1-d.r10_pr)+.15*d.r3_pr+.10*d.amount20_pr),
'consensus_sponsor_trend':dict(hold=12, cond=lambda d:(d.foreign_trust_agree>0)&(d.inst10>0)&(d.inst20>0)&(d.ma20>d.ma60)&(d.r60_pr>.55)&(d.dist_ma20<.10), score=lambda d:.30*d.flow20_to_amount_pr+.20*d.flow10_to_amount_pr+.20*d.r60_pr+.15*d.r20_pr+.15*(1-d.vol20_pr)),
'liquidity_surge_continuation':dict(hold=7, cond=lambda d:(d.amount5_ratio>1.25)&(d.amount_ratio>.90)&(d.r5>0)&(d.r20>0)&(d.flow5_to_amount>0)&(d.aclose>d.ma20), score=lambda d:.30*d.amount5_ratio_pr+.25*d.r20_pr+.20*d.r5_pr+.15*d.flow5_to_amount_pr+.10*d.r60_pr),
'narrow_leadership_escape':dict(hold=8, cond=lambda d:(d.breadth20_ctxz<-.25)&(d.r60_pr>.88)&(d.r20_pr>.75)&(d.flow10_to_amount>0)&(d.aclose>d.ma20)&(d.amount_ratio>.90), score=lambda d:.30*d.r60_pr+.25*d.r20_pr+.20*d.flow10_to_amount_pr+.15*d.amount_ratio_pr+.10*((-d.breadth20_ctxz).clip(0,3)/3)),
'broad_participation_rotation':dict(hold=11, cond=lambda d:(d.breadth20_ctxz>.25)&(d.breadth60_ctxz>-.10)&(d.flow_breadth_ctxz>-.25)&(d.r20_pr>.45)&(d.r20_pr<.80)&(d.ma20>d.ma60), score=lambda d:.25*d.r20_pr+.20*d.r60_pr+.20*d.flow20_to_amount_pr+.15*d.amount20_pr+.10*(1-d.vol20_pr)+.10*(d.breadth20_ctxz.clip(0,3)/3)),
'high_dispersion_winner':dict(hold=9, cond=lambda d:(d.dispersion20_ctxz>.50)&(d.r60_pr>.82)&(d.r20_pr>.70)&(d.flow5_to_amount>0)&(d.aclose>d.ma20)&(d.dist_ma20<.12), score=lambda d:.30*d.r60_pr+.20*d.r20_pr+.20*d.flow5_to_amount_pr+.15*d.amount20_pr+.15*(d.dispersion20_ctxz.clip(0,3)/3)),
'low_vol_sponsor_reacceleration':dict(hold=14, cond=lambda d:(d.vol20_pr<.40)&(d.flow20_to_amount>0)&(d.flow_accel_value>0)&(d.ma20>d.ma60)&(d.r20>0)&(d.r5>0)&(d.dist_ma20<.06), score=lambda d:.30*d.flow20_to_amount_pr+.25*d.flow_accel_value_pr+.20*(1-d.vol20_pr)+.15*d.r60_pr+.10*d.r5_pr),
'multi_horizon_pause_resume':dict(hold=9, cond=lambda d:(d.r120_pr>.60)&(d.r60_pr>.60)&(d.r20_pr>.40)&(d.r5<.08)&(d.r3>0)&(d.flow_accel_value>0)&(d.aclose>d.ma20), score=lambda d:.25*d.r120_pr+.20*d.r60_pr+.20*d.flow_accel_value_pr+.15*d.r3_pr+.10*d.flow5_to_amount_pr+.10*(1-d.vol20_pr)),
}'''
src, n = re.subn(r"SETUPS=\{.*?\n\}\nif args\.hypothesis", setups+"\nif args.hypothesis", src, flags=re.S)
assert n == 1, n

# Label this batch distinctly while retaining the same validated output/accounting machinery.
src = src.replace("research_out_12_stock_setups", "research_out_12_stock_setups_v12")
compile(src, 'scripts/research_12_stock_repricing_v12_runtime.py', 'exec')
Path('scripts/research_12_stock_repricing_v12_runtime.py').write_text(src)
