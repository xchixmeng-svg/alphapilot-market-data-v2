from pathlib import Path
import re, subprocess

# Build on the repaired v12 shared-capital/accounting engine only.
subprocess.run(['python','scripts/build_repricing_v12_runtime.py'], check=True)
p = Path('scripts/research_12_stock_repricing_v12_runtime.py')
src = p.read_text()

# v13: structural variants seeded from the only v12 Pareto survivor (high-dispersion winner).
# These are not bull/bear gates and are not tiny threshold tweaks. Each hypothesis changes the
# context/selection mechanism materially. All context fields are causal T-close features whose
# rolling normalization is fit on T-1 and earlier in v10+. 2025 remains untouched evidence.
setups = r'''SETUPS={
'dispersion_flow_leader':dict(hold=9, cond=lambda d:(d.dispersion20_ctxz>.45)&(d.r120_pr>.70)&(d.r60_pr>.78)&(d.r20_pr>.62)&(d.flow_accel_value>0)&(d.flow5_to_amount>0)&(d.aclose>d.ma20), score=lambda d:.25*d.r120_pr+.20*d.r60_pr+.20*d.flow_accel_value_pr+.15*d.flow5_to_amount_pr+.10*d.r20_pr+.10*(d.dispersion20_ctxz.clip(0,3)/3)),
'dispersion_lowvol_leader':dict(hold=12, cond=lambda d:(d.dispersion20_ctxz>.45)&(d.r60_pr>.78)&(d.r20_pr>.58)&(d.vol20_pr<.45)&(d.flow10_to_amount>0)&(d.dist_ma20<.08), score=lambda d:.25*d.r60_pr+.20*d.r20_pr+.20*(1-d.vol20_pr)+.20*d.flow10_to_amount_pr+.15*d.r120_pr),
'dispersion_consensus_leader':dict(hold=10, cond=lambda d:(d.dispersion20_ctxz>.35)&(d.foreign_trust_agree>0)&(d.inst10>0)&(d.r60_pr>.75)&(d.r20_pr>.55)&(d.aclose>d.ma20), score=lambda d:.25*d.r60_pr+.20*d.r20_pr+.20*d.flow10_to_amount_pr+.15*d.flow20_to_amount_pr+.10*d.r120_pr+.10*(d.dispersion20_ctxz.clip(0,3)/3)),
'dispersion_liquidity_burst_leader':dict(hold=7, cond=lambda d:(d.dispersion20_ctxz>.45)&(d.liquidity_expansion_ctxz>.20)&(d.r60_pr>.78)&(d.r20_pr>.65)&(d.amount_ratio>1.15)&(d.flow5_to_amount>0), score=lambda d:.25*d.r60_pr+.20*d.r20_pr+.20*d.amount_ratio_pr+.15*d.flow5_to_amount_pr+.10*d.r120_pr+.10*(d.liquidity_expansion_ctxz.clip(0,3)/3)),
'dispersion_pullback_reentry':dict(hold=8, cond=lambda d:(d.dispersion20_ctxz>.35)&(d.r120_pr>.72)&(d.r60_pr>.68)&(d.draw20<-.03)&(d.draw20>-.13)&(d.prev_close<=d.prev_ma20)&(d.aclose>d.ma20)&(d.flow5_to_amount>0), score=lambda d:.25*d.r120_pr+.20*d.r60_pr+.20*d.flow5_to_amount_pr+.15*d.r3_pr+.10*(1-d.vol20_pr)+.10*(d.dispersion20_ctxz.clip(0,3)/3)),
'breadth_recovery_repricing':dict(hold=10, cond=lambda d:(d.breadth20_d5>0)&(d.breadth60_ctxz>-.60)&(d.r20_pr>.50)&(d.r20_pr<.88)&(d.r60_pr>.55)&(d.flow10_to_amount>0)&(d.ma20>d.ma60), score=lambda d:.25*d.r20_pr+.20*d.r60_pr+.20*d.flow10_to_amount_pr+.15*d.flow20_to_amount_pr+.10*d.amount20_pr+.10*((d.breadth20_d5*5).clip(-1,1)+1)/2),
'flow_breadth_recovery_leader':dict(hold=9, cond=lambda d:(d.flow_breadth_d5>0)&(d.flow_breadth_ctxz>-.50)&(d.r60_pr>.72)&(d.r20_pr>.55)&(d.flow5_to_amount>0)&(d.flow_accel_value>0)&(d.aclose>d.ma20), score=lambda d:.25*d.flow_accel_value_pr+.20*d.flow5_to_amount_pr+.20*d.r60_pr+.15*d.r20_pr+.10*d.r120_pr+.10*((d.flow_breadth_d5*5).clip(-1,1)+1)/2),
'liquidity_contraction_leader':dict(hold=12, cond=lambda d:(d.liquidity_expansion_ctxz<-.20)&(d.r60_pr>.78)&(d.r20_pr>.58)&(d.amount_ratio<1.10)&(d.amount5_ratio<1.10)&(d.flow10_to_amount>0)&(d.dist_ma20<.07), score=lambda d:.25*d.r60_pr+.20*d.r20_pr+.20*d.flow10_to_amount_pr+.15*(1-d.amount_ratio_pr)+.10*(1-d.vol20_pr)+.10*d.r120_pr),
'dispersion_flow_concentration':dict(hold=9, cond=lambda d:(d.dispersion20_ctxz>.45)&(d.flow_dispersion_ctxz>.45)&(d.r60_pr>.72)&(d.flow5_to_amount_pr>.75)&(d.flow10_to_amount>0)&(d.aclose>d.ma20), score=lambda d:.25*d.flow5_to_amount_pr+.20*d.flow10_to_amount_pr+.20*d.r60_pr+.15*d.r20_pr+.10*(d.flow_dispersion_ctxz.clip(0,3)/3)+.10*(d.dispersion20_ctxz.clip(0,3)/3)),
'multi_horizon_leader_persistence':dict(hold=11, cond=lambda d:(d.r120_pr>.75)&(d.r60_pr>.72)&(d.r20_pr>.60)&(d.r10_pr>.50)&(d.flow10_to_amount>0)&(d.amount_ratio<1.60)&(d.dist_ma20<.10), score=lambda d:.25*d.r120_pr+.20*d.r60_pr+.15*d.r20_pr+.15*d.r10_pr+.15*d.flow10_to_amount_pr+.10*(1-d.vol20_pr)),
'moderate_momentum_sponsored':dict(hold=14, cond=lambda d:(d.r60_pr>.58)&(d.r60_pr<.88)&(d.r20_pr>.45)&(d.r20_pr<.78)&(d.flow20_to_amount>0)&(d.flow10_to_amount>0)&(d.ma20>d.ma60)&(d.dist_ma20<.07), score=lambda d:.25*d.flow20_to_amount_pr+.20*d.flow10_to_amount_pr+.20*d.r60_pr+.15*d.r20_pr+.10*(1-d.vol20_pr)+.10*d.amount20_pr),
'participation_escape_leader':dict(hold=8, cond=lambda d:(d.breadth20_ctxz<-.20)&(d.breadth20_d5>0)&(d.r60_pr>.82)&(d.aclose>d.prior_high20)&(d.flow5_to_amount>0)&(d.amount_ratio>1.0), score=lambda d:.25*d.r60_pr+.20*d.r20_pr+.20*d.flow5_to_amount_pr+.15*d.amount_ratio_pr+.10*((-d.breadth20_ctxz).clip(0,3)/3)+.10*((d.breadth20_d5*5).clip(-1,1)+1)/2),
}'''
src, n = re.subn(r"SETUPS=\{.*?\n\}\nif args\.hypothesis", setups+"\nif args.hypothesis", src, flags=re.S)
assert n == 1, n

src = src.replace("research_out_12_stock_setups_v12", "research_out_12_stock_setups_v13")
compile(src, 'scripts/research_12_stock_repricing_v13_runtime.py', 'exec')
Path('scripts/research_12_stock_repricing_v13_runtime.py').write_text(src)
