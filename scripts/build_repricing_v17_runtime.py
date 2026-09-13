from pathlib import Path
import runpy,re

runpy.run_path('scripts/build_repricing_v16_runtime.py', run_name='__main__')
p=Path('scripts/research_12_stock_repricing_v16_runtime.py')
src=p.read_text()

# Add causal stock-vs-market residual features. Current-day market medians are known at T close;
# percentile ranks are cross-sectional at the same T close, with no future information.
ins="""
stocks['resid20']=stocks['r20']-stocks['med_r20']
stocks['resid60']=stocks['r60']-stocks['med_r60']
stocks['resid20_pr']=stocks.groupby('date')['resid20'].rank(pct=True)
stocks['resid60_pr']=stocks.groupby('date')['resid60'].rank(pct=True)
"""
src,n=re.subn(r"(stocks=stocks\.drop\(columns=ctx_cols,errors='ignore'\)\.merge\(daily,on='date',how='left'\)\n)",r"\1"+ins,src,count=1)
assert n==1,n

setups="""SETUPS={
'breadth_expansion_residual_leader':dict(hold=8, cond=lambda d:(d.breadth20_d5>0)&(d.flow_breadth_d5>0)&(d.resid20_pr>.72)&(d.resid60_pr>.68)&(d.flow10_to_amount>0)&(d.dist_ma20<.10), score=lambda d:.28*d.resid20_pr+.22*d.resid60_pr+.20*d.flow10_to_amount_pr+.12*d.amount20_pr+.10*((d.breadth20_d5.clip(-.25,.25)+.25)/.5)+.08*((d.flow_breadth_d5.clip(-.25,.25)+.25)/.5)),
'breadth_expansion_catchup_sponsor':dict(hold=9, cond=lambda d:(d.breadth20_d5>0)&(d.r60_pr>.52)&(d.r60_pr<.82)&(d.resid20_pr>.62)&(d.flow5_to_amount>0)&(d.flow_accel_value>0)&(d.r5>0), score=lambda d:.28*d.flow_accel_value_pr+.22*d.resid20_pr+.18*d.r60_pr+.17*d.flow5_to_amount_pr+.15*d.r5_pr),
'breadth_contraction_resilient_leader':dict(hold=7, cond=lambda d:(d.breadth20_d5<0)&(d.resid20_pr>.82)&(d.resid60_pr>.75)&(d.flow10_to_amount>0)&(d.aclose>d.ma20)&(d.vol20_pr<.72), score=lambda d:.32*d.resid20_pr+.25*d.resid60_pr+.20*d.flow10_to_amount_pr+.13*(1-d.vol20_pr)+.10*d.amount20_pr),
'flow_breadth_expansion_early_rotation':dict(hold=8, cond=lambda d:(d.flow_breadth_d5>0)&(d.breadth20_d5>=-.02)&(d.r20_pr>.38)&(d.r20_pr<.72)&(d.resid20_pr>.60)&(d.flow_accel_value>0)&(d.amount_ratio<1.65), score=lambda d:.30*d.flow_accel_value_pr+.22*d.resid20_pr+.18*d.r20_pr+.15*d.amount_ratio_pr+.15*((d.flow_breadth_d5.clip(-.25,.25)+.25)/.5)),
'flow_breadth_concentration_top_leader':dict(hold=10, cond=lambda d:(d.flow_breadth_d5<0)&(d.resid60_pr>.82)&(d.r120_pr>.72)&(d.flow20_to_amount>0)&(d.flow10_to_amount>0)&(d.dist_ma20<.10), score=lambda d:.28*d.resid60_pr+.25*d.r120_pr+.20*d.flow20_to_amount_pr+.17*d.flow10_to_amount_pr+.10*d.amount20_pr),
'liquidity_expansion_orderly_breakout':dict(hold=7, cond=lambda d:(d.liquidity_expansion_d5>0)&(d.aclose>d.prior_high20)&(d.resid20_pr>.65)&(d.flow5_to_amount>0)&(d.amount_ratio>1.05)&(d.amount_ratio<1.75), score=lambda d:.26*d.resid20_pr+.24*d.flow5_to_amount_pr+.20*d.amount_ratio_pr+.18*d.r60_pr+.12*((d.liquidity_expansion_d5.clip(-.25,.25)+.25)/.5)),
'liquidity_contraction_sponsored_reclaim':dict(hold=8, cond=lambda d:(d.liquidity_expansion_d5<0)&(d.resid60_pr>.68)&(d.draw20<-.03)&(d.draw20>-.12)&(d.flow10_to_amount>0)&(d.prev_close<=d.prev_ma20)&(d.aclose>d.ma20), score=lambda d:.30*d.resid60_pr+.26*d.flow10_to_amount_pr+.20*d.r3_pr+.14*(1-d.vol20_pr)+.10*d.amount20_pr),
'high_dispersion_residual_winner':dict(hold=9, cond=lambda d:(d.dispersion20_ctxz>.45)&(d.resid20_pr>.85)&(d.resid60_pr>.70)&(d.flow10_to_amount>0)&(d.r20_pr<.90)&(d.dist_ma20<.12), score=lambda d:.34*d.resid20_pr+.22*d.resid60_pr+.20*d.flow10_to_amount_pr+.14*d.r60_pr+.10*(d.dispersion20_ctxz.clip(0,3)/3)),
'high_dispersion_sponsored_reclaim':dict(hold=7, cond=lambda d:(d.dispersion20_ctxz>.45)&(d.draw20<-.04)&(d.draw20>-.14)&(d.resid60_pr>.65)&(d.flow_accel_value>0)&(d.r3>0)&(d.aclose>d.ma20), score=lambda d:.28*d.flow_accel_value_pr+.24*d.resid60_pr+.20*d.r3_pr+.16*(1-d.vol20_pr)+.12*(d.dispersion20_ctxz.clip(0,3)/3)),
'low_dispersion_compression_escape':dict(hold=8, cond=lambda d:(d.dispersion20_ctxz<-.35)&(d.vol_compress<.72)&(d.aclose>d.prior_high20)&(d.flow10_to_amount>0)&(d.amount_ratio>1.05)&(d.resid20_pr>.58), score=lambda d:.28*(1-d.vol_compress_pr)+.24*d.flow10_to_amount_pr+.20*d.amount_ratio_pr+.18*d.resid20_pr+.10*d.r60_pr),
'weak_market_positive_residual':dict(hold=8, cond=lambda d:(d.med_r20<0)&(d.resid20_pr>.88)&(d.resid60_pr>.75)&(d.flow5_to_amount>0)&(d.aclose>d.ma20)&(d.r5>0), score=lambda d:.34*d.resid20_pr+.24*d.resid60_pr+.20*d.flow5_to_amount_pr+.12*d.r5_pr+.10*d.amount20_pr),
'market_recovery_sponsor':dict(hold=9, cond=lambda d:(d.med_r20_d5>0)&(d.breadth20_d5>0)&(d.resid20_pr>.62)&(d.flow10_to_amount>0)&(d.r20_pr>.40)&(d.r20_pr<.80)&(d.dist_ma20<.09), score=lambda d:.28*d.resid20_pr+.22*d.flow10_to_amount_pr+.18*d.r60_pr+.14*d.r20_pr+.10*((d.breadth20_d5.clip(-.25,.25)+.25)/.5)+.08*((d.med_r20_d5.clip(-.12,.12)+.12)/.24)),
}"""
src,n=re.subn(r"SETUPS=\{.*?\n\}\nif args\.hypothesis",setups+"\nif args.hypothesis",src,flags=re.S)
assert n==1,n
compile(src,'research_12_stock_repricing_v17_runtime.py','exec')
Path('scripts/research_12_stock_repricing_v17_runtime.py').write_text(src)
