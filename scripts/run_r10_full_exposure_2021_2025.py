#!/usr/bin/env python3
"""Research-only full-exposure overlay for the locked 2021-2025 R10 MAX engine."""
from pathlib import Path
import hashlib, json

root=Path(__file__).resolve().parent.parent
formal_path=root/"scripts"/"r10_max_formal.py"
locked_bytes=formal_path.read_bytes()
locked_blob_hash=hashlib.sha1(
    f"blob {len(locked_bytes)}".encode("ascii")+bytes([0])+locked_bytes
).hexdigest()
if locked_blob_hash!="cc5d3ee1f59f44914a74bd2c3b379e3b17f2f034":
    raise RuntimeError("locked engine blob changed: "+locked_blob_hash)

src=locked_bytes.decode("utf-8")

old_dd="""def dd_multiplier(dd):
    if dd <= -0.15: return 0.40
    if dd <= -0.09: return 0.45
    if dd <= -0.06: return 0.85
    return 1.0
"""
new_dd="""def dd_multiplier(dd):
    return 1.0
"""
if old_dd not in src:
    raise RuntimeError("drawdown multiplier signature changed")
src=src.replace(old_dd,new_dd,1)

old_r7_exit="""            elif float(r.get('r7_exposure', 1.0)) <= 0.0 or not bool(r.get('r7_hard', False)):
                reason = 'R7_REB'
"""
new_r7_exit="""            elif not bool(r.get('r7_hard', False)):
                reason = 'R7_REB'
"""
if old_r7_exit not in src:
    raise RuntimeError("R7 market exit signature changed")
src=src.replace(old_r7_exit,new_r7_exit,1)

old_force="if dd <= -FORCE_DD and i >= force_cooldown_until:"
if old_force not in src:
    raise RuntimeError("forced drawdown signature changed")
src=src.replace(old_force,"if False and dd <= -FORCE_DD and i >= force_cooldown_until:",1)

old_exposure="""            r7_exposure = float(sub['r7_exposure'].iloc[0]) if len(sub) and np.isfinite(sub['r7_exposure'].iloc[0]) else 0.0
            r7_slots_regime = int(sub['r7_slots'].iloc[0]) if len(sub) and np.isfinite(sub['r7_slots'].iloc[0]) else 0
"""
new_exposure="""            r7_exposure = 0.95
            r7_slots_regime = MAX_POSITIONS
"""
if old_exposure not in src:
    raise RuntimeError("R7 exposure assignment signature changed")
src=src.replace(old_exposure,new_exposure,1)

generated=Path("r10_max_full_exposure_2021_2025_generated.py")
generated.write_text(src,encoding="utf-8")
print("LOCKED_ENGINE_BLOB PASS",locked_blob_hash,flush=True)
exec(compile(src,str(generated),"exec"),{"__name__":"__main__","__file__":str(generated)})

experiment={
    "name":"R10 MAX Full Exposure 2021-2025 Research",
    "formal_strategy_modified":False,
    "base_locked_commit":"3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9",
    "evaluation":"2021-2025; 2020 warm-up",
    "research_changes":{
        "r7_slots":"always 5 when qualifying signals exist",
        "portfolio_cap":0.95,
        "single_stock_cap":0.25,
        "drawdown_sizing_multiplier":1.0,
        "forced_drawdown_liquidation":False,
        "market_exposure_only_exit":False,
        "selection_rules":"unchanged",
        "r05_risk_on_filter":"unchanged",
        "execution":"T+1, integer shares, original fees/tax/slippage",
    },
}
Path("full_exposure_experiment.json").write_text(
    json.dumps(experiment,ensure_ascii=False,indent=2),encoding="utf-8"
)
summary_path=Path("r10max_formal_summary.json")
summary=json.loads(summary_path.read_text(encoding="utf-8"))
summary["research_variant"]=experiment
summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
