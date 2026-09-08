#!/usr/bin/env python3
"""Research-only full-exposure overlay on the locked R10 MAX selection engine."""
from pathlib import Path
import hashlib, json, re

root=Path(__file__).resolve().parent.parent
src=(root/"scripts"/"r10_max_formal.py").read_text(encoding="utf-8")
manifest=json.loads(Path("input_manifest.json").read_text(encoding="utf-8"))
if manifest.get("status")!="PASS":
    raise RuntimeError("validation inputs did not pass audit")

# Bind the audited 2015-2020 dataset while preserving the locked engine source.
block=re.compile(r"EXPECTED_INPUT_HASHES = \{.*?\n\}\n\nfor filename, expected in EXPECTED_INPUT_HASHES.items\(\):.*?raise RuntimeError\(f'input SHA mismatch: \{filename\}: \{actual\} != \{expected\}'\)\n",re.S)
replacement="""EXPECTED_INPUT_HASHES = {
    'institutional_2015_2020.parquet': MANIFEST_HASHES['institutional'],
    'ohlcv_2015.parquet': MANIFEST_HASHES['ohlcv_2015'],
    'ohlcv_2016.parquet': MANIFEST_HASHES['ohlcv_2016'],
    'ohlcv_2017.parquet': MANIFEST_HASHES['ohlcv_2017'],
    'ohlcv_2018.parquet': MANIFEST_HASHES['ohlcv_2018'],
    'ohlcv_2019.parquet': MANIFEST_HASHES['ohlcv_2019'],
    'ohlcv_2020.parquet': MANIFEST_HASHES['ohlcv_2020'],
}
for filename, expected in EXPECTED_INPUT_HASHES.items():
    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'input SHA mismatch: {filename}: {actual} != {expected}')
"""
if not block.search(src):
    raise RuntimeError("locked hash block signature changed")
src=block.sub(replacement,src,count=1)
src=src.replace("for y in range(2020, 2026):","for y in range(2015, 2021):",1)
src=src.replace("official_corporate_actions_2020_2025.csv","official_corporate_actions_2015_2020.csv")
src=src.replace("ohlcv_causal_2020_2025.csv.gz","ohlcv_causal_2015_2020.csv.gz")
src=src.replace("institutional_2020_2025.parquet","institutional_2015_2020.parquet")
src=src.replace(
    "eval_dates = [d for d in all_dates if 20210104 <= int(d) <= 20251231]",
    "eval_dates = [d for d in all_dates if 20160104 <= int(d) <= 20201231]",
)
src=src.replace(
    "assert eval_dates[0] == 20210104 and eval_dates[-1] == 20251231",
    "assert eval_dates[0] == 20160104 and eval_dates[-1] == 20201231",
)
src=src.replace("for year in range(2021, 2026):","for year in range(2016, 2021):")
src=src.replace(
    "'exact_evaluation_window': eval_dates[0] == 20210104 and eval_dates[-1] == 20251231",
    "'exact_evaluation_window': eval_dates[0] == 20160104 and eval_dates[-1] == 20201231",
)

# Research overlay only:
# 1) never scale order sizing down because of portfolio drawdown;
# 2) always make five R7 slots available with a 95% portfolio ceiling;
# 3) do not force-sell/cool down on portfolio drawdown;
# 4) do not exit an R7 position solely because the market exposure switch is zero.
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

h={"institutional":manifest["institutional"]["sha256"]}
for row in manifest["ohlcv"]:
    h[f"ohlcv_{row['year']}"]=row["sha256"]
prefix="MANIFEST_HASHES="+repr(h)+"\n"
generated=Path("r10_max_full_exposure_2016_2020_generated.py")
generated.write_text(prefix+src,encoding="utf-8")
locked_bytes=(root/"scripts"/"r10_max_formal.py").read_bytes()
locked_blob_hash=hashlib.sha1(
    f"blob {len(locked_bytes)}\\0".encode("ascii")+locked_bytes
).hexdigest()
if locked_blob_hash!="cc5d3ee1f59f44914a74bd2c3b379e3b17f2f034":
    raise RuntimeError("locked engine blob changed: "+locked_blob_hash)
print("LOCKED_ENGINE_BLOB PASS",locked_blob_hash,flush=True)
exec(compile(generated.read_text(encoding="utf-8"),str(generated),"exec"),{"__name__":"__main__","__file__":str(generated)})

experiment={
    "name":"R10 MAX Full Exposure Research",
    "formal_strategy_modified":False,
    "base_locked_commit":"3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9",
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
