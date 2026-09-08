#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
FORMAL=ROOT/"scripts"/"r10_max_formal.py"
EXPECTED_BLOB="cc5d3ee1f59f44914a74bd2c3b379e3b17f2f034"

PROFILES={
    "formal":None,
    "defensive_mix":{
        "Strong Bull":(1.00,4,1),"Normal Bull":(0.85,3,1),
        "Repair":(0.65,2,1),"Weak":(0.25,1,0),
        "Bear":(0.0,0,0),"Fallback/Bear":(0.0,0,0),"Unknown":(0.0,0,0),
    },
    "balanced_core":{
        "Strong Bull":(1.00,4,1),"Normal Bull":(0.90,4,1),
        "Repair":(0.75,3,1),"Weak":(0.35,2,0),
        "Bear":(0.0,0,0),"Fallback/Bear":(0.0,0,0),"Unknown":(0.0,0,0),
    },
    "momentum_guard":{
        "Strong Bull":(1.00,5,0),"Normal Bull":(0.90,4,0),
        "Repair":(0.70,3,0),"Weak":(0.25,1,0),
        "Bear":(0.0,0,0),"Fallback/Bear":(0.0,0,0),"Unknown":(0.0,0,0),
    },
    "mixed_reserve":{
        "Strong Bull":(0.95,3,2),"Normal Bull":(0.80,3,1),
        "Repair":(0.60,2,1),"Weak":(0.20,1,0),
        "Bear":(0.0,0,0),"Fallback/Bear":(0.0,0,0),"Unknown":(0.0,0,0),
    },
}

def git_blob_sha(data:bytes)->str:
    return hashlib.sha1(f"blob {len(data)}".encode("ascii")+bytes([0])+data).hexdigest()

def bind_old(src:str,start:int,end:int,first_year:int,last_year_exclusive:int,manifest:dict)->str:
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
        raise RuntimeError("hash block signature changed")
    src=block.sub(replacement,src,count=1)
    src=src.replace("for y in range(2020, 2026):","for y in range(2015, 2021):",1)
    src=src.replace("official_corporate_actions_2020_2025.csv","official_corporate_actions_2015_2020.csv")
    src=src.replace("ohlcv_causal_2020_2025.csv.gz",f"ohlcv_causal_{start}_{end}.csv.gz")
    src=src.replace("institutional_2020_2025.parquet","institutional_2015_2020.parquet")
    src=src.replace(
        "eval_dates = [d for d in all_dates if 20210104 <= int(d) <= 20251231]",
        f"eval_dates = [d for d in all_dates if {start} <= int(d) <= {end}]",
    )
    src=src.replace(
        "assert eval_dates[0] == 20210104 and eval_dates[-1] == 20251231",
        f"assert eval_dates[0] == {start} and eval_dates[-1] == {end}",
    )
    src=src.replace("for year in range(2021, 2026):",f"for year in range({first_year}, {last_year_exclusive}):")
    src=src.replace(
        "'exact_evaluation_window': eval_dates[0] == 20210104 and eval_dates[-1] == 20251231",
        f"'exact_evaluation_window': eval_dates[0] == {start} and eval_dates[-1] == {end}",
    )
    hashes={"institutional":manifest["institutional"]["sha256"]}
    for row in manifest["ohlcv"]:
        hashes[f"ohlcv_{row['year']}"]=row["sha256"]
    return "MANIFEST_HASHES="+repr(hashes)+"\n"+src

def apply_profile(src:str,name:str)->str:
    profile=PROFILES[name]
    if profile is None:
        return src
    old="""            r7_exposure = float(sub['r7_exposure'].iloc[0]) if len(sub) and np.isfinite(sub['r7_exposure'].iloc[0]) else 0.0
            r7_slots_regime = int(sub['r7_slots'].iloc[0]) if len(sub) and np.isfinite(sub['r7_slots'].iloc[0]) else 0
"""
    new=f"""            regime_now = str(sub['regime'].iloc[0]) if len(sub) else 'Unknown'
            regime_allocation = {repr(profile)}
            r7_exposure, r7_slots_regime, r05_slots_regime = regime_allocation.get(regime_now, (0.0, 0, 0))
"""
    if old not in src:
        raise RuntimeError("exposure assignment signature changed")
    src=src.replace(old,new,1)
    old_free="r7_free, r05_free = max(0, r7_slots_regime - n_r7), max(0, R05_MAX_SLOTS - n_r05)"
    new_free="r7_free, r05_free = max(0, r7_slots_regime - n_r7), max(0, r05_slots_regime - n_r05)"
    if old_free not in src:
        raise RuntimeError("slot allocation signature changed")
    return src.replace(old_free,new_free,1)

def link_inputs(case:Path,mode:str):
    case.mkdir(parents=True,exist_ok=True)
    if mode=="old":
        source=ROOT/"validation_input"
        for p in source.iterdir():
            os.symlink(p.resolve(),case/p.name)
    else:
        hist=ROOT/"data"/"history"/"2020-2025"
        for y in range(2020,2026):
            p=hist/f"ohlcv_{y}.parquet"
            os.symlink(p.resolve(),case/p.name)
        p=hist/"institutional_2020_2025.parquet"
        os.symlink(p.resolve(),case/p.name)
        p=ROOT/"data"/"reference"/"official_corporate_actions_2020_2025.csv"
        os.symlink(p.resolve(),case/p.name)

def run_case(label:str,src:str,mode:str,profile:str)->dict:
    case=ROOT/"multi_regime_runs"/label
    if case.exists():
        shutil.rmtree(case)
    link_inputs(case,mode)
    generated=case/"generated.py"
    generated.write_text(src,encoding="utf-8")
    proc=subprocess.run([sys.executable,str(generated)],cwd=case,text=True,capture_output=True)
    (case/"execution.log").write_text(proc.stdout+"\nSTDERR\n"+proc.stderr,encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{label} failed; see {case/'execution.log'}")
    summary=json.loads((case/"r10max_formal_summary.json").read_text())
    audit=json.loads((case/"contract_audit.json").read_text())
    if not audit.get("all_pass"):
        raise RuntimeError(f"{label} contract failed")
    trades=pd.read_csv(case/"r10max_formal_trades.csv")
    nav=pd.read_csv(case/"r10max_formal_nav.csv")
    ret=pd.to_numeric(trades.get("return"),errors="coerce")
    result={
        "label":label,"profile":profile,
        **summary["strategy"],
        "benchmark":summary["benchmark_0050"],
        "forced_drawdown_events":summary["forced_drawdown_events"],
        "average_exposure":float(nav.exposure.mean()),
        "median_exposure":float(nav.exposure.median()),
        "maximum_exposure":float(nav.exposure.max()),
        "loss_le_12":int((ret<=-0.12).sum()),
        "loss_le_20":int((ret<=-0.20).sum()),
        "worst_trade":float(ret.min()),
        "annual":summary["annual"],
    }
    return result

def main():
    data=FORMAL.read_bytes()
    if git_blob_sha(data)!=EXPECTED_BLOB:
        raise RuntimeError("formal engine blob changed")
    base=data.decode("utf-8")
    manifest=json.loads((ROOT/"validation_input"/"input_manifest.json").read_text())
    dev_base=bind_old(base,20160104,20191231,2016,2020,manifest)
    results=[]
    for profile in PROFILES:
        print("[DEV]",profile,flush=True)
        results.append(run_case("dev_"+profile,apply_profile(dev_base,profile),"old",profile))
    # Contract gate: the development period must meet the formal balance targets.
    eligible=[r for r in results if r["max_drawdown"]>=-0.22 and r["profit_factor"]>=1.50]
    if not eligible:
        # Still run the least-bad candidate through later periods for diagnosis,
        # but never promote it as an accepted balanced strategy.
        diagnostic=[r for r in results if r["max_drawdown"]>=-0.25]
        selected=max(diagnostic or results,key=lambda r:(r["max_drawdown"],r["profit_factor"],r["cagr"]))
        gate="NO_PROFILE_MET_MAXDD_22_AND_PF_150"
    else:
        selected=max(eligible,key=lambda r:(r["cagr"],r["profit_factor"]))
        gate="PASS"
    profile=selected["profile"]
    print("[SELECTED]",profile,gate,flush=True)
    stress_base=bind_old(base,20200102,20201231,2020,2021,manifest)
    stress=run_case("stress_2020_"+profile,apply_profile(stress_base,profile),"old",profile)
    holdout=run_case("holdout_2021_2025_"+profile,apply_profile(base,profile),"new",profile)
    report={
        "method":"2016-2019 development; 2020 stress; 2021-2025 sealed holdout",
        "formal_engine_blob":EXPECTED_BLOB,
        "profiles":PROFILES,
        "development_results":results,
        "selection_gate":gate,
        "selected_profile":profile,
        "stress_2020":stress,
        "holdout_2021_2025":holdout,
        "status":"PASS" if gate=="PASS" else "RESEARCH_INCONCLUSIVE",
    }
    out=ROOT/"multi_regime_results"
    out.mkdir(exist_ok=True)
    (out/"multi_regime_search_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame([{k:v for k,v in r.items() if k not in {"benchmark","annual"}} for r in results]).to_csv(out/"development_matrix.csv",index=False)
    print(json.dumps({"status":report["status"],"selected":profile,"dev":selected,"stress_2020":stress,"holdout":holdout},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
