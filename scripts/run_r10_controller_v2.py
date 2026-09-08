#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FORMAL = ROOT / "scripts" / "r10_max_formal.py"
EXPECTED_BLOB = "cc5d3ee1f59f44914a74bd2c3b379e3b17f2f034"
OUT = ROOT / "controller_v2_results"
RUNS = ROOT / "controller_v2_runs"


def blob_sha(data: bytes) -> str:
    header = ("blob %d" % len(data)).encode("ascii") + bytes([0])
    return hashlib.sha1(header + data).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_case(case: Path) -> dict[str, str]:
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True)
    old = ROOT / "validation_input"
    hist = ROOT / "data" / "history" / "2020-2025"
    for y in range(2015, 2021):
        os.symlink((old / f"ohlcv_{y}.parquet").resolve(), case / f"ohlcv_{y}.parquet")
    for y in range(2021, 2026):
        os.symlink((hist / f"ohlcv_{y}.parquet").resolve(), case / f"ohlcv_{y}.parquet")

    inst_old = pd.read_parquet(old / "institutional_2015_2020.parquet")
    inst_new = pd.read_parquet(hist / "institutional_2020_2025.parquet")
    inst = pd.concat([inst_old, inst_new], ignore_index=True)
    inst["code"] = inst["code"].astype(str).str.zfill(4)
    if pd.api.types.is_numeric_dtype(inst["date"]):
        inst["date"] = pd.to_datetime(inst["date"].astype("Int64").astype(str), format="%Y%m%d")
    else:
        inst["date"] = pd.to_datetime(inst["date"])
    inst = inst.sort_values(["date", "code"]).drop_duplicates(["date", "code"], keep="last")
    inst.to_parquet(case / "institutional_2015_2025.parquet", index=False)

    action_files = [
        old / "official_corporate_actions_2015_2020.csv",
        ROOT / "data" / "reference" / "official_corporate_actions_2020_2025.csv",
    ]
    actions = [pd.read_csv(p, dtype=str) for p in action_files if p.exists()]
    if not actions:
        raise RuntimeError("official corporate-action inputs missing")
    ca = pd.concat(actions, ignore_index=True).drop_duplicates()
    ca.to_csv(case / "official_corporate_actions_2015_2025.csv", index=False)

    hashes = {"institutional_2015_2025.parquet": sha256(case / "institutional_2015_2025.parquet")}
    for y in range(2015, 2026):
        hashes[f"ohlcv_{y}.parquet"] = sha256(case / f"ohlcv_{y}.parquet")
    return hashes


def bind_window(src: str, hashes: dict[str, str], start: int, end: int) -> str:
    block = re.compile(
        r"EXPECTED_INPUT_HASHES = \{.*?\n\}\n\nfor filename, expected in EXPECTED_INPUT_HASHES.items\(\):.*?raise RuntimeError\(f'input SHA mismatch: \{filename\}: \{actual\} != \{expected\}'\)\n",
        re.S,
    )
    replacement = "EXPECTED_INPUT_HASHES = " + repr(hashes) + "\n" + (
        "for filename, expected in EXPECTED_INPUT_HASHES.items():\n"
        "    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()\n"
        "    if actual != expected:\n"
        "        raise RuntimeError(f'input SHA mismatch: {filename}: {actual} != {expected}')\n"
    )
    if not block.search(src):
        raise RuntimeError("formal hash block signature changed")
    src = block.sub(replacement, src, count=1)
    src = src.replace("for y in range(2020, 2026):", "for y in range(2015, 2026):", 1)
    src = src.replace("official_corporate_actions_2020_2025.csv", "official_corporate_actions_2015_2025.csv")
    src = src.replace("ohlcv_causal_2020_2025.csv.gz", f"ohlcv_causal_{start}_{end}.csv.gz")
    src = src.replace("institutional_2020_2025.parquet", "institutional_2015_2025.parquet")
    src = src.replace(
        "eval_dates = [d for d in all_dates if 20210104 <= int(d) <= 20251231]",
        f"eval_dates = [d for d in all_dates if {start} <= int(d) <= {end}]",
    )
    src = src.replace(
        "assert eval_dates[0] == 20210104 and eval_dates[-1] == 20251231",
        f"assert eval_dates[0] == {start} and eval_dates[-1] == {end}",
    )
    first_year, last_year = start // 10000, end // 10000
    src = src.replace("for year in range(2021, 2026):", f"for year in range({first_year}, {last_year + 1}):")
    src = src.replace(
        "'exact_evaluation_window': eval_dates[0] == 20210104 and eval_dates[-1] == 20251231",
        f"'exact_evaluation_window': eval_dates[0] == {start} and eval_dates[-1] == {end}",
    )
    return src


def controller_a(src: str) -> str:
    needle = "bm[['regime', 'r7_exposure', 'r7_slots']] = bm.apply(classify, axis=1)"
    replacement = "bm['regime']='No Classification'\nbm['r7_exposure']=0.95\nbm['r7_slots']=5"
    if needle not in src:
        raise RuntimeError("controller assignment signature changed")
    src = src.replace(needle, replacement, 1)
    free = "r7_free, r05_free = max(0, r7_slots_regime - n_r7), max(0, R05_MAX_SLOTS - n_r05)"
    return src.replace(free, "r7_free, r05_free = max(0, r7_slots_regime - n_r7), max(0, 2 - n_r05)", 1)


def controller_c(src: str, ma: int, short: int, long: int) -> str:
    src = src.replace("bm['mkt_ma120'] = bm['mkt'].rolling(120, min_periods=120).mean()", f"bm['mkt_ma120'] = bm['mkt'].rolling({ma}, min_periods={ma}).mean()", 1)
    src = src.replace("bm['mr20'] = bm['mkt'].pct_change(20)", f"bm['mr20'] = bm['mkt'].pct_change({short})", 1)
    src = src.replace("bm['mr60'] = bm['mkt'].pct_change(60)", f"bm['mr60'] = bm['mkt'].pct_change({long})", 1)
    pattern = re.compile(r"def classify\(row\):\n.*?return pd.Series\(\['Fallback/Bear', 0\.0, 0\]\)\n", re.S)
    replacement = """def classify(row):
    m, ma120, rshort, rlong = row['mkt'], row['mkt_ma120'], row['mr20'], row['mr60']
    if pd.isna(ma120) or pd.isna(rshort) or pd.isna(rlong):
        return pd.Series(['Unknown', 0.0, 0])
    if m > ma120 and rshort > 0 and rlong > 0:
        return pd.Series(['Risk-On', 0.95, 5])
    if m < ma120 and rlong < 0:
        return pd.Series(['Risk-Off', 0.0, 0])
    return pd.Series(['Transition', 0.50, 3])
"""
    if not pattern.search(src):
        raise RuntimeError("five-state classifier signature changed")
    src = pattern.sub(replacement, src, count=1)
    free = "r7_free, r05_free = max(0, r7_slots_regime - n_r7), max(0, R05_MAX_SLOTS - n_r05)"
    replacement_free = (
        "r05_controller_slots = {'Risk-On': 2, 'Transition': 1}.get(regime_now, 0)\n"
        "            r7_free, r05_free = max(0, r7_slots_regime - n_r7), max(0, r05_controller_slots - n_r05)"
    )
    regime_line = "r7_exposure = float(sub['r7_exposure'].iloc[0]) if len(sub) and np.isfinite(sub['r7_exposure'].iloc[0]) else 0.0"
    if regime_line not in src or free not in src:
        raise RuntimeError("daily controller signature changed")
    src = src.replace(regime_line, "regime_now = str(sub['regime'].iloc[0]) if len(sub) else 'Unknown'\n            " + regime_line, 1)
    return src.replace(free, replacement_free, 1)


def metrics(case: Path, label: str, arm: str, window: dict) -> dict:
    summary = json.loads((case / "r10max_formal_summary.json").read_text())
    audit = json.loads((case / "contract_audit.json").read_text())
    if not audit.get("all_pass"):
        raise RuntimeError(f"{label}: contract audit failed")
    trades = pd.read_csv(case / "r10max_formal_trades.csv")
    nav = pd.read_csv(case / "r10max_formal_nav.csv")
    ret = pd.to_numeric(trades.get("return"), errors="coerce")
    pnl_col = next((c for c in ["net_pnl", "pnl", "profit"] if c in trades.columns), None)
    top5_share = None
    if pnl_col:
        pnl = pd.to_numeric(trades[pnl_col], errors="coerce").dropna()
        positive = pnl[pnl > 0]
        top5_share = float(positive.nlargest(5).sum() / positive.sum()) if positive.sum() else None
    return {
        "label": label,
        "arm": arm,
        "window": window,
        **summary["strategy"],
        "benchmark_0050": summary["benchmark_0050"],
        "average_exposure": float(nav["exposure"].mean()),
        "maximum_exposure": float(nav["exposure"].max()),
        "worst_trade": float(ret.min()) if ret.notna().any() else None,
        "loss_le_12": int((ret <= -0.12).sum()),
        "loss_le_20": int((ret <= -0.20).sum()),
        "loss_le_25": int((ret <= -0.25).sum()),
        "top5_winner_gross_profit_share": top5_share,
        "annual": summary["annual"],
        "contract_all_pass": True,
    }


def run_case(base: str, label: str, arm: str, start: int, end: int, spec=(120, 20, 60)) -> dict:
    case = RUNS / label
    hashes = prepare_case(case)
    src = bind_window(base, hashes, start, end)
    if arm == "A":
        src = controller_a(src)
    elif arm == "C":
        src = controller_c(src, *spec)
    elif arm != "B":
        raise ValueError(arm)
    generated = case / "generated.py"
    generated.write_text(src, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(generated)], cwd=case, text=True, capture_output=True)
    (case / "execution.log").write_text(proc.stdout + "\nSTDERR\n" + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{label} failed; see {case / 'execution.log'}")
    return metrics(case, label, arm, {"start": start, "end": end, "ma": spec[0], "short": spec[1], "long": spec[2]})


def main() -> None:
    data = FORMAL.read_bytes()
    if blob_sha(data) != EXPECTED_BLOB:
        raise RuntimeError("formal engine blob changed")
    base = data.decode("utf-8")
    OUT.mkdir(exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    results = []
    slices = {
        "2016_2019": (20160104, 20191231),
        "2020": (20200102, 20201231),
        "2021_2022": (20210104, 20221230),
        "2023_2025": (20230103, 20251231),
        "2016_2025": (20160104, 20251231),
    }
    for arm in ("A", "B", "C"):
        for name, (start, end) in slices.items():
            label = f"arm_{arm}_{name}"
            print("[RUN]", label, flush=True)
            results.append(run_case(base, label, arm, start, end))

    expanding = []
    year_ends = {2016: 20161230, 2017: 20171229, 2018: 20181228, 2019: 20191231, 2020: 20201231,
                 2021: 20211230, 2022: 20221230, 2023: 20231229, 2024: 20241231, 2025: 20251231}
    for year, end in year_ends.items():
        label = f"expanding_C_2016_{year}"
        print("[RUN]", label, flush=True)
        expanding.append(run_case(base, label, "C", 20160104, end))

    sensitivity = []
    for ma in (100, 120, 140):
        for short, long in ((15, 45), (20, 60), (25, 75)):
            label = f"sensitivity_C_ma{ma}_r{short}_{long}"
            print("[RUN]", label, flush=True)
            sensitivity.append(run_case(base, label, "C", 20160104, 20251231, (ma, short, long)))

    primary = next(r for r in results if r["label"] == "arm_C_2016_2025")
    hard = primary["max_drawdown"] >= -0.25
    target = primary["max_drawdown"] >= -0.22 and primary["profit_factor"] >= 1.50
    stable = sum(r["max_drawdown"] >= -0.25 and r["profit_factor"] >= 1.0 for r in sensitivity) >= 7
    dependence = primary["top5_winner_gross_profit_share"]
    not_dependent = dependence is None or dependence <= 0.60
    passed = bool(hard and target and stable and not_dependent)
    report = {
        "status": "CANDIDATE_PASS" if passed else "CANNOT_PROVE",
        "research_only": True,
        "formal_engine_blob": EXPECTED_BLOB,
        "preregistered_contract": "research/R10_CONTROLLER_V2_PREREGISTERED_CONTRACT.md",
        "primary_parameters_fixed": {"ma": 120, "short": 20, "long": 60},
        "candidate_gate": {"max_dd_le_22": primary["max_drawdown"] >= -0.22,
                           "hard_reject_gt_25": hard, "pf_ge_1_50": primary["profit_factor"] >= 1.50,
                           "sensitivity_stable_cells_ge_7_of_9": stable,
                           "top5_winner_share_le_60pct_or_unavailable": not_dependent},
        "slice_results": results,
        "expanding_results": expanding,
        "sensitivity_results": sensitivity,
        "notes": ["Sensitivity is robustness evidence only; primary parameters remain MA120 and 20/60.",
                  "No result modifies formal R10 rules.",
                  "2021-2025 is reused historical validation, not pristine unseen OOS."],
    }
    (OUT / "controller_v2_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = [{k: v for k, v in r.items() if k not in {"annual", "benchmark_0050", "window"}} for r in results]
    pd.DataFrame(flat).to_csv(OUT / "slice_matrix.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k not in {"annual", "benchmark_0050", "window"}} for r in sensitivity]).to_csv(OUT / "sensitivity_matrix.csv", index=False)
    print(json.dumps({"status": report["status"], "gate": report["candidate_gate"], "primary": primary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
