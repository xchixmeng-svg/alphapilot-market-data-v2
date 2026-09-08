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

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FORMAL = ROOT / "scripts" / "r10_max_formal.py"
EXPECTED_FORMAL_BLOB = "cc5d3ee1f59f44914a74bd2c3b379e3b17f2f034"
INITIAL = 1_300_000.0

VARIANTS = {
    "always_on": {"kind": "always", "ma": 120, "short": 20, "long": 60},
    "locked_five": {"kind": "locked", "ma": 120, "short": 20, "long": 60},
    "simple_3": {"kind": "simple", "ma": 120, "short": 20, "long": 60},
    "sens_ma100": {"kind": "simple", "ma": 100, "short": 20, "long": 60},
    "sens_ma140": {"kind": "simple", "ma": 140, "short": 20, "long": 60},
    "sens_ret15_50": {"kind": "simple", "ma": 120, "short": 15, "long": 50},
    "sens_ret25_70": {"kind": "simple", "ma": 120, "short": 25, "long": 70},
}

SLICES = {
    "2016_2019": (20160104, 20191231),
    "stress_2020": (20200102, 20201231),
    "2021_2022": (20210104, 20221230),
    "2023_2025": (20230103, 20251231),
}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}".encode("ascii") + bytes([0])
    return hashlib.sha1(header + data).hexdigest()


def normalized_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y%m%d").astype(int)
    return pd.to_datetime(s.astype(str).str.replace("-", "", regex=False)).dt.strftime("%Y%m%d").astype(int)


def prepare_combined_inputs() -> tuple[Path, dict[str, str]]:
    out = ROOT / "combined_input"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()

    for year in range(2015, 2026):
        if year <= 2020:
            src = ROOT / "validation_input" / f"ohlcv_{year}.parquet"
        else:
            src = ROOT / "data" / "history" / "2020-2025" / f"ohlcv_{year}.parquet"
        if not src.exists():
            raise FileNotFoundError(src)
        os.symlink(src.resolve(), out / src.name)

    old_inst = pd.read_parquet(ROOT / "validation_input" / "institutional_2015_2020.parquet")
    new_inst = pd.read_parquet(ROOT / "data" / "history" / "2020-2025" / "institutional_2020_2025.parquet")
    for frame in (old_inst, new_inst):
        frame["date"] = normalized_date(frame["date"])
        frame["code"] = frame["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    inst = pd.concat([old_inst, new_inst], ignore_index=True)
    inst = inst.sort_values(["date", "code"]).drop_duplicates(["date", "code"], keep="last")
    inst_path = out / "institutional_2020_2025.parquet"
    inst.to_parquet(inst_path, index=False)

    old_ca = pd.read_csv(ROOT / "validation_input" / "official_corporate_actions_2015_2020.csv", dtype={"code": str})
    new_ca = pd.read_csv(ROOT / "data" / "reference" / "official_corporate_actions_2020_2025.csv", dtype={"code": str})
    ca = pd.concat([old_ca, new_ca], ignore_index=True)
    ca["date"] = pd.to_numeric(ca["date"], errors="raise").astype(int)
    ca["code"] = ca["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    ca = ca.sort_values(["date", "code", "source"]).drop_duplicates(["date", "code"], keep="last")
    ca_path = out / "official_corporate_actions_2020_2025.csv"
    ca.to_csv(ca_path, index=False)

    hashes = {p.name: file_sha(p) for p in out.iterdir()}
    manifest = {
        "status": "PASS",
        "evaluation": "2016-2025 continuous common pool",
        "warmup": 2015,
        "ohlcv_years": list(range(2015, 2026)),
        "institutional_rows": len(inst),
        "corporate_action_rows": len(ca),
        "hashes": hashes,
    }
    (out / "combined_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, hashes


def bind_continuous(src: str, hashes: dict[str, str]) -> str:
    block = re.compile(
        r"EXPECTED_INPUT_HASHES = \{.*?\n\}\n\nfor filename, expected in EXPECTED_INPUT_HASHES.items\(\):.*?raise RuntimeError\(f'input SHA mismatch: \{filename\}: \{actual\} != \{expected\}'\)\n",
        re.S,
    )
    expected = {k: v for k, v in hashes.items() if k.endswith(".parquet") or k.endswith(".csv")}
    replacement = "EXPECTED_INPUT_HASHES = " + repr(expected) + "\n\n" + (
        "for filename, expected in EXPECTED_INPUT_HASHES.items():\n"
        "    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()\n"
        "    if actual != expected:\n"
        "        raise RuntimeError(f'input SHA mismatch: {filename}: {actual} != {expected}')\n"
    )
    if not block.search(src):
        raise RuntimeError("formal hash block signature changed")
    src = block.sub(replacement, src, count=1)
    replacements = [
        ("for y in range(2020, 2026):", "for y in range(2015, 2026):"),
        ("ohlcv_causal_2020_2025.csv.gz", "ohlcv_causal_2015_2025.csv.gz"),
        ("eval_dates = [d for d in all_dates if 20210104 <= int(d) <= 20251231]", "eval_dates = [d for d in all_dates if 20160104 <= int(d) <= 20251231]"),
        ("assert eval_dates[0] == 20210104 and eval_dates[-1] == 20251231", "assert eval_dates[0] == 20160104 and eval_dates[-1] == 20251231"),
        ("for year in range(2021, 2026):", "for year in range(2016, 2026):"),
        ("'exact_evaluation_window': eval_dates[0] == 20210104 and eval_dates[-1] == 20251231", "'exact_evaluation_window': eval_dates[0] == 20160104 and eval_dates[-1] == 20251231"),
    ]
    for old, new in replacements:
        if old not in src:
            raise RuntimeError(f"formal signature changed: {old}")
        src = src.replace(old, new, 1)
    # The formal engine references the causal cache in its save, log and reload paths.
    # Replace every remaining occurrence so the widened 2015-2025 cache name stays consistent.
    src = src.replace("ohlcv_causal_2020_2025.csv.gz", "ohlcv_causal_2015_2025.csv.gz")
    return src


def replace_classifier(src: str, cfg: dict) -> str:
    if cfg["kind"] == "locked":
        return src

    ma = int(cfg["ma"])
    short = int(cfg["short"])
    long = int(cfg["long"])
    src = src.replace(
        "bm['mkt_ma120'] = bm['mkt'].rolling(120, min_periods=120).mean()",
        f"bm['mkt_ma120'] = bm['mkt'].rolling({ma}, min_periods={ma}).mean()",
        1,
    )
    src = src.replace("bm['mr20'] = bm['mkt'].pct_change(20)", f"bm['mr20'] = bm['mkt'].pct_change({short})", 1)
    src = src.replace("bm['mr60'] = bm['mkt'].pct_change(60)", f"bm['mr60'] = bm['mkt'].pct_change({long})", 1)

    block = re.compile(r"def classify\(row\):.*?bm\[\['regime', 'r7_exposure', 'r7_slots'\]\] = bm\.apply\(classify, axis=1\)", re.S)
    if cfg["kind"] == "always":
        classifier = """def classify(row):
    if pd.isna(row['mkt_ma120']):
        return pd.Series(['Unknown', 0.0, 0])
    return pd.Series(['Always-On', 1.0, 5])

bm[['regime', 'r7_exposure', 'r7_slots']] = bm.apply(classify, axis=1)"""
        r05 = {"Always-On": 1, "Unknown": 0}
    else:
        classifier = """def classify(row):
    m, ma120, mr20, mr60 = row['mkt'], row['mkt_ma120'], row['mr20'], row['mr60']
    if pd.isna(ma120) or pd.isna(mr20) or pd.isna(mr60):
        return pd.Series(['Unknown', 0.0, 0])
    if m > ma120 and mr20 > 0 and mr60 > 0:
        return pd.Series(['Risk-On', 1.0, 5])
    if m < ma120 and mr60 < 0:
        return pd.Series(['Risk-Off', 0.0, 0])
    return pd.Series(['Transition', 0.40, 2])

bm[['regime', 'r7_exposure', 'r7_slots']] = bm.apply(classify, axis=1)"""
        r05 = {"Risk-On": 1, "Transition": 1, "Risk-Off": 0, "Unknown": 0}
    if not block.search(src):
        raise RuntimeError("classifier signature changed")
    src = block.sub(classifier, src, count=1)

    old = """            r7_exposure = float(sub['r7_exposure'].iloc[0]) if len(sub) and np.isfinite(sub['r7_exposure'].iloc[0]) else 0.0
            r7_slots_regime = int(sub['r7_slots'].iloc[0]) if len(sub) and np.isfinite(sub['r7_slots'].iloc[0]) else 0
"""
    new = old + f"            regime_now = str(sub['regime'].iloc[0]) if len(sub) else 'Unknown'\n            r05_slots_regime = {repr(r05)}.get(regime_now, 0)\n"
    if old not in src:
        raise RuntimeError("slot assignment signature changed")
    src = src.replace(old, new, 1)
    old_free = "r7_free, r05_free = max(0, r7_slots_regime - n_r7), max(0, R05_MAX_SLOTS - n_r05)"
    new_free = "r7_free, r05_free = max(0, r7_slots_regime - n_r7), max(0, r05_slots_regime - n_r05)"
    if old_free not in src:
        raise RuntimeError("free-slot signature changed")
    return src.replace(old_free, new_free, 1)


def link_case_inputs(case: Path, combined: Path) -> None:
    case.mkdir(parents=True, exist_ok=True)
    for p in combined.iterdir():
        if p.suffix in {".parquet", ".csv"}:
            os.symlink(p.resolve(), case / p.name)


def profit_factor(t: pd.DataFrame) -> float | None:
    if t.empty or "pnl" not in t:
        return None
    win = float(t.loc[t.pnl > 0, "pnl"].sum())
    loss = float(-t.loc[t.pnl < 0, "pnl"].sum())
    return win / loss if loss > 0 else None


def slice_metrics(nav: pd.DataFrame, trades: pd.DataFrame, start: int, end: int) -> dict:
    n = nav[(nav.date >= start) & (nav.date <= end)].copy()
    if n.empty:
        raise RuntimeError(f"empty slice {start}-{end}")
    before = nav[nav.date < start]
    start_nav = float(before.iloc[-1].nav) if len(before) else INITIAL
    end_nav = float(n.iloc[-1].nav)
    curve = pd.concat([pd.Series([start_nav]), n.nav.reset_index(drop=True)], ignore_index=True)
    dd = float((curve / curve.cummax() - 1).min())
    days = max(1, (pd.to_datetime(str(int(n.iloc[-1].date))) - pd.to_datetime(str(int(n.iloc[0].date)))).days)
    ret = end_nav / start_nav - 1
    tr = trades[(trades.exit_date >= start) & (trades.exit_date <= end)].copy() if len(trades) else trades
    rr = pd.to_numeric(tr.get("return"), errors="coerce") if len(tr) else pd.Series(dtype=float)
    return {
        "start_nav": start_nav,
        "end_nav": end_nav,
        "total_return": ret,
        "cagr": (1 + ret) ** (365.25 / days) - 1 if 1 + ret > 0 else -1.0,
        "max_drawdown": dd,
        "completed_trades": int(len(tr)),
        "win_rate": float((tr.pnl > 0).mean()) if len(tr) else None,
        "profit_factor": profit_factor(tr),
        "worst_trade": float(rr.min()) if len(rr) else None,
        "loss_le_12": int((rr <= -0.12).sum()),
        "loss_le_20": int((rr <= -0.20).sum()),
    }


def run_variant(name: str, src: str, combined: Path) -> dict:
    case = ROOT / "regime_v2_runs" / name
    if case.exists():
        shutil.rmtree(case)
    link_case_inputs(case, combined)
    generated = case / "generated.py"
    generated.write_text(src, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(generated)], cwd=case, text=True, capture_output=True)
    (case / "execution.log").write_text(proc.stdout + "\nSTDERR\n" + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{name} failed; see {case / 'execution.log'}")
    audit = json.loads((case / "contract_audit.json").read_text())
    if not audit.get("all_pass"):
        raise RuntimeError(f"{name} contract failed: {audit}")
    summary = json.loads((case / "r10max_formal_summary.json").read_text())
    nav = pd.read_csv(case / "r10max_formal_nav.csv")
    trades = pd.read_csv(case / "r10max_formal_trades.csv")
    ret = pd.to_numeric(trades.get("return"), errors="coerce")
    winners = trades.loc[trades.pnl > 0, "pnl"].sort_values(ascending=False)
    top5_share = float(winners.head(5).sum() / winners.sum()) if len(winners) and winners.sum() > 0 else None
    result = {
        "variant": name,
        **summary["strategy"],
        "benchmark_0050": summary["benchmark_0050"],
        "average_exposure": float(nav.exposure.mean()),
        "median_exposure": float(nav.exposure.median()),
        "maximum_exposure": float(nav.exposure.max()),
        "worst_trade": float(ret.min()),
        "loss_le_12": int((ret <= -0.12).sum()),
        "loss_le_20": int((ret <= -0.20).sum()),
        "top5_winner_share_of_gross_profit": top5_share,
        "annual": summary["annual"],
        "slices": {k: slice_metrics(nav, trades, a, b) for k, (a, b) in SLICES.items()},
    }
    return result


def main() -> None:
    data = FORMAL.read_bytes()
    if git_blob_sha(data) != EXPECTED_FORMAL_BLOB:
        raise RuntimeError("formal engine blob changed")
    combined, hashes = prepare_combined_inputs()
    base = bind_continuous(data.decode("utf-8"), hashes)
    results = []
    for name, cfg in VARIANTS.items():
        print("[RUN]", name, flush=True)
        results.append(run_variant(name, replace_classifier(base, cfg), combined))

    by_name = {r["variant"]: r for r in results}
    candidate = by_name["simple_3"]
    sensitivity = [by_name[n] for n in VARIANTS if n.startswith("sens_")]
    sensitivity_pf = [r["profit_factor"] for r in sensitivity if r["profit_factor"] is not None]
    checks = {
        "candidate_maxdd_le_22": candidate["max_drawdown"] >= -0.22,
        "candidate_pf_ge_150": candidate["profit_factor"] is not None and candidate["profit_factor"] >= 1.50,
        "all_sensitivity_maxdd_le_25": all(r["max_drawdown"] >= -0.25 for r in sensitivity),
        "median_sensitivity_pf_ge_135": bool(sensitivity_pf) and float(np.median(sensitivity_pf)) >= 1.35,
    }
    report = {
        "status": "CANDIDATE_PASS" if all(checks.values()) else "RESEARCH_FAIL",
        "formal_engine_blob": EXPECTED_FORMAL_BLOB,
        "preregistered_candidate": "simple_3",
        "selection_policy": "Sensitivity variants are diagnostic and cannot replace simple_3.",
        "checks": checks,
        "variants": VARIANTS,
        "results": results,
    }
    out = ROOT / "regime_v2_results"
    out.mkdir(exist_ok=True)
    (out / "regime_controller_v2_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = []
    for r in results:
        flat.append({k: v for k, v in r.items() if k not in {"benchmark_0050", "annual", "slices"}})
    pd.DataFrame(flat).to_csv(out / "variant_matrix.csv", index=False)
    slice_rows = []
    for r in results:
        for label, metrics in r["slices"].items():
            slice_rows.append({"variant": r["variant"], "slice": label, **metrics})
    pd.DataFrame(slice_rows).to_csv(out / "time_slice_matrix.csv", index=False)
    print(json.dumps({"status": report["status"], "checks": checks, "simple_3": candidate}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
