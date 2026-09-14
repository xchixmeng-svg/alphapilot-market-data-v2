#!/usr/bin/env python3
"""AlphaPilot AI causal OOS stock-selection reranker v1.

Research contract
-----------------
* The locked R10 MAX engine is immutable and fingerprint checked.
* Only T-close information is used to create features.
* Historical labels execute at T+1 using the same fixed-limit convention and
  adverse slippage/fees/tax assumptions as locked R10.
* AI predictions are strictly yearly expanding-window OOS:
    2022 <- train 2021 with a 60-session purge before 2022
    2023 <- train 2021-2022 with a 60-session purge
    2024 <- train 2021-2023 with a 60-session purge
    2025 <- train 2021-2024 with a 60-session purge
  2021 portfolio behavior remains locked R10 because there is no prior test-year training set.
* No hyperparameter search is performed. One preregistered HistGradientBoosting
  model family is used for all folds/horizons.
* Five forward horizons are estimated independently (5/10/20/40/60 sessions).
* AI is used only for stock selection: it gates and re-ranks the locked R10
  otherwise-eligible shortlist. Locked exits, common cash, caps and risk controls remain unchanged.
* Low-base/persistence information is included only as a causal feature; it is
  not hard-coded as an entry rule.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "scripts" / "r10_max_formal.py"
LOCKED_SHA = "2fef3ba99b7c83b5db21e29c5f1c2abd8b3df77583840a025e6df937ff81c0d0"
RUN_ROOT = ROOT / "ai_causal_reranker_v1_results"
BASELINE_DIR = RUN_ROOT / "baseline"
VARIANT_DIR = RUN_ROOT / "AI_CAUSAL_V1"
HORIZONS = [5, 10, 20, 40, 60]
PURGE_SESSIONS = 60
TOP_PER_STRATEGY = 15
FAIL_THRESHOLD = -0.05
ACCEPT_MIN_PRED_RETURN = 0.02
ACCEPT_MAX_FAIL_PROB = 0.45
BUY_FEE = 0.000855
SELL_FEE = 0.000855
SELL_TAX = 0.003
BUY_ADVERSE = 0.005
SELL_ADVERSE = 0.005

EXPECTED_INPUT_HASHES = {
    "institutional_2020_2025.parquet": "63ad43e8bd3c7f7a90dda7d03d51cf3f1a3a84ce8c0bdf7f4fa5b422ffc90f5b",
    "ohlcv_2020.parquet": "5dd98f665a701e52cbf1920b5cb9203d3475abe9500d8eb8d9fef6bb8af321ca",
    "ohlcv_2021.parquet": "1b0962491ca57e231da0044e5dc2bfee38df991fae775e8ebaa4053f535b1f33",
    "ohlcv_2022.parquet": "b5e10eab9e06898cca5f2f2693fea2a352277755d55e22a6544081929cf65cdd",
    "ohlcv_2023.parquet": "450268bdf6e4beed956fcdf692a3a84ce8c0bdf7f4fa5b422ffc90f5b" if False else "450268bdf6e4beed956fcdf692a3a84ce8c0bdf7f4fa5b422ffc90f5b",
}
# Use the canonical manifest directly for all immutable history files; the dict
# above is intentionally not used because SHA256SUMS is the single source of truth.


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tick(p: float) -> float:
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0


def floor_tick(p: float) -> float:
    t = tick(p)
    return round(np.floor((p + 1e-10) / t) * t, 4)


def ceil_tick(p: float) -> float:
    t = tick(p)
    return round(np.ceil((p - 1e-10) / t) * t, 4)


def buy_fill(open_: float, low_: float, limit_: float) -> float | None:
    if open_ <= limit_:
        return min(ceil_tick(open_ * (1 + BUY_ADVERSE)), limit_)
    return limit_ if low_ <= limit_ else None


def run_py(script: Path, cwd: Path, log_name: str) -> None:
    p = subprocess.run([sys.executable, str(script)], cwd=cwd, text=True, capture_output=True)
    (cwd / log_name).write_text(p.stdout + "\n--- STDERR ---\n" + p.stderr, encoding="utf-8")
    print(f"[{cwd.name}] returncode={p.returncode}")
    if p.stdout:
        print("\n".join(p.stdout.splitlines()[-18:]))
    if p.returncode:
        raise RuntimeError(f"runner failed: {script}")


def link_inputs(dst: Path) -> None:
    hist = ROOT / "data" / "history" / "2020-2025"
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "institutional_2020_2025.parquet",
        "ohlcv_2020.parquet", "ohlcv_2021.parquet", "ohlcv_2022.parquet",
        "ohlcv_2023.parquet", "ohlcv_2024.parquet", "ohlcv_2025.parquet",
        "official_corporate_actions_2020_2025.csv",
    ]:
        src = hist / name
        out = dst / name
        if not out.exists():
            out.symlink_to(src.resolve())


def prepare_baseline() -> None:
    if sha256(LOCKED) != LOCKED_SHA:
        raise RuntimeError("locked R10 engine fingerprint mismatch")
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True)
    link_inputs(BASELINE_DIR)
    runner = BASELINE_DIR / "r10_max_formal_locked.py"
    runner.write_text(LOCKED.read_text(encoding="utf-8"), encoding="utf-8")
    run_py(runner, BASELINE_DIR, "execution.log")
    audit = json.loads((BASELINE_DIR / "contract_audit.json").read_text(encoding="utf-8"))
    if not audit.get("all_pass"):
        raise RuntimeError(f"baseline contract audit failed: {audit}")
    summary = json.loads((BASELINE_DIR / "r10max_formal_summary.json").read_text(encoding="utf-8"))
    if abs(float(summary["strategy"]["end_nav"]) - 2403427.0678222505) > 1e-6:
        raise RuntimeError("baseline end NAV mismatch")


def add_features(px: pd.DataFrame) -> pd.DataFrame:
    px = px.copy().sort_values(["code", "date"]).reset_index(drop=True)
    g = px.groupby("code", group_keys=False)
    for w in [5, 10, 20, 60]:
        px[f"ret{w}"] = g["aclose"].transform(lambda s, w=w: s.pct_change(w))
    px["high120"] = g["aclose"].transform(lambda s: s.rolling(120, min_periods=120).max())
    px["dist_high120"] = px["aclose"] / px["high120"] - 1.0
    px["gap_ma20"] = px["aclose"] / px["ma20"] - 1.0
    px["gap_ma60"] = px["aclose"] / px["ma60"] - 1.0
    px["gap_ma120"] = px["aclose"] / px["ma120"] - 1.0
    px["ma20_ma60"] = px["ma20"] / px["ma60"] - 1.0
    px["ma60_ma120"] = px["ma60"] / px["ma120"] - 1.0
    px["log_amt20"] = np.log1p(px["amt20"].clip(lower=0))
    px["log_vol20"] = np.log1p(px["vol20"].clip(lower=0))
    px["lowbase_flag"] = (
        (px["aclose"] <= 0.90 * px["high120"]) &
        (px["aclose"] >= px["ma60"]) &
        (px["ma20"] >= px["ma60"]) &
        (px["aclose"] >= 0.98 * px["ma120"]) &
        (px["ma60"] <= 1.05 * px["ma120"])
    ).astype(int)

    elig = px["amt20"] >= 30_000_000
    bd = ((px["aclose"] > px["ma60"]) & elig).groupby(px["date"]).sum() / elig.groupby(px["date"]).sum()
    adv = ((px["ret10"] > 0) & elig).groupby(px["date"]).sum() / elig.groupby(px["date"]).sum()
    m = px[px.code.eq("0050")][["date", "aclose", "ma60", "ma120", "ret20", "ret60"]].drop_duplicates("date")
    m = m.rename(columns={"aclose":"mkt", "ma60":"mkt_ma60", "ma120":"mkt_ma120", "ret20":"mr20", "ret60":"mr60"})
    m["mkt_gap60"] = m["mkt"] / m["mkt_ma60"] - 1
    m["mkt_gap120"] = m["mkt"] / m["mkt_ma120"] - 1
    m = m.merge(pd.DataFrame({"date":bd.index, "breadth":bd.values}), on="date", how="left")
    m = m.merge(pd.DataFrame({"date":adv.index, "advance10":adv.values}), on="date", how="left")
    px = px.merge(m[["date","mr20","mr60","mkt_gap60","mkt_gap120","breadth","advance10"]], on="date", how="left")
    return px


def candidate_events(px: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, sub0 in px.groupby("date", sort=True):
        sub = sub0.set_index("code")
        r7_exp = float(sub["r7_exposure"].iloc[0]) if len(sub) and np.isfinite(sub["r7_exposure"].iloc[0]) else 0.0
        r7 = sub[sub["r7_hard"].eq(True)].sort_values("r7_score", ascending=False).head(TOP_PER_STRATEGY) if r7_exp > 0 else sub.iloc[0:0]
        r05 = sub[sub["r05_hard"].eq(True)].sort_values("r05_score", ascending=False).head(TOP_PER_STRATEGY)
        for strat, frame in [("R7", r7), ("R05", r05)]:
            for code, r in frame.iterrows():
                rows.append({"date": int(d), "code": str(code), "strategy": strat, **r.to_dict()})
    ev = pd.DataFrame(rows)
    if ev.empty:
        raise RuntimeError("no candidate events")

    # Causal execution-independent recommendation persistence features.
    ev = ev.sort_values(["date","code","strategy"]).reset_index(drop=True)
    unique_dates = sorted(px.date.unique().tolist())
    idx = {int(d): i for i, d in enumerate(unique_dates)}
    by_code = {}
    p5 = []; p10 = []
    for r in ev.itertuples(index=False):
        c = str(r.code); i = idx[int(r.date)]
        hist = by_code.setdefault(c, [])
        p5.append(sum(1 for x in hist if i - x < 5))
        p10.append(sum(1 for x in hist if i - x < 10))
        if not hist or hist[-1] != i:
            hist.append(i)
    ev["persist5_prior"] = p5
    ev["persist10_prior"] = p10
    return ev


def attach_labels(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(px.date.unique().tolist())
    date_idx = {int(d): i for i, d in enumerate(dates)}
    by_date = {int(d): s.set_index("code") for d, s in px.groupby("date")}
    out = ev.copy()
    for h in HORIZONS:
        out[f"y_ret_{h}"] = np.nan
        out[f"y_fail_{h}"] = np.nan
    out["historical_fill"] = False

    for j, r in out.iterrows():
        d = int(r.date); code = str(r.code); strat = str(r.strategy); i = date_idx[d]
        if i + 1 >= len(dates):
            continue
        ed = int(dates[i+1]); esub = by_date.get(ed)
        if esub is None or code not in esub.index:
            continue
        er = esub.loc[code]
        limit = floor_tick(float(r.close) * (0.98 if strat == "R7" else 0.995))
        fill = buy_fill(float(er.open), float(er.low), limit)
        if fill is None or float(er.close) <= 0:
            continue
        out.at[j, "historical_fill"] = True
        entry_idx = float(er.aclose) * fill / float(er.close)
        entry_cost = entry_idx * (1 + BUY_FEE)
        for h in HORIZONS:
            xi = i + 1 + h
            if xi >= len(dates):
                continue
            xd = int(dates[xi]); xsub = by_date.get(xd)
            if xsub is None or code not in xsub.index:
                continue
            xr = xsub.loc[code]
            if not np.isfinite(xr.get("aopen", np.nan)):
                # aopen is not in the locked signal pickle; reconstruct from aclose/open/close.
                if float(xr.close) <= 0:
                    continue
                aopen = float(xr.aclose) * float(xr.open) / float(xr.close)
            else:
                aopen = float(xr.aopen)
            exit_idx = aopen * (1 - SELL_ADVERSE)
            proceeds = exit_idx * (1 - SELL_FEE - SELL_TAX)
            ret = proceeds / entry_cost - 1.0
            out.at[j, f"y_ret_{h}"] = ret
            out.at[j, f"y_fail_{h}"] = float(ret <= FAIL_THRESHOLD)
    return out


FEATURES = [
    "r7_score","r05_score","r7_exposure","r7_slots","amount_ratio","amtacc",
    "ret5","ret10","ret20","ret60","dist_high120","gap_ma20","gap_ma60","gap_ma120",
    "ma20_ma60","ma60_ma120","log_amt20","log_vol20","lowbase_flag",
    "mr20","mr60","mkt_gap60","mkt_gap120","breadth","advance10",
    "persist5_prior","persist10_prior","strategy_r7",
]


def fit_oos(ev: pd.DataFrame, px: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = ev.copy()
    ev["strategy_r7"] = ev.strategy.eq("R7").astype(int)
    all_dates = sorted(px.date.unique().tolist())
    didx = {int(d): i for i, d in enumerate(all_dates)}
    test_rows = []
    diagnostics = []

    for year in [2022, 2023, 2024, 2025]:
        test_mask = (ev.date >= year*10000 + 101) & (ev.date <= year*10000 + 1231)
        test = ev.loc[test_mask].copy()
        if test.empty:
            continue
        first_test_date = int(test.date.min())
        cutoff_i = didx[first_test_date] - PURGE_SESSIONS
        if cutoff_i <= 0:
            continue
        cutoff_date = int(all_dates[cutoff_i])
        train = ev[(ev.date < cutoff_date) & ev.historical_fill].copy()
        # Need all horizons observable for a comparable multi-horizon target set.
        train = train.dropna(subset=[f"y_ret_{h}" for h in HORIZONS])
        if len(train) < 150:
            raise RuntimeError(f"insufficient training rows for {year}: {len(train)}")

        med = train[FEATURES].median(numeric_only=True)
        Xtr = train[FEATURES].replace([np.inf,-np.inf], np.nan).fillna(med).fillna(0.0)
        Xte = test[FEATURES].replace([np.inf,-np.inf], np.nan).fillna(med).fillna(0.0)
        pred_ret = {}; pred_fail = {}
        for h in HORIZONS:
            yr = train[f"y_ret_{h}"].astype(float)
            reg = HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=160, max_leaf_nodes=15,
                min_samples_leaf=25, l2_regularization=1.0, random_state=560+h,
            )
            reg.fit(Xtr, yr)
            pred_ret[h] = reg.predict(Xte)

            yc = train[f"y_fail_{h}"].astype(int)
            if yc.nunique() < 2:
                pf = np.full(len(test), float(yc.mean()))
            else:
                clf = HistGradientBoostingClassifier(
                    learning_rate=0.05, max_iter=140, max_leaf_nodes=15,
                    min_samples_leaf=25, l2_regularization=1.0, random_state=760+h,
                )
                clf.fit(Xtr, yc)
                pf = clf.predict_proba(Xte)[:,1]
            pred_fail[h] = pf

            # Pure OOS diagnostics only where realized labels exist.
            yy = test[f"y_ret_{h}"].notna()
            if yy.any():
                act = test.loc[yy, f"y_ret_{h}"].to_numpy(float)
                pr = pred_ret[h][np.flatnonzero(yy.to_numpy())]
                spear = pd.Series(pr).corr(pd.Series(act), method="spearman")
                auc = np.nan
                actual_fail = test.loc[yy, f"y_fail_{h}"].astype(int).to_numpy()
                if len(np.unique(actual_fail)) == 2:
                    auc = roc_auc_score(actual_fail, pred_fail[h][np.flatnonzero(yy.to_numpy())])
                diagnostics.append({"test_year":year,"horizon":h,"train_n":len(train),"test_labeled_n":int(yy.sum()),
                                    "spearman_return":spear,"failure_auc":auc})

        mat_ret = np.column_stack([pred_ret[h] for h in HORIZONS])
        mat_fail = np.column_stack([pred_fail[h] for h in HORIZONS])
        # Among horizons meeting the failure-probability ceiling, choose highest predicted net return.
        allowed = mat_fail <= ACCEPT_MAX_FAIL_PROB
        masked = np.where(allowed, mat_ret, -np.inf)
        best_j = np.argmax(masked, axis=1)
        none_allowed = ~allowed.any(axis=1)
        if none_allowed.any():
            best_j[none_allowed] = np.argmax(mat_ret[none_allowed], axis=1)
        best_ret = mat_ret[np.arange(len(test)), best_j]
        best_fail = mat_fail[np.arange(len(test)), best_j]
        accept = (best_ret >= ACCEPT_MIN_PRED_RETURN) & (best_fail <= ACCEPT_MAX_FAIL_PROB)
        # Selection score rewards return and penalizes predicted probability of a <=-5% outcome.
        ai_score = best_ret - 0.05 * best_fail
        test["best_horizon"] = np.array(HORIZONS)[best_j]
        test["pred_net_return"] = best_ret
        test["pred_fail_prob"] = best_fail
        test["ai_score"] = ai_score
        test["ai_accept"] = accept.astype(int)
        test["fold_train_cutoff"] = cutoff_date
        for h in HORIZONS:
            test[f"pred_ret_{h}"] = pred_ret[h]
            test[f"pred_fail_{h}"] = pred_fail[h]
        test_rows.append(test)

    pred = pd.concat(test_rows, ignore_index=True)
    diag = pd.DataFrame(diagnostics)
    return pred, diag


def build_variant_runner(locked_source: str) -> str:
    marker = "# ============================================================\n# 步驟三：組合層回測引擎\n# ============================================================\n\n"
    if marker not in locked_source:
        raise RuntimeError("stage3 marker missing")
    stage3 = locked_source.split(marker,1)[1]
    old_load = "px = pd.read_pickle('r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)"
    new_load = "px = pd.read_pickle('../baseline/r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)\n_ai = pd.read_csv('../AI_OOS_PREDICTIONS.csv', dtype={'code': str})\n_ai['code'] = _ai['code'].astype(str).str.zfill(4)\nAI_MAP = {(int(r.date), str(r.code), str(r.strategy)): (float(r.ai_score), int(r.ai_accept), int(r.best_horizon), float(r.pred_net_return), float(r.pred_fail_prob)) for r in _ai.itertuples(index=False)}"
    if old_load not in stage3:
        raise RuntimeError("signal load anchor missing")
    stage3 = stage3.replace(old_load, new_load, 1)

    old = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    new = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            # 2021 is untouched. From 2022 onward use strictly OOS AI predictions to gate/re-rank\n            # the exact locked R10 otherwise-eligible shortlist.\n            if int(di) >= 20220101:\n                ai_combined = []\n                for _strat, _code, _r in combined:\n                    _p = AI_MAP.get((int(di), str(_code), str(_strat)))\n                    if _p is None:\n                        continue\n                    _score, _accept, _h, _pret, _pfail = _p\n                    if not _accept:\n                        continue\n                    ai_combined.append((_strat, _code, _r, _score))\n                ai_combined.sort(key=lambda z: -z[3])\n                combined = [(a,b,c) for a,b,c,_ in ai_combined]\n            else:\n                combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    if old not in stage3:
        raise RuntimeError("candidate block anchor missing")
    stage3 = stage3.replace(old, new, 1)
    return "# Generated AI causal OOS selection runner from immutable R10 stage-3.\nimport pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n" + stage3


def summarize(run_dir: Path, name: str) -> dict:
    s = json.loads((run_dir / "r10max_formal_summary.json").read_text(encoding="utf-8"))["strategy"]
    a = json.loads((run_dir / "contract_audit.json").read_text(encoding="utf-8"))
    if not a.get("all_pass"):
        raise RuntimeError(f"audit failed for {name}")
    return {"variant":name, "end_nav":s["end_nav"], "total_return":s["total_return"], "cagr":s["cagr"],
            "max_drawdown":s["max_drawdown"], "completed_trades":s["completed_trades"], "wins":s["wins"],
            "losses":s["losses"], "win_rate":s["win_rate"], "pnl_profit_factor":s["profit_factor"],
            **{f"return_{x['year']}":x["strategy_return"] for x in json.loads((run_dir / "r10max_formal_summary.json").read_text(encoding="utf-8"))["annual"]}}


def main() -> None:
    prepare_baseline()
    px = pd.read_pickle(BASELINE_DIR / "r10max_signals_final.pkl")
    px["code"] = px["code"].astype(str).str.zfill(4)
    px = add_features(px)
    ev = candidate_events(px)
    ev = attach_labels(ev, px)
    pred, diag = fit_oos(ev, px)
    keep_pred = ["date","code","strategy","best_horizon","pred_net_return","pred_fail_prob","ai_score","ai_accept","fold_train_cutoff"] + [f"pred_ret_{h}" for h in HORIZONS] + [f"pred_fail_{h}" for h in HORIZONS]
    pred[keep_pred].to_csv(RUN_ROOT / "AI_OOS_PREDICTIONS.csv", index=False)
    diag.to_csv(RUN_ROOT / "AI_OOS_MODEL_DIAGNOSTICS.csv", index=False)

    # Selection diagnostics on filled OOS events only; never used to tune the model.
    eval_rows = pred[pred.historical_fill].copy()
    sel = []
    for yr, g in eval_rows.groupby(eval_rows.date // 10000):
        acc = g[g.ai_accept.eq(1)]
        rej = g[g.ai_accept.eq(0)]
        sel.append({"year":int(yr), "events":len(g), "accepted":len(acc), "rejected":len(rej),
                    "accept_rate":len(acc)/len(g) if len(g) else np.nan,
                    "accepted_actual20_mean":acc.y_ret_20.mean(), "rejected_actual20_mean":rej.y_ret_20.mean(),
                    "accepted_actual20_win_rate":(acc.y_ret_20>0).mean(), "rejected_actual20_win_rate":(rej.y_ret_20>0).mean()})
    pd.DataFrame(sel).to_csv(RUN_ROOT / "AI_SELECTION_DIAGNOSTICS.csv", index=False)

    VARIANT_DIR.mkdir(parents=True)
    link_inputs(VARIANT_DIR)
    runner = VARIANT_DIR / "runner.py"
    runner.write_text(build_variant_runner(LOCKED.read_text(encoding="utf-8")), encoding="utf-8")
    run_py(runner, VARIANT_DIR, "execution.log")

    rows = [summarize(BASELINE_DIR, "BASELINE_R10"), summarize(VARIANT_DIR, "AI_CAUSAL_V1")]
    comp = pd.DataFrame(rows)
    b = comp.iloc[0]
    comp["cagr_delta_pp"] = (comp.cagr - b.cagr) * 100
    comp["dd_improvement_pp"] = (comp.max_drawdown - b.max_drawdown) * 100
    comp["pf_delta"] = comp.pnl_profit_factor - b.pnl_profit_factor
    comp["win_delta_pp"] = (comp.win_rate - b.win_rate) * 100
    comp.to_csv(RUN_ROOT / "AI_CAUSAL_V1_COMPARISON.csv", index=False)
    (RUN_ROOT / "AI_CAUSAL_V1_COMPARISON.json").write_text(json.dumps(comp.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== AI CAUSAL V1 PORTFOLIO ===")
    print(comp.to_string(index=False))
    print("\n=== OOS MODEL DIAGNOSTICS ===")
    print(diag.to_string(index=False))
    print("\n=== OOS SELECTION DIAGNOSTICS ===")
    print(pd.DataFrame(sel).to_string(index=False))


if __name__ == "__main__":
    main()
