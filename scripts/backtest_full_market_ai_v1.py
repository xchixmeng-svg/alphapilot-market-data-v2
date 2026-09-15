#!/usr/bin/env python3
"""AlphaPilot Full-Market AI V1.

AI sees the full executable Taiwan equity universe; R10 hard candidate flags are
NOT eligibility gates. R10 remains immutable as benchmark/data pipeline and its
execution/risk mechanics are reused by a generated portfolio runner.

See research/FULL_MARKET_AI_V1_PREREGISTRATION.md for the locked contract.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v2 = load("ai_v2", ROOT / "scripts" / "backtest_ai_causal_reranker_v2.py")
base = v2.mod

RUN_ROOT = ROOT / "full_market_ai_v1_results"
BASELINE_DIR = RUN_ROOT / "baseline"
VARIANT_DIR = RUN_ROOT / "FULL_MARKET_AI_V1"

v2.RUN_ROOT = RUN_ROOT
v2.BASELINE_DIR = BASELINE_DIR
v2.VARIANTS = {"FULL_MARKET_AI_V1": VARIANT_DIR}
base.RUN_ROOT = RUN_ROOT
base.BASELINE_DIR = BASELINE_DIR
base.VARIANT_DIR = VARIANT_DIR

PURGE = 60
FAIL_THRESHOLD = -0.05
BUY_FEE, SELL_FEE, SELL_TAX = 0.000855, 0.000855, 0.003
BUY_ADVERSE, SELL_ADVERSE = 0.005, 0.005
CONTEXT_FEATURES = ["mr20", "mr60", "mkt_gap60", "mkt_gap120", "breadth", "advance10"]
MODEL_FEATURES = [
    "r7_score", "r05_score", "amount_ratio", "amtacc",
    "ret5", "ret10", "ret20", "ret60", "dist_high120",
    "gap_ma20", "gap_ma60", "gap_ma120", "ma20_ma60", "ma60_ma120",
    "log_amt20", "log_vol20", "lowbase_flag",
    "mr20", "mr60", "mkt_gap60", "mkt_gap120", "breadth", "advance10",
]


def tick_vec(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return np.select(
        [a < 10, a < 50, a < 100, a < 500, a < 1000],
        [0.01, 0.05, 0.1, 0.5, 1.0],
        default=5.0,
    )


def floor_tick_vec(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    t = tick_vec(a)
    return np.floor((a + 1e-10) / t) * t


def ceil_tick_vec(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    t = tick_vec(a)
    return np.ceil((a - 1e-10) / t) * t


def build_full_market_events(px: pd.DataFrame) -> pd.DataFrame:
    """Build execution-feasible universe and vectorized T+1 / 20-session labels."""
    x = px.copy()
    valid = (
        x["code"].astype(str).str.fullmatch(r"[1-9]\d{3}")
        & ~x["name"].astype(str).str.contains("KY", case=False, na=False)
        & (x["amt20"] >= 30_000_000)
        & x["ma120"].notna()
    )
    keep = ["date", "code", "name", "open", "low", "close", "aclose", "amt20", "vol20",
            "r7_hard", "r05_hard"] + MODEL_FEATURES
    keep = list(dict.fromkeys(keep))
    ev = x.loc[valid, keep].copy()
    ev["code"] = ev["code"].astype(str).str.zfill(4)
    ev["outside_r10"] = ~(ev["r7_hard"].fillna(False) | ev["r05_hard"].fillna(False))

    dates = np.array(sorted(int(d) for d in x.date.unique()), dtype=np.int64)
    didx = {int(d): i for i, d in enumerate(dates)}
    next_map = {int(dates[i]): int(dates[i + 1]) for i in range(len(dates) - 1)}
    exit_map = {int(dates[i]): int(dates[i + 21]) for i in range(len(dates) - 21)}
    ev["entry_date"] = ev["date"].map(next_map)
    ev["exit_date20"] = ev["date"].map(exit_map)

    q = x[["date", "code", "open", "low", "close", "aclose"]].copy()
    q["code"] = q["code"].astype(str).str.zfill(4)
    en = q.rename(columns={"date": "entry_date", "open": "entry_open", "low": "entry_low",
                           "close": "entry_close", "aclose": "entry_aclose"})
    ex = q[["date", "code", "open", "close", "aclose"]].rename(
        columns={"date": "exit_date20", "open": "exit_open", "close": "exit_close", "aclose": "exit_aclose"})
    ev = ev.merge(en, on=["entry_date", "code"], how="left")
    ev = ev.merge(ex, on=["exit_date20", "code"], how="left")

    limit = floor_tick_vec(ev["close"].to_numpy(float) * 0.995)
    eopen = ev["entry_open"].to_numpy(float)
    elow = ev["entry_low"].to_numpy(float)
    adverse = ceil_tick_vec(eopen * (1 + BUY_ADVERSE))
    fill = np.where(eopen <= limit, np.minimum(adverse, limit), np.where(elow <= limit, limit, np.nan))
    fill[~np.isfinite(eopen) | ~np.isfinite(elow)] = np.nan
    ev["historical_fill"] = np.isfinite(fill)
    ev["entry_fill"] = fill

    entry_close = ev["entry_close"].to_numpy(float)
    entry_aclose = ev["entry_aclose"].to_numpy(float)
    entry_idx = np.where((entry_close > 0) & np.isfinite(fill), entry_aclose * fill / entry_close, np.nan)
    entry_cost = entry_idx * (1 + BUY_FEE)

    exit_close = ev["exit_close"].to_numpy(float)
    exit_open = ev["exit_open"].to_numpy(float)
    exit_aclose = ev["exit_aclose"].to_numpy(float)
    exit_aopen = np.where(exit_close > 0, exit_aclose * exit_open / exit_close, np.nan)
    proceeds = exit_aopen * (1 - SELL_ADVERSE) * (1 - SELL_FEE - SELL_TAX)
    yret = proceeds / entry_cost - 1.0
    yret[~np.isfinite(entry_cost) | ~np.isfinite(proceeds)] = np.nan
    ev["y_ret20"] = yret
    ev["y_fail20"] = np.where(np.isfinite(yret), (yret <= FAIL_THRESHOLD).astype(float), np.nan)
    ev["market_i"] = ev["date"].map(didx).astype(int)
    return ev


def fit_models(train: pd.DataFrame):
    med = train[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    X = train[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    yr = train["y_ret20"].astype(float)
    reg = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=120, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=2.0, random_state=420,
    )
    reg.fit(X, yr)
    yc = train["y_fail20"].astype(int)
    clf = None
    constant_fail = None
    if yc.nunique() < 2:
        constant_fail = float(yc.mean())
    else:
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=110, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=2.0, random_state=421,
        )
        clf.fit(X, yc)
    return reg, clf, constant_fail, med


def predict_pair(bundle, frame: pd.DataFrame):
    reg, clf, const, med = bundle
    X = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    pr = reg.predict(X)
    pf = np.full(len(frame), const, dtype=float) if clf is None else clf.predict_proba(X)[:, 1]
    return pr, pf


def expanding_oos(ev: pd.DataFrame, px: pd.DataFrame):
    dates = np.array(sorted(int(d) for d in px.date.unique()), dtype=np.int64)
    didx = {int(d): i for i, d in enumerate(dates)}
    ctx = px[["date"] + CONTEXT_FEATURES].drop_duplicates("date").sort_values("date").copy()
    outputs, diagnostics, daily_rows = [], [], []

    for year in [2021, 2022, 2023, 2024, 2025]:
        test = ev[(ev.date >= year * 10000 + 101) & (ev.date <= year * 10000 + 1231)].copy().reset_index(drop=True)
        if test.empty:
            raise RuntimeError(f"no full-market test rows for {year}")
        first_date = int(test.date.min())
        ci = didx[first_date] - PURGE
        if ci <= 0:
            raise RuntimeError(f"insufficient purge history for {year}")
        cutoff = int(dates[ci])

        tr = ev[(ev.date < cutoff) & ev.historical_fill & ev.y_ret20.notna()].copy()
        tr = tr[(tr.market_i % 5) == 0].copy()  # deterministic compute sample, all executable names on sampled dates
        if len(tr) < 2000:
            raise RuntimeError(f"insufficient training rows for {year}: {len(tr)}")

        prior_ctx = ctx[ctx.date < cutoff].copy()
        cmed = prior_ctx[CONTEXT_FEATURES].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        Ctr = prior_ctx[CONTEXT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(cmed).fillna(0.0)
        scaler = StandardScaler().fit(Ctr)
        km = KMeans(n_clusters=3, n_init=20, random_state=42).fit(scaler.transform(Ctr))

        needed_dates = pd.concat([tr[["date"]], test[["date"]]]).drop_duplicates()
        need_ctx = needed_dates.merge(ctx, on="date", how="left")
        Cneed = need_ctx[CONTEXT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(cmed).fillna(0.0)
        need_ctx["context_cluster"] = km.predict(scaler.transform(Cneed))
        cmap = dict(zip(need_ctx.date.astype(int), need_ctx.context_cluster.astype(int)))
        tr["context_cluster"] = tr.date.map(cmap).astype(int)
        test["context_cluster"] = test.date.map(cmap).astype(int)

        global_bundle = fit_models(tr)
        cluster_bundle = {}
        for c in range(3):
            tc = tr[tr.context_cluster.eq(c)]
            cluster_bundle[c] = fit_models(tc) if len(tc) >= 5000 else global_bundle

        pred_ret = np.empty(len(test), dtype=float)
        pred_fail = np.empty(len(test), dtype=float)
        for c in range(3):
            mask = test.context_cluster.eq(c).to_numpy()
            if not mask.any():
                continue
            pr, pf = predict_pair(cluster_bundle[c], test.loc[mask])
            pred_ret[mask] = pr
            pred_fail[mask] = pf
        test["pred_return20"] = pred_ret
        test["pred_fail_prob"] = pred_fail
        test["ret_rank"] = test.groupby("date")["pred_return20"].rank(method="average", pct=True)
        test["safety_rank"] = test.groupby("date")["pred_fail_prob"].rank(method="average", pct=True, ascending=False)
        test["ai_score"] = 0.5 * test["ret_rank"] + 0.5 * test["safety_rank"]
        test["ai_accept"] = ((test.pred_return20 > 0) & (test.pred_fail_prob < 0.50)).astype(int)
        test["fold_train_cutoff"] = cutoff

        labeled = test[test.y_ret20.notna() & test.historical_fill].copy()
        spear = labeled.pred_return20.corr(labeled.y_ret20, method="spearman") if len(labeled) else np.nan
        auc = np.nan
        if len(labeled) and labeled.y_fail20.nunique() == 2:
            auc = roc_auc_score(labeled.y_fail20.astype(int), labeled.pred_fail_prob)
        q90 = labeled.ai_score.quantile(0.90) if len(labeled) else np.nan
        top = labeled[labeled.ai_score >= q90] if np.isfinite(q90) else labeled.iloc[0:0]
        diagnostics.append({
            "test_year": year, "train_cutoff": cutoff, "train_rows_sampled": len(tr),
            "test_universe_rows": len(test), "spearman_return20": spear, "failure_auc": auc,
            "top_decile_actual20_mean": top.y_ret20.mean(),
            "top_decile_actual20_win": (top.y_ret20 > 0).mean() if len(top) else np.nan,
        })

        accepted = test[test.ai_accept.eq(1)].copy().sort_values(["date", "ai_score"], ascending=[True, False])
        top50 = accepted.groupby("date", group_keys=False).head(50).copy()
        outputs.append(top50)

        for d, gd in test.groupby("date"):
            ga = gd[gd.ai_accept.eq(1)]
            gt = top50[top50.date.eq(d)]
            daily_rows.append({
                "date": int(d), "year": year, "context_cluster": int(gd.context_cluster.iloc[0]),
                "full_executable_universe_n": int(len(gd)), "accepted_n": int(len(ga)),
                "stored_top50_n": int(len(gt)), "top50_outside_r10_n": int(gt.outside_r10.sum()) if len(gt) else 0,
                "top_score": float(gt.ai_score.max()) if len(gt) else np.nan,
            })
        print(f"FULL_MARKET fold={year} cutoff={cutoff} train={len(tr)} test={len(test)} top50={len(top50)}", flush=True)

    pred = pd.concat(outputs, ignore_index=True)
    return pred, pd.DataFrame(diagnostics), pd.DataFrame(daily_rows)


def build_runner(locked_source: str) -> str:
    marker = "# ============================================================\n# 步驟三：組合層回測引擎\n# ============================================================\n\n"
    if marker not in locked_source:
        raise RuntimeError("stage3 marker missing")
    stage3 = locked_source.split(marker, 1)[1]
    old_load = "px = pd.read_pickle('r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)"
    new_load = (
        "px = pd.read_pickle('../baseline/r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)\n"
        "_ai = pd.read_csv('../FULL_MARKET_AI_OOS_TOP50.csv', dtype={'code': str})\n"
        "_ai['code'] = _ai['code'].astype(str).str.zfill(4)\n"
        "AI_BY_DATE = {int(d): g.set_index('code').sort_values('ai_score', ascending=False) for d, g in _ai.groupby('date')}"
    )
    if old_load not in stage3:
        raise RuntimeError("signal load anchor missing")
    stage3 = stage3.replace(old_load, new_load, 1)

    exit_anchor = "        if p['strategy'] == 'R7':\n"
    if exit_anchor not in stage3:
        raise RuntimeError("exit anchor missing")
    stage3 = stage3.replace(
        exit_anchor,
        "        if p['strategy'] == 'AI':\n"
        "            if p['hold_days'] >= 20:\n"
        "                reason = 'AI_MAXHOLD'\n"
        "        elif p['strategy'] == 'R7':\n",
        1,
    )

    start = "            mult = dd_multiplier(dd)\n"
    end = "                used.add(code); slots_free -= 1\n"
    si = stage3.find(start)
    ei0 = stage3.find(end, si)
    if si < 0 or ei0 < 0:
        raise RuntimeError("entry block anchors missing")
    ei = ei0 + len(end)
    new_entry = """            mult = dd_multiplier(dd)
            slot_diag.append({'date': di, 'positions': n_open, 'slots_free': slots_free,
                              'r7_slots_free': 0, 'r05_slots_free': 0})
            held = set(positions)
            _cand = AI_BY_DATE.get(int(di))
            combined = []
            if _cand is not None:
                for code, _ar in _cand.iterrows():
                    code = str(code)
                    if int(_ar.get('ai_accept', 0)) != 1 or code in held or code not in sub.index:
                        continue
                    combined.append(('AI', code, sub.loc[code]))
            created_mv, reserved_cost, used = 0.0, 0.0, set()
            for strat, code, r in combined:
                if slots_free <= 0: break
                if code in used: continue
                limit = floor_tick(float(r['close']) * 0.995)
                base = 0.20 * nav * mult
                target = min(base, nav * SINGLE_CAP, nav * TOTAL_CAP - mv - created_mv,
                             max(0.0, cash - reserved_cost - 1000.0) / (1 + BUY_FEE))
                if target <= 0 or limit <= 0: continue
                shares = int(target / limit) if limit * 1000 > target else int(target / limit / 1000) * 1000
                vol20 = r.get('vol20', np.nan)
                if np.isfinite(vol20) and vol20 > 0:
                    cap = int(vol20 * 0.02 / 1000) * 1000
                    if cap < 1000: cap = int(vol20 * 0.02)
                    shares = min(shares, cap) if cap > 0 else 0
                if shares <= 0: continue
                gross = shares * limit
                post_total, post_code = (mv + created_mv + gross) / nav, gross / nav
                if post_total > TOTAL_CAP + 1e-12 or post_code > SINGLE_CAP + 1e-12: continue
                o = submit('BUY', di, exdate, code, strat, shares, 'ENTRY', limit,
                           nav, post_total, post_code)
                pending_buys.setdefault(exdate, []).append(o)
                created_mv += gross; reserved_cost += gross * (1 + BUY_FEE)
                used.add(code); slots_free -= 1
"""
    stage3 = stage3[:si] + new_entry + stage3[ei:]
    return "# Generated from immutable R10 stage-3; Full-Market AI selection replaces R10 entry eligibility.\nimport pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n" + stage3


def main() -> None:
    base.prepare_baseline()
    px = pd.read_pickle(BASELINE_DIR / "r10max_signals_final.pkl")
    px["code"] = px["code"].astype(str).str.zfill(4)
    px = base.add_features(px)
    events = build_full_market_events(px)
    pred, diag, daily = expanding_oos(events, px)

    keep = ["date", "code", "name", "context_cluster", "pred_return20", "pred_fail_prob",
            "ret_rank", "safety_rank", "ai_score", "ai_accept", "outside_r10", "fold_train_cutoff"]
    pred[keep].to_csv(RUN_ROOT / "FULL_MARKET_AI_OOS_TOP50.csv", index=False)
    diag.to_csv(RUN_ROOT / "FULL_MARKET_AI_MODEL_DIAGNOSTICS.csv", index=False)
    daily.to_csv(RUN_ROOT / "FULL_MARKET_AI_DAILY_AUDIT.csv", index=False)

    coverage = {
        "stored_top50_rows": int(len(pred)),
        "stored_top50_outside_r10_rows": int(pred.outside_r10.sum()),
        "stored_top50_outside_r10_share": float(pred.outside_r10.mean()) if len(pred) else np.nan,
        "mean_daily_executable_universe": float(daily.full_executable_universe_n.mean()),
        "mean_daily_accepted": float(daily.accepted_n.mean()),
    }
    (RUN_ROOT / "FULL_MARKET_AI_COVERAGE.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    locked = base.LOCKED.read_text(encoding="utf-8")
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    v2.link_inputs(VARIANT_DIR)
    runner = VARIANT_DIR / "runner.py"
    runner.write_text(build_runner(locked), encoding="utf-8")
    v2.run_py_verbose(runner, VARIANT_DIR, "execution.log")

    comp = pd.DataFrame([
        base.summarize(BASELINE_DIR, "BASELINE_R10"),
        base.summarize(VARIANT_DIR, "FULL_MARKET_AI_V1"),
    ])
    b, r = comp.iloc[0], comp.iloc[1]
    comp["cagr_delta_pp"] = (comp.cagr - b.cagr) * 100
    comp["dd_improvement_pp"] = (comp.max_drawdown - b.max_drawdown) * 100
    comp["pf_delta"] = comp.pnl_profit_factor - b.pnl_profit_factor
    comp["win_delta_pp"] = (comp.win_rate - b.win_rate) * 100
    comp.to_csv(RUN_ROOT / "FULL_MARKET_AI_V1_COMPARISON.csv", index=False)

    gate = {
        "cagr_better": bool(r.cagr > b.cagr),
        "pf_not_worse": bool(r.pnl_profit_factor >= b.pnl_profit_factor),
        "dd_not_worse_over_1pp": bool(r.max_drawdown >= b.max_drawdown - 0.01),
    }
    gate["pass"] = all(gate.values())
    (RUN_ROOT / "FULL_MARKET_AI_V1_SUCCESS_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print("=== FULL-MARKET COVERAGE ===")
    print(json.dumps(coverage, indent=2))
    print("\n=== FULL-MARKET OOS DIAGNOSTICS ===")
    print(diag.to_string(index=False))
    print("\n=== FULL-MARKET PORTFOLIO ===")
    print(comp.to_string(index=False))
    print("\n=== SUCCESS GATE ===")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
