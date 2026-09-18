#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import backtest_ai_market_reasoning_v6_1_corp_safe as v61

OUT = ROOT / "v6_2_prototype"
OUT.mkdir(exist_ok=True)

BARRIER_TASKS = [
    (0.05, -0.03, 5), (0.05, -0.03, 10), (0.05, -0.03, 20),
    (0.10, -0.05, 10), (0.10, -0.05, 20), (0.10, -0.05, 40), (0.10, -0.05, 60),
    (0.20, -0.10, 20), (0.20, -0.10, 40), (0.20, -0.10, 60), (0.20, -0.10, 120),
]
FAMILIES = [
    "price_volume_structure",
    "institutional_flow",
    "revenue_earnings",
    "eps_revisions",
    "valuation",
    "industry_cycle",
    "industry_pricing_supply_demand",
    "peer_sector_condition",
    "macro_rates_fx_commodities",
    "corporate_events",
    "news_disclosed_catalysts",
    "seasonality_historical_context",
]

def stable_id(date: int, code: str, family: str, key: str) -> str:
    raw = f"{date}|{code}|{family}|{key}".encode()
    return "E_" + hashlib.sha256(raw).hexdigest()[:16]

def finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def family(status, date, code, name, fields, missing_reason=None):
    ev = []
    for k, v in fields.items():
        if v is not None:
            ev.append({"evidence_id": stable_id(date, code, name, k), "key": k, "value": v})
    out = {"family": name, "status": status, "evidence": ev}
    if missing_reason:
        out["missing_reason"] = missing_reason
    return out

def build_bundle(r) -> dict:
    date = int(r.date); code = str(r.code).zfill(4)
    p = family("AVAILABLE", date, code, FAMILIES[0], {
        "close": finite(r.close), "volume": finite(r.volume),
        "ret1": finite(r.ret1), "ret20_obs": finite(r.ret20_obs), "ret60_obs": finite(r.ret60_obs),
        "vol20_obs": finite(r.vol20_obs), "vol60_obs": finite(r.vol60_obs),
        "close_pos20": finite(r.close_pos20), "close_pos60": finite(r.close_pos60),
        "range_width20": finite(r.range_width20), "range_width60": finite(r.range_width60),
    })
    inst = family("UNAVAILABLE", date, code, FAMILIES[1], {},
        "Frozen Stage0F feature registry contains no stock-level institutional-flow fields.")
    rev_fields = {
        "revenue_thousand": finite(r.rev_revenue_thousand),
        "revenue_mom_pct": finite(r.rev_mom_pct),
        "revenue_yoy_pct": finite(r.rev_yoy_pct),
        "revenue_stale_days": finite(r.rev_stale_days),
    }
    rev = family("PARTIAL", date, code, FAMILIES[2], rev_fields,
        "Revenue is present; earnings/margins/EPS are not present in frozen Stage0F.")
    eps = family("UNAVAILABLE", date, code, FAMILIES[3], {},
        "No PIT analyst EPS-revision/consensus series in frozen Stage0F.")
    val = family("AVAILABLE", date, code, FAMILIES[4], {
        "pe": finite(r.valuation_pe), "pb": finite(r.valuation_pb),
        "dividend_yield_pct": finite(r.valuation_dividend_yield_pct),
        "stale_days": finite(r.valuation_stale_days),
    })
    ind_cols = [c for c in r.index if str(c).startswith("industry_ret1_")]
    ind_values = {c: finite(r[c]) for c in ind_cols if finite(r[c]) is not None}
    ind = family("PARTIAL", date, code, FAMILIES[5], ind_values,
        "Industry-index context exists, but historical company-to-industry mapping is intentionally not in frozen 0F.")
    pricing = family("UNAVAILABLE", date, code, FAMILIES[6], {},
        "No structured industry pricing/inventory/supply-demand series in frozen Stage0F.")
    peer = family("PARTIAL", date, code, FAMILIES[7], {
        "sector_adv": finite(r.sector_adv) if "sector_adv" in r.index else None,
        "sector_median1": finite(r.sector_median1) if "sector_median1" in r.index else None,
        "sector_dispersion1": finite(r.sector_dispersion1) if "sector_dispersion1" in r.index else None,
    }, "Global industry breadth/context only; no PIT stock peer map.")
    macro = family("PARTIAL", date, code, FAMILIES[8], {
        "fed_funds": finite(r.macro_fed_funds), "us2y": finite(r.macro_us2y),
        "us10y": finite(r.macro_us10y), "us10y2y": finite(r.macro_us10y2y),
    }, "Rates are present; FX/commodity series are not in frozen Stage0F.")
    evt = family("PARTIAL", date, code, FAMILIES[9], {
        "event_cum_count": finite(r.event_cum_count), "event_last_age_hours": finite(r.event_last_age_hours),
    }, "Event timing/count exists; event semantic text/type is not present in 0F model panel.")
    news = family("UNAVAILABLE", date, code, FAMILIES[10], {},
        "No PIT news/disclosure text bundle in frozen Stage0F model panel.")
    seas = family("PARTIAL", date, code, FAMILIES[11], {
        "season_ret20_1y": finite(r.season_ret20_1y), "season_ret60_1y": finite(r.season_ret60_1y),
        "season_ret20_2y": finite(r.season_ret20_2y),
    }, "Price-history seasonality only; no semantic seasonal business context.")

    return {
        "schema": "V6.2-EVIDENCE-BUNDLE-v0",
        "decision_date": date,
        "code": code,
        "price_segment_id": int(r.price_segment_id),
        "families": [p, inst, rev, eps, val, ind, pricing, peer, macro, evt, news, seas],
        "numerical_outputs": {},
    }

def barrier_label(g: pd.DataFrame, pos: int, up: float, down: float, horizon: int, max_outcome_date: int = 20241231) -> tuple[str, int|None]:
    entry = float(g.iloc[pos]["close"])
    seg = int(g.iloc[pos]["price_segment_id"])
    for step in range(1, horizon + 1):
        j = pos + step
        if j >= len(g):
            return "CENSORED_RESET_OR_END", None
        rr = g.iloc[j]
        if int(rr["date"]) > max_outcome_date:
            return "CENSORED_RESET_OR_END", None
        if int(rr["price_segment_id"]) != seg:
            return "CENSORED_RESET_OR_END", None
        hi = float(rr["high"]) / entry - 1.0
        lo = float(rr["low"]) / entry - 1.0
        hit_up = hi >= up
        hit_dn = lo <= down
        if hit_up and hit_dn:
            return "AMBIGUOUS_SAME_BAR", step
        if hit_up:
            return "UP_FIRST", step
        if hit_dn:
            return "DOWN_FIRST", step
    return "NEITHER", None

def main():
    panel, base_feats, _ = v61.load_frozen_panel(require_lock=False)
    ds, feats = v61.add_safe_observable_transforms(panel, base_feats)

    # Barrier audit on deterministic sample: 10 late-June-2024 decision dates, capped per date; outcomes hard-censored after 2024-12-31.
    dates = sorted(ds.loc[(ds["date"] >= 20240101) & (ds["date"] <= 20241231) & ds["universe_ok"], "date"].unique())
    safe_dates = [int(d) for d in dates if int(d) <= 20240630]
    sample_dates = safe_dates[-10:]
    sample = ds[ds["date"].isin(sample_dates) & ds["universe_ok"]].copy()
    sample = sample.sort_values(["date","code"]).groupby("date", group_keys=False).head(100)

    groups = {str(c): g.sort_values("date").reset_index(drop=True) for c, g in ds.groupby("code", sort=False)}
    positions = {}
    for code,g in groups.items():
        positions[code] = {int(d): i for i,d in enumerate(g["date"].to_numpy())}

    counts = []
    for up,down,h in BARRIER_TASKS:
        c = {"up":up,"down":down,"horizon":h,"UP_FIRST":0,"DOWN_FIRST":0,"NEITHER":0,
             "AMBIGUOUS_SAME_BAR":0,"CENSORED_RESET_OR_END":0}
        for r in sample.itertuples(index=False):
            code=str(r.code).zfill(4); pos=positions[code][int(r.date)]
            state,_=barrier_label(groups[code],pos,up,down,h)
            c[state]+=1
        counts.append(c)
    pd.DataFrame(counts).to_csv(OUT/"V6_2_BARRIER_LABEL_DISTRIBUTION.csv",index=False)

    # Latest-date full-market evidence bundles; deterministic first 200 also saved as JSONL sample.
    # Development seal: never inspect 2025+ bundles during V6.2 prototype.
    dev_mask = ds["universe_ok"] & (ds["date"] <= 20241231)
    latest = int(ds.loc[dev_mask,"date"].max())
    day = ds[(ds["date"]==latest)&ds["universe_ok"]].sort_values("code").copy()
    bundles = [build_bundle(r) for _,r in day.iterrows()]

    coverage = []
    for fam in FAMILIES:
        states=[next(x for x in b["families"] if x["family"]==fam)["status"] for b in bundles]
        coverage.append({
            "family":fam,"stocks":len(states),
            "available":states.count("AVAILABLE"),
            "partial":states.count("PARTIAL"),
            "unavailable":states.count("UNAVAILABLE"),
        })
    pd.DataFrame(coverage).to_csv(OUT/"V6_2_EVIDENCE_FAMILY_COVERAGE.csv",index=False)

    payloads=[json.dumps(b,ensure_ascii=False,separators=(",",":")) for b in bundles]
    sizes=np.array([len(p.encode("utf-8")) for p in payloads],dtype=float)
    # Rough only: not provider tokenizer benchmark.
    token_est=sizes/4.0
    size_meta={
        "decision_date":latest,"eligible_stocks":len(bundles),
        "bytes_p50":float(np.quantile(sizes,.50)) if len(sizes) else None,
        "bytes_p95":float(np.quantile(sizes,.95)) if len(sizes) else None,
        "rough_tokens_p50_chars_over_4":float(np.quantile(token_est,.50)) if len(sizes) else None,
        "rough_tokens_p95_chars_over_4":float(np.quantile(token_est,.95)) if len(sizes) else None,
        "warning":"Token estimate is heuristic only; real provider tokenizer/latency/cost benchmark still required."
    }
    (OUT/"V6_2_BUNDLE_SIZE_AUDIT.json").write_text(json.dumps(size_meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (OUT/"V6_2_SAMPLE_EVIDENCE_BUNDLES.jsonl").open("w",encoding="utf-8") as f:
        for p in payloads[:200]:
            f.write(p+"\n")

    summary={
      "status":"PROTOTYPE_PASS",
      "not_locked":True,
      "latest_bundle_date":latest,
      "eligible_stocks_latest_date":len(bundles),
      "sample_barrier_rows":int(len(sample)),
      "barrier_tasks":len(BARRIER_TASKS),
      "evidence_family_coverage":coverage,
      "bundle_size":size_meta,
      "development_data_cutoff":20241231,
      "2025_opened_for_model_evaluation":False,
      "notes":[
        "No LLM was called in this workflow.",
        "This run tests deterministic retrieval, barrier-label semantics, missing-data honesty, and bundle scale only.",
        "Real LLM batch invariance/latency/cost benchmark requires a provider/model endpoint and must happen before LOCK."
      ]
    }
    (OUT/"V6_2_PROTOTYPE_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("[V6.2 PROTOTYPE]",json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
