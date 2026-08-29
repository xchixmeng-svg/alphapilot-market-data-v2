#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from clean_event_loop import PortfolioEngine
from clean_research_run import (
    BarStore,
    EVAL_END,
    EVAL_START,
    FEATURE_COLS,
    INITIAL,
    DD_GATE,
    candidate_pool,
    metrics,
)
from clean_strategy_research import ResearchStrategy, build_feature_store

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".clean_cache"
OUT = ROOT / "clean_results" / "lanes"
OUT.mkdir(parents=True, exist_ok=True)
FAMILIES = ["REGIME_MOM","PERSIST_MOM","MOM_LOWVOL","SMID_LIQ_RS","MOM_RS_FLOW","BREAK_FLOW","PULLBACK_RS"]


def load_inputs():
    raw = pd.concat(
        [pd.read_parquet(CACHE / f"ohlcv_{y}.parquet") for y in range(2021, 2026)],
        ignore_index=True,
    )
    raw = raw[raw.date.between(EVAL_START, EVAL_END)].copy()
    raw = raw[~raw.code.astype(str).str.startswith("00")].copy()
    features = pd.read_parquet(CACHE / "features_2020_2025.parquet", columns=FEATURE_COLS)
    features = features[features.date.between(20200101, EVAL_END)].copy()
    dates = [str(int(x)) for x in sorted(raw.date.unique())]
    if len(dates) < 1200:
        raise RuntimeError(f"too few evaluation dates {len(dates)}")
    return raw, dates, build_feature_store(features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--pool-size", type=int, default=420)
    args = ap.parse_args()
    if not 0 <= args.shard < args.shards:
        raise SystemExit("invalid shard")

    raw, dates, store = load_inputs()
    bars = BarStore(raw)
    family_pool = [p for p in candidate_pool(args.pool_size) if p.family == args.family]
    lane_pool = [p for i, p in enumerate(family_pool) if i % args.shards == args.shard]
    if not lane_pool:
        raise RuntimeError(f"empty lane {args.family} shard {args.shard}")

    rows = []
    for i, p in enumerate(lane_pool, 1):
        result = PortfolioEngine(INITIAL).run(dates, bars, ResearchStrategy(store, p))
        m = metrics(result)
        qualified = (
            m["max_dd"] > DD_GATE
            and m["positive_years"] >= 3
            and m["completed_trades"] >= 20
        )
        rows.append({"family": args.family,"shard": args.shard,"lane_trial": i,"params": asdict(p),"qualified": bool(qualified),**m})
        print(json.dumps({"family": args.family,"shard": args.shard,"trial": i,"cagr": m["cagr"],"dd": m["max_dd"],"trades": m["completed_trades"],"qualified": qualified}),flush=True)

    rows.sort(key=lambda x: (x["qualified"], x["cagr"], -abs(x["max_dd"])), reverse=True)
    out = {"family": args.family,"shard": args.shard,"shards": args.shards,"tested": len(rows),"qualified": sum(bool(x["qualified"]) for x in rows),"best": rows[0] if rows else None,"top": rows[:25]}
    path = OUT / f"{args.family.lower()}_s{args.shard}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "tested": len(rows), "qualified": out["qualified"]}))


if __name__ == "__main__":
    main()
