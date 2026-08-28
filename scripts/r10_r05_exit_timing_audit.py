#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-run audit of formal R0.5 exit dates against the documented T-close state machine.

Reference trades are comparison-only. They never feed the clean trading engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import r10_true_validation as true

bt = true.bt
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "stress_results" / "latest" / "true_validation2021_2025"
REF = ROOT / "tests" / "fixtures" / "golden_2021_2025"


def load_parts(prefix: str) -> pd.DataFrame:
    ps = sorted(REF.glob(f"{prefix}_part*.csv"))
    if not ps:
        raise RuntimeError(f"missing {prefix} reference parts")
    return pd.concat([pd.read_csv(p, dtype={"code": str}) for p in ps], ignore_index=True)


def row_at(idx, di: int, code: str):
    try:
        r = idx.loc[(int(di), str(code).zfill(4))]
        return r.iloc[-1] if isinstance(r, pd.DataFrame) else r
    except KeyError:
        return None


def main() -> None:
    orders = load_parts("orders")
    exits = load_parts("exits")
    for q in (orders, exits):
        q["code"] = q.code.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    for c in ("decision_date", "execute_date"):
        orders[c] = pd.to_numeric(orders[c], errors="raise").astype(int)
    for c in ("entry_date", "sell_decision_date", "exit_date"):
        exits[c] = pd.to_numeric(exits[c], errors="raise").astype(int)

    r05_exits = exits[exits.strategy.astype(str).eq("R05")].copy().sort_values(["entry_date", "code"])
    r05_buys = orders[
        orders.strategy.astype(str).eq("R05") & orders.status.astype(str).str.upper().eq("FILLED")
    ].copy()
    buy_map = {
        (str(r.code), int(r.execute_date)): r
        for r in r05_buys.itertuples(index=False)
    }

    cfg = bt.SCENARIOS["validation2021_2025"]
    raw = bt.load_scenario_ohlcv(cfg)
    feat, _, _ = bt.build_features(raw)
    idx = feat.set_index(["date", "code"]).sort_index()
    dates = sorted(int(x) for x in feat.date.unique())
    next_date = {dates[i]: dates[i+1] for i in range(len(dates)-1)}

    rows = []
    for g in r05_exits.itertuples(index=False):
        code = str(g.code)
        entry_date = int(g.entry_date)
        formal_decision = int(g.sell_decision_date)
        formal_exit = int(g.exit_date)
        b = buy_map.get((code, entry_date))
        if b is None:
            rows.append({
                "code": code, "entry_date": entry_date,
                "formal_decision_date": formal_decision, "formal_exit_date": formal_exit,
                "status": "MISSING_FORMAL_BUY",
            })
            continue

        er = row_at(idx, entry_date, code)
        if er is None:
            rows.append({
                "code": code, "entry_date": entry_date,
                "formal_decision_date": formal_decision, "formal_exit_date": formal_exit,
                "status": "MISSING_ENTRY_MARKET_ROW",
            })
            continue

        entry_price = float(b.fill_price)
        factor = float(er.aclose / er.close) if np.isfinite(er.aclose) and er.close else 1.0
        entry_adj = entry_price * factor
        p = bt.Position(
            "R05", code, str(g.name), int(g.shares), entry_date,
            entry_price, entry_adj,
            entry_price * int(g.shares) * (1.0 + bt.BUY_FEE),
            entry_adj,
        )

        first_trigger_date = None
        first_reason = None
        formal_day_reason = None
        formal_day_close = None
        formal_day_low = None
        formal_day_hard = entry_adj * 0.90

        sim_dates = [d for d in dates if entry_date <= d <= formal_decision]
        for di in sim_dates:
            r = row_at(idx, di, code)
            if r is None:
                continue
            reason = bt.r05_exit_reason(p, r)
            if di == formal_decision:
                formal_day_reason = reason
                formal_day_close = float(r.aclose)
                f = float(r.aclose / r.close) if r.close else 1.0
                formal_day_low = float(r.low) * f
            if reason is not None and first_trigger_date is None:
                first_trigger_date = di
                first_reason = reason
                # A causal engine exits at T+1, so no later state is relevant.
                break

        expected_exit = next_date.get(first_trigger_date) if first_trigger_date is not None else None
        if first_trigger_date is None:
            status = "FORMAL_EXIT_BEFORE_ANY_DOCUMENTED_TRIGGER"
        elif first_trigger_date < formal_decision:
            status = "FORMAL_DECISION_LATER_THAN_FIRST_TRIGGER"
        elif first_trigger_date > formal_decision:
            status = "FORMAL_DECISION_EARLY"
        else:
            status = "DECISION_MATCH"
        if first_trigger_date is None and formal_day_reason is None:
            status = "FORMAL_DECISION_HAS_NO_T_CLOSE_TRIGGER"

        rows.append({
            "code": code, "name": str(g.name), "entry_date": entry_date,
            "entry_price": entry_price,
            "formal_decision_date": formal_decision,
            "formal_exit_date": formal_exit,
            "first_causal_trigger_date": first_trigger_date,
            "first_causal_reason": first_reason,
            "causal_t1_exit_date": expected_exit,
            "formal_day_reason": formal_day_reason,
            "formal_day_aclose": formal_day_close,
            "formal_day_low_adj": formal_day_low,
            "hard_threshold_adj": formal_day_hard,
            "status": status,
        })

    q = pd.DataFrame(rows)
    counts = q.status.value_counts(dropna=False).to_dict()
    report = {
        "reference_is_engine_input": False,
        "audit_rule": "R0.5 exit decision may use T-close-known state only; execution is next trading day",
        "formal_r05_exits": int(len(q)),
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "decision_match": int((q.status == "DECISION_MATCH").sum()),
        "no_t_close_trigger_on_formal_decision": int((q.status == "FORMAL_DECISION_HAS_NO_T_CLOSE_TRIGGER").sum()),
        "rows": q.to_dict("records"),
    }
    q.to_csv(OUT / "r05_exit_timing_audit.csv", index=False, encoding="utf-8-sig")
    (OUT / "r05_exit_timing_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in (
        "formal_r05_exits", "status_counts", "decision_match", "no_t_close_trigger_on_formal_decision"
    )}, ensure_ascii=False, indent=2))
    bad = q[q.status != "DECISION_MATCH"].head(10)
    if not bad.empty:
        print("FIRST NON-MATCHING R0.5 EXITS")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
