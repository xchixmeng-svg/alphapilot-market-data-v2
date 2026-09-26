#!/usr/bin/env python3
"""Build read-only causal context for already-consumed V6.2 packets.

No cases are selected here.  The supplied packet files are the immutable
allow-list.  Hidden outcomes and future paths are neither accepted nor read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from v6_2_v4_2_contract import (
    CONTRACT_VERSION,
    LAUNCH_WINDOWS,
    OUTCOME_HORIZONS,
    ContractError,
    index_exact,
    load_jsonl,
    norm_code,
    norm_date,
    validate_packet,
    validate_sidecar,
)

FORBIDDEN = re.compile(r"hidden|outcome|future|forward|realized|label|target|y_hit|mfe|mae|time_to_hit", re.I)
RAW_MULTIPLE = re.compile(r"(^|_)(raw_)?(pe|pb|p_e|p_b|price_earnings|price_book)($|_)", re.I)

FAMILY_PATTERNS = {
    "market_context": re.compile(r"^(market|twse|taiex|tpex|otc|benchmark|breadth|regime|index_market|mkt|0050)_|^(equity_count|valid_return_count|extreme_return_excluded|advancers|decliners|unchanged|total_amount|median_amount)$", re.I),
    "industry_index_context": re.compile(r"^(industry_ret1|industry_index|sector_index|industry_idx|sector_idx|industry_market|sector_market)_", re.I),
    "macro_context": re.compile(r"^(macro|fx|usd|twd|rate|yield|vix|oil|commodity|policy)_", re.I),
    "event_timeliness_context": re.compile(r"^(event|mops)_", re.I),
    "revenue_context": re.compile(r"^(rev|revenue|monthly_revenue)_", re.I),
}


def scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def safe_columns(columns: list[str]) -> dict[str, list[str]]:
    routed = {name: [] for name in FAMILY_PATTERNS}
    for col in columns:
        if col in {"date", "code"} or FORBIDDEN.search(col) or RAW_MULTIPLE.search(col):
            continue
        for family, pattern in FAMILY_PATTERNS.items():
            if pattern.search(col):
                routed[family].append(col)
                break
    return routed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--cases", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    packets = []
    for path in args.cases:
        if FORBIDDEN.search(path.name):
            raise ContractError(f"refusing forbidden input path: {path}")
        packets.extend(load_jsonl(path))
    for packet in packets:
        # The already-consumed v1 transport contained raw PE/PB.  It is accepted
        # only at this import boundary and removed by the V4.2 upgrader before
        # Stage C.  Every production validation remains fail closed.
        validate_packet(packet, allow_legacy_raw_multiples=True)
    packet_index = index_exact(packets)

    df = pd.read_parquet(args.evidence)
    if "date" not in df or "code" not in df:
        raise ContractError("evidence bundle lacks date/code")
    df = df.copy()
    df["date"] = df["date"].map(norm_date)
    df["code"] = df["code"].map(norm_code)
    if int(df["date"].max()) > 20241231:
        raise ContractError("evidence bundle exposes data after 2024-12-31")
    if df.duplicated(["date", "code"]).any():
        raise ContractError("evidence bundle has duplicate date/code keys")
    evidence_index = df.set_index(["date", "code"], drop=False)
    routed = safe_columns(list(df.columns))
    expected_counts = {
        "market_context": (10, None),
        "industry_index_context": (37, 37),
        "macro_context": (3, None),
        "event_timeliness_context": (4, None),
        "revenue_context": (7, None),
    }
    for family, (minimum, exact) in expected_counts.items():
        count = len(routed[family])
        if count < minimum or (exact is not None and count != exact):
            raise ContractError(f"incomplete {family}: routed={count}, required={exact or ('>='+str(minimum))}")

    rows = []
    for exact, packet in sorted(packet_index.items()):
        case_id, date, code = exact
        if (date, code) not in evidence_index.index:
            raise ContractError(f"exact evidence row missing: {exact}")
        src = evidence_index.loc[(date, code)]
        if isinstance(src, pd.DataFrame):
            raise ContractError(f"ambiguous evidence row: {exact}")
        context = {}
        for family, cols in routed.items():
            values = {col: scalar(src[col]) for col in cols if scalar(src[col]) is not None}
            context[family] = {
                "values": values,
                "source_columns": sorted(values),
            }
        context["industry_index_context"].update({
            "relationship_to_company": "CONTEXT_ONLY_NOT_COMPANY_CLASSIFICATION",
            "company_industry_mapping": None,
        })
        sidecar = {
            "contract_version": CONTRACT_VERSION,
            "case_id": case_id,
            "decision_date": date,
            "code": code,
            "causal_context_read_only": context,
            "launch_observation_windows": LAUNCH_WINDOWS,
            "profit_outcome_horizons": OUTCOME_HORIZONS,
            "h120_is_holding_period": False,
            "outcomes_opened": False,
            "selection_recomputed": False,
            "frozen_layers_modified": False,
        }
        validate_sidecar(sidecar)
        rows.append(sidecar)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    args.out.write_text(text, encoding="utf-8")
    summary = {
        "status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "cases": len(rows),
        "exact_case_keys_only": True,
        "outcomes_opened": False,
        "selection_recomputed": False,
        "frozen_layers_modified": False,
        "source_sha256": hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        "sidecar_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "routed_column_counts": {k: len(v) for k, v in routed.items()},
    }
    (args.out.parent / "CAUSAL_CONTEXT_BUILD_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
