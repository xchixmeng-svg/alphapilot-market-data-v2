#!/usr/bin/env python3
"""Fail-closed transport contract for AlphaPilot V6.2 / V4.2.

This module validates only packet transport and isolation.  It does not score,
select, label, or inspect outcomes.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

CONTRACT_VERSION = "V6.2-CAUSAL-CONTEXT-PACK-v1"
LAUNCH_WINDOWS = [1, 3, 5, 10, 20, 30]
OUTCOME_HORIZONS = [20, 30, 40, 60, 120]
REQUIRED_CONTEXT = {
    "market_context",
    "industry_index_context",
    "macro_context",
    "event_timeliness_context",
    "revenue_context",
}
FALSE_FLAGS = ("outcomes_opened", "selection_recomputed", "frozen_layers_modified")

_FORBIDDEN_KEY = re.compile(
    r"hidden|outcome|future|forward_path|realized|label|target|y_hit|mfe|mae|time_to_hit",
    re.I,
)
_RAW_MULTIPLE_KEY = re.compile(r"^(raw_)?(pe|pb|p_e|p_b|price_earnings|price_book)$", re.I)


class ContractError(RuntimeError):
    pass


def norm_code(value: Any) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(4)


def norm_date(value: Any) -> int:
    if hasattr(value, "strftime"):
        return int(value.strftime("%Y%m%d"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = str(int(value))
        if len(numeric) == 8:
            return int(numeric)
    s = re.sub(r"[^0-9]", "", str(value))
    if len(s) >= 8:
        s = s[:8]
    if len(s) != 8:
        raise ContractError(f"invalid decision date: {value!r}")
    return int(s)


def _walk(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield child_path, child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _walk(child, f"{path}[{i}]")


def _assert_no_leak(value: Any, *, where: str, allow_raw_multiples: bool = False) -> None:
    for path, child in _walk(value):
        key = re.sub(r".*\.", "", path).split("[")[0]
        if key in {"profit_outcome_horizons", "outcomes_opened"}:
            continue
        if _FORBIDDEN_KEY.search(key):
            raise ContractError(f"{where}: forbidden future/outcome field: {path}")
        if not allow_raw_multiples and _RAW_MULTIPLE_KEY.match(key) and child is not None:
            raise ContractError(f"{where}: raw PE/PB field leaked: {path}")


def validate_packet(packet: dict[str, Any], *, allow_legacy_raw_multiples: bool = False) -> None:
    for key in ("case_id", "decision_date", "code", "evidence", "numerical_reference_read_only"):
        if key not in packet:
            raise ContractError(f"packet missing {key}")
    if norm_date(packet["decision_date"]) > 20241231:
        raise ContractError("packet decision date exceeds 2024-12-31")
    if not isinstance(packet["evidence"], dict):
        raise ContractError("packet evidence must be an object")
    _assert_no_leak(packet, where="packet", allow_raw_multiples=allow_legacy_raw_multiples)


def validate_sidecar(sidecar: dict[str, Any]) -> None:
    if sidecar.get("contract_version") != CONTRACT_VERSION:
        raise ContractError("wrong or missing causal-context contract version")
    for key in ("case_id", "decision_date", "code", "causal_context_read_only"):
        if key not in sidecar:
            raise ContractError(f"sidecar missing {key}")
    if norm_date(sidecar["decision_date"]) > 20241231:
        raise ContractError("sidecar decision date exceeds 2024-12-31")
    for flag in FALSE_FLAGS:
        if sidecar.get(flag) is not False:
            raise ContractError(f"sidecar {flag} must be literal false")
    if sidecar.get("launch_observation_windows") != LAUNCH_WINDOWS:
        raise ContractError("launch observation windows drifted")
    if sidecar.get("profit_outcome_horizons") != OUTCOME_HORIZONS:
        raise ContractError("profit/outcome horizons drifted")
    if sidecar.get("h120_is_holding_period") is not False:
        raise ContractError("120 sessions must not be represented as a holding period")
    context = sidecar["causal_context_read_only"]
    if not isinstance(context, dict) or not REQUIRED_CONTEXT.issubset(context):
        raise ContractError("causal_context_read_only is missing required context families")
    industry = context["industry_index_context"]
    if not isinstance(industry, dict):
        raise ContractError("industry_index_context must be an object")
    if industry.get("relationship_to_company") != "CONTEXT_ONLY_NOT_COMPANY_CLASSIFICATION":
        raise ContractError("industry index context may not impersonate company industry")
    if len(industry.get("source_columns", [])) != 37:
        raise ContractError("causal context must carry exactly 37 frozen industry indices")
    for path, child in _walk(context):
        key = re.sub(r".*\.", "", path).split("[")[0].lower()
        if key in {"company_industry", "company_sector", "company_industry_name"} and child not in (None, ""):
            raise ContractError(f"unverified company-industry claim: {path}")
    _assert_no_leak(context, where="causal_context_read_only")


def validate_pair(packet: dict[str, Any], sidecar: dict[str, Any]) -> None:
    validate_packet(packet)
    validate_sidecar(sidecar)
    exact_packet = (int(packet["case_id"]), norm_date(packet["decision_date"]), norm_code(packet["code"]))
    exact_sidecar = (int(sidecar["case_id"]), norm_date(sidecar["decision_date"]), norm_code(sidecar["code"]))
    if exact_packet != exact_sidecar:
        raise ContractError(f"packet/sidecar key mismatch: {exact_packet} != {exact_sidecar}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_exact(rows: Iterable[dict[str, Any]]) -> dict[tuple[int, int, str], dict[str, Any]]:
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["case_id"]), norm_date(row["decision_date"]), norm_code(row["code"]))
        if key in out:
            raise ContractError(f"duplicate exact key: {key}")
        out[key] = row
    return out
