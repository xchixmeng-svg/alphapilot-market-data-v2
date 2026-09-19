#!/usr/bin/env python3
"""
audit_v6_2_decision_collapse.py

AlphaPilot V6.2 -- Taiwan-equity AI Opportunity Detection System
Historical replay audit (2024, 64-case diagnostic batch).

PURPOSE
-------
This script does NOT evaluate "does a stock rise +10% within 120 sessions".
It diagnoses a specific, already-observed behavioral anomaly:

    64 cases replayed. AI decisions were:
        REJECT           = 60
        WATCH            =  4
        CANDIDATE        =  0
        HIGH_CONVICTION  =  0

    Yet a large share of those 64 stocks subsequently achieved +10% (and
    larger) moves. This script asks *why* the AI decision layer collapsed
    toward REJECT/WATCH, using only deterministic, already-computed
    ground-truth outcome data -- it does not re-run any model, does not
    search for an optimal probability cutoff, and does not build a new
    selection rule.

Two time dimensions are kept strictly separate throughout, per the V6.2
architecture definition:

    A. LAUNCH   -- did the stock *begin* a successful up-leg, observed over
                   a 1/3/5/10/20/30 trading-session window after the
                   decision date?
    B. PROFIT   -- did price eventually reach +10/+20/+30/+50%, observed
                   over 20/30/40/60/120 trading-session horizons?

    120 trading sessions is one outcome-validation horizon among several.
    It is NOT a holding period, NOT a forced exit rule, and NOT the sole
    pass/fail criterion for whether the AI "got it right".

THIS SCRIPT DOES NOT:
    - modify the AI's original decision
    - re-run any LLM
    - search for / propose a probability or launch-probability threshold
    - build a new buy/sell rule
    - open or read any 2025 data
    - touch frozen 0A-0F layers, R10, or any model logic
    - silently drop duplicate columns or silently continue past bad data

Everything below is read-only, local-file, deterministic diagnostics.

DATA LAYOUT (all local, no GitHub access is attempted)
--------------------------------------------------------
v6_2_audit/
    audit_v6_2_decision_collapse.py      (this file)
    data/
        prepared/
            cases_shard_0.jsonl ... cases_shard_3.jsonl
            HIDDEN_OUTCOMES.csv
            HIDDEN_FUTURE_PATHS.parquet
            CASE_MANIFEST.csv
            PREP_SUMMARY.json
        responses/
            responses_shard_0.json ... responses_shard_3.json
    output/                              (created by this script)

Run:
    python -m py_compile audit_v6_2_decision_collapse.py
    python audit_v6_2_decision_collapse.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# ============================================================================
# Constants
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PREPARED_DIR = Path(os.environ.get("V62_PREPARED_DIR", DATA_DIR / "prepared"))
RESPONSES_DIR = Path(os.environ.get("V62_RESPONSES_DIR", DATA_DIR / "responses"))
OUTPUT_DIR = Path(os.environ.get("V62_OUTPUT_DIR", BASE_DIR / "output"))

SOURCE_RUN_ID = 35435870750
SOURCE_HEAD_SHA = "1d43e2a3c0fbb420d2218d838544e7810c8c491c"
PREPARED_ARTIFACT_ID = 10582014726
RESPONSE_ARTIFACT_IDS = (10583306332, 10582917794, 10581414968, 10582976677)

EXPECTED_N_CASES = 64
DECISION_DATE_CUTOFF = 20241231  # inclusive; this batch is a 2024 replay

PROFIT_TARGETS = (10, 20, 30, 50)
PROFIT_HORIZONS = (20, 30, 40, 60, 120)
LAUNCH_WINDOWS = (1, 3, 5, 10, 20, 30)

DECISION_ORDER = ["REJECT", "WATCH", "CANDIDATE", "HIGH_CONVICTION"]
ACTIONABLE_DECISIONS = {"CANDIDATE", "HIGH_CONVICTION"}
NON_ACTIONABLE_DECISIONS = {"REJECT", "WATCH"}

EVIDENCE_QUALITY_ORDER = ["WEAK", "MODERATE", "STRONG"]

AVAILABLE_EVIDENCE_FAMILIES = [
    "price_volume_structure",
    "institutional_flow",
    "revenue",
    "valuation",
    "mops_material_information",
]
KNOWN_MISSING_EVIDENCE_FAMILIES = [
    "eps_revisions",
    "analyst_consensus",
    "industry_pricing",
    "inventory_supply_demand",
    "broad_news_semantics",
]

SIZE_BUCKET_ORDER = [
    "SMALL",
    "BASE_WINNER",
    "MEDIUM_WINNER",
    "LARGE_WINNER",
    "VERY_LARGE_WINNER",
]

# Deterministic keyword/phrase diagnostics (Section 24). Supports EN + ZH.
# Each pattern is a compiled case-insensitive regex tried against the
# concatenated text of decision_reason + bear_thesis + invalidation +
# hypothesis. Chinese keywords do not need case-insensitivity but are
# included in the same regex set for simplicity.
REASON_LANGUAGE_PATTERNS: dict[str, str] = {
    "missing": r"missing|不足|缺乏|缺少|缺(?!口)",
    "insufficient": r"insufficient|不充分|不夠充分|不夠完整",
    "unavailable": r"unavailable|無法取得|無資料|尚無資料",
    "lack": r"\black(?:s|ing)?\b|缺乏|欠缺",
    "uncertain": r"uncertain|不確定|未明朗|尚不明朗",
    "caution": r"caution|謹慎|保守|審慎",
    "risk": r"\brisk\b|風險",
    "wait": r"\bwait(?:ing)?\b|等待|觀望",
    "confirmation": r"confirmation|確認|待確認|尚待確認",
    "valuation": r"valuation|估值|本益比|股價淨值比",
    "revenue": r"revenue|營收",
    "institutional": r"institutional|法人",
    "price": r"\bprice\b|價格|股價",
    "trend": r"\btrend\b|趨勢",
    "support": r"\bsupport\b|支撐",
    "resistance": r"\bresistance\b|壓力",
    "volume": r"\bvolume\b|成交量|量能",
    # References to evidence families that the packet explicitly marks as
    # unavailable.  These are references-to-missing-data diagnostics; they do
    # not by themselves prove a factual hallucination.
    "eps_revision_reference": r"\beps\b|earnings per share|每股盈餘|每股收益|獲利預估|盈利预测",
    "analyst_consensus_reference": r"analyst|consensus|分析師|分析师|市場共識|市场共识|機構評級|机构评级",
    "industry_pricing_reference": r"industry pricing|產業報價|产业报价|行業價格|行业价格|產品報價|产品报价",
    "inventory_supply_demand_reference": r"inventory|supply.?demand|庫存|库存|供需|缺貨|缺货",
    "broad_news_reference": r"\bnews\b|新聞|新闻",
    "net_profit_reference": r"net profit|淨利|净利|淨利润|净利润",
}
_COMPILED_REASON_PATTERNS = {
    name: re.compile(pat, flags=re.IGNORECASE) for name, pat in REASON_LANGUAGE_PATTERNS.items()
}


# ============================================================================
# Utility: fail loudly, never silently continue
# ============================================================================


def _fail(msg: str) -> None:
    raise RuntimeError(f"[audit_v6_2_decision_collapse] {msg}")


def _require_columns_unique(df: pd.DataFrame, stage: str) -> None:
    if not df.columns.is_unique:
        dupes = df.columns[df.columns.duplicated()].tolist()
        _fail(
            f"Duplicate columns detected after stage '{stage}': {dupes}. "
            f"Refusing to continue (silently dropping duplicate columns is "
            f"explicitly disallowed for this audit)."
        )


# ============================================================================
# 1. Code normalization
# ============================================================================


def normalize_code(raw: Any) -> str:
    """
    Canonicalize a Taiwan equity code to a fixed string form.

    Rules (all mandatory, in order):
      1. If the value is already a string, use it directly -- never round-trip
         through int(), because that is exactly the operation that silently
         turns "0050" into 50 or "009829" into 9829.
      2. Strip whitespace.
      3. Remove an accidental trailing ".0" that appears when a code was at
         some point coerced to a pandas/numpy float (e.g. "2330.0" -> "2330").
      4. Zero-pad to a *minimum* of 4 characters (zfill(4)). Codes that are
         already longer than 4 characters (e.g. 6-digit codes such as
         "009829") are left untouched -- zfill only pads, it never truncates.

    A code that arrives as a Python int/float/np.integer/np.floating is
    accepted defensively (in case some upstream file was written with
    numeric-typed codes), but this path is inherently lossy for any code
    that originally had a leading zero, so it is only used as a fallback
    and is flagged via a warning collected by the caller if desired.
    """
    if raw is None:
        _fail("normalize_code() received None; a stock code must never be null.")

    if isinstance(raw, str):
        s = raw.strip()
    elif isinstance(raw, (int, np.integer)):
        s = str(int(raw))
    elif isinstance(raw, (float, np.floating)):
        if float(raw).is_integer():
            s = str(int(raw))
        else:
            _fail(f"normalize_code() received non-integer float code: {raw!r}")
    else:
        _fail(f"normalize_code() received unsupported type {type(raw)!r} for value {raw!r}")

    s = s.strip()
    if s.endswith(".0"):
        s = s[:-2]
    if not s.isdigit():
        _fail(f"normalize_code() produced a non-numeric code string: {s!r} (raw={raw!r})")
    if len(s) < 4:
        s = s.zfill(4)
    return s


def _normalize_code_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_code).astype(str)


# ============================================================================
# 2. Loaders
# ============================================================================


def load_prep_summary() -> dict:
    path = PREPARED_DIR / "PREP_SUMMARY.json"
    if not path.exists():
        _fail(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if summary.get("2025_opened", None) is not True and summary.get("2025_opened", None) is not False:
        _fail(
            "PREP_SUMMARY.json is missing an explicit boolean '2025_opened' field. "
            "Refusing to proceed without an explicit seal-status flag."
        )
    if summary.get("2025_opened") is not False:
        _fail(
            "PREP_SUMMARY.json declares 2025_opened != false. "
            "This audit is only authorized to run against a sealed 2024 replay. "
            "Aborting."
        )
    return summary


def _flatten_price_volume_structure(d: dict) -> dict:
    # No family prefix for this family, per the explicit examples in the
    # spec (pkt__current_price, pkt__ret20_pct, ...).
    out = {}
    for k, v in (d or {}).items():
        out[k] = v
    return out


def _flatten_family(d: dict, prefix: str) -> dict:
    out = {}
    for k, v in (d or {}).items():
        out[f"{prefix}_{k}"] = v
    return out


def _flatten_case_record(rec: dict) -> dict:
    """
    Flatten one raw prepared-case JSON object into a single-level dict of
    *unprefixed* field names (the pkt__ namespace prefix is applied later,
    uniformly, right before the merge -- see build_canonical_table()).
    """
    for required in ("case_id", "decision_date", "code"):
        if required not in rec:
            _fail(f"Prepared case record missing required field '{required}': {rec}")

    flat: dict[str, Any] = {
        "case_id": rec["case_id"],
        "decision_date": rec["decision_date"],
        "code": rec["code"],
        "validation_stratum": rec.get("validation_stratum"),
    }

    evidence = rec.get("evidence", {})
    if not isinstance(evidence, dict):
        _fail(f"case_id={rec.get('case_id')}: 'evidence' block is not an object.")

    flat["available_evidence_ids"] = evidence.get("available_evidence_ids")
    flat["missing_evidence_ids"] = evidence.get("missing_evidence_ids")

    flat.update(_flatten_price_volume_structure(evidence.get("price_volume_structure", {})))
    flat.update(_flatten_family(evidence.get("institutional_flow", {}), "institutional"))
    flat.update(_flatten_family(evidence.get("revenue", {}), "revenue"))
    flat.update(_flatten_family(evidence.get("valuation", {}), "valuation"))
    flat.update(_flatten_family(evidence.get("mops_material_information", {}), "mops"))

    numref = rec.get("numerical_reference_read_only", {})
    if not isinstance(numref, dict):
        _fail(f"case_id={rec.get('case_id')}: 'numerical_reference_read_only' block is not an object.")

    for target in PROFIT_TARGETS:
        for horizon in PROFIT_HORIZONS:
            key = f"p_hit{target}_h{horizon}"
            flat[key] = numref.get(key, np.nan)

    for n in LAUNCH_WINDOWS:
        src_key = f"p_successful_launch_by_{n}"
        dst_key = f"launch_by_{n}"
        flat[dst_key] = numref.get(src_key, np.nan)

    return flat


def load_packets() -> pd.DataFrame:
    shard_paths = sorted(PREPARED_DIR.glob("cases_shard_*.jsonl"))
    if not shard_paths:
        _fail(f"No cases_shard_*.jsonl files found under {PREPARED_DIR}")

    rows: list[dict] = []
    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    _fail(f"{path.name}:{line_no}: invalid JSON ({e})")
                rows.append(_flatten_case_record(rec))

    df = pd.DataFrame(rows)
    if df.empty:
        _fail("load_packets(): produced an empty DataFrame.")

    df["case_id"] = df["case_id"].astype("int64")
    df["decision_date"] = df["decision_date"].astype("int64")
    df["code"] = _normalize_code_series(df["code"])

    if df["case_id"].duplicated().any():
        dupes = df.loc[df["case_id"].duplicated(), "case_id"].tolist()
        _fail(f"load_packets(): duplicate case_id values found: {dupes}")

    bad_dates = df.loc[df["decision_date"] > DECISION_DATE_CUTOFF, ["case_id", "decision_date"]]
    if not bad_dates.empty:
        _fail(
            "load_packets(): found decision_date(s) beyond the 2024 cutoff "
            f"({DECISION_DATE_CUTOFF}). This audit must not touch 2025 data. "
            f"Offending rows:\n{bad_dates}"
        )

    _require_columns_unique(df, "load_packets (pre-namespace)")
    return df


def _flatten_response_record(rec: dict) -> dict:
    for required in ("case_id", "date", "code", "decision"):
        if required not in rec:
            _fail(f"AI response record missing required field '{required}': {rec}")

    flat: dict[str, Any] = {}
    entry = rec.get("entry", {}) or {}
    failure_exit = rec.get("failure_exit", {}) or {}

    for k, v in rec.items():
        if k in ("entry", "failure_exit"):
            continue
        flat[k] = v

    flat["entry_status"] = entry.get("status")
    flat["entry_ideal_low"] = entry.get("ideal_low")
    flat["entry_ideal_high"] = entry.get("ideal_high")

    flat["failure_exit_price"] = failure_exit.get("exit_price")
    flat["failure_exit_reason"] = failure_exit.get("reason")
    flat["failure_exit_trigger_type"] = failure_exit.get("trigger_type")

    return flat


def load_responses() -> pd.DataFrame:
    shard_paths = sorted(RESPONSES_DIR.glob("responses_shard_*.json"))
    if len(shard_paths) != 4:
        _fail(
            f"Expected exactly four responses_shard_*.json files under "
            f"{RESPONSES_DIR}, found {len(shard_paths)}."
        )

    error_paths = sorted(RESPONSES_DIR.glob("errors_shard_*.json"))
    if len(error_paths) != 4:
        _fail(
            f"Expected exactly four errors_shard_*.json files under "
            f"{RESPONSES_DIR}, found {len(error_paths)}."
        )
    recorded_errors: list[Any] = []
    for path in error_paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            _fail(f"{path.name}: expected a top-level JSON error array.")
        recorded_errors.extend(payload)
    if recorded_errors:
        _fail(f"AI response artifacts contain recorded errors: {recorded_errors[:5]}")

    rows: list[dict] = []
    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as f:
            try:
                arr = json.load(f)
            except json.JSONDecodeError as e:
                _fail(f"{path.name}: invalid JSON ({e})")
        if not isinstance(arr, list):
            _fail(f"{path.name}: expected a top-level JSON array of response objects.")
        for rec in arr:
            rows.append(_flatten_response_record(rec))

    df = pd.DataFrame(rows)
    if df.empty:
        _fail("load_responses(): produced an empty DataFrame.")

    df["case_id"] = df["case_id"].astype("int64")
    df["date"] = df["date"].astype("int64")
    df["code"] = _normalize_code_series(df["code"])

    if df["case_id"].duplicated().any():
        dupes = df.loc[df["case_id"].duplicated(), "case_id"].tolist()
        _fail(f"load_responses(): duplicate case_id values found: {dupes}")

    bad_dates = df.loc[df["date"] > DECISION_DATE_CUTOFF, ["case_id", "date"]]
    if not bad_dates.empty:
        _fail(
            "load_responses(): found date(s) beyond the 2024 cutoff "
            f"({DECISION_DATE_CUTOFF}). Aborting.\n{bad_dates}"
        )

    bad_decisions = df.loc[~df["decision"].isin(DECISION_ORDER), ["case_id", "decision"]]
    if not bad_decisions.empty:
        _fail(f"load_responses(): unrecognized decision value(s):\n{bad_decisions}")

    _require_columns_unique(df, "load_responses (AI namespace)")
    return df


def load_outcomes() -> pd.DataFrame:
    path = PREPARED_DIR / "HIDDEN_OUTCOMES.csv"
    if not path.exists():
        _fail(f"Required file not found: {path}")

    df = pd.read_csv(path, dtype={"code": str})
    if df.empty:
        _fail("load_outcomes(): HIDDEN_OUTCOMES.csv is empty.")

    for required in ("case_id", "date", "code"):
        if required not in df.columns:
            _fail(f"HIDDEN_OUTCOMES.csv missing required column '{required}'")

    df["case_id"] = df["case_id"].astype("int64")
    df["date"] = df["date"].astype("int64")
    df["code"] = _normalize_code_series(df["code"])

    expected_cols = []
    for target in PROFIT_TARGETS:
        for horizon in PROFIT_HORIZONS:
            expected_cols.append(f"y_hit{target}_h{horizon}")
            expected_cols.append(f"p_hit{target}_h{horizon}")
    for horizon in (20, 40, 60, 120):
        expected_cols.append(f"mfe_h{horizon}")
        expected_cols.append(f"mae_h{horizon}")
    for target in PROFIT_TARGETS:
        expected_cols.append(f"time_to_hit{target}")

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        _fail(f"HIDDEN_OUTCOMES.csv is missing expected columns: {missing}")

    if df["case_id"].duplicated().any():
        dupes = df.loc[df["case_id"].duplicated(), "case_id"].tolist()
        _fail(f"load_outcomes(): duplicate case_id values found: {dupes}")

    bad_dates = df.loc[df["date"] > DECISION_DATE_CUTOFF, ["case_id", "date"]]
    if not bad_dates.empty:
        _fail(f"load_outcomes(): found date(s) beyond the 2024 cutoff. Aborting.\n{bad_dates}")

    _require_columns_unique(df, "load_outcomes (pre-namespace)")
    return df


def load_future_paths() -> pd.DataFrame:
    path = PREPARED_DIR / "HIDDEN_FUTURE_PATHS.parquet"
    if not path.exists():
        _fail(f"Required file not found: {path}")

    df = pd.read_parquet(path)
    for required in ("case_id", "session", "open", "high", "low", "close"):
        if required not in df.columns:
            _fail(f"HIDDEN_FUTURE_PATHS.parquet missing required column '{required}'")

    df["case_id"] = df["case_id"].astype("int64")
    df["session"] = df["session"].astype("int64")

    if (df["session"] < 1).any() or (df["session"] > 120).any():
        _fail("HIDDEN_FUTURE_PATHS.parquet contains session values outside [1, 120].")

    dupe_mask = df.duplicated(subset=["case_id", "session"])
    if dupe_mask.any():
        _fail(
            "HIDDEN_FUTURE_PATHS.parquet contains duplicate (case_id, session) rows: "
            f"{df.loc[dupe_mask, ['case_id', 'session']].to_dict('records')}"
        )

    if df[["open", "high", "low", "close"]].isna().any().any():
        _fail("HIDDEN_FUTURE_PATHS.parquet contains null OHLC values.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        _fail("HIDDEN_FUTURE_PATHS.parquet contains non-positive OHLC values.")

    # The prepared replay was explicitly selected from rows with complete
    # 120-session evaluability.  Require every case to contain exactly the
    # contiguous sessions 1..120; otherwise timing diagnostics can look valid
    # while silently using a shortened path.
    expected_sessions = list(range(1, 121))
    for case_id, g in df.groupby("case_id", sort=False):
        got = sorted(g["session"].tolist())
        if got != expected_sessions:
            _fail(
                f"HIDDEN_FUTURE_PATHS.parquet case_id={case_id} does not contain "
                "the exact contiguous session range 1..120."
            )

    _require_columns_unique(df, "load_future_paths")

    return df


def load_case_manifest() -> pd.DataFrame:
    path = PREPARED_DIR / "CASE_MANIFEST.csv"
    if not path.exists():
        _fail(f"Required file not found: {path}")
    df = pd.read_csv(path, dtype={"code": str} if "code" in pd.read_csv(path, nrows=0).columns else None)
    if "case_id" not in df.columns:
        _fail("CASE_MANIFEST.csv missing required column 'case_id'")
    df["case_id"] = df["case_id"].astype("int64")
    if len(df) != EXPECTED_N_CASES or df["case_id"].nunique() != EXPECTED_N_CASES:
        _fail(
            f"CASE_MANIFEST.csv must contain exactly {EXPECTED_N_CASES} unique cases; "
            f"got rows={len(df)}, unique_case_id={df['case_id'].nunique()}."
        )
    _require_columns_unique(df, "load_case_manifest")
    return df


# ============================================================================
# 3. Source validation
# ============================================================================


def validate_sources(
    packets: pd.DataFrame,
    responses: pd.DataFrame,
    outcomes: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    for name, df in (("packets", packets), ("responses", responses), ("outcomes", outcomes)):
        n = len(df)
        n_unique = df["case_id"].nunique()
        if n != EXPECTED_N_CASES:
            _fail(f"validate_sources(): {name} has {n} rows, expected exactly {EXPECTED_N_CASES}.")
        if n_unique != EXPECTED_N_CASES:
            _fail(
                f"validate_sources(): {name} has {n_unique} unique case_id values, "
                f"expected exactly {EXPECTED_N_CASES}."
            )

    manifest_ids = set(manifest["case_id"].tolist())
    packet_ids = set(packets["case_id"].tolist())
    response_ids = set(responses["case_id"].tolist())
    outcome_ids = set(outcomes["case_id"].tolist())

    if not (manifest_ids == packet_ids == response_ids == outcome_ids):
        _fail(
            "validate_sources(): case_id sets differ across sources.\n"
            f"  manifest only:  {sorted(manifest_ids - packet_ids - response_ids - outcome_ids)}\n"
            f"  packets only:   {sorted(packet_ids - manifest_ids)}\n"
            f"  responses only: {sorted(response_ids - manifest_ids)}\n"
            f"  outcomes only:  {sorted(outcome_ids - manifest_ids)}"
        )

    if len(manifest_ids) != EXPECTED_N_CASES:
        _fail(
            f"validate_sources(): manifest has {len(manifest_ids)} unique case_ids, "
            f"expected {EXPECTED_N_CASES}."
        )


# ============================================================================
# 4. Canonical table construction
# ============================================================================


def build_canonical_table(
    packets: pd.DataFrame, responses: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    """
    Single merge chain:  AI responses -> prepared packets -> hidden outcomes.

    Namespacing (no _x / _y ever):
        AI response fields   -> kept as-is (decision, decision_reason, ...)
        packet fields        -> pkt__<field>   (except merge key case_id)
        outcome fields        -> out__<field>   (except merge keys
                                  case_id / date / code)

    A consistency check between responses.date and packets.decision_date is
    performed explicitly before dropping the redundant column, rather than
    silently trusting one source.
    """
    pkt = packets.copy()
    pkt_rename = {c: f"pkt__{c}" for c in pkt.columns if c != "case_id"}
    pkt = pkt.rename(columns=pkt_rename)
    _require_columns_unique(pkt, "packets namespace")

    out = outcomes.copy()
    out_rename = {c: f"out__{c}" for c in out.columns if c not in ("case_id", "date", "code")}
    out = out.rename(columns=out_rename)
    _require_columns_unique(out, "outcomes namespace")

    ai = responses.copy()
    _require_columns_unique(ai, "AI responses namespace")

    # --- merge 1: AI response -> packet -------------------------------
    m = ai.merge(
        pkt,
        on="case_id",
        how="left",
        validate="one_to_one",
        indicator="_merge_pkt",
    )
    _require_columns_unique(m, "after AI -> packet merge")
    miss = m.loc[m["_merge_pkt"] != "both"]
    if not miss.empty:
        _fail(
            "build_canonical_table(): AI -> packet merge produced rows without a "
            f"matching packet. case_ids: {miss['case_id'].tolist()}"
        )
    m = m.drop(columns=["_merge_pkt"])

    # cross-check code / date consistency between the two sources instead of
    # silently keeping one and discarding the other
    code_mismatch = m.loc[m["code"] != m["pkt__code"], ["case_id", "code", "pkt__code"]]
    if not code_mismatch.empty:
        _fail(f"build_canonical_table(): code mismatch between AI response and packet:\n{code_mismatch}")
    date_mismatch = m.loc[m["date"] != m["pkt__decision_date"], ["case_id", "date", "pkt__decision_date"]]
    if not date_mismatch.empty:
        _fail(
            f"build_canonical_table(): date mismatch between AI response.date and "
            f"packet.decision_date:\n{date_mismatch}"
        )

    # The reasoning layer is forbidden to invent or alter calibrated
    # numerical probabilities.  The response repeats p_hit10_h120, so verify
    # exact semantic immutability (within serialization tolerance) against the
    # prepared packet before using any decision-layer result.
    if "p_hit10_h120" not in m.columns or "pkt__p_hit10_h120" not in m.columns:
        _fail("Missing response/packet p_hit10_h120 fields required for immutability audit.")
    p_ai = pd.to_numeric(m["p_hit10_h120"], errors="coerce")
    p_pkt = pd.to_numeric(m["pkt__p_hit10_h120"], errors="coerce")
    prob_bad = p_ai.isna() | p_pkt.isna() | ~np.isclose(p_ai, p_pkt, rtol=0.0, atol=1e-12)
    if prob_bad.any():
        _fail(
            "AI response p_hit10_h120 differs from the immutable prepared value for "
            f"case_ids {m.loc[prob_bad, 'case_id'].tolist()}."
        )

    for _, row in m.iterrows():
        available = row.get("pkt__available_evidence_ids")
        missing_ids = row.get("pkt__missing_evidence_ids")
        if not isinstance(available, list) or not isinstance(missing_ids, list):
            _fail(
                f"case_id={row['case_id']}: available/missing evidence IDs must be JSON lists."
            )
        available_set = set(available)
        explicit_missing_set = {f"missing:{x}" for x in missing_ids}
        for col in ("primary_evidence_ids", "secondary_evidence_ids", "counter_evidence_ids"):
            cited = row.get(col)
            if not isinstance(cited, list):
                _fail(f"case_id={row['case_id']}: {col} must be a JSON list.")
            allowed = available_set | (explicit_missing_set if col == "counter_evidence_ids" else set())
            invalid = sorted(set(cited) - allowed)
            if invalid:
                _fail(
                    f"case_id={row['case_id']}: {col} contains invalid evidence IDs {invalid}."
                )

    # --- merge 2: (AI+packet) -> outcome -------------------------------
    m = m.merge(
        out,
        on=["case_id", "date", "code"],
        how="left",
        validate="one_to_one",
        indicator="_merge_out",
    )
    _require_columns_unique(m, "after (AI+packet) -> outcome merge")
    miss = m.loc[m["_merge_out"] != "both"]
    if not miss.empty:
        _fail(
            "build_canonical_table(): (AI+packet) -> outcome merge produced rows "
            f"without a matching outcome. case_ids: {miss['case_id'].tolist()}"
        )
    m = m.drop(columns=["_merge_out"])

    if len(m) != EXPECTED_N_CASES:
        _fail(f"build_canonical_table(): final row count is {len(m)}, expected {EXPECTED_N_CASES}.")
    if m["case_id"].nunique() != EXPECTED_N_CASES:
        _fail(
            f"build_canonical_table(): final unique case_id count is "
            f"{m['case_id'].nunique()}, expected {EXPECTED_N_CASES}."
        )
    _require_columns_unique(m, "final canonical table")

    for target in PROFIT_TARGETS:
        target_prob_cols = []
        target_y_cols = []
        for horizon in PROFIT_HORIZONS:
            ycol = f"out__y_hit{target}_h{horizon}"
            pcol = f"pkt__p_hit{target}_h{horizon}"
            out_pcol = f"out__p_hit{target}_h{horizon}"
            for col in (ycol, pcol, out_pcol):
                if col not in m.columns:
                    _fail(f"build_canonical_table(): missing required column '{col}'.")
                if m[col].isna().any():
                    bad = m.loc[m[col].isna(), "case_id"].tolist()
                    _fail(f"build_canonical_table(): column '{col}' has NA for case_ids {bad}.")
            if not m[ycol].isin([0, 1]).all():
                _fail(f"build_canonical_table(): '{ycol}' contains values outside {{0,1}}.")
            if not m[pcol].between(0, 1, inclusive="both").all():
                _fail(f"build_canonical_table(): '{pcol}' contains probabilities outside [0,1].")
            if not np.allclose(m[pcol], m[out_pcol], rtol=0.0, atol=1e-12):
                _fail(f"build_canonical_table(): prepared/outcome probability mismatch for '{pcol}'.")
            target_prob_cols.append(pcol)
            target_y_cols.append(ycol)
        if (m[target_prob_cols].diff(axis=1).iloc[:, 1:] < -1e-12).any().any():
            _fail(
                f"build_canonical_table(): p_hit{target} is not nondecreasing across "
                f"Profit horizons {PROFIT_HORIZONS}."
            )
        if (m[target_y_cols].diff(axis=1).iloc[:, 1:] < 0).any().any():
            _fail(
                f"build_canonical_table(): y_hit{target} is not nondecreasing across "
                f"Profit horizons {PROFIT_HORIZONS}."
            )

    for horizon in PROFIT_HORIZONS:
        ycols = [f"out__y_hit{t}_h{horizon}" for t in PROFIT_TARGETS]
        pcols = [f"pkt__p_hit{t}_h{horizon}" for t in PROFIT_TARGETS]
        if (m[ycols].diff(axis=1).iloc[:, 1:] > 0).any().any():
            _fail(f"build_canonical_table(): target-hit hierarchy is invalid at horizon {horizon}.")
        if (m[pcols].diff(axis=1).iloc[:, 1:] > 1e-12).any().any():
            _fail(f"build_canonical_table(): probability target hierarchy is invalid at horizon {horizon}.")

    for target in PROFIT_TARGETS:
        time_col = f"out__time_to_hit{target}"
        if time_col not in m.columns:
            _fail(f"build_canonical_table(): missing '{time_col}'.")
        time = pd.to_numeric(m[time_col], errors="coerce")
        y120 = m[f"out__y_hit{target}_h120"]
        if ((y120 == 1) & (~time.between(1, 120, inclusive="both"))).any():
            _fail(f"build_canonical_table(): {time_col} is invalid for a 120-session hit.")
        if ((y120 == 0) & time.notna()).any():
            _fail(f"build_canonical_table(): {time_col} is present for a non-hit case.")
        for horizon in PROFIT_HORIZONS:
            expected_hit = time.le(horizon).fillna(False).astype(int)
            if not expected_hit.equals(m[f"out__y_hit{target}_h{horizon}"].astype(int)):
                _fail(
                    f"build_canonical_table(): {time_col} disagrees with "
                    f"out__y_hit{target}_h{horizon}."
                )

    launch_prob_cols = [f"pkt__launch_by_{h}" for h in LAUNCH_WINDOWS]
    for col in launch_prob_cols:
        if col not in m.columns or m[col].isna().any() or not m[col].between(0, 1, inclusive="both").all():
            _fail(f"build_canonical_table(): invalid required launch probability column '{col}'.")
    if (m[launch_prob_cols].diff(axis=1).iloc[:, 1:] < -1e-12).any().any():
        _fail("build_canonical_table(): launch cumulative probabilities are not monotone.")

    return m


# ============================================================================
# 5. Realized launch + path-quality derivation (uses HIDDEN_FUTURE_PATHS)
# ============================================================================


def _trough_before_hit(
    path_for_case: pd.DataFrame, basis_price: float, time_to_hit: int
) -> tuple[int | None, str]:
    """
    Deterministic algorithm (documented in README_AUDIT.txt):

      Given the future OHLC path for a single case (sessions 1..120) and a
      ground-truth 'time_to_hit' session (the session on which the relevant
      profit target was first achieved, as recorded in HIDDEN_OUTCOMES.csv),
      find the "final trough before the successful up-leg":

        1. Form [entry close at session 0, lows at sessions 1..time_to_hit].
        2. trough_session = the LAST occurrence of the minimum price in that
           path.  This exactly matches the already-audited V6.2 launch-label
           implementation; using the first tied trough would systematically
           make launches look earlier than the frozen definition.
        3. If trough_session == time_to_hit (the trough and the hit session
           are the same bar), the launch point is ambiguous -- return
           (None, "AMBIGUOUS_SAME_SESSION").
        4. Otherwise the realized launch session is trough_session + 1 (the
           next trading session after the trough), status "DETERMINED".

    IMPORTANT LIMITATION (also stated in README): 'time_to_hit' is taken
    from HIDDEN_OUTCOMES.csv as an external ground truth and is NOT
    recomputed here; this function only locates a trough using the 'low'
    column of HIDDEN_FUTURE_PATHS.parquet within that externally-defined
    window. Because the two data products may use different price bases
    (e.g. close-based hit detection vs low-based trough detection), this
    realized-launch label is an APPROXIMATE, retrospective diagnostic --
    not a re-derivation of ground truth, and not a claim of an
    independently verified hit definition.
    """
    if time_to_hit is None or pd.isna(time_to_hit):
        return None, "UNAVAILABLE_NO_TIME_TO_HIT"
    if basis_price is None or pd.isna(basis_price) or float(basis_price) <= 0:
        return None, "UNAVAILABLE_NO_BASIS_PRICE"
    time_to_hit = int(time_to_hit)
    if time_to_hit < 1 or time_to_hit > 120:
        return None, "UNAVAILABLE_TIME_TO_HIT_OUT_OF_RANGE"

    window = path_for_case.loc[path_for_case["session"] <= time_to_hit]
    if window.empty:
        return None, "UNAVAILABLE_NO_PATH_DATA"
    if window["session"].max() < time_to_hit:
        return None, "UNAVAILABLE_INCOMPLETE_PATH_DATA"

    window = window.sort_values("session")
    lows = window["low"].to_numpy(dtype=float)
    sessions = window["session"].to_numpy(dtype=int)
    path_prices = np.concatenate(([float(basis_price)], lows))
    path_sessions = np.concatenate(([0], sessions))
    min_low = float(np.min(path_prices))
    tol = max(1e-8, abs(min_low) * 1e-12)
    tied = np.flatnonzero(np.isclose(path_prices, min_low, rtol=1e-10, atol=tol))
    trough_session = int(path_sessions[tied[-1]])

    if trough_session == time_to_hit:
        return None, "AMBIGUOUS_SAME_SESSION"

    return trough_session + 1, "DETERMINED"


def _max_drawdown_before_hit(path_for_case: pd.DataFrame, basis_price: float, time_to_hit: int) -> float | None:
    """
    Path-quality helper (Section 28): the maximum adverse excursion (as a
    negative percentage relative to basis_price) observed in sessions
    [1, time_to_hit] before a target was hit. Uses the 'low' column.
    Returns None if the window/basis is unavailable.
    """
    if time_to_hit is None or pd.isna(time_to_hit) or basis_price is None or pd.isna(basis_price):
        return None
    time_to_hit = int(time_to_hit)
    if time_to_hit < 1 or time_to_hit > 120 or basis_price <= 0:
        return None
    window = path_for_case.loc[path_for_case["session"] <= time_to_hit]
    if window.empty:
        return None
    min_low = window["low"].min()
    return float(min_low / basis_price - 1.0)


def derive_realized_launch(canonical: pd.DataFrame, future_paths: pd.DataFrame) -> pd.DataFrame:
    """
    Adds, per case_id:
        realized_launch_session   (int or NaN)
        realized_launch_status    (str; one of DETERMINED / AMBIGUOUS_SAME_SESSION /
                                    UNAVAILABLE_NO_TIME_TO_HIT / UNAVAILABLE_TIME_TO_HIT_OUT_OF_RANGE /
                                    UNAVAILABLE_NO_PATH_DATA / UNAVAILABLE_INCOMPLETE_PATH_DATA /
                                    CENSORED_NO_HIT10)
        launch_by_{1,3,5,10,20,30}_actual   (1/0/NaN; NaN = censored/undetermined)

    Realized launch is anchored to the +10% target (out__time_to_hit10),
    consistent with "Launch" being defined against the primary +10% success
    criterion elsewhere in this audit. This does not affect the separate
    +20/+30/+50 profit-horizon diagnostics, which are computed independently
    in audit_profit_horizons().
    """
    paths_by_case = {cid: g for cid, g in future_paths.groupby("case_id")}

    sessions: list[int | None] = []
    statuses: list[str] = []

    for _, row in canonical.iterrows():
        cid = row["case_id"]
        y_hit10 = row["out__y_hit10_h120"]
        if pd.isna(y_hit10) or int(y_hit10) != 1:
            sessions.append(None)
            statuses.append("CENSORED_NO_HIT10")
            continue

        path = paths_by_case.get(cid)
        if path is None or path.empty:
            sessions.append(None)
            statuses.append("UNAVAILABLE_NO_PATH_DATA")
            continue

        time_to_hit = row.get("out__time_to_hit10")
        basis_price = row.get("pkt__current_price")
        sess, status = _trough_before_hit(path, basis_price, time_to_hit)
        sessions.append(sess)
        statuses.append(status)

    out = canonical.copy()
    out["realized_launch_session"] = pd.array(sessions, dtype="Int64")
    out["realized_launch_status"] = statuses

    for n in LAUNCH_WINDOWS:
        col = f"launch_by_{n}_actual"
        vals = []
        for sess, status in zip(sessions, statuses):
            if status != "DETERMINED":
                vals.append(np.nan)
            else:
                vals.append(1.0 if sess <= n else 0.0)
        out[col] = vals

    actual_cols = [f"launch_by_{n}_actual" for n in LAUNCH_WINDOWS]
    determined_rows = out["realized_launch_status"] == "DETERMINED"
    if determined_rows.any() and (
        out.loc[determined_rows, actual_cols].diff(axis=1).iloc[:, 1:] < 0
    ).any().any():
        _fail("derive_realized_launch(): realized cumulative launch labels are not monotone.")

    return out


def derive_path_metrics(canonical: pd.DataFrame, future_paths: pd.DataFrame) -> pd.DataFrame:
    """
    Adds, per case_id, best-effort "max drawdown before first hit" metrics
    for each profit target, using out__time_to_hit{target} as the window
    end and pkt__current_price as the basis price. NaN where unavailable
    (target never hit, or path data incomplete). These are diagnostic-only
    path/risk fields (Section 28) and never override the primary
    out__y_hit{target}_h{horizon} success labels.
    """
    paths_by_case = {cid: g for cid, g in future_paths.groupby("case_id")}
    out = canonical.copy()

    for target in PROFIT_TARGETS:
        colname = f"mae_before_hit{target}"
        values = []
        for _, row in out.iterrows():
            cid = row["case_id"]
            time_to_hit = row.get(f"out__time_to_hit{target}")
            basis = row.get("pkt__current_price")
            path = paths_by_case.get(cid)
            if path is None or path.empty:
                values.append(np.nan)
                continue
            values.append(_max_drawdown_before_hit(path, basis, time_to_hit))
        out[colname] = values

    return out


# ============================================================================
# 6. Audit sections
# ============================================================================


def _safe_rate(numer_mask: pd.Series, denom_mask: pd.Series) -> float | None:
    denom = int(denom_mask.sum())
    if denom == 0:
        return None
    return float((numer_mask & denom_mask).sum()) / denom


def _group_profit_stats(df: pd.DataFrame) -> dict:
    stats: dict[str, Any] = {"n": int(len(df))}
    for target in PROFIT_TARGETS:
        for horizon in PROFIT_HORIZONS:
            col = f"out__y_hit{target}_h{horizon}"
            if col in df.columns:
                stats[f"hit{target}_h{horizon}_rate"] = (
                    float(df[col].mean()) if len(df) else None
                )
                time_col = f"out__time_to_hit{target}"
                hit_times = df.loc[df[col] == 1, time_col] if time_col in df.columns else pd.Series(dtype=float)
                stats[f"median_time_to_hit{target}_among_h{horizon}_hits"] = (
                    float(hit_times.median()) if hit_times.notna().any() else None
                )
    for horizon in (20, 40, 60, 120):
        mfe_col = f"out__mfe_h{horizon}"
        mae_col = f"out__mae_h{horizon}"
        if mfe_col in df.columns:
            stats[f"mean_mfe_h{horizon}"] = float(df[mfe_col].mean()) if len(df) else None
            stats[f"median_mfe_h{horizon}"] = float(df[mfe_col].median()) if len(df) else None
        if mae_col in df.columns:
            stats[f"mean_mae_h{horizon}"] = float(df[mae_col].mean()) if len(df) else None
            stats[f"median_mae_h{horizon}"] = float(df[mae_col].median()) if len(df) else None
    return stats


def audit_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """Section A/B: decision counts x profit-horizon performance."""
    rows = []
    for decision in DECISION_ORDER:
        sub = df.loc[df["decision"] == decision]
        row = {"decision": decision}
        row.update(_group_profit_stats(sub))
        rows.append(row)

    actionable = df.loc[df["decision"].isin(ACTIONABLE_DECISIONS)]
    non_actionable = df.loc[df["decision"].isin(NON_ACTIONABLE_DECISIONS)]
    rows.append({"decision": "ACTIONABLE_TOTAL", **_group_profit_stats(actionable)})
    rows.append({"decision": "NON_ACTIONABLE_TOTAL", **_group_profit_stats(non_actionable)})
    rows.append({"decision": "ALL", **_group_profit_stats(df)})

    return pd.DataFrame(rows)


def audit_launch(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Section A (16): launch diagnosis by decision bucket."""
    rows = []
    for decision in DECISION_ORDER:
        sub = df.loc[df["decision"] == decision]
        determined_sub = sub.loc[sub["realized_launch_status"] == "DETERMINED"]
        row: dict[str, Any] = {"decision": decision, "n": int(len(sub))}
        for n in LAUNCH_WINDOWS:
            pred_col = f"pkt__launch_by_{n}"
            row[f"predicted_launch_by{n}_mean_conditional_win120"] = (
                float(determined_sub[pred_col].mean())
                if pred_col in determined_sub.columns and len(determined_sub)
                else None
            )
            row[f"predicted_launch_by{n}_n"] = int(len(determined_sub))
        for n in LAUNCH_WINDOWS:
            actual_col = f"launch_by_{n}_actual"
            if actual_col in sub.columns and sub[actual_col].notna().any():
                row[f"realized_launch_by{n}_rate"] = float(sub[actual_col].mean(skipna=True))
                row[f"realized_launch_by{n}_n_determined"] = int(sub[actual_col].notna().sum())
            else:
                row[f"realized_launch_by{n}_rate"] = None
                row[f"realized_launch_by{n}_n_determined"] = 0
        determined = sub.loc[sub["realized_launch_status"] == "DETERMINED"]
        row["median_realized_launch_session"] = (
            float(determined["realized_launch_session"].median()) if not determined.empty else None
        )
        rows.append(row)

    launch_table = pd.DataFrame(rows)

    summary: dict[str, Any] = {}
    determined_all = df.loc[df["realized_launch_status"] == "DETERMINED"]
    for n in LAUNCH_WINDOWS:
        pred_col = f"pkt__launch_by_{n}"
        actual_col = f"launch_by_{n}_actual"
        summary[f"predicted_launch_by{n}_mean_conditional_win120"] = (
            float(determined_all[pred_col].mean())
            if pred_col in determined_all.columns and len(determined_all)
            else None
        )
        summary[f"predicted_launch_by{n}_n"] = int(len(determined_all))
        if actual_col in df.columns and df[actual_col].notna().any():
            summary[f"realized_launch_by{n}_rate"] = float(df[actual_col].mean(skipna=True))
        else:
            summary[f"realized_launch_by{n}_rate"] = None

    for n in (5, 10, 20, 30):
        actual_col = f"launch_by_{n}_actual"
        mask_missed = (
            df["decision"].isin(NON_ACTIONABLE_DECISIONS)
            & (df[actual_col] == 1.0)
        )
        summary[f"missed_launch_by{n}_count"] = int(mask_missed.sum())

    return launch_table, summary


def audit_false_negatives_launch(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "case_id", "date", "code", "decision", "evidence_quality",
        "realized_launch_session", "realized_launch_status",
        "pkt__launch_by_5", "pkt__launch_by_10", "pkt__launch_by_20", "pkt__launch_by_30",
        "launch_by_5_actual", "launch_by_10_actual", "launch_by_20_actual", "launch_by_30_actual",
        "decision_reason", "bull_thesis", "bear_thesis", "invalidation",
    ]
    cols = [c for c in cols if c in df.columns]
    mask = df["decision"].isin(NON_ACTIONABLE_DECISIONS) & (df["launch_by_30_actual"] == 1.0)
    return df.loc[mask, cols].copy()


def audit_false_negatives_profit(df: pd.DataFrame) -> pd.DataFrame:
    """Union of FN_HIT{target}_{horizon} flags; long-format table."""
    records = []
    for target in PROFIT_TARGETS:
        for horizon in PROFIT_HORIZONS:
            ycol = f"out__y_hit{target}_h{horizon}"
            if ycol not in df.columns:
                continue
            mask = df["decision"].isin(NON_ACTIONABLE_DECISIONS) & (df[ycol] == 1)
            sub = df.loc[mask]
            for _, row in sub.iterrows():
                records.append(
                    {
                        "fn_label": f"FN_HIT{target}_{horizon}",
                        "case_id": row["case_id"],
                        "date": row["date"],
                        "code": row["code"],
                        "decision": row["decision"],
                        "evidence_quality": row.get("evidence_quality"),
                        "pkt__p_hit10_h120": row.get("pkt__p_hit10_h120"),
                        "pkt__p_hit20_h120": row.get("pkt__p_hit20_h120"),
                        "pkt__p_hit30_h120": row.get("pkt__p_hit30_h120"),
                        "pkt__p_hit50_h120": row.get("pkt__p_hit50_h120"),
                        "pkt__launch_by_30": row.get("pkt__launch_by_30"),
                        f"out__y_hit{target}_h{horizon}": row.get(ycol),
                        "out__mfe_h120": row.get("out__mfe_h120"),
                        "out__mae_h120": row.get("out__mae_h120"),
                        f"out__time_to_hit{target}": row.get(f"out__time_to_hit{target}"),
                        "decision_reason": row.get("decision_reason"),
                        "bull_thesis": row.get("bull_thesis"),
                        "bear_thesis": row.get("bear_thesis"),
                        "invalidation": row.get("invalidation"),
                        "primary_evidence_ids": row.get("primary_evidence_ids"),
                        "secondary_evidence_ids": row.get("secondary_evidence_ids"),
                        "counter_evidence_ids": row.get("counter_evidence_ids"),
                    }
                )
    return pd.DataFrame(records)


def audit_fast_success_misses(df: pd.DataFrame) -> pd.DataFrame:
    """Section 19: +10% within 20 or 30 sessions, but AI non-actionable."""
    cols = [
        "case_id", "date", "code", "decision", "evidence_quality",
        "out__y_hit10_h20", "out__y_hit10_h30",
        "out__time_to_hit10", "pkt__p_hit10_h20", "pkt__p_hit10_h30",
        "decision_reason",
    ]
    cols = [c for c in cols if c in df.columns]
    mask = df["decision"].isin(NON_ACTIONABLE_DECISIONS) & (
        (df["out__y_hit10_h20"] == 1) | (df["out__y_hit10_h30"] == 1)
    )
    return df.loc[mask, cols].copy()


def _size_bucket(mfe120: float | None) -> str:
    if mfe120 is None or pd.isna(mfe120):
        return "UNKNOWN"
    if mfe120 < 0.10:
        return "SMALL"
    if mfe120 < 0.20:
        return "BASE_WINNER"
    if mfe120 < 0.30:
        return "MEDIUM_WINNER"
    if mfe120 < 0.50:
        return "LARGE_WINNER"
    return "VERY_LARGE_WINNER"


def audit_missed_winners(df: pd.DataFrame, min_pct: float) -> pd.DataFrame:
    target_map = {0.20: 20, 0.30: 30, 0.50: 50}
    target = target_map[min_pct]
    ycol = f"out__y_hit{target}_h120"
    cols = [
        "case_id", "date", "code", "decision", "evidence_quality",
        "out__mfe_h120", "out__mae_h120", ycol,
        "pkt__p_hit10_h120", f"pkt__p_hit{target}_h120",
        "decision_reason", "bull_thesis", "bear_thesis",
    ]
    cols = [c for c in cols if c in df.columns]
    # Formal target-hit labels are authoritative.  MFE remains a magnitude
    # diagnostic and must not silently redefine the success label.
    mask = (~df["decision"].isin(ACTIONABLE_DECISIONS)) & (df[ycol] == 1)
    return df.loc[mask, cols].copy()


def audit_size_distribution(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["size_bucket"] = tmp["out__mfe_h120"].map(_size_bucket)
    table = (
        tmp.groupby(["size_bucket", "decision"])
        .size()
        .rename("n")
        .reset_index()
    )
    pivot = table.pivot(index="size_bucket", columns="decision", values="n").fillna(0).astype(int)
    for decision in DECISION_ORDER:
        if decision not in pivot.columns:
            pivot[decision] = 0
    pivot = pivot.reindex(SIZE_BUCKET_ORDER + (["UNKNOWN"] if "UNKNOWN" in pivot.index else []))
    pivot = pivot[DECISION_ORDER]
    pivot = pivot.reset_index()
    return pivot


def _quartile_labels(series: pd.Series) -> pd.Series:
    """Return deterministic Q1..Q4 labels, breaking probability ties by row order."""
    if len(series) < 4:
        return pd.Series("INSUFFICIENT_VALUES", index=series.index, dtype="object")
    ranks = series.rank(method="first")
    return pd.Series(
        pd.qcut(ranks, q=4, labels=["Q1", "Q2", "Q3", "Q4"]),
        index=series.index,
        dtype="object",
    )


def audit_probability_quartiles(
    df: pd.DataFrame,
    prob_col: str,
    label: str,
    actual_cols: dict[str, str],
) -> pd.DataFrame:
    """Quartile diagnostics for one prediction and its matching outcomes.

    Diagnostic only -- no threshold is chosen or recommended here.  Equal
    probabilities are deterministically ordered by the canonical case order
    solely to keep four equally sized descriptive bins.
    """
    required_actual = list(actual_cols.values())
    actual_known = df[required_actual].notna().all(axis=1) if required_actual else True
    sub = df.loc[df[prob_col].notna() & actual_known].copy()
    sub = sub.sort_values([prob_col, "case_id"], kind="mergesort")
    sub["quartile"] = _quartile_labels(sub[prob_col])

    rows = []
    for q, g in sub.groupby("quartile", observed=True):
        row: dict[str, Any] = {
            "metric": label,
            "quartile": q,
            "n": int(len(g)),
            "mean_prob": float(g[prob_col].mean()),
        }
        for out_name, actual_col in actual_cols.items():
            if actual_col in g.columns:
                valid = g[actual_col].notna()
                row[f"{out_name}_n"] = int(valid.sum())
                row[f"{out_name}_rate"] = float(g.loc[valid, actual_col].mean()) if valid.any() else None
        for decision in DECISION_ORDER:
            row[f"{decision.lower()}_rate"] = float((g["decision"] == decision).mean())
        row["actionable_rate"] = float(g["decision"].isin(ACTIONABLE_DECISIONS).mean())
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        quartile_order = ["Q1", "Q2", "Q3", "Q4", "INSUFFICIENT_VALUES"]
        out["quartile"] = pd.Categorical(out["quartile"], categories=quartile_order, ordered=True)
        out = out.sort_values("quartile").reset_index(drop=True)
    return out


def audit_evidence_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Section 23: evidence-availability bias, split by winner/loser and decision."""
    tmp = df.copy()

    def _has(row, family):
        avail = row.get("pkt__available_evidence_ids")
        if isinstance(avail, (list, tuple, set)):
            return int(family in avail)
        if isinstance(avail, str):
            return int(family in avail)
        return 0

    for fam in AVAILABLE_EVIDENCE_FAMILIES:
        tmp[f"has_{fam}"] = tmp.apply(lambda r: _has(r, fam), axis=1)

    tmp["available_evidence_count"] = tmp[[f"has_{f}" for f in AVAILABLE_EVIDENCE_FAMILIES]].sum(axis=1)
    tmp["is_winner_hit10_h120"] = tmp["out__y_hit10_h120"] == 1

    groups = {
        "ALL": tmp,
        "WINNER_HIT10": tmp.loc[tmp["is_winner_hit10_h120"]],
        "NON_WINNER_HIT10": tmp.loc[~tmp["is_winner_hit10_h120"]],
        "DECISION_REJECT": tmp.loc[tmp["decision"] == "REJECT"],
        "DECISION_WATCH": tmp.loc[tmp["decision"] == "WATCH"],
        "DECISION_ACTIONABLE": tmp.loc[tmp["decision"].isin(ACTIONABLE_DECISIONS)],
    }
    if "launch_by_30_actual" in tmp.columns:
        groups["EARLY_LAUNCH_BY30"] = tmp.loc[tmp["launch_by_30_actual"] == 1.0]
        groups["NO_LAUNCH_OR_LATE"] = tmp.loc[tmp["launch_by_30_actual"] != 1.0]

    rows = []
    for gname, g in groups.items():
        if g.empty:
            row = {"group": gname, "n": 0}
        else:
            row = {
                "group": gname,
                "n": int(len(g)),
                "mean_available_evidence_count": float(g["available_evidence_count"].mean()),
            }
            for fam in AVAILABLE_EVIDENCE_FAMILIES:
                row[f"pct_has_{fam}"] = float(g[f"has_{fam}"].mean())
        rows.append(row)

    return pd.DataFrame(rows)


def _reason_text(row: pd.Series) -> str:
    parts = []
    for col in ("decision_reason", "bear_thesis", "invalidation", "hypothesis"):
        v = row.get(col)
        if isinstance(v, str):
            parts.append(v)
    return " \n ".join(parts)


def audit_reason_language(df: pd.DataFrame) -> pd.DataFrame:
    """Section 24: deterministic keyword/phrase audit (no LLM)."""
    texts = df.apply(_reason_text, axis=1)

    rows = []
    for name, pattern in _COMPILED_REASON_PATTERNS.items():
        hit_mask = texts.str.contains(pattern, regex=True, na=False)
        n_hit = int(hit_mask.sum())
        row: dict[str, Any] = {
            "pattern": name,
            "case_count": n_hit,
            "reject_rate": _safe_rate(df["decision"] == "REJECT", hit_mask),
            "watch_rate": _safe_rate(df["decision"] == "WATCH", hit_mask),
        }
        if "launch_by_30_actual" in df.columns:
            sub = df.loc[hit_mask]
            row["launch_by30_actual_rate"] = (
                float(sub["launch_by_30_actual"].mean(skipna=True))
                if sub["launch_by_30_actual"].notna().any()
                else None
            )
        for target, horizon in ((10, 20), (10, 30), (10, 60), (10, 120), (20, 120), (30, 120)):
            ycol = f"out__y_hit{target}_h{horizon}"
            if ycol in df.columns:
                sub = df.loc[hit_mask]
                row[f"hit{target}_h{horizon}_rate"] = (
                    float(sub[ycol].mean()) if len(sub) else None
                )
        rows.append(row)

    return pd.DataFrame(rows)


def audit_evidence_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Section 25."""
    rows = []
    for eq in EVIDENCE_QUALITY_ORDER:
        sub = df.loc[df["evidence_quality"] == eq]
        row: dict[str, Any] = {"evidence_quality": eq, "n": int(len(sub))}
        for decision in DECISION_ORDER:
            row[f"{decision.lower()}_rate"] = (
                float((sub["decision"] == decision).mean()) if len(sub) else None
            )
        if "pkt__launch_by_30" in sub.columns:
            launch_valid = sub.loc[sub["launch_by_30_actual"].notna()]
            row["mean_predicted_launch_by30_conditional_win120"] = (
                float(launch_valid["pkt__launch_by_30"].mean()) if len(launch_valid) else None
            )
        if "launch_by_30_actual" in sub.columns:
            row["realized_launch_by30_rate"] = (
                float(sub["launch_by_30_actual"].mean(skipna=True))
                if sub["launch_by_30_actual"].notna().any()
                else None
            )
        for target, horizon in ((10, 20), (10, 30), (10, 60), (10, 120), (20, 120), (30, 120)):
            ycol = f"out__y_hit{target}_h{horizon}"
            if ycol in sub.columns:
                row[f"hit{target}_h{horizon}_rate"] = float(sub[ycol].mean()) if len(sub) else None
        for horizon in (20, 60, 120):
            mfe_col, mae_col = f"out__mfe_h{horizon}", f"out__mae_h{horizon}"
            if mfe_col in sub.columns:
                row[f"mean_mfe_h{horizon}"] = float(sub[mfe_col].mean()) if len(sub) else None
            if mae_col in sub.columns:
                row[f"mean_mae_h{horizon}"] = float(sub[mae_col].mean()) if len(sub) else None
        rows.append(row)

    out = pd.DataFrame(rows)
    strong = out.loc[out["evidence_quality"] == "STRONG"]
    if not strong.empty:
        n = strong["n"].iloc[0]
        reject_rate = strong["reject_rate"].iloc[0]
        if n > 0 and reject_rate is not None and reject_rate >= 0.95:
            out.attrs["strong_evidence_still_rejected"] = True
        else:
            out.attrs["strong_evidence_still_rejected"] = False
    return out


def audit_hypothesis_types(df: pd.DataFrame) -> pd.DataFrame:
    """Section 26."""
    rows = []
    for htype, sub in df.groupby(df["hypothesis_type"].fillna("UNSPECIFIED")):
        row: dict[str, Any] = {"hypothesis_type": htype, "n": int(len(sub))}
        row["reject_rate"] = float((sub["decision"] == "REJECT").mean())
        row["watch_rate"] = float((sub["decision"] == "WATCH").mean())
        row["actionable_rate"] = float(sub["decision"].isin(ACTIONABLE_DECISIONS).mean())
        if "launch_by_30_actual" in sub.columns:
            row["actual_launch_by30_rate"] = (
                float(sub["launch_by_30_actual"].mean(skipna=True))
                if sub["launch_by_30_actual"].notna().any()
                else None
            )
        for target, horizon in ((10, 20), (10, 30), (10, 60), (10, 120), (20, 120), (30, 120)):
            ycol = f"out__y_hit{target}_h{horizon}"
            if ycol in sub.columns:
                row[f"hit{target}_h{horizon}_rate"] = float(sub[ycol].mean())
        if "out__mfe_h120" in sub.columns:
            row["mean_mfe_h120"] = float(sub["out__mfe_h120"].mean())
        if "out__mae_h120" in sub.columns:
            row["median_mae_h120"] = float(sub["out__mae_h120"].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def audit_calibration_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Full calibration grid for every Profit and Launch horizon.

    Residual is actual_rate - mean_prediction.  Launch calibration is
    conditional on cases with a determined realized launch label, matching
    the frozen launch head's conditioning on eventual +10% winners by 120.
    """
    records: list[dict[str, Any]] = []
    metrics: list[tuple[str, int, int | None, str, str]] = []
    for target in PROFIT_TARGETS:
        for horizon in PROFIT_HORIZONS:
            metrics.append(
                ("PROFIT", target, horizon, f"pkt__p_hit{target}_h{horizon}", f"out__y_hit{target}_h{horizon}")
            )
    for horizon in LAUNCH_WINDOWS:
        metrics.append(
            ("LAUNCH_CONDITIONAL_WIN120", horizon, None, f"pkt__launch_by_{horizon}", f"launch_by_{horizon}_actual")
        )

    for metric_type, target_or_window, horizon, pcol, ycol in metrics:
        if pcol not in df.columns or ycol not in df.columns:
            continue
        tmp = df.loc[df[pcol].notna() & df[ycol].notna()].copy()
        if tmp.empty:
            continue
        tmp["prediction_quartile"] = _quartile_labels(tmp[pcol])
        groups: list[tuple[str, str, pd.DataFrame]] = [("ALL", "ALL", tmp)]
        groups.extend(("decision", v, tmp.loc[tmp["decision"] == v]) for v in DECISION_ORDER)
        groups.extend(
            ("evidence_quality", v, tmp.loc[tmp["evidence_quality"] == v])
            for v in EVIDENCE_QUALITY_ORDER
        )
        groups.extend(
            ("prediction_quartile", v, tmp.loc[tmp["prediction_quartile"] == v])
            for v in ("Q1", "Q2", "Q3", "Q4")
        )
        for slice_column, slice_value, sub in groups:
            if sub.empty:
                continue
            pred = pd.to_numeric(sub[pcol], errors="raise")
            actual = pd.to_numeric(sub[ycol], errors="raise")
            records.append(
                {
                    "metric_type": metric_type,
                    "profit_target_pct_or_launch_window": target_or_window,
                    "profit_horizon": horizon,
                    "prediction_column": pcol,
                    "actual_column": ycol,
                    "slice_column": slice_column,
                    "slice_value": slice_value,
                    "n": int(len(sub)),
                    "mean_prediction": float(pred.mean()),
                    "actual_rate": float(actual.mean()),
                    "calibration_residual_actual_minus_pred": float((actual - pred).mean()),
                    "brier_score": float(((actual - pred) ** 2).mean()),
                }
            )

    return pd.DataFrame(records)


def audit_path_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Section 28 summary table (per-target mean/median MAE-before-hit)."""
    rows = []
    for target in PROFIT_TARGETS:
        col = f"mae_before_hit{target}"
        ycol = f"out__y_hit{target}_h120"
        if col not in df.columns:
            continue
        sub = df.loc[(df[ycol] == 1) & df[col].notna()] if ycol in df.columns else df.loc[df[col].notna()]
        rows.append(
            {
                "target_pct": target,
                "n_with_hit_and_path": int(len(sub)),
                "mean_mae_before_hit": float(sub[col].mean()) if len(sub) else None,
                "median_mae_before_hit": float(sub[col].median()) if len(sub) else None,
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# 7. Case-level export + scientific diagnosis
# ============================================================================


def build_case_level_export(df: pd.DataFrame) -> pd.DataFrame:
    preferred_first = [
        "case_id", "date", "code", "pkt__validation_stratum",
        "decision", "evidence_quality", "hypothesis_type",
        "pkt__p_hit10_h120", "pkt__p_hit20_h120", "pkt__p_hit30_h120", "pkt__p_hit50_h120",
        "pkt__launch_by_5", "pkt__launch_by_10", "pkt__launch_by_20", "pkt__launch_by_30",
        "realized_launch_session", "realized_launch_status",
        "launch_by_5_actual", "launch_by_10_actual", "launch_by_20_actual", "launch_by_30_actual",
        "out__y_hit10_h120", "out__y_hit20_h120", "out__y_hit30_h120", "out__y_hit50_h120",
        "out__mfe_h120", "out__mae_h120",
        "out__time_to_hit10", "out__time_to_hit20", "out__time_to_hit30", "out__time_to_hit50",
        "decision_reason", "bull_thesis", "bear_thesis", "invalidation",
    ]
    ordered = [c for c in preferred_first if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining].sort_values("case_id").reset_index(drop=True)


def compute_scientific_diagnosis(
    df: pd.DataFrame,
    launch_summary: dict,
    evidence_quality_table: pd.DataFrame,
    p10_quartile_table: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Returns (diagnosis_labels, evidence_lines). Every label emitted must be
    backed by a concrete evidence_line derived from already-computed audit
    numbers -- no free-form guessing.
    """
    labels: list[str] = []
    evidence: list[str] = []

    n = len(df)
    actionable_rate = float(df["decision"].isin(ACTIONABLE_DECISIONS).mean())
    hit10_h120_rate = float(df["out__y_hit10_h120"].mean())

    if actionable_rate <= 0.05 and hit10_h120_rate >= 0.30:
        labels.append("DECISION_COLLAPSE_CONFIRMED")
        evidence.append(
            f"actionable_rate={actionable_rate:.3f} over n={n} cases while "
            f"actual_hit10_h120_rate={hit10_h120_rate:.3f}."
        )

    missed_early = launch_summary.get("missed_launch_by30_count", 0)
    n_launch30_actual_1 = int((df["launch_by_30_actual"] == 1.0).sum())
    if n_launch30_actual_1 > 0 and missed_early / max(n_launch30_actual_1, 1) >= 0.5:
        labels.append("EARLY_LAUNCH_OPPORTUNITIES_REJECTED")
        evidence.append(
            f"missed_launch_by30_count={missed_early} out of "
            f"{n_launch30_actual_1} cases with realized launch_by_30_actual=1."
        )

    fast_success_mask = (df["out__y_hit10_h20"] == 1) | (df["out__y_hit10_h30"] == 1)
    n_fast = int(fast_success_mask.sum())
    n_fast_missed = int((fast_success_mask & df["decision"].isin(NON_ACTIONABLE_DECISIONS)).sum())
    if n_fast > 0 and n_fast_missed / n_fast >= 0.5:
        labels.append("SHORT_HORIZON_WINNERS_REJECTED")
        evidence.append(
            f"{n_fast_missed} of {n_fast} cases with a +10% hit within 20-30 "
            f"sessions were nonetheless REJECT/WATCH."
        )

    q4 = p10_quartile_table.loc[p10_quartile_table["quartile"] == "Q4"]
    if not q4.empty:
        q4_actionable = q4["actionable_rate"].iloc[0]
        q4_actual_hit10 = q4["actual_hit10_rate"].iloc[0] if "actual_hit10_rate" in q4.columns else None
        if q4_actionable is not None and q4_actual_hit10 is not None:
            if q4_actionable <= 0.10 and q4_actual_hit10 >= 0.30:
                labels.append("NUMERICAL_SIGNAL_IGNORED")
                evidence.append(
                    f"Top p_hit10_h120 quartile (Q4): actionable_rate={q4_actionable:.3f}, "
                    f"actual_hit10_rate={q4_actual_hit10:.3f}."
                )

    strong_row = evidence_quality_table.loc[evidence_quality_table["evidence_quality"] == "STRONG"]
    if not strong_row.empty:
        n_strong = strong_row["n"].iloc[0]
        strong_reject_rate = strong_row["reject_rate"].iloc[0]
        strong_actionable_rate = (
            strong_row["candidate_rate"].iloc[0] + strong_row["high_conviction_rate"].iloc[0]
        )
        if n_strong and n_strong > 0 and strong_actionable_rate is not None and strong_actionable_rate <= 0.10:
            labels.append("STRONG_EVIDENCE_DID_NOT_CREATE_ACTIONABILITY")
            evidence.append(
                f"evidence_quality=STRONG: n={n_strong}, reject_rate={strong_reject_rate:.3f}, "
                f"actionable_rate={strong_actionable_rate:.3f}."
            )

    large_winner_mask = df["out__y_hit30_h120"] == 1
    large_winner_missed = large_winner_mask & (~df["decision"].isin(ACTIONABLE_DECISIONS))
    n_large_winner = int(large_winner_mask.sum())
    n_large_missed = int(large_winner_missed.sum())
    if n_large_winner > 0 and n_large_missed / n_large_winner >= 0.5:
        labels.append("LARGE_WINNER_BLINDNESS")
        evidence.append(
            f"{n_large_missed} of {n_large_winner} cases with formal y_hit30_h120=1 "
            f"were non-actionable."
        )

    reject_texts = df.loc[df["decision"] == "REJECT"].apply(_reason_text, axis=1)
    if len(reject_texts) > 0:
        caution_hits = (
            reject_texts.str.contains(_COMPILED_REASON_PATTERNS["missing"], na=False)
            | reject_texts.str.contains(_COMPILED_REASON_PATTERNS["insufficient"], na=False)
            | reject_texts.str.contains(_COMPILED_REASON_PATTERNS["unavailable"], na=False)
            | reject_texts.str.contains(_COMPILED_REASON_PATTERNS["lack"], na=False)
        )
        caution_share = float(caution_hits.mean())
        if caution_share >= 0.5:
            labels.append("OVER_CAUTION_ON_MISSING_EVIDENCE")
            evidence.append(
                f"{caution_share:.1%} of REJECT decision_reason/bear_thesis/invalidation/"
                f"hypothesis text contains missing/unavailable-evidence language."
            )

    if not labels:
        labels.append("MIXED")
        evidence.append(
            "No single dominant pattern crossed the diagnostic thresholds used above; "
            "see the per-section audit tables for the full breakdown."
        )

    return labels, evidence


# ============================================================================
# 8. Output writing
# ============================================================================


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if pd.isna(o):
        return None
    raise TypeError(f"Object of type {type(o)} is not JSON serializable: {o!r}")


def write_outputs(
    canonical: pd.DataFrame,
    decision_performance: pd.DataFrame,
    launch_table: pd.DataFrame,
    launch_summary: dict,
    p10_quartile_table: pd.DataFrame,
    launch30_quartile_table: pd.DataFrame,
    fn_launch: pd.DataFrame,
    fn_profit: pd.DataFrame,
    fast_success_misses: pd.DataFrame,
    missed20: pd.DataFrame,
    missed30: pd.DataFrame,
    missed50: pd.DataFrame,
    evidence_availability: pd.DataFrame,
    reason_language: pd.DataFrame,
    evidence_quality_table: pd.DataFrame,
    hypothesis_table: pd.DataFrame,
    calibration_residuals: pd.DataFrame,
    path_quality: pd.DataFrame,
    size_distribution: pd.DataFrame,
    diagnosis_labels: list[str],
    diagnosis_evidence: list[str],
    prep_summary: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_level = build_case_level_export(canonical)
    case_level.to_csv(OUTPUT_DIR / "02_CASE_LEVEL_AUDIT.csv", index=False)
    case_level.to_json(
        OUTPUT_DIR / "02_CASE_LEVEL_AUDIT.json", orient="records", force_ascii=False, indent=2,
        default_handler=_json_default,
    )

    decision_performance.to_csv(OUTPUT_DIR / "03_DECISION_PERFORMANCE_BY_HORIZON.csv", index=False)
    launch_table.to_csv(OUTPUT_DIR / "04_LAUNCH_AUDIT.csv", index=False)
    launch30_quartile_table.to_csv(OUTPUT_DIR / "05_LAUNCH_PROBABILITY_QUARTILES.csv", index=False)
    p10_quartile_table.to_csv(OUTPUT_DIR / "06_PROFIT_PROBABILITY_QUARTILES.csv", index=False)
    fn_launch.to_csv(OUTPUT_DIR / "07_FALSE_NEGATIVES_LAUNCH.csv", index=False)
    fn_profit.to_csv(OUTPUT_DIR / "08_FALSE_NEGATIVES_PROFIT.csv", index=False)
    fast_success_misses.to_csv(OUTPUT_DIR / "09_FAST_SUCCESS_MISSES.csv", index=False)
    missed20.to_csv(OUTPUT_DIR / "10_MISSED_20PCT_WINNERS.csv", index=False)
    missed30.to_csv(OUTPUT_DIR / "11_MISSED_30PCT_WINNERS.csv", index=False)
    missed50.to_csv(OUTPUT_DIR / "12_MISSED_50PCT_WINNERS.csv", index=False)
    evidence_availability.to_csv(OUTPUT_DIR / "13_EVIDENCE_AVAILABILITY_AUDIT.csv", index=False)
    reason_language.to_csv(OUTPUT_DIR / "14_REASON_LANGUAGE_AUDIT.csv", index=False)
    evidence_quality_table.to_csv(OUTPUT_DIR / "15_EVIDENCE_QUALITY_AUDIT.csv", index=False)
    hypothesis_table.to_csv(OUTPUT_DIR / "16_HYPOTHESIS_TYPE_AUDIT.csv", index=False)
    calibration_residuals.to_csv(OUTPUT_DIR / "17_CALIBRATION_RESIDUAL_AUDIT.csv", index=False)
    path_quality.to_csv(OUTPUT_DIR / "18_PATH_QUALITY_AUDIT.csv", index=False)
    size_distribution.to_csv(OUTPUT_DIR / "20_SIZE_DISTRIBUTION_AUDIT.csv", index=False)

    decision_counts = canonical["decision"].value_counts().reindex(DECISION_ORDER, fill_value=0).to_dict()
    actionable_count = int(canonical["decision"].isin(ACTIONABLE_DECISIONS).sum())

    summary = {
        "status": "COMPLETE",
        "source_provenance": {
            "run_id": SOURCE_RUN_ID,
            "head_sha": SOURCE_HEAD_SHA,
            "prepared_artifact_id": PREPARED_ARTIFACT_ID,
            "response_artifact_ids": list(RESPONSE_ARTIFACT_IDS),
        },
        "n_cases": int(len(canonical)),
        "decision_counts": {k: int(v) for k, v in decision_counts.items()},
        "actionable_count": actionable_count,
        "actionable_rate": actionable_count / len(canonical),
        "launch": launch_summary,
        "profit": {
            f"actual_hit{target}_h{horizon}_rate": float(canonical[f"out__y_hit{target}_h{horizon}"].mean())
            for target in PROFIT_TARGETS
            for horizon in PROFIT_HORIZONS
            if f"out__y_hit{target}_h{horizon}" in canonical.columns
        },
        "misses": {
            f"missed_hit{target}_h{horizon}_count": int(
                (
                    (canonical[f"out__y_hit{target}_h{horizon}"] == 1)
                    & (~canonical["decision"].isin(ACTIONABLE_DECISIONS))
                ).sum()
            )
            for target in PROFIT_TARGETS
            for horizon in PROFIT_HORIZONS
            if f"out__y_hit{target}_h{horizon}" in canonical.columns
        },
        "path": {
            f"median_mae_h{h}": (
                float(canonical[f"out__mae_h{h}"].median()) if f"out__mae_h{h}" in canonical.columns else None
            )
            for h in (20, 60, 120)
        }
        | {
            f"mean_mfe_h{h}": (
                float(canonical[f"out__mfe_h{h}"].mean()) if f"out__mfe_h{h}" in canonical.columns else None
            )
            for h in (20, 60, 120)
        },
        "q4_p_hit10_h120": (
            p10_quartile_table.loc[p10_quartile_table["quartile"] == "Q4"].iloc[0].to_dict()
            if (p10_quartile_table["quartile"] == "Q4").any()
            else None
        ),
        "evidence_quality_summary": evidence_quality_table.to_dict(orient="records"),
        "reason_pattern_summary": reason_language.to_dict(orient="records"),
        "scientific_diagnosis": diagnosis_labels,
        "scientific_diagnosis_evidence": diagnosis_evidence,
        "2025_opened": prep_summary.get("2025_opened", None),
        "data_integrity_pass": True,
    }

    with open(OUTPUT_DIR / "01_AUDIT_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

    integrity_report = {
        "source_run_id": SOURCE_RUN_ID,
        "source_head_sha": SOURCE_HEAD_SHA,
        "prepared_artifact_id": PREPARED_ARTIFACT_ID,
        "response_artifact_ids": list(RESPONSE_ARTIFACT_IDS),
        "n_cases": int(len(canonical)),
        "expected_n_cases": EXPECTED_N_CASES,
        "case_id_unique": bool(canonical["case_id"].nunique() == EXPECTED_N_CASES),
        "columns_unique": bool(canonical.columns.is_unique),
        "decision_date_cutoff": DECISION_DATE_CUTOFF,
        "max_decision_date_seen": int(canonical["date"].max()),
        "2025_opened": prep_summary.get("2025_opened", None),
        "required_outcome_columns_non_na": {
            f"out__y_hit{t}_h{h}": bool(canonical[f"out__y_hit{t}_h{h}"].notna().all())
            for t in PROFIT_TARGETS
            for h in PROFIT_HORIZONS
        },
        "response_probability_immutable": bool(
            np.allclose(
                canonical["p_hit10_h120"],
                canonical["pkt__p_hit10_h120"],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "response_evidence_ids_subset_of_available": True,
        "realized_launch_status_counts": canonical["realized_launch_status"].value_counts().to_dict(),
    }
    with open(OUTPUT_DIR / "19_DATA_INTEGRITY_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(integrity_report, f, ensure_ascii=False, indent=2, default=_json_default)

    readme = _build_readme(summary, integrity_report, diagnosis_labels, diagnosis_evidence)
    with open(OUTPUT_DIR / "README_AUDIT.txt", "w", encoding="utf-8") as f:
        f.write(readme)


def _build_readme(summary: dict, integrity: dict, labels: list[str], evidence: list[str]) -> str:
    lines = []
    lines.append("AlphaPilot V6.2 -- Decision Collapse Audit (2024, 64-case replay)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        f"Source: Run {SOURCE_RUN_ID}, prepared artifact {PREPARED_ARTIFACT_ID}, "
        f"response artifacts {list(RESPONSE_ARTIFACT_IDS)}."
    )
    lines.append(f"Source head SHA: {SOURCE_HEAD_SHA}.")
    lines.append("")
    lines.append("1. LAUNCH vs PROFIT are different concepts and are audited separately.")
    lines.append("   LAUNCH = did the stock begin a successful up-leg, observed over a")
    lines.append("   1/3/5/10/20/30 trading-session window after the decision date.")
    lines.append("   PROFIT = did price reach +10/+20/+30/+50%, observed over")
    lines.append("   20/30/40/60/120 trading-session horizons.")
    lines.append("")
    lines.append("2. The formal LAUNCH observation window is 1-30 trading sessions.")
    lines.append("")
    lines.append("3. 30 trading sessions is NOT a forced time-stop and NOT a holding limit.")
    lines.append("")
    lines.append("4. 120 trading sessions is one of several PROFIT outcome-validation")
    lines.append("   horizons (alongside 20/30/40/60). It is not a holding period and")
    lines.append("   is not, by itself, the sole criterion for whether a decision was")
    lines.append("   'correct'.")
    lines.append("")
    lines.append("5. All of +10% / +20% / +30% / +50% are analyzed, at every horizon")
    lines.append("   where the corresponding outcome column exists.")
    lines.append("")
    lines.append("6. Interim MAE (maximum adverse excursion) never re-labels an eventual")
    lines.append("   winner as a failure. A case that fell -8% and later rose +50% is")
    lines.append("   recorded as a LARGE_WINNER; the -8% is reported separately as a")
    lines.append("   path-quality / risk diagnostic (see 18_PATH_QUALITY_AUDIT.csv and the")
    lines.append("   mae_before_hit{target} columns in the case-level export).")
    lines.append("")
    lines.append("7. This 64-case batch is an architecture / decision-behavior diagnostic.")
    lines.append("   It is NOT a statistically powered validation of AlphaPilot V6.2, and")
    lines.append("   no threshold, cutoff, or selection rule may be derived from it.")
    lines.append("")
    lines.append(f"8. 2025 data: 2025_opened = {integrity.get('2025_opened')!r}. This run does")
    lines.append("   not read, reference, or depend on any 2025 data.")
    lines.append("")
    lines.append("-" * 72)
    lines.append("REALIZED LAUNCH -- deterministic algorithm and its limitation")
    lines.append("-" * 72)
    lines.append("Realized launch is only defined for cases where out__y_hit10_h120 == 1")
    lines.append("(anchored to the +10% target, consistent with 'launch' elsewhere in this")
    lines.append("audit). Given the externally-provided out__time_to_hit10 session and the")
    lines.append("HIDDEN_FUTURE_PATHS.parquet OHLC path for that case:")
    lines.append("  1. Build the path from entry close at session 0 through daily lows at")
    lines.append("     sessions [1, time_to_hit10].")
    lines.append("  2. trough_session is the LAST occurrence of the minimum path price.")
    lines.append("  3. If trough_session == time_to_hit10, the launch point is ambiguous")
    lines.append("     (status AMBIGUOUS_SAME_SESSION) and is left undetermined.")
    lines.append("  4. Otherwise realized_launch_session = trough_session + 1, status")
    lines.append("     DETERMINED.")
    lines.append("LIMITATION: out__time_to_hit10 is taken as given, external ground truth")
    lines.append("and is not re-derived here; the trough itself uses entry close plus the")
    lines.append("future daily-low path. Because the two data products may use different")
    lines.append("price bases for their definitions (e.g. high-based hit vs low-based")
    lines.append("trough detection), realized_launch_session/realized_launch_status is an")
    lines.append("APPROXIMATE, retrospective diagnostic -- not an independently re-verified")
    lines.append("ground-truth label. All realized_launch_status values other than")
    lines.append("DETERMINED are reported explicitly rather than silently treated as zero.")
    lines.append("")
    lines.append("-" * 72)
    lines.append("SCIENTIFIC DIAGNOSIS")
    lines.append("-" * 72)
    for label, ev in zip(labels, evidence):
        lines.append(f"  [{label}]")
        lines.append(f"    evidence: {ev}")
    lines.append("")
    lines.append("Every diagnosis label above is backed by a specific number computed from")
    lines.append("this run's own audit tables (see the 'evidence' line under each label).")
    lines.append("No label was assigned by inspection or judgement outside the computed")
    lines.append("audit statistics.")
    lines.append("")
    lines.append("-" * 72)
    lines.append("INTEGRITY ASSERTIONS THAT PASSED FOR THIS RUN")
    lines.append("-" * 72)
    for k, v in integrity.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("-" * 72)
    lines.append("EXPLICITLY NOT DONE IN THIS RUN")
    lines.append("-" * 72)
    lines.append("  - No threshold or cutoff (p_hit10, launch probability, or otherwise)")
    lines.append("    was searched for or recommended.")
    lines.append("  - No AI decision was modified.")
    lines.append("  - No LLM was invoked.")
    lines.append("  - No 2025 data was opened or read.")
    lines.append("  - No frozen 0A-0F layer, R10 logic, or model weights were touched.")
    lines.append("  - No duplicate columns were silently dropped; any such condition")
    lines.append("    raises RuntimeError and aborts the run.")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# 9. main
# ============================================================================


def main() -> None:
    print("[1/9] Loading PREP_SUMMARY.json and validating seal status ...")
    prep_summary = load_prep_summary()

    print("[2/9] Loading prepared case packets ...")
    packets = load_packets()

    print("[3/9] Loading AI responses ...")
    responses = load_responses()

    print("[4/9] Loading hidden outcomes ...")
    outcomes = load_outcomes()

    print("[5/9] Loading case manifest and future paths ...")
    manifest = load_case_manifest()
    future_paths = load_future_paths()

    print("[6/9] Validating sources (row counts, case_id sets) ...")
    validate_sources(packets, responses, outcomes, manifest)

    print("[7/9] Building canonical table (single merge chain, namespaced) ...")
    canonical = build_canonical_table(packets, responses, outcomes)
    canonical = derive_realized_launch(canonical, future_paths)
    canonical = derive_path_metrics(canonical, future_paths)
    _require_columns_unique(canonical, "final canonical table (post-derivation)")

    print("[8/9] Running audit sections ...")
    decision_performance = audit_decisions(canonical)
    launch_table, launch_summary = audit_launch(canonical)
    p10_quartile_table = audit_probability_quartiles(
        canonical,
        "pkt__p_hit10_h120",
        "p_hit10_h120",
        {"actual_hit10": "out__y_hit10_h120"},
    )
    launch30_quartile_table = audit_probability_quartiles(
        canonical,
        "pkt__launch_by_30",
        "launch_by_30_conditional_win120",
        {"actual_launch_by30": "launch_by_30_actual"},
    )
    fn_launch = audit_false_negatives_launch(canonical)
    fn_profit = audit_false_negatives_profit(canonical)
    fast_success_misses = audit_fast_success_misses(canonical)
    missed20 = audit_missed_winners(canonical, 0.20)
    missed30 = audit_missed_winners(canonical, 0.30)
    missed50 = audit_missed_winners(canonical, 0.50)
    evidence_availability = audit_evidence_availability(canonical)
    reason_language = audit_reason_language(canonical)
    evidence_quality_table = audit_evidence_quality(canonical)
    hypothesis_table = audit_hypothesis_types(canonical)
    calibration_residuals = audit_calibration_residuals(canonical)
    path_quality = audit_path_quality(canonical)
    size_distribution = audit_size_distribution(canonical)

    diagnosis_labels, diagnosis_evidence = compute_scientific_diagnosis(
        canonical, launch_summary, evidence_quality_table, p10_quartile_table
    )

    print("[9/9] Writing outputs ...")
    write_outputs(
        canonical=canonical,
        decision_performance=decision_performance,
        launch_table=launch_table,
        launch_summary=launch_summary,
        p10_quartile_table=p10_quartile_table,
        launch30_quartile_table=launch30_quartile_table,
        fn_launch=fn_launch,
        fn_profit=fn_profit,
        fast_success_misses=fast_success_misses,
        missed20=missed20,
        missed30=missed30,
        missed50=missed50,
        evidence_availability=evidence_availability,
        reason_language=reason_language,
        evidence_quality_table=evidence_quality_table,
        hypothesis_table=hypothesis_table,
        calibration_residuals=calibration_residuals,
        path_quality=path_quality,
        size_distribution=size_distribution,
        diagnosis_labels=diagnosis_labels,
        diagnosis_evidence=diagnosis_evidence,
        prep_summary=prep_summary,
    )

    print(f"Done. Outputs written to: {OUTPUT_DIR}")
    print(f"Scientific diagnosis: {diagnosis_labels}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"AUDIT ABORTED: {e}", file=sys.stderr)
        sys.exit(1)
