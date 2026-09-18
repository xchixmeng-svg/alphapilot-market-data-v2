#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v6_2_institutional_pit"

SOURCES = {
    "TWSE": OUT / "twse_institutional_pit_2020_2024.parquet",
    "TPEX": OUT / "tpex_institutional_pit_2020_2024.parquet",
}
AUDITS = {
    "TWSE": OUT / "TWSE_PIT_AUDIT.json",
    "TPEX": OUT / "TPEX_PIT_AUDIT.json",
}
REQUIRED = ["date", "market", "code", "available_session", "volume", "foreign_net", "trust_net", "dealer_net", "foreign_net_ratio", "trust_net_ratio", "dealer_net_ratio"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def audit_market(market: str) -> dict:
    p = SOURCES[market]
    a = AUDITS[market]
    if not p.exists() or not a.exists():
        raise RuntimeError(f"missing institutional inputs for {market}")
    prior = json.loads(a.read_text(encoding="utf-8"))
    if prior.get("status") != "PASS":
        raise RuntimeError(f"upstream PIT audit not PASS for {market}: {prior.get('status')}")
    z = pd.read_parquet(p)
    missing_cols = [c for c in REQUIRED if c not in z.columns]
    if missing_cols:
        raise RuntimeError(f"{market} missing columns: {missing_cols}")
    dup = int(z.duplicated(["date", "market", "code"], keep=False).sum())
    same_day = int((pd.to_numeric(z["available_session"], errors="coerce") <= pd.to_numeric(z["date"], errors="coerce")).sum())
    missing_avail = int(z["available_session"].isna().sum())
    market_bad = int((z["market"].astype(str).str.upper() != market).sum())
    ratios = {}
    for raw, ratio in [("foreign_net", "foreign_net_ratio"), ("trust_net", "trust_net_ratio"), ("dealer_net", "dealer_net_ratio")]:
        valid = z["volume"].notna() & (pd.to_numeric(z["volume"], errors="coerce") > 0) & z[raw].notna()
        expected = pd.to_numeric(z.loc[valid, raw], errors="coerce") / pd.to_numeric(z.loc[valid, "volume"], errors="coerce")
        actual = pd.to_numeric(z.loc[valid, ratio], errors="coerce")
        mismatch = int(((expected - actual).abs() > 1e-12).sum())
        ratios[ratio] = {"eligible_rows": int(valid.sum()), "nonnull_rows": int(z[ratio].notna().sum()), "formula_mismatch": mismatch}
    status = "PASS" if dup == 0 and same_day == 0 and missing_avail == 0 and market_bad == 0 and all(x["formula_mismatch"] == 0 for x in ratios.values()) else "FAIL"
    return {
        "market": market,
        "rows": int(len(z)),
        "sha256": sha256(p),
        "duplicate_key_violations": dup,
        "same_day_or_future_visibility_violations": same_day,
        "missing_available_session": missing_avail,
        "market_label_violations": market_bad,
        "ratio_formula": "net shares / same-day volume",
        "ratio_checks": ratios,
        "missingness": {c: int(z[c].isna().sum()) for c in REQUIRED},
        "status": status,
    }


def main() -> None:
    results = {m: audit_market(m) for m in SOURCES}
    status = "PASS" if all(x["status"] == "PASS" for x in results.values()) else "FAIL"
    manifest = {
        "layer": "V6.2 Institutional PIT supplemental-source admission",
        "scope": "2020-2024 only",
        "admission_semantics": "source trade-date flow becomes visible at next trading session; decision_date must be >= available_session",
        "tpex_foreign_mapping": "foreign_net = foreign+China excluding foreign dealers; foreign dealers are already represented in dealer flow and are excluded to avoid double counting",
        "tpex_semantic_fail_closed_checks": ["foreign_ex_dealer + foreign_dealer == foreign_total", "dealer_proprietary + dealer_hedge == dealer_total"],
        "rolling_boundary_rule": "any downstream rolling institutional feature must reset at V6.1 price_segment_id",
        "sources": results,
        "future_join_violations": sum(x["same_day_or_future_visibility_violations"] for x in results.values()),
        "duplicate_key_violations": sum(x["duplicate_key_violations"] for x in results.values()),
        "status": status,
    }
    out = OUT / "INSTITUTIONAL_SUPPLEMENTAL_ADMISSION.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    if status != "PASS":
        raise RuntimeError("institutional supplemental-source admission failed")


if __name__ == "__main__":
    main()
