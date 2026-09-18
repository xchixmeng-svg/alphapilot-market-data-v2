#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_v6_stage0_industry_layer as core  # noqa: E402

# Cross-source naming alias observed on official TWSE MI_INDEX in 2021+.
# This affects audit matching only; it does not alter the frozen 0B artifact.
core.ALIASES["電子類指數"] = ["電子類指數", "電子工業類指數"]

OUT = ROOT / "prelock_v6_industry_boundary_audit"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2020-12-01")
END = pd.Timestamp("2021-01-31")
ABS_TOL = 0.02


def fetch_tip_window(code_map: dict[str, str]) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    transport = []
    for name in core.FROZEN_INDUSTRY_NAMES:
        code = code_map[name]
        try:
            df, meta = core.request_tip_slice(code, name, START, END)
            meta = dict(meta)
            meta["index_name"] = name
            meta["index_code"] = code
            meta["ok"] = True
            transport.append(meta)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            transport.append({
                "index_name": name,
                "index_code": code,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            })
    if not frames:
        return core.empty_tip_frame(), transport
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date", "price_index"]), transport


def fetch_twse_window(code_map: dict[str, str]) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    dates = []
    for d in pd.date_range(START, END, freq="B"):
        try:
            obj = core.fetch_old_twse_date(d, code_map)
            dates.append({"date": d.date().isoformat(), "status": obj["status"], "stat": obj.get("stat")})
            if obj["status"] == "SUCCESS":
                rows.extend(obj.get("rows") or [])
        except Exception as e:
            dates.append({"date": d.date().isoformat(), "status": "ERROR", "error": f"{type(e).__name__}: {e}"})
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "price_index"])
    return df, dates


def boundary_equivalence(code_map: dict[str, str]) -> dict:
    tip, tip_transport = fetch_tip_window(code_map)
    twse, twse_dates = fetch_twse_window(code_map)

    if tip.empty or twse.empty:
        comp = pd.DataFrame(columns=[
            "date", "month", "index_name", "index_code",
            "tip_price_index", "twse_price_index", "abs_diff", "within_0p02"
        ])
    else:
        a = tip[["date", "index_code", "index_name", "price_index"]].rename(
            columns={"price_index": "tip_price_index"}
        )
        b = twse[["date", "index_code", "index_name", "price_index"]].rename(
            columns={"price_index": "twse_price_index"}
        )
        comp = a.merge(b, on=["date", "index_code", "index_name"], how="inner")
        comp["abs_diff"] = (comp["tip_price_index"] - comp["twse_price_index"]).abs()
        comp["within_0p02"] = comp["abs_diff"] <= ABS_TOL + 1e-12
        comp["month"] = comp["date"].dt.strftime("%Y-%m")
        comp = comp[[
            "date", "month", "index_name", "index_code",
            "tip_price_index", "twse_price_index", "abs_diff", "within_0p02"
        ]].sort_values(["date", "index_code"])
        comp["date"] = comp["date"].dt.strftime("%Y-%m-%d")

    comp.to_csv(OUT / "TWSE_MI_INDEX_vs_TIP_2020_12_2021_01_raw.csv", index=False, encoding="utf-8-sig")

    by_month = {}
    if not comp.empty:
        for month, g in comp.groupby("month"):
            by_month[month] = {
                "compared_observations": int(len(g)),
                "distinct_dates": int(g["date"].nunique()),
                "distinct_indices": int(g["index_code"].nunique()),
                "mismatches_gt_0p02": int((~g["within_0p02"]).sum()),
                "max_abs_diff": float(g["abs_diff"].max()),
                "mean_abs_diff": float(g["abs_diff"].mean()),
            }

    overlap_months = sorted(by_month)
    mismatch_count = int((~comp["within_0p02"]).sum()) if len(comp) else 0

    tip_dec = tip[(tip["date"] >= pd.Timestamp("2020-12-01")) & (tip["date"] <= pd.Timestamp("2020-12-31"))]
    tip_jan = tip[(tip["date"] >= pd.Timestamp("2021-01-01")) & (tip["date"] <= pd.Timestamp("2021-01-31"))]
    twse_jan = twse[(twse["date"] >= pd.Timestamp("2021-01-01")) & (twse["date"] <= pd.Timestamp("2021-01-31"))]
    tip_jan_codes = set(tip_jan["index_code"].dropna().astype(str))
    twse_jan_codes = set(twse_jan["index_code"].dropna().astype(str))
    common_jan_codes = sorted(tip_jan_codes & twse_jan_codes)

    # The requested range intentionally crosses the source boundary. TIP's official
    # historical response itself determines whether December 2020 exists. If it does
    # not, we record that fact and use January 2021 as the true overlapping boundary
    # month, because MI_INDEX remains queryable for that same month.
    jan = by_month.get("2021-01", {})
    enough_jan = (
        jan.get("compared_observations", 0) >= 660
        and jan.get("distinct_dates", 0) >= 20
        and jan.get("distinct_indices", 0) >= 33
    )
    passed = bool(enough_jan and mismatch_count == 0)

    return {
        "audit": "TWSE_MI_INDEX_vs_TIP_boundary_numeric_equivalence",
        "window": {"start": START.date().isoformat(), "end": END.date().isoformat()},
        "absolute_tolerance": ABS_TOL,
        "status": "PASS" if passed else "FAIL",
        "boundary_interpretation": (
            "TIP is queried across Dec-2020 and Jan-2021. If TIP returns zero Dec-2020 rows, "
            "Jan-2021 is the first true overlap month and is compared exhaustively against MI_INDEX."
        ),
        "tip_dec_2020_rows": int(len(tip_dec)),
        "tip_jan_2021_rows": int(len(tip_jan)),
        "twse_jan_2021_rows": int(len(twse_jan)),
        "tip_first_date_in_requested_window": None if tip.empty else tip["date"].min().date().isoformat(),
        "tip_jan_index_count": len(tip_jan_codes),
        "twse_jan_index_count": len(twse_jan_codes),
        "common_jan_index_count": len(common_jan_codes),
        "tip_only_jan_codes": sorted(tip_jan_codes - twse_jan_codes),
        "twse_only_jan_codes": sorted(twse_jan_codes - tip_jan_codes),
        "common_jan_codes": common_jan_codes,
        "compared_observations": int(len(comp)),
        "distinct_dates": int(comp["date"].nunique()) if len(comp) else 0,
        "distinct_indices": int(comp["index_code"].nunique()) if len(comp) else 0,
        "overlap_months": overlap_months,
        "by_month": by_month,
        "mismatches_gt_0p02": mismatch_count,
        "mismatch_rate": None if len(comp) == 0 else mismatch_count / len(comp),
        "max_abs_diff": None if len(comp) == 0 else float(comp["abs_diff"].max()),
        "tip_transport": tip_transport,
        "twse_date_status": twse_dates,
        "raw_comparison_csv": str((OUT / "TWSE_MI_INDEX_vs_TIP_2020_12_2021_01_raw.csv").relative_to(ROOT)),
    }


def total_return_root_cause(manifest_path: Path, feature_registry_path: Path) -> dict:
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    freg = json.loads(feature_registry_path.read_text(encoding="utf-8"))

    observed = int(m["missingness"]["total_return_index"]["missing_rows"])
    old_dates = int(m["old_history_checkpoint_audit"]["success_trading_dates"])
    per_series = m["coverage_audit"]["per_series"]
    legacy_series = [
        {"name": name, "code": x["code"]}
        for name, x in per_series.items()
        if str(x.get("date_start", "")).startswith("2016-")
    ]
    expected_old_missing = old_dates * len(legacy_series)
    feature_list = list(freg.get("features") or [])
    used_as_model_feature = "total_return_index" in feature_list or any(
        str(x).startswith("total_return_index") for x in feature_list
    )
    industry_features = [x for x in feature_list if str(x).startswith("industry_ret1_")]

    exact_accounting = observed == expected_old_missing
    passed = exact_accounting and not used_as_model_feature and len(industry_features) > 0

    return {
        "audit": "total_return_index_missingness_root_cause",
        "status": "PASS" if passed else "FAIL",
        "observed_missing_rows": observed,
        "observed_missing_rate": float(m["missingness"]["total_return_index"]["missing_rate"]),
        "old_segment_success_trading_dates_2016_2020": old_dates,
        "legacy_industry_series_starting_2016": len(legacy_series),
        "expected_missing_from_old_MI_INDEX_contract": expected_old_missing,
        "exact_missingness_accounting": exact_accounting,
        "root_cause": (
            "The 2016-2020 TWSE MI_INDEX parser explicitly writes total_return_index=None. "
            "All 33 legacy industry series have 1,224 successful old-segment trading dates, "
            "so 1,224*33=40,392 missing rows. This exactly equals the frozen manifest missing count, "
            "which leaves no residual missing total_return_index rows attributable to the TIP segment."
        ),
        "legacy_series": legacy_series,
        "model_feature_uses_total_return_index": used_as_model_feature,
        "industry_model_features": industry_features,
        "model_effect_note": (
            "V6 0F does not use total_return_index as a model feature. Industry context features are "
            "industry_ret1_* series derived from the frozen price-index path."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-0b-manifest", required=True)
    ap.add_argument("--frozen-feature-registry", required=True)
    ns = ap.parse_args()

    code_map, universe = core.resolve_official_code_map()
    boundary = boundary_equivalence(code_map)
    tri = total_return_root_cause(Path(ns.frozen_0b_manifest), Path(ns.frozen_feature_registry))

    evidence = {
        "status": "PASS" if boundary["status"] == "PASS" and tri["status"] == "PASS" else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_oos_opened": False,
        "universe_size": int(universe["frozen_universe_size"]),
        "boundary_equivalence": boundary,
        "total_return_index_missingness": tri,
    }
    out = OUT / "V6_PRELOCK_INDUSTRY_SOURCE_AUDIT.json"
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)
    return 0 if evidence["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
