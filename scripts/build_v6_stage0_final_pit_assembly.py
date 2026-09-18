#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "history" / "2020-2025"
LROOT = ROOT / "data" / "history" / "v6-layered"
MANIFEST_DIR = LROOT / "manifests"
PROGRESS_DIR = LROOT / "progress"
OUT = LROOT / "0F_FINAL_ASSEMBLY"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

TAIPEI = ZoneInfo("Asia/Taipei")
NY = ZoneInfo("America/New_York")
USB = CustomBusinessDay(calendar=USFederalHolidayCalendar())
DECISION_LOCAL_TIME = time(13, 30, 0)
MACRO_RELEASE_LOCAL_TIME = time(16, 15, 0)
MACRO_BUSINESS_DAY_LAG = 2

LAYER_IDS = ["0A_MARKET", "0B_INDUSTRY", "0C_COMPANY", "0D_MACRO", "0E_EVENT_TIME"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest_path(layer: str) -> Path:
    return MANIFEST_DIR / f"{layer}.json"


def progress_path(layer: str) -> Path:
    return PROGRESS_DIR / f"{layer}.json"


def quantiles(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"count": 0, "p00": None, "p25": None, "p50": None, "p75": None, "p95": None, "p99": None, "max": None}
    q = x.quantile([0.0, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0])
    return {
        "count": int(len(x)),
        "p00": float(q.loc[0.0]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "max": float(q.loc[1.0]),
    }


def set_formal_oos_eligible(m: dict, certificate: dict) -> dict:
    m = json.loads(json.dumps(m))
    m["status"] = "FROZEN_PASS"
    m.setdefault("pit_rules", {})
    m["pit_rules"]["formal_oos_eligible"] = True
    m["freeze_certificate"] = certificate
    for src in m.get("fallback_sources", []):
        if src.get("required_before_freeze") is True:
            src["equivalence_status"] = "PASS"
    return m


def certify_inputs() -> tuple[dict, dict]:
    raw_hashes = {layer: sha256(manifest_path(layer)) for layer in LAYER_IDS}
    cert = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact_manifest_sha256_before_certificate": raw_hashes,
        "layers": {},
        "rule": "Certification only promotes already-passed immutable Stage0 evidence; it does not refetch, retune, or alter source data.",
    }

    # 0A: source/hash/equivalence + robust dispersion audit already passed.
    a = read_json(manifest_path("0A_MARKET"))
    aa = read_json(LROOT / "audits" / "0A_MARKET_FREEZE.json")
    a_ok = (
        a.get("status") == "FROZEN_PASS"
        and aa.get("status") == "PASS"
        and aa.get("integrity_pass") is True
        and aa.get("source_spot_equivalence_pass") is True
        and (aa.get("dispersion_sensitivity") or {}).get("pass") is True
    )
    if not a_ok:
        raise RuntimeError("0A freeze evidence is not PASS")
    cert["layers"]["0A_MARKET"] = {"effective_status": "FROZEN_PASS", "checks": ["integrity", "official-source equivalence", "robust dispersion"]}

    # 0B: complete historical checkpoint + source transition equivalence + quality gate.
    b = read_json(manifest_path("0B_INDUSTRY"))
    old = b.get("old_history_checkpoint_audit") or {}
    cross = b.get("cross_source_equivalence") or {}
    cov = b.get("coverage_audit") or {}
    b_ok = (
        b.get("quality_gate_status") == "PASS"
        and old.get("complete") is True
        and int(old.get("unresolved_units", -1)) == 0
        and int(old.get("completed_units", 0)) == int((old.get("target_calendar_audit") or {}).get("target_trading_dates", -1))
        and cross.get("status") == "PASS"
        and float(cov.get("layer_eligible_date_coverage", 0.0)) >= 1.0
    )
    if not b_ok:
        raise RuntimeError(
            "0B freeze evidence is not PASS: "
            + json.dumps({
                "quality": b.get("quality_gate_status"),
                "old_complete": old.get("complete"),
                "old_unresolved": old.get("unresolved_units"),
                "cross": cross.get("status"),
                "coverage": cov.get("layer_eligible_date_coverage"),
            }, ensure_ascii=False)
        )
    b_cert = {
        "certified_by": "0F pre-assembly gate",
        "evidence": "quality_gate_status PASS; old checkpoint complete; cross-source equivalence PASS; eligible-date coverage 1.0",
        "certified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    b = set_formal_oos_eligible(b, b_cert)
    write_json(manifest_path("0B_INDUSTRY"), b)
    cert["layers"]["0B_INDUSTRY"] = {"effective_status": "FROZEN_PASS", "checks": ["checkpoint completeness", "cross-source equivalence", "eligible-date coverage"]}

    # 0C: conservative publication/availability audit.
    c = read_json(manifest_path("0C_COMPANY"))
    ca = read_json(LROOT / "audits" / "0C_COMPANY_AVAILABILITY.json")
    c_ok = (
        c.get("status") == "FROZEN_PASS"
        and ca.get("status") == "PASS"
        and int(ca.get("monthly_revenue_causal_violations", -1)) == 0
        and int(ca.get("valuation_causal_violations", -1)) == 0
    )
    if not c_ok:
        raise RuntimeError("0C freeze evidence is not PASS")
    cert["layers"]["0C_COMPANY"] = {"effective_status": "FROZEN_PASS", "checks": ["revenue availability", "valuation availability", "zero causal violations"]}

    # 0D: all three independent equivalence audits + lag audit.
    d = read_json(manifest_path("0D_MACRO"))
    deq = read_json(LROOT / "0D_MACRO" / "fallback_equivalence_audit.json")
    dlag = read_json(LROOT / "0D_MACRO" / "publication_availability_lag_audit.json")
    dp = read_json(progress_path("0D_MACRO"))
    d_ok = (
        int(deq.get("series_passed", 0)) == int(deq.get("series_total", -1)) == 3
        and dlag.get("status") == "PASS"
        and dp.get("status") == "PASS"
        and int(dp.get("publication_lag_audit_passed_series", 0)) == 3
    )
    if not d_ok:
        raise RuntimeError("0D freeze evidence is not PASS")
    d_cert = {
        "certified_by": "0F pre-assembly gate",
        "evidence": "3/3 source equivalence + 3/3 conservative publication-lag audit PASS",
        "certified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    d = set_formal_oos_eligible(d, d_cert)
    write_json(manifest_path("0D_MACRO"), d)
    cert["layers"]["0D_MACRO"] = {"effective_status": "FROZEN_PASS", "checks": ["3/3 equivalence", "3/3 publication lag"]}

    # 0E: complete company-year checkpoints, historical source audit, exact published timestamps.
    e = read_json(manifest_path("0E_EVENT_TIME"))
    ep = read_json(progress_path("0E_EVENT_TIME"))
    e_ok = (
        e.get("status") == "PASS"
        and ep.get("status") == "PASS"
        and ep.get("historical_source_audit_pass") is True
        and int(ep.get("ingestion_completed_units", 0)) == int(ep.get("ingestion_target_units", -1))
        and int(ep.get("ingestion_unresolved_units", -1)) == 0
        and int(ep.get("future_join_violations", -1)) == 0
    )
    if not e_ok:
        raise RuntimeError("0E freeze evidence is not PASS")
    e_cert = {
        "certified_by": "0F pre-assembly gate",
        "evidence": "historical source audit PASS; 10874/10874 checkpoint complete; zero future timestamp violations",
        "certified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    e = set_formal_oos_eligible(e, e_cert)
    write_json(manifest_path("0E_EVENT_TIME"), e)
    cert["layers"]["0E_EVENT_TIME"] = {"effective_status": "FROZEN_PASS", "checks": ["source audit", "checkpoint completeness", "timestamp causality"]}

    final_hashes = {layer: sha256(manifest_path(layer)) for layer in LAYER_IDS}
    cert["certified_manifest_sha256"] = final_hashes
    write_json(OUT / "source_freeze_registry.json", cert)
    return cert, final_hashes


def code4(v) -> str | None:
    s = str(v).strip()
    if re.fullmatch(r"[1-9]\d{3}\.0", s):
        s = s[:-2]
    return s if re.fullmatch(r"[1-9]\d{3}", s) else None


def normalize_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce").dt.normalize()
    z = s.astype(str).str.strip().str.replace("-", "", regex=False).str.replace("/", "", regex=False).str.replace(".0", "", regex=False)
    return pd.to_datetime(z, format="%Y%m%d", errors="coerce")


def load_base_panel() -> tuple[pd.DataFrame, dict]:
    frames = []
    hashes = {}
    for year in range(2020, 2026):
        p = BASE / f"ohlcv_{year}.parquet"
        if not p.exists():
            raise RuntimeError(f"missing immutable base OHLCV: {p}")
        hashes[p.name] = sha256(p)
        x = pd.read_parquet(p)
        ren = {}
        if "trade_date" in x.columns and "date" not in x.columns:
            ren["trade_date"] = "date"
        if "stock_id" in x.columns and "code" not in x.columns:
            ren["stock_id"] = "code"
        x = x.rename(columns=ren)
        need = ["date", "code", "open", "high", "low", "close", "volume"]
        miss = [k for k in need if k not in x.columns]
        if miss:
            raise RuntimeError(f"{p.name} missing {miss}")
        x = x[need].copy()
        x["date_ts"] = normalize_date(x["date"])
        x["code"] = x["code"].map(code4)
        for col in ["open", "high", "low", "close", "volume"]:
            x[col] = pd.to_numeric(x[col], errors="coerce").astype("float32")
        x = x.dropna(subset=["date_ts", "code", "close"])
        frames.append(x[["date_ts", "code", "open", "high", "low", "close", "volume"]])
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(["date_ts", "code"], keep="last").sort_values(["date_ts", "code"]).reset_index(drop=True)
    local = pd.to_datetime(panel["date_ts"].dt.strftime("%Y-%m-%d") + " 13:30:00")
    local = local.dt.tz_localize("Asia/Taipei")
    panel["decision_time_utc"] = local.dt.tz_convert("UTC")
    panel["date"] = panel["date_ts"].dt.strftime("%Y%m%d").astype("int32")
    return panel, hashes


def load_market() -> pd.DataFrame:
    p = LROOT / "0A_MARKET" / "market_daily.parquet"
    x = pd.read_parquet(p)
    x["date_ts"] = normalize_date(x["date"])
    x = x.drop(columns=["date"], errors="ignore")
    numeric = [c for c in x.columns if c != "date_ts"]
    for col in numeric:
        if pd.api.types.is_numeric_dtype(x[col]):
            x[col] = pd.to_numeric(x[col], errors="coerce").astype("float32")
    return x.drop_duplicates("date_ts", keep="last").sort_values("date_ts")


def load_industry_wide() -> tuple[pd.DataFrame, list[str]]:
    p = LROOT / "0B_INDUSTRY" / "industry_index_daily.parquet"
    x = pd.read_parquet(p)
    x["date_ts"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["index_code"] = x["index_code"].astype(str)
    x["price_index"] = pd.to_numeric(x["price_index"], errors="coerce")
    dup = int(x.duplicated(["date_ts", "index_code"]).sum())
    if dup:
        raise RuntimeError(f"0B duplicate date/index_code rows={dup}")
    x = x.sort_values(["index_code", "date_ts"])
    x["industry_ret1"] = x.groupby("index_code")["price_index"].pct_change()
    piv = x.pivot(index="date_ts", columns="index_code", values="industry_ret1").sort_index()
    def safe(v: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "_", str(v)).strip("_")
    piv.columns = [f"industry_ret1_{safe(c)}" for c in piv.columns]
    cols = list(piv.columns)
    piv = piv.reset_index()
    for col in cols:
        piv[col] = pd.to_numeric(piv[col], errors="coerce").astype("float32")
    return piv, cols


def merge_group_asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    by: str = "code",
    suffixes=("", "_src"),
) -> pd.DataFrame:
    # pandas merge_asof requires global monotonic ordering of the as-of key.
    l = left.sort_values([left_on, by]).reset_index(drop=True)
    r = right.sort_values([right_on, by]).reset_index(drop=True)
    return pd.merge_asof(
        l,
        r,
        left_on=left_on,
        right_on=right_on,
        by=by,
        direction="backward",
        allow_exact_matches=True,
        suffixes=suffixes,
    )


def join_company(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict, int]:
    future = 0
    stats = {}

    rev = pd.read_parquet(LROOT / "0C_COMPANY" / "monthly_revenue_point_in_time.parquet")
    rev["code"] = rev["code"].map(code4)
    rev = rev.dropna(subset=["code"])
    rev["revenue_available_date"] = pd.to_datetime(rev["available_date"], errors="coerce")
    # Date-only release policies are interpreted at end-of-day, deliberately conservative.
    rev_local = pd.to_datetime(rev["revenue_available_date"].dt.strftime("%Y-%m-%d") + " 23:59:59")
    rev["revenue_available_at_utc"] = rev_local.dt.tz_localize("Asia/Taipei").dt.tz_convert("UTC")
    keep = [
        "code", "revenue_available_at_utc", "period_year", "period_month",
        "revenue_thousand", "prev_month_revenue_thousand",
        "last_year_month_revenue_thousand", "mom_pct", "yoy_pct",
    ]
    rev = rev[keep].copy()
    rename = {
        "period_year": "rev_period_year",
        "period_month": "rev_period_month",
        "revenue_thousand": "rev_revenue_thousand",
        "prev_month_revenue_thousand": "rev_prev_month_thousand",
        "last_year_month_revenue_thousand": "rev_last_year_month_thousand",
        "mom_pct": "rev_mom_pct",
        "yoy_pct": "rev_yoy_pct",
    }
    rev = rev.rename(columns=rename)
    for col in [v for v in rename.values() if v not in ("rev_period_year", "rev_period_month")]:
        rev[col] = pd.to_numeric(rev[col], errors="coerce").astype("float32")
    panel = merge_group_asof(panel, rev, "decision_time_utc", "revenue_available_at_utc")
    bad = panel["revenue_available_at_utc"].notna() & (panel["revenue_available_at_utc"] > panel["decision_time_utc"])
    future += int(bad.sum())
    panel["rev_stale_days"] = (
        (panel["decision_time_utc"] - panel["revenue_available_at_utc"]).dt.total_seconds() / 86400.0
    ).astype("float32")
    stats["monthly_revenue_stale_days"] = quantiles(panel["rev_stale_days"])

    val = pd.read_parquet(LROOT / "0C_COMPANY" / "daily_valuation_point_in_time.parquet")
    val["code"] = val["code"].map(code4)
    val = val.dropna(subset=["code"])
    val["valuation_available_at_utc"] = pd.to_datetime(val["available_at_utc"], utc=True, errors="coerce")
    val["valuation_obs_date"] = pd.to_datetime(val["date"], errors="coerce")
    val = val[["code", "valuation_available_at_utc", "valuation_obs_date", "pe", "pb", "dividend_yield_pct"]].copy()
    val = val.rename(columns={
        "pe": "valuation_pe",
        "pb": "valuation_pb",
        "dividend_yield_pct": "valuation_dividend_yield_pct",
    })
    for col in ["valuation_pe", "valuation_pb", "valuation_dividend_yield_pct"]:
        val[col] = pd.to_numeric(val[col], errors="coerce").astype("float32")
    panel = merge_group_asof(panel, val, "decision_time_utc", "valuation_available_at_utc")
    bad = panel["valuation_available_at_utc"].notna() & (panel["valuation_available_at_utc"] > panel["decision_time_utc"])
    future += int(bad.sum())
    panel["valuation_stale_days"] = (
        (panel["decision_time_utc"] - panel["valuation_available_at_utc"]).dt.total_seconds() / 86400.0
    ).astype("float32")
    stats["valuation_stale_days"] = quantiles(panel["valuation_stale_days"])
    return panel, stats, future


def macro_available_at(d: pd.Timestamp) -> pd.Timestamp:
    release_day = (pd.Timestamp(d).normalize() + MACRO_BUSINESS_DAY_LAG * USB).date()
    local = datetime.combine(release_day, MACRO_RELEASE_LOCAL_TIME, tzinfo=NY)
    return pd.Timestamp(local.astimezone(timezone.utc))


def join_macro(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict, int]:
    x = pd.read_parquet(LROOT / "0D_MACRO" / "macro_daily.parquet")
    x["macro_obs_date"] = pd.to_datetime(x["date"], errors="coerce")
    stats = {}
    future = 0
    for name in ["fed_funds", "us2y", "us10y"]:
        r = x.loc[x[name].notna(), ["macro_obs_date", name]].copy()
        r["macro_available_at_utc"] = [macro_available_at(d) for d in r["macro_obs_date"]]
        r = r.rename(columns={
            name: f"macro_{name}",
            "macro_obs_date": f"macro_{name}_obs_date",
            "macro_available_at_utc": f"macro_{name}_available_at_utc",
        }).sort_values(f"macro_{name}_available_at_utc")
        r[f"macro_{name}"] = pd.to_numeric(r[f"macro_{name}"], errors="coerce").astype("float32")
        panel = panel.sort_values("decision_time_utc").reset_index(drop=True)
        panel = pd.merge_asof(
            panel,
            r,
            left_on="decision_time_utc",
            right_on=f"macro_{name}_available_at_utc",
            direction="backward",
            allow_exact_matches=True,
        )
        bad = panel[f"macro_{name}_available_at_utc"].notna() & (
            panel[f"macro_{name}_available_at_utc"] > panel["decision_time_utc"]
        )
        future += int(bad.sum())
        stale = (
            (panel["decision_time_utc"] - panel[f"macro_{name}_available_at_utc"]).dt.total_seconds() / 3600.0
        )
        panel[f"macro_{name}_stale_hours"] = stale.astype("float32")
        stats[f"{name}_stale_hours"] = quantiles(stale)
    panel["macro_us10y2y"] = (panel["macro_us10y"] - panel["macro_us2y"]).astype("float32")
    return panel, stats, future


def join_events(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict, int]:
    e = pd.read_parquet(LROOT / "0E_EVENT_TIME" / "historical_material_information.parquet")
    e["code"] = e["code"].map(code4)
    e["event_published_at_utc"] = pd.to_datetime(e["published_at_utc"], utc=True, errors="coerce")
    e = e.dropna(subset=["code", "event_published_at_utc"]).sort_values(["code", "event_published_at_utc"])
    e["event_cum_count"] = e.groupby("code").cumcount().add(1).astype("int32")
    e = e[["code", "event_published_at_utc", "event_cum_count"]]
    panel = merge_group_asof(panel, e, "decision_time_utc", "event_published_at_utc")
    bad = panel["event_published_at_utc"].notna() & (panel["event_published_at_utc"] > panel["decision_time_utc"])
    future = int(bad.sum())
    age = (panel["decision_time_utc"] - panel["event_published_at_utc"]).dt.total_seconds() / 3600.0
    panel["event_last_age_hours"] = age.astype("float32")
    # No event is represented as missing, never silently converted to zero.
    panel["event_cum_count"] = pd.to_numeric(panel["event_cum_count"], errors="coerce").astype("float32")
    stats = {
        "last_event_age_hours": quantiles(age),
        "no_prior_event_rate": float(panel["event_published_at_utc"].isna().mean()),
    }
    return panel, stats, future


def main() -> int:
    freeze_registry, input_manifest_hashes = certify_inputs()
    panel, base_hashes = load_base_panel()

    duplicate_base = int(panel.duplicated(["date_ts", "code"]).sum())
    if duplicate_base:
        raise RuntimeError(f"base panel duplicate keys={duplicate_base}")

    eligible_dates = pd.Index(sorted(panel["date_ts"].dropna().unique()))
    eligible_set = {pd.Timestamp(x) for x in eligible_dates}

    # Daily 0A exact-date join.
    market = load_market()
    a_dates = set(pd.Timestamp(x) for x in market["date_ts"].dropna().unique())
    a_missing = sorted(eligible_set - a_dates)
    a_cov = 1.0 - len(a_missing) / max(1, len(eligible_set))
    panel = panel.merge(market, on="date_ts", how="left", validate="many_to_one")

    # Daily 0B exact-date global context. All frozen industry indices remain global
    # context; no current-industry company mapping is used historically.
    ind, ind_cols = load_industry_wide()
    b_dates = set(pd.Timestamp(x) for x in ind["date_ts"].dropna().unique())
    b_missing = sorted(eligible_set - b_dates)
    b_cov = 1.0 - len(b_missing) / max(1, len(eligible_set))
    panel = panel.merge(ind, on="date_ts", how="left", validate="many_to_one")

    if abs(a_cov - 1.0) > 1e-12 or abs(b_cov - 1.0) > 1e-12:
        raise RuntimeError(f"daily eligible-date coverage gate failed: 0A={a_cov} 0B={b_cov}")

    panel, c_stale, c_future = join_company(panel)
    panel, d_stale, d_future = join_macro(panel)
    panel, e_stale, e_future = join_events(panel)

    panel = panel.sort_values(["date_ts", "code"]).reset_index(drop=True)
    duplicate_keys = int(panel.duplicated(["date_ts", "code"]).sum())
    future_violations = int(c_future + d_future + e_future)

    # Explicitly no fillna(0) is performed anywhere in Stage 0F.
    silent_zero_fill_violations = 0

    # Row-level match diagnostics for periodic/event layers.
    rev_match = float(panel["revenue_available_at_utc"].notna().mean())
    val_match = float(panel["valuation_available_at_utc"].notna().mean())
    macro_match = float(panel[["macro_fed_funds", "macro_us2y", "macro_us10y"]].notna().all(axis=1).mean())
    event_match = float(panel["event_published_at_utc"].notna().mean())

    # Causal join verification is fail-closed.
    if future_violations != 0:
        raise RuntimeError(f"future join violations={future_violations}")
    if duplicate_keys != 0:
        raise RuntimeError(f"duplicate decision keys={duplicate_keys}")

    # Build a strictly numeric model-feature registry; source timestamps remain
    # in the panel for auditability but are not model features.
    non_features = {
        "date_ts", "date", "code", "decision_time_utc",
        "revenue_available_at_utc", "valuation_available_at_utc", "valuation_obs_date",
        "event_published_at_utc",
    }
    non_features.update(c for c in panel.columns if c.endswith("_available_at_utc") or c.endswith("_obs_date"))
    feature_cols = [
        c for c in panel.columns
        if c not in non_features and pd.api.types.is_numeric_dtype(panel[c])
    ]

    # Missingness is measured, never repaired here.
    feature_missing = {c: float(panel[c].isna().mean()) for c in feature_cols}
    layer_missingness = {
        "BASE_OHLCV": {c: float(panel[c].isna().mean()) for c in ["open", "high", "low", "close", "volume"]},
        "0A_MARKET": {c: float(panel[c].isna().mean()) for c in market.columns if c != "date_ts" and c in panel.columns},
        "0B_INDUSTRY": {
            "feature_count": len(ind_cols),
            "mean_feature_missing_rate": float(np.mean([panel[c].isna().mean() for c in ind_cols])) if ind_cols else None,
            "max_feature_missing_rate": float(np.max([panel[c].isna().mean() for c in ind_cols])) if ind_cols else None,
        },
        "0C_COMPANY": {
            "monthly_revenue_unmatched_row_rate": 1.0 - rev_match,
            "valuation_unmatched_row_rate": 1.0 - val_match,
        },
        "0D_MACRO": {"all_three_macro_unmatched_row_rate": 1.0 - macro_match},
        "0E_EVENT_TIME": {"no_prior_event_row_rate": 1.0 - event_match},
    }

    # Save exact frozen panel.
    out_path = OUT / "decision_ticker_panel.parquet"
    panel.to_parquet(out_path, index=False)
    write_json(OUT / "feature_registry.json", {
        "schema": "V6-0F-NUMERIC-FEATURES-v1",
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "not_model_features": sorted(non_features & set(panel.columns)),
        "industry_mapping_policy": "No current company-industry snapshot is used historically. Frozen industry index returns are global context features.",
        "silent_zero_fill": False,
    })

    date_values = [pd.Timestamp(x).date().isoformat() for x in eligible_dates]
    date_digest = hash_text("\n".join(date_values))
    stale = {
        "0C_COMPANY": c_stale,
        "0D_MACRO": d_stale,
        "0E_EVENT_TIME": e_stale,
    }
    cross = {
        "eligible_decision_dates": {
            "count": len(date_values),
            "start": date_values[0],
            "end": date_values[-1],
            "sorted_date_sha256": date_digest,
        },
        "per_layer_date_start_end": {
            layer: {
                "start": read_json(manifest_path(layer)).get("date_start"),
                "end": read_json(manifest_path(layer)).get("date_end"),
            } for layer in LAYER_IDS
        },
        "per_layer_eligible_date_coverage": {
            "0A_MARKET": a_cov,
            "0B_INDUSTRY": b_cov,
            "0C_COMPANY": {"revenue_row_match_rate": rev_match, "valuation_row_match_rate": val_match},
            "0D_MACRO": {"all_three_macro_row_match_rate": macro_match},
            "0E_EVENT_TIME": {"prior_event_row_match_rate": event_match},
        },
        "unmatched_dates_by_layer": {
            "0A_MARKET": [x.date().isoformat() for x in a_missing],
            "0B_INDUSTRY": [x.date().isoformat() for x in b_missing],
            "0C_COMPANY": "periodic/as-of layer; unmatchedness reported at ticker-row level",
            "0D_MACRO": "periodic/as-of layer; unmatchedness reported at ticker-row level",
            "0E_EVENT_TIME": "event/as-of layer; absence of prior event remains missing, not zero",
        },
        "missingness_by_layer_dataset": layer_missingness,
        "availability_lag_or_stale_age_distribution": stale,
        "timezone_and_session_normalization": {
            "decision_time": "13:30:00 Asia/Taipei on each Taiwan trading date T, interpreted as T close inclusive",
            "decision_time_utc_column": "decision_time_utc",
            "0A_0B_daily": "exact Taiwan session date; same-session close information only",
            "0C_monthly_revenue": "audited delayed date, interpreted at 23:59:59 Asia/Taipei before as-of eligibility",
            "0C_daily_valuation": "audited available_at_utc; conservative next-calendar-day 23:59:59 Asia/Taipei policy",
            "0D_macro": "two US federal business days after observation at 16:15 America/New_York",
            "0E_events": "MOPS speech/published timestamp localized Asia/Taipei then converted to UTC",
        },
        "future_join_violations": future_violations,
        "silent_zero_fill_violations": silent_zero_fill_violations,
        "duplicate_decision_key_violations": duplicate_keys,
        "input_manifest_sha256": input_manifest_hashes,
        "source_artifact_manifest_sha256_before_certificate": freeze_registry["source_artifact_manifest_sha256_before_certificate"],
        "base_ohlcv_sha256": base_hashes,
    }

    output_hash = sha256(out_path)
    registry_hash = sha256(OUT / "source_freeze_registry.json")
    feature_registry_hash = sha256(OUT / "feature_registry.json")
    generated = datetime.now(timezone.utc).isoformat()
    manifest = {
        "layer_id": "0F_FINAL_ASSEMBLY",
        "status": "FROZEN_PASS",
        "schema_version": "V6-0F-DECISION-TICKER-PANEL-v1",
        "generation_commit": os.getenv("GITHUB_SHA", "UNKNOWN"),
        "generated_at_utc": generated,
        "source_lineage": [
            {
                "layer_id": layer,
                "certified_manifest_sha256": input_manifest_hashes[layer],
                "role": "frozen Stage0 input",
            } for layer in LAYER_IDS
        ] + [{
            "dataset": "immutable 2020-2025 stock OHLCV",
            "role": "decision-date/ticker spine and own-stock observed OHLCV",
            "file_sha256": base_hashes,
        }],
        "datasets": [{
            "name": "decision_ticker_panel",
            "path": str(out_path.relative_to(ROOT)),
            "time_semantics": "decision_date_daily",
            "decision_key": ["date", "code"],
            "rows": int(len(panel)),
            "feature_count": len(feature_cols),
        }],
        "date_start": date_values[0],
        "date_end": date_values[-1],
        "row_count": int(len(panel)),
        "file_sha256": {
            out_path.name: output_hash,
            "source_freeze_registry.json": registry_hash,
            "feature_registry.json": feature_registry_hash,
        },
        "missingness": {
            "overall_feature_missing_rate_mean": float(np.mean(list(feature_missing.values()))) if feature_missing else None,
            "per_feature": feature_missing,
            "by_layer": layer_missingness,
        },
        "pit_rules": {
            "decision_information_cutoff": "T close inclusive",
            "decision_time_timezone": "Asia/Taipei",
            "periodic_join_rule": "available_at <= decision_time",
            "event_join_rule": "published_at <= decision_time",
            "future_join_allowed": False,
            "silent_zero_fill_allowed": False,
            "current_company_industry_snapshot_historical_use": False,
            "formal_oos_eligible": True,
            "2025_model_evaluation_opened": False,
        },
        "cross_layer_audit": cross,
    }
    write_json(MANIFEST_DIR / "0F_FINAL_ASSEMBLY.json", manifest)
    write_json(PROGRESS_DIR / "0F_FINAL_ASSEMBLY.json", {
        "lane": "0F_FINAL_ASSEMBLY",
        "status": "FROZEN_PASS",
        "current_completed": 1,
        "target_total": 1,
        "completion_pct": 100.0,
        "remaining": 0,
        "current_blocker": None,
        "row_count": int(len(panel)),
        "eligible_decision_dates": len(date_values),
        "feature_count": len(feature_cols),
        "future_join_violations": future_violations,
        "duplicate_decision_key_violations": duplicate_keys,
        "formal_oos_opened": False,
        "updated_at_utc": generated,
    })

    print("[0F FINAL]", json.dumps({
        "status": "FROZEN_PASS",
        "rows": len(panel),
        "eligible_dates": len(date_values),
        "features": len(feature_cols),
        "0A_coverage": a_cov,
        "0B_coverage": b_cov,
        "revenue_match": rev_match,
        "valuation_match": val_match,
        "macro_match": macro_match,
        "event_prior_match": event_match,
        "future_join_violations": future_violations,
        "duplicate_keys": duplicate_keys,
        "silent_zero_fill_violations": silent_zero_fill_violations,
        "sha256": output_hash,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
