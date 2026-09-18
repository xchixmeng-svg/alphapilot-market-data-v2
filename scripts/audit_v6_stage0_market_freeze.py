#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data" / "history" / "2007-2019"
NEW = ROOT / "data" / "history" / "2020-2025"
MANIFEST = ROOT / "data" / "history" / "v6-layered" / "manifests" / "0A_MARKET.json"
AUDIT_DIR = ROOT / "data" / "history" / "v6-layered" / "audits"
PROGRESS = ROOT / "data" / "history" / "v6-layered" / "progress" / "0A_MARKET.json"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 AlphaPilot-V6-0A-Audit/1.0"})
SPOT_DATES = ["2016-03-15", "2017-06-15", "2018-09-17", "2019-12-16"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def num(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", "").replace("+", "")
    if s in ("", "--", "---", "-", "nan", "None"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def code4(x):
    s = str(x).strip().replace(".0", "")
    return s if re.fullmatch(r"[1-9]\d{3}", s) else None


def load_old_year(year: int) -> pd.DataFrame:
    p = OLD / "raw" / f"yearly_{year}.zip"
    with zipfile.ZipFile(p) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(member) as fh:
            x = pd.read_csv(fh, low_memory=False)
    x["code"] = x["code"].map(code4)
    x["date"] = pd.to_numeric(x["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=["code", "date", "close"])


def load_new_year(year: int) -> pd.DataFrame:
    x = pd.read_parquet(NEW / f"ohlcv_{year}.parquet")
    if "trade_date" in x.columns and "date" not in x.columns:
        x = x.rename(columns={"trade_date": "date"})
    if "stock_id" in x.columns and "code" not in x.columns:
        x = x.rename(columns={"stock_id": "code"})
    x["code"] = x["code"].map(code4)
    d = x["date"]
    if pd.api.types.is_numeric_dtype(d):
        x["date"] = pd.to_numeric(d, errors="coerce")
    else:
        x["date"] = pd.to_datetime(d, errors="coerce").dt.strftime("%Y%m%d")
        x["date"] = pd.to_numeric(x["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=["code", "date", "close"])


def verify_integrity():
    old_manifest = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    old_map = {int(x["year"]): x for x in old_manifest["files"]}
    old_checks = []
    for y in range(2016, 2020):
        p = OLD / "raw" / f"yearly_{y}.zip"
        got = sha256(p)
        exp = old_map[y]["sha256"]
        old_checks.append({"year": y, "sha256_match": got == exp, "sha256": got})

    sums = {}
    for line in (NEW / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            sums[parts[-1]] = parts[0]
    new_checks = []
    for y in range(2020, 2026):
        name = f"ohlcv_{y}.parquet"
        p = NEW / name
        got = sha256(p)
        exp = sums.get(name)
        new_checks.append({"year": y, "sha256_match": bool(exp) and got == exp, "sha256": got})
    return old_checks, new_checks


def parse_twse(d: str) -> dict[str, float]:
    ds = d.replace("-", "")
    r = S.get(
        "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
        params={"response": "json", "date": ds, "type": "ALLBUT0999"},
        timeout=(15, 60),
    )
    r.raise_for_status()
    j = r.json()
    out = {}
    for t in j.get("tables") or []:
        fields = [str(x) for x in (t.get("fields") or [])]
        if "證券代號" not in fields or not any("收盤價" in x for x in fields):
            continue
        ci = fields.index("證券代號")
        pi = next(i for i, x in enumerate(fields) if "收盤價" in x)
        for row in t.get("data") or []:
            if ci >= len(row) or pi >= len(row):
                continue
            c = code4(row[ci])
            v = num(row[pi])
            if c and v is not None:
                out[c] = v
    return out


def parse_tpex(d: str) -> dict[str, float]:
    y, m, day = map(int, d.split("-"))
    roc = f"{y-1911:03d}/{m:02d}/{day:02d}"
    urls = [
        (
            "https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php",
            {"l": "zh-tw", "o": "json", "d": roc, "se": "AL", "s": "0,asc,0"},
        ),
        (
            "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php",
            {"l": "zh-tw", "d": roc, "se": "EW", "t": "D"},
        ),
        (
            "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
            {"date": d, "id": "", "response": "json"},
        ),
    ]
    last = None
    for url, params in urls:
        for attempt in range(3):
            try:
                r = S.get(
                    url,
                    params=params,
                    headers={"Connection": "close"},
                    timeout=(15, 60),
                )
                r.raise_for_status()
                j = r.json()
                fields = []
                rows = j.get("aaData") or j.get("data") or []
                if j.get("tables"):
                    table = j["tables"][0]
                    fields = [str(x) for x in (table.get("fields") or [])]
                    rows = rows or table.get("data") or []
                out = {}
                close_i = None
                if fields:
                    close_i = next(
                        (i for i, x in enumerate(fields) if "收盤" in x),
                        2 if len(fields) > 2 else None,
                    )
                for row in rows:
                    if isinstance(row, dict):
                        cc = code4(
                            row.get("SecuritiesCompanyCode")
                            or row.get("股票代號")
                            or row.get("代號")
                            or row.get("Code")
                        )
                        v = num(
                            row.get("Close")
                            or row.get("收盤")
                            or row.get("收盤價")
                            or row.get("ClosePrice")
                        )
                    elif isinstance(row, list) and len(row) >= 3:
                        cc = code4(row[0])
                        idx = close_i if close_i is not None and close_i < len(row) else 2
                        v = num(row[idx])
                    else:
                        continue
                    if cc and v is not None:
                        out[cc] = v
                if out:
                    return out
                last = RuntimeError(f"empty TPEx payload via {url}")
            except Exception as e:
                last = e
    raise RuntimeError(f"TPEx historical query failed for {d}: {last}")


def source_spot_equivalence(frames: dict[int, pd.DataFrame]):
    rows = []
    source_errors = []
    for d in SPOT_DATES:
        y = int(d[:4])
        ymd = int(d.replace("-", ""))
        local = frames[y]
        local = local[local["date"].astype(int) == ymd]
        local_map = dict(zip(local["code"], local["close"]))
        for market, fetcher in (("TWSE", parse_twse), ("TPEX", parse_tpex)):
            try:
                official = fetcher(d)
                common = sorted(set(local_map) & set(official))
                matches = 0
                compared = 0
                max_abs = 0.0
                for c in common:
                    a = num(local_map[c])
                    b = num(official[c])
                    if a is None or b is None:
                        continue
                    compared += 1
                    diff = abs(a - b)
                    max_abs = max(max_abs, diff)
                    if diff <= 1e-6:
                        matches += 1
                rows.append({
                    "date": d,
                    "year": y,
                    "market": market,
                    "official_codes": len(official),
                    "overlap": len(common),
                    "compared": compared,
                    "exact_close_matches": matches,
                    "match_rate": matches / compared if compared else 0.0,
                    "max_abs_close_diff": max_abs,
                })
            except Exception as e:
                source_errors.append({"date": d, "market": market, "error": str(e)})
    good = [r for r in rows if r["compared"] >= 10 and r["match_rate"] >= 0.995]
    years = {r["year"] for r in good}
    total = sum(r["compared"] for r in good)
    twse_years = {r["year"] for r in good if r["market"] == "TWSE"}
    tpex_years = {r["year"] for r in good if r["market"] == "TPEX"}
    passed = total >= 60 and len(years) >= 3 and len(twse_years) >= 3 and len(tpex_years) >= 3
    return rows, source_errors, passed


def _mad(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    med = float(x.median())
    return 1.4826 * float((x - med).abs().median())


def dispersion_audit(frames: dict[int, pd.DataFrame]):
    px = pd.concat([frames[y] for y in range(2016, 2026)], ignore_index=True)
    px = px.sort_values(["code", "date"]).drop_duplicates(["date", "code"], keep="last")
    px["ret1_raw"] = px.groupby("code")["close"].pct_change()
    eq = px[px["code"].str.fullmatch(r"[1-9]\d{3}", na=False)].copy()
    eq["ret1_clean"] = eq["ret1_raw"].where(eq["ret1_raw"].abs() <= 0.20)

    rows = []
    for d, g in eq.groupby("date"):
        raw = pd.to_numeric(g["ret1_raw"], errors="coerce")
        clean = pd.to_numeric(g["ret1_clean"], errors="coerce")
        rows.append({
            "date": int(d),
            "raw_std": float(raw.std()) if raw.notna().sum() >= 2 else np.nan,
            "clean_std": float(clean.std()) if clean.notna().sum() >= 2 else np.nan,
            "raw_mad": _mad(raw),
            "clean_mad": _mad(clean),
            "extreme": int((raw.abs() > 0.20).sum()),
        })
    daily = pd.DataFrame(rows).dropna(subset=["raw_mad", "clean_mad"])
    daily["std_abs_delta"] = (daily["raw_std"] - daily["clean_std"]).abs()
    daily["mad_abs_delta"] = (daily["raw_mad"] - daily["clean_mad"]).abs()

    extreme = int((eq["ret1_raw"].abs() > 0.20).sum())
    raw_std_p99 = float(daily["std_abs_delta"].quantile(0.99)) if len(daily) else 999.0
    mad_p99 = float(daily["mad_abs_delta"].quantile(0.99)) if len(daily) else 999.0
    mad_max = float(daily["mad_abs_delta"].max()) if len(daily) else 999.0

    out_path = ROOT / "data" / "history" / "v6-layered" / "0A_MARKET" / "market_daily.parquet"
    out = pd.read_parquet(out_path)
    chk = daily[["date", "clean_mad", "extreme"]].merge(
        out[["date", "mkt_dispersion", "extreme_return_excluded"]],
        on="date",
        how="inner",
    )
    output_delta = float(
        (chk["clean_mad"] - chk["mkt_dispersion"]).abs().max()
    ) if len(chk) else 999.0
    excluded_match = bool(
        (chk["extreme"].astype(int) == chk["extreme_return_excluded"].astype(int)).all()
    ) if len(chk) else False

    passed = mad_p99 <= 0.001 and output_delta <= 1e-12 and excluded_match
    return {
        "extreme_abs_return_gt_20pct_rows": extreme,
        "daily_count": int(len(daily)),
        "raw_standard_deviation_p99_contamination_delta": raw_std_p99,
        "robust_mad_p99_delta_after_extreme_exclusion": mad_p99,
        "robust_mad_max_delta_after_extreme_exclusion": mad_max,
        "output_mkt_dispersion_max_recompute_delta": output_delta,
        "output_extreme_exclusion_count_match": excluded_match,
        "pass": passed,
        "interpretation": "raw standard deviation contamination is reported diagnostically; formal 0A uses robust MAD and excludes abs(return)>20% from market-state statistics",
    }


def main() -> int:
    old_checks, new_checks = verify_integrity()
    integrity_pass = all(x["sha256_match"] for x in old_checks + new_checks)

    frames = {}
    for y in range(2016, 2020):
        frames[y] = load_old_year(y)
    for y in range(2020, 2026):
        frames[y] = load_new_year(y)

    spots, source_errors, source_pass = source_spot_equivalence(frames)
    dispersion = dispersion_audit(frames)
    audit_pass = integrity_pass and source_pass and bool(dispersion["pass"])

    evidence = {
        "lane": "0A_MARKET",
        "audit": "source_lineage_coverage_and_dispersion_freeze",
        "status": "PASS" if audit_pass else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_pass": integrity_pass,
        "old_archive_checks": old_checks,
        "normalized_parquet_checks": new_checks,
        "official_source_spot_equivalence": spots,
        "official_source_errors": source_errors,
        "source_spot_equivalence_pass": source_pass,
        "dispersion_sensitivity": dispersion,
        "formal_oos_opened": False,
    }
    ep = AUDIT_DIR / "0A_MARKET_FREEZE.json"
    ep.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not MANIFEST.exists():
        raise RuntimeError("0A manifest missing; build market layer before audit")
    mp = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mp["status"] = "FROZEN_PASS" if audit_pass else "PROVISIONAL"
    mp["quality_gate_status"] = "PASS" if audit_pass else "PROVISIONAL_SOURCE_EQUIVALENCE_PENDING"
    mp.setdefault("pit_rules", {})
    mp["pit_rules"]["formal_oos_eligible"] = audit_pass
    mp["freeze_audit_path"] = str(ep.relative_to(ROOT))
    mp["freeze_blockers"] = [] if audit_pass else [
        "0A freeze audit did not pass; inspect source spot equivalence and dispersion sensitivity evidence"
    ]
    MANIFEST.write_text(json.dumps(mp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    progress = {
        "lane": "0A_MARKET",
        "status": "FROZEN_PASS" if audit_pass else "PROVISIONAL",
        "current_completed": 4 if audit_pass else 3,
        "target_total": 4,
        "completion_pct": 100.0 if audit_pass else 75.0,
        "remaining": 0 if audit_pass else 1,
        "current_blocker": None if audit_pass else "0A freeze audit failed",
        "freeze_audit_path": str(ep.relative_to(ROOT)),
        "formal_oos_opened": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)
    return 0 if audit_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
