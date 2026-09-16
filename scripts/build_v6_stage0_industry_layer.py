#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from lxml import html as lxml_html

ROOT = Path(__file__).resolve().parents[1]
LAYER_DIR = ROOT / "data" / "history" / "v6-layered" / "0B_INDUSTRY"
MANIFEST_DIR = ROOT / "data" / "history" / "v6-layered" / "manifests"
CACHE_ROOT = ROOT / ".cache" / "v6-stage0-industry"
OLD_CACHE = CACHE_ROOT / "twse_old_daily"
TIP_CACHE = CACHE_ROOT / "tip_recent"
for p in (LAYER_DIR, MANIFEST_DIR, OLD_CACHE, TIP_CACHE):
    p.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2016-01-01")
OLD_END = pd.Timestamp("2020-12-31")
TIP_START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-09-15")
MAX_OLD_DATES_PER_RUN = int(os.getenv("V6_OLD_IND_MAX_DATES", "120"))
OLD_REQUEST_DELAY_SECONDS = float(os.getenv("V6_OLD_IND_DELAY", "0.55"))

TIP_FRONT = "https://taiwanindex.com.tw"
TIP_DOWNLOAD = "https://backend.taiwanindex.com.tw/api/download/history"
TWSE_MI_INDEX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"

FROZEN_INDUSTRY_NAMES = [
    "食品類指數", "紡織纖維類指數", "造紙類指數", "建材營造類指數",
    "金融保險類指數", "水泥窯製類指數", "塑膠化工類指數", "機電類指數",
    "電子類指數", "水泥類指數", "塑膠類指數", "電機機械類指數",
    "電器電纜類指數", "化學生技醫療類指數", "玻璃陶瓷類指數", "鋼鐵類指數",
    "橡膠類指數", "汽車類指數", "航運類指數", "觀光類指數",
    "貿易百貨類指數", "半導體類指數", "電腦及週邊設備類指數", "光電類指數",
    "通信網路類指數", "電子零組件類指數", "電子通路類指數", "資訊服務類指數",
    "其他電子類指數", "化學類指數", "生技醫療類指數", "油電燃氣類指數",
    "其他類指數", "綠能環保類指數", "數位雲端類指數", "運動休閒類指數",
    "居家生活類指數",
]
ALIASES = {
    "觀光類指數": ["觀光類指數", "觀光餐旅類指數"],
}

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 AlphaPilot-V6-Stage0-Industry/2.0",
    "Accept": "application/json,text/csv,text/html,*/*",
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def generation_commit() -> str:
    if os.getenv("GITHUB_SHA", "").strip():
        return os.environ["GITHUB_SHA"].strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def get(url: str, *, params=None, timeout=45, tries=2) -> requests.Response:
    last = None
    for attempt in range(tries):
        try:
            r = S.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if attempt + 1 < tries:
                time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"GET failed {url}: {type(last).__name__}: {last}")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).replace("臺", "台")


def aliases_for(name: str) -> list[str]:
    return ALIASES.get(name, [name])


def canonical_from_raw(raw: str) -> str | None:
    n = norm_text(raw)
    for canonical in FROZEN_INDUSTRY_NAMES:
        if any(n == norm_text(a) for a in aliases_for(canonical)):
            return canonical
    return None


def resolve_official_code_map() -> tuple[dict[str, str], dict]:
    selector_html = get(TIP_FRONT + "/indexes/multipleHistory").text
    comparison_html = get(TIP_FRONT + "/indexes/comparison").text
    selector_norm = norm_text(selector_html)
    selector_missing = [
        n for n in FROZEN_INDUSTRY_NAMES
        if not any(norm_text(a) in selector_norm for a in aliases_for(n))
    ]
    if selector_missing:
        raise RuntimeError(f"TIP selector missing frozen industry names: {selector_missing}")

    doc = lxml_html.fromstring(comparison_html)
    mapping: dict[str, str] = {}
    evidence: dict[str, dict] = {}
    for tr in doc.xpath("//tr"):
        cells = [re.sub(r"\s+", " ", "".join(td.itertext())).strip() for td in tr.xpath("./td")]
        hrefs = tr.xpath(".//a/@href")
        codes = []
        for href in hrefs:
            m = re.fullmatch(r"/indexes/([A-Za-z0-9_-]+)", href.split("?")[0])
            if m:
                codes.append(m.group(1))
        if not codes:
            continue
        code = codes[0]
        ncells = {norm_text(x): x for x in cells}
        for canonical in FROZEN_INDUSTRY_NAMES:
            if canonical in mapping:
                continue
            for alias in aliases_for(canonical):
                if norm_text(alias) in ncells:
                    mapping[canonical] = code
                    evidence[canonical] = {"code": code, "matched_cell": ncells[norm_text(alias)]}
                    break
    missing = [n for n in FROZEN_INDUSTRY_NAMES if n not in mapping]
    counts: dict[str, int] = {}
    for code in mapping.values():
        counts[code] = counts.get(code, 0) + 1
    dup = sorted(k for k, v in counts.items() if v > 1)
    if missing or dup or len(mapping) != 37:
        raise RuntimeError(f"industry code map failed: mapped={len(mapping)} missing={missing} duplicate_codes={dup}")
    return mapping, {"frozen_universe_size": 37, "selector_missing": selector_missing, "mapping_evidence": evidence}


def parse_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
        .replace({"": None, "-": None, "--": None, "nan": None}), errors="coerce"
    )


def parse_tip_csv(content: bytes, code: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    cols = {norm_text(c).lower(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if k in {"date", "日期"}), None)
    price_col = next((cols[k] for k in cols if k in {"priceindex", "價格指數"}), None)
    total_col = next((cols[k] for k in cols if k in {"totalreturnindex", "報酬指數"}), None)
    change_col = next((cols[k] for k in cols if k in {"change", "漲跌點數"}), None)
    pct_col = next((cols[k] for k in cols if k in {"%change", "change%", "漲跌百分比"}), None)
    if date_col is None or price_col is None:
        raise RuntimeError(f"{name}/{code}: required TIP columns absent: {df.columns.tolist()}")
    n = len(df)
    nan = pd.Series([float("nan")] * n)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "index_code": code,
        "index_name": name,
        "price_index": parse_num_series(df[price_col]),
        "total_return_index": parse_num_series(df[total_col]) if total_col else nan,
        "change": parse_num_series(df[change_col]) if change_col else nan,
        "change_pct": parse_num_series(df[pct_col]) if pct_col else nan,
        "source": "TIP_OFFICIAL_CSV",
    })
    return out.dropna(subset=["date", "price_index"])


def fetch_tip_series(code: str, name: str) -> pd.DataFrame:
    r = get(TIP_DOWNLOAD, params={
        "lang": "zh-tw", "code": code,
        "start": TIP_START.date().isoformat(), "end": END.date().isoformat(),
    }, timeout=60, tries=2)
    ctype = (r.headers.get("content-type") or "").lower()
    if "csv" not in ctype and not r.content.startswith(b"\xef\xbb\xbf"):
        raise RuntimeError(f"{name}/{code}: unexpected TIP content-type={ctype}")
    out = parse_tip_csv(r.content, code, name)
    out = out[(out["date"] >= TIP_START) & (out["date"] <= END)]
    if out.empty:
        raise RuntimeError(f"{name}/{code}: empty TIP recent history")
    return out.sort_values("date").reset_index(drop=True)


def load_or_build_tip_recent(code_map: dict[str, str]) -> tuple[pd.DataFrame, pd.Series, dict]:
    key = hashlib.sha256(json.dumps(code_map, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    p = TIP_CACHE / f"tip_2021_2026_{key}.parquet"
    meta_p = TIP_CACHE / f"tip_2021_2026_{key}.json"
    if p.exists() and meta_p.exists():
        df = pd.read_parquet(p)
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if set(df["index_name"].unique()) == set(FROZEN_INDUSTRY_NAMES) and str(meta.get("end")) == END.date().isoformat():
            print(f"[0B TIP CACHE] rows={len(df)} series={df['index_name'].nunique()}", flush=True)
        else:
            p.unlink(missing_ok=True); meta_p.unlink(missing_ok=True); df = None
    else:
        df = None
    if df is None:
        frames = []
        for i, name in enumerate(FROZEN_INDUSTRY_NAMES, 1):
            part = fetch_tip_series(code_map[name], name)
            frames.append(part)
            print(f"[0B TIP] {i}/37 {code_map[name]} {name} rows={len(part)} {part['date'].min().date()}..{part['date'].max().date()}", flush=True)
            time.sleep(0.12)
        df = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "index_code"], keep="last")
        df = df.sort_values(["date", "index_code"]).reset_index(drop=True)
        df.to_parquet(p, index=False)
        meta_p.write_text(json.dumps({
            "start": TIP_START.date().isoformat(), "end": END.date().isoformat(),
            "rows": len(df), "series": int(df["index_name"].nunique()), "sha256": sha256(p),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    # t00 is only used as the recent official trading calendar / overlap audit reference.
    bench_p = TIP_CACHE / "t00_2021_2026.parquet"
    if bench_p.exists():
        benchmark = pd.read_parquet(bench_p)
    else:
        benchmark = fetch_tip_series("t00", "__TAIEX_BENCHMARK__")
        benchmark.to_parquet(bench_p, index=False)
    benchmark_dates = benchmark["date"].drop_duplicates().sort_values().reset_index(drop=True)
    return df, benchmark_dates, {"cache_file": str(p.relative_to(ROOT)), "cache_sha256": sha256(p)}


def parse_scalar(x):
    if x is None:
        return None
    s = str(x).replace(",", "").replace("%", "").strip()
    if s in {"", "-", "--", "nan", "None"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_twse_json(j: dict, d: pd.Timestamp, code_map: dict[str, str]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    raw_names: list[str] = []
    for table in j.get("tables") or []:
        if not isinstance(table, dict):
            continue
        fields = [str(x).strip() for x in (table.get("fields") or [])]
        data = table.get("data") or []
        if not fields or not data:
            continue
        name_i = next((i for i, x in enumerate(fields) if x in ("指數", "指數名稱") or "指數" in x), None)
        close_i = next((i for i, x in enumerate(fields) if "收盤指數" in x or x == "收盤"), None)
        pct_i = next((i for i, x in enumerate(fields) if "漲跌百分比" in x or "漲跌幅" in x), None)
        change_i = next((i for i, x in enumerate(fields) if "漲跌點數" in x or "漲跌價差" in x), None)
        if name_i is None or close_i is None:
            continue
        for vals in data:
            if not isinstance(vals, list) or max(name_i, close_i) >= len(vals):
                continue
            raw = str(vals[name_i]).strip()
            if not raw:
                continue
            raw_names.append(raw)
            canonical = canonical_from_raw(raw)
            if canonical is None:
                continue
            close = parse_scalar(vals[close_i])
            if close is None:
                continue
            rows.append({
                "date": d,
                "index_code": code_map[canonical],
                "index_name": canonical,
                "price_index": close,
                "total_return_index": float("nan"),
                "change": parse_scalar(vals[change_i]) if change_i is not None and change_i < len(vals) else None,
                "change_pct": parse_scalar(vals[pct_i]) if pct_i is not None and pct_i < len(vals) else None,
                "source": "TWSE_MI_INDEX_OFFICIAL",
            })
    rows = list({(r["date"], r["index_name"]): r for r in rows}.values())
    return rows, sorted(set(raw_names))


def old_cache_path(d: pd.Timestamp) -> Path:
    return OLD_CACHE / f"{d.date().isoformat()}.json"


def load_old_checkpoint(d: pd.Timestamp) -> dict | None:
    p = old_cache_path(d)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("status") in {"SUCCESS", "VERIFIED_NO_DATA"}:
            return obj
    except Exception:
        pass
    p.unlink(missing_ok=True)
    return None


def save_old_checkpoint(d: pd.Timestamp, obj: dict):
    p = old_cache_path(d)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(p)


def fetch_old_twse_date(d: pd.Timestamp, code_map: dict[str, str]) -> dict:
    r = get(TWSE_MI_INDEX, params={"response": "json", "date": d.strftime("%Y%m%d"), "type": "IND"}, timeout=30, tries=2)
    ctype = (r.headers.get("content-type") or "").lower()
    if "json" not in ctype:
        raise RuntimeError(f"{d.date()}: HTTP 200/non-JSON is not valid no-data: {ctype}")
    j = r.json()
    stat = str(j.get("stat", "")).strip()
    if stat.upper() == "OK":
        rows, raw_names = parse_twse_json(j, d, code_map)
        if not rows:
            raise RuntimeError(f"{d.date()}: stat OK but zero frozen industry rows")
        return {
            "date": d.date().isoformat(), "status": "SUCCESS", "stat": stat,
            "rows": rows, "frozen_rows": len(rows), "raw_index_name_count": len(raw_names),
        }
    no_data_markers = ("沒有符合條件", "查無資料", "無資料")
    if any(x in stat for x in no_data_markers):
        return {"date": d.date().isoformat(), "status": "VERIFIED_NO_DATA", "stat": stat, "rows": []}
    raise RuntimeError(f"{d.date()}: unclassified TWSE stat={stat!r}")


def old_weekdays() -> list[pd.Timestamp]:
    return list(pd.date_range(START, OLD_END, freq="B"))


def advance_old_history(code_map: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    targets = old_weekdays()
    completed = {d.date().isoformat(): load_old_checkpoint(d) for d in targets}
    unresolved = [d for d in targets if completed[d.date().isoformat()] is None]
    batch = unresolved[:MAX_OLD_DATES_PER_RUN]
    fresh_success = fresh_no_data = transient_fail = 0
    errors = []
    print(f"[0B OLD RESUME] targets={len(targets)} completed={len(targets)-len(unresolved)} unresolved={len(unresolved)} this_run={len(batch)}", flush=True)
    for i, d in enumerate(batch, 1):
        try:
            obj = fetch_old_twse_date(d, code_map)
            save_old_checkpoint(d, obj)
            completed[d.date().isoformat()] = obj
            if obj["status"] == "SUCCESS": fresh_success += 1
            else: fresh_no_data += 1
        except Exception as e:
            transient_fail += 1
            errors.append({"date": d.date().isoformat(), "error": f"{type(e).__name__}: {e}"})
        if i < len(batch):
            time.sleep(OLD_REQUEST_DELAY_SECONDS)
        if i % 20 == 0 or i == len(batch):
            print(f"[0B OLD PROGRESS] {i}/{len(batch)} success={fresh_success} verified_no_data={fresh_no_data} transient_fail={transient_fail}", flush=True)

    completed_now = {d.date().isoformat(): load_old_checkpoint(d) for d in targets}
    unresolved_after = [d for d in targets if completed_now[d.date().isoformat()] is None]
    rows = []
    success_dates = no_data_dates = 0
    for d in targets:
        obj = completed_now[d.date().isoformat()]
        if obj is None:
            continue
        if obj["status"] == "SUCCESS":
            success_dates += 1
            for r in obj.get("rows") or []:
                rr = dict(r)
                rr["date"] = pd.Timestamp(rr["date"])
                rows.append(rr)
        elif obj["status"] == "VERIFIED_NO_DATA":
            no_data_dates += 1
    df = pd.DataFrame(rows, columns=["date","index_code","index_name","price_index","total_return_index","change","change_pct","source"])
    audit = {
        "target_weekdays": len(targets),
        "completed_units": len(targets) - len(unresolved_after),
        "success_trading_dates": success_dates,
        "verified_no_data_dates": no_data_dates,
        "unresolved_units": len(unresolved_after),
        "unresolved_sample": [d.date().isoformat() for d in unresolved_after[:30]],
        "this_run_requested": len(batch),
        "this_run_success": fresh_success,
        "this_run_verified_no_data": fresh_no_data,
        "this_run_transient_fail": transient_fail,
        "this_run_errors": errors[:30],
        "complete": len(unresolved_after) == 0,
        "checkpoint_root": str(OLD_CACHE.relative_to(ROOT)),
    }
    return df, audit


def cross_source_equivalence(tip_df: pd.DataFrame, code_map: dict[str, str]) -> dict:
    # Six 2021-2026 dates x up to 37 series => enough observations across >3 years.
    candidates = []
    tip_dates = pd.Series(tip_df["date"].drop_duplicates().sort_values().unique())
    for year in (2021, 2022, 2023, 2024, 2025, 2026):
        y = tip_dates[tip_dates.dt.year == year]
        if not y.empty:
            candidates.append(pd.Timestamp(y.iloc[-1]))
    comparisons, transport_failures = [], []
    for d in candidates:
        try:
            obj = fetch_old_twse_date(d, code_map)
        except Exception as e:
            transport_failures.append({"date": d.date().isoformat(), "error": f"{type(e).__name__}: {e}"})
            continue
        if obj["status"] != "SUCCESS":
            continue
        twse = {r["index_name"]: float(r["price_index"]) for r in obj["rows"]}
        day = tip_df[tip_df["date"] == d]
        for name, tv in twse.items():
            x = day[day["index_name"] == name]
            if x.empty:
                continue
            tip_value = float(x.iloc[0]["price_index"])
            diff = abs(tip_value - tv)
            comparisons.append({"date": d.date().isoformat(), "index_name": name, "tip": tip_value, "twse": tv, "abs_diff": diff})
        time.sleep(0.25)
    n = len(comparisons)
    mismatches = [x for x in comparisons if x["abs_diff"] > 0.02 + 1e-12]
    years = len(set(x["date"][:4] for x in comparisons))
    status = "PASS" if n >= 60 and years >= 3 and not mismatches else ("FAIL" if n >= 60 and years >= 3 else "PROVISIONAL")
    return {
        "status": status, "compared_observations": n, "distinct_calendar_years": years,
        "absolute_tolerance": 0.02, "mismatches": len(mismatches),
        "mismatch_rate": None if n == 0 else len(mismatches) / n,
        "transport_failures": transport_failures,
        "sample_failures": mismatches[:20],
        "note": "0.02 is a display-rounding equivalence audit for index levels, not the macro fallback tolerance.",
    }


def coverage_audit(long_df: pd.DataFrame, full_trading_dates: pd.Series, code_map: dict[str, str]) -> tuple[dict, list[str]]:
    failures = []
    per_series = {}
    for name in FROZEN_INDUSTRY_NAMES:
        df = long_df[long_df["index_name"] == name].sort_values("date")
        if df.empty:
            failures.append(f"{name}: no observations")
            continue
        first, last = df["date"].min(), df["date"].max()
        expected = full_trading_dates[(full_trading_dates >= first) & (full_trading_dates <= last)]
        obs = set(df["date"].tolist())
        missing = [d for d in expected.tolist() if d not in obs]
        coverage = 1.0 if len(expected) == 0 else (len(expected)-len(missing))/len(expected)
        per_series[name] = {
            "code": code_map[name], "date_start": first.date().isoformat(), "date_end": last.date().isoformat(),
            "rows": int(len(df)), "expected_dates_post_start": int(len(expected)),
            "missing_dates_post_start": int(len(missing)), "post_start_date_coverage": float(coverage),
            "missing_date_sample": [d.date().isoformat() for d in missing[:20]],
        }
        if coverage < 1.0:
            failures.append(f"{name}: post-start coverage={coverage}")
    union = set(long_df["date"].tolist())
    missing_layer = [d for d in full_trading_dates.tolist() if d not in union]
    layer_cov = 1.0 if len(full_trading_dates) == 0 else (len(full_trading_dates)-len(missing_layer))/len(full_trading_dates)
    if layer_cov < 1.0:
        failures.append(f"layer date coverage={layer_cov}")
    return {
        "benchmark": "TWSE MI_INDEX successful trading dates 2016-2020 + TIP t00 dates 2021-2026",
        "benchmark_date_count": int(len(full_trading_dates)),
        "layer_union_date_count": int(long_df["date"].nunique()),
        "layer_eligible_date_coverage": float(layer_cov),
        "missing_layer_date_count": len(missing_layer),
        "missing_layer_date_sample": [d.date().isoformat() for d in missing_layer[:20]],
        "minimum_post_start_series_coverage": min((x["post_start_date_coverage"] for x in per_series.values()), default=0.0),
        "per_series": per_series,
    }, failures


def main():
    code_map, universe_audit = resolve_official_code_map()
    tip_df, tip_benchmark_dates, tip_cache_audit = load_or_build_tip_recent(code_map)
    old_df, old_audit = advance_old_history(code_map)

    frames = [tip_df]
    if not old_df.empty:
        frames.insert(0, old_df)
    long_df = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "index_code"], keep="last")
    long_df = long_df.sort_values(["date", "index_code"]).reset_index(drop=True)

    old_trading_dates = pd.Series(old_df["date"].drop_duplicates().sort_values().unique()) if not old_df.empty else pd.Series([], dtype="datetime64[ns]")
    full_trading_dates = pd.Series(pd.to_datetime(pd.concat([old_trading_dates, tip_benchmark_dates], ignore_index=True)).drop_duplicates().sort_values().reset_index(drop=True))

    if old_audit["complete"]:
        equivalence = cross_source_equivalence(tip_df, code_map)
        coverage, coverage_failures = coverage_audit(long_df, full_trading_dates, code_map)
        failures = list(coverage_failures)
        if equivalence["status"] == "FAIL":
            failures.append("TIP/TWSE overlap numeric equivalence failed")
        quality = "FAIL" if failures else ("PASS" if equivalence["status"] == "PASS" else "PROVISIONAL")
        status = "PROVISIONAL"
    else:
        equivalence = {"status": "NOT_RUN_UNTIL_OLD_HISTORY_COMPLETE"}
        coverage = {
            "benchmark": "partial checkpoint; final coverage audit deferred until old-history complete",
            "benchmark_date_count": int(len(full_trading_dates)),
            "layer_union_date_count": int(long_df["date"].nunique()),
            "layer_eligible_date_coverage": None,
            "minimum_post_start_series_coverage": None,
            "per_series": {},
        }
        failures = []
        quality = "BUILDING"
        status = "BUILDING"

    out_path = LAYER_DIR / "industry_index_daily.parquet"
    long_df.to_parquet(out_path, index=False)
    missingness = {
        c: {"missing_rows": int(long_df[c].isna().sum()), "missing_rate": float(long_df[c].isna().mean())}
        for c in ["price_index", "total_return_index", "change", "change_pct"]
    }
    manifest = {
        "layer_id": "0B_INDUSTRY", "status": status, "schema_version": "v2-hybrid-official",
        "generation_commit": generation_commit(), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_lineage": [
            {"owner": "Taiwan Stock Exchange", "transport": "MI_INDEX IND per-date official JSON",
             "url": TWSE_MI_INDEX, "coverage_role": "PRIMARY_2016_2020"},
            {"owner": "Taiwan Stock Exchange / Taiwan Index Plus", "transport": "TIP official historical CSV endpoint",
             "url": TIP_DOWNLOAD, "code_map_source": TIP_FRONT + "/indexes/comparison",
             "frozen_universe_source": TIP_FRONT + "/indexes/multipleHistory", "coverage_role": "PRIMARY_2021_2026"},
        ],
        "datasets": [{"name": n, "index_code": code_map[n], "time_semantics": "decision_date_daily"} for n in FROZEN_INDUSTRY_NAMES],
        "date_start": long_df["date"].min().date().isoformat(), "date_end": long_df["date"].max().date().isoformat(),
        "row_count": int(len(long_df)), "file_sha256": {out_path.name: sha256(out_path)}, "missingness": missingness,
        "pit_rules": {
            "decision_date_definition": "TWSE trading date",
            "value_availability": "official closing index value published after the same trading session",
            "future_join_allowed": False, "forward_fill_in_layer_builder": False,
            "schema_universe_frozen_before_oos": True, "formal_oos_eligible": False,
            "reason_not_yet_eligible": "0B remains BUILDING/PROVISIONAL until complete checkpoint coverage, overlap equivalence, source freeze and lock",
        },
        "universe_audit": universe_audit,
        "tip_recent_cache_audit": tip_cache_audit,
        "old_history_checkpoint_audit": old_audit,
        "cross_source_equivalence": equivalence,
        "coverage_audit": coverage,
        "quality_gate_status": quality,
        "quality_gate_failures": failures,
    }
    mp = MANIFEST_DIR / "0B_INDUSTRY.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if quality == "FAIL":
        raise RuntimeError("0B industry quality gate FAIL: " + "; ".join(failures))

    print(json.dumps({
        "layer": "0B_INDUSTRY", "status": status, "quality_gate_status": quality,
        "rows": int(len(long_df)), "industry_series": int(long_df["index_name"].nunique()),
        "date_start": manifest["date_start"], "date_end": manifest["date_end"],
        "old_completed_units": old_audit["completed_units"], "old_target_weekdays": old_audit["target_weekdays"],
        "old_unresolved_units": old_audit["unresolved_units"], "this_run_requested": old_audit["this_run_requested"],
        "this_run_success": old_audit["this_run_success"], "this_run_verified_no_data": old_audit["this_run_verified_no_data"],
        "this_run_transient_fail": old_audit["this_run_transient_fail"],
        "cross_source_equivalence": equivalence.get("status"),
        "formal_oos_eligible": False,
        "sha256": manifest["file_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
