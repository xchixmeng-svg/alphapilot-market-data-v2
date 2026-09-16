#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from lxml import html as lxml_html

ROOT = Path(__file__).resolve().parents[1]
LAYER_DIR = ROOT / "data" / "history" / "v6-layered" / "0B_INDUSTRY"
MANIFEST_DIR = ROOT / "data" / "history" / "v6-layered" / "manifests"
LAYER_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2016-01-01")
END = pd.Timestamp("2026-09-15")

TIP_FRONT = "https://taiwanindex.com.tw"
TIP_BACK = "https://backend.taiwanindex.com.tw"
TIP_DOWNLOAD = TIP_BACK + "/api/download/history"
TWSE_MI_INDEX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"

# Frozen before OOS. These are the 37 official TWSE industry-index series exposed by
# TIP's "發行量加權股價指數系列指數" historical selector at V6 preregistration time.
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
    "User-Agent": "Mozilla/5.0 AlphaPilot-V6-Stage0-Industry/1.0",
    "Accept": "text/html,text/csv,application/json,*/*",
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def generation_commit() -> str:
    env = os.getenv("GITHUB_SHA", "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def get(url: str, *, params=None, timeout=45) -> requests.Response:
    r = S.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r


def norm_text(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).replace("臺", "台")


def aliases_for(name: str) -> list[str]:
    return ALIASES.get(name, [name])


def resolve_official_code_map() -> tuple[dict[str, str], dict]:
    selector_html = get(TIP_FRONT + "/indexes/multipleHistory").text
    comparison_html = get(TIP_FRONT + "/indexes/comparison").text

    selector_missing = [
        name for name in FROZEN_INDUSTRY_NAMES
        if not any(norm_text(alias) in norm_text(selector_html) for alias in aliases_for(name))
    ]
    if selector_missing:
        raise RuntimeError(f"TIP selector missing frozen industry names: {selector_missing}")

    doc = lxml_html.fromstring(comparison_html)
    mapping: dict[str, str] = {}
    evidence = {}
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
        normalized_cells = {norm_text(x): x for x in cells}
        for canonical in FROZEN_INDUSTRY_NAMES:
            if canonical in mapping:
                continue
            for alias in aliases_for(canonical):
                if norm_text(alias) in normalized_cells:
                    mapping[canonical] = code
                    evidence[canonical] = {
                        "code": code,
                        "matched_cell": normalized_cells[norm_text(alias)],
                    }
                    break

    missing = [n for n in FROZEN_INDUSTRY_NAMES if n not in mapping]
    counts = {}
    for code in mapping.values():
        counts[code] = counts.get(code, 0) + 1
    duplicate_codes = sorted(code for code, n in counts.items() if n > 1)
    if missing or duplicate_codes or len(mapping) != len(FROZEN_INDUSTRY_NAMES):
        raise RuntimeError(
            f"industry code-map audit failed: mapped={len(mapping)} missing={missing} duplicate_codes={duplicate_codes}"
        )
    return mapping, {
        "frozen_universe_size": len(FROZEN_INDUSTRY_NAMES),
        "selector_missing": selector_missing,
        "mapping_evidence": evidence,
    }


def parse_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": None, "-": None, "--": None, "nan": None}),
        errors="coerce",
    )


def fetch_tip_history(code: str, name: str) -> pd.DataFrame:
    r = get(
        TIP_DOWNLOAD,
        params={
            "lang": "zh-tw",
            "code": code,
            "start": START.date().isoformat(),
            "end": END.date().isoformat(),
        },
        timeout=60,
    )
    ctype = (r.headers.get("content-type") or "").lower()
    if "csv" not in ctype and not r.content.startswith(b"\xef\xbb\xbf"):
        raise RuntimeError(f"{name}/{code}: unexpected content type {ctype}")

    df = pd.read_csv(io.BytesIO(r.content), encoding="utf-8-sig")
    col_norm = {norm_text(c).lower(): c for c in df.columns}
    date_col = next((col_norm[k] for k in col_norm if k in {"date", "日期"}), None)
    price_col = next((col_norm[k] for k in col_norm if k in {"priceindex", "價格指數"}), None)
    total_col = next((col_norm[k] for k in col_norm if k in {"totalreturnindex", "報酬指數"}), None)
    change_col = next((col_norm[k] for k in col_norm if k in {"change", "漲跌點數"}), None)
    pct_col = next((col_norm[k] for k in col_norm if k in {"%change", "change%", "漲跌百分比"}), None)
    if date_col is None or price_col is None:
        raise RuntimeError(f"{name}/{code}: required columns absent: {df.columns.tolist()}")

    n = len(df)
    nan_col = pd.Series([float("nan")] * n)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "index_code": code,
        "index_name": name,
        "price_index": parse_num(df[price_col]),
        "total_return_index": parse_num(df[total_col]) if total_col else nan_col,
        "change": parse_num(df[change_col]) if change_col else nan_col,
        "change_pct": parse_num(df[pct_col]) if pct_col else nan_col,
    })
    out = out.dropna(subset=["date", "price_index"]).drop_duplicates(["date", "index_code"], keep="last")
    out = out[(out["date"] >= START) & (out["date"] <= END)].sort_values("date").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"{name}/{code}: empty history")
    return out


def parse_twse_industry_on_date(d: pd.Timestamp) -> dict[str, float]:
    r = get(
        TWSE_MI_INDEX,
        params={"response": "json", "date": d.strftime("%Y%m%d"), "type": "IND"},
        timeout=30,
    )
    j = r.json()
    if str(j.get("stat", "")).upper() not in ("OK", ""):
        return {}
    out = {}
    for table in j.get("tables") or []:
        fields = [str(x).strip() for x in (table.get("fields") or [])]
        data = table.get("data") or []
        if not fields or not data:
            continue
        name_i = next((i for i, x in enumerate(fields) if x in ("指數", "指數名稱") or "指數" in x), None)
        close_i = next((i for i, x in enumerate(fields) if "收盤指數" in x or x == "收盤"), None)
        if name_i is None or close_i is None:
            continue
        for vals in data:
            if not isinstance(vals, list) or max(name_i, close_i) >= len(vals):
                continue
            name = str(vals[name_i]).strip()
            raw = str(vals[close_i]).replace(",", "").strip()
            try:
                out[norm_text(name)] = float(raw)
            except Exception:
                pass
    return out


def choose_crosscheck_dates(benchmark_dates: pd.Series) -> list[pd.Timestamp]:
    s = pd.Series(pd.to_datetime(benchmark_dates).dropna().sort_values().unique())
    chosen = []
    for year in (2021, 2022, 2023, 2024, 2025, 2026):
        y = s[s.dt.year == year]
        if not y.empty:
            chosen.append(pd.Timestamp(y.iloc[-1]))
    return chosen


def cross_source_spotcheck(long_df: pd.DataFrame, benchmark_dates: pd.Series) -> dict:
    rows = []
    transport_failures = []
    for d in choose_crosscheck_dates(benchmark_dates):
        try:
            twse = parse_twse_industry_on_date(d)
        except Exception as e:
            transport_failures.append({"date": d.date().isoformat(), "error": f"{type(e).__name__}: {e}"})
            continue
        day = long_df[long_df["date"] == d]
        for canonical in FROZEN_INDUSTRY_NAMES:
            tip_row = day[day["index_name"] == canonical]
            if tip_row.empty:
                continue
            twse_value = None
            matched_alias = None
            for alias in aliases_for(canonical):
                k = norm_text(alias)
                if k in twse:
                    twse_value = twse[k]
                    matched_alias = alias
                    break
            if twse_value is None:
                continue
            tip_value = float(tip_row.iloc[0]["price_index"])
            diff = abs(tip_value - float(twse_value))
            rows.append({
                "date": d.date().isoformat(),
                "index_name": canonical,
                "matched_twse_name": matched_alias,
                "tip": tip_value,
                "twse": float(twse_value),
                "abs_diff": diff,
                "match_within_0p02": diff <= 0.02 + 1e-12,
            })

    compared = len(rows)
    mismatches = sum(not x["match_within_0p02"] for x in rows)
    if compared >= 100:
        status = "PASS" if mismatches == 0 else "FAIL"
    else:
        status = "PROVISIONAL"
    return {
        "status": status,
        "compared_observations": compared,
        "mismatches": mismatches,
        "mismatch_rate": None if compared == 0 else mismatches / compared,
        "absolute_tolerance": 0.02,
        "transport_failures": transport_failures,
        "sample_failures": [x for x in rows if not x["match_within_0p02"]][:20],
    }


def main():
    code_map, universe_audit = resolve_official_code_map()

    benchmark = fetch_tip_history("t00", "__TAIEX_BENCHMARK__")
    benchmark_dates = benchmark["date"].drop_duplicates().sort_values().reset_index(drop=True)
    if benchmark_dates.empty:
        raise RuntimeError("empty t00 benchmark trading calendar")

    frames = []
    series_audit = {}
    for i, name in enumerate(FROZEN_INDUSTRY_NAMES, 1):
        code = code_map[name]
        df = fetch_tip_history(code, name)
        first = df["date"].min()
        last = df["date"].max()
        expected = benchmark_dates[(benchmark_dates >= first) & (benchmark_dates <= last)]
        observed_dates = set(df["date"].tolist())
        missing_dates = [d for d in expected.tolist() if d not in observed_dates]
        coverage = 1.0 if len(expected) == 0 else (len(expected) - len(missing_dates)) / len(expected)
        series_audit[name] = {
            "code": code,
            "date_start": first.date().isoformat(),
            "date_end": last.date().isoformat(),
            "rows": int(len(df)),
            "expected_benchmark_dates_post_start": int(len(expected)),
            "missing_benchmark_dates_post_start": int(len(missing_dates)),
            "post_start_date_coverage": float(coverage),
            "missing_date_sample": [d.date().isoformat() for d in missing_dates[:20]],
        }
        frames.append(df)
        print(f"[0B INDUSTRY] {i}/{len(FROZEN_INDUSTRY_NAMES)} {code} {name} rows={len(df)} coverage={coverage:.6f}", flush=True)

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.drop_duplicates(["date", "index_code"], keep="last").sort_values(["date", "index_code"]).reset_index(drop=True)

    union_dates = pd.Series(long_df["date"].drop_duplicates().sort_values().unique())
    eligible = benchmark_dates[(benchmark_dates >= START) & (benchmark_dates <= END)]
    union_set = set(pd.to_datetime(union_dates).tolist())
    missing_layer_dates = [d for d in eligible.tolist() if d not in union_set]
    layer_date_coverage = 1.0 if len(eligible) == 0 else (len(eligible) - len(missing_layer_dates)) / len(eligible)

    cross = cross_source_spotcheck(long_df, benchmark_dates)
    min_series_coverage = min(x["post_start_date_coverage"] for x in series_audit.values())
    quality_gate_status = "PASS"
    quality_failures = []
    if len(code_map) != len(FROZEN_INDUSTRY_NAMES):
        quality_gate_status = "FAIL"
        quality_failures.append("frozen universe mapping incomplete")
    if layer_date_coverage < 1.0:
        quality_gate_status = "FAIL"
        quality_failures.append(f"layer eligible-date coverage {layer_date_coverage}")
    if min_series_coverage < 1.0:
        quality_gate_status = "FAIL"
        quality_failures.append(f"min post-start series coverage {min_series_coverage}")
    if cross["status"] == "FAIL":
        quality_gate_status = "FAIL"
        quality_failures.append("TIP/TWSE numeric spot-check mismatch")
    elif cross["status"] == "PROVISIONAL" and quality_gate_status == "PASS":
        quality_gate_status = "PROVISIONAL"

    out_path = LAYER_DIR / "industry_index_daily.parquet"
    long_df.to_parquet(out_path, index=False)

    missingness = {
        c: {
            "missing_rows": int(long_df[c].isna().sum()),
            "missing_rate": float(long_df[c].isna().mean()),
        }
        for c in ["price_index", "total_return_index", "change", "change_pct"]
    }

    manifest = {
        "layer_id": "0B_INDUSTRY",
        "status": "PROVISIONAL",
        "schema_version": "v1",
        "generation_commit": generation_commit(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_lineage": [
            {
                "owner": "Taiwan Stock Exchange / Taiwan Index Plus",
                "transport": "TIP official historical CSV endpoint",
                "url": TIP_DOWNLOAD,
                "code_map_source": TIP_FRONT + "/indexes/comparison",
                "frozen_universe_source": TIP_FRONT + "/indexes/multipleHistory",
                "source_role": "PRIMARY_OFFICIAL",
            },
            {
                "owner": "Taiwan Stock Exchange",
                "transport": "MI_INDEX IND daily JSON",
                "url": TWSE_MI_INDEX,
                "source_role": "INDEPENDENT_NUMERIC_SPOTCHECK",
            },
        ],
        "datasets": [
            {
                "name": name,
                "index_code": code_map[name],
                "time_semantics": "decision_date_daily",
            }
            for name in FROZEN_INDUSTRY_NAMES
        ],
        "date_start": long_df["date"].min().date().isoformat(),
        "date_end": long_df["date"].max().date().isoformat(),
        "row_count": int(len(long_df)),
        "file_sha256": {out_path.name: sha256(out_path)},
        "missingness": missingness,
        "pit_rules": {
            "decision_date_definition": "TWSE trading date",
            "value_availability": "official closing index value published after the same trading session",
            "future_join_allowed": False,
            "forward_fill_in_layer_builder": False,
            "schema_universe_frozen_before_oos": True,
            "formal_oos_eligible": False,
            "reason_not_yet_eligible": "layer checkpoint/source-freeze not yet committed; manifest remains provisional even when quality gates pass",
        },
        "universe_audit": universe_audit,
        "coverage_audit": {
            "benchmark": "TIP t00 official trading-date history",
            "benchmark_date_count": int(len(eligible)),
            "layer_union_date_count": int(len(union_dates)),
            "layer_eligible_date_coverage": float(layer_date_coverage),
            "missing_layer_date_count": int(len(missing_layer_dates)),
            "missing_layer_date_sample": [d.date().isoformat() for d in missing_layer_dates[:20]],
            "minimum_post_start_series_coverage": float(min_series_coverage),
            "per_series": series_audit,
        },
        "cross_source_spotcheck": cross,
        "quality_gate_status": quality_gate_status,
        "quality_gate_failures": quality_failures,
    }

    manifest_path = MANIFEST_DIR / "0B_INDUSTRY.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if quality_gate_status == "FAIL":
        raise RuntimeError("0B industry quality gate FAIL: " + "; ".join(quality_failures))

    print(json.dumps({
        "layer": "0B_INDUSTRY",
        "status": manifest["status"],
        "quality_gate_status": quality_gate_status,
        "rows": manifest["row_count"],
        "industry_series": len(FROZEN_INDUSTRY_NAMES),
        "date_start": manifest["date_start"],
        "date_end": manifest["date_end"],
        "layer_date_coverage": layer_date_coverage,
        "min_series_coverage": min_series_coverage,
        "crosscheck_status": cross["status"],
        "crosscheck_n": cross["compared_observations"],
        "sha256": manifest["file_sha256"],
        "formal_oos_eligible": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
