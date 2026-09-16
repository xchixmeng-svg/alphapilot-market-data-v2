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
LAYER_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2016-01-01")
END = pd.Timestamp("2026-09-15")
TIP_FRONT = "https://taiwanindex.com.tw"
TIP_BACK = "https://backend.taiwanindex.com.tw"
TIP_DOWNLOAD = TIP_BACK + "/api/download/history"
TWSE_MI_INDEX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"

# Frozen before formal OOS. The names are the research schema; codes are resolved
# from the official TIP comparison page and audited on every build.
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
ALIASES = {"觀光類指數": ["觀光類指數", "觀光餐旅類指數"]}
DOWNLOAD_AUDIT: dict[str, dict] = {}

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 AlphaPilot-V6-Stage0-Industry/1.1",
    "Accept": "text/html,text/csv,application/json,*/*",
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


def get(url: str, *, params=None, timeout=45, tries=3) -> requests.Response:
    last = None
    for attempt in range(tries):
        try:
            r = S.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if attempt + 1 < tries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed {url}: {type(last).__name__}: {last}")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).replace("臺", "台")


def aliases_for(name: str) -> list[str]:
    return ALIASES.get(name, [name])


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
        normalized_cells = {norm_text(x): x for x in cells}
        for canonical in FROZEN_INDUSTRY_NAMES:
            if canonical in mapping:
                continue
            for alias in aliases_for(canonical):
                if norm_text(alias) in normalized_cells:
                    mapping[canonical] = code
                    evidence[canonical] = {"code": code, "matched_cell": normalized_cells[norm_text(alias)]}
                    break

    missing = [n for n in FROZEN_INDUSTRY_NAMES if n not in mapping]
    counts: dict[str, int] = {}
    for code in mapping.values():
        counts[code] = counts.get(code, 0) + 1
    duplicate_codes = sorted(k for k, v in counts.items() if v > 1)
    if missing or duplicate_codes or len(mapping) != 37:
        raise RuntimeError(f"industry code map failed: mapped={len(mapping)} missing={missing} dup={duplicate_codes}")
    return mapping, {"frozen_universe_size": 37, "selector_missing": selector_missing, "mapping_evidence": evidence}


def parse_num(s: pd.Series) -> pd.Series:
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
        raise RuntimeError(f"{name}/{code}: required columns absent: {df.columns.tolist()}")
    n = len(df)
    nan = pd.Series([float("nan")] * n)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "index_code": code,
        "index_name": name,
        "price_index": parse_num(df[price_col]),
        "total_return_index": parse_num(df[total_col]) if total_col else nan,
        "change": parse_num(df[change_col]) if change_col else nan,
        "change_pct": parse_num(df[pct_col]) if pct_col else nan,
    })
    return out.dropna(subset=["date", "price_index"])


def request_tip_slice(code: str, name: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    r = get(TIP_DOWNLOAD, params={
        "lang": "zh-tw", "code": code,
        "start": start.date().isoformat(), "end": end.date().isoformat(),
    }, timeout=60, tries=2)
    ctype = (r.headers.get("content-type") or "").lower()
    if "csv" not in ctype and not r.content.startswith(b"\xef\xbb\xbf"):
        raise RuntimeError(f"{name}/{code}: unexpected content-type={ctype}")
    df = parse_tip_csv(r.content, code, name)
    return df, {
        "request_start": start.date().isoformat(), "request_end": end.date().isoformat(),
        "rows": int(len(df)), "bytes": len(r.content),
        "actual_start": None if df.empty else df["date"].min().date().isoformat(),
        "actual_end": None if df.empty else df["date"].max().date().isoformat(),
    }


def primary_chunks() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks = []
    y = START.year
    while y <= END.year:
        a = max(START, pd.Timestamp(f"{y}-01-01"))
        b = min(END, pd.Timestamp(f"{min(y + 2, END.year)}-12-31"))
        chunks.append((a, b))
        y += 3
    return chunks


def fetch_tip_history(code: str, name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    evidence: list[dict] = []
    for start, end in primary_chunks():
        try:
            df, ev = request_tip_slice(code, name, start, end)
            frames.append(df)
            evidence.append({**ev, "mode": "3Y_PRIMARY", "ok": True})
        except Exception as e:
            evidence.append({
                "request_start": start.date().isoformat(), "request_end": end.date().isoformat(),
                "mode": "3Y_PRIMARY", "ok": False, "error": f"{type(e).__name__}: {e}",
            })
            for year in range(start.year, end.year + 1):
                ys = max(start, pd.Timestamp(f"{year}-01-01"))
                ye = min(end, pd.Timestamp(f"{year}-12-31"))
                try:
                    df, ev = request_tip_slice(code, name, ys, ye)
                    frames.append(df)
                    evidence.append({**ev, "mode": "1Y_FALLBACK", "ok": True})
                except Exception as e2:
                    evidence.append({
                        "request_start": ys.date().isoformat(), "request_end": ye.date().isoformat(),
                        "mode": "1Y_FALLBACK", "ok": False, "error": f"{type(e2).__name__}: {e2}",
                    })
                    raise RuntimeError(f"{name}/{code}: yearly fallback failed {year}: {e2}") from e2
    if not frames:
        raise RuntimeError(f"{name}/{code}: no history chunks")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["date", "index_code"], keep="last")
    out = out[(out["date"] >= START) & (out["date"] <= END)].sort_values("date").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"{name}/{code}: empty after chunk merge")
    DOWNLOAD_AUDIT[name] = {
        "code": code, "chunk_count": len(evidence), "chunks": evidence,
        "merged_rows": int(len(out)),
        "merged_start": out["date"].min().date().isoformat(),
        "merged_end": out["date"].max().date().isoformat(),
    }
    return out


def parse_twse_industry_on_date(d: pd.Timestamp) -> dict[str, float]:
    r = get(TWSE_MI_INDEX, params={"response": "json", "date": d.strftime("%Y%m%d"), "type": "IND"}, timeout=30, tries=2)
    j = r.json()
    if str(j.get("stat", "")).upper() not in ("OK", ""):
        return {}
    out: dict[str, float] = {}
    for table in j.get("tables") or []:
        fields = [str(x).strip() for x in (table.get("fields") or [])]
        data = table.get("data") or []
        name_i = next((i for i, x in enumerate(fields) if x in ("指數", "指數名稱") or "指數" in x), None)
        close_i = next((i for i, x in enumerate(fields) if "收盤指數" in x or x == "收盤"), None)
        if name_i is None or close_i is None:
            continue
        for vals in data:
            if not isinstance(vals, list) or max(name_i, close_i) >= len(vals):
                continue
            try:
                out[norm_text(vals[name_i])] = float(str(vals[close_i]).replace(",", "").strip())
            except Exception:
                pass
    return out


def choose_crosscheck_dates(dates: pd.Series) -> list[pd.Timestamp]:
    s = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    out = []
    for year in (2021, 2022, 2023, 2024, 2025, 2026):
        y = s[s.dt.year == year]
        if not y.empty:
            out.append(pd.Timestamp(y.iloc[-1]))
    return out


def cross_source_spotcheck(long_df: pd.DataFrame, benchmark_dates: pd.Series) -> dict:
    rows, transport_failures = [], []
    for d in choose_crosscheck_dates(benchmark_dates):
        try:
            twse = parse_twse_industry_on_date(d)
        except Exception as e:
            transport_failures.append({"date": d.date().isoformat(), "error": f"{type(e).__name__}: {e}"})
            continue
        day = long_df[long_df["date"] == d]
        for canonical in FROZEN_INDUSTRY_NAMES:
            tip = day[day["index_name"] == canonical]
            if tip.empty:
                continue
            twse_value, matched_alias = None, None
            for alias in aliases_for(canonical):
                if norm_text(alias) in twse:
                    twse_value, matched_alias = twse[norm_text(alias)], alias
                    break
            if twse_value is None:
                continue
            tip_value = float(tip.iloc[0]["price_index"])
            diff = abs(tip_value - float(twse_value))
            rows.append({
                "date": d.date().isoformat(), "index_name": canonical, "matched_twse_name": matched_alias,
                "tip": tip_value, "twse": float(twse_value), "abs_diff": diff,
                "match_within_0p02": diff <= 0.02 + 1e-12,
            })
    n = len(rows)
    mismatches = sum(not x["match_within_0p02"] for x in rows)
    status = "PROVISIONAL" if n < 100 else ("PASS" if mismatches == 0 else "FAIL")
    return {
        "status": status, "compared_observations": n, "mismatches": mismatches,
        "mismatch_rate": None if n == 0 else mismatches / n,
        "absolute_tolerance": 0.02, "transport_failures": transport_failures,
        "sample_failures": [x for x in rows if not x["match_within_0p02"]][:20],
    }


def main():
    code_map, universe_audit = resolve_official_code_map()
    benchmark = fetch_tip_history("t00", "__TAIEX_BENCHMARK__")
    benchmark_dates = benchmark["date"].drop_duplicates().sort_values().reset_index(drop=True)
    if benchmark_dates.empty:
        raise RuntimeError("empty t00 benchmark trading calendar")

    frames, series_audit = [], {}
    for i, name in enumerate(FROZEN_INDUSTRY_NAMES, 1):
        code = code_map[name]
        df = fetch_tip_history(code, name)
        first, last = df["date"].min(), df["date"].max()
        expected = benchmark_dates[(benchmark_dates >= first) & (benchmark_dates <= last)]
        observed = set(df["date"].tolist())
        missing = [d for d in expected.tolist() if d not in observed]
        coverage = 1.0 if len(expected) == 0 else (len(expected) - len(missing)) / len(expected)
        series_audit[name] = {
            "code": code, "date_start": first.date().isoformat(), "date_end": last.date().isoformat(),
            "rows": int(len(df)), "expected_benchmark_dates_post_start": int(len(expected)),
            "missing_benchmark_dates_post_start": int(len(missing)), "post_start_date_coverage": float(coverage),
            "missing_date_sample": [d.date().isoformat() for d in missing[:20]],
        }
        frames.append(df)
        print(f"[0B INDUSTRY] {i}/37 {code} {name} rows={len(df)} coverage={coverage:.6f}", flush=True)

    long_df = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "index_code"], keep="last")
    long_df = long_df.sort_values(["date", "index_code"]).reset_index(drop=True)
    union_dates = pd.Series(long_df["date"].drop_duplicates().sort_values().unique())
    eligible = benchmark_dates[(benchmark_dates >= START) & (benchmark_dates <= END)]
    union_set = set(pd.to_datetime(union_dates).tolist())
    missing_layer_dates = [d for d in eligible.tolist() if d not in union_set]
    layer_date_coverage = 1.0 if len(eligible) == 0 else (len(eligible) - len(missing_layer_dates)) / len(eligible)

    cross = cross_source_spotcheck(long_df, benchmark_dates)
    min_series_coverage = min(x["post_start_date_coverage"] for x in series_audit.values())
    failures = []
    if len(code_map) != 37:
        failures.append("frozen universe mapping incomplete")
    if layer_date_coverage < 1.0:
        failures.append(f"layer eligible-date coverage {layer_date_coverage}")
    if min_series_coverage < 1.0:
        failures.append(f"min post-start series coverage {min_series_coverage}")
    if cross["status"] == "FAIL":
        failures.append("TIP/TWSE numeric spot-check mismatch")
    quality = "FAIL" if failures else ("PROVISIONAL" if cross["status"] == "PROVISIONAL" else "PASS")

    out_path = LAYER_DIR / "industry_index_daily.parquet"
    long_df.to_parquet(out_path, index=False)
    missingness = {
        c: {"missing_rows": int(long_df[c].isna().sum()), "missing_rate": float(long_df[c].isna().mean())}
        for c in ["price_index", "total_return_index", "change", "change_pct"]
    }
    manifest = {
        "layer_id": "0B_INDUSTRY", "status": "PROVISIONAL", "schema_version": "v1",
        "generation_commit": generation_commit(), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_lineage": [
            {"owner": "Taiwan Stock Exchange / Taiwan Index Plus", "transport": "TIP official historical CSV endpoint",
             "url": TIP_DOWNLOAD, "code_map_source": TIP_FRONT + "/indexes/comparison",
             "frozen_universe_source": TIP_FRONT + "/indexes/multipleHistory", "source_role": "PRIMARY_OFFICIAL"},
            {"owner": "Taiwan Stock Exchange", "transport": "MI_INDEX IND daily JSON",
             "url": TWSE_MI_INDEX, "source_role": "INDEPENDENT_NUMERIC_SPOTCHECK"},
        ],
        "datasets": [{"name": n, "index_code": code_map[n], "time_semantics": "decision_date_daily"} for n in FROZEN_INDUSTRY_NAMES],
        "date_start": long_df["date"].min().date().isoformat(), "date_end": long_df["date"].max().date().isoformat(),
        "row_count": int(len(long_df)), "file_sha256": {out_path.name: sha256(out_path)}, "missingness": missingness,
        "pit_rules": {
            "decision_date_definition": "TWSE trading date",
            "value_availability": "official closing index value published after the same trading session",
            "future_join_allowed": False, "forward_fill_in_layer_builder": False,
            "schema_universe_frozen_before_oos": True, "formal_oos_eligible": False,
            "reason_not_yet_eligible": "layer checkpoint/source-freeze not yet committed; manifest remains provisional even when quality gates pass",
        },
        "universe_audit": universe_audit, "download_chunk_audit": DOWNLOAD_AUDIT,
        "coverage_audit": {
            "benchmark": "TIP t00 official trading-date history", "benchmark_date_count": int(len(eligible)),
            "layer_union_date_count": int(len(union_dates)), "layer_eligible_date_coverage": float(layer_date_coverage),
            "missing_layer_date_count": int(len(missing_layer_dates)),
            "missing_layer_date_sample": [d.date().isoformat() for d in missing_layer_dates[:20]],
            "minimum_post_start_series_coverage": float(min_series_coverage), "per_series": series_audit,
        },
        "cross_source_spotcheck": cross, "quality_gate_status": quality, "quality_gate_failures": failures,
    }
    manifest_path = MANIFEST_DIR / "0B_INDUSTRY.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if quality == "FAIL":
        raise RuntimeError("0B industry quality gate FAIL: " + "; ".join(failures))
    print(json.dumps({
        "layer": "0B_INDUSTRY", "status": "PROVISIONAL", "quality_gate_status": quality,
        "rows": int(len(long_df)), "industry_series": 37,
        "date_start": manifest["date_start"], "date_end": manifest["date_end"],
        "benchmark_dates": int(len(eligible)), "layer_date_coverage": layer_date_coverage,
        "min_series_coverage": min_series_coverage, "crosscheck_status": cross["status"],
        "crosscheck_n": cross["compared_observations"], "sha256": manifest["file_sha256"],
        "formal_oos_eligible": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
