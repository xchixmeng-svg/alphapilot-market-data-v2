#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data" / "history" / "2007-2019" / "raw"
NEW = ROOT / "data" / "history" / "2020-2025"
YTD = ROOT / "data" / "history" / "2026-YTD"
OUT = ROOT / "data" / "history" / "v6-layered" / "0A_MARKET"
MANIFEST_DIR = ROOT / "data" / "history" / "v6-layered" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2016
END_DATE = 20260915
SCHEMA_VERSION = "V6-0A-MARKET-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ymd(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y%m%d").astype("Int64")
    z = s.astype(str).str.strip().str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    return pd.to_numeric(z, errors="coerce").astype("Int64")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    ren = {
        "trade_date": "date",
        "stock_id": "code",
        "Trading_Volume": "volume",
        "Trade_Volume": "volume",
    }
    x = x.rename(columns={k: v for k, v in ren.items() if k in x.columns})
    need = ["date", "code", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in x.columns]
    if missing:
        raise RuntimeError(f"OHLCV missing columns {missing}; got={list(x.columns)}")
    x = x[need].copy()
    x["date"] = ymd(x["date"])
    x["code"] = x["code"].astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(4)
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["date", "code", "close"]).copy()
    x["date"] = x["date"].astype(np.int64)
    return x[(x["date"] >= START_YEAR * 10000 + 101) & (x["date"] <= END_DATE) & (x["close"] > 0)]


def load_zip(year: int) -> tuple[pd.DataFrame, dict]:
    p = OLD / f"yearly_{year}.zip"
    if not p.exists():
        raise RuntimeError(f"missing immutable history {p}")
    with zipfile.ZipFile(p) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"no CSV in {p}")
        with zf.open(members[0]) as fh:
            df = normalize(pd.read_csv(fh, low_memory=False))
    return df, {
        "path": str(p.relative_to(ROOT)),
        "sha256": sha256(p),
        "role": "historical_stock_ohlcv_source",
        "lineage": "immutable upstream yearly archive; source manifest states TWSE MI_INDEX + TPEx OTC daily OHLCV",
        "year": year,
    }


def load_parquet(year: int) -> tuple[pd.DataFrame, dict]:
    p = NEW / f"ohlcv_{year}.parquet"
    if not p.exists():
        raise RuntimeError(f"missing {p}")
    return normalize(pd.read_parquet(p)), {
        "path": str(p.relative_to(ROOT)),
        "sha256": sha256(p),
        "role": "historical_stock_ohlcv_source",
        "lineage": "existing AlphaPilot normalized historical parquet",
        "year": year,
    }


def load_2026(parts: list[pd.DataFrame], lineage: list[dict]) -> None:
    yp = YTD / "ohlcv_2026_ytd.csv"
    if yp.exists():
        parts.append(normalize(pd.read_csv(yp, low_memory=False)))
        lineage.append({
            "path": str(yp.relative_to(ROOT)),
            "sha256": sha256(yp),
            "role": "2026_ytd_stock_ohlcv_source",
            "lineage": "existing AlphaPilot 2026 YTD normalized OHLCV",
        })

    for d in sorted((ROOT / "data").glob("2026-??-??")):
        n = d / "normalized"
        for p in (n / "twse_ohlcv.csv", n / "tpex_ohlcv.csv"):
            if not p.exists():
                continue
            try:
                parts.append(normalize(pd.read_csv(p, low_memory=False)))
                lineage.append({
                    "path": str(p.relative_to(ROOT)),
                    "sha256": sha256(p),
                    "role": "2026_daily_stock_ohlcv_source",
                    "lineage": "official-source daily normalized AlphaPilot snapshot",
                })
            except Exception as e:
                print(f"[0A WARN] skip {p}: {e}", flush=True)


def build_market(px: pd.DataFrame) -> pd.DataFrame:
    px = px.sort_values(["code", "date"]).drop_duplicates(["date", "code"], keep="last").copy()
    g = px.groupby("code", group_keys=False)
    px["ret1"] = px["close"] / g["close"].shift(1) - 1.0
    px["amount"] = (px["close"] * px["volume"]).where(px["volume"] >= 0)

    # Equity cross-section deliberately excludes ETF/fund-style leading-zero codes.
    eq = px[px["code"].str.fullmatch(r"[1-9]\d{3}", na=False)].copy()
    daily = eq.groupby("date").agg(
        mkt_ret1=("ret1", "median"),
        mkt_advance=("ret1", lambda s: float((s > 0).mean())),
        mkt_dispersion=("ret1", "std"),
        equity_count=("code", "nunique"),
        advancers=("ret1", lambda s: int((s > 0).sum())),
        decliners=("ret1", lambda s: int((s < 0).sum())),
        unchanged=("ret1", lambda s: int((s == 0).sum())),
        total_amount=("amount", "sum"),
        median_amount=("amount", "median"),
    ).reset_index()

    f = px[px["code"] == "0050"][["date", "close", "volume", "amount"]].copy()
    f = f.sort_values("date").drop_duplicates("date", keep="last")
    f["0050_ret1"] = f["close"].pct_change()
    f = f.rename(columns={"close": "0050_close", "volume": "0050_volume", "amount": "0050_amount"})
    daily = daily.merge(f, on="date", how="left")
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def main() -> int:
    parts: list[pd.DataFrame] = []
    lineage: list[dict] = []
    for year in range(START_YEAR, 2020):
        d, meta = load_zip(year)
        parts.append(d)
        lineage.append(meta)
        print(f"[0A LOAD] {year} rows={len(d)}", flush=True)
    for year in range(2020, 2026):
        d, meta = load_parquet(year)
        parts.append(d)
        lineage.append(meta)
        print(f"[0A LOAD] {year} rows={len(d)}", flush=True)
    load_2026(parts, lineage)

    px = pd.concat(parts, ignore_index=True)
    market = build_market(px)
    if market.empty:
        raise RuntimeError("0A market output empty")

    out = OUT / "market_daily.parquet"
    market.to_parquet(out, index=False)
    output_sha = sha256(out)

    value_cols = [c for c in market.columns if c != "date"]
    missing = {c: float(market[c].isna().mean()) for c in value_cols}
    date_start = str(int(market["date"].min()))
    date_end = str(int(market["date"].max()))
    date_start = f"{date_start[:4]}-{date_start[4:6]}-{date_start[6:8]}"
    date_end = f"{date_end[:4]}-{date_end[4:6]}-{date_end[6:8]}"

    manifest = {
        "layer_id": "0A_MARKET",
        "status": "PROVISIONAL",
        "quality_gate_status": "PROVISIONAL_SOURCE_EQUIVALENCE_PENDING",
        "schema_version": SCHEMA_VERSION,
        "generation_commit": os.getenv("GITHUB_SHA", "LOCAL_UNCOMMITTED"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_lineage": lineage,
        "datasets": [{
            "name": "market_daily",
            "path": str(out.relative_to(ROOT)),
            "time_semantics": "decision_date_daily",
            "columns": list(market.columns),
        }],
        "date_start": date_start,
        "date_end": date_end,
        "row_count": int(len(market)),
        "file_sha256": {"market_daily.parquet": output_sha},
        "missingness": missing,
        "pit_rules": {
            "decision_time": "Taiwan market close on date T",
            "same_day_ohlcv_allowed": True,
            "future_join_allowed": False,
            "silent_zero_fill_allowed": False,
            "formal_oos_eligible": False,
        },
        "construction": {
            "market_cross_section": "Taiwan four-digit equity codes beginning 1-9; excludes ETF/fund-style leading-zero codes",
            "mkt_ret1": "cross-sectional median same-day close-to-close return",
            "mkt_advance": "share of valid equities with positive same-day close-to-close return",
            "mkt_dispersion": "cross-sectional standard deviation of same-day close-to-close return",
            "0050": "separate observed 0050 close/return/volume/amount series",
        },
        "freeze_blockers": [
            "2016-2019 upstream mirror must receive official-source spot/equivalence audit before FROZEN PASS",
            "2020-2025 normalized parquet lineage/hash/coverage audit must be reconciled with archive manifests",
            "corporate-action impact on cross-sectional dispersion must be audited before source freeze",
        ],
    }
    mp = MANIFEST_DIR / "0A_MARKET.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "layer": "0A_MARKET",
        "status": manifest["status"],
        "rows": len(market),
        "date_start": date_start,
        "date_end": date_end,
        "equity_count_min": int(market["equity_count"].min()),
        "equity_count_max": int(market["equity_count"].max()),
        "sha256": output_sha,
        "formal_oos_eligible": False,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
