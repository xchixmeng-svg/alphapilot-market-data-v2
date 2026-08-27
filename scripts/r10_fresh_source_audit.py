#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh-source audit for AlphaPilot R10 historical validation.

Verifies cached/backtest inputs against fresh official TWSE/TPEx responses on
fixed, deterministic sample dates. It never reads any performance target,
Golden order, exit, trade, or NAV reference.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import r10_fast_validation as fast

bt = fast.bt
core = fast.core
OUT = bt.OUT_ROOT / "latest" / "true_validation2021_2025"

TWSE_MI_LEGACY = "https://www.twse.com.tw/exchangeReport/MI_INDEX"
TWSE_T86_LEGACY = "https://www.twse.com.tw/fund/T86"
S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.5",
    "Referer": "https://www.twse.com.tw/",
})


def _legacy_json(url: str, params: dict, tries: int = 4):
    last = None
    for k in range(tries):
        try:
            r = S.get(url, params=params, timeout=(20, 90))
            r.raise_for_status()
            text = r.text.lstrip("\ufeff \t\r\n")
            if not text.startswith(("{", "[")):
                raise RuntimeError(f"non-json status={r.status_code} content-type={r.headers.get('content-type')} prefix={text[:80]!r}")
            return r.json()
        except Exception as exc:
            last = exc
            if k + 1 < tries:
                time.sleep(2 ** k)
    raise RuntimeError(f"official GET failed {url}: {last}")


def _twse_day_fresh(d) -> list[dict]:
    p = _legacy_json(TWSE_MI_LEGACY, {
        "response": "json", "date": d.strftime("%Y%m%d"), "type": "ALLBUT0999"
    })
    f, r = core.select(p, ["證券代號", "開盤價", "最高價", "最低價", "收盤價"])
    return [] if not f else core.norm_ohlcv(core.dicts(f, r), d)


def _twse_inst_fresh(d) -> list[dict]:
    p = _legacy_json(TWSE_T86_LEGACY, {
        "response": "json", "date": d.strftime("%Y%m%d"), "selectType": "ALLBUT0999"
    })
    f, r = core.select(p, ["證券代號", "外資", "投信", "自營商"])
    return [] if not f else core.norm_inst(core.dicts(f, r), d, "TWSE")


def _official_day_fresh(d) -> pd.DataFrame:
    a = _twse_day_fresh(d)
    time.sleep(0.2)
    b = core.tpex_day(d)
    return pd.DataFrame(a + b, columns=core.COLS) if (a or b) else pd.DataFrame(columns=core.COLS)


def _norm(q: pd.DataFrame) -> pd.DataFrame:
    if q.empty:
        return q.copy()
    z = q.copy()
    z["code"] = z["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    for c in ("open", "high", "low", "close", "volume"):
        if c in z:
            z[c] = pd.to_numeric(z[c], errors="coerce")
    return z


def _inst_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["market", "code", "foreign_net", "trust_net", "dealer_net"])
    q = pd.DataFrame(rows).copy()
    q["code"] = q["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    for c in ("foreign_net", "trust_net", "dealer_net"):
        q[c] = pd.to_numeric(q[c], errors="coerce").fillna(0.0)
    return q[["market", "code", "foreign_net", "trust_net", "dealer_net"]].drop_duplicates(["market", "code"], keep="last")


def main() -> None:
    cfg = bt.SCENARIOS["validation2021_2025"]
    raw = _norm(bt.load_scenario_ohlcv(cfg))
    lo, hi = bt.intdate(cfg["eval_start"]), bt.intdate(cfg["eval_end"])
    dates = np.array(sorted(int(x) for x in raw.date.unique() if lo <= int(x) <= hi), dtype=int)
    if len(dates) < 1000:
        raise RuntimeError(f"unexpectedly short evaluation calendar: {len(dates)}")

    idx = np.linspace(0, len(dates) - 1, 10, dtype=int)
    sample_dates = [int(dates[i]) for i in idx]
    day_audits, inst_audits, hard_failures = [], [], []

    for di in sample_dates:
        d = bt.dt_from_int(di)
        hist = _norm(raw[raw.date.eq(di)][["code", "open", "high", "low", "close", "volume"]])
        try:
            fresh = _norm(_official_day_fresh(d)[["code", "open", "high", "low", "close", "volume"]])
        except Exception as exc:
            hard_failures.append(f"{di} official OHLCV transport: {exc}")
            continue
        if fresh.empty:
            hard_failures.append(f"{di} official OHLCV empty")
            continue
        m = hist.merge(fresh, on="code", suffixes=("_hist", "_fresh"), how="inner")
        if m.empty:
            hard_failures.append(f"{di} no OHLCV code intersection")
            continue
        px_bad = pd.Series(False, index=m.index)
        for c in ("open", "high", "low", "close"):
            px_bad |= (m[f"{c}_hist"] - m[f"{c}_fresh"]).abs() > 1e-8
        vol_bad = (m["volume_hist"] - m["volume_fresh"]).abs() > 0.5
        mismatch = px_bad | vol_bad
        hist_codes = set(hist.code[hist.code.str.fullmatch(r"\d{4}")])
        fresh_codes = set(fresh.code[fresh.code.str.fullmatch(r"\d{4}")])
        coverage = len(hist_codes & fresh_codes) / max(1, len(hist_codes))
        day_audits.append({
            "date": di, "historical_rows": int(len(hist)), "official_rows": int(len(fresh)),
            "intersection_rows": int(len(m)), "four_digit_coverage": float(coverage),
            "ohlcv_mismatch_rows": int(mismatch.sum()),
            "sample_mismatches": m.loc[mismatch].head(5).to_dict("records"),
        })
        if coverage < 0.90:
            hard_failures.append(f"{di} OHLCV code coverage {coverage:.2%} < 90%")
        if int(mismatch.sum()) != 0:
            hard_failures.append(f"{di} OHLCV mismatches={int(mismatch.sum())}")

        try:
            fresh_rows: list[dict] = []
            fresh_rows.extend(_twse_inst_fresh(d))
            time.sleep(0.2)
            fresh_rows.extend(core.tpex_inst(d))
            fi = _inst_frame(fresh_rows)
        except Exception as exc:
            hard_failures.append(f"{di} official institutional transport: {exc}")
            continue
        cache_path = fast.INST_DAY_CACHE / f"{di}.csv"
        if not cache_path.exists():
            hard_failures.append(f"{di} institutional cache missing")
            continue
        ci = _inst_frame(pd.read_csv(cache_path, dtype={"code": str}).to_dict("records"))
        im = ci.merge(fi, on=["market", "code"], suffixes=("_cache", "_fresh"), how="inner")
        if im.empty:
            hard_failures.append(f"{di} institutional no code intersection")
            continue
        ibad = pd.Series(False, index=im.index)
        for c in ("foreign_net", "trust_net", "dealer_net"):
            ibad |= (im[f"{c}_cache"] - im[f"{c}_fresh"]).abs() > 0.5
        cache_keys = set(zip(ci.market.astype(str), ci.code.astype(str)))
        fresh_keys = set(zip(fi.market.astype(str), fi.code.astype(str)))
        icoverage = len(cache_keys & fresh_keys) / max(1, len(cache_keys))
        inst_audits.append({
            "date": di, "cache_rows": int(len(ci)), "official_rows": int(len(fi)),
            "intersection_rows": int(len(im)), "key_coverage": float(icoverage),
            "institutional_mismatch_rows": int(ibad.sum()),
            "sample_mismatches": im.loc[ibad].head(5).to_dict("records"),
        })
        if icoverage < 0.90:
            hard_failures.append(f"{di} institutional key coverage {icoverage:.2%} < 90%")
        if int(ibad.sum()) != 0:
            hard_failures.append(f"{di} institutional mismatches={int(ibad.sum())}")
        time.sleep(0.2)

    report = {
        "status": "PASS" if not hard_failures else "FAIL",
        "purpose": "fresh official input verification only",
        "performance_reference_used": False,
        "twse_historical_endpoint": "legacy official /exchangeReport + /fund endpoints",
        "sample_method": "10 evenly spaced dates from actual 2021-2025 backtest trading calendar",
        "sample_dates": sample_dates, "ohlcv": day_audits, "institutional": inst_audits,
        "failures": hard_failures,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fresh_source_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
