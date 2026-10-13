#!/usr/bin/env python3
"""Evaluate Whale Radar bullish events against causally later market prices.

Causality rule:
- event time comes from first_seen_at / event_date
- simulated follower entry is NEXT trading day's OPEN
- horizons are trading-day closes after entry: 5/10/20/40/60/120
- MFE/MAE use high/low from entry day through each horizon

The script only uses price files already present in this repository.  Missing
history is reported as INSUFFICIENT instead of guessed or fetched silently.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

HORIZONS = (5, 10, 20, 40, 60, 120)
POSITIVE_TYPES = {"sec43_1_disclosure", "new_10pct_shareholder"}


def parse_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def fnum(v: Any) -> float | None:
    try:
        s = str(v).replace(",", "").strip()
        if not s or s in {"--", "---", "nan", "None"}:
            return None
        return float(s)
    except Exception:
        return None


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(x, dict) or x.get("event_type") not in POSITIVE_TYPES:
            continue
        d = parse_date(str(x.get("first_seen_at") or x.get("event_date") or ""))
        if not d:
            continue
        x = dict(x)
        x["_event_date"] = d
        out.append(x)
    return out


def load_price_history(data_root: Path, wanted_codes: set[str]) -> dict[str, list[dict[str, Any]]]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in sorted(data_root.glob("????-??-??/normalized/*_ohlcv.csv")):
        try:
            with p.open("r", encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    code = str(r.get("stock_id", "")).strip()
                    if code not in wanted_codes:
                        continue
                    d = parse_date(str(r.get("trade_date", "")))
                    o, h, l, c = map(fnum, (r.get("open"), r.get("high"), r.get("low"), r.get("close")))
                    if d and None not in (o, h, l, c):
                        by_code[code].append({"date": d, "open": o, "high": h, "low": l, "close": c})
        except (OSError, UnicodeDecodeError):
            continue
    for code in by_code:
        uniq = {x["date"]: x for x in by_code[code]}
        by_code[code] = [uniq[d] for d in sorted(uniq)]
    return by_code


def evaluate_event(ev: dict[str, Any], bars: list[dict[str, Any]]) -> dict[str, Any]:
    event_date: date = ev["_event_date"]
    later = [b for b in bars if b["date"] > event_date]
    base = {
        "event_id": ev.get("event_id", ""),
        "event_type": ev.get("event_type", ""),
        "actor": ev.get("actor", ""),
        "stock_code": ev.get("stock_code", ""),
        "company_name": ev.get("company_name", ""),
        "event_date": event_date.isoformat(),
        "status": "INSUFFICIENT",
    }
    if not later:
        return base
    entry = later[0]
    entry_price = entry["open"]
    base.update({
        "entry_date": entry["date"].isoformat(),
        "entry_open": round(entry_price, 6),
        "available_trading_days": len(later),
    })
    complete = 0
    for hz in HORIZONS:
        if len(later) < hz:
            continue
        window = later[:hz]
        end = window[-1]
        ret = end["close"] / entry_price - 1.0
        mfe = max(x["high"] for x in window) / entry_price - 1.0
        mae = min(x["low"] for x in window) / entry_price - 1.0
        base[f"ret{hz}"] = round(ret, 6)
        base[f"mfe{hz}"] = round(mfe, 6)
        base[f"mae{hz}"] = round(mae, 6)
        complete += 1
    base["status"] = "OK" if complete else "INSUFFICIENT"
    return base


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_actor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        actor = str(r.get("actor", "")).strip()
        if actor:
            by_actor[actor].append(r)
    out = []
    for actor, rs in by_actor.items():
        row: dict[str, Any] = {"actor": actor, "events": len(rs)}
        scored = 0
        score_parts = []
        for hz in HORIZONS:
            vals = [float(r[f"ret{hz}"]) for r in rs if f"ret{hz}" in r]
            mfes = [float(r[f"mfe{hz}"]) for r in rs if f"mfe{hz}" in r]
            maes = [float(r[f"mae{hz}"]) for r in rs if f"mae{hz}" in r]
            if not vals:
                continue
            wins = sum(v > 0 for v in vals)
            hit15 = sum(m >= 0.15 for m in mfes)
            row[f"n{hz}"] = len(vals)
            row[f"win_rate{hz}"] = round(wins / len(vals), 4)
            row[f"hit15_rate{hz}"] = round(hit15 / len(vals), 4)
            row[f"avg_ret{hz}"] = round(mean(vals), 6)
            row[f"median_ret{hz}"] = round(median(vals), 6)
            row[f"median_mfe{hz}"] = round(median(mfes), 6)
            row[f"median_mae{hz}"] = round(median(maes), 6)
            if hz in (20, 40, 60):
                scored += len(vals)
                score_parts.append((wins / len(vals)) * 50 + min(max(mean(vals), -0.2), 0.4) * 100)
        row["status"] = "QUALIFIED" if scored >= 9 else "INSUFFICIENT"
        row["person_alpha_score"] = round(max(0.0, min(100.0, mean(score_parts) if score_parts else 0.0)), 2)
        out.append(row)
    out.sort(key=lambda x: (x["status"] != "QUALIFIED", -x["person_alpha_score"], -x["events"], x["actor"]))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()}) if rows else ["actor", "events", "status", "person_alpha_score"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def self_test(tmp: Path) -> None:
    ev = {"event_id": "x", "event_type": "sec43_1_disclosure", "actor": "甲", "stock_code": "1234", "company_name": "測試", "_event_date": date(2026, 1, 1)}
    bars = []
    for i in range(1, 25):
        px = 100 + i
        bars.append({"date": date(2026, 1, 1 + i), "open": float(px), "high": float(px + 2), "low": float(px - 2), "close": float(px + 1)})
    r = evaluate_event(ev, bars)
    assert r["entry_open"] == 101.0
    assert "ret20" in r and r["mfe20"] > 0
    a = aggregate([r, r, r])
    assert a and a[0]["actor"] == "甲"
    print("history-score self-test: OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/whale_radar")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    if args.self_test:
        self_test(root)
        return 0
    events = load_events(root / "events.jsonl")
    codes = {str(x.get("stock_code", "")) for x in events if x.get("stock_code")}
    prices = load_price_history(Path(args.data_root), codes)
    event_rows = [evaluate_event(ev, prices.get(str(ev.get("stock_code", "")), [])) for ev in events]
    people = aggregate(event_rows)
    latest = root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "event_outcomes.json").write_text(json.dumps(event_rows, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (latest / "person_alpha_scores.json").write_text(json.dumps(people, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(latest / "event_outcomes.csv", event_rows)
    write_csv(latest / "person_alpha_scores.csv", people)
    print(f"history score: {len(events)} bullish events; {len(people)} named actors; repo price codes={len(prices)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
