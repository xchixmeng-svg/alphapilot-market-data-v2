#!/usr/bin/env python3
"""
AlphaPilot Whale Radar v0 collector.

Purpose
-------
Collect only public, official Taiwan market disclosures and turn them into a
deduplicated event ledger. This version deliberately does NOT infer a person's
private brokerage account or claim that "no disclosure" means "no holding".

Sources (TWSE OpenAPI):
- t187ap02_L: listed companies' >10% shareholders
- t187ap11_L: directors/supervisors shareholdings
- t187ap12_L: daily insider share transfer pre-declarations
- t187ap04_L: material information; filtered for Securities and Exchange Act 43-1

The script uses only the Python standard library so it can run cheaply in
GitHub Actions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TZ_TAIPEI = timezone(timedelta(hours=8))
BASE = "https://openapi.twse.com.tw/v1/opendata"
DATASETS = {
    "large_shareholders_10pct": f"{BASE}/t187ap02_L",
    "director_holdings": f"{BASE}/t187ap11_L",
    "insider_transfers": f"{BASE}/t187ap12_L",
    "material_news": f"{BASE}/t187ap04_L",
}

SEC43_PATTERNS = (
    "43條之1",
    "43 條之 1",
    "第四十三條之一",
    "43-1",
    "43之1",
)


def now_taipei() -> datetime:
    return datetime.now(TZ_TAIPEI)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def fetch_json(url: str, timeout: int = 30, attempts: int = 3) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": "AlphaPilot-WhaleRadar/0.1 (+github-actions)",
        "Accept": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
                raw = resp.read().decode("utf-8-sig")
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError(f"expected JSON list from {url}, got {type(data).__name__}")
            return [row for row in data if isinstance(row, dict)]
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def get_first(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = clean(row.get(key))
        if value:
            return value
    return ""


def stock_code(row: dict[str, Any]) -> str:
    return get_first(row, ("公司代號", "公司代碼", "證券代號", "股票代號", "Code"))


def company_name(row: dict[str, Any]) -> str:
    return get_first(row, ("公司名稱", "公司簡稱", "證券名稱", "股票名稱", "Name"))


def row_fingerprint(dataset: str, row: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"dataset": dataset, "row": row},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def contains_sec43(row: dict[str, Any]) -> bool:
    text = " ".join(clean(v) for v in row.values())
    return any(p in text for p in SEC43_PATTERNS)


def extract_sec43_actor(row: dict[str, Any]) -> str:
    text = " ".join(clean(v) for v in row.values())
    patterns = (
        r"代取得人(.+?)依證券交易法",
        r"取得人(.+?)依證券交易法",
        r"共同取得人(.+?)依證券交易法",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            actor = match.group(1).strip(" ：:，,。")
            if actor:
                return actor[:120]
    return ""


def find_previous_snapshot(root: Path, today: str, dataset: str) -> list[dict[str, Any]]:
    snapshots = root / "snapshots"
    if not snapshots.exists():
        return []
    candidates = []
    for child in snapshots.iterdir():
        if not child.is_dir() or child.name >= today:
            continue
        p = child / f"{dataset}.json"
        if p.exists():
            candidates.append((child.name, p))
    if not candidates:
        return []
    _, path = max(candidates, key=lambda item: item[0])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def event(
    *,
    event_type: str,
    dataset: str,
    row: dict[str, Any],
    actor: str = "",
    score: int = 0,
    direction: str = "neutral",
    note: str = "",
) -> dict[str, Any]:
    eid = row_fingerprint(dataset + ":" + event_type, row)
    return {
        "event_id": eid,
        "event_type": event_type,
        "dataset": dataset,
        "stock_code": stock_code(row),
        "company_name": company_name(row),
        "actor": actor,
        "signal_score": score,
        "direction": direction,
        "note": note,
        "raw": row,
    }


def build_events(
    current: dict[str, list[dict[str, Any]]],
    previous_large: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    current_large = current.get("large_shareholders_10pct", [])
    prev_keys = {
        (
            stock_code(r),
            get_first(r, ("大股東名稱", "股東名稱", "姓名")),
        )
        for r in previous_large
    }
    for row in current_large:
        actor = get_first(row, ("大股東名稱", "股東名稱", "姓名"))
        key = (stock_code(row), actor)
        if actor and key not in prev_keys:
            events.append(
                event(
                    event_type="new_10pct_shareholder",
                    dataset="large_shareholders_10pct",
                    row=row,
                    actor=actor,
                    score=35,
                    direction="bullish_watch",
                    note="New name versus the most recent stored >10% shareholder snapshot.",
                )
            )

    for row in current.get("material_news", []):
        if contains_sec43(row):
            events.append(
                event(
                    event_type="sec43_1_disclosure",
                    dataset="material_news",
                    row=row,
                    actor=extract_sec43_actor(row),
                    score=50,
                    direction="bullish_watch",
                    note="Material disclosure matched Securities and Exchange Act Article 43-1 text.",
                )
            )

    for row in current.get("insider_transfers", []):
        actor = get_first(
            row,
            ("申報人姓名", "姓名", "內部人姓名", "申報人", "持有人"),
        )
        events.append(
            event(
                event_type="insider_transfer_predeclare",
                dataset="insider_transfers",
                row=row,
                actor=actor,
                score=-20,
                direction="risk_watch",
                note="Insider transfer pre-declaration. Treat as a risk/exit signal, not a buy signal.",
            )
        )

    dedup: dict[str, dict[str, Any]] = {}
    for item in events:
        dedup[item["event_id"]] = item
    return list(dedup.values())


def read_existing_event_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("event_id"):
                ids.add(str(obj["event_id"]))
        except json.JSONDecodeError:
            continue
    return ids


def append_new_events(path: Path, events: list[dict[str, Any]], fetched_at: str) -> list[dict[str, Any]]:
    existing = read_existing_event_ids(path)
    new_events = [e for e in events if e["event_id"] not in existing]
    if not new_events:
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for item in new_events:
            payload = dict(item)
            payload["first_seen_at"] = fetched_at
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return new_events


def write_csv(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event_id",
        "event_type",
        "stock_code",
        "company_name",
        "actor",
        "signal_score",
        "direction",
        "note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for item in events:
            writer.writerow({k: item.get(k, "") for k in fields})


def self_test() -> None:
    row = {
        "公司代號": "4194",
        "公司簡稱": "禾生技",
        "主旨": "代取得人隆欣投資(股)公司依證券交易法第43條之1第1項規定辦理公告",
    }
    assert stock_code(row) == "4194"
    assert company_name(row) == "禾生技"
    assert contains_sec43(row)
    assert "隆欣投資" in extract_sec43_actor(row)

    old = [{"公司代號": "1101", "大股東名稱": "甲"}]
    cur = {
        "large_shareholders_10pct": [
            {"公司代號": "1101", "公司名稱": "台泥", "大股東名稱": "甲"},
            {"公司代號": "1101", "公司名稱": "台泥", "大股東名稱": "乙"},
        ],
        "material_news": [row],
        "insider_transfers": [],
    }
    events = build_events(cur, old)
    types = {x["event_type"] for x in events}
    assert "new_10pct_shareholder" in types
    assert "sec43_1_disclosure" in types
    print("self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/whale_radar", help="output root")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="continue when an endpoint fails; failures are recorded in status.json",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    root = Path(args.out)
    now = now_taipei()
    today = now.date().isoformat()
    fetched_at = now.isoformat(timespec="seconds")
    current: dict[str, list[dict[str, Any]]] = {}
    status: dict[str, Any] = {
        "fetched_at": fetched_at,
        "sources": {},
        "version": "whale-radar-v0.1",
    }

    failed = False
    for name, url in DATASETS.items():
        try:
            rows = fetch_json(url)
            current[name] = rows
            status["sources"][name] = {"ok": True, "rows": len(rows), "url": url}
            json_dump(root / "latest" / f"{name}.json", rows)
            json_dump(root / "snapshots" / today / f"{name}.json", rows)
            print(f"{name}: {len(rows)} rows")
        except Exception as exc:
            failed = True
            current[name] = []
            status["sources"][name] = {"ok": False, "error": str(exc), "url": url}
            print(f"{name}: ERROR {exc}", file=sys.stderr)
            if not args.allow_partial:
                json_dump(root / "latest" / "status.json", status)
                return 2

    previous_large = find_previous_snapshot(root, today, "large_shareholders_10pct")
    events = build_events(current, previous_large)
    ledger = root / "events.jsonl"
    new_events = append_new_events(ledger, events, fetched_at)

    json_dump(root / "latest" / "events_seen_today.json", events)
    json_dump(root / "latest" / "new_events.json", new_events)
    write_csv(root / "latest" / "new_events.csv", new_events)

    status["events_seen_today"] = len(events)
    status["new_events"] = len(new_events)
    status["partial"] = failed
    json_dump(root / "latest" / "status.json", status)
    print(f"events: {len(events)} seen, {len(new_events)} new")
    return 0 if (not failed or args.allow_partial) else 2


if __name__ == "__main__":
    raise SystemExit(main())
