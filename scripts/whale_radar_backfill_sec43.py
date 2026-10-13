#!/usr/bin/env python3
"""Backfill historical Securities and Exchange Act 43-1 disclosures from MOPS.

The goal is evidence collection, not inference.  We query MOPS' historical
material-information keyword search by ROC year/month, retain only rows whose
text explicitly contains an Article 43-1 spelling, and store a separate
historical event ledger.  The live events.jsonl remains untouched.

Coverage is deliberately recorded in a manifest.  If MOPS throttles or changes
HTML, failed months are marked FAILED/PARTIAL rather than guessed.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import html
import json
import re
import time
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MOPS_ENDPOINTS = (
    "https://mopsov.twse.com.tw/mops/web/ajax_t51sb10",
    "https://mops.twse.com.tw/mops/web/ajax_t51sb10",
)
PATTERNS = ("43條之1", "43 條之 1", "第四十三條之一", "43-1", "43之1")
UA = "Mozilla/5.0 (compatible; AlphaPilot-WhaleRadar/1.0; +github-actions)"


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            text = re.sub(r"\s+", " ", html.unescape("".join(self._cell))).strip()
            self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(x.strip() for x in self._row):
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def contains_43(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return any(re.sub(r"\s+", "", p) in compact for p in PATTERNS)


def extract_actor(subject: str) -> str:
    s = re.sub(r"\s+", " ", subject).strip()
    pats = (
        r"代取得人(.+?)依(?:證券交易法|證交法)",
        r"共同取得人(.+?)依(?:證券交易法|證交法)",
        r"取得人(.+?)依(?:證券交易法|證交法)",
        r"代(.+?)依(?:證券交易法|證交法)第?43",
    )
    for pat in pats:
        m = re.search(pat, s)
        if m:
            actor = m.group(1).strip(" ：:，,。()（）")
            if actor and len(actor) <= 120:
                return actor
    return ""


def parse_roc_date(text: str, fallback_year: int, fallback_month: int) -> str:
    # Common MOPS form: 115/10/12; tolerate Gregorian as well.
    for pat in (r"(?<!\d)(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})(?!\d)", r"(?<!\d)(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?!\d)"):
        m = re.search(pat, text)
        if not m:
            continue
        y, mo, d = map(int, m.groups())
        if y < 1911:
            y += 1911
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            pass
    # Do not fabricate a day; empty means unusable for causal scoring.
    return ""


def event_id(stock_code: str, event_date: str, subject: str, actor: str) -> str:
    raw = "|".join((stock_code, event_date, subject, actor))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def parse_rows(page: str, roc_year: int, month: int) -> list[dict[str, Any]]:
    parser = TableParser()
    parser.feed(page)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cells in parser.rows:
        joined = " | ".join(cells)
        if not contains_43(joined):
            continue
        code = ""
        for cell in cells:
            m = re.search(r"(?<!\d)(\d{4})(?!\d)", cell)
            if m:
                code = m.group(1)
                break
        if not code:
            continue
        subject = next((c for c in cells if contains_43(c)), joined)
        actor = extract_actor(subject)
        event_date = parse_roc_date(joined, roc_year, month)
        eid = event_id(code, event_date, subject, actor)
        if eid in seen:
            continue
        seen.add(eid)
        # Company name is usually adjacent to code. Keep best-effort raw text as evidence.
        company_name = ""
        for i, cell in enumerate(cells):
            if code in cell and i + 1 < len(cells):
                nxt = cells[i + 1].strip()
                if nxt and not re.fullmatch(r"[\d/: -]+", nxt):
                    company_name = nxt[:80]
                    break
        out.append({
            "event_id": eid,
            "event_type": "sec43_1_disclosure",
            "dataset": "mops_historical_keyword_search",
            "stock_code": code,
            "company_name": company_name,
            "actor": actor,
            "event_date": event_date,
            "signal_score": 50,
            "direction": "bullish_watch",
            "subject": subject,
            "raw_cells": cells,
            "source": "MOPS",
        })
    return out


def fetch_month(roc_year: int, month: int, timeout: int = 35) -> tuple[str, str]:
    last: Exception | None = None
    params = {
        "encodeURIComponent": "1",
        "firstin": "true",
        "id": "",
        "key": "",
        "TYPEK": "",
        "Stp": "4",
        "go": "false",
        "COMPANY_ID": "",
        "r1": "1",
        "KIND": "",
        "CODE": "",
        "keyWord": "43條之1",
        "year": str(roc_year),
        "month1": str(month),
        "begin_day": "1",
        "end_day": str(calendar.monthrange(roc_year + 1911, month)[1]),
        "PCount": "200",
    }
    qs = urlencode(params)
    for endpoint in MOPS_ENDPOINTS:
        url = endpoint + "?" + qs
        for attempt in range(1, 4):
            try:
                req = Request(url, headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"})
                with urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
                text = raw.decode("utf-8", errors="replace")
                if "查詢過於頻繁" in text or "Service Unavailable" in text:
                    raise RuntimeError("MOPS throttle")
                if len(text) < 200:
                    raise RuntimeError(f"unexpected short MOPS response: {len(text)} bytes")
                return text, url
            except Exception as exc:
                last = exc
                time.sleep(min(12.0, 1.5 * attempt))
    raise RuntimeError(f"MOPS fetch failed for {roc_year}/{month}: {last}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def self_test() -> None:
    sample = """
    <table><tr><th>公司代號</th><th>公司簡稱</th><th>發布時間</th><th>主旨</th></tr>
    <tr><td>4194</td><td>禾生技</td><td>115/10/12 10:01</td>
    <td>代取得人隆欣投資(股)公司依證券交易法第43條之1第1項規定辦理公告</td></tr></table>
    """
    rows = parse_rows(sample, 115, 10)
    assert len(rows) == 1
    assert rows[0]["stock_code"] == "4194"
    assert rows[0]["event_date"] == "2026-10-12"
    assert "隆欣投資" in rows[0]["actor"]
    print("sec43-backfill self-test: OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/whale_radar")
    ap.add_argument("--start-year", type=int, default=110, help="ROC year, default 110 (2021)")
    ap.add_argument("--end-year", type=int, default=datetime.now().year - 1911)
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0

    root = Path(args.root)
    all_rows: dict[str, dict[str, Any]] = {}
    months: list[dict[str, Any]] = []
    failures = 0
    now = date.today()
    for ry in range(args.start_year, args.end_year + 1):
        gy = ry + 1911
        last_month = now.month if gy == now.year else 12
        for mo in range(1, last_month + 1):
            try:
                page, url = fetch_month(ry, mo)
                rows = parse_rows(page, ry, mo)
                for row in rows:
                    all_rows[row["event_id"]] = row
                months.append({"roc_year": ry, "month": mo, "status": "OK", "events": len(rows), "url": url})
                print(f"MOPS {ry}/{mo:02d}: {len(rows)} events")
            except Exception as exc:
                failures += 1
                months.append({"roc_year": ry, "month": mo, "status": "FAILED", "events": 0, "error": str(exc)})
                print(f"MOPS {ry}/{mo:02d}: FAILED: {exc}")
                if not args.allow_partial:
                    raise
            time.sleep(max(0.0, args.sleep))

    rows = sorted(all_rows.values(), key=lambda x: (x.get("event_date", ""), x.get("stock_code", ""), x.get("event_id", "")))
    write_jsonl(root / "historical_sec43_events.jsonl", rows)
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "MOPS historical material-information keyword search",
        "query_keyword": "43條之1",
        "start_roc_year": args.start_year,
        "end_roc_year": args.end_year,
        "months_attempted": len(months),
        "months_failed": failures,
        "status": "OK" if failures == 0 else "PARTIAL",
        "events": len(rows),
        "named_actors": len({x.get("actor") for x in rows if x.get("actor")}),
        "months": months,
    }
    latest = root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "sec43_backfill_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (latest / "historical_sec43_events.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"historical sec43: {len(rows)} events; named actors={manifest['named_actors']}; failures={failures}/{len(months)}")
    return 0 if (failures == 0 or args.allow_partial) else 2


if __name__ == "__main__":
    raise SystemExit(main())
