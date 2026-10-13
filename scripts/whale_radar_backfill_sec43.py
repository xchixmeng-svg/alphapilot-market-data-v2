#!/usr/bin/env python3
"""Backfill historical Securities and Exchange Act 43-1 disclosures from MOPS.

This version deliberately validates the *current* MOPS historical material-info
route before trusting research output.  MOPS changed to a SPA/API gateway in
2025; the old t51sb10 keyword route can return a normal HTML response while not
returning the intended disclosure corpus.  We therefore use t05st01, the
historical material-information page, and require a known positive control.

Research scope for the first pass is listed + OTC companies.  That matches the
tradable universe we care about.  The positive control itself is queried with
TYPEK=all so source validation is independent of listing status.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import html
import http.cookiejar
import json
import re
import time
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

MOPSOV_FORM = "https://mopsov.twse.com.tw/mops/web/t05st01"
MOPSOV_AJAX = "https://mopsov.twse.com.tw/mops/web/ajax_t05st01"
MOPS_GATEWAY = "https://mops.twse.com.tw/mops/api/redirectToOld"
MOPS_HOME = "https://mops.twse.com.tw/mops/"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
PATTERNS = (
    "43條之1",
    "43 條之 1",
    "第43條之1",
    "第四十三條之一",
    "43-1",
    "43之1",
)
MARKETS = {"listed": "sii", "otc": "otc"}
POSITIVE_CONTROL = {
    "roc_year": 115,
    "month": 10,
    "stock_code": "4194",
    "subject_token": "隆欣投資",
    "date": "2026-10-12",
}


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
            value = re.sub(r"\s+", " ", html.unescape("".join(self._cell))).strip()
            self._row.append(value)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(x.strip() for x in self._row):
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def contains_43(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return any(re.sub(r"\s+", "", p) in compact for p in PATTERNS)


def extract_actor(subject: str) -> str:
    s = clean(subject)
    patterns = (
        r"代取得人(.+?)依(?:證券交易法|證交法)",
        r"共同取得人(.+?)依(?:證券交易法|證交法)",
        r"取得人(.+?)依(?:證券交易法|證交法)",
        r"代(.+?)依(?:證券交易法|證交法)第?43",
    )
    for pattern in patterns:
        m = re.search(pattern, s)
        if m:
            actor = m.group(1).strip(" ：:，,。()（）")
            if actor and len(actor) <= 120:
                return actor
    return ""


def parse_roc_date(text: str) -> str:
    text = clean(text)
    m = re.search(r"(?<!\d)(\d{2,3})/(\d{1,2})/(\d{1,2})(?!\d)", text)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return date(y + 1911, mo, d).isoformat()
        except ValueError:
            pass
    m = re.search(r"(?<!\d)(20\d{2})/(\d{1,2})/(\d{1,2})(?!\d)", text)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            pass
    return ""


def event_id(stock_code: str, event_date: str, subject: str, actor: str) -> str:
    raw = "|".join((stock_code, event_date, subject, actor))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _row_code(cells: list[str]) -> str:
    for cell in cells:
        m = re.fullmatch(r"\s*(\d{4,6})\s*", cell)
        if m:
            return m.group(1)
    for cell in cells:
        m = re.search(r"(?<!\d)(\d{4})(?!\d)", cell)
        if m:
            return m.group(1)
    return ""


def _row_subject(cells: list[str]) -> str:
    # Prefer an explicit 43-1 cell; otherwise use the longest plausible text cell.
    for cell in cells:
        if contains_43(cell):
            return clean(cell)
    candidates = [
        clean(c)
        for c in cells
        if len(clean(c)) >= 8
        and not parse_roc_date(c)
        and not re.fullmatch(r"[\d:./ -]+", clean(c))
        and clean(c) not in {"詳細資料", "查詢"}
    ]
    return max(candidates, key=len) if candidates else ""


def parse_page(page: str, market: str = "") -> tuple[list[dict[str, Any]], int]:
    if "FOR SECURITY" in page or "查詢過於頻繁" in page:
        raise RuntimeError("MOPS security/throttle response")
    parser = TableParser()
    parser.feed(page)
    events: dict[str, dict[str, Any]] = {}
    generic_rows = 0
    for cells in parser.rows:
        code = _row_code(cells)
        event_date = parse_roc_date(" | ".join(cells))
        subject = _row_subject(cells)
        if not code or not event_date or not subject:
            continue
        generic_rows += 1
        if not contains_43(subject):
            continue
        actor = extract_actor(subject)
        company_name = ""
        for i, cell in enumerate(cells):
            if clean(cell) == code and i + 1 < len(cells):
                nxt = clean(cells[i + 1])
                if nxt and not parse_roc_date(nxt) and not re.fullmatch(r"[\d:./ -]+", nxt):
                    company_name = nxt[:80]
                    break
        eid = event_id(code, event_date, subject, actor)
        events[eid] = {
            "event_id": eid,
            "event_type": "sec43_1_disclosure",
            "dataset": "mops_t05st01_historical",
            "stock_code": code,
            "company_name": company_name,
            "actor": actor,
            "event_date": event_date,
            "available_at": event_date,
            "market": market,
            "signal_score": 50,
            "direction": "bullish_watch",
            "subject": subject,
            "raw_cells": cells,
            "source": "MOPS historical material information",
        }
    return list(events.values()), generic_rows


def build_opener_with_session():
    jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))
    headers = {
        "User-Agent": UA,
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    for url in (MOPS_HOME, MOPSOV_FORM):
        try:
            opener.open(Request(url, headers=headers), timeout=15).read(2048)
        except Exception:
            pass
    return opener


def direct_post(opener, params: dict[str, str], timeout: int = 30) -> str:
    body = urlencode(params).encode("utf-8")
    headers = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": MOPSOV_FORM,
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    with opener.open(Request(MOPSOV_AJAX, data=body, headers=headers, method="POST"), timeout=timeout) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", errors="replace")
    if len(text) < 200:
        raise RuntimeError(f"short direct MOPS response: {len(text)} bytes")
    return text


def gateway_fetch(opener, params: dict[str, str], timeout: int = 30) -> str:
    payload = json.dumps({"apiName": "ajax_t05st01", "parameters": params}, ensure_ascii=False).encode("utf-8")
    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Content-Type": "application/json",
        "Origin": "https://mops.twse.com.tw",
        "Referer": MOPS_HOME,
    }
    with opener.open(Request(MOPS_GATEWAY, data=payload, headers=headers, method="POST"), timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    if data.get("code") != 200:
        raise RuntimeError(f"MOPS gateway code={data.get('code')} message={data.get('message')}")
    url = (data.get("result") or {}).get("url")
    if not url:
        raise RuntimeError("MOPS gateway returned no legacy URL")
    with opener.open(Request(url, headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}), timeout=timeout) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", errors="replace")
    if len(text) < 200:
        raise RuntimeError(f"short gateway MOPS response: {len(text)} bytes")
    return text


def company_params(stock_code: str, roc_year: int, month: int, firstin: str = "true") -> dict[str, str]:
    return {
        "encodeURIComponent": "1",
        "step": "1",
        "firstin": firstin,
        "off": "1",
        "keyword4": "",
        "code1": "",
        "TYPEK2": "",
        "checkbtn": "",
        "queryName": "co_id",
        "inpuType": "co_id",
        "TYPEK": "all",
        "isnew": "false",
        "co_id": stock_code,
        "year": str(roc_year),
        "month": f"{month:02d}",
        "b_date": "",
        "e_date": "",
        "type": "",
    }


def market_month_params(typek: str, roc_year: int, month: int, firstin: str = "true") -> dict[str, str]:
    days = calendar.monthrange(roc_year + 1911, month)[1]
    return {
        "encodeURIComponent": "1",
        "step": "1",
        "firstin": firstin,
        "off": "1",
        "TYPEK": typek,
        "year": str(roc_year),
        "month": f"{month:02d}",
        "b_date": "01",
        "e_date": f"{days:02d}",
        "key_word": "",
        "queryName": "date",
        "inpuType": "date",
        "co_id": "",
        "keyword4": "",
        "code1": "",
        "TYPEK2": "",
        "checkbtn": "",
    }


def fetch_with_transport(opener, transport: str, params: dict[str, str]) -> str:
    if transport == "gateway":
        return gateway_fetch(opener, params)
    return direct_post(opener, params)


def probe_positive_control() -> tuple[str | None, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    c = POSITIVE_CONTROL
    for transport in ("gateway", "direct"):
        for firstin in ("true", "1", "ture"):
            opener = build_opener_with_session()
            params = company_params(c["stock_code"], c["roc_year"], c["month"], firstin)
            try:
                page = fetch_with_transport(opener, transport, params)
                hit = c["stock_code"] in page and c["subject_token"] in page and contains_43(page)
                events, generic_rows = parse_page(page, "positive_control")
                diagnostics.append({
                    "transport": transport,
                    "firstin": firstin,
                    "bytes": len(page),
                    "generic_rows": generic_rows,
                    "events": len(events),
                    "positive_control": hit,
                })
                if hit:
                    return transport, diagnostics
            except Exception as exc:
                diagnostics.append({
                    "transport": transport,
                    "firstin": firstin,
                    "positive_control": False,
                    "error": str(exc),
                })
    return None, diagnostics


def fetch_market_month(transport: str, typek: str, roc_year: int, month: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    best_events: list[dict[str, Any]] = []
    best_generic = 0
    for firstin in ("true", "1", "ture"):
        opener = build_opener_with_session()
        try:
            page = fetch_with_transport(opener, transport, market_month_params(typek, roc_year, month, firstin))
            events, generic_rows = parse_page(page, typek)
            diagnostics.append({
                "firstin": firstin,
                "bytes": len(page),
                "generic_rows": generic_rows,
                "events": len(events),
                "ok": True,
            })
            if generic_rows > best_generic:
                best_generic = generic_rows
                best_events = events
            # A populated monthly material-info table proves this variant works.
            if generic_rows > 0:
                return events, {"generic_rows": generic_rows, "attempts": diagnostics}
            if "查無資料" in page:
                return [], {"generic_rows": 0, "attempts": diagnostics}
        except Exception as exc:
            diagnostics.append({"firstin": firstin, "ok": False, "error": str(exc)})
    if best_generic > 0:
        return best_events, {"generic_rows": best_generic, "attempts": diagnostics}
    raise RuntimeError(f"no usable t05st01 monthly table; attempts={diagnostics}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def self_test() -> None:
    sample = """
    <table><tr><th>公司代號</th><th>公司簡稱</th><th>發言日期</th><th>發言時間</th><th>主旨</th></tr>
    <tr><td>4194</td><td>禾生技</td><td>115/10/12</td><td>10:01:00</td>
    <td>代取得人隆欣投資(股)公司依證券交易法第43條之1第1項規定辦理公告</td></tr></table>
    """
    rows, generic = parse_page(sample)
    assert generic == 1
    assert len(rows) == 1
    assert rows[0]["stock_code"] == "4194"
    assert rows[0]["event_date"] == "2026-10-12"
    assert "隆欣投資" in rows[0]["actor"]
    print("sec43-backfill self-test: OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/whale_radar")
    ap.add_argument("--start-year", type=int, default=113, help="ROC year; default 113 (2024) for first research pass")
    ap.add_argument("--end-year", type=int, default=datetime.now().year - 1911)
    ap.add_argument("--sleep", type=float, default=0.20)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0

    root = Path(args.root)
    latest = root / "latest"
    latest.mkdir(parents=True, exist_ok=True)

    transport, probe = probe_positive_control()
    (latest / "sec43_positive_control_probe.json").write_text(
        json.dumps({"control": POSITIVE_CONTROL, "transport": transport, "diagnostics": probe}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not transport:
        manifest = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": "BROKEN_SOURCE",
            "reason": "current t05st01 route could not retrieve known positive control 4194/隆欣投資/115-10",
            "events": 0,
            "named_actors": 0,
            "positive_control": POSITIVE_CONTROL,
        }
        (latest / "sec43_backfill_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("historical sec43: BROKEN_SOURCE; t05st01 positive control not retrieved")
        return 3

    print(f"positive control OK via t05st01/{transport}")
    all_rows: dict[str, dict[str, Any]] = {}
    coverage: list[dict[str, Any]] = []
    failures = 0
    total_generic_rows = 0
    now = date.today()

    for ry in range(args.start_year, args.end_year + 1):
        gy = ry + 1911
        last_month = now.month if gy == now.year else 12
        for mo in range(1, last_month + 1):
            for market_name, typek in MARKETS.items():
                try:
                    rows, diag = fetch_market_month(transport, typek, ry, mo)
                    total_generic_rows += int(diag.get("generic_rows", 0))
                    for row in rows:
                        row["market"] = market_name
                        all_rows[row["event_id"]] = row
                    coverage.append({
                        "roc_year": ry,
                        "month": mo,
                        "market": market_name,
                        "status": "OK",
                        "events": len(rows),
                        **diag,
                    })
                    print(f"MOPS {market_name} {ry}/{mo:02d}: rows={diag.get('generic_rows', 0)} sec43={len(rows)}")
                except Exception as exc:
                    failures += 1
                    coverage.append({
                        "roc_year": ry,
                        "month": mo,
                        "market": market_name,
                        "status": "FAILED",
                        "events": 0,
                        "error": str(exc),
                    })
                    print(f"MOPS {market_name} {ry}/{mo:02d}: FAILED: {exc}")
                    if not args.allow_partial:
                        raise
                time.sleep(max(0.0, args.sleep))

    rows = sorted(all_rows.values(), key=lambda x: (x.get("event_date", ""), x.get("stock_code", ""), x.get("event_id", "")))
    status = "OK" if failures == 0 else "PARTIAL"
    if total_generic_rows == 0:
        status = "BROKEN_SOURCE"

    write_jsonl(root / "historical_sec43_events.jsonl", rows)
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "MOPS historical material information (t05st01)",
        "transport": transport,
        "scope": ["listed", "otc"],
        "start_roc_year": args.start_year,
        "end_roc_year": args.end_year,
        "market_month_queries": len(coverage),
        "queries_failed": failures,
        "generic_material_info_rows": total_generic_rows,
        "status": status,
        "events": len(rows),
        "named_actors": len({x.get("actor") for x in rows if x.get("actor")}),
        "positive_control": POSITIVE_CONTROL,
        "coverage": coverage,
    }
    (latest / "sec43_backfill_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (latest / "historical_sec43_events.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"historical sec43: status={status}; material_rows={total_generic_rows}; "
        f"events={len(rows)}; named_actors={manifest['named_actors']}; failures={failures}/{len(coverage)}"
    )
    if status == "BROKEN_SOURCE":
        return 3
    return 0 if (failures == 0 or args.allow_partial) else 2


if __name__ == "__main__":
    raise SystemExit(main())
