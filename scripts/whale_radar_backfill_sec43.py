#!/usr/bin/env python3
"""Backfill Article 43-1 disclosures from the official MOPS announcement search.

Research safety rules:
- Never trust an empty history until a real, independently verified 43-1 control passes.
- Do not brute-force hundreds of MOPS requests from GitHub Actions; fail fast on cloud throttling.
- Keep source/coverage diagnostics so a transport block is never misread as 'no events'.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import html
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path

HOSTS = ("https://mopsov.twse.com.tw", "https://mops.twse.com.tw")
PAGE = "/mops/web/t146sb10"
PATH = "/mops/web/ajax_t146sb10"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36"

# These controls are real public 43-1 notices that are independently indexed outside
# our self-test.  The old 4194/隆欣投資 example was synthetic/unverified and must not be
# used as evidence.
CONTROLS = (
    {
        "roc_year": 115,
        "month": 10,
        "day": 6,
        "stock_code": "5355",
        "actor_token": "李茂昌",
        "market": "otc",
        "date": "2026-10-06",
        "company": "佳總",
    },
    {
        "roc_year": 115,
        "month": 9,
        "day": 22,
        "stock_code": "6214",
        "actor_token": "劉興漢",
        "market": "sii",
        "date": "2026-09-22",
        "company": "精誠",
    },
)


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            text = re.sub(r"\s+", " ", html.unescape("".join(self._cell))).strip()
            self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def contains_43(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return "43條之1" in compact or "第四十三條之一" in compact or "43-1" in compact


def parse_roc_date(text: str) -> str:
    for pat in (
        r"(?<!\d)(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})(?!\d)",
        r"(?<!\d)(\d{3})(\d{2})(\d{2})(?!\d)",
        r"(?<!\d)(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?!\d)",
    ):
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
    return ""


def extract_actor(text: str) -> str:
    for pat in (
        r"代取得人\s*[:：]?\s*(.+?)(?:依|$)",
        r"共同取得人\s*[:：]?\s*(.+?)(?:依|$)",
        r"取得人\s*[:：]?\s*(.+?)(?:依|$)",
    ):
        m = re.search(pat, text)
        if m:
            return m.group(1).strip(" ：:，,。()（）")
    return ""


def parse_rows(page: str):
    p = TableParser()
    p.feed(page)
    out = {}
    for cells in p.rows:
        joined = " | ".join(cells)
        if not contains_43(joined):
            continue
        code = ""
        for c in cells:
            m = re.fullmatch(r"\s*(\d{4,6})\s*", c)
            if m:
                code = m.group(1)
                break
        event_date = parse_roc_date(joined)
        if not code or not event_date:
            continue
        actor = extract_actor(joined)
        subject = next((c for c in cells if contains_43(c)), max(cells, key=len))
        company_name = ""
        for i, c in enumerate(cells):
            if c.strip() == code and i + 1 < len(cells):
                company_name = cells[i + 1].strip()
                break
        eid = hashlib.sha256(f"{code}|{event_date}|{joined}".encode()).hexdigest()[:24]
        out[eid] = {
            "event_id": eid,
            "event_type": "sec43_1_disclosure",
            "dataset": "mops_t146sb10",
            "stock_code": code,
            "company_name": company_name,
            "actor": actor,
            "event_date": event_date,
            "available_at": event_date,
            "signal_score": 50,
            "direction": "bullish_watch",
            "subject": subject,
            "raw_cells": cells,
            "source": "MOPS t146sb10 announcement search",
        }
    return list(out.values())


def curl(host: str, path: str, payload=None, jar=None, max_time=20):
    cmd = [
        "curl", "-sS", "-L", "--http1.1", "--compressed",
        "--connect-timeout", "5", "--max-time", str(max_time),
        "-A", UA,
    ]
    if jar:
        cmd += ["-c", jar, "-b", jar]
    if payload is not None:
        cmd += [
            "-H", "Content-Type: application/x-www-form-urlencoded",
            "-H", "Origin: " + host,
            "-e", host + PAGE,
        ]
        for k, v in payload.items():
            cmd += ["--data-urlencode", f"{k}={v}"]
    cmd.append(host + path)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + 10)
    if p.returncode:
        raise RuntimeError((p.stderr or "")[-300:])
    text = p.stdout
    if len(text) < 200:
        raise RuntimeError(f"short response {len(text)}")
    if "查詢過於頻繁" in text or "FOR SECURITY" in text or "FOR SECURITY REASONS" in text:
        raise RuntimeError("MOPS throttle/security")
    return text


def discover_43_kinds(page: str):
    kinds = []
    for val, label in re.findall(
        r"<option[^>]*value=[\"']?([^\"' >]+)[\"']?[^>]*>(.*?)</option>",
        page, re.I | re.S,
    ):
        label = re.sub(r"\s+", "", re.sub("<[^>]+>", "", html.unescape(label)))
        if "43條之1" in label or "第四十三條之一" in label or "43-1" in label:
            kinds.append(val)
    return list(dict.fromkeys(kinds))


def base_payload(control, kind, d1, d2, notice_date="1", selecttype="0"):
    market = control["market"]
    company = control["stock_code"]
    return {
        "encodeURIComponent": "1",
        "step": "1",
        "firstin": "true",
        "off": "1",
        "keyword4": "",
        "code1": company,
        "TYPEK2": market,
        "checkbtn": "",
        "queryName": "co_id_1",
        "inpuType": "co_id",
        "scope": "1",
        "co_id_1": company,
        "co_id_2": company,
        "typek": market,
        "TYPEK": market,
        "selecttype": selecttype,
        "date": "4",
        "noticeDate": notice_date,
        "yymmdd1": d1,
        "yymmdd2": d2,
        "date1": d1,
        "date2": d2,
        "noticeKind": kind,
        "sort": "1",
    }


def control_date_variants(c):
    ry, mo, d = c["roc_year"], c["month"], c["day"]
    gy = ry + 1911
    values = (
        f"{ry:03d}{mo:02d}{d:02d}",
        f"{ry:03d}/{mo:02d}/{d:02d}",
        f"{gy:04d}{mo:02d}{d:02d}",
        f"{gy:04d}/{mo:02d}/{d:02d}",
    )
    for v in values:
        for notice_date in ("1", "2"):
            for selecttype in ("0", "1"):
                yield v, notice_date, selecttype


def probe_control():
    diagnostics = []
    for host in HOSTS:
        fd, jar = tempfile.mkstemp(prefix="whale-mops-", suffix=".cookie")
        os.close(fd)
        try:
            try:
                form = curl(host, PAGE, jar=jar, max_time=12)
                kinds = discover_43_kinds(form)
                diagnostics.append({"host": host, "stage": "form", "bytes": len(form), "kinds": kinds})
            except Exception as exc:
                diagnostics.append({"host": host, "stage": "form", "error": repr(exc)})
                continue

            # If the official form does not expose an Article 43-1 category, do not guess
            # category numbers. Guessing is what caused the earlier hundreds of useless calls.
            if not kinds:
                diagnostics.append({"host": host, "stage": "category", "error": "43-1 noticeKind not discovered"})
                continue

            for control in CONTROLS:
                for kind in kinds:
                    for i, (d, notice_date, selecttype) in enumerate(control_date_variants(control)):
                        q = base_payload(control, kind, d, d, notice_date, selecttype)
                        try:
                            text = curl(host, PATH, q, jar=jar, max_time=15)
                            compact = re.sub(r"\s+", "", text)
                            hit = (
                                control["stock_code"] in compact
                                and control["actor_token"] in compact
                                and contains_43(compact)
                            )
                            diagnostics.append({
                                "host": host,
                                "control": control["stock_code"],
                                "kind": kind,
                                "try": i,
                                "date": d,
                                "noticeDate": notice_date,
                                "selecttype": selecttype,
                                "bytes": len(text),
                                "hit": hit,
                            })
                            if hit:
                                contract = {
                                    "host": host,
                                    "kind": kind,
                                    "market": control["market"],
                                    "noticeDate": notice_date,
                                    "selecttype": selecttype,
                                    "date_style": "slash" if "/" in d else "compact",
                                    "gregorian": d.startswith("20"),
                                    "verified_control": control,
                                }
                                return parse_rows(text), diagnostics, contract
                        except Exception as exc:
                            diagnostics.append({
                                "host": host,
                                "control": control["stock_code"],
                                "kind": kind,
                                "try": i,
                                "error": repr(exc),
                            })
                            if "throttle/security" in repr(exc):
                                # Stop hammering this host. A cloud-IP block is a transport
                                # condition, not evidence that the filing does not exist.
                                break
                        time.sleep(0.35)
                    else:
                        continue
                    break
                else:
                    continue
                break
        finally:
            try:
                os.unlink(jar)
            except OSError:
                pass
    return [], diagnostics, None


def format_date(ry, mo, d, contract):
    y = ry + 1911 if contract["gregorian"] else ry
    if contract["date_style"] == "slash":
        return f"{y:04d}/{mo:02d}/{d:02d}" if contract["gregorian"] else f"{y:03d}/{mo:02d}/{d:02d}"
    return f"{y:04d}{mo:02d}{d:02d}" if contract["gregorian"] else f"{y:03d}{mo:02d}{d:02d}"


def query_month(ry, mo, contract):
    last = calendar.monthrange(ry + 1911, mo)[1]
    d1 = format_date(ry, mo, 1, contract)
    d2 = format_date(ry, mo, last, contract)
    pseudo_control = {"market": contract["market"], "stock_code": ""}
    q = base_payload(
        pseudo_control,
        contract["kind"],
        d1,
        d2,
        contract["noticeDate"],
        contract["selecttype"],
    )
    fd, jar = tempfile.mkstemp(prefix="whale-mops-", suffix=".cookie")
    os.close(fd)
    try:
        curl(contract["host"], PAGE, jar=jar, max_time=12)
        text = curl(contract["host"], PATH, q, jar=jar, max_time=18)
        return parse_rows(text), {"bytes": len(text), "d1": d1, "d2": d2}
    finally:
        try:
            os.unlink(jar)
        except OSError:
            pass


def self_test():
    sample = """
    <table><tr><td>5355</td><td>佳總</td><td>115/10/06</td>
    <td>代取得人李茂昌依證券交易法第43條之1第1項規定辦理公告</td></tr></table>
    """
    rows = parse_rows(sample)
    assert len(rows) == 1
    assert rows[0]["stock_code"] == "5355"
    assert rows[0]["event_date"] == "2026-10-06"
    assert "李茂昌" in rows[0]["actor"]
    print("sec43-backfill self-test: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/whale_radar")
    ap.add_argument("--start-year", type=int, default=113)
    ap.add_argument("--end-year", type=int, default=datetime.now().year - 1911)
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    root = Path(args.root)
    latest = root / "latest"
    latest.mkdir(parents=True, exist_ok=True)

    rows, diagnostics, contract = probe_control()
    probe = {
        "controls": CONTROLS,
        "positive_control": contract is not None,
        "route": "t146sb10",
        "contract": contract,
        "diagnostics": diagnostics,
    }
    (latest / "sec43_positive_control_probe.json").write_text(
        json.dumps(probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if contract is None:
        throttled = any("throttle/security" in str(x.get("error", "")) for x in diagnostics)
        status = "MOPS_CLOUD_BLOCKED" if throttled else "BROKEN_SOURCE"
        manifest = {
            "status": status,
            "reason": "verified controls were not retrievable from this runner",
            "events": 0,
            "named_actors": 0,
            **probe,
        }
        (latest / "sec43_backfill_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"historical sec43: {status}; verified controls not retrieved")
        return 3

    print("positive control OK; contract=" + json.dumps(contract, ensure_ascii=False))
    allrows = {}
    coverage = []
    failures = 0
    now = date.today()

    for ry in range(args.start_year, args.end_year + 1):
        last_month = now.month if ry + 1911 == now.year else 12
        for mo in range(1, last_month + 1):
            try:
                rr, meta = query_month(ry, mo, contract)
                for row in rr:
                    allrows[row["event_id"]] = row
                coverage.append({"roc_year": ry, "month": mo, "status": "OK", "events": len(rr), **meta})
                print(f"MOPS t146 {ry}/{mo:02d}: 43-1={len(rr)}")
            except Exception as exc:
                failures += 1
                coverage.append({"roc_year": ry, "month": mo, "status": "FAILED", "events": 0, "error": str(exc)})
                print(f"MOPS t146 {ry}/{mo:02d}: FAILED {exc}")
                if not args.allow_partial:
                    raise
            time.sleep(max(0.0, args.sleep))

    out = sorted(allrows.values(), key=lambda x: (x["event_date"], x["stock_code"]))
    root.mkdir(parents=True, exist_ok=True)
    with (root / "historical_sec43_events.jsonl").open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "MOPS t146sb10 dedicated announcement search",
        "contract": contract,
        "start_roc_year": args.start_year,
        "end_roc_year": args.end_year,
        "months_attempted": len(coverage),
        "months_failed": failures,
        "status": "OK" if failures == 0 else "PARTIAL",
        "events": len(out),
        "named_actors": len({x["actor"] for x in out if x["actor"]}),
        "verified_controls": CONTROLS,
        "coverage": coverage,
    }
    (latest / "sec43_backfill_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (latest / "historical_sec43_events.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"historical sec43: {len(out)} events; named actors={manifest['named_actors']}; "
        f"failures={failures}/{len(coverage)}"
    )
    return 0 if failures == 0 or args.allow_partial else 2


if __name__ == "__main__":
    raise SystemExit(main())
