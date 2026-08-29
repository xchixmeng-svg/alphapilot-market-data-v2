#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

START_YEAR = 2007
END_YEAR = 2019
BASE_URL = "https://github.com/yukishirotsubasa/tw-stock-data-release/releases/download/daily-close-csv/yearly_{year}.zip"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "history" / "2007-2019" / "raw"
MANIFEST = ROOT / "data" / "history" / "2007-2019" / "manifest.json"

# SHA-256 digests published by the upstream GitHub Release.
EXPECTED = {
    2007: "ae2848f0a5484e418d85edeafcbcf8d47c41dbf877288ce160bb197c2dc71370",
    2008: "4e895ae5dc3fbca080d6aea62ae595cffe80eae89d9b39b22ad43db89fef4a4e",
    2009: "820fb45b206f2444916fcd7ed9ffd20ca408eb6ea2e38003b60c4885dbbc1402",
    2010: "dbd99850cf8c2893cdc52e5a974c16e5231977d89f0f0920b7bc03dfc9fa64ae",
    2011: "382f8a0768e7e877f1335b7650e1288b4ec08511a81d88049d7c5feec1745f1d",
    2012: "ff20850e04580c1a3d4761d52e2e6822924deef08f64a51973ca535a94aacfb0",
    2013: "d9cf0192ad149a63d13437e307b5a024aefe592f03b148d6b6f3012a50563b49",
    2014: "2ccc5e2a5d3fcaeaaa303a60310282407a68f05846fc38d276b257ddac7e04ca",
    2015: "a0680d885f6313a240b2587f07ac40064eab6ea7772f53fcfd8bdab9fa76430d",
    2016: "6f55396eb9ebeb3906136fa4d8cdc50b3c8b7964f8db464cb010c7692b9c6710",
    2017: "d895d8d992151e52e58254c484270ce4c3be9f3653151525982edfc40cadeb9a",
    2018: "41bc0410ed5882f4e2764776aae95f5f8f2bde71d0c7ce52b0eb8a149d0fe5ab",
    2019: "6ff9ef9b46b359fb2ecc3a55ad95300ae09c0355c458edfc78b7aeaa9d2b2438",
}

REQUIRED_COLUMNS = {"date", "code", "name", "volume", "open", "high", "low", "close"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "AlphaPilot-Data-Archive/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def audit_zip(data: bytes, year: int) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"{year}: corrupt ZIP member: {bad}")
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"{year}: ZIP has no CSV")
        member = csv_names[0]
        with zf.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            fields = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - fields)
            if missing:
                raise RuntimeError(f"{year}: missing columns {missing}")
            rows = 0
            min_date = None
            max_date = None
            stock_0050_rows = 0
            for row in reader:
                rows += 1
                raw_date = str(row.get("date", "")).strip()
                try:
                    d = int(float(raw_date))
                except Exception:
                    continue
                min_date = d if min_date is None else min(min_date, d)
                max_date = d if max_date is None else max(max_date, d)
                if str(row.get("code", "")).strip().replace(".0", "") == "0050":
                    stock_0050_rows += 1
        if rows <= 0 or min_date is None or max_date is None:
            raise RuntimeError(f"{year}: empty/invalid CSV")
        if min_date // 10000 != year or max_date // 10000 != year:
            raise RuntimeError(f"{year}: date range mismatch {min_date}..{max_date}")
        return {
            "csv_member": member,
            "rows": rows,
            "min_date": min_date,
            "max_date": max_date,
            "0050_rows": stock_0050_rows,
            "columns": sorted(fields),
        }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "AlphaPilot Taiwan OHLCV historical stress archive",
        "years": [START_YEAR, END_YEAR],
        "source": "yukishirotsubasa/tw-stock-data-release GitHub Release daily-close-csv",
        "source_coverage_note": "Upstream release states TWSE MI_INDEX + TPEx OTC daily OHLCV.",
        "immutable_policy": "2007-2019 raw ZIPs are preserved after SHA-256 verification; 2020-2025 archive is untouched.",
        "files": [],
    }

    for year in range(START_YEAR, END_YEAR + 1):
        url = BASE_URL.format(year=year)
        path = OUT / f"yearly_{year}.zip"
        expected = EXPECTED[year]

        if path.exists():
            data = path.read_bytes()
            digest = sha256(data)
            if digest != expected:
                raise RuntimeError(f"{year}: existing file digest mismatch {digest} != {expected}")
            status = "existing_verified"
        else:
            print(f"[download] {year} {url}", flush=True)
            data = download(url)
            digest = sha256(data)
            if digest != expected:
                raise RuntimeError(f"{year}: downloaded digest mismatch {digest} != {expected}")
            path.write_bytes(data)
            status = "downloaded_verified"

        audit = audit_zip(data, year)
        manifest["files"].append({
            "year": year,
            "path": str(path.relative_to(ROOT)),
            "url": url,
            "sha256": digest,
            "expected_sha256": expected,
            "bytes": len(data),
            "status": status,
            **audit,
        })
        print(f"[PASS] {year} rows={audit['rows']} dates={audit['min_date']}..{audit['max_date']} sha256={digest}", flush=True)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] wrote {MANIFEST.relative_to(ROOT)} for {len(manifest['files'])} years", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
