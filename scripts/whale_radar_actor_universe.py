#!/usr/bin/env python3
"""Build the first Whale Radar tracking universe from public >10% shareholder data.

This is NOT a performance ranking.  It answers a narrower question first:
"which named people/entities are worth monitoring for future public changes?"

Inputs are produced by whale_radar_collect.py.  The script classifies holders,
counts how many distinct listed companies each holder currently appears in, and
creates a transparent tracking priority.  Strategic corporate cross-holdings
and government holdings are deliberately separated from investment-style
holders so they are not mislabeled as stock-picking skill.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

GOV_TOKENS = (
    "財政部", "交通部", "經濟部", "行政院", "國家發展基金", "退除役官兵",
    "國軍", "地方政府", "市政府", "縣政府", "公務人員退休", "勞動基金",
)
ASSET_TOKENS = ("資產管理", "資產經營", "資本管理", "基金管理")
INVEST_TOKENS = ("投資", "創投", "創業投資", "控股")
COMPANY_SUFFIXES = ("股份有限公司", "有限公司", "公司", "財團法人", "基金會", "銀行")


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    return [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []


def classify(name: str) -> str:
    if any(t in name for t in GOV_TOKENS):
        return "public_sector"
    if any(t in name for t in ASSET_TOKENS):
        return "asset_manager"
    if any(t in name for t in INVEST_TOKENS):
        return "investment_vehicle"
    if any(name.endswith(s) for s in COMPANY_SUFFIXES):
        return "operating_company"
    return "individual"


def priority(kind: str, n_companies: int) -> tuple[int, list[str]]:
    score = 20
    reasons: list[str] = []
    if n_companies >= 2:
        add = 12 * min(n_companies - 1, 5)
        score += add
        reasons.append(f"currently appears as >10% holder in {n_companies} companies")
    if kind == "asset_manager":
        score += 25
        reasons.append("asset-management style holder")
    elif kind == "investment_vehicle":
        score += 18
        reasons.append("investment/holding vehicle")
    elif kind == "individual" and n_companies >= 2:
        score += 15
        reasons.append("same named individual appears across multiple companies")
    elif kind == "operating_company":
        score -= 8
        reasons.append("may be strategic/group cross-holding rather than trading alpha")
    elif kind == "public_sector":
        score -= 35
        reasons.append("public-sector holding; low value for copy-trading research")
    return max(0, min(100, score)), reasons


def build(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_actor: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for r in rows:
        actor = str(r.get("大股東名稱", "")).strip()
        code = str(r.get("公司代號", "")).strip()
        name = str(r.get("公司名稱", "")).strip()
        if actor and code:
            by_actor[actor][code] = (code, name)

    out: list[dict[str, Any]] = []
    for actor, companies in by_actor.items():
        kind = classify(actor)
        n = len(companies)
        score, reasons = priority(kind, n)
        ordered = sorted(companies.values())
        watch = kind != "public_sector" and (
            n >= 2 or kind in {"asset_manager", "investment_vehicle"}
        )
        out.append({
            "actor": actor,
            "actor_kind": kind,
            "distinct_companies": n,
            "tracking_priority": score,
            "watch": watch,
            "stock_codes": " | ".join(x[0] for x in ordered),
            "companies": " | ".join(x[1] for x in ordered),
            "reasons": "; ".join(reasons) if reasons else "single current >10% holding",
        })
    out.sort(key=lambda x: (-int(x["watch"]), -x["tracking_priority"], -x["distinct_companies"], x["actor"]))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["actor", "actor_kind", "distinct_companies", "tracking_priority", "watch", "stock_codes", "companies", "reasons"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def self_test() -> None:
    rows = [
        {"公司代號": "1504", "公司名稱": "東元", "大股東名稱": "寶佳資產管理股份有限公司"},
        {"公司代號": "2820", "公司名稱": "華票", "大股東名稱": "寶佳資產管理股份有限公司"},
        {"公司代號": "2603", "公司名稱": "長榮", "大股東名稱": "張國華"},
        {"公司代號": "2211", "公司名稱": "長榮鋼", "大股東名稱": "張國華"},
        {"公司代號": "2002", "公司名稱": "中鋼", "大股東名稱": "經濟部"},
    ]
    out = {x["actor"]: x for x in build(rows)}
    assert out["寶佳資產管理股份有限公司"]["actor_kind"] == "asset_manager"
    assert out["寶佳資產管理股份有限公司"]["watch"] is True
    assert out["張國華"]["actor_kind"] == "individual"
    assert out["經濟部"]["watch"] is False
    print("actor-universe self-test: OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/whale_radar")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    root = Path(args.root)
    source = root / "latest" / "large_shareholders_10pct.json"
    rows = build(load_json(source))
    watch = [x for x in rows if x["watch"]]
    (root / "latest").mkdir(parents=True, exist_ok=True)
    (root / "latest" / "actor_universe.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "latest" / "watchlist_actors.json").write_text(json.dumps(watch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(root / "latest" / "actor_universe.csv", rows)
    write_csv(root / "latest" / "watchlist_actors.csv", watch)
    print(f"actor universe: {len(rows)} named holders; watchlist: {len(watch)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
