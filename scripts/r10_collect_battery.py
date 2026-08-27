#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate matrix stress artifacts and apply the R10 regression trust gate."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

ORDER = [
    "validation2021_2025", "gfc2008", "euro2011", "china2015",
    "tradewar2018", "covid2020", "bear2022", "crash2024", "tariff2025",
    "synthetic_tail",
]


def find_summaries(root: Path):
    found = {}
    for p in root.rglob("summary.json"):
        try: obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception: continue
        name = obj.get("scenario")
        if not name and obj.get("type") == "SYNTHETIC_TAIL_DIAGNOSTIC": name = "synthetic_tail"
        if name: found[name] = obj
    return found


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output",default="stress_results/full_battery")
    a=ap.parse_args(); inp=Path(a.input); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    s=find_summaries(inp); validation=s.get("validation2021_2025",{}); gate=(validation.get("regression_gate") or {}).get("grade","FAIL")
    results=[]
    for name in ORDER:
        r=s.get(name)
        if not r:
            results.append({"scenario":name,"status":"MISSING","trust":"NOT_AVAILABLE"}); continue
        rr=dict(r)
        if name=="synthetic_tail": trust="DETERMINISTIC_DIAGNOSTIC"
        elif name=="validation2021_2025": trust="REGRESSION_GATE"
        elif str(r.get("mode","")).startswith("PARTIAL"):
            trust="PARTIAL_R10_R7_ONLY"
        elif gate=="PASS": trust="REGRESSION_VALIDATED_RECONSTRUCTION"
        else: trust="RESEARCH_RECONSTRUCTION_ONLY"
        rr["trust"]=trust; results.append(rr)
    payload={"generated_at":datetime.now().astimezone().isoformat(),"regression_gate":gate,"policy":"Historical FULL_R10 reconstruction is promoted only when the 2021-2025 regression gate passes. Partial pre-T86 periods never become full R10.","results":results}
    (out/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# AlphaPilot R10-MAX Full Stress Battery","",f"**Regression Gate: {gate}**","", "| Scenario | Status | Trust | End NAV | CAGR | Max DD | Trades |", "|---|---|---|---:|---:|---:|---:|"]
    for r in results:
        if r.get("status")!="PASS":
            lines.append(f"| {r.get('scenario')} | {r.get('status')} | {r.get('trust')} | — | — | — | — |")
            continue
        if r.get("scenario")=="synthetic_tail" or r.get("type")=="SYNTHETIC_TAIL_DIAGNOSTIC":
            c=r.get("critical_cases",{}).get("95pct_exposure_limit_down",{})
            lines.append(f"| synthetic_tail | PASS | {r.get('trust')} | — | — | 2LD {float(c.get('2',0)):.2%}; 3LD {float(c.get('3',0)):.2%} | — |")
        else:
            lines.append(f"| {r.get('scenario')} | PASS | {r.get('trust')} | {float(r.get('end_nav',0)):,.0f} | {float(r.get('cagr',0)):.2%} | {float(r.get('max_dd',0)):.2%} | {int(r.get('completed_trades',0))} |")
    lines += ["", "## Trust rule", "", "- PASS regression: post-2012 FULL_R10 historical reconstructions may be used as validated reconstructions.", "- FAIL regression: all reconstructed historical performance remains Research only; no parameter tuning is allowed to force a match.", "- 2008/2011 remain R7-only partial tests because required TWSE daily stock-level institutional inputs do not exist.", "- 2024/2025 overlap the locked development sample and are event diagnostics, not independent OOS evidence."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__": main()
