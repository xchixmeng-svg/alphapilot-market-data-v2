#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate R10 battery with hard FULL-R10 completion gates."""
from __future__ import annotations
import argparse,json
from datetime import datetime
from pathlib import Path

HIST=["gfc2008","euro2011","china2015","tradewar2018","covid2020","bear2022","crash2024","tariff2025"]
ORDER=["finmind_calibration","validation2021_2025"]+HIST+["synthetic_tail"]

def find_summaries(root:Path):
    found={}
    for p in root.rglob("summary.json"):
        try:o=json.loads(p.read_text(encoding="utf-8"))
        except Exception:continue
        name=o.get("scenario")
        if not name and o.get("type")=="SYNTHETIC_TAIL_DIAGNOSTIC":name="synthetic_tail"
        if not name and o.get("type")=="FINMIND_OFFICIAL_CALIBRATION":name="finmind_calibration"
        if name:found[name]=o
    return found

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--output",default="stress_results/full_battery");a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);s=find_summaries(Path(a.input))
    validation=s.get("validation2021_2025",{});reg=(validation.get("regression_gate") or {}).get("grade","FAIL")
    cal=s.get("finmind_calibration",{});calgate="PASS" if cal.get("status")=="PASS" and cal.get("gate") is True else "FAIL"
    results=[];full_ok=True
    for name in ORDER:
        r=s.get(name)
        if not r:
            results.append({"scenario":name,"status":"MISSING","trust":"NOT_COMPLETE"});full_ok=False;continue
        rr=dict(r);mode=str(r.get("mode",""))
        if name=="finmind_calibration":trust="SOURCE_CALIBRATION_GATE"
        elif name=="synthetic_tail":trust="DETERMINISTIC_DIAGNOSTIC"
        elif name=="validation2021_2025":trust="REGRESSION_GATE"
        elif mode.startswith("PARTIAL"):
            trust="PROHIBITED_PARTIAL";full_ok=False
        elif r.get("status")!="PASS":trust="NOT_COMPLETE";full_ok=False
        elif reg!="PASS":trust="RESEARCH_ONLY_REGRESSION_NOT_PASSED"
        elif name in {"gfc2008","euro2011"} and calgate!="PASS":trust="RESEARCH_ONLY_SOURCE_NOT_CALIBRATED"
        else:trust="FULL_R10_VALIDATED_RECONSTRUCTION"
        rr["trust"]=trust;results.append(rr)
        if name in HIST and (r.get("status")!="PASS" or mode.startswith("PARTIAL") or not bool(r.get("r05_enabled",False))):full_ok=False
    if reg!="PASS" or calgate!="PASS":full_ok=False
    overall="PASS" if full_ok else "INCOMPLETE"
    payload={"status":overall,"generated_at":datetime.now().astimezone().isoformat(),"regression_gate":reg,"finmind_calibration_gate":calgate,"policy":"A complete battery requires FULL R10 (R7+R0.5+Portfolio Layer) for every historical scenario, PASS 2021-2025 regression, and PASS FinMind-vs-official calibration. Partial/R7-only results are prohibited from the final performance table.","results":results}
    (out/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# AlphaPilot R10-MAX Complete Stress Battery","",f"**Overall: {overall}**",f"**Regression Gate: {reg}**",f"**FinMind Calibration Gate: {calgate}**","","| Scenario | Status | Mode/Trust | End NAV | CAGR | Max DD | Trades |","|---|---|---|---:|---:|---:|---:|"]
    for r in results:
        n=r.get("scenario") or r.get("type")
        if n=="finmind_calibration":lines.append(f"| finmind_calibration | {r.get('status')} | {r.get('trust')} | — | — | — | {r.get('matched_pairs','—')} pairs |")
        elif n=="synthetic_tail":
            c=r.get("critical_cases",{}).get("95pct_exposure_limit_down",{});lines.append(f"| synthetic_tail | {r.get('status')} | {r.get('trust')} | — | — | 2LD {float(c.get('2',0)):.2%}; 3LD {float(c.get('3',0)):.2%} | — |")
        elif r.get("status")!="PASS":lines.append(f"| {n} | {r.get('status')} | {r.get('trust')} | — | — | — | — |")
        else:lines.append(f"| {n} | PASS | {r.get('trust')} | {float(r.get('end_nav',0)):,.0f} | {float(r.get('cagr',0)):.2%} | {float(r.get('max_dd',0)):.2%} | {int(r.get('completed_trades',0))} |")
    lines += ["","## Hard completion rule","","- No R7-only / Partial result is accepted as completed R10.","- 2008 and 2011 must include R0.5 using 2005+ historical institutional data and must pass the FinMind-vs-official overlap calibration gate.","- The reconstructed engine must pass the locked 2021-2025 regression gate before any historical performance is promoted as validated.","- 2024/2025 overlap the locked research sample and remain event diagnostics even when technically complete."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print("\n".join(lines))
    if overall!="PASS":raise SystemExit(2)
if __name__=="__main__":main()
