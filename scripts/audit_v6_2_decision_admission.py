#!/usr/bin/env python3
"""V6.2 decision-layer scientific audit.

This is deliberately separate from 0A-0F and R10.  It does not change the
selector, labels, forecasts, or frozen evidence.  It prevents an engineering
PASS from being mistaken for a scientifically useful decision layer.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ACTIONABLE = {"CANDIDATE", "HIGH_CONVICTION"}


def rows(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    obj = json.loads(path.read_text())
    if isinstance(obj, list): return obj
    for key in ("results", "rows", "cases"):
        if isinstance(obj.get(key), list): return obj[key]
    return [obj]


def audit(rs, min_cases=4):
    n=len(rs); counts={k:0 for k in ("REJECT","WATCH","CANDIDATE","HIGH_CONVICTION")}
    contradictions=[]; action_calls=0
    for r in rs:
        d=r.get("decision")
        if d in counts: counts[d]+=1
        entry=(r.get("entry") or {}).get("status")
        fx=r.get("failure_exit") or {}
        if d in ACTIONABLE:
            action_calls+=1
            if entry in (None,"NOT_ACTIONABLE") or fx.get("exit_price") is None:
                contradictions.append(r.get("case_id"))
        elif d in ("REJECT","WATCH"):
            if entry not in (None,"NOT_ACTIONABLE"):
                contradictions.append(r.get("case_id"))
    actionable=counts["CANDIDATE"]+counts["HIGH_CONVICTION"]
    gates={
      "semantic_consistency": not contradictions,
      "actionable_branch_covered": action_calls>0,
      # Diagnostic, not a quota: zero admissions in a nontrivial evaluation set
      # means admission value has not been demonstrated and must not be called PASS.
      "admission_value_demonstrated": not (n>=min_cases and actionable==0),
    }
    status="PASS" if all(gates.values()) else "FAIL"
    return {"status":status,"n":n,"counts":counts,"actionable_rate": actionable/n if n else None,
            "gates":gates,"semantic_contradiction_case_ids":contradictions,
            "interpretation":"Engineering success is insufficient unless actionable semantics are exercised and admission does not collapse to zero."}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("input", type=Path); ap.add_argument("--out", type=Path)
    a=ap.parse_args(); result=audit(rows(a.input)); text=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if a.out: a.out.write_text(text)
    print(text,end="")
    raise SystemExit(0 if result["status"]=="PASS" else 2)

if __name__=="__main__": main()
