#!/usr/bin/env python3
"""Fail-closed audit for the consumed case_id=4 V6.2/V4.2 two-stage regression.

This is an architecture regression only. It never opens outcomes or future paths and
must not be interpreted as predictive-value evidence.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ACTIONABLE={"CANDIDATE","HIGH_CONVICTION"}

def load1(path: Path):
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows)!=1:
        raise SystemExit(f"FAIL expected exactly one result, got {len(rows)}")
    return rows[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("result",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    r=load1(args.result)
    if int(r.get("case_id",-1))!=4:
        raise SystemExit("FAIL regression must be exact consumed case_id=4")
    if r.get("stage_c_received_causal_context_read_only") is not True:
        raise SystemExit("FAIL Stage C did not receive causal context")
    d=r.get("decision")
    if d not in {"REJECT","WATCH","CANDIDATE","HIGH_CONVICTION"}:
        raise SystemExit("FAIL invalid decision")
    actionable=d in ACTIONABLE
    entry=r.get("entry") or {}
    failure=r.get("failure_exit") or {}
    if actionable:
        if entry.get("status")=="NOT_ACTIONABLE":
            raise SystemExit("FAIL actionable_entry_not_actionable")
        lo,hi=entry.get("ideal_low"),entry.get("ideal_high")
        xp=failure.get("exit_price")
        if not all(isinstance(x,(int,float)) for x in (lo,hi,xp)):
            raise SystemExit("FAIL actionable branch missing numeric entry/failure exit")
        if not (0 < xp < hi and 0 < lo <= hi):
            raise SystemExit("FAIL invalid actionable price geometry")
        if float(r.get("action_seconds",0))<=0:
            raise SystemExit("FAIL actionable decision did not execute stage 2")
    else:
        if entry.get("status")!="NOT_ACTIONABLE" or failure.get("exit_price") is not None:
            raise SystemExit("FAIL non-actionable branch leaked action plan")
        if float(r.get("action_seconds",-1))!=0:
            raise SystemExit("FAIL non-actionable branch executed stage 2")
    for f in ("hypothesis","bull_thesis","bear_thesis","invalidation","decision_reason"):
        if not str(r.get(f,"")).strip():
            raise SystemExit(f"FAIL missing {f}")
    summary={
        "status":"PASS",
        "scope":"CONSUMED_CASE_4_ARCHITECTURE_REGRESSION_ONLY",
        "case_id":4,
        "decision":d,
        "actionable_branch_covered":actionable,
        "two_stage_semantics_consistent":True,
        "causal_context_received":True,
        "outcomes_opened":False,
        "future_paths_opened":False,
        "predictive_value_claimed":False,
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))
    if not actionable:
        raise SystemExit("FAIL case_id=4 did not cover actionable branch; do not expand replay")

if __name__=="__main__":
    main()
