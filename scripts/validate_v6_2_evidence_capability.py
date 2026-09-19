#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--registry",required=True)
    ap.add_argument("--response",required=True)
    ap.add_argument("--out",required=True)
    ns=ap.parse_args()
    reg=json.loads(Path(ns.registry).read_text())
    rsp=json.loads(Path(ns.response).read_text())
    fam=reg["families"]
    used=set(rsp.get("evidence_used") or rsp.get("primary_evidence_ids") or [])
    unsupported=[]
    for x in used:
        if x in fam and fam[x]["status"]=="UNAVAILABLE":
            unsupported.append(x)
    txt=json.dumps(rsp,ensure_ascii=False).lower()
    semantic={
      "eps_revisions":["eps revision","eps revisions","analyst estimate revision"],
      "analyst_consensus":["analyst consensus","consensus estimate"],
      "industry_pricing":["industry pricing","product price increase","spot price"],
      "inventory_supply_demand":["inventory destocking","supply shortage","supply-demand","inventory cycle"],
      "broad_news_semantics":["news sentiment","media sentiment"]
    }
    for k,terms in semantic.items():
        if fam[k]["status"]=="UNAVAILABLE" and any(t in txt for t in terms):
            unsupported.append(k)
    unsupported=sorted(set(unsupported))
    audit={"status":"PASS" if not unsupported else "FAIL","unsupported_claim_families":unsupported}
    Path(ns.out).write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(audit,ensure_ascii=False))
    if unsupported: raise SystemExit(2)
if __name__=="__main__": main()
