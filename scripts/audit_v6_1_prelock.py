#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/"scripts"))
import backtest_ai_market_reasoning_v6_1_corp_safe as v61

OUT=ROOT/"v6_1_prelock_audit"
OUT.mkdir(exist_ok=True)

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    panel,base_feats,_=v61.load_frozen_panel(require_lock=False)
    ds,feats=v61.add_safe_observable_transforms(panel,base_feats)

    reset_rows=ds.loc[ds["price_reset"],["date","code","close","raw_ret1_for_reset","price_segment_id"]].copy()
    max_safe=float(ds["ret1"].abs().max(skipna=True))
    if max_safe>v61.RESET_ABS_RAW_RETURN+1e-7:
        raise RuntimeError(f"safe ret1 still exceeds reset threshold: {max_safe}")

    cross={}
    for h in v61.AUDIT_HORIZONS:
        g=ds.groupby("code",group_keys=False)
        target_seg=g["price_segment_id"].shift(-h)
        raw_has_target=target_seg.notna()
        safe=v61.fwd_return(ds,h)
        violation=(safe.notna() & raw_has_target & (target_seg.to_numpy()!=ds["price_segment_id"].to_numpy()))
        n=int(violation.sum())
        cross[str(h)]=n
        if n:
            raise RuntimeError(f"cross-reset forward label violations h={h}: {n}")

    support=[]
    for year in (2021,2022,2023,2024):
        rec={"year":year}
        try:
            train,test,cutoff=v61.year_context(ds,year)
            rec.update({"cutoff":cutoff,"train_rows":int(len(train)),"test_rows":int(len(test))})
            all_ok=True
            for h in v61.AUDIT_HORIZONS:
                y=v61.fwd_return(train,h).to_numpy(dtype=float)
                n=int(np.isfinite(y).sum())
                rec[f"naive_support_h{h}"]=n
                if n<1000: all_ok=False
            rec["all_six_horizons_support_ge_1000"]=bool(all_ok)
        except Exception as e:
            rec["error"]=str(e); rec["all_six_horizons_support_ge_1000"]=False
        support.append(rec)

    for rec in support:
        if rec["year"] in (2022,2023,2024) and not rec["all_six_horizons_support_ge_1000"]:
            raise RuntimeError(f"formal repair-validation support insufficient: {rec}")

    by_year=(reset_rows.assign(year=(reset_rows["date"]//10000).astype(int))
             .groupby("year").size().rename("reset_events").reset_index())

    result={
      "status":"PASS",
      "panel_sha256":sha(v61.PANEL_PATH),
      "feature_count":len(feats),
      "reset_threshold_abs_raw_close_return":v61.RESET_ABS_RAW_RETURN,
      "reset_events_total":int(ds["price_reset"].sum()),
      "max_abs_safe_ret1":max_safe,
      "cross_reset_forward_label_violations":cross,
      "support":support,
      "2021_formal_score_authorized":False,
      "repair_validation_years":[2022,2023,2024],
      "2025_opened":False,
      "2026_opened":False
    }
    reset_rows.to_csv(OUT/"V6_1_RESET_BOUNDARIES.csv",index=False,encoding="utf-8-sig")
    by_year.to_csv(OUT/"V6_1_RESET_BOUNDARIES_BY_YEAR.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(support).to_csv(OUT/"V6_1_YEAR_SUPPORT.csv",index=False,encoding="utf-8-sig")
    (OUT/"V6_1_PRELOCK_AUDIT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    files=[
      ROOT/"research"/"V6_1_CORP_SAFE_PREREGISTRATION.md",
      ROOT/"scripts"/"backtest_ai_market_reasoning_v6_1_corp_safe.py",
      ROOT/"scripts"/"audit_v6_1_prelock.py",
      ROOT/".github"/"workflows"/"research_ai_market_reasoning_v6_1_prelock.yml"
    ]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    (OUT/"V6_1_PRELOCK_FILE_HASHES.json").write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("[V6.1 PRELOCK PASS]",json.dumps(result,ensure_ascii=False),flush=True)
    print("[V6.1 FILE HASHES]",json.dumps(hashes,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
