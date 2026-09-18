#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT=Path("v6_2_evidence_bundle_v2")
ROOT.mkdir(exist_ok=True)

V1=Path("inputs/v1/EVIDENCE_BUNDLE_V1.parquet")
EVENTS=Path("inputs/0e/0E_EVENT_TIME/historical_material_information.parquet")
EVENT_MANIFEST=Path("inputs/0e/manifests/0E_EVENT_TIME.json")

EXPECTED_EVENT_SHA="5342861ec5a532c3a8d40ef194d8515c781b806e66101ec16c7161e579a44c53"

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

for p in [V1,EVENTS,EVENT_MANIFEST]:
    if not p.exists():
        raise FileNotFoundError(p)

manifest0e=json.loads(EVENT_MANIFEST.read_text(encoding="utf-8"))
if manifest0e.get("status")!="PASS":
    raise RuntimeError("0E event artifact manifest is not PASS")
if manifest0e.get("pit_rules",{}).get("future_join_violations")!=0:
    raise RuntimeError("0E event source has future_join violations")
if manifest0e.get("pit_rules",{}).get("formal_oos_eligible") is not True:
    raise RuntimeError("0E event source not formal_oos_eligible")
if sha(EVENTS)!=EXPECTED_EVENT_SHA:
    raise RuntimeError(f"0E historical event hash mismatch: {sha(EVENTS)}")

v1=pd.read_parquet(V1)
if "decision_time_utc" not in v1.columns or "event_published_at_utc" not in v1.columns:
    raise RuntimeError("v1/0F spine missing decision_time_utc or event_published_at_utc")

v1["decision_time_utc"]=pd.to_datetime(v1["decision_time_utc"],utc=True,errors="coerce")
v1["event_published_at_utc"]=pd.to_datetime(v1["event_published_at_utc"],utc=True,errors="coerce")
if v1["decision_time_utc"].isna().any():
    raise RuntimeError("decision_time_utc contains null")
if v1["decision_time_utc"].dt.year.max()>2024:
    raise RuntimeError("sealed/live year exposure in V1")

# Read only pre-2025 event content. 2025 remains sealed for V6.2 development.
cutoff=pd.Timestamp("2025-01-01T00:00:00Z")
events=pd.read_parquet(
    EVENTS,
    filters=[("published_at_utc","<",cutoff.to_pydatetime())],
)
events["published_at_utc"]=pd.to_datetime(events["published_at_utc"],utc=True,errors="coerce")
events["available_at_utc"]=pd.to_datetime(events["available_at_utc"],utc=True,errors="coerce")
events["code"]=events["code"].astype(str).str.strip().str.replace(r"\.0$","",regex=True)
events["title"]=events["title"].astype(str).fillna("").str.strip()

events=events.dropna(subset=["published_at_utc","available_at_utc","code"])
if (events["available_at_utc"] < events["published_at_utc"]).any():
    raise RuntimeError("event available_at precedes published_at")
if events["published_at_utc"].dt.year.max()>2024:
    raise RuntimeError("2025 event content leaked into development scan")

# Multiple disclosures may share the exact same timestamp for one company.
# Collapse them deterministically into a JSON array to preserve all titles
# without multiplying the frozen 0F decision spine.
def pack_titles(s):
    vals=sorted({x for x in s if x})
    return json.dumps(vals,ensure_ascii=False,separators=(",",":"))

agg=(events.groupby(["code","published_at_utc"],as_index=False)
     .agg(
         event_titles_json=("title",pack_titles),
         event_title_count=("title",lambda s: int(len({x for x in s if x}))),
         event_source=("source",lambda s: "|".join(sorted({str(x) for x in s if pd.notna(x)}))),
         event_available_at_utc=("available_at_utc","max"),
     ))

key=v1[["code","event_published_at_utc"]].copy()
key["code"]=key["code"].astype(str).str.strip().str.replace(r"\.0$","",regex=True)
key["_rowid"]=range(len(key))

j=key.merge(
    agg,
    left_on=["code","event_published_at_utc"],
    right_on=["code","published_at_utc"],
    how="left",
    validate="many_to_one",
)

if len(j)!=len(v1):
    raise RuntimeError("event-title join multiplied spine rows")

# Fail closed: any matched raw event must have been public no later than the
# already-frozen 0F selected event timestamp and decision cutoff.
future_raw=((j["event_available_at_utc"].notna()) &
            (j["event_available_at_utc"]>v1["decision_time_utc"].to_numpy())).sum()
if future_raw:
    raise RuntimeError(f"event_future_join_violations={future_raw}")

out=v1.copy()
out["event_titles_json"]=j["event_titles_json"].to_numpy()
out["event_title_count"]=j["event_title_count"].astype("Int64").to_numpy()
out["event_text_source"]=j["event_source"].to_numpy()
out["event_text_available_at_utc"]=j["event_available_at_utc"].to_numpy()

# Rows with no prior event are expected to have no text. If 0F says there is a
# prior event timestamp but the raw-title lookup misses, expose it explicitly.
has_prior=v1["event_published_at_utc"].notna()
matched=out["event_titles_json"].notna()
lookup_miss=int((has_prior & ~matched).sum())
matched_rows=int(matched.sum())

out_path=ROOT/"EVIDENCE_BUNDLE_V2.parquet"
out.to_parquet(out_path,index=False)

audit={
  "layer":"V6.2 Evidence Bundle v2",
  "scope":"2020-2024 development only",
  "status":"PASS" if future_raw==0 and len(out)==len(v1) else "FAIL",
  "parent_v1_sha256":sha(V1),
  "frozen_0e_event_sha256":sha(EVENTS),
  "event_manifest_sha256":sha(EVENT_MANIFEST),
  "join_policy":"Use the exact event_published_at_utc already selected by frozen 0F; lookup raw MOPS title(s) at the same code+timestamp. No independent historical lookback/ranking rule added.",
  "rows":int(len(out)),
  "event_raw_rows_pre2025":int(len(events)),
  "event_timestamp_groups_pre2025":int(len(agg)),
  "rows_with_prior_event_timestamp":int(has_prior.sum()),
  "rows_with_event_text_matched":matched_rows,
  "event_title_lookup_miss":lookup_miss,
  "event_future_join_violations":int(future_raw),
  "duplicate_output_key_violations":int(out.duplicated(["date","code"]).sum()),
  "sealed_2025_event_content_read":False,
  "output_sha256":sha(out_path),
}
if audit["duplicate_output_key_violations"]!=0:
    raise RuntimeError("duplicate output key")
if audit["status"]!="PASS":
    raise RuntimeError(audit)
(ROOT/"EVIDENCE_BUNDLE_V2_AUDIT.json").write_text(
    json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
)
print(json.dumps(audit,ensure_ascii=False),flush=True)
