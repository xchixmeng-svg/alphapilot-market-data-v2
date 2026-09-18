#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"research"/"V6_1_PREREG_LOCK.json"

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    if not LOCK.exists(): raise SystemExit("V6.1 LOCK FAIL: lock file missing")
    lock=json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("lock_status")!="LOCKED": raise SystemExit("V6.1 LOCK FAIL: lock_status")
    if lock.get("2025_opened") is not False or lock.get("2026_opened") is not False:
        raise SystemExit("V6.1 LOCK FAIL: sealed years opened")
    src=str(lock.get("source_freeze_commit") or "")
    if len(src)<7: raise SystemExit("V6.1 LOCK FAIL: source_freeze_commit")
    r=subprocess.run(["git","merge-base","--is-ancestor",src,"HEAD"])
    if r.returncode!=0: raise SystemExit("V6.1 LOCK FAIL: source freeze is not ancestor of HEAD")
    bad=[]
    files=lock.get("locked_files") or {}
    if not files: raise SystemExit("V6.1 LOCK FAIL: locked_files empty")
    for rel,expected in files.items():
        p=ROOT/rel
        got=sha(p) if p.exists() else None
        if got!=expected: bad.append([rel,expected,got])
    if bad: raise SystemExit("V6.1 LOCK FAIL: file hash mismatch: "+json.dumps(bad))
    runtime=lock.get("runtime_versions") or {}
    expected={"pandas":"2.2.3","numpy":"2.3.5","scipy":"1.18.1","scikit-learn":"1.5.2","pyarrow":"18.1.0"}
    if runtime!=expected: raise SystemExit("V6.1 LOCK FAIL: runtime versions")
    print(json.dumps({"lock_verified":True,"lock_id":lock.get("lock_id"),"source_freeze_commit":src,
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
      "lock_sha256":sha(LOCK),"locked_file_count":len(files)},ensure_ascii=False))

if __name__=="__main__": main()
