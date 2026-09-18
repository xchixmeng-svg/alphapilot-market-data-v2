#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='research/V6_PREREG_LOCK.json')
    ns = ap.parse_args()
    mp = ROOT / ns.manifest
    if not mp.exists():
        raise SystemExit(f'LOCK FAIL: missing {ns.manifest}')

    m = json.loads(mp.read_text(encoding='utf-8'))
    if m.get('lock_status') != 'LOCKED':
        raise SystemExit(f"LOCK FAIL: lock_status={m.get('lock_status')!r}, expected LOCKED")

    source = str(m.get('source_freeze_commit') or '').strip()
    if len(source) < 7:
        raise SystemExit('LOCK FAIL: source_freeze_commit missing')

    try:
        subprocess.check_call(['git', 'merge-base', '--is-ancestor', source, 'HEAD'], cwd=ROOT)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f'LOCK FAIL: SOURCE FREEZE {source} is not an ancestor of HEAD') from e

    locked = m.get('locked_files') or {}
    if not locked:
        raise SystemExit('LOCK FAIL: locked_files is empty')

    bad = []
    for rel, expected in locked.items():
        p = ROOT / rel
        if not p.exists():
            bad.append((rel, 'MISSING', expected))
            continue
        got = sha256(p)
        if got != expected:
            bad.append((rel, got, expected))
    if bad:
        raise SystemExit('LOCK FAIL: file hash mismatch: ' + json.dumps(bad, ensure_ascii=False))

    registry = ROOT / str(m.get('hypothesis_registry', 'research/V6_HYPOTHESIS_REGISTRY.json'))
    if not registry.exists():
        raise SystemExit(f'LOCK FAIL: missing hypothesis registry {registry.relative_to(ROOT)}')
    reg = json.loads(registry.read_text(encoding='utf-8'))
    if reg.get('status') != 'LOCKED':
        raise SystemExit(f"LOCK FAIL: hypothesis registry status={reg.get('status')!r}")
    families = reg.get('formal_families') or reg.get('families') or []
    if not families:
        raise SystemExit('LOCK FAIL: hypothesis registry has no formal families')

    expected_runtime = m.get('runtime_versions') or {}
    runtime_bad = []
    for pkg, expected in expected_runtime.items():
        try:
            got = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            got = 'MISSING'
        if got != expected:
            runtime_bad.append((pkg, got, expected))
    if runtime_bad:
        raise SystemExit('LOCK FAIL: runtime mismatch: ' + json.dumps(runtime_bad))

    head = git('rev-parse', 'HEAD')
    print(json.dumps({
        'lock_verified': True,
        'lock_id': m.get('lock_id'),
        'source_freeze_commit': source,
        'execution_commit': head,
        'manifest_sha256': sha256(mp),
        'locked_file_count': len(locked),
        'hypothesis_family_count': len(families),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
