#!/usr/bin/env python3
"""Plan every missing R10 paper session, never leap over an unbooked day."""
import argparse
import json
import os
import re
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent.parent
SOURCE_KEYS = {f"{market}_{kind}" for market in ("twse", "tpex")
               for kind in ("ohlcv", "institutional")}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def expected_session():
    response = requests.get("https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule", timeout=30)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("DATA_GATE_FAIL TWSE holiday calendar unavailable")
    closed = {date(int(x["Date"][:3]) + 1911, int(x["Date"][3:5]), int(x["Date"][5:]))
              for x in rows if "放假" in x.get("Description", "") or "無交易" in x.get("Name", "")}
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    day = now.date() if now.hour >= 18 else now.date() - timedelta(days=1)
    while day.weekday() >= 5 or day in closed:
        day -= timedelta(days=1)
    return day.isoformat()


def plan(target, require_current=False):
    if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", target):
        raise RuntimeError(f"PAPER_GATE_FAIL invalid target date: {target!r}")
    if require_current and target != expected_session():
        raise RuntimeError(f"DATA_GATE_FAIL latest main snapshot {target} is stale; expected {expected_session()}")
    states = sorted((ROOT / "forward_runtime").glob("????-??-??/paper_state.json"))
    if not states:
        raise RuntimeError("PAPER_GATE_FAIL no prior paper seed; explicit reconciliation required")
    last = states[-1].parent.name
    state = json.loads(states[-1].read_text(encoding="utf-8"))
    if state.get("trade_date") != last or state.get("account") != "PAPER_SIMULATION_NOT_BROKER":
        raise RuntimeError("PAPER_GATE_FAIL prior paper state invalid")
    if target < last:
        raise RuntimeError(f"PAPER_GATE_FAIL refusing historical rewrite: state {last}, target {target}")
    if target == last:
        return []
    paths = git("ls-tree", "-r", "--name-only", "origin/main", "data").splitlines()
    dates = sorted({m.group(1) for p in paths if (m := re.fullmatch(r"data/(20\d{2}-\d{2}-\d{2})/manifest\.json", p))
                    and last < m.group(1) <= target})
    if not dates or dates[-1] != target:
        raise RuntimeError(f"DATA_GATE_FAIL no latest manifest {target}")
    for day in dates:
        m = json.loads(git("show", f"origin/main:data/{day}/manifest.json"))
        if m.get("status") != "PASS" or set(m.get("source_dates", {})) != SOURCE_KEYS or any(v != day for v in m["source_dates"].values()):
            raise RuntimeError(f"DATA_GATE_FAIL four-source manifest {day}")
    return dates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--require-current", action="store_true")
    args = parser.parse_args()
    dates = plan(args.target, args.require_current)
    print(json.dumps({"target": args.target, "pending_dates": dates}, ensure_ascii=False))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"dates={','.join(dates)}\n")
