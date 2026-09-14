#!/usr/bin/env python3
"""Run AI causal reranker v1 with code-date persistence counted across prior sessions only."""
from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "scripts" / "backtest_ai_causal_reranker_v1.py"
spec = importlib.util.spec_from_file_location("ai_v1", src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_orig_candidate_events = mod.candidate_events


def candidate_events_fixed(px: pd.DataFrame) -> pd.DataFrame:
    ev = _orig_candidate_events(px)
    dates = sorted(int(x) for x in px.date.unique().tolist())
    didx = {d: i for i, d in enumerate(dates)}
    unique = ev[["date", "code"]].drop_duplicates().sort_values(["date", "code"]).copy()
    p5_map = {}
    p10_map = {}
    history = {}
    for r in unique.itertuples(index=False):
        d = int(r.date); c = str(r.code); i = didx[d]
        hist = history.setdefault(c, [])
        p5_map[(d,c)] = sum(1 for x in hist if i - x < 5)
        p10_map[(d,c)] = sum(1 for x in hist if i - x < 10)
        hist.append(i)
    ev["persist5_prior"] = [p5_map[(int(d), str(c))] for d,c in zip(ev.date, ev.code)]
    ev["persist10_prior"] = [p10_map[(int(d), str(c))] for d,c in zip(ev.date, ev.code)]
    return ev

mod.candidate_events = candidate_events_fixed
mod.main()
