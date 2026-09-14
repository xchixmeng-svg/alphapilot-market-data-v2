#!/usr/bin/env python3
"""Corrected low-base repeated-recommendation research (v2).

Purpose
-------
Test the user's actual idea: a low-base/base-completed stock is first detected by an
execution-independent R10 shadow recommendation list, then the live/common-cash
portfolio is allowed to buy it only on a later shadow recommendation after the required
count is reached. The shadow list keeps running regardless of holdings/cash/slots.

Key correction versus v1
-------------------------
Once a low-base watch starts, the portfolio entry gate remains active on every session
inside the persistence window. A watched stock may enter only on a CURRENT shadow
recommendation whose count has reached the threshold. It can no longer bypass the gate
on a day where shadow_state[(date, code)] was missing.

Locked R10 strategy/exits/execution are otherwise unchanged. This is a research overlay
on the immutable formal stage-3 engine only.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_r10_persistence_gate as base

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "lowbase_shadow_gate_v2_results"
BASELINE_DIR = RUN_ROOT / "baseline"
VARIANTS = [
    {"name": "SHADOW2_LB_P2_5", "min_count": 2, "window": 5, "consecutive": False},
    {"name": "SHADOW2_LB_P2_10", "min_count": 2, "window": 10, "consecutive": False},
    {"name": "SHADOW2_LB_P3_5", "min_count": 3, "window": 5, "consecutive": False},
    {"name": "SHADOW2_LB_P3_10", "min_count": 3, "window": 10, "consecutive": False},
]


def build_runner(locked_source: str, cfg: dict) -> str:
    marker = (
        "# ============================================================\n"
        "# 步驟三：組合層回測引擎\n"
        "# ============================================================\n\n"
    )
    if marker not in locked_source:
        raise RuntimeError("stage-3 marker not found")
    stage3 = locked_source.split(marker, 1)[1]

    old_load = "px = pd.read_pickle('r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)"
    new_load = (
        "px = pd.read_pickle('../baseline/r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)\n"
        "px['research_high120'] = px.groupby('code')['aclose'].transform(lambda s: s.rolling(120, min_periods=120).max())"
    )
    if old_load not in stage3:
        raise RuntimeError("signal-load anchor not found")
    stage3 = stage3.replace(old_load, new_load, 1)

    state_anchor = "cash = INITIAL_CAPITAL\n"
    if state_anchor not in stage3:
        raise RuntimeError("cash-state anchor not found")

    injection = f'''# ---- Research-only corrected shadow recommendation gate v2 ----
PERSISTENCE_MIN_COUNT = {int(cfg['min_count'])}
PERSISTENCE_WINDOW = {int(cfg['window'])}
shadow_gate = {{}}
shadow_rows = []
watch_hist = {{}}
watch_episode = {{}}
episode_seq = 0


def lowbase_completed(r):
    vals = [r.get('aclose', np.nan), r.get('ma20', np.nan), r.get('ma60', np.nan),
            r.get('ma120', np.nan), r.get('research_high120', np.nan)]
    if not all(np.isfinite(v) and v > 0 for v in vals):
        return False
    aclose, ma20, ma60, ma120, high120 = map(float, vals)
    return bool(
        aclose <= 0.90 * high120 and
        aclose >= ma60 and
        ma20 >= ma60 and
        aclose >= 0.98 * ma120 and
        ma60 <= 1.05 * ma120
    )

# Build an execution-independent daily shadow recommendation list.
# The same hard filters, scores and market regime are used, but holdings/cash are ignored.
for _d in all_dates:
    _i = date_idx[_d]
    _sub = by_date.get(_d)
    if _sub is None or len(_sub) == 0:
        continue

    # Expire watches before today's signal processing.
    for _code in list(watch_hist):
        _hist = [x for x in watch_hist[_code] if _i - x < PERSISTENCE_WINDOW]
        if _hist:
            watch_hist[_code] = _hist
        else:
            del watch_hist[_code]
            watch_episode.pop(_code, None)

    _r7_exp = float(_sub['r7_exposure'].iloc[0]) if np.isfinite(_sub['r7_exposure'].iloc[0]) else 0.0
    _r7_slots = int(_sub['r7_slots'].iloc[0]) if np.isfinite(_sub['r7_slots'].iloc[0]) else 0
    _r7 = (_sub[_sub['r7_hard'] == True].sort_values('r7_score', ascending=False).head(_r7_slots)
           if _r7_exp > 0 else _sub.iloc[0:0])
    _r05 = _sub[_sub['r05_hard'] == True].sort_values('r05_score', ascending=False).head(R05_MAX_SLOTS)
    _combined = [('R7', c, r) for c, r in _r7.iterrows()] + [('R05', c, r) for c, r in _r05.iterrows()]
    _combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))

    _seen = set(); _daily = []
    for _strat, _code, _r in _combined:
        _code = str(_code)
        if _code in _seen:
            continue
        _seen.add(_code)
        _daily.append((_strat, _code, _r))
        if len(_daily) >= MAX_POSITIONS:
            break

    _daily_codes = set()
    for _strat, _code, _r in _daily:
        _daily_codes.add(_code)
        _is_lb_now = lowbase_completed(_r)
        if _code not in watch_hist:
            if not _is_lb_now:
                shadow_rows.append({{'date': _d, 'date_index': _i, 'code': _code, 'strategy': _strat,
                                    'episode': 0, 'lowbase_watch': False, 'is_lowbase_now': False,
                                    'current_recommendation': True, 'count': 0, 'accepted': True,
                                    'aclose': float(_r.get('aclose', np.nan))}})
                continue
            episode_seq += 1
            watch_hist[_code] = []
            watch_episode[_code] = episode_seq

        _hist = watch_hist[_code]
        if not _hist or _hist[-1] != _i:
            _hist.append(_i)
        _hist[:] = [x for x in _hist if _i - x < PERSISTENCE_WINDOW]
        _count = len(_hist)
        shadow_gate[(_d, _code)] = {{'watch_active': True, 'current_recommendation': True,
                                     'count': _count, 'episode': watch_episode[_code]}}
        shadow_rows.append({{'date': _d, 'date_index': _i, 'code': _code, 'strategy': _strat,
                            'episode': watch_episode[_code], 'lowbase_watch': True,
                            'is_lowbase_now': bool(_is_lb_now), 'current_recommendation': True,
                            'count': _count, 'accepted': bool(_count >= PERSISTENCE_MIN_COUNT),
                            'aclose': float(_r.get('aclose', np.nan))}})

    # Critical correction: watched names remain gated even on sessions where they are
    # not in today's shadow recommendation list. This prevents the v1 bypass.
    for _code, _hist in list(watch_hist.items()):
        if _code in _daily_codes:
            continue
        shadow_gate[(_d, _code)] = {{'watch_active': True, 'current_recommendation': False,
                                     'count': len(_hist), 'episode': watch_episode[_code]}}

cash = INITIAL_CAPITAL
'''
    stage3 = stage3.replace(state_anchor, injection, 1)

    gate_anchor = (
        "                if post_total > TOTAL_CAP + 1e-12 or post_code > SINGLE_CAP + 1e-12: continue\n"
        "                o = submit('BUY', di, exdate, code, strat, shares, 'ENTRY', limit,\n"
    )
    gate_replacement = (
        "                if post_total > TOTAL_CAP + 1e-12 or post_code > SINGLE_CAP + 1e-12: continue\n"
        "                _sg = shadow_gate.get((di, str(code)))\n"
        "                if _sg is not None and bool(_sg.get('watch_active', False)):\n"
        "                    _confirmed_now = bool(_sg.get('current_recommendation', False)) and int(_sg.get('count', 0)) >= PERSISTENCE_MIN_COUNT\n"
        "                    if not _confirmed_now:\n"
        "                        # The watched low-base name consumes its original rank/slot opportunity; no backfill.\n"
        "                        used.add(code); slots_free -= 1\n"
        "                        continue\n"
        "                o = submit('BUY', di, exdate, code, strat, shares, 'ENTRY', limit,\n"
    )
    if gate_anchor not in stage3:
        raise RuntimeError("BUY submit anchor not found")
    stage3 = stage3.replace(gate_anchor, gate_replacement, 1)

    dump_anchor = "Path('contract_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')\n"
    if dump_anchor not in stage3:
        raise RuntimeError("output anchor not found")
    stage3 = stage3.replace(
        dump_anchor,
        dump_anchor + "pd.DataFrame(shadow_rows).to_csv('shadow_recommendations.csv', index=False)\n",
        1,
    )

    header = (
        "# Generated corrected shadow-persistence gate v2 from immutable R10 stage-3.\n"
        "import pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n"
    )
    return header + stage3


def occurrence_quality(sh: pd.DataFrame, causal: pd.DataFrame, window: int) -> pd.DataFrame:
    """Paired descriptive outcome table for 1st/2nd/3rd shadow recommendation.

    This is diagnostic only; future returns are never used by the strategy.
    Each watch episode contributes at most one observation at each occurrence number.
    """
    watched = sh[sh['lowbase_watch'].astype(bool) & sh['current_recommendation'].astype(bool)].copy()
    if watched.empty:
        return pd.DataFrame()
    watched['date'] = watched['date'].astype(int)
    watched['code'] = watched['code'].astype(str).str.zfill(4)
    watched['episode'] = watched['episode'].astype(int)
    watched = watched.sort_values(['episode', 'date'])
    watched = watched[watched['count'].isin([1, 2, 3])].drop_duplicates(['episode', 'count'], keep='first')

    px = causal[['date', 'code', 'aclose']].copy().sort_values(['code', 'date'])
    px['code'] = px['code'].astype(str).str.zfill(4)
    horizons = [5, 10, 20, 40, 60]
    for h in horizons:
        px[f'fwd_{h}'] = px.groupby('code')['aclose'].shift(-h) / px['aclose'] - 1.0
    out = watched.merge(px[['date', 'code'] + [f'fwd_{h}' for h in horizons]], on=['date', 'code'], how='left')
    out['window_sessions'] = int(window)
    return out


def summarize_quality(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for n, g in events.groupby('count'):
        row = {'occurrence': int(n), 'n_events': int(len(g)), 'n_unique_episodes': int(g['episode'].nunique())}
        for h in [5, 10, 20, 40, 60]:
            s = g[f'fwd_{h}'].dropna()
            row[f'fwd{h}_mean'] = float(s.mean()) if len(s) else np.nan
            row[f'fwd{h}_median'] = float(s.median()) if len(s) else np.nan
            row[f'fwd{h}_win_rate'] = float((s > 0).mean()) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    actual = base.sha256(base.LOCKED_ENGINE)
    if actual != base.LOCKED_ENGINE_SHA256:
        raise RuntimeError(f"locked engine SHA mismatch: {actual}")

    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True)
    base.RUN_ROOT = RUN_ROOT
    base.BASELINE_DIR = BASELINE_DIR

    base.link_inputs(BASELINE_DIR)
    baseline_script = BASELINE_DIR / "r10_max_formal_locked.py"
    baseline_script.write_text(base.LOCKED_ENGINE.read_text(encoding="utf-8"), encoding="utf-8")
    base.run_python(baseline_script, BASELINE_DIR, "execution.log")
    base.verify_baseline()

    locked_source = base.LOCKED_ENGINE.read_text(encoding="utf-8")
    for cfg in VARIANTS:
        rd = RUN_ROOT / cfg['name']
        rd.mkdir(parents=True)
        runner = rd / 'runner.py'
        runner.write_text(build_runner(locked_source, cfg), encoding='utf-8')
        base.run_python(runner, rd, 'execution.log')

    causal = pd.read_csv(
        BASELINE_DIR / 'ohlcv_causal_2020_2025.csv.gz',
        dtype={'code': str},
        usecols=lambda c: c in {'date', 'code', 'close', 'aclose', 'alow', 'ahigh'},
        low_memory=False,
    )
    causal['code'] = causal['code'].astype(str).str.zfill(4)
    causal['date'] = causal['date'].astype(int)

    rows = [base.collect_metrics('BASELINE', BASELINE_DIR, None, causal)]
    quality_outputs = []
    for cfg in VARIANTS:
        rd = RUN_ROOT / cfg['name']
        row = base.collect_metrics(cfg['name'], rd, cfg, causal)
        sh = pd.read_csv(rd / 'shadow_recommendations.csv', dtype={'code': str})
        watched = sh[sh['lowbase_watch'].astype(bool) & sh['current_recommendation'].astype(bool)]
        row['shadow_lowbase_recommendations'] = int(len(watched))
        row['shadow_lowbase_unique_codes'] = int(watched.code.nunique()) if len(watched) else 0
        row['shadow_unique_episodes'] = int(watched.episode.nunique()) if len(watched) else 0
        row['episodes_reach_2'] = int(watched.loc[watched['count'] >= 2, 'episode'].nunique()) if len(watched) else 0
        row['episodes_reach_3'] = int(watched.loc[watched['count'] >= 3, 'episode'].nunique()) if len(watched) else 0
        rows.append(row)

        ev = occurrence_quality(sh, causal, cfg['window'])
        if not ev.empty:
            ev['variant'] = cfg['name']
            quality_outputs.append(ev)

    comp = pd.DataFrame(rows)
    b = comp.iloc[0]
    comp['cagr_delta_pp_vs_baseline'] = (comp.cagr - b.cagr) * 100
    comp['max_dd_improvement_pp_vs_baseline'] = (comp.max_drawdown - b.max_drawdown) * 100
    comp['pf_delta_vs_baseline'] = comp.pnl_profit_factor - b.pnl_profit_factor
    comp['win_rate_delta_pp_vs_baseline'] = (comp.win_rate - b.win_rate) * 100
    comp.to_csv(RUN_ROOT / 'LOWBASE_SHADOW_GATE_V2_COMPARISON.csv', index=False)
    (RUN_ROOT / 'LOWBASE_SHADOW_GATE_V2_COMPARISON.json').write_text(
        json.dumps(comp.replace({np.nan: None}).to_dict(orient='records'), ensure_ascii=False, indent=2),
        encoding='utf-8')

    if quality_outputs:
        events = pd.concat(quality_outputs, ignore_index=True)
        events.to_csv(RUN_ROOT / 'LOWBASE_SHADOW_OCCURRENCE_EVENTS.csv', index=False)
        # Since event sets are identical for P2/P3 at a given window, summarize by window once.
        qrows = []
        for w in sorted(events.window_sessions.unique()):
            one = events[(events.window_sessions == w) & events.variant.str.contains(f'_{int(w)}$')].copy()
            # Deduplicate same event repeated by P2/P3 variants.
            one = one.drop_duplicates(['window_sessions', 'episode', 'count', 'date', 'code'])
            qs = summarize_quality(one)
            qs['window_sessions'] = int(w)
            qrows.append(qs)
        qsum = pd.concat(qrows, ignore_index=True)
        qsum.to_csv(RUN_ROOT / 'LOWBASE_SHADOW_OCCURRENCE_QUALITY.csv', index=False)
    else:
        qsum = pd.DataFrame()

    print('=== CORRECTED LOW-BASE SHADOW GATE V2 ===')
    show_cols = ['variant','end_nav','cagr','max_drawdown','pnl_profit_factor','win_rate','completed_trades',
                 'shadow_lowbase_recommendations','shadow_unique_episodes','episodes_reach_2','episodes_reach_3',
                 'cagr_delta_pp_vs_baseline','max_dd_improvement_pp_vs_baseline']
    print(comp.reindex(columns=show_cols).to_string(index=False))
    print('\n=== OCCURRENCE QUALITY (diagnostic; not used by strategy) ===')
    if len(qsum):
        print(qsum.to_string(index=False))


if __name__ == '__main__':
    main()
