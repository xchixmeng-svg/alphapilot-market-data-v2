#!/usr/bin/env python3
"""Causal crash/late-entry overlays for the immutable R10 MAX formal engine.

Every feature is computed from data available at T close. Entry vetoes apply to
orders submitted after that close. Holding alarms submit a sell for T+1. The
locked formal source is copied and instrumented at runtime; it is never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FORMAL_SHA = "2fef3ba99b7c83b5db21e29c5f1c2abd8b3df77583840a025e6df937ff81c0d0"
POLICIES = {
    "BASE": {"group": "baseline", "label": "locked baseline"},
    "E_MARKET_SHOCK": {"group": "entry", "label": "broad-market deterioration veto"},
    "E_EXHAUSTION": {"group": "entry", "label": "late-stage exhaustion veto"},
    "E_DISTRIBUTION": {"group": "entry", "label": "price-volume/institutional distribution veto"},
    "H_FAST_FAIL": {"group": "holding", "label": "first-three-session failed-breakout exit"},
    "H_STRESS_FAIL": {"group": "holding", "label": "failed position plus market stress exit"},
    "C_SELECTIVE": {"group": "combined", "label": "selective entry veto plus early failure exit"},
}

OFFICIAL_EVENT_SUPPLEMENTS = [{
    "date": 20250721, "code": "6919", "market": "TWSE",
    "event_type": "SPLIT:PAR_VALUE_5_TO_0.5", "official_prev_close": 1215.0,
    "reference_price": 121.5, "cash_dividend_per_share": 0.0,
    "stock_shares_per_1000": 9000.0, "continuity_bridge": 10.0,
    "source": "MOPS_20250529_6919_PAR_VALUE_CHANGE",
}]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inject(formal: str, policy: str) -> str:
    group = POLICIES[policy]["group"]
    # Features deliberately use fixed, interpretable thresholds. No outcome data
    # or future bars are referenced. The full matrix is always reported.
    feature_hook = r'''

# ---- isolated causal crash-risk research hook ----
_cg = px.groupby('code', group_keys=False)
px['risk_ret5'] = _cg['aclose'].transform(lambda s: s.pct_change(5))
px['risk_ret20'] = _cg['aclose'].transform(lambda s: s.pct_change(20))
px['risk_day_ret'] = _cg['aclose'].transform(lambda s: s.pct_change())
px['risk_vol_ratio'] = px['volume'] / px['vol20'].replace(0, np.nan)
_range = (px['high'] - px['low']).replace(0, np.nan)
px['risk_clv'] = ((2 * px['close'] - px['high'] - px['low']) / _range).fillna(0).clip(-1, 1)
_eligible = px['amt20'].ge(30_000_000) & px['is_valid_universe']
_adv = (px['risk_day_ret'] > 0).where(_eligible).groupby(px['date']).mean()
_limitdown = (px['risk_day_ret'] <= -0.085).where(_eligible).groupby(px['date']).mean()
_breadth = (px['aclose'] > px['ma60']).where(_eligible).groupby(px['date']).mean()
_market = px.loc[px.code.eq('0050'), ['date','aclose']].drop_duplicates('date').sort_values('date')
_market['risk_mkt_ret3'] = _market['aclose'].pct_change(3)
_market['risk_mkt_ret5'] = _market['aclose'].pct_change(5)
_daily = pd.DataFrame({'date': _adv.index, 'risk_adv1': _adv.values,
                       'risk_limitdown_share': _limitdown.reindex(_adv.index).values,
                       'risk_breadth': _breadth.reindex(_adv.index).values})
_daily = _daily.sort_values('date')
_daily['risk_breadth_drop5'] = _daily['risk_breadth'].diff(5)
_daily = _daily.merge(_market[['date','risk_mkt_ret3','risk_mkt_ret5']], on='date', how='left')
px = px.merge(_daily, on='date', how='left')
px['risk_inst_sell'] = ((px['Foreign3D'] < 0) & (px['Trust5D'] <= 0))
px['risk_market_shock'] = ((px['risk_mkt_ret3'] <= -0.035) |
                           ((px['risk_breadth_drop5'] <= -0.15) & (px['risk_adv1'] <= 0.38)) |
                           (px['risk_limitdown_share'] >= 0.025))
px['risk_exhaustion'] = ((px['risk_ret20'] >= 0.35) & (px['ma20gap'] >= 0.22) &
                         (px['risk_vol_ratio'] >= 1.6) & (px['risk_clv'] <= 0.15))
px['risk_distribution'] = ((px['risk_ret20'] >= 0.15) & (px['risk_vol_ratio'] >= 1.5) &
                            (px['risk_clv'] <= -0.35) & px['risk_inst_sell'])
'''
    marker = "\n# ---------- 0050大盤基準：修正用還原後的aclose ----------"
    if marker not in formal:
        raise RuntimeError("feature marker changed")
    formal = formal.replace(marker, feature_hook + marker, 1)
    formal = formal.replace(
        "'cash_dividend_per_share', 'share_factor', 'event_type', 'source', 'reference_price']",
        "'cash_dividend_per_share', 'share_factor', 'event_type', 'source', 'reference_price',\n"
        "             'risk_day_ret', 'risk_ret5', 'risk_ret20', 'risk_vol_ratio', 'risk_clv',\n"
        "             'risk_mkt_ret3', 'risk_mkt_ret5', 'risk_adv1', 'risk_limitdown_share',\n"
        "             'risk_breadth', 'risk_breadth_drop5', 'risk_inst_sell',\n"
        "             'risk_market_shock', 'risk_exhaustion', 'risk_distribution']",
        1,
    )
    formal = formal.replace(
        "nav_rows, trade_rows, order_rows, corp_rows, slot_diag = [], [], [], [], []",
        "nav_rows, trade_rows, order_rows, corp_rows, slot_diag = [], [], [], [], []\n"
        "risk_entry_rows, risk_exit_rows = [], []",
        1,
    )
    formal = formal.replace(
        "'hold_days': 0, 'runner': None, 'pending_sell': False}",
        "'hold_days': 0, 'runner': None, 'pending_sell': False,\n"
        "                           'risk_min_ret': 0.0}",
        1,
    )
    formal = formal.replace(
        "p['hold_days'] += 1",
        "p['hold_days'] += 1\n"
        "        if sub is not None and code in sub.index:\n"
        "            p['risk_min_ret'] = min(p.get('risk_min_ret', 0.0),\n"
        "                                    float(sub.loc[code]['aclose']) / p['entry_index'] - 1.0)",
        1,
    )

    hold_marker = "    for code, p in list(positions.items()):\n        if p['pending_sell'] or p['hold_days'] < MIN_HOLD_DAYS or sub is None or code not in sub.index:\n            continue"
    hold_repl = f'''    for code, p in list(positions.items()):
        if p['pending_sell'] or sub is None or code not in sub.index:
            continue
        _rr = sub.loc[code]
        _rret = float(_rr['aclose']) / p['entry_index'] - 1.0
        _fast_fail = (p['hold_days'] <= 3 and _rret <= -0.06 and
                      float(_rr.get('risk_day_ret', 0.0)) <= -0.025 and
                      (bool(_rr.get('risk_market_shock', False)) or
                       (float(_rr.get('risk_vol_ratio', 0.0)) >= 1.35 and
                        float(_rr.get('risk_clv', 0.0)) <= -0.30)))
        _stress_fail = (_rret <= -0.075 and bool(_rr.get('risk_market_shock', False)))
        _do_risk_exit = (({policy!r} in ('H_FAST_FAIL','C_SELECTIVE') and _fast_fail) or
                         ({policy!r} == 'H_STRESS_FAIL' and _stress_fail))
        if _do_risk_exit and exdate is not None:
            o = submit('SELL', di, exdate, code, p['strategy'], p['shares'], 'CRASH_RISK_EXIT')
            pending_sells.setdefault(exdate, []).append(o)
            p['pending_sell'] = True
            risk_exit_rows.append({{'signal_date': di, 'scheduled_date': exdate, 'code': code,
                                   'strategy': p['strategy'], 'hold_days': p['hold_days'],
                                   'return_at_signal': _rret, 'market_shock': bool(_rr.get('risk_market_shock', False))}})
            continue
        if p['hold_days'] < MIN_HOLD_DAYS:
            continue'''
    if hold_marker not in formal:
        raise RuntimeError("holding marker changed")
    formal = formal.replace(hold_marker, hold_repl, 1)

    entry_marker = """            for strat, code, r in combined:
                if slots_free <= 0: break
                if code in used: continue"""
    entry_repl = f'''            for strat, code, r in combined:
                if slots_free <= 0: break
                if code in used: continue
                _veto = (({policy!r} == 'E_MARKET_SHOCK' and bool(r.get('risk_market_shock', False))) or
                         ({policy!r} == 'E_EXHAUSTION' and bool(r.get('risk_exhaustion', False))) or
                         ({policy!r} == 'E_DISTRIBUTION' and bool(r.get('risk_distribution', False))) or
                         ({policy!r} == 'C_SELECTIVE' and
                          (bool(r.get('risk_exhaustion', False)) or bool(r.get('risk_distribution', False))) and
                          bool(r.get('risk_market_shock', False))))
                if _veto:
                    risk_entry_rows.append({{'signal_date': di, 'scheduled_date': exdate, 'code': code,
                                            'strategy': strat, 'market_shock': bool(r.get('risk_market_shock', False)),
                                            'exhaustion': bool(r.get('risk_exhaustion', False)),
                                            'distribution': bool(r.get('risk_distribution', False))}})
                    continue'''
    if entry_marker not in formal:
        raise RuntimeError("entry marker changed")
    formal = formal.replace(entry_marker, entry_repl, 1)
    formal = formal.replace(
        "pd.DataFrame(slot_diag).to_csv('r10max_formal_slot_diag.csv', index=False)",
        "pd.DataFrame(slot_diag).to_csv('r10max_formal_slot_diag.csv', index=False)\n"
        "pd.DataFrame(risk_entry_rows).to_csv('crash_risk_entry_vetoes.csv', index=False)\n"
        "pd.DataFrame(risk_exit_rows).to_csv('crash_risk_exits.csv', index=False)",
        1,
    )
    if group in ("holding", "combined"):
        audit_marker = "'minimum_hold_three_sessions': bool((trades_df.hold_days >= 3).all()) if len(trades_df) else True,"
        audit_repl = "'minimum_hold_three_sessions': bool((trades_df.loc[trades_df.reason != 'CRASH_RISK_EXIT', 'hold_days'] >= 3).all()) if len(trades_df) else True,"
        if audit_marker not in formal:
            raise RuntimeError("audit marker changed")
        formal = formal.replace(audit_marker, audit_repl, 1)
    return formal


def summarize(run_dir: Path, policy: str) -> dict:
    nav = pd.read_csv(run_dir / "r10max_formal_nav.csv")
    trades = pd.read_csv(run_dir / "r10max_formal_trades.csv", dtype={"code": str})
    ret = trades["return"]
    end_nav = float(nav.nav.iloc[-1])
    years = (pd.to_datetime(str(int(nav.date.iloc[-1]))) - pd.to_datetime(str(int(nav.date.iloc[0])))).days / 365.25
    wins = trades.loc[trades.pnl > 0, "pnl"].sum()
    losses = -trades.loc[trades.pnl < 0, "pnl"].sum()
    veto_file, exit_file = run_dir / "crash_risk_entry_vetoes.csv", run_dir / "crash_risk_exits.csv"
    return {
        "policy": policy, **POLICIES[policy], "initial_nav": 1_300_000.0, "end_nav": end_nav,
        "total_return": end_nav / 1_300_000.0 - 1,
        "cagr": (end_nav / 1_300_000.0) ** (1 / years) - 1,
        "max_drawdown": float(nav.drawdown.min()),
        "profit_factor": float(wins / losses) if losses else None,
        "win_rate": float((trades.pnl > 0).mean()), "trades": int(len(trades)),
        "loss_le_12": int((ret <= -.12).sum()), "loss_le_15": int((ret <= -.15).sum()),
        "loss_le_20": int((ret <= -.20).sum()), "loss_le_25": int((ret <= -.25).sum()),
        "max_single_loss": float(ret.min()),
        "risk_exit_trades": int((trades.reason == "CRASH_RISK_EXIT").sum()),
        "entry_vetoes": max(0, sum(1 for _ in veto_file.open()) - 1) if veto_file.exists() and veto_file.stat().st_size else 0,
        "exit_signals": max(0, sum(1 for _ in exit_file.open()) - 1) if exit_file.exists() and exit_file.stat().st_size else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=sorted(POLICIES))
    ap.add_argument("--formal", default="scripts/r10_max_formal.py", type=Path)
    ap.add_argument("--history", default="data/history/2020-2025", type=Path)
    ap.add_argument("--events", default="data/reference/official_corporate_actions_2020_2025.csv", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    assert sha256(args.formal) == FORMAL_SHA
    run_dir = args.output.resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    for path in args.history.glob("*.parquet"):
        os.symlink(path.resolve(), run_dir / path.name)
    events = pd.read_csv(args.events, dtype={"code": str})
    events["code"] = events["code"].str.zfill(4)
    events = pd.concat([events, pd.DataFrame(OFFICIAL_EVENT_SUPPLEMENTS)], ignore_index=True)
    events.sort_values(["date", "code", "source"]).drop_duplicates(["date", "code"], keep="last").to_csv(
        run_dir / args.events.name, index=False)
    variant = inject(args.formal.read_text(encoding="utf-8"), args.policy)
    (run_dir / "derived_crash_overlay_engine.py").write_text(variant, encoding="utf-8")
    log_path = run_dir / "execution.log"
    with log_path.open("w", encoding="utf-8") as log:
        done = subprocess.run([sys.executable, "derived_crash_overlay_engine.py"], cwd=run_dir,
                              stdout=log, stderr=subprocess.STDOUT)
    if done.returncode:
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-120:]), file=sys.stderr)
        raise subprocess.CalledProcessError(done.returncode, done.args)
    result = summarize(run_dir, args.policy)
    (run_dir / "crash_overlay_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.policy == "BASE":
        assert abs(result["end_nav"] - 2403427.0678222505) < 1e-6, result
        assert result["trades"] == 150, result


if __name__ == "__main__":
    main()
