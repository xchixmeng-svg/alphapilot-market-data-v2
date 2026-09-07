#!/usr/bin/env python3
"""Run one isolated causal margin-risk overlay on the locked formal R10 engine.

The formal source is never edited.  This runner injects two narrow hooks into a
temporary copy: a pre-entry block or a holding-period T-close warning whose sell
is submitted for T+1.  All execution/accounting remains in the locked engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


FORMAL_SHA = "2fef3ba99b7c83b5db21e29c5f1c2abd8b3df77583840a025e6df937ff81c0d0"
POLICIES = {
    "BASE": {"group": "baseline", "rule": "none"},
    "E1_M5_10_P5NEG": {"group": "entry", "rule": "m5>=10%; p5<=0%"},
    "E2_M10_20_P10NEG": {"group": "entry", "rule": "m10>=20%; p10<=0%"},
    "E3_USE50_M5_05_P5NEG": {"group": "entry", "rule": "usage>=50%; m5>=5%; p5<=0%"},
    "E4_M20_30_P20LE5": {"group": "entry", "rule": "m20>=30%; p20<=5%"},
    "E5_M5_10_S5_20_P5NEG": {"group": "entry", "rule": "m5>=10%; short5>=20%; p5<=0%"},
    "H1_M5_10_P5NEG": {"group": "holding", "rule": "m5>=10%; p5<=0%"},
    "H2_M10_20_P10NEG": {"group": "holding", "rule": "m10>=20%; p10<=0%"},
    "H3_USE50_M5_05_P5NEG": {"group": "holding", "rule": "usage>=50%; m5>=5%; p5<=0%"},
    "H4_M20_30_P20LE5": {"group": "holding", "rule": "m20>=30%; p20<=5%"},
    "H5_M5_10_S5_20_P5NEG": {"group": "holding", "rule": "m5>=10%; short5>=20%; p5<=0%"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition_expr(policy: str) -> str:
    key = policy[1:] if policy != "BASE" else ""
    return {
        "1_M5_10_P5NEG": "(px['margin_chg5'] >= .10) & (px['price_ret5'] <= 0)",
        "2_M10_20_P10NEG": "(px['margin_chg10'] >= .20) & (px['price_ret10'] <= 0)",
        "3_USE50_M5_05_P5NEG": "(px['margin_usage_calc'] >= 50) & (px['margin_chg5'] >= .05) & (px['price_ret5'] <= 0)",
        "4_M20_30_P20LE5": "(px['margin_chg20'] >= .30) & (px['price_ret20'] <= .05)",
        "5_M5_10_S5_20_P5NEG": "(px['margin_chg5'] >= .10) & (px['short_chg5'] >= .20) & (px['price_ret5'] <= 0)",
    }[key]


def inject(formal: str, policy: str, margin_path: Path) -> str:
    group = POLICIES[policy]["group"]
    expr = "False" if policy == "BASE" else condition_expr(policy)
    feature_hook = f'''

# ---- isolated causal margin-risk research hook ----
_m = pd.read_csv({str(margin_path)!r}, dtype={{'code': str}})
_m['code'] = _m['code'].str.zfill(4)
_m['date'] = pd.to_datetime(_m['date']).dt.strftime('%Y%m%d').astype(int)
_m = _m.sort_values(['code', 'date']).drop_duplicates(['date', 'code'], keep='last')
for _c in ('margin_balance','margin_limit','margin_usage_pct','short_balance'):
    _m[_c] = pd.to_numeric(_m[_c], errors='coerce')
_mg = _m.groupby('code', group_keys=False)
for _w in (5, 10, 20):
    _prev = _mg['margin_balance'].shift(_w)
    _m[f'margin_chg{{_w}}'] = (_m['margin_balance'] - _prev) / _prev.abs().clip(lower=1)
_sprev = _mg['short_balance'].shift(5)
_m['short_chg5'] = (_m['short_balance'] - _sprev) / _sprev.abs().clip(lower=1)
_m['margin_usage_calc'] = _m['margin_usage_pct'].where(
    _m['margin_usage_pct'].notna(), 100 * _m['margin_balance'] / _m['margin_limit'].replace(0, np.nan))
_cols = ['date','code','margin_chg5','margin_chg10','margin_chg20','short_chg5','margin_usage_calc']
px = px.merge(_m[_cols], on=['date','code'], how='left')
_pg = px.groupby('code', group_keys=False)
px['price_ret5'] = _pg['aclose'].transform(lambda s: s.pct_change(5))
px['price_ret10'] = _pg['aclose'].transform(lambda s: s.pct_change(10))
px['price_ret20'] = _pg['aclose'].transform(lambda s: s.pct_change(20))
_risk = ({expr}).fillna(False) if hasattr(({expr}), 'fillna') else False
px['margin_entry_block'] = _risk if {group!r} == 'entry' else False
px['margin_hold_exit'] = _risk if {group!r} == 'holding' else False
'''
    marker = "\n# ---------- 0050大盤基準：修正用還原後的aclose ----------"
    formal = formal.replace(marker, feature_hook + marker, 1)
    formal = formal.replace(
        "'cash_dividend_per_share', 'share_factor', 'event_type', 'source', 'reference_price']",
        "'cash_dividend_per_share', 'share_factor', 'event_type', 'source', 'reference_price',\n"
        "             'margin_entry_block', 'margin_hold_exit', 'margin_chg5', 'margin_chg10',\n"
        "             'margin_chg20', 'short_chg5', 'margin_usage_calc', 'price_ret5',\n"
        "             'price_ret10', 'price_ret20']",
        1,
    )
    formal = formal.replace(
        "nav_rows, trade_rows, order_rows, corp_rows, slot_diag = [], [], [], [], []",
        "nav_rows, trade_rows, order_rows, corp_rows, slot_diag = [], [], [], [], []\nentry_block_rows = []",
        1,
    )
    hold_marker = "    for code, p in list(positions.items()):\n        if p['pending_sell'] or p['hold_days'] < MIN_HOLD_DAYS or sub is None or code not in sub.index:\n            continue"
    hold_repl = """    for code, p in list(positions.items()):
        if p['pending_sell'] or sub is None or code not in sub.index:
            continue
        # Overlay warnings are allowed from the first holding close; execution is still exact T+1.
        if bool(sub.loc[code].get('margin_hold_exit', False)) and exdate is not None:
            o = submit('SELL', di, exdate, code, p['strategy'], p['shares'], 'MARGIN_EXIT')
            pending_sells.setdefault(exdate, []).append(o)
            p['pending_sell'] = True
            continue
        if p['hold_days'] < MIN_HOLD_DAYS:
            continue"""
    if hold_marker not in formal:
        raise RuntimeError("formal holding hook marker changed")
    formal = formal.replace(hold_marker, hold_repl, 1)
    entry_marker = """            for strat, code, r in combined:
                if slots_free <= 0: break
                if code in used: continue"""
    entry_repl = """            for strat, code, r in combined:
                if slots_free <= 0: break
                if code in used: continue
                if bool(r.get('margin_entry_block', False)):
                    entry_block_rows.append({'signal_date': di, 'scheduled_date': exdate,
                                             'code': code, 'strategy': strat})
                    continue"""
    if entry_marker not in formal:
        raise RuntimeError("formal entry hook marker changed")
    formal = formal.replace(entry_marker, entry_repl, 1)
    formal = formal.replace(
        "pd.DataFrame(slot_diag).to_csv('r10max_formal_slot_diag.csv', index=False)",
        "pd.DataFrame(slot_diag).to_csv('r10max_formal_slot_diag.csv', index=False)\n"
        "pd.DataFrame(entry_block_rows, columns=['signal_date','scheduled_date','code','strategy']).to_csv('margin_entry_blocks.csv', index=False)",
        1,
    )
    if group == "holding":
        # Preserve the locked three-session minimum for every formal exit.
        # MARGIN_EXIT is the separately defined overlay intervention: a T-close
        # warning submitted for T+1, including during the first three sessions.
        audit_marker = "'minimum_hold_three_sessions': bool((trades_df.hold_days >= 3).all()) if len(trades_df) else True,"
        audit_repl = "'minimum_hold_three_sessions': bool((trades_df.loc[trades_df.reason != 'MARGIN_EXIT', 'hold_days'] >= 3).all()) if len(trades_df) else True,"
        if audit_marker not in formal:
            raise RuntimeError("formal minimum-hold audit marker changed")
        formal = formal.replace(audit_marker, audit_repl, 1)
    return formal


def summarize(run_dir: Path, policy: str, manifest: dict) -> dict:
    nav = pd.read_csv(run_dir / "r10max_formal_nav.csv")
    trades = pd.read_csv(run_dir / "r10max_formal_trades.csv", dtype={"code": str})
    end_nav = float(nav.nav.iloc[-1])
    years = (pd.to_datetime(str(int(nav.date.iloc[-1]))) - pd.to_datetime(str(int(nav.date.iloc[0])))).days / 365.25
    wins = trades.loc[trades.pnl > 0, "pnl"].sum()
    losses = -trades.loc[trades.pnl < 0, "pnl"].sum()
    returns = trades["return"]
    result = {
        "policy": policy,
        **POLICIES[policy],
        "initial_nav": 1_300_000.0,
        "end_nav": end_nav,
        "total_return": end_nav / 1_300_000.0 - 1,
        "cagr": (end_nav / 1_300_000.0) ** (1 / years) - 1,
        "max_drawdown": float(nav.drawdown.min()),
        "profit_factor": float(wins / losses) if losses else None,
        "win_rate": float((trades.pnl > 0).mean()),
        "trades": int(len(trades)),
        "loss_le_12": int((returns <= -.12).sum()),
        "loss_le_15": int((returns <= -.15).sum()),
        "loss_le_20": int((returns <= -.20).sum()),
        "loss_le_25": int((returns <= -.25).sum()),
        "max_single_loss": float(returns.min()),
        "margin_exit_trades": int((trades.reason == "MARGIN_EXIT").sum()),
        "margin_coverage": manifest["coverage"],
        "margin_sha256": manifest["sha256"],
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=sorted(POLICIES))
    ap.add_argument("--margin", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--formal", default="scripts/r10_max_formal.py", type=Path)
    ap.add_argument("--history", default="data/history/2020-2025", type=Path)
    ap.add_argument("--events", default="data/reference/official_corporate_actions_2020_2025.csv", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert manifest["coverage"]["TWSE"]["ratio"] == 1.0
    assert manifest["coverage"]["TPEX"]["ratio"] == 1.0
    assert sha256(args.margin) == manifest["sha256"]
    assert sha256(args.formal) == FORMAL_SHA

    run_dir = args.output.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for path in args.history.glob("*.parquet"):
        os.symlink(path.resolve(), run_dir / path.name)
    os.symlink(args.events.resolve(), run_dir / args.events.name)
    variant = inject(args.formal.read_text(encoding="utf-8"), args.policy, args.margin.resolve())
    variant_path = run_dir / "derived_overlay_engine.py"
    variant_path.write_text(variant, encoding="utf-8")
    execution_log = run_dir / "execution.log"
    with execution_log.open("w", encoding="utf-8") as log:
        completed = subprocess.run([sys.executable, variant_path.name], cwd=run_dir,
                                   stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        # Surface the inner engine traceback in Actions logs while retaining the
        # complete execution log in the policy directory.
        tail = execution_log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
        print("\n".join(tail), file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    result = summarize(run_dir, args.policy, manifest)
    (run_dir / "overlay_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.policy == "BASE":
        assert abs(result["end_nav"] - 2403427.0678222505) < 1e-6, result
        assert result["trades"] == 150, result


if __name__ == "__main__":
    main()
