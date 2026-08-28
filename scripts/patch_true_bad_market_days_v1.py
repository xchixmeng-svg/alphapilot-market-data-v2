#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parent
fp = root / 'r10_fast_validation.py'
tp = root / 'r10_true_validation.py'
fs = fp.read_text(encoding='utf-8')
ts = tp.read_text(encoding='utf-8')

# ---------- FAST signal precompute: exact R6.2 BAD-day detector ----------
anchor = '''def precompute_signals(feat: pd.DataFrame, bm: pd.DataFrame, ins: pd.DataFrame, eval_dates: list[int]):
'''
helper = '''def detect_bad_market_days(feat: pd.DataFrame) -> set[int]:
    """Exact R6.2 incomplete-market-day rule.

    Universe first follows the normal 4-digit common-stock/KY exclusion mask.
    BAD when daily row count < 80% of its 20D rolling median (min_periods=5).
    """
    dates = [int(x) for x in sorted(feat.date.unique())]
    u = feat[core.common(feat)].copy()
    counts = u.groupby("date").size().reindex(dates, fill_value=0).astype(float)
    med = counts.rolling(20, min_periods=5).median()
    bad = counts < 0.8 * med
    bad[med.isna()] = False
    return {int(d) for d in counts.index[bad]}


def precompute_signals(feat: pd.DataFrame, bm: pd.DataFrame, ins: pd.DataFrame, eval_dates: list[int]):
'''
if 'def detect_bad_market_days(' not in fs:
    if anchor not in fs:
        raise SystemExit('precompute anchor missing')
    fs = fs.replace(anchor, helper, 1)

old = '''    # R7 breadth and benchmark state, all rolling operations are backward-looking only.
    eligible = feat[(feat.amt20 >= 30_000_000) & feat.aclose.notna()].copy()
    breadth = eligible.groupby("date").apply(
        lambda g: pd.Series({"breadth": float((g.aclose > g.ma60).mean()), "advance10": float((g.r10 > 0).mean())}),
        include_groups=False,
    ).reset_index().sort_values("date")
    breadth["breadth_mean20"] = breadth.breadth.rolling(20, min_periods=10).mean()
    bmap = breadth.set_index("date")
'''
new = '''    # R7 breadth and benchmark state, all rolling operations are backward-looking only.
    bad_market_days = detect_bad_market_days(feat)
    all_dates = [int(x) for x in sorted(feat.date.unique())]
    eligible = feat[(feat.amt20 >= 30_000_000) & feat.aclose.notna()].copy()
    breadth = eligible.groupby("date").apply(
        lambda g: pd.Series({"breadth": float((g.aclose > g.ma60).mean()), "advance10": float((g.r10 > 0).mean())}),
        include_groups=False,
    ).reset_index().set_index("date").reindex(all_dates)
    # Exact R6.2 behavior: on BAD dates carry prior Breadth/Advance instead of
    # computing them from the severely incomplete cross-section.
    for j, d in enumerate(all_dates):
        if d in bad_market_days and j > 0:
            breadth.loc[d, "breadth"] = breadth.loc[all_dates[j-1], "breadth"]
            breadth.loc[d, "advance10"] = breadth.loc[all_dates[j-1], "advance10"]
    breadth = breadth.reset_index().rename(columns={"index":"date"})
    breadth["breadth_mean20"] = breadth.breadth.rolling(20, min_periods=10).mean()
    bmap = breadth.set_index("date")
'''
if old not in fs:
    raise SystemExit('breadth anchor missing')
fs = fs.replace(old, new, 1)

old = '''        x = x0[core.common(x0)].copy()
        r = zmap.loc[di]
        v = bmap.loc[di]
        m, ma60, ma120, mr20, mr60, br, adv, bmean = [float(q) for q in (r.mkt, r.ma60, r.ma120, r.mr20, r.mr60, v.breadth, v.advance10, v.breadth_mean20)]
        if mr20 <= -.08 or (m < ma120 and mr60 < 0 and br < .40): reg, expo, slots = "Bear", 0., 0
'''
new = '''        x = x0[core.common(x0)].copy()
        r = zmap.loc[di]
        v = bmap.loc[di]
        m, ma60, ma120, mr20, mr60, br, adv, bmean = [float(q) for q in (r.mkt, r.ma60, r.ma120, r.mr20, r.mr60, v.breadth, v.advance10, v.breadth_mean20)]
        if di in bad_market_days:
            if i <= 0 or eval_dates[i-1] not in r7_states:
                raise RuntimeError(f"BAD market day has no prior regime: {di}")
            prev = dict(r7_states[eval_dates[i-1]])
            prev["bad_market_day"] = True
            r7_states[di] = prev
            ec = x.iloc[0:0].copy()
            ec["r7_score"] = np.nan
            ec["r7_rank"] = np.array([], dtype=int)
            r7_cands[di] = ec
            pr05 = dict(r05_states.get(eval_dates[i-1], {"risk_on": False}))
            pr05["bad_market_day"] = True
            r05_states[di] = pr05
            er = x.iloc[0:0].copy()
            er["r05_score"] = np.nan
            er["r05_rank"] = np.array([], dtype=int)
            r05_cands[di] = er
            continue
        if mr20 <= -.08 or (m < ma120 and mr60 < 0 and br < .40): reg, expo, slots = "Bear", 0., 0
'''
if old not in fs:
    raise SystemExit('BAD regime insertion anchor missing')
fs = fs.replace(old, new, 1)

# Make normal states explicitly auditable.
fs = fs.replace('"advance10":adv}\n        r7_cands[di] = c', '"advance10":adv,"bad_market_day":False}\n        r7_cands[di] = c', 1)

fp.write_text(fs, encoding='utf-8')

# ---------- TRUE engine: no new T decisions on BAD dates ----------
old = '''    by_date, r7_states, r7_cands_map, r05_states, r05_cands_map = fast.precompute_signals(
        feat, bm, ins, eval_dates
    )

    cash = float(bt.INITIAL_CAPITAL)
'''
new = '''    by_date, r7_states, r7_cands_map, r05_states, r05_cands_map = fast.precompute_signals(
        feat, bm, ins, eval_dates
    )
    bad_market_days = fast.detect_bad_market_days(feat)
    bt.log(f"[BAD-MARKET] {sorted(d for d in bad_market_days if eval_start <= d <= eval_end)}")

    cash = float(bt.INITIAL_CAPITAL)
'''
if old not in ts:
    raise SystemExit('true precompute anchor missing')
ts = ts.replace(old, new, 1)

old = '''        exposure = stock_mv / nav if nav > 0 else 0.0

        # Peaks are updated using information known at T close only.
        for p in positions.values():
            r = bt.row_lookup(feat_idx, di, p.code)
            if r is not None and np.isfinite(r.aclose):
                p.peak_adj = max(p.peak_adj, float(r.aclose))

        r7_state = r7_states[di]
'''
new = '''        exposure = stock_mv / nav if nav > 0 else 0.0
        is_bad_market_day = di in bad_market_days

        # A severely incomplete market date is not valid signal information.
        # It may mark NAV, but it must not update trailing peaks.
        if not is_bad_market_day:
            for p in positions.values():
                r = bt.row_lookup(feat_idx, di, p.code)
                if r is not None and np.isfinite(r.aclose):
                    p.peak_adj = max(p.peak_adj, float(r.aclose))

        r7_state = r7_states[di]
'''
if old not in ts:
    raise SystemExit('true peak anchor missing')
ts = ts.replace(old, new, 1)

old = '''        r05_score = {} if r05_cands.empty else {
            str(x.code): float(x.r05_score) for x in r05_cands[["code", "r05_score"]].itertuples(index=False)
        }

        sell_map: dict[str, bt.SellOrder] = {}
'''
new = '''        r05_score = {} if r05_cands.empty else {
            str(x.code): float(x.r05_score) for x in r05_cands[["code", "r05_score"]].itertuples(index=False)
        }

        if is_bad_market_day:
            # Time passes, but no new signal/exit/entry/force-DD decision may be
            # formed from an incomplete cross-section. Yesterday's orders were
            # already executed above and NAV was already marked.
            for p in positions.values():
                p.hold_days += 1
            nav_rows.append({
                "date": di, "nav": nav, "cash": cash, "stock_mv": stock_mv,
                "exposure": exposure, "drawdown": dd,
                "positions": len({p.code for p in positions.values()}),
                "r7_positions": sum(p.strategy == "R7" for p in positions.values()),
                "r05_positions": sum(p.strategy == "R05" for p in positions.values()),
                "r7_regime": regime, "r7_regime_exposure": r7_state["exposure"],
                "r7_rebalance_due": False,
                "r05_risk_on": bool(r05_state.get("risk_on", False)),
                "dd_multiplier": bt.dd_multiplier(dd), "no_buy_active": i < no_buy_until,
                "bad_market_day": True,
            })
            if (i + 1) % 100 == 0 or i + 1 == len(eval_dates):
                bt.log(f"[TRUE-SIM] {i+1}/{len(eval_dates)} date={di} nav={nav:,.0f} trades={len(trade_rows)} BAD")
            continue

        sell_map: dict[str, bt.SellOrder] = {}
'''
if old not in ts:
    raise SystemExit('true BAD decision anchor missing')
ts = ts.replace(old, new, 1)

# Normal NAV rows carry an explicit False flag for auditability.
old = '''            "dd_multiplier": bt.dd_multiplier(dd), "no_buy_active": i < no_buy_until,
        })
'''
new = '''            "dd_multiplier": bt.dd_multiplier(dd), "no_buy_active": i < no_buy_until,
            "bad_market_day": False,
        })
'''
if old not in ts:
    raise SystemExit('normal nav row anchor missing')
ts = ts.replace(old, new, 1)

tp.write_text(ts, encoding='utf-8')
print('PATCHED', fp)
print('PATCHED', tp)
