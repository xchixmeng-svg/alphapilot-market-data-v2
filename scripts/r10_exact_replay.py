#!/usr/bin/env python3
from __future__ import annotations

import base64, gzip, json, math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / '.clean_cache'
PARTS = ROOT / 'exact_reference' / 'compact_parts'
OUT = ROOT / 'clean_results'
OUT.mkdir(exist_ok=True)
INITIAL = 1_300_000.0


def d8(v):
    return int(str(v).replace('-', '')[:8])


def tick(p: float) -> float:
    if p < 10: return .01
    if p < 50: return .05
    if p < 100: return .1
    if p < 500: return .5
    if p < 1000: return 1.0
    return 5.0


def ceil_tick(p: float) -> float:
    t = tick(p)
    return round(math.ceil((p - 1e-12) / t) * t, 8)


def floor_tick(p: float) -> float:
    t = tick(p)
    return round(math.floor((p + 1e-12) / t) * t, 8)


def load_bundle():
    files = sorted(PARTS.glob('gzpart*.b64'))
    if [p.name for p in files] != ['gzpart00.b64', 'gzpart01.b64', 'gzpart02.b64']:
        raise RuntimeError(f'exact reference parts missing: {[p.name for p in files]}')
    enc = ''.join(p.read_text(encoding='utf-8').strip() for p in files)
    b = json.loads(gzip.decompress(base64.b64decode(enc)).decode('utf-8'))
    if len(b.get('orders', [])) != 410 or len(b.get('trades', [])) != 241:
        raise RuntimeError(f"reference cardinality invalid orders={len(b.get('orders', []))} trades={len(b.get('trades', []))}")
    return b


def load_raw():
    parts = []
    for y in range(2020, 2026):
        p = CACHE / f'ohlcv_{y}.parquet'
        if not p.exists(): raise RuntimeError(f'missing {p}')
        parts.append(pd.read_parquet(p))
    q = pd.concat(parts, ignore_index=True)
    q['code'] = q.code.astype(str).str.strip().str.zfill(4)
    q['date'] = q.date.astype(int)
    return q.drop_duplicates(['date','code'], keep='last').sort_values(['date','code']).reset_index(drop=True)


def order_frame(b):
    return pd.DataFrame(b['orders'], columns=['strategy','order_date','code','t1_limit','status','fill_price'])


def trade_frame(b):
    return pd.DataFrame(b['trades'], columns=['strategy','code','buy_date','limit','buy_price','shares','cost','sell_date','sell_open','sell_price','proceeds'])


def validate_orders(raw, orders):
    bars = raw.set_index(['date','code'])
    rows = []
    for r in orders.itertuples(index=False):
        od = d8(r.order_date); code = str(r.code).zfill(4); limit = float(r.t1_limit); expected = str(r.status)
        key = (od, code)
        if key not in bars.index:
            rows.append({'strategy':r.strategy,'order_date':od,'code':code,'limit':limit,'expected':expected,'actual':'NO_BAR','status_match':False,'fill_match':False,'match':False,'reason':'NO_BAR'})
            continue
        b = bars.loc[key]
        if isinstance(b, pd.DataFrame): b = b.iloc[-1]
        op, lo = float(b.open), float(b.low)
        touched = op <= limit + 1e-9 or lo <= limit + 1e-9
        actual = 'FILLED' if touched else 'MISSED'
        # Formal workbook executions show 0.5% adverse buy slippage on gap-better opens,
        # rounded up to a legal tick and capped by the precommitted limit.
        actual_fill = min(limit, ceil_tick(op * 1.005)) if touched else None
        status_match = expected == actual
        exp_fill = None if pd.isna(r.fill_price) else float(r.fill_price)
        fill_match = (expected != 'FILLED') or (touched and exp_fill is not None and abs(exp_fill - actual_fill) < 1e-6)
        contradiction = bool(expected == 'MISSED' and touched)
        rows.append({'strategy':r.strategy,'order_date':od,'code':code,'limit':limit,'open':op,'low':lo,'expected':expected,'actual':actual,'expected_fill':exp_fill,'actual_fill':actual_fill,'status_match':status_match,'fill_match':fill_match,'match':status_match and fill_match,'formal_touch_miss_contradiction':contradiction,'reason':''})
    return pd.DataFrame(rows)


def validate_trades(raw, trades):
    bars = raw.set_index(['date','code'])
    rows = []
    for r in trades.itertuples(index=False):
        code = str(r.code).zfill(4); bd = d8(r.buy_date); sd = d8(r.sell_date)
        bk, sk = (bd,code), (sd,code)
        buy_bar_ok, sell_bar_ok = bk in bars.index, sk in bars.index
        buy_touch = buy_price_ok = sell_open_ok = sell_price_market_ok = False
        expected_buy = None
        if buy_bar_ok:
            bb = bars.loc[bk]; bb = bb.iloc[-1] if isinstance(bb,pd.DataFrame) else bb
            buy_touch = float(bb.open) <= float(r.limit) + 1e-9 or float(bb.low) <= float(r.limit) + 1e-9
            if buy_touch:
                expected_buy = min(float(r.limit), ceil_tick(float(bb.open) * 1.005))
                buy_price_ok = abs(float(r.buy_price) - expected_buy) < 1e-6
        if sell_bar_ok:
            sb = bars.loc[sk]; sb = sb.iloc[-1] if isinstance(sb,pd.DataFrame) else sb
            sell_open_ok = abs(float(sb.open) - float(r.sell_open)) < 1e-6
            # Market-plausibility audit only: workbook sale must not exceed T+1 open and
            # cannot be worse than 0.5% adverse slippage plus one legal tick of rounding.
            lower = floor_tick(float(r.sell_open) * 0.995) - tick(float(r.sell_open)) - 1e-9
            sell_price_market_ok = float(r.sell_price) <= float(r.sell_open) + 1e-9 and float(r.sell_price) >= lower
        ok = buy_bar_ok and buy_price_ok and sell_bar_ok and sell_open_ok and sell_price_market_ok
        rows.append({'strategy':r.strategy,'code':code,'buy_date':bd,'sell_date':sd,'buy_bar_ok':buy_bar_ok,'buy_touch':buy_touch,'expected_buy_from_raw':expected_buy,'recorded_buy':float(r.buy_price),'buy_price_ok':buy_price_ok,'sell_bar_ok':sell_bar_ok,'raw_sell_open':None if not sell_bar_ok else float(sb.open),'recorded_sell_open':float(r.sell_open),'sell_open_ok':sell_open_ok,'sell_price_market_ok':sell_price_market_ok,'match':ok})
    return pd.DataFrame(rows)


def replay_nav(raw, trades):
    t = trades.copy()
    t['buy_date'] = t.buy_date.map(d8); t['sell_date'] = t.sell_date.map(d8); t['code'] = t.code.astype(str).str.zfill(4)
    buys, sells = {}, {}
    for r in t.to_dict('records'):
        buys.setdefault(int(r['buy_date']), []).append(r); sells.setdefault(int(r['sell_date']), []).append(r)
    test = raw[(raw.date >= 20210104) & (raw.date <= 20251231)]
    dates = sorted(int(x) for x in test.date.unique())
    daily = {int(d):g[['code','close']].set_index('code')['close'].to_dict() for d,g in test.groupby('date')}
    cash, pos, last_close, rows, peak = INITIAL, {}, {}, [], INITIAL
    min_cash, max_pos, errors = INITIAL, 0, []
    for d in dates:
        marks = daily.get(d, {})
        last_close.update({str(c).zfill(4):float(px) for c,px in marks.items()})
        for r in sells.get(d, []):
            key = (str(r['strategy']), str(r['code']))
            if key not in pos:
                errors.append(f'sell_without_position:{d}:{key[0]}:{key[1]}')
            else:
                cash += float(r['proceeds']); del pos[key]
        for r in buys.get(d, []):
            key = (str(r['strategy']), str(r['code'])); sh = int(r['shares']); cost = float(r['cost'])
            if key in pos: errors.append(f'duplicate_position:{d}:{key[0]}:{key[1]}')
            if cost > cash + 1e-6: errors.append(f'cash_overdraft:{d}:{key[0]}:{key[1]}:{cost-cash:.6f}')
            cash -= cost; pos[key] = sh
        mv, missing = 0.0, []
        for (strategy,c), sh in pos.items():
            if c not in last_close: missing.append(f'{strategy}:{c}')
            else: mv += sh * last_close[c]
        if missing: errors.append(f'missing_mark:{d}:{"|".join(missing[:5])}')
        nav = cash + mv; peak = max(peak,nav); dd = nav/peak - 1.0
        min_cash = min(min_cash,cash); max_pos = max(max_pos,len(pos))
        rows.append({'date':d,'nav':nav,'cash':cash,'market_value':mv,'positions':len(pos),'peak':peak,'drawdown':dd})
    return pd.DataFrame(rows), min_cash, max_pos, errors


def annual_returns(nav):
    z = nav.copy(); z['year'] = z.date.astype(str).str[:4]; out = {}; prev = INITIAL
    for y,g in z.groupby('year',sort=True):
        end = float(g.iloc[-1].nav); out[str(y)] = end/prev - 1.0; prev = end
    return out


def main():
    b = load_bundle(); raw = load_raw(); orders = order_frame(b); trades = trade_frame(b)
    oa = validate_orders(raw,orders); ta = validate_trades(raw,trades)
    nav,min_cash,max_pos,replay_errors = replay_nav(raw,trades)
    end = float(nav.iloc[-1].nav); maxdd = float(nav.drawdown.min()); yrs = annual_returns(nav)
    target_end,target_dd,target_orders,target_trades = b['target']
    contradictions = oa[oa.get('formal_touch_miss_contradiction',False) == True] if 'formal_touch_miss_contradiction' in oa.columns else pd.DataFrame()
    summary = {
      'strategy':'AlphaPilot R10-MAX Exact Replay','audit_execution_completed':True,
      'source_workbook_sha256':b['source_sha256'],'reference_orders':len(orders),'reference_trades':len(trades),
      'order_status_matches':int(oa.status_match.sum()),'order_fill_matches':int(oa.fill_match.sum()),'order_exact_matches':int(oa.match.sum()),'order_mismatches':int((~oa.match).sum()),
      'formal_touch_miss_contradictions':int(len(contradictions)),
      'formal_touch_miss_cases':contradictions[['strategy','order_date','code','limit','open','low']].to_dict('records') if len(contradictions) else [],
      'trade_market_checks_passed':int(ta.match.sum()),'trade_market_check_failures':int((~ta.match).sum()),
      'initial_nav':INITIAL,'ending_nav_replayed':end,'ending_nav_target':float(target_end),'ending_nav_diff':end-float(target_end),
      'max_drawdown_replayed':maxdd,'target_max_drawdown':float(target_dd),'annual_returns_replayed':yrs,
      'min_cash_replayed':float(min_cash),'max_positions_replayed':int(max_pos),'replay_errors':replay_errors[:50],
    }
    summary['orders_410_gate'] = len(orders)==int(target_orders) and int(oa.match.sum())==int(target_orders)
    summary['trades_241_gate'] = len(trades)==int(target_trades) and int(ta.match.sum())==int(target_trades)
    summary['cash_nonnegative_gate'] = min_cash >= -1e-6 and not any(x.startswith('cash_overdraft') for x in replay_errors)
    summary['ending_nav_gate'] = abs(summary['ending_nav_diff']) < 1.0
    summary['exact_replay_pass'] = bool(summary['orders_410_gate'] and summary['trades_241_gate'] and summary['cash_nonnegative_gate'] and summary['ending_nav_gate'])
    summary['validation_verdict'] = 'PASS' if summary['exact_replay_pass'] else 'FAIL_WITH_EVIDENCE'
    oa.to_csv(OUT/'r10_exact_order_audit.csv',index=False); ta.to_csv(OUT/'r10_exact_trade_audit.csv',index=False); nav.to_csv(OUT/'r10_exact_nav.csv',index=False)
    (OUT/'r10_exact_replay_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    # Audit execution itself is successful even when the strategy-validation verdict is FAIL.
    # The verdict is carried in exact_replay_pass/validation_verdict; do not turn evidence into a CI crash.

if __name__=='__main__': main()
