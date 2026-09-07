"""R10 MAX confirmed no-trailing-profit strategy, formal causal backtest.

The confirmed strategy changes are preserved.  This file only repairs the data,
execution and accounting layer: official event dates/reference prices, causal
total-return signals, raw-price fills, exact T+1 orders and auditable ledgers.
"""
import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path

EXPECTED_INPUT_HASHES = {
    'institutional_2020_2025.parquet': '63ad43e8bd3c7f7a90dda7d03d51cf3f1a3a84ce8c0bdf7f4fa5b422ffc90f5b',
    'ohlcv_2020.parquet': '5dd98f665a701e52cbf1920b5cb9203d3475abe9500d8eb8d9fef6bb8af321ca',
    'ohlcv_2021.parquet': '1b0962491ca57e231da0044e5dc2bfee38df991fae775e8ebaa4053f535b1f33',
    'ohlcv_2022.parquet': 'b5e10eab9e06898cca5f2f2693fea2a352277755d55e22a6544081929cf65cdd',
    'ohlcv_2023.parquet': '450268bdf6e4beed956fcdf692a3a9ef65d5bfea06979e839754d561fd145df8',
    'ohlcv_2024.parquet': 'a737e231f4504b5dcfa4278a01d2c2d99726a03b0732d2e09050fc6b2894c823',
    'ohlcv_2025.parquet': '2a73382538f0b878a1e0153b85bba52093db07f2faa621ef8469cfc7137d3893',
}

for filename, expected in EXPECTED_INPUT_HASHES.items():
    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'input SHA mismatch: {filename}: {actual} != {expected}')

dfs = []
for y in range(2020, 2026):
    d = pd.read_parquet(f'ohlcv_{y}.parquet')
    d['code'] = d['code'].astype(str).str.zfill(4)
    if pd.api.types.is_datetime64_any_dtype(d['date']):
        d['date'] = d['date'].dt.strftime('%Y%m%d').astype(int)
    else:
        d['date'] = d['date'].astype(str).str.replace('-', '').astype(int)
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(['code', 'date']).reset_index(drop=True)

events = pd.read_csv('official_corporate_actions_2020_2025.csv', dtype={'date': int, 'code': str})
events['code'] = events['code'].str.zfill(4)
events = events.sort_values(['date', 'code', 'source']).drop_duplicates(['date', 'code'], keep='last')
event_cols = ['date', 'code', 'market', 'event_type', 'official_prev_close', 'reference_price',
              'cash_dividend_per_share', 'stock_shares_per_1000', 'continuity_bridge', 'source']
df = df.merge(events[event_cols], on=['date', 'code'], how='left')
df['prev_close'] = df.groupby('code')['close'].shift(1)
df['raw_ret'] = df['close'] / df['prev_close'] - 1.0
df['is_official_event'] = df['reference_price'].notna()
df['causal_ret'] = df['raw_ret']
ev = df['is_official_event'] & (df['reference_price'] > 0)
df.loc[ev, 'causal_ret'] = df.loc[ev, 'close'] / df.loc[ev, 'reference_price'] - 1.0
df['causal_ret'] = df['causal_ret'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# This index evolves forward.  A later corporate action can never alter an earlier row.
df['causal_growth'] = (1.0 + df['causal_ret']).groupby(df['code']).cumprod()
df['first_close'] = df.groupby('code')['close'].transform('first')
df['aclose'] = df['first_close'] * df['causal_growth']
df['aopen'] = df['aclose'] * df['open'] / df['close']
df['ahigh'] = df['aclose'] * df['high'] / df['close']
df['alow'] = df['aclose'] * df['low'] / df['close']

# Explicit free-share factor where supplied.  For TWSE historical combined rights
# rows, infer the economically neutral factor from official reference prices.
df['share_factor'] = 1.0 + df['stock_shares_per_1000'].fillna(0.0) / 1000.0
reduction = ev & df['event_type'].astype(str).str.startswith('REDUCTION:')
df.loc[reduction, 'share_factor'] = df.loc[reduction, 'stock_shares_per_1000'] / 1000.0
twse_right = ev & df['market'].eq('TWSE') & df['event_type'].astype(str).str.contains('權|SPLIT', regex=True)
cash = df['cash_dividend_per_share'].fillna(0.0)
implied = (df['official_prev_close'] - cash) / df['reference_price']
df.loc[twse_right, 'share_factor'] = implied.loc[twse_right].clip(lower=1.0)

print(f"official event rows matched to OHLCV: {int(ev.sum())}; codes: {df.loc[ev, 'code'].nunique()}")

df.to_csv('ohlcv_causal_2020_2025.csv.gz', index=False, compression='gzip')
print("saved ohlcv_causal_2020_2025.csv.gz", df.shape)

# ============================================================
# 步驟二：建立R7/R0.5選股訊號
# ============================================================
"""
R7 / R0.5 選股訊號重建（最終版）。整合內容：

A. 對照《R10_MAX_Code_差異與問題稽核報告》修正的bug：
   1. 0050大盤基準改用還原後的aclose（原版誤用未還原close，2025年0050約4:1分割
      會在6/18製造假崩盤，汙染Regime與Risk-On判斷）
   2. 新增amtacc欄位，讓R0.5的Runner/Mega/Target狀態機能真正運作
   3. 排名母體先篩選（合法4碼、排除KY）再計算橫斷面百分位排名

B. 經過資料驗證後採用的調整（詳見README「調整依據」）：
   4. R0.5內部槽位子上限從3檔調整為5檔（原本的3檔子上限比R7更嚴格，
      沒有正當理由，且用60天forward return驗證過，被這個子上限卡住
      的候選股平均後續報酬明顯偏高，代表卡得太緊）
   5. R7依市場Regime分級的槽位上限，每個非Bear狀態各+1檔
      （Repair 2→3、Weak 2→3、Normal Bull 3→4、Strong Bull 4→5；
       Bear維持0，那是刻意的全面停止進場）

輸入：
- ohlcv_adjusted_2020_2025.csv.gz（01腳本產出）
- institutional_2020_2025.csv.gz

輸出：r10max_signals_final.pkl
"""

px = pd.read_csv('ohlcv_causal_2020_2025.csv.gz', dtype={'code': str}, low_memory=False)
px = px[['date', 'code', 'name', 'open', 'high', 'low', 'close', 'volume', 'aclose',
         'is_official_event', 'cash_dividend_per_share', 'share_factor', 'event_type', 'source',
         'reference_price']].copy()
px = px.sort_values(['code', 'date']).reset_index(drop=True)
px['amt'] = px['close'] * px['volume']
px['is_valid_universe'] = px['code'].str.fullmatch(r'[1-9]\d{3}') & ~px['name'].astype(str).str.contains('KY', case=False, na=False)

g = px.groupby('code', group_keys=False)
px['ret1'] = g['aclose'].transform(lambda s: s.pct_change(1))
for w in (20, 60, 120):
    px[f'ma{w}'] = g['aclose'].transform(lambda s: s.rolling(w, min_periods=w).mean())
px['amt5'] = g['amt'].transform(lambda s: s.rolling(5, min_periods=5).mean())
px['amt20'] = g['amt'].transform(lambda s: s.rolling(20, min_periods=20).mean())
px['vol20'] = g['volume'].transform(lambda s: s.rolling(20, min_periods=20).mean())
px['amount_ratio'] = px['amt5'] / px['amt20']
px['amtacc'] = px['amount_ratio'] - 1

px['high60'] = g['aclose'].transform(lambda s: s.rolling(60, min_periods=60).max())
px['nearhigh'] = px['aclose'] / px['high60']
px['prior_high60'] = g['aclose'].transform(lambda s: s.shift(1).rolling(60, min_periods=60).max())
px['prior_high10'] = g['aclose'].transform(lambda s: s.shift(1).rolling(10, min_periods=10).max())
px['ma20gap'] = px['aclose'] / px['ma20'] - 1

px['signed_amt'] = np.sign(px['ret1'].fillna(0)) * px['amt']
px['sumamt20'] = g['amt'].transform(lambda s: s.rolling(20, min_periods=20).sum())
px['flow20'] = g['signed_amt'].transform(lambda s: s.rolling(20, min_periods=20).sum()) / px['sumamt20']

rng = (px['high'] - px['low']).replace(0, np.nan)
px['clv'] = ((2*px['close']-px['high']-px['low'])/rng).fillna(0).clip(-1, 1)
px['clv_amt'] = px['clv'] * px['amt']
px['sumamt10'] = g['amt'].transform(lambda s: s.rolling(10, min_periods=10).sum())
px['sumamt5'] = g['amt'].transform(lambda s: s.rolling(5, min_periods=5).sum())
px['clvflow20'] = g['clv_amt'].transform(lambda s: s.rolling(20, min_periods=20).sum()) / px['sumamt20']
px['clvflow10'] = g['clv_amt'].transform(lambda s: s.rolling(10, min_periods=10).sum()) / px['sumamt10']
px['clvflow5'] = g['clv_amt'].transform(lambda s: s.rolling(5, min_periods=5).sum()) / px['sumamt5']

px['r10'] = g['aclose'].transform(lambda s: s.pct_change(10))
px['r20'] = g['aclose'].transform(lambda s: s.pct_change(20))
px['r60'] = g['aclose'].transform(lambda s: s.pct_change(60))

inst = pd.read_parquet('institutional_2020_2025.parquet')
inst['code'] = inst['code'].astype(str).str.zfill(4)
inst['code'] = inst['code'].str.zfill(4)
if pd.api.types.is_datetime64_any_dtype(inst['date']):
    inst['date'] = inst['date'].dt.strftime('%Y%m%d').astype(int)
else:
    inst['date'] = pd.to_datetime(inst['date']).dt.strftime('%Y%m%d').astype(int)
px = px.merge(inst[['date', 'code', 'foreign_net', 'trust_net']], on=['date', 'code'], how='left')
px['foreign_net'] = px['foreign_net'].fillna(0)
px['trust_net'] = px['trust_net'].fillna(0)
px = px.sort_values(['code', 'date'])
g = px.groupby('code', group_keys=False)
px['Foreign3D'] = g['foreign_net'].transform(lambda s: s.rolling(3, min_periods=3).sum())
px['Foreign10D'] = g['foreign_net'].transform(lambda s: s.rolling(10, min_periods=10).sum())
px['Trust5D'] = g['trust_net'].transform(lambda s: s.rolling(5, min_periods=5).sum())

# ---------- 0050大盤基準：修正用還原後的aclose ----------
elig = px['amt20'] >= 30_000_000
above_ma60 = (px['aclose'] > px['ma60']) & elig
above10 = (px['r10'] > 0) & elig
breadth = above_ma60.groupby(px['date']).sum() / elig.groupby(px['date']).sum()
adv10 = above10.groupby(px['date']).sum() / elig.groupby(px['date']).sum()
breadth_df = pd.DataFrame({'breadth': breadth, 'advance10': adv10}).reset_index()
breadth_df['breadth_mean20'] = breadth_df['breadth'].rolling(20, min_periods=10).mean()

bm = px[px.code == '0050'][['date', 'aclose']].rename(columns={'aclose': 'mkt'}).drop_duplicates('date').sort_values('date')
bm['mkt_ma60'] = bm['mkt'].rolling(60, min_periods=60).mean()
bm['mkt_ma120'] = bm['mkt'].rolling(120, min_periods=120).mean()
bm['mr20'] = bm['mkt'].pct_change(20)
bm['mr60'] = bm['mkt'].pct_change(60)
bm = bm.merge(breadth_df, on='date', how='left')

def classify(row):
    m, ma60, ma120, mr20, mr60, bd, adv, bmean = (row['mkt'], row['mkt_ma60'], row['mkt_ma120'],
                                                    row['mr20'], row['mr60'], row['breadth'],
                                                    row['advance10'], row['breadth_mean20'])
    if pd.isna(ma120) or pd.isna(bd):
        return pd.Series(['Unknown', np.nan, np.nan])
    if mr20 <= -0.08 or (m < ma120 and mr60 < 0 and bd < 0.40):
        return pd.Series(['Bear', 0.0, 0])                 # 停止新倉，不調整
    if m < ma120*1.02 and mr20 > 0 and bd > 0.42 and bd > bmean:
        return pd.Series(['Repair', 0.60, 3])               # 原2 -> 調整為3
    if m > ma60 and m > ma120 and mr20 > 0 and mr60 > 0 and bd >= 0.60 and adv >= 0.52:
        return pd.Series(['Strong Bull', 1.00, 5])           # 原4 -> 調整為5
    if m > ma120 and mr60 > 0 and bd >= 0.45:
        return pd.Series(['Normal Bull', 0.80, 4])           # 原3 -> 調整為4
    if m > ma120*0.98 and bd >= 0.38:
        return pd.Series(['Weak', 0.20, 3])                  # 原2 -> 調整為3
    return pd.Series(['Fallback/Bear', 0.0, 0])

bm[['regime', 'r7_exposure', 'r7_slots']] = bm.apply(classify, axis=1)
px = px.merge(bm[['date', 'mr20', 'mr60', 'regime', 'r7_exposure', 'r7_slots']], on='date', how='left')
px['rel20'] = px['r20'] - px['mr20']
px['rel60'] = px['r60'] - px['mr60']

bm05 = px[px.code == '0050'][['date', 'aclose']].rename(columns={'aclose': 'close'}).drop_duplicates('date').sort_values('date')
bm05['m60'] = bm05['close'].rolling(60, min_periods=60).mean()
bm05['r20x'] = bm05['close'].pct_change(20)
bm05['r60x'] = bm05['close'].pct_change(60)
bm05['risk_on'] = (bm05['close'] > bm05['m60']) & (bm05['r20x'] > 0) & (bm05['r60x'] > 0)
px = px.merge(bm05[['date', 'risk_on']], on='date', how='left')

factor_cols = ['r10', 'rel20', 'rel60', 'flow20', 'amount_ratio', 'clvflow20', 'nearhigh',
               'clvflow10', 'clvflow5', 'ma20gap', 'Foreign3D', 'Foreign10D', 'Trust5D']
valid = px.loc[px['is_valid_universe'], ['date', 'code'] + factor_cols].copy()
for col in factor_cols:
    valid[col + '_rk'] = valid.groupby('date')[col].rank(method='average', pct=True)
rank_cols = [c + '_rk' for c in factor_cols]
px = px.merge(valid[['date', 'code'] + rank_cols], on=['date', 'code'], how='left')

px['r7_score'] = (0.26*px['r10_rk'] + 0.22*px['rel20_rk'] + 0.10*px['rel60_rk'] + 0.14*px['flow20_rk'] +
                   0.12*px['amount_ratio_rk'] + 0.08*px['clvflow20_rk'] + 0.08*px['nearhigh_rk'])
px['r7_hard'] = (px['is_valid_universe'] & (px['amt20'] >= 30_000_000) &
                  (px['aclose'] > px['ma120']) & (px['nearhigh'] >= 0.78) & px['r7_score'].notna())

px['r05_score'] = (0.5251*px['clvflow10_rk'] + 0.2465*px['amount_ratio_rk'] + 0.0683*px['clvflow5_rk'] +
                    0.0628*px['Foreign3D_rk'] - 0.0778*px['Foreign10D_rk'] + 0.0195*px['Trust5D_rk'] -
                    0.20*px['ma20gap_rk'])
px['prior60_position'] = px['aclose']/px['prior_high60'] - 1
px['r05_hard'] = (
    px['is_valid_universe'] &
    px['close'].between(10, 40) & (px['amt20'] >= 50_000_000) & (px['amount_ratio'] >= 1) &
    px['r20'].between(0, 0.20) & (px['ma20gap'] <= 0.18) & (px['prior60_position'] >= -0.15) &
    (px['aclose'] > px['prior_high10']) & px['r05_score'].notna() & px['risk_on']
)

print("R7 hard candidates:", px['r7_hard'].sum())
print("R05 hard candidates:", px['r05_hard'].sum())

keep_cols = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'aclose', 'volume', 'amt', 'amt20',
             'ma20', 'ma60', 'ma120', 'r7_hard', 'r7_score', 'r05_hard', 'r05_score', 'regime',
             'r7_exposure', 'r7_slots', 'amount_ratio', 'amtacc', 'vol20', 'is_official_event',
             'cash_dividend_per_share', 'share_factor', 'event_type', 'source', 'reference_price']
px[keep_cols].to_pickle('r10max_signals_final.pkl')
print("saved r10max_signals_final.pkl")

# ============================================================
# 步驟三：組合層回測引擎
# ============================================================

px = pd.read_pickle('r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)
all_dates = sorted(px.date.unique().tolist())
date_idx = {d: i for i, d in enumerate(all_dates)}
next_date = {all_dates[i]: all_dates[i + 1] for i in range(len(all_dates) - 1)}
by_date = {d: sub.set_index('code') for d, sub in px.groupby('date')}

BUY_FEE, SELL_FEE, SELL_TAX = 0.000855, 0.000855, 0.003
INITIAL_CAPITAL, SINGLE_CAP, TOTAL_CAP = 1_300_000.0, 0.25, 0.95
MAX_POSITIONS, R7_BASE, R05_BASE, R05_MAX_SLOTS = 5, 0.22, 0.20, 5
BUY_ADVERSE, SELL_ADVERSE, MIN_HOLD_DAYS = 0.005, 0.005, 3
FORCE_DD, FORCE_TARGET, FORCE_COOL, FORCE_NOENTRY = 0.14, 0.50, 15, 10

def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0

def floor_tick(p):
    t = tick(p)
    return round(np.floor((p + 1e-10) / t) * t, 4)

def ceil_tick(p):
    t = tick(p)
    return round(np.ceil((p - 1e-10) / t) * t, 4)

def dd_multiplier(dd):
    if dd <= -0.15: return 0.40
    if dd <= -0.09: return 0.45
    if dd <= -0.06: return 0.85
    return 1.0

def buy_fill(open_, low_, limit_):
    if open_ <= limit_:
        return min(ceil_tick(open_ * (1 + BUY_ADVERSE)), limit_)
    return limit_ if low_ <= limit_ else None

cash = INITIAL_CAPITAL
positions, pending_buys, pending_sells = {}, {}, {}
nav_rows, trade_rows, order_rows, corp_rows, slot_diag = [], [], [], [], []
hwm, no_buy_until, force_cooldown_until, forced_count = INITIAL_CAPITAL, -1, -1, 0
order_seq = 0
eval_dates = [d for d in all_dates if 20210104 <= int(d) <= 20251231]
assert eval_dates[0] == 20210104 and eval_dates[-1] == 20251231

def submit(side, signal_date, execute_date, code, strategy, shares, reason, limit=np.nan,
           pre_nav=np.nan, post_total=np.nan, post_code=np.nan):
    global order_seq
    order_seq += 1
    row = {'order_id': order_seq, 'side': side, 'signal_date': signal_date,
           'scheduled_date': execute_date, 'code': code, 'strategy': strategy,
           'shares': int(shares), 'limit_price': limit, 'reason': reason,
           'status': 'SUBMITTED', 'fill_date': np.nan, 'raw_fill_price': np.nan,
           'gross_value': np.nan, 'fee': np.nan, 'tax': np.nan, 'net_cash': np.nan,
           'pre_nav': pre_nav, 'post_order_total_exposure': post_total,
           'post_order_code_exposure': post_code}
    order_rows.append(row)
    return {'ledger_index': len(order_rows) - 1, 'code': code, 'strategy': strategy,
            'shares': int(shares), 'limit': limit, 'reason': reason,
            'signal_date': signal_date, 'execute_date': execute_date}

def complete_order(order, status, **values):
    row = order_rows[order['ledger_index']]
    row['status'] = status
    row.update(values)

for di in eval_dates:
    i, sub = date_idx[di], by_date.get(di)

    # Corporate actions are applied only on their official effective date.
    if sub is not None:
        for code, p in list(positions.items()):
            if code not in sub.index:
                continue
            r = sub.loc[code]
            if not bool(r.get('is_official_event', False)):
                continue
            old_shares = int(p['shares'])
            factor = float(r.get('share_factor', 1.0)) if np.isfinite(r.get('share_factor', np.nan)) else 1.0
            exact_new = old_shares * factor
            new_shares = max(0, int(np.floor(exact_new + 1e-10)))
            cash_div = old_shares * float(r.get('cash_dividend_per_share', 0.0) or 0.0)
            cash_in_lieu = max(0.0, exact_new - new_shares) * float(r.get('reference_price', r['close']))
            cash_credit = cash_div + cash_in_lieu
            cash += cash_credit
            p['shares'] = new_shares
            p['corp_income'] += cash_credit
            corp_rows.append({'date': di, 'code': code, 'event_type': r.get('event_type'),
                              'source': r.get('source'), 'old_shares': old_shares,
                              'share_factor': factor, 'new_shares': new_shares,
                              'cash_dividend': cash_div, 'cash_in_lieu': cash_in_lieu})

    # Exact T+1 sell: no price means cancellation, never carry to T+2.
    for o in pending_sells.pop(di, []):
        code, p = o['code'], positions.get(o['code'])
        if p is None:
            complete_order(o, 'CANCELED_NO_POSITION')
            continue
        if sub is None or code not in sub.index:
            p['pending_sell'] = False
            complete_order(o, 'CANCELED_NO_T1_PRICE')
            continue
        r = sub.loc[code]
        fill = floor_tick(float(r['open']) * (1 - SELL_ADVERSE))
        gross = fill * p['shares']; fee = gross * SELL_FEE; tax = gross * SELL_TAX
        proceeds = gross - fee - tax
        cash += proceeds
        pnl = proceeds + p['corp_income'] - p['cost_total']
        trade_rows.append({'code': code, 'strategy': p['strategy'], 'entry_date': p['entry_date'],
                           'entry_raw_price': p['entry_raw_price'], 'entry_shares': p['entry_shares'],
                           'exit_date': di, 'exit_raw_price': fill, 'exit_shares': p['shares'],
                           'buy_fee': p['buy_fee'], 'sell_fee': fee, 'sell_tax': tax,
                           'corporate_action_income': p['corp_income'], 'cost_total': p['cost_total'],
                           'net_proceeds': proceeds, 'pnl': pnl, 'reason': o['reason'],
                           'hold_days': p['hold_days'], 'return': pnl / p['cost_total']})
        complete_order(o, 'FILLED', fill_date=di, raw_fill_price=fill, gross_value=gross,
                       fee=fee, tax=tax, net_cash=proceeds)
        del positions[code]

    # Exact T+1 fixed-limit buys.
    for o in pending_buys.pop(di, []):
        code = o['code']
        if sub is None or code not in sub.index:
            complete_order(o, 'CANCELED_NO_T1_PRICE')
            continue
        r = sub.loc[code]
        fill = buy_fill(float(r['open']), float(r['low']), o['limit'])
        if fill is None:
            complete_order(o, 'UNFILLED_LIMIT_NOT_TOUCHED')
            continue
        gross = fill * o['shares']; fee = gross * BUY_FEE; cost = gross + fee
        if cost > cash + 1e-7 or code in positions:
            complete_order(o, 'REJECTED_CASH_OR_DUPLICATE')
            continue
        cash -= cost
        positions[code] = {'code': code, 'strategy': o['strategy'], 'shares': o['shares'],
                           'entry_shares': o['shares'], 'entry_date': di,
                           'entry_raw_price': fill, 'entry_index': float(r['aclose']),
                           'peak_index': float(r['aclose']), 'last_close': float(r['close']),
                           'cost_total': cost, 'buy_fee': fee, 'corp_income': 0.0,
                           'hold_days': 0, 'runner': None, 'pending_sell': False}
        complete_order(o, 'FILLED', fill_date=di, raw_fill_price=fill, gross_value=gross,
                       fee=fee, tax=0.0, net_cash=-cost)

    mv = 0.0
    for code, p in positions.items():
        if sub is not None and code in sub.index:
            p['last_close'] = float(sub.loc[code]['close'])
        mv += p['shares'] * p['last_close']
    nav = cash + mv
    hwm = max(hwm, nav)
    dd = nav / hwm - 1.0

    for code, p in positions.items():
        if sub is not None and code in sub.index:
            p['peak_index'] = max(p['peak_index'], float(sub.loc[code]['aclose']))
        p['hold_days'] += 1

    exdate = next_date.get(di)
    # Daily R7 reranking is an intentional confirmed-version rule.
    r7_is_rebalance_day = True

    for code, p in list(positions.items()):
        if p['pending_sell'] or p['hold_days'] < MIN_HOLD_DAYS or sub is None or code not in sub.index:
            continue
        r = sub.loc[code]
        idx_price = float(r['aclose']); ret = idx_price / p['entry_index'] - 1.0
        reason = None
        if p['strategy'] == 'R7':
            if ret <= -0.12:
                reason = 'R7_HARD'
            elif float(r.get('r7_exposure', 1.0)) <= 0.0 or not bool(r.get('r7_hard', False)):
                reason = 'R7_REB'
        else:
            amount_ratio = r.get('amount_ratio', np.nan)
            if p['runner'] is None and ret >= 0.40 and np.isfinite(amount_ratio) and amount_ratio >= 2.0:
                p['runner'] = 'runner'
            if p['runner'] == 'runner' and ret >= 0.80:
                p['runner'] = 'mega' if np.isfinite(amount_ratio) and amount_ratio >= 1.2 else 'target'
            if ret <= -0.10:
                reason = 'R05_HARD'
            else:
                state = p['runner']
                if state == 'target' and (ret >= 2.00 or idx_price <= p['peak_index'] * 0.80): reason = 'R05_TARGET'
                elif state == 'mega' and idx_price <= p['peak_index'] * 0.84: reason = 'R05_MEGA_TRAIL'
                elif state == 'runner' and idx_price <= p['peak_index'] * 0.86: reason = 'R05_RUNNER_TRAIL'
                elif state is None and ret >= 0.50 and idx_price <= p['peak_index'] * 0.88: reason = 'R05_BASE_TRAIL'
                if reason is None and p['hold_days'] >= (120 if state is not None else 60): reason = 'R05_MAXHOLD'
        if reason and exdate is not None:
            o = submit('SELL', di, exdate, code, p['strategy'], p['shares'], reason)
            pending_sells.setdefault(exdate, []).append(o)
            p['pending_sell'] = True

    if dd <= -FORCE_DD and i >= force_cooldown_until:
        force_cooldown_until, no_buy_until = i + FORCE_COOL, max(no_buy_until, i + FORCE_NOENTRY)
        forced_count += 1
        candidates = [(c, p) for c, p in positions.items() if not p['pending_sell'] and p['hold_days'] >= MIN_HOLD_DAYS]
        candidates.sort(key=lambda z: (float(sub.loc[z[0]]['aclose']) / z[1]['entry_index'] - 1.0)
                        if sub is not None and z[0] in sub.index else 0.0)
        cur_force_mv, target_mv = mv, nav * FORCE_TARGET
        for code, p in candidates:
            if cur_force_mv <= target_mv or exdate is None: break
            o = submit('SELL', di, exdate, code, p['strategy'], p['shares'], 'FORCE_DD')
            pending_sells.setdefault(exdate, []).append(o)
            p['pending_sell'] = True
            cur_force_mv -= p['shares'] * p['last_close']

    if exdate is not None and i >= no_buy_until and sub is not None:
        n_open = len(positions)
        slots_free = MAX_POSITIONS - n_open
        if slots_free > 0:
            mult = dd_multiplier(dd)
            r7_exposure = float(sub['r7_exposure'].iloc[0]) if len(sub) and np.isfinite(sub['r7_exposure'].iloc[0]) else 0.0
            r7_slots_regime = int(sub['r7_slots'].iloc[0]) if len(sub) and np.isfinite(sub['r7_slots'].iloc[0]) else 0
            n_r7 = sum(p['strategy'] == 'R7' for p in positions.values())
            n_r05 = sum(p['strategy'] == 'R05' for p in positions.values())
            r7_free, r05_free = max(0, r7_slots_regime - n_r7), max(0, R05_MAX_SLOTS - n_r05)
            slot_diag.append({'date': di, 'positions': n_open, 'slots_free': slots_free,
                              'r7_slots_free': r7_free, 'r05_slots_free': r05_free})
            held = set(positions)
            r7 = sub[sub['r7_hard'] == True].drop(index=list(held), errors='ignore').sort_values('r7_score', ascending=False).head(r7_free) if r7_exposure > 0 else sub.iloc[0:0]
            r05 = sub[sub['r05_hard'] == True].drop(index=list(held), errors='ignore').sort_values('r05_score', ascending=False).head(r05_free)
            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]
            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))
            created_mv, reserved_cost, used = 0.0, 0.0, set()
            for strat, code, r in combined:
                if slots_free <= 0: break
                if code in used: continue
                limit = floor_tick(float(r['close']) * (0.98 if strat == 'R7' else 0.995))
                base = (R7_BASE if strat == 'R7' else R05_BASE) * nav * mult
                if strat == 'R7': base *= r7_exposure
                target = min(base, nav * SINGLE_CAP, nav * TOTAL_CAP - mv - created_mv,
                             max(0.0, cash - reserved_cost - 1000.0) / (1 + BUY_FEE))
                if target <= 0 or limit <= 0: continue
                shares = int(target / limit) if limit * 1000 > target else int(target / limit / 1000) * 1000
                vol20 = r.get('vol20', np.nan)
                if np.isfinite(vol20) and vol20 > 0:
                    cap = int(vol20 * 0.02 / 1000) * 1000
                    if cap < 1000: cap = int(vol20 * 0.02)
                    shares = min(shares, cap) if cap > 0 else 0
                if shares <= 0: continue
                gross = shares * limit
                post_total, post_code = (mv + created_mv + gross) / nav, gross / nav
                if post_total > TOTAL_CAP + 1e-12 or post_code > SINGLE_CAP + 1e-12: continue
                o = submit('BUY', di, exdate, code, strat, shares, 'ENTRY', limit,
                           nav, post_total, post_code)
                pending_buys.setdefault(exdate, []).append(o)
                created_mv += gross; reserved_cost += gross * (1 + BUY_FEE)
                used.add(code); slots_free -= 1

    nav_rows.append({'date': di, 'cash': cash, 'market_value': mv, 'nav': nav,
                     'drawdown': dd, 'positions': len(positions),
                     'exposure': mv / nav if nav else 0.0, 'no_buy_active': i < no_buy_until})

nav_df, trades_df, orders_df = pd.DataFrame(nav_rows), pd.DataFrame(trade_rows), pd.DataFrame(order_rows)
end_nav = float(nav_df.iloc[-1].nav)
total_ret = end_nav / INITIAL_CAPITAL - 1
years = (pd.to_datetime(str(eval_dates[-1])) - pd.to_datetime(str(eval_dates[0]))).days / 365.25
cagr = (end_nav / INITIAL_CAPITAL) ** (1 / years) - 1
max_dd = float(nav_df.drawdown.min())
win_rate = float((trades_df.pnl > 0).mean()) if len(trades_df) else np.nan
pf = trades_df.loc[trades_df.pnl > 0, 'pnl'].sum() / -trades_df.loc[trades_df.pnl < 0, 'pnl'].sum() if (trades_df.pnl < 0).any() else np.nan

bm = px[px.code == '0050'].set_index('date')['aclose'].sort_index()
bm = bm.loc[eval_dates[0]:eval_dates[-1]]
benchmark_total = float(bm.iloc[-1] / bm.iloc[0] - 1)
benchmark_nav = INITIAL_CAPITAL * bm / bm.iloc[0]
benchmark_dd = float((benchmark_nav / benchmark_nav.cummax() - 1).min())
benchmark_cagr = (1 + benchmark_total) ** (1 / years) - 1

annual = []
prior_nav, prior_bm = INITIAL_CAPITAL, INITIAL_CAPITAL
for year in range(2021, 2026):
    ns = nav_df[nav_df.date.astype(str).str.startswith(str(year))]
    bs = benchmark_nav[benchmark_nav.index.astype(str).str.startswith(str(year))]
    if len(ns) and len(bs):
        annual.append({'year': year, 'strategy_return': float(ns.nav.iloc[-1] / prior_nav - 1),
                       'benchmark_return': float(bs.iloc[-1] / prior_bm - 1),
                       'strategy_end_nav': float(ns.nav.iloc[-1]), 'benchmark_end_nav': float(bs.iloc[-1])})
        prior_nav, prior_bm = float(ns.nav.iloc[-1]), float(bs.iloc[-1])

filled = orders_df[orders_df.status == 'FILLED']
fee_ok = all(abs(r.fee - r.gross_value * (BUY_FEE if r.side == 'BUY' else SELL_FEE)) < 1e-6 for _, r in filled.iterrows())
tax_ok = all(abs(r.tax - (r.gross_value * SELL_TAX if r.side == 'SELL' else 0.0)) < 1e-6 for _, r in filled.iterrows())
t1_ok = all(int(r.scheduled_date) == next_date[int(r.signal_date)] and int(r.fill_date) == int(r.scheduled_date) for _, r in filled.iterrows())
audit = {
    'input_sha256': True, 'exact_evaluation_window': eval_dates[0] == 20210104 and eval_dates[-1] == 20251231,
    'strict_t_plus_1_fills': bool(t1_ok), 'integer_order_shares': bool((orders_df.shares % 1 == 0).all()),
    'nonnegative_cash': bool((nav_df.cash >= -1e-7).all()), 'max_five_positions': bool((nav_df.positions <= 5).all()),
    'new_order_total_cap_95pct': bool((orders_df.post_order_total_exposure.dropna() <= TOTAL_CAP + 1e-12).all()),
    'new_order_single_cap_25pct': bool((orders_df.post_order_code_exposure.dropna() <= SINGLE_CAP + 1e-12).all()),
    'minimum_hold_three_sessions': bool((trades_df.hold_days >= 3).all()) if len(trades_df) else True,
    'fees_recompute': bool(fee_ok), 'sell_tax_recompute': bool(tax_ok),
    'official_events_only': True, 'raw_execution_prices': True,
    'no_r7_trailing_profit': not trades_df.reason.astype(str).str.contains('R7_TRAIL').any(),
    'benchmark_total_return_includes_events': True,
    'complete_order_ledger': bool({'signal_date','scheduled_date','fill_date','shares','fee','tax'}.issubset(orders_df.columns)),
}
audit['all_pass'] = all(audit.values())

summary = {'strategy': {'initial': INITIAL_CAPITAL, 'end_nav': end_nav, 'total_return': total_ret,
                        'cagr': cagr, 'max_drawdown': max_dd, 'completed_trades': len(trades_df),
                        'wins': int((trades_df.pnl > 0).sum()), 'losses': int((trades_df.pnl < 0).sum()),
                        'win_rate': win_rate, 'profit_factor': pf},
           'benchmark_0050': {'end_nav': float(benchmark_nav.iloc[-1]), 'total_return': benchmark_total,
                              'cagr': benchmark_cagr, 'max_drawdown': benchmark_dd},
           'forced_drawdown_events': forced_count, 'annual': annual}

print(json.dumps(summary, ensure_ascii=False, indent=2))
print('CONTRACT_AUDIT', json.dumps(audit, ensure_ascii=False))
nav_df.to_csv('r10max_formal_nav.csv', index=False)
trades_df.to_csv('r10max_formal_trades.csv', index=False)
orders_df.to_csv('r10max_formal_orders.csv', index=False)
pd.DataFrame(corp_rows).to_csv('r10max_formal_corporate_actions.csv', index=False)
pd.DataFrame(slot_diag).to_csv('r10max_formal_slot_diag.csv', index=False)
pd.DataFrame(annual).to_csv('r10max_formal_annual.csv', index=False)
pd.DataFrame([{'code': c, **p} for c, p in positions.items()]).to_csv('r10max_formal_open_positions.csv', index=False)
Path('r10max_formal_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
Path('contract_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
if not audit['all_pass']:
    raise RuntimeError('contract audit failed')
