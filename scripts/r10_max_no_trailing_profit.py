"""
還原股價處理：偵測分割/減資/長期停牌恢復造成的價格斷點，往回調整歷史價格使序列連續。

輸入：ohlcv_2020.csv.gz ~ ohlcv_2025.csv.gz（欄位：date,code,name,volume,open,high,low,close）
輸出：ohlcv_adjusted_2020_2025.csv.gz（新增 aclose/aopen/ahigh/alow/factor/event_ratio 欄位）

方法：
- 用「收盤對收盤」比值，超過±15%視為非正常交易造成的跳空（台股單日漲跌停約±10%，
  超過這個範圍幾乎確定是分割/減資/停牌恢復）
- 偵測到事件後，把事件當天(含)之前的所有歷史價格，乘上對應倍數，讓價格序列連續
"""
import pandas as pd
import numpy as np

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

df['prev_close'] = df.groupby('code')['close'].shift(1)
df['ratio'] = df['close'] / df['prev_close']

THRESH = 0.15
df['event_ratio'] = np.where(
    df['prev_close'].notna() & ((df['ratio'] - 1).abs() > THRESH),
    df['ratio'],
    1.0
)

def calc_factor(g):
    e = g['event_ratio'].values
    suffix_incl = np.cumprod(e[::-1])[::-1]
    F = np.ones(len(e))
    F[:-1] = suffix_incl[1:]
    return pd.Series(F, index=g.index)

df['factor'] = df.groupby('code', group_keys=False).apply(calc_factor)
df['aclose'] = df['close'] * df['factor']
df['aopen'] = df['open'] * df['factor']
df['ahigh'] = df['high'] * df['factor']
df['alow'] = df['low'] * df['factor']

n_events = (df['event_ratio'] != 1.0).sum()
print(f"detected {n_events} split/corp-action events across {df.code.nunique()} codes")

df.to_csv('ohlcv_adjusted_2020_2025.csv.gz', index=False, compression='gzip')
print("saved ohlcv_adjusted_2020_2025.csv.gz", df.shape)

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

px = pd.read_csv('ohlcv_adjusted_2020_2025.csv.gz', dtype={'code': str})
px = px[['date', 'code', 'name', 'open', 'high', 'low', 'close', 'volume', 'aclose']].copy()
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
             'r7_exposure', 'r7_slots', 'amount_ratio', 'amtacc', 'vol20']
px[keep_cols].to_pickle('r10max_signals_final.pkl')
print("saved r10max_signals_final.pkl")

# ============================================================
# 步驟三：組合層回測引擎
# ============================================================

px = pd.read_pickle('r10max_signals_final.pkl')
px = px.sort_values(['date', 'code']).reset_index(drop=True)

all_dates = sorted(px.date.unique().tolist())
date_idx = {d: i for i, d in enumerate(all_dates)}
next_date = {all_dates[i]: all_dates[i+1] for i in range(len(all_dates)-1)}
by_date = {d: sub.set_index('code') for d, sub in px.groupby('date')}

BUY_FEE = 0.000855
SELL_FEE = 0.000855
SELL_TAX = 0.003
INITIAL_CAPITAL = 1_300_000.0
SINGLE_CAP = 0.25
TOTAL_CAP = 0.95
MAX_POSITIONS = 5
R7_BASE = 0.22
R05_BASE = 0.20
R05_MAX_SLOTS = 5
BUY_ADVERSE = 0.005
SELL_ADVERSE = 0.005
MIN_HOLD_DAYS = 3

FORCE_DD = 0.14
FORCE_TARGET = 0.50
FORCE_COOL = 15
FORCE_NOENTRY = 10

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

# 修正: 買進套用0.5%不利滑價，向上取合法tick，並以限價封頂
def buy_fill(open_, low_, limit_):
    if open_ <= limit_:
        raw = open_ * (1 + BUY_ADVERSE)
        filled = ceil_tick(raw)
        return min(filled, limit_)
    elif low_ <= limit_:
        return limit_
    else:
        return None

cash = INITIAL_CAPITAL
positions = {}
pending_buys, pending_sells = {}, {}
nav_rows, trade_rows, unfilled_rows, no_buy_log, slot_diag = [], [], [], [], []
hwm = INITIAL_CAPITAL
no_buy_until = -1
force_cooldown_until = -1
forced_count = 0
r7_last_rebalance = -999
r7_last_regime = None

warmup = 260
eval_dates = all_dates[warmup:]

for di in eval_dates:
    i = date_idx[di]
    sub = by_date.get(di)

    # ---- 執行昨天決定、今天要賣的單 ----
    for o in pending_sells.pop(di, []):
        k = o['code']; p = positions.get(k)
        if p is None:
            continue
        if sub is None or k not in sub.index:
            nd = next_date.get(di)
            if nd:
                o['execute_date'] = nd
                pending_sells.setdefault(nd, []).append(o)
            continue
        r = sub.loc[k]
        px_fill = floor_tick(float(r['open']) * (1 - SELL_ADVERSE))
        proceeds = px_fill * p['shares'] * (1 - SELL_FEE - SELL_TAX)
        cash += proceeds
        pnl = proceeds - p['cost_total']
        trade_rows.append({'code': k, 'strategy': p['strategy'], 'entry_date': p['entry_date'],
                            'entry_price': p['entry_adj'], 'exit_date': di, 'exit_price': px_fill,
                            'pnl': pnl, 'reason': o['reason'],
                            'hold_days': p['hold_days'], 'return': proceeds / p['cost_total'] - 1})
        del positions[k]

    # ---- 執行昨天決定、今天要買的單 ----
    for o in pending_buys.pop(di, []):
        if sub is None or o['code'] not in sub.index:
            unfilled_rows.append({'code': o['code'], 'strategy': o['strategy'], 'order_date': di, 'limit': o['limit'], 'reason': 'NO_PRICE_DATA'})
            continue
        r = sub.loc[o['code']]
        fill = buy_fill(float(r['open']), float(r['low']), o['limit'])
        if fill is None:
            unfilled_rows.append({'code': o['code'], 'strategy': o['strategy'], 'order_date': di, 'limit': o['limit'],
                                    'day_low': float(r['low']), 'day_open': float(r['open']), 'reason': 'PRICE_NOT_TOUCHED'})
        if fill is not None:
            cost = fill * o['shares'] * (1 + BUY_FEE)
            if cost <= cash + 1e-6 and o['code'] not in positions:
                cash -= cost
                positions[o['code']] = {'code': o['code'], 'strategy': o['strategy'], 'shares': o['shares'],
                                         'entry_date': di, 'entry_adj': fill, 'cost_total': cost,
                                         'peak_adj': fill, 'hold_days': 0, 'runner': None,
                                         'pending_sell': False}

    # ---- 標記市值 ----
    mv = 0.0
    for k, p in positions.items():
        r = sub.loc[k] if (sub is not None and k in sub.index) else None
        pxnow = float(r['close']) if r is not None else p['entry_adj']
        mv += p['shares'] * pxnow
    nav = cash + mv
    hwm = max(hwm, nav)
    dd = nav / hwm - 1.0

    for p in positions.values():
        r = sub.loc[p['code']] if (sub is not None and p['code'] in sub.index) else None
        if r is not None and np.isfinite(r['aclose']):
            p['peak_adj'] = max(p['peak_adj'], float(r['aclose']))
        p['hold_days'] += 1

    exdate = next_date.get(di)

    # ---- R7 15個交易日換手節奏：只在初次、Regime改變、或滿15個交易日才重新排名 ----
    cur_regime = sub['regime'].iloc[0] if (sub is not None and len(sub)) else r7_last_regime
    r7_is_rebalance_day = True  # 停用換手節奏限制，維持每天重算
    if r7_is_rebalance_day:
        r7_last_rebalance = i
        r7_last_regime = cur_regime

    sell_list = []
    for k, p in list(positions.items()):
        # 修正: 最短持有3個交易日才能產生賣出決策
        if p['hold_days'] < MIN_HOLD_DAYS or sub is None or k not in sub.index:
            continue
        r = sub.loc[k]
        adj = float(r['aclose']); ret = adj / p['entry_adj'] - 1.0
        reason = None
        if p['strategy'] == 'R7':
            if ret <= -0.12:
                reason = 'R7_HARD'
            elif r7_is_rebalance_day and ((float(r.get('r7_exposure', 1.0)) <= 0.0) or (not bool(r.get('r7_hard', False)))):
                reason = 'R7_REB'
        else:  # R0.5 — Runner/Mega/Target 狀態機（以 amount_ratio 判斷，一次性吸收態）
            amount_ratio = r.get('amount_ratio', np.nan)
            if p['runner'] in (None, 'runner') and ret >= 0.80:
                # 首次到達約+80%，一次性分類成 MEGA 或 TARGET，之後不再改變
                if p['runner'] != 'mega' and p['runner'] != 'target':
                    if amount_ratio is not None and np.isfinite(amount_ratio) and amount_ratio >= 1.2:
                        p['runner'] = 'mega'
                    else:
                        p['runner'] = 'target'
            elif p['runner'] is None and ret >= 0.40 and amount_ratio is not None and np.isfinite(amount_ratio) and amount_ratio >= 2.0:
                p['runner'] = 'runner'

            if ret <= -0.10:
                reason = 'R05_HARD'
            else:
                state = p['runner']
                if state == 'target':
                    if ret >= 2.00 or adj <= p['peak_adj']*(1-0.20):
                        reason = 'R05_TARGET'
                elif state == 'mega':
                    if adj <= p['peak_adj']*(1-0.16):
                        reason = 'R05_MEGA_TRAIL'
                elif state == 'runner':
                    if adj <= p['peak_adj']*(1-0.14):
                        reason = 'R05_RUNNER_TRAIL'
                elif ret >= 0.50:
                    if adj <= p['peak_adj']*(1-0.12):
                        reason = 'R05_BASE_TRAIL'
                if reason is None:
                    cap = 120 if state is not None else 60
                    if p['hold_days'] >= cap:
                        reason = 'R05_MAXHOLD'
        if reason and exdate is not None:
            sell_list.append({'code': k, 'reason': reason})
            positions[k]['pending_sell'] = True
    if sell_list and exdate is not None:
        pending_sells.setdefault(exdate, []).extend(sell_list)

    # ---- 強制回撤降曝險：真正賣到約50%曝險，不是只鎖倉 ----
    if dd <= -FORCE_DD and i >= force_cooldown_until:
        force_cooldown_until = i + FORCE_COOL
        no_buy_until = max(no_buy_until, i + FORCE_NOENTRY)
        forced_count += 1
        # 弱勢優先賣出，賣到約50%曝險為止（尊重最短持有天數與待賣狀態）
        candidates = [(k, p) for k, p in positions.items() if not p['pending_sell'] and p['hold_days'] >= MIN_HOLD_DAYS]
        def _ret(item):
            k, p = item
            r = sub.loc[k] if (sub is not None and k in sub.index) else None
            return (float(r['aclose'])/p['entry_adj'] - 1) if r is not None and np.isfinite(r.get('aclose', np.nan)) else 0
        candidates.sort(key=_ret)
        cur_mv_force = sum(p['shares'] * (float(sub.loc[k]['close']) if sub is not None and k in sub.index else p['entry_adj'])
                            for k, p in positions.items())
        target_mv = nav * FORCE_TARGET
        for k, p in candidates:
            if cur_mv_force <= target_mv or exdate is None:
                break
            r = sub.loc[k] if (sub is not None and k in sub.index) else None
            px_now = float(r['close']) if r is not None else p['entry_adj']
            sell_list.append({'code': k, 'reason': 'FORCE_DD'})
            positions[k]['pending_sell'] = True
            pending_sells.setdefault(exdate, []).append({'code': k, 'reason': 'FORCE_DD'})
            cur_mv_force -= p['shares'] * px_now

    # ---- 進場：待賣持倉在成交前，仍計入槽位與曝險（修正：不提前釋放）----
    created = []
    if exdate is not None and i >= no_buy_until and sub is not None:
        n_open = len(positions)  # 待賣但未成交的部位仍算持有
        n_r7_open = sum(1 for p in positions.values() if p['strategy']=='R7')
        n_r05_open = sum(1 for p in positions.values() if p['strategy']=='R05')
        slots_free = MAX_POSITIONS - n_open
        if slots_free > 0:
            mult = dd_multiplier(dd)
            r7_exposure = float(sub['r7_exposure'].iloc[0]) if len(sub) and np.isfinite(sub['r7_exposure'].iloc[0]) else 0.0
            r7_slots_regime = int(sub['r7_slots'].iloc[0]) if len(sub) and np.isfinite(sub['r7_slots'].iloc[0]) else 0
            r7_slots_free = max(0, r7_slots_regime - n_r7_open)
            r05_slots_free = max(0, R05_MAX_SLOTS - n_r05_open)
            slot_diag.append({'date': di, 'total_slots_free': slots_free, 'r7_slots_free': r7_slots_free, 'r05_slots_free': r05_slots_free})
            already_held = set(positions.keys())

            r7_cand = sub[sub['r7_hard'] == True].copy() if (r7_exposure > 0 and r7_slots_free > 0 and r7_is_rebalance_day) else sub.iloc[0:0]
            r7_cand = r7_cand[~r7_cand.index.isin(already_held)]
            r7_cand = r7_cand.sort_values('r7_score', ascending=False).head(r7_slots_free)

            r05_cand = sub[sub['r05_hard'] == True].copy() if r05_slots_free > 0 else sub.iloc[0:0]
            r05_cand = r05_cand[~r05_cand.index.isin(already_held)]
            r05_cand = r05_cand.sort_values('r05_score', ascending=False).head(r05_slots_free)

            combined = [('R7', c, r) for c, r in r7_cand.iterrows()] + [('R05', c, r) for c, r in r05_cand.iterrows()]
            combined_sorted = sorted(combined, key=lambda x: -(x[2]['r7_score'] if x[0]=='R7' else x[2]['r05_score']))

            cur_mv = sum(p['shares'] * (float(sub.loc[k]['close']) if k in sub.index else p['entry_adj'])
                         for k, p in positions.items())
            created_mv = 0.0
            used_codes = set()
            for strat, code, row in combined_sorted:
                if slots_free <= 0:
                    break
                if code in used_codes:
                    continue
                base_pct = R7_BASE if strat == 'R7' else R05_BASE
                if strat == 'R7':
                    limit = floor_tick(float(row['aclose']) * 0.98)
                    base_target = base_pct * nav * mult * r7_exposure
                else:
                    limit = floor_tick(float(row['close']) * 0.995)
                    base_target = base_pct * nav * mult
                rem_single = nav * SINGLE_CAP
                rem_total = nav * TOTAL_CAP - cur_mv - created_mv
                target_amt = min(base_target, rem_single, rem_total, cash - created_mv - 1000)
                if target_amt <= 0:
                    continue
                # 高價股：若1張超過目標資金，允許整數零股；否則以1000股為單位
                if limit * 1000 > target_amt:
                    shares = int(target_amt / limit)
                else:
                    shares = int(target_amt / limit / 1000) * 1000
                # 流動性上限：單筆股數 <= 當日20日均量的2%
                vol20 = row.get('vol20', np.nan)
                if pd.notna(vol20) and vol20 > 0:
                    liq_cap = int(vol20 * 0.02 / 1000) * 1000
                    if liq_cap < 1000:
                        liq_cap = int(vol20 * 0.02)  # 允許零股，避免完全排除低量股
                    shares = min(shares, liq_cap) if liq_cap > 0 else 0
                if shares <= 0:
                    continue
                created.append({'code': code, 'strategy': strat, 'limit': limit, 'shares': shares})
                created_mv += shares * limit
                used_codes.add(code)
                slots_free -= 1
            if created:
                pending_buys.setdefault(exdate, []).extend(created)

    no_buy_log.append({'date': di, 'no_buy_active': i < no_buy_until})
    nav_rows.append({'date': di, 'nav': nav, 'drawdown': dd})

nav_df = pd.DataFrame(nav_rows)
trades_df = pd.DataFrame(trade_rows)
end_nav = float(nav_df.iloc[-1].nav)
total_ret = end_nav/INITIAL_CAPITAL - 1
years = (pd.to_datetime(str(eval_dates[-1]), format='%Y%m%d') - pd.to_datetime(str(eval_dates[0]), format='%Y%m%d')).days/365.25
cagr = (end_nav/INITIAL_CAPITAL)**(1/years) - 1
max_dd = float(nav_df.drawdown.min())
win_rate = float((trades_df.pnl>0).mean()) if len(trades_df) else np.nan
pf = (trades_df[trades_df.pnl>0].pnl.sum() / -trades_df[trades_df.pnl<=0].pnl.sum()) if len(trades_df) and (trades_df.pnl<=0).any() else np.nan

print(f"起始: {INITIAL_CAPITAL:,.0f}  終止: {end_nav:,.0f}")
print(f"總報酬: {total_ret:.2%}  CAGR: {cagr:.2%}  MaxDD: {max_dd:.2%}")
print(f"交易數: {len(trades_df)}  勝率: {win_rate:.2%}  PF: {pf:.2f}")
print(f"強制降曝險觸發次數: {forced_count}")
if len(trades_df):
    print(trades_df.groupby('strategy').agg(n=('pnl','count'), win=('pnl', lambda s:(s>0).mean()), sum_pnl=('pnl','sum')))
    print(trades_df.groupby('reason').agg(n=('pnl','count'), sum_pnl=('pnl','sum')))

nav_df.to_csv('r10max_nav.csv', index=False)
trades_df.to_csv('r10max_trades.csv', index=False)
pd.DataFrame(unfilled_rows).to_csv('r10max_unfilled.csv', index=False)
pd.DataFrame(no_buy_log).to_csv('r10max_no_buy_log.csv', index=False)
pd.DataFrame(slot_diag).to_csv('r10max_slot_diag.csv', index=False)
