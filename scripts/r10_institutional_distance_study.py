from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

def normalize_date(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime('%Y%m%d').astype(int)
    return pd.to_datetime(s.astype(str), errors='coerce').dt.strftime('%Y%m%d').astype('Int64')

def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not m.any():
        return np.nan
    values, weights = values[m], weights[m]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    c = np.cumsum(weights)
    return float(values[np.searchsorted(c, q * c[-1], side='left')])

def pf_from_pnl(s):
    s = pd.Series(s).dropna()
    gp, gl = s[s > 0].sum(), -s[s < 0].sum()
    return float(gp / gl) if gl > 0 else np.nan

def safe_qcut(s, q=4):
    out = pd.Series(index=s.index, dtype='float64')
    v = s.dropna()
    if len(v) < q * 4 or v.nunique() < q:
        return out
    out.loc[v.index] = pd.qcut(v.rank(method='first'), q, labels=False) + 1
    return out

def group_summary(df, group_col):
    rows = []
    for g, x in df.dropna(subset=[group_col]).groupby(group_col):
        rows.append({
            'group': int(g), 'n': int(len(x)),
            'win_rate': float((x['return'] > 0).mean()),
            'avg_return': float(x['return'].mean()),
            'median_return': float(x['return'].median()),
            'profit_factor': pf_from_pnl(x['pnl']),
            'avg_mfe': float(x['mfe'].mean()),
            'avg_mae': float(x['mae'].mean()),
            'avg_distance': float(x['distance'].mean()),
            'avg_abs_distance': float(x['abs_distance'].mean()),
        })
    return rows

def permutation_pvalue(a, b, rng, n_perm=5000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b]); na = len(a); hits = 0
    for _ in range(n_perm):
        p = rng.permutation(pool)
        hits += abs(p[:na].mean() - p[na:].mean()) >= obs - 1e-15
    return (hits + 1) / (n_perm + 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal-dir', default='formal_run')
    ap.add_argument('--institutional', default='data/history/2020-2025/institutional_2020_2025.parquet')
    ap.add_argument('--out-dir', default='formal_run/inst_distance_study')
    args = ap.parse_args()

    formal = Path(args.formal_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(formal / 'r10max_formal_trades.csv', dtype={'code': str})
    orders = pd.read_csv(formal / 'r10max_formal_orders.csv', dtype={'code': str})
    px = pd.read_csv(formal / 'ohlcv_causal_2020_2025.csv.gz', dtype={'code': str}, low_memory=False)
    inst = pd.read_parquet(args.institutional)

    for d in (trades, orders, px, inst):
        if 'code' in d.columns:
            d['code'] = d['code'].astype(str).str.zfill(4)

    for d, c in [(trades, 'entry_date'), (trades, 'exit_date'), (orders, 'signal_date'),
                 (orders, 'scheduled_date'), (orders, 'fill_date'), (px, 'date'), (inst, 'date')]:
        if c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                d[c] = pd.to_numeric(d[c], errors='coerce').astype('Int64')
            else:
                d[c] = normalize_date(d[c])

    buys = orders[(orders['side'] == 'BUY') & (orders['status'] == 'FILLED')].copy()
    buys = buys[['order_id','code','strategy','signal_date','fill_date','raw_fill_price','shares']]
    buys = buys.rename(columns={'fill_date':'entry_date','raw_fill_price':'buy_order_fill','shares':'buy_order_shares'})
    merged = trades.merge(buys, on=['code','strategy','entry_date'], how='left', validate='many_to_one')
    if merged['signal_date'].isna().any():
        miss = merged.loc[merged['signal_date'].isna(), ['code','strategy','entry_date']].to_dict('records')[:10]
        raise RuntimeError(f"Could not match filled BUY orders for {merged.signal_date.isna().sum()} trades: {miss}")

    need = {'date','code','close','high','low','aclose','ahigh','alow'}
    missing = need - set(px.columns)
    if missing:
        raise RuntimeError(f"ohlcv causal file missing columns: {sorted(missing)}")
    px['scale'] = px['aclose'] / px['close'].replace(0, np.nan)
    px['atypical'] = (px['ahigh'] + px['alow'] + px['aclose']) / 3.0
    px = px.sort_values(['code','date'])

    inst['date'] = normalize_date(inst['date']) if not pd.api.types.is_numeric_dtype(inst['date']) else pd.to_numeric(inst['date'], errors='coerce').astype('Int64')
    available_inst = [c for c in ['foreign_net','trust_net','dealer_net'] if c in inst.columns]
    if not {'foreign_net','trust_net'}.issubset(available_inst):
        raise RuntimeError(f"Expected foreign_net and trust_net; available columns={list(inst.columns)}")

    base = px[['date','code','atypical']].merge(inst[['date','code'] + available_inst], on=['date','code'], how='left')
    for c in available_inst:
        base[c] = pd.to_numeric(base[c], errors='coerce').fillna(0.0)
    base['combined_net'] = base[available_inst].clip(lower=0).sum(axis=1)
    entities = [('foreign','foreign_net'), ('trust','trust_net'), ('combined','combined_net')]
    if 'dealer_net' in available_inst:
        entities.insert(2, ('dealer','dealer_net'))

    by_code = {c: g.reset_index(drop=True) for c,g in base.groupby('code', sort=False)}
    px_by_code = {c: g.reset_index(drop=True) for c,g in px.groupby('code', sort=False)}
    feature_rows, windows = [], [5,10,20]

    for idx, tr in merged.reset_index(drop=True).iterrows():
        code = tr['code']; sig = int(tr['signal_date']); ent = int(tr['entry_date']); ex = int(tr['exit_date'])
        g, pg = by_code.get(code), px_by_code.get(code)
        if g is None or pg is None:
            continue
        erow = pg[pg['date'] == ent]
        if erow.empty:
            continue
        scale = float(erow.iloc[0]['scale'])
        entry_index = float(tr['entry_raw_price']) * scale
        hold = pg[(pg['date'] >= ent) & (pg['date'] < ex)]
        mfe = float(hold['ahigh'].max() / entry_index - 1) if len(hold) else np.nan
        mae = float(hold['alow'].min() / entry_index - 1) if len(hold) else np.nan
        hist_all = g[g['date'] <= sig]

        for entity, col in entities:
            for w in windows:
                hist = hist_all.tail(w).copy()
                weights = hist[col].clip(lower=0).to_numpy(float)
                vals = hist['atypical'].to_numpy(float)
                pos = weights > 0
                if not pos.any():
                    continue
                center = float(np.average(vals[pos], weights=weights[pos]))
                low = weighted_quantile(vals[pos], weights[pos], 0.20)
                high = weighted_quantile(vals[pos], weights[pos], 0.80)
                distance = entry_index / center - 1.0
                if entry_index < low:
                    zone_distance = entry_index / low - 1.0
                elif entry_index > high:
                    zone_distance = entry_index / high - 1.0
                else:
                    zone_distance = 0.0
                feature_rows.append({
                    'trade_id': idx, 'code': code, 'strategy': tr['strategy'],
                    'signal_date': sig, 'entry_date': ent, 'exit_date': ex,
                    'entry_raw_price': float(tr['entry_raw_price']),
                    'return': float(tr['return']), 'pnl': float(tr['pnl']),
                    'mfe': mfe, 'mae': mae, 'entity': entity, 'window': w,
                    'positive_buy_days': int(pos.sum()),
                    'positive_buy_weight': float(weights[pos].sum()),
                    'accum_center_index': center, 'accum_low_index': low, 'accum_high_index': high,
                    'distance': distance, 'abs_distance': abs(distance),
                    'distance_to_zone': zone_distance, 'abs_distance_to_zone': abs(zone_distance),
                    'entry_inside_zone': bool(low <= entry_index <= high),
                    'position_vs_zone': 'below' if entry_index < low else ('above' if entry_index > high else 'inside'),
                    'overextension_above_zone': max(zone_distance, 0.0),
                })

    feat = pd.DataFrame(feature_rows)
    if feat.empty:
        raise RuntimeError('No institutional distance features produced')
    feat.to_csv(out / 'institutional_distance_trade_features.csv', index=False)

    rng = np.random.default_rng(20260929)
    summary_rows, detail = [], {}
    for (entity, window), z0 in feat.groupby(['entity','window']):
        for segment in ['ALL','R7','R05']:
            z = z0 if segment == 'ALL' else z0[z0['strategy'] == segment]
            if len(z) < 20:
                continue
            z = z.copy()
            z['signed_q'] = safe_qcut(z['distance'])
            z['close_q'] = safe_qcut(z['abs_distance'])
            z['zone_q'] = safe_qcut(z['abs_distance_to_zone'])
            q = group_summary(z, 'close_q')
            if len(q) < 4:
                continue
            q1, q4 = z[z['close_q'] == 1], z[z['close_q'] == 4]
            s1, s4 = z[z['signed_q'] == 1], z[z['signed_q'] == 4]
            above = z[z['position_vs_zone'] == 'above']
            not_above = z[z['position_vs_zone'] != 'above']
            corr = z['abs_distance'].rank(method='average').corr(z['return'].rank(method='average'))
            signed_corr = z['distance'].rank(method='average').corr(z['return'].rank(method='average'))
            p = permutation_pvalue(q1['return'], q4['return'], rng)
            p_signed = permutation_pvalue(s1['return'], s4['return'], rng)
            row = {
                'entity': entity, 'window': int(window), 'segment': segment, 'n': int(len(z)),
                'spearman_abs_distance_vs_return': float(corr),
                'spearman_signed_distance_vs_return': float(signed_corr),
                'closest_q_n': int(len(q1)), 'farthest_q_n': int(len(q4)),
                'closest_q_win_rate': float((q1['return'] > 0).mean()),
                'farthest_q_win_rate': float((q4['return'] > 0).mean()),
                'win_rate_diff_closest_minus_farthest': float((q1['return'] > 0).mean() - (q4['return'] > 0).mean()),
                'closest_q_avg_return': float(q1['return'].mean()),
                'farthest_q_avg_return': float(q4['return'].mean()),
                'avg_return_diff_closest_minus_farthest': float(q1['return'].mean() - q4['return'].mean()),
                'closest_q_pf': pf_from_pnl(q1['pnl']), 'farthest_q_pf': pf_from_pnl(q4['pnl']),
                'closest_q_avg_mfe': float(q1['mfe'].mean()), 'farthest_q_avg_mfe': float(q4['mfe'].mean()),
                'closest_q_avg_mae': float(q1['mae'].mean()), 'farthest_q_avg_mae': float(q4['mae'].mean()),
                'permutation_p_avg_return_diff': float(p),
                'lowest_signed_q_n': int(len(s1)), 'highest_signed_q_n': int(len(s4)),
                'lowest_signed_q_win_rate': float((s1['return'] > 0).mean()),
                'highest_signed_q_win_rate': float((s4['return'] > 0).mean()),
                'lowest_signed_q_avg_return': float(s1['return'].mean()),
                'highest_signed_q_avg_return': float(s4['return'].mean()),
                'avg_return_diff_lowest_minus_highest_signed': float(s1['return'].mean() - s4['return'].mean()),
                'permutation_p_signed_return_diff': float(p_signed),
                'above_zone_n': int(len(above)),
                'above_zone_win_rate': float((above['return'] > 0).mean()) if len(above) else np.nan,
                'not_above_zone_win_rate': float((not_above['return'] > 0).mean()) if len(not_above) else np.nan,
                'above_zone_avg_return': float(above['return'].mean()) if len(above) else np.nan,
                'not_above_zone_avg_return': float(not_above['return'].mean()) if len(not_above) else np.nan,
                'inside_zone_n': int(z['entry_inside_zone'].sum()),
                'inside_zone_win_rate': float((z.loc[z['entry_inside_zone'], 'return'] > 0).mean()) if z['entry_inside_zone'].any() else np.nan,
                'outside_zone_win_rate': float((z.loc[~z['entry_inside_zone'], 'return'] > 0).mean()) if (~z['entry_inside_zone']).any() else np.nan,
                'inside_zone_avg_return': float(z.loc[z['entry_inside_zone'], 'return'].mean()) if z['entry_inside_zone'].any() else np.nan,
                'outside_zone_avg_return': float(z.loc[~z['entry_inside_zone'], 'return'].mean()) if (~z['entry_inside_zone']).any() else np.nan,
            }
            summary_rows.append(row)
            detail[f"{entity}_{window}_{segment}"] = {'close_quartiles': q}

    summary = pd.DataFrame(summary_rows).sort_values(['segment','entity','window'])
    summary.to_csv(out / 'institutional_distance_summary.csv', index=False)
    ranked = summary.copy()
    ranked['diagnostic_score'] = ranked['avg_return_diff_closest_minus_farthest'].fillna(0) + 0.25 * ranked['win_rate_diff_closest_minus_farthest'].fillna(0)
    best = ranked.sort_values('diagnostic_score', ascending=False).head(12)

    coverage_by_entity = {}
    for entity, z in feat.groupby('entity'):
        coverage_by_entity[entity] = {
            'unique_trades': int(z['trade_id'].nunique()),
            'R7_unique_trades': int(z.loc[z['strategy']=='R7','trade_id'].nunique()),
            'R05_unique_trades': int(z.loc[z['strategy']=='R05','trade_id'].nunique()),
        }

    result = {
        'study': 'R10 MAX institutional accumulation distance overlay, research only',
        'formal_commit': '3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9',
        'completed_r10_trades': int(len(trades)),
        'matched_trades': int(merged['signal_date'].notna().sum()),
        'institutional_columns': available_inst,
        'completed_trade_counts': trades['strategy'].value_counts().to_dict(),
        'feature_trade_counts': feat.drop_duplicates('trade_id')['strategy'].value_counts().to_dict(),
        'coverage_by_entity': coverage_by_entity,
        'method': {
            'causality': 'for each T+1 entry, accumulation features use institutional and price data only through signal day T',
            'cost_proxy': 'positive net-buy weighted causal-adjusted typical price; not actual institutional cost',
            'windows': windows,
            'groups': 'distance quartiles formed only within already-selected R10 trades; Q1=closest, Q4=farthest',
            'warning': 'research overlay; no R10 signal, ranking, sizing, or exit rule changed'
        },
        'top_diagnostic_contrasts': best.replace({np.nan: None}).to_dict('records'),
        'detail': detail,
    }
    (out / 'institutional_distance_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print('INSTITUTIONAL_COLUMNS', available_inst)
    print('FEATURE_ROWS', len(feat), 'UNIQUE_TRADES', feat.trade_id.nunique())
    print('\n=== CLOSEST Q1 MINUS FARTHEST Q4: diagnostic summary ===')
    show = summary[['entity','window','segment','n','closest_q_win_rate','farthest_q_win_rate',
                    'win_rate_diff_closest_minus_farthest','closest_q_avg_return','farthest_q_avg_return',
                    'avg_return_diff_closest_minus_farthest','closest_q_pf','farthest_q_pf',
                    'spearman_abs_distance_vs_return','spearman_signed_distance_vs_return',
                    'permutation_p_avg_return_diff','avg_return_diff_lowest_minus_highest_signed',
                    'permutation_p_signed_return_diff','above_zone_avg_return','not_above_zone_avg_return']]
    print(show.to_string(index=False))

if __name__ == '__main__':
    main()
