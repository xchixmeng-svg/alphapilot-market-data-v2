"""Real-history cases for lessons 64-100 of 《從菜雞到股神》.

Every case is built from real Taiwan market data and carries `asof` (the decision date) plus, for each
non-price fact, the date it became public (`available`). Nothing after `asof` is ever put in a case's
visible part; the future is kept in a separate `reveal` block that the game only shows after the
player has committed to an answer.

Sources: repo parquet (OHLCV + institutional 2020-2025), FinMind (TAIEX, industry, monthly revenue,
financial statements, dividends/ex-rights, PER, foreign holding, news).
Publication rules used when a source has no exact timestamp (conservative = never earlier than real):
  monthly revenue  -> 10th of the following month (legal deadline)
  quarterly report -> Q1 05-15, Q2 08-14, Q3 11-14, Q4 03-31 of next year (legal deadlines)
"""
import json, os, sys, time, math, hashlib, traceback
import requests
import pandas as pd
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else 'out'
os.makedirs(OUT, exist_ok=True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
H = os.path.join(ROOT, 'data/history/2020-2025')
LOG = []
import atexit
atexit.register(lambda: open(os.path.join(OUT, 'cases100.log'), 'w').write('\n'.join(LOG)))
def _hook(t, v, tb): LOG.append('FATAL ' + ''.join(traceback.format_exception(t, v, tb))[-3000:]); sys.__excepthook__(t, v, tb)
sys.excepthook = _hook
def log(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)

# ───────────────────────── data ─────────────────────────
o = pd.concat([pd.read_parquet(os.path.join(H, f'ohlcv_{y}.parquet')) for y in range(2020, 2026)], ignore_index=True)
o['code'] = o['code'].astype(str)
o = o[o['close'] > 0].copy()
o['date'] = o['date'].astype(int)
EQ = o[o['code'].str.fullmatch(r'[1-9]\d{3}')].copy()           # common stocks (4 digits, not ETF 00xx)
EQ.sort_values(['code', 'date'], inplace=True)
EQ['prev'] = EQ.groupby('code')['close'].shift(1)
EQ['ret'] = EQ['close'] / EQ['prev'] - 1
EQ['value'] = EQ['volume'] * EQ['close']
DATES = sorted(EQ['date'].unique())
DPOS = {d: i for i, d in enumerate(DATES)}
NAME = EQ.groupby('code')['name'].last().to_dict()
log('ohlcv equities', len(EQ), 'codes', EQ['code'].nunique(), 'dates', DATES[0], DATES[-1])

PIV = {k: EQ.pivot(index='date', columns='code', values=k) for k in ['open', 'high', 'low', 'close', 'volume', 'value', 'ret']}
inst = pd.read_parquet(os.path.join(H, 'institutional_2020_2025.parquet'))
inst['code'] = inst['code'].astype(str); inst['date'] = inst['date'].astype(int)
INST = {k: inst.pivot_table(index='date', columns='code', values=k, aggfunc='sum') for k in ['foreign_net', 'trust_net', 'dealer_net']}

FM = 'https://api.finmindtrade.com/api/v4/data'
TOKEN = os.environ.get('FINMIND_TOKEN', '').strip()
NREQ = [0]
def fm(ds, sid=None, a=None, b=None, tries=6):
    p = {'dataset': ds}
    if sid: p['data_id'] = sid
    if a: p['start_date'] = a
    if b: p['end_date'] = b
    if TOKEN: p['token'] = TOKEN
    for k in range(tries):
        try:
            NREQ[0] += 1
            r = requests.get(FM, params=p, timeout=60)
            j = r.json()
            if r.status_code == 200 and j.get('msg') == 'success':
                time.sleep(0.4 if TOKEN else 1.3)
                return j.get('data') or []
            log('finmind', ds, sid, a, r.status_code, str(j.get('msg'))[:120])
            if r.status_code in (402, 429): time.sleep(90 * (k + 1)); continue
            return []
        except Exception as e:
            log('finmind error', ds, sid, e); time.sleep(5)
    return []

def ymd(d): s = str(d); return f'{s[:4]}-{s[4:6]}-{s[6:]}'
def dint(s): return int(str(s)[:10].replace('-', ''))

tx = fm('TaiwanStockPrice', 'TAIEX', '2019-12-01', '2025-12-31')
TAIEX = pd.Series({dint(r['date']): r['close'] for r in tx}).sort_index()
log('taiex rows', len(TAIEX))
info = fm('TaiwanStockInfo')
IND = {}
for r in sorted(info, key=lambda r: r.get('date') or ''):
    c = str(r['stock_id'])
    if r.get('industry_category') and r['industry_category'] not in ('ETF', '大盤', 'Index', '上櫃指數股票型基金(ETF)', '受益證券', '存託憑證'):
        IND[c] = r['industry_category']
log('industry map', len(IND))

# breadth per day from all common stocks
up = (PIV['ret'] > 0.0005).sum(axis=1); dn = (PIV['ret'] < -0.0005).sum(axis=1)
lim_up = ((PIV['ret'] > 0.095) & (PIV['close'] >= PIV['high'])).sum(axis=1)
lim_dn = ((PIV['ret'] < -0.095) & (PIV['close'] <= PIV['low'])).sum(axis=1)
BREADTH = pd.DataFrame({'up': up, 'down': dn, 'limitUp': lim_up, 'limitDown': lim_dn})

# equal-weight industry index (daily mean return of members with value >= 5m that day)
MEMBERS = {}
for c, i in IND.items():
    if c in PIV['close'].columns: MEMBERS.setdefault(i, []).append(c)
IND_IDX = {}
for i, cs in MEMBERS.items():
    if len(cs) < 6: continue
    r = PIV['ret'][cs].where(PIV['value'][cs] >= 5e6)
    IND_IDX[i] = (1 + r.mean(axis=1).fillna(0)).cumprod() * 100
log('industry indexes', len(IND_IDX), sorted(((len(v), k) for k, v in MEMBERS.items()), reverse=True)[:25])

# ───────────────────────── helpers ─────────────────────────
def rnd(x, n=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))): return None
    return round(float(x), n)
def pos(d): return DPOS.get(d)
def window(end_d, n):
    k = pos(end_d); return DATES[max(0, k - n + 1):k + 1]
def after(d, n):
    k = pos(d); return DATES[k + 1:k + 1 + n]
def ser(df, code, dates): return [rnd(df[code].get(d)) if code in df.columns else None for d in dates]
def norm(vals):
    b = next((v for v in vals if v), None)
    return [rnd(v / b * 100) if v and b else None for v in vals]
def tx_ser(dates): return [rnd(TAIEX.get(d)) for d in dates]
def ret(series_get, d0, d1):
    a, b = series_get(d0), series_get(d1)
    return (b / a - 1) if a and b else None
def cret(code, d0, d1): return ret(lambda d: PIV['close'][code].get(d), d0, d1)
def tret(d0, d1): return ret(lambda d: TAIEX.get(d), d0, d1)
def iret(ind, d0, d1): return ret(lambda d: IND_IDX[ind].get(d), d0, d1)
def candles(code, dates, vol=True):
    c = {'d': [int(x) for x in dates], 'o': ser(PIV['open'], code, dates), 'h': ser(PIV['high'], code, dates),
         'l': ser(PIV['low'], code, dates), 'c': ser(PIV['close'], code, dates)}
    if vol: c['v'] = [int(x / 1000) if x == x and x is not None else None for x in (PIV['volume'][code].get(d) for d in dates)]
    return c
VALN = {}
def liquid(code, d, n=20, min_value=3e7):
    if n not in VALN: VALN[n] = PIV['value'].rolling(n, min_periods=n).mean()
    try: v = VALN[n].at[d, code]
    except KeyError: return False
    return v == v and v >= min_value
def seeded(items, key):   # deterministic spread across years: sort by hash
    return sorted(items, key=lambda x: hashlib.md5((key + str(x)).encode()).hexdigest())
def pick_spread(cands, k, key):
    out, years = [], set()
    for c in seeded(cands, key):
        y = str(c[1])[:4] if isinstance(c, tuple) else ''
        if y in years and len(cands) > k * 2: continue
        out.append(c); years.add(y)
        if len(out) >= k: break
    return out

CASES = {}
def put(lesson, case):
    CASES.setdefault(str(lesson), []).append(case)

VALID_DATES = [d for d in DATES if d >= 20200801 and d <= 20251031]   # room for 120-day history and reveal

def safe(fn):
    try:
        fn(); log('OK', fn.__name__, {k: len(v) for k, v in CASES.items()})
    except Exception:
        log('FAILED', fn.__name__, traceback.format_exc()[-1500:])

# ───────────────────────── Module 9 ─────────────────────────
def m09():
    # 64 market cap: real shares issued (foreign holding table) × close
    for code in ['2330', '2317', '1101']:
        d = 20240628
        sh = fm('TaiwanStockShareholding', code, ymd(d), ymd(d))
        if not sh: continue
        n = sh[0]['NumberOfSharesIssued']; px = PIV['close'][code].get(d)
        put(64, {'asof': d, 'code': code, 'name': NAME.get(code), 'shares': n, 'close': px, 'cap': rnd(n * px / 1e8, 0)})
    # 65 lots: real closes for budget questions
    for code in ['2330', '2884', '2603', '3008']:
        d = 20250102
        put(65, {'asof': d, 'code': code, 'name': NAME.get(code), 'close': PIV['close'][code].get(d)})
    # 68 locked limit-up / limit-down: one-price days (open=high=low=close) at ±10%
    lu, ld = [], []
    for d in VALID_DATES[::3]:
        r = PIV['ret'].loc[d]; op, hi, lo, cl = PIV['open'].loc[d], PIV['high'].loc[d], PIV['low'].loc[d], PIV['close'].loc[d]
        one = (op == hi) & (hi == lo) & (lo == cl)
        for c in r[(r > 0.095) & one].index[:3]:
            if liquid(c, d, 20, 1e7): lu.append((c, d))
        for c in r[(r < -0.095) & one].index[:3]:
            if liquid(c, d, 20, 1e7): ld.append((c, d))
    for kind, arr in (('limitUp', lu), ('limitDown', ld)):
        for c, d in pick_spread(arr, 2, 'lim' + kind):
            w = window(d, 15) + after(d, 5)
            put(68, {'kind': kind, 'asof': d, 'code': c, 'name': NAME.get(c), 'chart': candles(c, w), 'mark': int(d),
                     'volLots': int(PIV['volume'][c].get(d) / 1000), 'avgVolLots': int(PIV['volume'][c].reindex(window(d, 21)[:-1]).mean() / 1000)})
    # 70 ex-dividend / ex-rights: real dividend results + raw prices
    for code in ['2330', '2412', '2882', '1216', '2886', '2002']:
        rows = fm('TaiwanStockDividendResult', code, '2021-01-01', '2025-09-30')
        rows = [r for r in rows if dint(r['date']) in DPOS and r.get('stock_and_cache_dividend')]
        rows.sort(key=lambda r: -float(r['stock_and_cache_dividend']) / float(r['before_price']))
        for r in rows[:1]:
            d = dint(r['date']); w = window(d, 15) + after(d, 10)
            put(70, {'asof': d, 'code': code, 'name': NAME.get(code), 'kind': r['stock_or_cache_dividend'], 'before': r['before_price'],
                     'reference': r['reference_price'], 'dividend': r['stock_and_cache_dividend'], 'chart': candles(code, w), 'mark': d})

# ───────────────────────── Module 10 ─────────────────────────
def m10():
    t = TAIEX
    ma60 = t.rolling(60).mean()
    envs = {'up': [], 'down': [], 'range': []}
    for d in VALID_DATES[::5]:
        w = window(d, 80)
        if len(w) < 80 or t.get(w[0]) is None: continue
        r60 = tret(w[-60], d); sl = (ma60.get(d) / ma60.get(w[-20]) - 1) if ma60.get(w[-20]) else 0
        vals = t.reindex(w[-60:]); width = vals.max() / vals.min() - 1
        if r60 > 0.10 and sl > 0.02: envs['up'].append(('TAIEX', d))
        elif r60 < -0.10 and sl < -0.02: envs['down'].append(('TAIEX', d))
        elif abs(r60) < 0.03 and width < 0.09 and abs(sl) < 0.01: envs['range'].append(('TAIEX', d))
    for k, arr in envs.items():
        for _, d in pick_spread(arr, 2, 'env' + k):
            w = window(d, 80)
            put(71, {'env': k, 'asof': d, 'line': tx_ser(w), 'd': [int(x) for x in w], 'reveal': {'after': tx_ser(after(d, 20))}})
    # 72 relative strength vs TAIEX (20 days)
    rs = {'laggard': [], 'leader': [], 'defensive': []}
    for d in VALID_DATES[::7]:
        w = window(d, 21); tr = tret(w[0], d)
        if tr is None: continue
        r = PIV['close'].loc[d] / PIV['close'].loc[w[0]] - 1
        for c in r.dropna().index:
            if not liquid(c, d, 20, 5e7): continue
            x = r[c]
            if tr >= 0.05 and 0 < x <= tr - 0.03: rs['laggard'].append((c, d))
            elif tr <= 0.02 and x >= tr + 0.08 and x <= 0.2: rs['leader'].append((c, d))
            elif tr <= -0.05 and -0.01 <= x <= 0.02: rs['defensive'].append((c, d))
        if all(len(v) > 40 for v in rs.values()): break
    for k, arr in rs.items():
        for c, d in pick_spread(arr, 2, 'rs' + k):
            w = window(d, 21)
            put(72, {'kind': k, 'asof': d, 'code': c, 'name': NAME.get(c), 'd': [int(x) for x in w], 'stock': norm(ser(PIV['close'], c, w)),
                     'market': norm(tx_ser(w)), 'stockRet': rnd(cret(c, w[0], d) * 100, 1), 'marketRet': rnd(tret(w[0], d) * 100, 1)})
    # 73 stock vs industry vs market
    sc = {'withSector': [], 'ownStrength': [], 'lagSector': []}
    for d in VALID_DATES[::9]:
        w = window(d, 21)
        for ind, idx in IND_IDX.items():
            ir = iret(ind, w[0], d)
            if ir is None: continue
            for c in MEMBERS[ind]:
                x = cret(c, w[0], d)
                if x is None or not liquid(c, d, 20, 5e7): continue
                if ir >= 0.08 and abs(x - ir) <= 0.02: sc['withSector'].append((c, d))
                elif ir <= 0.01 and x >= 0.10 and x <= 0.25: sc['ownStrength'].append((c, d))
                elif ir >= 0.10 and 0 <= x <= ir - 0.07: sc['lagSector'].append((c, d))
        if all(len(v) > 40 for v in sc.values()): break
    for k, arr in sc.items():
        for c, d in pick_spread(arr, 2, 'sec' + k):
            w = window(d, 21); ind = IND[c]
            put(73, {'kind': k, 'asof': d, 'code': c, 'name': NAME.get(c), 'industry': ind, 'd': [int(x) for x in w],
                     'stock': norm(ser(PIV['close'], c, w)), 'sector': norm([rnd(IND_IDX[ind].get(x)) for x in w]), 'market': norm(tx_ser(w)),
                     'stockRet': rnd(cret(c, w[0], d) * 100, 1), 'sectorRet': rnd(iret(ind, w[0], d) * 100, 1), 'marketRet': rnd(tret(w[0], d) * 100, 1)})
    # 74 persistent relative strength vs one-day spike
    pairs = []
    for d in VALID_DATES[::11]:
        w = window(d, 41); tr = tret(w[0], d)
        if tr is None: continue
        r = PIV['close'].loc[d] / PIV['close'].loc[w[0]] - 1
        mx = PIV['ret'].reindex(w[1:]).max()
        spike = [c for c in r.dropna().index if mx.get(c, 0) > 0.095 and r[c] < tr - 0.03 and liquid(c, d, 40, 5e7)]
        steady = [c for c in r.dropna().index if mx.get(c, 1) < 0.05 and r[c] > tr + 0.12 and liquid(c, d, 40, 5e7)]
        if spike and steady: pairs.append((seeded(spike, str(d))[0], d, seeded(steady, str(d))[0]))
        if len(pairs) > 20: break
    for a, d, b in pick_spread(pairs, 3, 'persist'):
        w = window(d, 41)
        def relline(c): return [rnd(PIV['close'][c].get(x) / PIV['close'][c].get(w[0]) / (TAIEX.get(x) / TAIEX.get(w[0])) * 100) for x in w]
        put(74, {'asof': d, 'd': [int(x) for x in w], 'spike': {'code': a, 'name': NAME.get(a), 'rel': relline(a), 'maxDay': rnd(PIV['ret'][a].reindex(w[1:]).max() * 100, 1), 'ret': rnd(cret(a, w[0], d) * 100, 1)},
                 'steady': {'code': b, 'name': NAME.get(b), 'rel': relline(b), 'maxDay': rnd(PIV['ret'][b].reindex(w[1:]).max() * 100, 1), 'ret': rnd(cret(b, w[0], d) * 100, 1)},
                 'marketRet': rnd(tret(w[0], d) * 100, 1)})
    # 75 breadth vs index
    br = {'narrowUp': [], 'hiddenStrength': [], 'broadUp': []}
    for d in VALID_DATES:
        k = pos(d); tr = TAIEX.get(d) / TAIEX.get(DATES[k - 1]) - 1 if TAIEX.get(DATES[k - 1]) else None
        if tr is None or d not in BREADTH.index: continue
        u, dn_ = BREADTH.loc[d, 'up'], BREADTH.loc[d, 'down']
        if tr >= 0.008 and dn_ > u * 1.15: br['narrowUp'].append(('B', d))
        elif tr <= -0.008 and u > dn_ * 1.15: br['hiddenStrength'].append(('B', d))
        elif tr >= 0.015 and u > (u + dn_) * 0.72: br['broadUp'].append(('B', d))
    for k, arr in br.items():
        for _, d in pick_spread(arr, 2, 'br' + k):
            w = window(d, 20)
            put(75, {'kind': k, 'asof': d, 'd': [int(x) for x in w], 'index': tx_ser(w), 'up': [int(BREADTH.loc[x, 'up']) for x in w], 'down': [int(BREADTH.loc[x, 'down']) for x in w],
                     'limitUp': int(BREADTH.loc[d, 'limitUp']), 'limitDown': int(BREADTH.loc[d, 'limitDown']),
                     'indexChg': rnd((TAIEX.get(d) / TAIEX.get(DATES[pos(d) - 1]) - 1) * 100, 2)})
    # 76 share volume vs traded value
    for d in [20230601, 20240603, 20250602]:
        if d not in DPOS: d = DATES[min(len(DATES) - 1, pos(max(x for x in DATES if x <= d)))]
        vol = PIV['volume'].loc[d].dropna(); val = PIV['value'].loc[d].dropna(); px = PIV['close'].loc[d]
        cheap = [c for c in vol.sort_values(ascending=False).index[:80] if px[c] < 25 and c in val.index][:1]
        rich = [c for c in val.sort_values(ascending=False).index[:40] if px[c] > 300 and vol[c] < vol[cheap[0]] * 0.5][:1] if cheap else []
        thin = [c for c in val.sort_values().index if px[c] > 100 and liquid(c, d, 20, 1e6) and not liquid(c, d, 20, 1.5e7)][:1]
        if cheap and rich:
            def row(c): return {'code': c, 'name': NAME.get(c), 'close': rnd(px[c]), 'volLots': int(vol[c] / 1000), 'valueYi': rnd(val[c] / 1e8, 2),
                                'avg20ValueYi': rnd(PIV['value'][c].reindex(window(d, 20)).mean() / 1e8, 2)}
            put(76, {'asof': d, 'cheap': row(cheap[0]), 'rich': row(rich[0]), 'thin': row(thin[0]) if thin else None})
    # 77 top-down: market / sector / stock alignment
    td = {'aligned': [], 'headwind': [], 'sectorDrag': []}
    for d in VALID_DATES[::13]:
        w = window(d, 41); m40 = tret(w[0], d); w20 = window(d, 21)
        if m40 is None: continue
        for ind in IND_IDX:
            s20 = iret(ind, w20[0], d)
            if s20 is None: continue
            for c in MEMBERS[ind][:60]:
                x = cret(c, w20[0], d)
                if x is None or not liquid(c, d, 20, 5e7): continue
                ma20 = PIV['close'][c].reindex(w20).mean(); dist = PIV['close'][c].get(d) / ma20 - 1
                if m40 > 0.05 and s20 > tret(w20[0], d) + 0.02 and x > s20 and 0 < dist < 0.08: td['aligned'].append((c, d))
                elif m40 < -0.06 and x > 0.08: td['headwind'].append((c, d))
                elif m40 > 0.05 and s20 < -0.04 and x > 0.06: td['sectorDrag'].append((c, d))
        if all(len(v) > 30 for v in td.values()): break
    for k, arr in td.items():
        for c, d in pick_spread(arr, 2, 'td' + k):
            w = window(d, 41); ind = IND[c]
            put(77, {'kind': k, 'asof': d, 'code': c, 'name': NAME.get(c), 'industry': ind, 'd': [int(x) for x in w],
                     'market': tx_ser(w), 'sector': [rnd(IND_IDX[ind].get(x)) for x in w], 'stock': candles(c, w),
                     'marketRet40': rnd(tret(w[0], d) * 100, 1), 'sectorRet20': rnd(iret(ind, window(d, 21)[0], d) * 100, 1), 'stockRet20': rnd(cret(c, window(d, 21)[0], d) * 100, 1),
                     'reveal': {'stock': candles(c, after(d, 20)), 'ret20': rnd(cret(c, d, after(d, 20)[-1]) * 100, 1) if after(d, 20) else None}})

# ───────────────────────── Module 11 ─────────────────────────
NEWS_CACHE = {}
def news_on(code, d):
    k = (code, d)
    if k not in NEWS_CACHE:
        rows = fm('TaiwanStockNews', code, ymd(d))
        NEWS_CACHE[k] = [{'time': str(r.get('date')), 'title': r.get('title'), 'source': r.get('source'), 'link': r.get('link')} for r in rows if r.get('title')]
    return NEWS_CACHE[k]

def m11():
    # 78 theme: industry surge; members already-run / just-started / flat
    themes = []
    for d in VALID_DATES[::15]:
        w60 = window(d, 61)
        for ind, idx in IND_IDX.items():
            r40 = iret(ind, window(d, 41)[0], d)
            if r40 is None or r40 < 0.18: continue
            done, start, flat = [], [], []
            for c in MEMBERS[ind]:
                if not liquid(c, d, 20, 3e7): continue
                r60 = cret(c, w60[0], d); r10 = cret(c, window(d, 11)[0], d); pre = cret(c, w60[0], window(d, 11)[0])
                if None in (r60, r10, pre): continue
                if r60 > 0.6 and r10 < 0.05: done.append(c)
                elif pre < 0.08 and r10 > 0.12: start.append(c)
                elif abs(r60) < 0.06: flat.append(c)
            if done and start and flat: themes.append((ind, d, done[0], start[0], flat[0]))
        if len(themes) > 12: break
    for ind, d, a, b, c in pick_spread(themes, 3, 'theme'):
        w = window(d, 61)
        put(78, {'asof': d, 'industry': ind, 'd': [int(x) for x in w], 'sector': norm([rnd(IND_IDX[ind].get(x)) for x in w]),
                 'stocks': [{'role': r, 'code': x, 'name': NAME.get(x), 'line': norm(ser(PIV['close'], x, w)), 'ret60': rnd(cret(x, w[0], d) * 100, 1)} for r, x in (('done', a), ('start', b), ('flat', c))],
                 'reveal': {x: rnd(cret(x, d, after(d, 20)[-1]) * 100, 1) for x in (a, b, c)}})
    # 79 + 82 news timing: big move days of liquid stocks, with real news timestamps
    big = []
    for d in VALID_DATES[::2]:
        r = PIV['ret'].loc[d]
        for c in r[(r > 0.07)].index:
            if liquid(c, d, 20, 1e8): big.append((c, d))
    log('big move days', len(big))
    got79, got82 = 0, 0
    tries = 0
    for c, d in seeded(big, 'news'):
        tries += 1
        if (got79 >= 3 and got82 >= 3) or tries > 45: break
        w10 = window(d, 11); pre = cret(c, w10[0], window(d, 2)[0])
        nxt = after(d, 1)
        if not nxt: continue
        today = news_on(c, d); tomorrow = news_on(c, nxt[0])
        late = [n for n in today if n['time'][11:16] >= '13:35'] + [n for n in tomorrow if n['time'][11:16] < '09:00']
        early = [n for n in today if '00:00' < n['time'][11:16] < '09:00']
        if got79 < 3 and late:
            w = window(d, 15) + after(d, 3)
            put(79, {'asof': d, 'code': c, 'name': NAME.get(c), 'chart': candles(c, w), 'mark': d, 'news': late[0], 'early': early[0] if early else None,
                     'moveDay': rnd(PIV['ret'][c].get(d) * 100, 1)})
            got79 += 1
        if got82 < 3 and pre is not None and pre > 0.15 and (today or late):
            w = window(d, 25) + after(d, 10)
            n = (today or late)[0]
            put(82, {'asof': d, 'code': c, 'name': NAME.get(c), 'chart': candles(c, w), 'mark': d, 'news': n, 'preRet10': rnd(pre * 100, 1),
                     'reveal': {'ret10': rnd(cret(c, d, after(d, 10)[-1]) * 100, 1) if len(after(d, 10)) == 10 else None}})
            got82 += 1
    # 83 institutional: one-day spike that failed vs persistent buying
    spikes, persist = [], []
    for d in VALID_DATES[::6]:
        if d not in INST['foreign_net'].index: continue
        fn = INST['foreign_net'].loc[d]
        for c in fn.dropna().index:
            if c not in PIV['close'].columns or not liquid(c, d, 20, 2e8): continue
            hist = INST['foreign_net'][c].reindex(window(d, 61)[:-1]).abs().mean()
            if hist and fn[c] > 6 * hist and len(after(d, 10)) == 10 and cret(c, d, after(d, 10)[-1]) < -0.05: spikes.append((c, d))
        tn = INST['trust_net'].reindex(window(d, 12))
        cnt = (tn > 0).sum()
        for c in cnt[cnt >= 10].index:
            if c in PIV['close'].columns and liquid(c, d, 20, 1e8):
                x = cret(c, window(d, 12)[0], d)
                if x and 0.05 < x < 0.25: persist.append((c, d))
        if len(spikes) > 20 and len(persist) > 20: break
    for kind, arr in (('spike', spikes), ('persist', persist)):
        for c, d in pick_spread(arr, 2, 'inst' + kind):
            w = window(d, 25)
            put(83, {'kind': kind, 'asof': d, 'code': c, 'name': NAME.get(c), 'chart': candles(c, w),
                     'foreign': [int(v / 1000) if v == v else 0 for v in INST['foreign_net'][c].reindex(w)], 'trust': [int(v / 1000) if v == v else 0 for v in INST['trust_net'][c].reindex(w)],
                     'reveal': {'chart': candles(c, after(d, 10)), 'ret10': rnd(cret(c, d, after(d, 10)[-1]) * 100, 1) if after(d, 10) else None}})
    # 84 ratio: big lots small share vs small lots big share, same day
    for d in [20230818, 20240816, 20250815]:
        d = max(x for x in DATES if x <= d)
        fn = INST['foreign_net'].loc[d].dropna(); vol = PIV['volume'].loc[d]
        bigc = [c for c in fn.sort_values(ascending=False).index[:30] if c in vol.index and vol[c] > 0 and fn[c] / vol[c] < 0.08]
        smallc = [c for c in fn.sort_values(ascending=False).index[:400] if c in vol.index and vol[c] > 0 and 0.25 < fn[c] / vol[c] < 0.6 and fn[c] < (fn[bigc[0]] if bigc else 0) * 0.3 and liquid(c, d, 20, 2e7)]
        if bigc and smallc:
            def row(c): return {'code': c, 'name': NAME.get(c), 'netLots': int(fn[c] / 1000), 'volLots': int(vol[c] / 1000), 'pct': rnd(fn[c] / vol[c] * 100, 1),
                                'net5': int(INST['foreign_net'][c].reindex(window(d, 5)).sum() / 1000)}
            put(84, {'asof': d, 'big': row(bigc[0]), 'small': row(smallc[0])})
    # 85 money flow: industry turnover share up without price confirmation / after a big run
    tot = PIV['value'].sum(axis=1)
    share = {i: PIV['value'][cs].sum(axis=1) / tot * 100 for i, cs in MEMBERS.items() if i in IND_IDX}
    mf = {'noConfirm': [], 'lateChase': [], 'confirmed': []}
    for d in VALID_DATES[::8]:
        w = window(d, 41)
        for i, s in share.items():
            a, b = s.reindex(w[:10]).mean(), s.reindex(w[-10:]).mean()
            if not a or b < a * 1.6 or b < 1.5: continue
            p = iret(i, w[0], d); pre = iret(i, window(w[0], 61)[0], w[0])
            if p is None or pre is None: continue
            if p < 0.01: mf['noConfirm'].append((i, d))
            elif pre > 0.35 and p > 0.05: mf['lateChase'].append((i, d))
            elif 0.06 < p < 0.25 and pre < 0.1: mf['confirmed'].append((i, d))
    for k, arr in mf.items():
        for i, d in pick_spread(arr, 1, 'mf' + k):
            w = window(d, 41)
            put(85, {'kind': k, 'asof': d, 'industry': i, 'd': [int(x) for x in w], 'share': [rnd(share[i].get(x)) for x in w], 'sector': norm([rnd(IND_IDX[i].get(x)) for x in w]),
                     'priorRun60': rnd(iret(i, window(w[0], 61)[0], w[0]) * 100, 1), 'reveal': {'sectorRet20': rnd(iret(i, d, after(d, 20)[-1]) * 100, 1) if len(after(d, 20)) == 20 else None}})

# ───────────────────────── Module 12 ─────────────────────────
UNIVERSE = ['2330', '2317', '2454', '2308', '2382', '2357', '3231', '2376', '3711', '2303', '2412', '1301', '1303', '2002', '2603',
            '2609', '1216', '2912', '9910', '1101', '2105', '3034', '3037', '8046', '2049', '6415', '3661', '2408', '2409', '6669', '1513', '1605']
QDEAD = {3: '0515', 6: '0814', 9: '1114', 12: '0331'}
def qavail(qdate):   # quarter end 'YYYY-MM-DD' -> int date the report must be public
    y, m = int(qdate[:4]), int(qdate[5:7])
    return int(f'{y + 1}{QDEAD[12]}') if m == 12 else int(f'{y}{QDEAD[m]}')
def ravail(rev_year, rev_month):   # revenue month -> the 10th of next month
    y, m = (rev_year + 1, 1) if rev_month == 12 else (rev_year, rev_month + 1)
    return int(f'{y}{m:02d}10')
FIN = {}
def fin(code):
    if code in FIN: return FIN[code]
    st = fm('TaiwanStockFinancialStatements', code, '2020-01-01', '2025-09-30')
    cf = fm('TaiwanStockCashFlowsStatement', code, '2020-01-01', '2025-09-30')
    bs = fm('TaiwanStockBalanceSheet', code, '2020-01-01', '2025-09-30')
    q = {}
    for rows, pre in ((st, ''), (cf, 'cf_'), (bs, 'bs_')):
        for r in rows:
            q.setdefault(r['date'], {})[pre + r['type']] = r['value']
    FIN[code] = q; return q
REV = {}
def rev(code):
    if code not in REV:
        rows = fm('TaiwanStockMonthRevenue', code, '2019-01-01', '2025-10-31')
        REV[code] = {(int(r['revenue_year']), int(r['revenue_month'])): r['revenue'] for r in rows}
    return REV[code]

def m12():
    q0 = fin('2330'); k0 = sorted(q0)[-1] if q0 else None
    if k0: log('statement keys', sorted(q0[k0].keys()))
    # 86 revenue: MoM drop but YoY up / single-month spike / steady uptrend
    found = {'momDropYoyUp': [], 'spike': [], 'trend': []}
    for code in UNIVERSE:
        R = rev(code)
        for (y, m), v in R.items():
            if y < 2021 or (y, m) > (2025, 8): continue
            ly = R.get((y - 1, m)); pm = R.get((y, m - 1) if m > 1 else (y - 1, 12))
            last6 = [R.get((y if m - k > 0 else y - 1, (m - k - 1) % 12 + 1)) for k in range(6)]
            if not ly or not pm or None in last6: continue
            yoy, mom = v / ly - 1, v / pm - 1
            if mom < -0.12 and yoy > 0.15: found['momDropYoyUp'].append((code, y * 100 + m))
            prev3 = [R.get((y if m - k > 0 else y - 1, (m - k - 1) % 12 + 1)) for k in range(1, 4)]
            nxt = R.get((y, m + 1) if m < 12 else (y + 1, 1))
            if None not in prev3 and nxt and v > 1.6 * max(prev3) and nxt < v * 0.7: found['spike'].append((code, y * 100 + m))
            if all(last6[k] > last6[k + 1] * 0.98 for k in range(5)) and yoy > 0.1: found['trend'].append((code, y * 100 + m))
    for k, arr in found.items():
        for code, ym in pick_spread(arr, 2, 'rev' + k):
            y, m = divmod(ym, 100); R = rev(code)
            months = []
            for kk in range(13, -1, -1):
                yy, mm = (y, m - kk) if m - kk > 0 else (y - 1, m - kk + 12)
                if mm <= 0: yy, mm = yy - 1, mm + 12
                months.append((yy, mm))
            rows = [{'ym': f'{a}-{b:02d}', 'rev': rnd(R.get((a, b), 0) / 1e8, 2), 'avail': ravail(a, b)} for a, b in months]
            if k == 'spike':
                a, b = (y, m + 1) if m < 12 else (y + 1, 1)
                rows.append({'ym': f'{a}-{b:02d}', 'rev': rnd(R.get((a, b), 0) / 1e8, 2), 'avail': ravail(a, b), 'reveal': True})
            put(86, {'kind': k, 'code': code, 'name': NAME.get(code), 'asof': ravail(y, m), 'rows': rows})
    # 81 expectation gap: strong revenue but no rise (already priced in) / weak revenue but no fall
    g81 = []
    for code in UNIVERSE:
        R = rev(code)
        if code not in PIV['close'].columns: continue
        for (y, m), v in R.items():
            ly = R.get((y - 1, m))
            if not ly or y < 2021 or (y, m) > (2025, 9): continue
            yoy = v / ly - 1
            me = max([d for d in DATES if d <= int(f'{y}{m:02d}31')] or [0])
            av = ravail(y, m); da = min([d for d in DATES if d >= av] or [0])
            if not me or not da or me not in DPOS or len(after(da, 10)) < 10: continue
            pre = cret(code, window(me, 21)[0], me); post = cret(code, da, after(da, 10)[-1])
            if pre is None or post is None: continue
            if yoy > 0.3 and pre > 0.12 and post < -0.04: g81.append((code, y * 100 + m, 'goodNoRise'))
            if yoy < -0.15 and pre < -0.10 and post > 0.04: g81.append((code, y * 100 + m, 'badNoFall'))
    kinds = {}
    for code, ym, k in seeded(g81, 'exp81'):
        if kinds.get(k, 0) >= 2: continue
        kinds[k] = kinds.get(k, 0) + 1
        y, m = divmod(ym, 100); R = rev(code); me = max(d for d in DATES if d <= int(f'{y}{m:02d}31')); da = min(d for d in DATES if d >= ravail(y, m))
        w = window(me, 30) + [d for d in DATES if me < d <= da]
        put(81, {'kind': k, 'code': code, 'name': NAME.get(code), 'ym': f'{y}-{m:02d}', 'rev': rnd(R[(y, m)] / 1e8, 2), 'yoy': rnd((R[(y, m)] / R[(y - 1, m)] - 1) * 100, 1),
                 'asof': da, 'chart': candles(code, w), 'monthEnd': me, 'avail': da, 'preRet20': rnd(cret(code, window(me, 21)[0], me) * 100, 1),
                 'reveal': {'chart': candles(code, after(da, 10)), 'ret10': rnd(cret(code, da, after(da, 10)[-1]) * 100, 1)}})
    # 87-91 quarterly statements
    g87, g88, g89, g90, g91 = [], [], [], [], []
    for code in UNIVERSE:
        q = fin(code); ks = sorted(q)
        def gv(dq, *names):
            for n in names:
                if n in q.get(dq, {}): return q[dq][n]
            return None
        for i, dq in enumerate(ks):
            rv, gp, op = gv(dq, 'Revenue'), gv(dq, 'GrossProfit'), gv(dq, 'OperatingIncome')
            if not rv or rv <= 0 or gp is None: continue
            prev = ks[i - 4] if i >= 4 else None
            if prev and gv(prev, 'Revenue') and gv(prev, 'GrossProfit') is not None:
                g_now, g_prev = gp / rv, gv(prev, 'GrossProfit') / gv(prev, 'Revenue')
                rg = rv / gv(prev, 'Revenue') - 1
                if rg > 0.1 and g_now < g_prev - 0.03: g87.append((code, dq, 'volumeNotQuality'))
                if abs(rg) < 0.05 and g_now > g_prev + 0.03: g87.append((code, dq, 'qualityUp'))
            if op is not None and gp / rv > 0.35 and op / rv < 0.10: g88.append((code, dq, 'highGrossLowOp'))
            if op is not None and 0.15 < gp / rv < 0.30 and op / rv > 0.10: g88.append((code, dq, 'leanOps'))
            ni, nonop, pre = gv(dq, 'IncomeAfterTaxes', 'NetIncome'), gv(dq, 'TotalNonoperatingIncomeAndExpense'), gv(dq, 'PreTaxIncome')
            eps, eps_prev = gv(dq, 'EPS'), (gv(ks[i - 1], 'EPS') if i else None)
            if pre and nonop and pre > 0 and nonop / pre > 0.5 and eps and eps_prev and eps_prev > 0 and eps > eps_prev * 1.5: g89.append((code, dq, 'oneOff'))
            if i >= 3:
                last4 = ks[i - 3:i + 1]
                nis = [gv(x, 'IncomeAfterTaxes', 'NetIncome') for x in last4]
                ocf = gv(dq, 'cf_CashFlowsFromOperatingActivities', 'cf_NetCashInflowFromOperatingActivities')
                if None not in nis and all(x > 0 for x in nis) and ocf is not None and ocf < 0 and dq[5:7] == '12': g90.append((code, dq, 'cashGap'))
            ta, tl = gv(dq, 'bs_TotalAssets'), gv(dq, 'bs_Liabilities', 'bs_TotalLiabilities')
            if dq[5:7] == '12' and i >= 3 and ta and tl:
                yq = [k for k in ks[i - 3:i + 1] if k[:4] == dq[:4]]
                opy = sum(gv(k, 'OperatingIncome') or 0 for k in yq) if len(yq) == 4 else None
                fcy = gv(dq, 'cf_InterestExpense', 'cf_InterestPaid')
                if fcy is not None: fcy = abs(fcy)
                if opy and fcy and fcy > 0:
                    cov = opy / fcy; dr = tl / ta
                    rec = (code, dq, 'highDebtCovered' if dr > 0.6 and cov > 15 else 'thinCover' if dr < 0.55 and 0 < cov < 2.5 else None, opy, fcy)
                    if rec[2]: g91.append(rec)
    def fcase(code, dq, kind, lesson, fields):
        q = fin(code); ks = [k for k in sorted(q) if k <= dq][-6:]
        rows = []
        for k in ks:
            row = {'q': k[:7], 'avail': qavail(k)}
            for f, names in fields.items():
                v = None
                for n in names:
                    if n in q[k]: v = q[k][n]; break
                row[f] = rnd(v / 1e8, 2) if v is not None and f not in ('eps',) else rnd(v, 2)
            rows.append(row)
        put(lesson, {'kind': kind, 'code': code, 'name': NAME.get(code), 'asof': qavail(dq), 'rows': rows})
    F_IS = {'rev': ['Revenue'], 'gp': ['GrossProfit'], 'op': ['OperatingIncome'], 'nonop': ['TotalNonoperatingIncomeAndExpense'], 'pre': ['PreTaxIncome'], 'ni': ['IncomeAfterTaxes', 'NetIncome'], 'eps': ['EPS']}
    for lesson, arr in ((87, g87), (88, g88), (89, g89)):
        kinds = {}
        for c, dq, k in seeded(arr, str(lesson)):
            if kinds.get(k, 0) < 2: kinds[k] = kinds.get(k, 0) + 1; fcase(c, dq, k, lesson, F_IS)
    for c, dq, k in pick_spread([(a, b, c_) for a, b, c_ in g90], 2, 'cf'):
        fcase(c, dq, k, 90, dict(F_IS, ocf=['cf_CashFlowsFromOperatingActivities', 'cf_NetCashInflowFromOperatingActivities']))
    kinds = {}
    for c, dq, k, opy, fcy in seeded(g91, 'debt'):
        if kinds.get(k, 0) >= 2: continue
        kinds[k] = kinds.get(k, 0) + 1
        q = fin(c)
        put(91, {'kind': k, 'code': c, 'name': NAME.get(c), 'asof': qavail(dq), 'year': dq[:4], 'assets': rnd(q[dq].get('bs_TotalAssets') / 1e8, 1),
                 'liab': rnd((q[dq].get('bs_Liabilities') or q[dq].get('bs_TotalLiabilities')) / 1e8, 1), 'opYear': rnd(opy / 1e8, 1), 'interestYear': rnd(fcy / 1e8, 2)})
    # 92 valuation: low PER followed by EPS decline (trap) / high PER with growth
    for code in ['2603', '2609', '2615', '2002', '1303', '3661', '6669', '5269', '3711', '2454']:
        per = fm('TaiwanStockPER', code, '2021-01-01', '2025-06-30')
        if not per: continue
        q = fin(code); ks = sorted(q)
        for r in per[::40]:
            d = dint(r['date'])
            if d not in DPOS or not r.get('PER'): continue
            pub = [k for k in ks if qavail(k) <= d]
            fut = [k for k in ks if qavail(k) > d][:4]
            if len(pub) < 4 or len(fut) < 4: continue
            eps_ttm = sum(q[k].get('EPS', 0) for k in pub[-4:]); eps_next = sum(q[k].get('EPS', 0) for k in fut)
            if r['PER'] < 8 and eps_next < eps_ttm * 0.6: put(92, {'kind': 'trap', 'code': code, 'name': NAME.get(code), 'asof': d, 'per': r['PER'], 'pbr': r.get('PBR'), 'close': PIV['close'][code].get(d),
                                                                  'epsTTM': rnd(eps_ttm), 'reveal': {'epsNext4Q': rnd(eps_next), 'ret250': rnd(cret(code, d, after(d, 250)[-1]) * 100, 1) if len(after(d, 250)) == 250 else None}}); break
            if r['PER'] > 35 and eps_next > eps_ttm * 1.4: put(92, {'kind': 'growth', 'code': code, 'name': NAME.get(code), 'asof': d, 'per': r['PER'], 'pbr': r.get('PBR'), 'close': PIV['close'][code].get(d),
                                                                   'epsTTM': rnd(eps_ttm), 'reveal': {'epsNext4Q': rnd(eps_next), 'ret250': rnd(cret(code, d, after(d, 250)[-1]) * 100, 1) if len(after(d, 250)) == 250 else None}}); break

# ───────────────────────── Module 13 ─────────────────────────
def m13():
    # 93/94: same-industry basket vs cross-industry basket over the same year
    for y0, y1 in ((20220103, 20221230), (20240102, 20241231)):
        w = [d for d in DATES if y0 <= d <= y1]
        def top(ind, k):
            cs = [c for c in MEMBERS.get(ind, []) if liquid(c, w[-1], 60, 2e8)]
            return sorted(cs, key=lambda c: -PIV['value'][c].reindex(w).mean())[:k]
        semi = top('半導體業', 5)
        cross = [x[0] for x in [top(i, 1) for i in ('半導體業', '金融保險', '食品工業', '航運業', '鋼鐵工業', '建材營造', '塑膠工業', '觀光餐旅')] if x][:5]
        if len(semi) < 5 or len(cross) < 5: log('m13 basket skipped', y0, len(semi), len(cross)); continue
        def basket(cs):
            r = PIV['ret'][cs].reindex(w)
            corr = r.corr().values; n = len(cs); avgc = (corr.sum() - n) / (n * n - n)
            eq = r.mean(axis=1); worst = eq.idxmin()
            down_all = int((r < 0).all(axis=1).sum())
            return {'codes': cs, 'names': [NAME.get(c) for c in cs], 'avgCorr': rnd(avgc, 2), 'worstDay': int(worst), 'worstDayPct': rnd(eq.min() * 100, 2),
                    'allDownDays': down_all, 'days': len(w), 'line': norm([rnd(x) for x in (1 + eq.fillna(0)).cumprod().reindex(w[::5]) * 100])}
        A, B = basket(semi), basket(cross)
        if A['avgCorr'] > B['avgCorr'] + 0.1 and A['allDownDays'] > B['allDownDays']: put(93, {'from': w[0], 'to': w[-1], 'same': A, 'mixed': B})
        else: log('m13 basket not illustrative', y0, A['avgCorr'], B['avgCorr'], A['allDownDays'], B['allDownDays'])
    # 95 averaging into a broken thesis: support broken then a further fall
    cands = []
    for d in VALID_DATES[::10]:
        w = window(d, 60)
        for c in PIV['close'].columns[::7]:
            if not liquid(c, d, 20, 1e8): continue
            lo = PIV['low'][c].reindex(w[:-5]).min(); cl = PIV['close'][c].get(d)
            if cl and lo and cl < lo * 0.97 and PIV['close'][c].reindex(w[-5:-1]).min() >= lo * 0.99 and len(after(d, 30)) == 30:
                fut = cret(c, d, after(d, 30)[-1])
                if fut is not None and fut < -0.15: cands.append((c, d))
        if len(cands) > 15: break
    for c, d in pick_spread(cands, 3, 'avgdown'):
        w = window(d, 60)
        put(95, {'asof': d, 'code': c, 'name': NAME.get(c), 'chart': candles(c, w), 'support': rnd(PIV['low'][c].reindex(w[:-5]).min()),
                 'reveal': {'chart': candles(c, after(d, 30)), 'ret30': rnd(cret(c, d, after(d, 30)[-1]) * 100, 1)}})
    # 96 lucky chase vs good setup that lost
    lucky, unlucky = [], []
    for d in VALID_DATES[::9]:
        w = window(d, 41)
        for c in PIV['close'].columns[::5]:
            if not liquid(c, d, 20, 1e8) or len(after(d, 20)) < 20: continue
            cl = PIV['close'][c].get(d); ma = PIV['close'][c].reindex(w[-20:]).mean()
            if not cl or not ma: continue
            dist = cl / ma - 1; fut = cret(c, d, after(d, 20)[-1])
            if fut is None: continue
            if dist > 0.22 and fut > 0.12: lucky.append((c, d))
            lo = PIV['low'][c].reindex(w[-15:]).min()
            if 0.0 < dist < 0.05 and cl > PIV['close'][c].get(w[-20]) * 1.03 and lo >= cl * 0.9 and min(PIV['low'][c].reindex(after(d, 20))) < lo * 0.99 and fut < -0.06: unlucky.append((c, d))
        if len(lucky) > 15 and len(unlucky) > 15: break
    for kind, arr in (('luckyChase', lucky), ('goodLoss', unlucky)):
        for c, d in pick_spread(arr, 2, 'proc' + kind):
            w = window(d, 41)
            put(96, {'kind': kind, 'asof': d, 'code': c, 'name': NAME.get(c), 'chart': candles(c, w), 'ma20': rnd(PIV['close'][c].reindex(w[-20:]).mean()),
                     'swingLow': rnd(PIV['low'][c].reindex(w[-15:]).min()), 'reveal': {'chart': candles(c, after(d, 20)), 'ret20': rnd(cret(c, d, after(d, 20)[-1]) * 100, 1)}})

# ───────────────────────── Module 14 (blind cases) ─────────────────────────
def asof_fund(code, d):
    q = fin(code); pub = [k for k in sorted(q) if qavail(k) <= d][-4:]
    R = rev(code); months = sorted([k for k in R if ravail(*k) <= d])[-6:]
    fq = [{'q': k[:7], 'avail': qavail(k), 'rev': rnd(q[k].get('Revenue', 0) / 1e8, 1), 'gm': rnd(q[k].get('GrossProfit', 0) / q[k]['Revenue'] * 100, 1) if q[k].get('Revenue') else None,
           'om': rnd(q[k].get('OperatingIncome', 0) / q[k]['Revenue'] * 100, 1) if q[k].get('Revenue') else None, 'eps': rnd(q[k].get('EPS'))} for k in pub]
    fr = []
    for (y, m) in months:
        ly = R.get((y - 1, m))
        fr.append({'ym': f'{y}-{m:02d}', 'avail': ravail(y, m), 'rev': rnd(R[(y, m)] / 1e8, 1), 'yoy': rnd((R[(y, m)] / ly - 1) * 100, 1) if ly else None})
    return fq, fr

def m14():
    blind = []
    for d in VALID_DATES[::17]:
        if len(after(d, 25)) < 25: continue
        for c in UNIVERSE:
            if c not in PIV['close'].columns or c not in IND or IND[c] not in IND_IDX or not liquid(c, d, 20, 3e8): continue
            w = window(d, 81)
            if len(w) < 81: continue
            r20 = cret(c, window(d, 21)[0], d); ma = PIV['close'][c].reindex(w[-20:]).mean(); dist = PIV['close'][c].get(d) / ma - 1
            fut = cret(c, d, after(d, 20)[-1])
            if r20 is None or fut is None: continue
            tag = 'aligned' if (tret(w[-41], d) or 0) > 0.03 and r20 > 0.03 and dist < 0.08 else 'extended' if dist > 0.15 else 'weak' if r20 < -0.06 else 'mixed'
            blind.append((c, d, tag))
        if len(blind) > 300: break
    log('blind candidates', len(blind))
    chosen = {}
    for c, d, tag in seeded(blind, 'blind14'):
        if chosen.get(tag, 0) >= 3 or any(x['code'] == c for v in CASES.get('98', []) for x in [v]): continue
        chosen[tag] = chosen.get(tag, 0) + 1
        w = window(d, 81); ind = IND[c]; fut = after(d, 25)
        newsd = []
        for x in window(d, 2):
            for n in news_on(c, x)[:3]:
                if n['time'][:10].replace('-', '') <= str(d) and not (n['time'][:10].replace('-', '') == str(d) and n['time'][11:16] > '13:30'):
                    newsd.append(n)
        fq, fr = asof_fund(c, d)
        case = {'tag': tag, 'asof': d, 'code': c, 'name': NAME.get(c), 'industry': ind, 'd': [int(x) for x in w],
                'market': tx_ser(w), 'sector': [rnd(IND_IDX[ind].get(x)) for x in w], 'stock': candles(c, w),
                'foreign': [int(v / 1000) if v == v else 0 for v in INST['foreign_net'][c].reindex(w[-20:])],
                'trust': [int(v / 1000) if v == v else 0 for v in INST['trust_net'][c].reindex(w[-20:])],
                'news': newsd[:4], 'fundQ': fq, 'fundM': fr,
                'reveal': {'stock': candles(c, fut), 'market': tx_ser(fut), 'ret20': rnd(cret(c, d, fut[19]) * 100, 1)}}
        put(98, case)
        if sum(chosen.values()) >= 10: break

for fn in (m09, m10, m11, m12, m13, m14):
    safe(fn)

meta = {'built': time.strftime('%Y-%m-%d %H:%M:%S'), 'finmindRequests': NREQ[0], 'counts': {k: len(v) for k, v in CASES.items()},
        'rules': {'revenueAvailable': '10th of next month', 'quarterAvailable': 'Q1 05-15, Q2 08-14, Q3 11-14, Q4 03-31 next year', 'industry': 'FinMind TaiwanStockInfo latest category (classification as of build date)'}}
def _np(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if math.isnan(o) else float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
json.dump({'meta': meta, 'cases': CASES}, open(os.path.join(OUT, 'cases100.json'), 'w'), ensure_ascii=False, separators=(',', ':'), default=_np)
open(os.path.join(OUT, 'cases100.log'), 'w').write('\n'.join(LOG))
log('DONE', meta)
