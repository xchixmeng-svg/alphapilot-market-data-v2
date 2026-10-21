"""Probe which real historical sources the 100-lesson course can use (run in GitHub Actions)."""
import json, os, sys, time
import requests
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else 'probe_out'
os.makedirs(OUT, exist_ok=True)
H = os.path.join(os.path.dirname(__file__), '../../../../data/history/2020-2025')
rep = {}

def note(k, v):
    rep[k] = v
    print('==', k, json.dumps(v, ensure_ascii=False, default=str)[:1500], flush=True)

try:
    o = pd.read_parquet(os.path.join(H, 'ohlcv_2023.parquet'))
    codes = sorted(o['code'].astype(str).unique()) if 'code' in o else []
    note('ohlcv', {'columns': list(o.columns), 'rows': len(o), 'codes': len(codes), 'has_0050': '0050' in codes, 'has_2330': '2330' in codes,
                   'sample_codes': codes[:15], 'dates': [str(o['date'].min()), str(o['date'].max())], 'head': o.head(3).to_dict('records')})
except Exception as e:
    note('ohlcv_error', str(e))
try:
    i = pd.read_parquet(os.path.join(H, 'institutional_2020_2025.parquet'))
    note('institutional', {'columns': list(i.columns), 'rows': len(i), 'head': i.head(3).to_dict('records')})
except Exception as e:
    note('inst_error', str(e))

def get(name, url, **kw):
    try:
        r = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'}, **kw)
        txt = r.text
        note(name, {'status': r.status_code, 'len': len(txt), 'head': txt[:600]})
        return r
    except Exception as e:
        note(name, {'error': str(e)})

get('twse_taiex_month', 'https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date=20230601&response=json')
get('twse_industry_openapi', 'https://openapi.twse.com.tw/v1/opendata/t187ap03_L')
get('tpex_industry_openapi', 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O')
get('mops_revenue_html', 'https://mops.twse.com.tw/nas/t21/sii/t21sc03_112_6_0.html')
get('mopsov_revenue_html', 'https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_112_6_0.html')
get('twse_exright', 'https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate=20230601&endDate=20230630&response=json')
get('twse_qfii', 'https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?date=20230602&selectType=ALLBUT0999&response=json')
get('twse_breadth', 'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date=20230602&type=MS&response=json')

FM = 'https://api.finmindtrade.com/api/v4/data'
TOKEN = os.environ.get('FINMIND_TOKEN', '').strip()
for ds, sid, a, b in [('TaiwanStockNews', '2330', '2023-06-01', '2023-06-05'), ('TaiwanStockMonthRevenue', '2330', '2023-01-01', '2023-12-31'),
                      ('TaiwanStockFinancialStatements', '2330', '2023-01-01', '2023-12-31'), ('TaiwanStockCashFlowsStatement', '2330', '2023-01-01', '2023-12-31'),
                      ('TaiwanStockBalanceSheet', '2330', '2023-01-01', '2023-12-31'), ('TaiwanStockDividendResult', '2330', '2022-01-01', '2024-12-31'),
                      ('TaiwanStockPrice', 'TAIEX', '2023-06-01', '2023-06-10'), ('TaiwanStockInfo', '', '', ''),
                      ('TaiwanStockShareholding', '2330', '2023-06-01', '2023-06-05'), ('TaiwanStockPER', '2330', '2023-06-01', '2023-06-05')]:
    p = {'dataset': ds}
    if sid: p['data_id'] = sid
    if a: p['start_date'] = a
    if b: p['end_date'] = b
    if TOKEN: p['token'] = TOKEN
    try:
        r = requests.get(FM, params=p, timeout=60)
        j = r.json()
        d = j.get('data') or []
        note('finmind_' + ds, {'status': r.status_code, 'msg': j.get('msg'), 'rows': len(d), 'cols': list(d[0].keys()) if d else [], 'head': d[:3]})
    except Exception as e:
        note('finmind_' + ds, {'error': str(e)})
    time.sleep(1.5)

json.dump(rep, open(os.path.join(OUT, 'probe.json'), 'w'), ensure_ascii=False, indent=1, default=str)
