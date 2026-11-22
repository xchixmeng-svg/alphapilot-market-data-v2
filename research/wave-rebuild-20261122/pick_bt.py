"""抄底選股回測：燈號抓底訊號日收盤後選股，隔天開盤買，跟燈號系統同一天出場（系統出場訊號隔天開盤）。
分數＝三項排名百分位的平均（有資料的才算）：
  抗跌：從大盤一年高點那天到訊號日，個股漲跌幅（越抗跌越高）
  法人：最近 20 天外資＋投信買賣超合計 ÷ 20 天成交量（2020 起才有）
  營收：訊號日前已公布的最新月營收年增率（2013 起才有）
股票池：訊號日 20 天平均成交金額前 500 大（4 碼普通股）。推 8 檔，影子帳戶買前 5 檔，各 18%（共 90%），整股＋零股。"""
import pandas as pd, numpy as np, json
R = "/home/claude/xchixmeng-svg/alphapilot-market-data-v2"
P = pd.read_parquet("/home/claude/work/panel.parquet")
op = P.pivot_table(index="date", columns="code", values="open")
cl = P.pivot_table(index="date", columns="code", values="close")
vl = P.pivot_table(index="date", columns="code", values="volume")
tv = P.pivot_table(index="date", columns="code", values="value")
inst = pd.read_parquet(f"{R}/data/history/2020-2025/institutional_2020_2025.parquet")
inst["d"] = pd.to_datetime(inst.date.astype(str)); inst["net"] = inst.foreign_net.fillna(0) + inst.trust_net.fillna(0)
inet = inst.pivot_table(index="d", columns="code", values="net", aggfunc="sum")
rev = pd.read_csv(f"{R}/data/extra/revenue_history.csv", dtype={"code": str})
rev["avail"] = pd.to_datetime(rev.available_date.astype(str))
tx = pd.read_csv(f"{R}/data/extra/macro/TAIEX.csv", parse_dates=["date"]).set_index("date")
names = {}
import zipfile, glob
for f in sorted(glob.glob(f"{R}/data/history/2007-2019/raw/yearly_201*.zip"))[-1:] + []:
    z = zipfile.ZipFile(f); x = pd.read_csv(z.open(z.namelist()[0]), dtype={"code": str}, usecols=["code", "name"]); names.update(dict(zip(x.code, x.name)))
for f in sorted(glob.glob(f"{R}/data/history/2020-2025/ohlcv_2025.parquet")):
    x = pd.read_parquet(f, columns=["code", "name"]); names.update(dict(zip(x.code, x.name)))

EVENTS = [("2010-06-17", "2011-08-05"), ("2011-12-19", "2012-05-15"), ("2018-10-25", "2019-05-17"),
          ("2020-03-19", "2021-08-20"), ("2022-09-28", "2023-10-26"), ("2024-08-06", "2024-11-29")]
BUY, SELL = 0.001425, 0.001425 + 0.003
CAP0 = 1_000_000.0

def nxt(d): return cl.index[cl.index.get_loc(d) + 1]

def pick(sig):
    d = pd.Timestamp(sig); i = cl.index.get_loc(d)
    liq = tv.iloc[i - 19:i + 1].mean().dropna().sort_values(ascending=False)
    uni = liq.index[:500]
    txc = tx.close[tx.index <= d]; hi_d = txc.iloc[-250:].idxmax()
    base = cl.loc[:hi_d].iloc[-1]
    res = (cl.iloc[i] / base - 1).reindex(uni)
    S = pd.DataFrame({"抗跌": res})
    S["法人"] = np.nan
    if d >= pd.Timestamp("2020-02-01"):
        n20 = inet.loc[:d].iloc[-20:].sum().reindex(uni); v20 = vl.iloc[i - 19:i + 1].sum().reindex(uni)
        S["法人"] = n20 / v20
    S["營收"] = np.nan
    r = rev[rev.avail <= d].sort_values("period").groupby("code").tail(1).set_index("code")
    if len(r): S["營收"] = r.yoy_pct.reindex(uni)
    S = S.dropna(subset=["抗跌"])
    pct = S.rank(pct=True)
    S["分數"] = pct.mean(axis=1, skipna=True) * 100
    S["名稱"] = [names.get(c, "") for c in S.index]
    return S.sort_values("分數", ascending=False), hi_d

out = []; acct = CAP0; ev_rows = []; trade_rows = []
for sig, xsig in EVENTS:
    S, hi_d = pick(sig)
    d_in = nxt(pd.Timestamp(sig)); d_out = nxt(pd.Timestamp(xsig))
    prev = cl.loc[pd.Timestamp(sig)]
    cand = []
    for c in S.index:
        o = op.at[d_in, c] if c in op.columns else np.nan
        if not (o == o) or o <= 0: continue
        if o >= prev[c] * 1.095: continue          # 開盤漲停買不到
        cand.append(c)
        if len(cand) == 8: break
    buy = cand[:5]
    cash = acct; budget = acct * 0.90 / 5; pos = {}
    for c in buy:
        o = op.at[d_in, c]; sh = int(budget / (o * (1 + BUY)))
        pos[c] = sh; cash -= sh * o * (1 + BUY)
    # 出場：系統出場訊號隔天開盤；跌停開盤賣不掉就往後一天
    val = cash; per = []
    seg = cl.loc[d_in:d_out]
    for c in cand:
        o_in = op.at[d_in, c]
        k = cl.index.get_loc(d_out)
        while True:
            o_out = op.iloc[k][c]; pc = cl.iloc[k - 1][c]
            if o_out == o_out and o_out > 0 and not (o_out <= pc * 0.905): break
            k += 1
            if k >= len(cl) - 1: o_out = cl.iloc[-1][c]; break
        r_ = o_out * (1 - SELL) / (o_in * (1 + BUY)) - 1
        low = (seg[c].min() / o_in - 1)
        if c in pos: val += pos[c] * o_out * (1 - SELL)
        per.append(dict(事件=sig, 名次=cand.index(c) + 1, 代號=c, 名稱=names.get(c, ""), 分數=round(S.at[c, "分數"]),
                        抗跌=f"{S.at[c,'抗跌']*100:+.0f}%", 法人=("" if S.at[c, "法人"] != S.at[c, "法人"] else f"{S.at[c,'法人']*100:+.1f}%"),
                        營收年增=("" if S.at[c, "營收"] != S.at[c, "營收"] else f"{S.at[c,'營收']:+.0f}%"),
                        進場日=d_in.date(), 進場價=o_in, 出場價=o_out, 期間最低=f"{low*100:+.0f}%", 報酬=round(r_ * 100, 1), 影子帳戶買=("✓" if c in pos else ""),
                        張零股=(f"{pos[c]//1000} 張 {pos[c]%1000} 股" if c in pos else "")))
    tx_r = tx.open.loc[d_out] * (1 - 0.002425) / (tx.open.loc[d_in] * (1 + 0.001425)) - 1
    ev_rows.append(dict(事件=sig, 進場日=d_in.date(), 出場日=d_out.date(), 帳戶進場前=round(acct), 帳戶出場後=round(val),
                        影子5檔=round((val / acct - 1) * 100, 1), 候選8檔平均=round(np.mean([p["報酬"] for p in per]), 1),
                        大盤同期=round(tx_r * 100, 1)))
    acct = val; trade_rows += per
E = pd.DataFrame(ev_rows); T = pd.DataFrame(trade_rows)
E.to_csv("/home/claude/work/pick_events.csv", index=False); T.to_csv("/home/claude/work/pick_trades.csv", index=False)
pd.set_option("display.width", 300)
print(E.to_string()); print(T.to_string())
