import pandas as pd
t = pd.read_csv("lights_trades_raw.csv", parse_dates=["sig", "date"])
E = pd.read_csv("lights_equity.csv", parse_dates=["date"]).set_index("date")
tx = pd.read_csv("/home/claude/xchixmeng-svg/alphapilot-market-data-v2/data/extra/macro/TAIEX.csv", parse_dates=["date"]).set_index("date")
ex = t[t.date.notna()].reset_index(drop=True)
rows = []; acct = 1_000_000.0
for k in range(len(ex)):
    a = ex.iloc[k]
    if not str(a.kind).endswith("買進"): continue
    b = ex.iloc[k + 1] if k + 1 < len(ex) else None
    end = b.date if b is not None else E.index[-1]
    seg = tx.close[(tx.index >= a.date) & (tx.index <= end)]
    # 離底：進場前後 60 個交易日內的最低收盤
    win = tx.close[(tx.index >= a.date - pd.Timedelta(days=90)) & (tx.index <= a.date + pd.Timedelta(days=90))]
    lo = win.min(); lod = win.idxmin()
    outp = b.px if b is not None else E.equity.iloc[-1]
    ret = (outp * (1 - 0.002425) / (a.px / (1 - 0.001425)) - 1) if b is not None else None
    acct_after = E.equity.loc[:end].iloc[-1] if b is None else None
    rows.append(dict(進場類型=a.kind, 訊號日=a.sig.date() if a.sig == a.sig else "", 進場日=a.date.date(), 進場價=round(a.px),
                     離前後低點=f"{(a.px/lo-1)*100:+.1f}%（低點 {lod.date()} {lo:,.0f}）",
                     出場訊號日=(b.sig.date() if b is not None else "持有中"), 出場日=(b.date.date() if b is not None else ""),
                     出場價=(round(b.px) if b is not None else ""), 持有期間最低=f"{(seg.min()/a.px-1)*100:+.1f}%",
                     這筆報酬=(f"{ret*100:+.1f}%" if ret is not None else "")))
R = pd.DataFrame(rows)
# 帳戶金額：每筆出場那天的權益
vals = []
for r in rows:
    d = pd.Timestamp(r["出場日"]) if r["出場日"] else E.index[-1]
    vals.append(round(E.equity.loc[:d].iloc[-1] / 1e4))
R["出場後帳戶(萬)"] = vals
R.to_csv("lights_trades.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
print(R.to_string())
dd = E.equity / E.equity.cummax() - 1
print("最大回落", dd.idxmin().date(), round(dd.min()*100,1), "高點", E.equity.loc[:dd.idxmin()].idxmax().date())
y = E.equity.resample("YE").last(); print((y.pct_change()*100).round(1).to_dict())
