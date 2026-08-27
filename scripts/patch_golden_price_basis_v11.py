#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parent / "r10_fast_validation.py"
s = p.read_text(encoding="utf-8")

# The formal workbook records raw T+1 open for audit, but the historical
# execution engine applies the 0.5% adverse sell model to corporate-action-
# adjusted open (aopen).  Preserve both and assert the final execution ledger.
anchor = '''    if len(golden_exit) != 241:
        raise RuntimeError("golden exit key is not unique")
'''
insert = anchor + '''
    golden_exec = {
        (str(r.strategy), str(r.code).zfill(4), int(r.entry_date)): {
            "open_raw": float(r.open_raw),
            "exit_price": float(r.exit_price),
            "shares": int(r.shares),
            "proceeds": float(r.proceeds),
        }
        for r in gx.itertuples(index=False)
    }
    if len(golden_exec) != 241:
        raise RuntimeError("golden execution-price key is not unique")
'''
if anchor not in s:
    raise SystemExit("golden execution map anchor not found")
s = s.replace(anchor, insert, 1)

old_sell = '''            px=bt.legal_sell_price(float(r.open)); gross=px*p.shares; proceeds=gross*(1-bt.SELL_FEE-bt.SELL_TAX); cash+=proceeds
            pnl=proceeds-p.cost_total
'''
new_sell = '''            raw_open=float(r.open)
            sell_open=float(r.aopen) if hasattr(r,"aopen") and np.isfinite(r.aopen) else raw_open
            px=bt.legal_sell_price(sell_open)
            gkey=(str(p.strategy),str(p.code).zfill(4),int(p.entry_date))
            gexec=golden_exec.get(gkey)
            if gexec is None:
                raise RuntimeError(f"missing golden execution price for {gkey}")
            if int(p.shares)!=int(gexec["shares"]):
                raise RuntimeError(f"golden sell shares mismatch {di} {p.strategy} {p.code}: got={p.shares} expected={gexec['shares']}")
            if abs(raw_open-float(gexec["open_raw"]))>1e-8:
                raise RuntimeError(f"golden raw-open mismatch {di} {p.strategy} {p.code}: got={raw_open} expected={gexec['open_raw']}")
            if abs(float(px)-float(gexec["exit_price"]))>1e-8:
                raise RuntimeError(f"golden sell-price mismatch {di} {p.strategy} {p.code}: raw_open={raw_open} aopen={sell_open} got={px} expected={gexec['exit_price']}")
            gross=px*p.shares
            proceeds=gross*(1-bt.SELL_FEE-bt.SELL_TAX)
            if abs(float(proceeds)-float(gexec["proceeds"]))>0.05:
                raise RuntimeError(f"golden sell-proceeds mismatch {di} {p.strategy} {p.code}: got={proceeds} expected={gexec['proceeds']}")
            cash+=proceeds
            pnl=proceeds-p.cost_total
'''
if old_sell not in s:
    raise SystemExit("sell-price execution anchor not found")
s = s.replace(old_sell, new_sell, 1)

# R7 limits/signals are on the adjusted price series, so their T+1 fill test
# must use adjusted open/low. R0.5 is defined on raw close and remains raw.
old_buy = '''            r=bt.row_lookup(feat_idx,di,o.code); fill=None
            if r is not None: fill=bt.buy_fill(float(r.open),float(r.low),float(o.limit))
'''
new_buy = '''            r=bt.row_lookup(feat_idx,di,o.code); fill=None
            if r is not None:
                if o.strategy=="R7":
                    buy_open=float(r.aopen) if hasattr(r,"aopen") and np.isfinite(r.aopen) else float(r.open)
                    buy_low=float(r.alow) if hasattr(r,"alow") and np.isfinite(r.alow) else float(r.low)
                else:
                    buy_open=float(r.open); buy_low=float(r.low)
                fill=bt.buy_fill(buy_open,buy_low,float(o.limit))
'''
if old_buy not in s:
    raise SystemExit("buy-price basis anchor not found")
s = s.replace(old_buy, new_buy, 1)

s=s.replace('AlphaPilot-R10-FastValidation-v10-GOLDEN-FULL-STREAM','AlphaPilot-R10-FastValidation-v11-GOLDEN-PRICE-BASIS',1)
p.write_text(s,encoding='utf-8')
print('PATCHED',p)
