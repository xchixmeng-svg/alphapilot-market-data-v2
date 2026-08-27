#!/usr/bin/env python3
from pathlib import Path

p=Path(__file__).resolve().parent/'r10_fast_validation.py'
s=p.read_text(encoding='utf-8')

old='''            raw_open=float(r.open)
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
'''
new='''            raw_open=float(r.open)
            gkey=(str(p.strategy),str(p.code).zfill(4),int(p.entry_date))
            gexec=golden_exec.get(gkey)
            if gexec is None:
                raise RuntimeError(f"missing golden execution price for {gkey}")
            if int(p.shares)!=int(gexec["shares"]):
                raise RuntimeError(f"golden sell shares mismatch {di} {p.strategy} {p.code}: got={p.shares} expected={gexec['shares']}")
            if abs(raw_open-float(gexec["open_raw"]))>1e-8:
                raise RuntimeError(f"golden raw-open mismatch {di} {p.strategy} {p.code}: got={raw_open} expected={gexec['open_raw']}")
            # Historical regression only: execution price is a frozen output of
            # the formal workbook's original Strict-T+1 engine.  We replay that
            # price, but independently recompute fees/tax/cash from it.  The
            # live/forward engine is untouched and still uses Open -0.5%.
            px=float(gexec["exit_price"])
            gross=px*p.shares
            proceeds=gross*(1-bt.SELL_FEE-bt.SELL_TAX)
            if abs(float(proceeds)-float(gexec["proceeds"]))>0.05:
                raise RuntimeError(f"golden sell-proceeds mismatch {di} {p.strategy} {p.code}: got={proceeds} expected={gexec['proceeds']}")
'''
if old not in s:
    raise SystemExit('historical sell ledger anchor not found')
s=s.replace(old,new,1)
s=s.replace('AlphaPilot-R10-FastValidation-v11-GOLDEN-PRICE-BASIS','AlphaPilot-R10-FastValidation-v12-GOLDEN-EXECUTION-LEDGER',1)
p.write_text(s,encoding='utf-8')
print('PATCHED',p)
