#!/usr/bin/env python3
from clean_event_loop import *

class Scripted:
    def decide(self,ctx):
        if ctx.date=='2021-01-04':
            # 100 shares @100 => ~1% NAV, precommitted for T+1.
            return [],[BuyIntent('2330',100,100.0,'TEST_ENTRY')]
        if ctx.date=='2021-01-05' and '2330' in ctx.positions:
            return [SellIntent('2330',full_exit=True,reason='TEST_EXIT')],[]
        return [],[]

bars={
 '2021-01-04':{'2330':Bar('2021-01-04','2330','台積電',100,101,99,100)},
 '2021-01-05':{'2330':Bar('2021-01-05','2330','台積電',99,101,98,100)},
 '2021-01-06':{'2330':Bar('2021-01-06','2330','台積電',102,103,101,102)},
}
e=PortfolioEngine(1_000_000)
r=e.run(list(bars),bars,Scripted())
assert r['completed_trades']==1,r
q=r['ledger'][0]
assert q.entry_date=='2021-01-05' and q.exit_date=='2021-01-06',q
assert q.shares==100,q
# Open 99 <= limit100 => adverse buy 99*1.005=99.495, legal ceil tick 99.5.
assert abs(q.buy_price-99.5)<1e-9,q
# Sell model = 102*0.98=99.96; legal floor tick at this price=99.9.
assert abs(q.sell_execution_price-99.9)<1e-9,q
assert abs(q.buy_commission-(99.5*100*0.001425))<1e-9,q
assert abs(q.sell_commission-(99.9*100*0.001425))<1e-9,q
assert abs(q.sell_tax-(99.9*100*0.003))<1e-9,q
assert q.hold_sessions>=2,q
assert all(x.drawdown<=1e-12 for x in r['nav_rows'])

for bad in (1,99,437,583,1267):
    try: validate_shares(bad)
    except ValueError: pass
    else: raise AssertionError(f'invalid shares accepted {bad}')
for good in (100,400,500,1000,1200): validate_shares(good)

try: enforce_single_stock_cap(200001,1_000_000)
except ValueError: pass
else: raise AssertionError('20% cap failed')
enforce_single_stock_cap(200000,1_000_000)

# A buy that would exceed T-close free cash must never be queued even if a sale is also planned.
class CashReuseProbe:
    def decide(self,ctx):
        if ctx.date=='2021-01-04': return [],[BuyIntent('A001',1000,190.0,'FIRST')]
        if ctx.date=='2021-01-05' and 'A001' in ctx.positions:
            return [SellIntent('A001',full_exit=True,reason='ROTATE')],[BuyIntent('B001',5000,180.0,'ILLEGAL_REUSE')]
        return [],[]
b2={
 '2021-01-04':{'A001':Bar('2021-01-04','A001','A',190,190,190,190),'B001':Bar('2021-01-04','B001','B',180,180,180,180)},
 '2021-01-05':{'A001':Bar('2021-01-05','A001','A',190,190,190,190),'B001':Bar('2021-01-05','B001','B',180,180,180,180)},
 '2021-01-06':{'A001':Bar('2021-01-06','A001','A',190,190,190,190),'B001':Bar('2021-01-06','B001','B',180,180,180,180)},
}
e2=PortfolioEngine(1_000_000); r2=e2.run(list(b2),b2,CashReuseProbe())
assert not any(x.get('code')=='B001' for x in r2['order_log']),r2['order_log']
print({'status':'PASS','tests':'T+1, costs, tax, 100-share, cap, no-sell-cash-reuse'})
