#!/usr/bin/env python3
from clean_event_loop import *

class Scripted:
    def decide(self,ctx):
        if ctx.date=='2021-01-04':
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
assert abs(q.buy_price-99.5)<1e-9,q
assert abs(q.sell_execution_price-99.9)<1e-9,q
assert abs(q.buy_commission-(99.5*100*0.001425))<1e-9,q
assert abs(q.sell_commission-(99.9*100*0.001425))<1e-9,q
assert abs(q.sell_tax-(99.9*100*0.003))<1e-9,q
assert q.hold_sessions==1,q
assert all(x.drawdown<=1e-12 for x in r['nav_rows'])
assert all(x.reserved_cash>=-1e-9 for x in r['nav_rows'])

for bad in (1,99,437,583,1267):
    try: validate_shares(bad)
    except ValueError: pass
    else: raise AssertionError(f'invalid shares accepted {bad}')
for good in (100,400,500,1000,1200): validate_shares(good)

try: enforce_single_stock_cap(200001,1_000_000)
except ValueError: pass
else: raise AssertionError('20% cap failed')
enforce_single_stock_cap(200000,1_000_000)

# Isolate T-close reservation: order is within 20% NAV, but free cash is only 100k.
# It must be rejected rather than assuming tomorrow's sale or any future cash inflow.
e2=PortfolioEngine(1_000_000)
e2.cash=100_000.0
e2._queue_buys('2021-01-05','2021-01-06',[BuyIntent('B001',1000,190.0,'NO_FUTURE_CASH')],1_000_000.0)
assert not e2.pending_buys,e2.pending_buys

print({'status':'PASS','tests':'T+1, costs, tax, 100-share, cap, reserve, no-future-cash-reuse'})
