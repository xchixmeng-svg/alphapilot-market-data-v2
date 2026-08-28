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

# A T-close order whose T+1 security has no tradable bar must expire unfilled.
# Never synthesize a price and never crash the portfolio simulation.
class MissingBarBuy:
    def decide(self,ctx):
        return ([],[BuyIntent('6206',100,50.0,'MISSING_BAR_TEST')]) if ctx.date=='2021-01-04' else ([],[])
missing_bars={
 '2021-01-04':{'2330':Bar('2021-01-04','2330','台積電',100,101,99,100)},
 '2021-01-05':{'2330':Bar('2021-01-05','2330','台積電',101,102,100,101)},
}
e3=PortfolioEngine(1_000_000)
r3=e3.run(list(missing_bars),missing_bars,MissingBarBuy())
assert not e3.positions,e3.positions
miss=[x for x in r3['order_log'] if x.get('code')=='6206']
assert len(miss)==1 and not miss[0]['filled'] and miss[0]['reason']=='NO_TRADABLE_BAR',miss

# A binding sell instruction cannot fabricate an execution during suspension.
# It must stay pending and execute at the first later session with a valid open.
class SuspendedSell:
    def decide(self,ctx):
        if ctx.date=='2021-01-04': return [],[BuyIntent('3701',100,50.0,'ENTRY')]
        if ctx.date=='2021-01-05' and '3701' in ctx.positions: return [SellIntent('3701',True,None,'EXIT')],[]
        return [],[]
suspend_bars={
 '2021-01-04':{'3701':Bar('2021-01-04','3701','大眾控',50,51,49,50)},
 '2021-01-05':{'3701':Bar('2021-01-05','3701','大眾控',49,50,48,49)},
 '2021-01-06':{},
 '2021-01-07':{'3701':Bar('2021-01-07','3701','大眾控',47,48,46,47)},
}
e4=PortfolioEngine(1_000_000)
r4=e4.run(list(suspend_bars),suspend_bars,SuspendedSell())
assert r4['completed_trades']==1,r4
sq=r4['ledger'][0]
assert sq.exit_date=='2021-01-07',sq
assert abs(sq.sell_open_reference-47.0)<1e-12,sq
assert abs(sq.sell_execution_price-46.05)<1e-12,sq
sl=[x for x in r4['order_log'] if x.get('side')=='SELL' and x.get('code')=='3701']
assert len(sl)==2 and sl[0]['reason']=='NO_TRADABLE_BAR_DEFERRED' and not sl[0]['filled'] and sl[1]['filled'],sl

print({'status':'PASS','tests':'T+1, costs, tax, 100-share, cap, reserve, no-future-cash-reuse, missing-buy-no-fill, suspended-sell-defer'})
