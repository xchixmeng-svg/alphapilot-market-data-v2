#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Protocol, Sequence
import math

from clean_backtest_engine import (
    BuyOrder, SellOrder, buy_fill_price, sell_fill_price, commission,
    validate_shares, enforce_single_stock_cap, MAX_SINGLE_STOCK_NAV,
)

STOCK_SELL_TAX_RATE = 0.003
INITIAL_CAPITAL = 1_000_000.0


def tick_size(price: float) -> float:
    if price < 10: return 0.01
    if price < 50: return 0.05
    if price < 100: return 0.10
    if price < 500: return 0.50
    if price < 1000: return 1.00
    return 5.00


def floor_tick(price: float) -> float:
    step=tick_size(price)
    return round(math.floor((price+1e-12)/step)*step, 4)


def ceil_tick(price: float) -> float:
    step=tick_size(price)
    return round(math.ceil((price-1e-12)/step)*step, 4)


@dataclass(frozen=True)
class Bar:
    date: str
    code: str
    name: str
    open: float
    high: float
    low: float
    close: float


@dataclass
class Position:
    code: str
    name: str
    shares: int
    entry_date: str
    entry_price: float
    buy_commission: float
    cost_basis: float
    last_close: float
    hold_sessions: int = 1


@dataclass(frozen=True)
class BuyIntent:
    code: str
    shares: int
    limit_price: float
    reason: str


@dataclass(frozen=True)
class SellIntent:
    code: str
    shares: Optional[int] = None
    full_exit: bool = True
    reason: str = 'STRATEGY_EXIT'


@dataclass
class PendingBuy:
    order: BuyOrder
    reason: str
    reserved_cash: float


@dataclass
class PendingSell:
    order: SellOrder
    reason: str


@dataclass
class TradeLedgerRow:
    code: str
    name: str
    entry_date: str
    exit_date: str
    shares: int
    buy_price: float
    sell_open_reference: float
    sell_execution_price: float
    buy_commission: float
    sell_commission: float
    sell_tax: float
    gross_buy_value: float
    gross_sell_value: float
    total_costs: float
    net_pnl: float
    net_return: float
    hold_sessions: int
    exit_reason: str


@dataclass
class DailyNav:
    date: str
    cash: float
    reserved_cash: float
    market_value: float
    nav: float
    peak_nav: float
    drawdown: float
    holdings: int


@dataclass
class DecisionContext:
    date: str
    next_date: Optional[str]
    cash: float
    reserved_cash: float
    nav: float
    positions: Dict[str, Position]
    bars_by_code: Dict[str, Bar]


class Strategy(Protocol):
    def decide(self, ctx: DecisionContext) -> tuple[Sequence[SellIntent], Sequence[BuyIntent]]: ...


class PortfolioEngine:
    """One causal event loop. Strategy sees T data only; execution consumes T+1 bars."""
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        if initial_capital <= 0: raise ValueError('initial capital must be positive')
        self.initial_capital=float(initial_capital)
        self.cash=float(initial_capital)
        self.positions: Dict[str, Position]={}
        self.pending_buys: Dict[str,List[PendingBuy]]={}
        self.pending_sells: Dict[str,List[PendingSell]]={}
        self.ledger: List[TradeLedgerRow]=[]
        self.nav_rows: List[DailyNav]=[]
        self.order_log: List[dict]=[]
        self.peak_nav=float(initial_capital)

    def reserved_cash(self) -> float:
        return sum(x.reserved_cash for rows in self.pending_buys.values() for x in rows)

    def available_cash(self) -> float:
        return self.cash-self.reserved_cash()

    def mark_nav(self, bars: Dict[str,Bar]) -> float:
        mv=0.0
        for p in self.positions.values():
            b=bars.get(p.code)
            px=b.close if b is not None else p.last_close
            p.last_close=px; mv += p.shares*px
        return self.cash+mv

    def _execute_sells(self, date: str, bars: Dict[str,Bar]) -> None:
        for ps in self.pending_sells.pop(date,[]):
            o=ps.order; p=self.positions.get(o.code)
            if p is None: continue
            b=bars.get(o.code)
            if b is None or b.open<=0: raise RuntimeError(f'missing sell open {date} {o.code}')
            shares=p.shares if o.full_exit else min(o.shares,p.shares)
            validate_shares(int(shares))
            px=floor_tick(sell_fill_price(float(b.open)))
            gross=px*shares; sell_fee=commission(gross); tax=gross*STOCK_SELL_TAX_RATE
            proceeds=gross-sell_fee-tax; self.cash += proceeds
            portion=shares/p.shares
            allocated_cost=p.cost_basis*portion
            allocated_buy_fee=p.buy_commission*portion
            pnl=proceeds-allocated_cost
            self.ledger.append(TradeLedgerRow(
                code=p.code,name=p.name,entry_date=p.entry_date,exit_date=date,shares=shares,
                buy_price=p.entry_price,sell_open_reference=b.open,sell_execution_price=px,
                buy_commission=allocated_buy_fee,sell_commission=sell_fee,sell_tax=tax,
                gross_buy_value=p.entry_price*shares,gross_sell_value=gross,
                total_costs=allocated_buy_fee+sell_fee+tax,net_pnl=pnl,
                net_return=pnl/allocated_cost if allocated_cost else 0.0,
                hold_sessions=p.hold_sessions,exit_reason=ps.reason))
            self.order_log.append({'decision_date':o.decision_date,'execute_date':date,'side':'SELL','code':o.code,'shares':shares,'filled':True,'fill_price':px,'reason':ps.reason})
            if shares==p.shares: del self.positions[o.code]
            else:
                p.shares-=shares; p.cost_basis-=allocated_cost; p.buy_commission-=allocated_buy_fee

    def _execute_buys(self, date: str, bars: Dict[str,Bar], nav_before_buy: float) -> None:
        for pb in self.pending_buys.pop(date,[]):
            o=pb.order; b=bars.get(o.code)
            if b is None: raise RuntimeError(f'missing buy bar {date} {o.code}')
            if o.code in self.positions:
                self.cash += 0.0
                self.order_log.append({'decision_date':o.decision_date,'execute_date':date,'side':'BUY','code':o.code,'shares':o.shares,'filled':False,'reason':'ALREADY_HELD'})
                continue
            fill=buy_fill_price(o,float(b.open),float(b.low))
            if fill is None:
                self.order_log.append({'decision_date':o.decision_date,'execute_date':date,'side':'BUY','code':o.code,'shares':o.shares,'limit_price':o.limit_price,'filled':False,'reason':'LIMIT_NOT_TOUCHED'})
                continue
            px=ceil_tick(float(fill)); px=min(px,o.limit_price)
            gross=px*o.shares; fee=commission(gross); cost=gross+fee
            if cost>pb.reserved_cash+1e-6: raise RuntimeError('execution cost exceeds precommitted reserve')
            if cost>self.cash+1e-6: raise RuntimeError('cash shortfall: pending sell proceeds were improperly assumed')
            enforce_single_stock_cap(gross,nav_before_buy)
            self.cash-=cost
            self.positions[o.code]=Position(o.code,b.name,o.shares,date,px,fee,cost,b.close,1)
            self.order_log.append({'decision_date':o.decision_date,'execute_date':date,'side':'BUY','code':o.code,'shares':o.shares,'limit_price':o.limit_price,'filled':True,'fill_price':px,'reason':pb.reason})

    def _queue_sells(self, date: str, next_date: str, intents: Sequence[SellIntent]) -> None:
        seen=set()
        for x in intents:
            if x.code in seen: raise ValueError('duplicate sell intent')
            seen.add(x.code); p=self.positions.get(x.code)
            if p is None: continue
            # D1 may generate at close, but execution date is the following session D2.
            if p.entry_date==next_date: raise RuntimeError('same-day round trip scheduling detected')
            shares=p.shares if x.full_exit or x.shares is None else int(x.shares)
            validate_shares(shares)
            if shares>p.shares: raise ValueError('sell shares exceed position')
            o=SellOrder(date,next_date,x.code,shares,bool(x.full_exit))
            self.pending_sells.setdefault(next_date,[]).append(PendingSell(o,x.reason))

    def _queue_buys(self, date: str, next_date: str, intents: Sequence[BuyIntent], nav: float) -> None:
        free=self.available_cash(); seen=set(self.positions)
        for x in intents:
            if x.code in seen: continue
            seen.add(x.code); validate_shares(int(x.shares))
            if x.limit_price<=0: raise ValueError('limit price must be positive')
            gross=x.limit_price*x.shares; reserve=gross+commission(gross)
            enforce_single_stock_cap(gross,nav)
            if reserve>free+1e-6: continue
            # Reserve from T-close free cash only. T+1 sale proceeds are not counted.
            free-=reserve
            o=BuyOrder(date,next_date,x.code,int(x.shares),float(x.limit_price))
            self.pending_buys.setdefault(next_date,[]).append(PendingBuy(o,x.reason,reserve))

    def run(self, dates: Sequence[str], bars_by_date: Dict[str,Dict[str,Bar]], strategy: Strategy) -> dict:
        dates=list(dates)
        for i,date in enumerate(dates):
            bars=bars_by_date[date]
            # Execute orders committed at prior close. Sells are processed first operationally,
            # but buy affordability remains capped by its own T-close reservation.
            self._execute_sells(date,bars)
            nav_pre=self.mark_nav(bars)
            self._execute_buys(date,bars,nav_pre)
            nav=self.mark_nav(bars); self.peak_nav=max(self.peak_nav,nav)
            mv=sum(p.shares*(bars[p.code].close if p.code in bars else p.last_close) for p in self.positions.values())
            self.nav_rows.append(DailyNav(date,self.cash,self.reserved_cash(),mv,nav,self.peak_nav,nav/self.peak_nav-1.0,len(self.positions)))
            for p in self.positions.values():
                if p.entry_date!=date: p.hold_sessions+=1
            if i+1>=len(dates): continue
            nxt=dates[i+1]
            # Build a shallow position snapshot so strategy cannot mutate engine state.
            snap={k:Position(**asdict(v)) for k,v in self.positions.items()}
            ctx=DecisionContext(date,nxt,self.cash,self.reserved_cash(),nav,snap,dict(bars))
            sells,buys=strategy.decide(ctx)
            self._queue_sells(date,nxt,sells)
            self._queue_buys(date,nxt,buys,nav)
        navs=[r.nav for r in self.nav_rows]
        mdd=min((r.drawdown for r in self.nav_rows),default=0.0)
        return {'initial_capital':self.initial_capital,'end_nav':navs[-1] if navs else self.initial_capital,'max_drawdown':mdd,'completed_trades':len(self.ledger),'orders':len(self.order_log),'nav_rows':self.nav_rows,'ledger':self.ledger,'order_log':self.order_log}
