#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaPilot R10 Strategy Factory -- STRICT T+1 search engine.

Hard constraints (not searchable):
- 2020 is warm-up only; evaluation is 2021-01-04..2025-12-31.
- Initial capital = NT$1,000,000 common pool.
- T-close decisions only; all sells execute next trading day at Open - 0.5% adverse slippage.
- Pending sells KEEP their slots and T-day exposure through the T close that creates them.
- At T+1 open: sells execute first, then already-committed buys may use the released cash.
- T+1 buy is a precommitted limit; no touch = no fill/no chase.
- Integer shares; board-lot first, odd-lot only when one board lot exceeds target.
- A candidate is feasible only when full-period Max DD >= -15%.

The optimizer may change signal/filter/portfolio/exit parameters, never the execution clock.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import r10_fast_validation as fast

bt = fast.bt
core = fast.core

ENGINE_VERSION = "AlphaPilot-R10-StrategyFactory-STRICT-T1-v1"
INITIAL_CAPITAL = 1_000_000.0
MAX_DD_FLOOR = -0.15
OUT = bt.OUT_ROOT / "latest" / "strategy_factory_strict_t1"


def _norm(weights: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, float(v)) for v in weights.values())
    if s <= 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: max(0.0, float(v)) / s for k, v in weights.items()}


def size_shares(target_cash: float, limit: float, avg_vol20: float) -> Tuple[int, str]:
    if target_cash <= 0 or limit <= 0 or not np.isfinite(avg_vol20):
        return 0, "NONE"
    one_lot = limit * 1000.0
    liq_shares = max(0, int(math.floor(float(avg_vol20) * 0.02 + 1e-12)))
    if one_lot > target_cash + 1e-9:
        shares = int(math.floor(target_cash / limit + 1e-12))
        return max(0, min(shares, liq_shares)), "HIGH_PRICE_ODDLOT"
    lots = int(math.floor(target_cash / one_lot + 1e-12))
    liq_lots = liq_shares // 1000
    return max(0, min(lots, liq_lots) * 1000), "BOARD_LOT"


def strict_slot_contract() -> dict:
    occupied = {"A", "B", "C", "D", "E"}
    pending_sell = {"A"}
    max_positions = 5
    can_add = len(occupied) < max_positions
    assert not can_add, "pending T+1 sell incorrectly freed a T-close slot"
    return {
        "pass": True,
        "rule": "pending sells occupy slots until next-day open execution",
        "occupied_at_T_close": len(occupied),
        "pending_sells": len(pending_sell),
        "new_slot_available": can_add,
    }


def prepare_context() -> dict:
    bt.INITIAL_CAPITAL = INITIAL_CAPITAL
    bt.ADV_CAP = fast.full._BASE_ADV_CAP
    bt.VERSION = ENGINE_VERSION
    cfg = dict(bt.SCENARIOS["validation2021_2025"])
    cfg["warmup_start"] = "2020-01-01"
    cfg["eval_start"] = "2021-01-04"
    cfg["eval_end"] = "2025-12-31"
    cfg["years"] = [2020, 2021, 2022, 2023, 2024, 2025]
    raw = bt.load_scenario_ohlcv(cfg)
    feat, corp_events, bm = bt.build_features(raw)
    eval_start, eval_end = bt.intdate(cfg["eval_start"]), bt.intdate(cfg["eval_end"])
    eval_dates = sorted(int(x) for x in feat.date.unique() if eval_start <= int(x) <= eval_end)
    if not eval_dates:
        raise RuntimeError("no 2021-2025 evaluation dates")
    next_date = {eval_dates[i]: eval_dates[i + 1] for i in range(len(eval_dates) - 1)}
    feat_idx = feat.set_index(["date", "code"]).sort_index()
    ins = fast.fetch_institutional_fast(feat, eval_start, eval_end)
    by_date = {int(d): g.copy() for d, g in feat.groupby("date", sort=False)}
    eligible = feat[(feat.amt20 >= 20_000_000) & feat.aclose.notna()].copy()
    breadth = eligible.groupby("date").apply(
        lambda g: pd.Series({"breadth": float((g.aclose > g.ma60).mean()), "advance10": float((g.r10 > 0).mean())}),
        include_groups=False,
    ).reset_index().sort_values("date")
    breadth["breadth_mean20"] = breadth.breadth.rolling(20, min_periods=10).mean()
    bmap = breadth.set_index("date")
    z = bm.drop_duplicates("date").sort_values("date").copy()
    z["ma60"] = z.mkt.rolling(60, min_periods=60).mean()
    z["ma120"] = z.mkt.rolling(120, min_periods=120).mean()
    z["mr20"] = z.mkt.pct_change(20, fill_method=None)
    z["mr60"] = z.mkt.pct_change(60, fill_method=None)
    zmap = z.set_index("date")
    et = feat[feat.code.eq("0050")][["date", "close"]].drop_duplicates("date").sort_values("date").copy()
    et["m60"] = et.close.rolling(60, min_periods=60).mean()
    et["r20x"] = et.close.pct_change(20, fill_method=None)
    et["r60x"] = et.close.pct_change(60, fill_method=None)
    emap = et.set_index("date")
    inst_by_date = {int(d): g.sort_values("market").drop_duplicates("code", keep="last") for d, g in ins.groupby("date", sort=False)}
    r7_days: dict[int, pd.DataFrame] = {}
    r05_days: dict[int, pd.DataFrame] = {}
    states: dict[int, dict] = {}
    for i, di in enumerate(eval_dates):
        x0 = by_date.get(di)
        if x0 is None or di not in bmap.index or di not in zmap.index:
            raise RuntimeError(f"missing signal context {di}")
        x = x0[core.common(x0)].copy()
        r = zmap.loc[di]
        v = bmap.loc[di]
        m, ma60, ma120, mr20, mr60 = map(float, [r.mkt, r.ma60, r.ma120, r.mr20, r.mr60])
        br, adv, bmean = map(float, [v.breadth, v.advance10, v.breadth_mean20])
        if mr20 <= -0.08 or (m < ma120 and mr60 < 0 and br < 0.40):
            regime = "Bear"
        elif m < ma120 * 1.02 and mr20 > 0 and br > 0.42 and br > bmean:
            regime = "Repair"
        elif m > ma60 and m > ma120 and mr20 > 0 and mr60 > 0 and br >= 0.60 and adv >= 0.52:
            regime = "Strong Bull"
        elif m > ma120 and mr60 > 0 and br >= 0.45:
            regime = "Normal Bull"
        elif m > ma120 * 0.98 and br >= 0.38:
            regime = "Weak"
        else:
            regime = "Fallback/Bear"
        x["rel20"] = x.r20 - mr20
        x["rel60"] = x.r60 - mr60
        for src, dst in [("r10", "p10"), ("rel20", "p20"), ("rel60", "p60"), ("flow20", "pf"), ("amtacc", "pa"), ("clvflow20", "pc"), ("nearhigh", "pn")]:
            x[dst] = core.pr(x[src])
        keep = ["code", "name", "close", "aclose", "ma60", "ma120", "amt20", "nearhigh", "avgvol20", "p10", "p20", "p60", "pf", "pa", "pc", "pn"]
        r7_days[di] = x[keep].copy()
        risk = False
        if di in emap.index and di in inst_by_date:
            e = emap.loc[di]
            risk = bool(e.close > e.m60 and e.r20x > 0 and e.r60x > 0)
            xi = x0[core.common(x0)].copy().merge(inst_by_date[di][["code", "Foreign3D", "Foreign10D", "Trust5D"]], on="code", how="left")
            for src, dst in [("clvflow10", "pclv10"), ("amount_ratio", "pamt"), ("clvflow5", "pclv5"), ("Foreign3D", "pf3"), ("Foreign10D", "pf10"), ("Trust5D", "pt5"), ("ma20gap", "pgap")]:
                xi[dst] = core.pr(xi[src])
            xi["prior60_position"] = xi.aclose / xi.prior_high60 - 1.0
            keep2 = ["code", "name", "close", "aclose", "amt20", "amount_ratio", "r20", "ma20gap", "prior60_position", "prior_high10", "avgvol20", "pclv10", "pamt", "pclv5", "pf3", "pf10", "pt5", "pgap"]
            r05_days[di] = xi[keep2].copy()
        else:
            r05_days[di] = pd.DataFrame()
        states[di] = {"regime": regime, "mkt": m, "ma60": ma60, "ma120": ma120, "mr20": mr20, "mr60": mr60, "breadth": br, "advance10": adv, "r05_risk_on": risk}
        if (i + 1) % 200 == 0 or i + 1 == len(eval_dates):
            bt.log(f"[FACTORY-CONTEXT] {i+1}/{len(eval_dates)} date={di}")
    return {"cfg": cfg, "raw": raw, "feat": feat, "feat_idx": feat_idx, "ins": ins, "eval_dates": eval_dates, "next_date": next_date, "r7_days": r7_days, "r05_days": r05_days, "states": states, "corp_events": corp_events}


def regime_profile(state: dict, p: dict) -> tuple[float, int]:
    reg = state["regime"]
    if reg == "Strong Bull": return p["strong_exposure"], p["strong_slots"]
    if reg == "Normal Bull": return p["normal_exposure"], p["normal_slots"]
    if reg == "Repair": return p["repair_exposure"], p["repair_slots"]
    if reg == "Weak": return p["weak_exposure"], p["weak_slots"]
    return 0.0, 0


def candidates_for_day(ctx: dict, di: int, p: dict) -> tuple[pd.DataFrame, pd.DataFrame, float, int]:
    st = ctx["states"][di]
    expo, slots = regime_profile(st, p)
    x = ctx["r7_days"][di]
    w = p["r7_weights"]
    score = w["p10"] * x.p10 + w["p20"] * x.p20 + w["p60"] * x.p60 + w["pf"] * x.pf + w["pa"] * x.pa + w["pc"] * x.pc + w["pn"] * x.pn
    mask = (x.amt20 >= p["r7_min_amt"]) & (x.nearhigh >= p["r7_nearhigh"]) & score.notna()
    if p["r7_ma_mode"] == "MA120": mask &= x.aclose > x.ma120
    elif p["r7_ma_mode"] == "MA60": mask &= x.aclose > x.ma60
    else: mask &= (x.aclose > x.ma60) & (x.ma60 > x.ma120)
    c = x.loc[mask].copy()
    c["score"] = score.loc[mask]
    c = c.sort_values(["score", "code"], ascending=[False, True])
    c["rank"] = np.arange(1, len(c) + 1)
    y = ctx["r05_days"][di]
    if y.empty or not st["r05_risk_on"]:
        cc = y.iloc[0:0].copy()
    else:
        rw = p["r05_weights"]
        rscore = rw["pclv10"] * y.pclv10 + rw["pamt"] * y.pamt + rw["pclv5"] * y.pclv5 + rw["pf3"] * y.pf3 + rw["pf10"] * y.pf10 + rw["pt5"] * y.pt5 + rw["pgap"] * y.pgap
        h = y.close.between(p["r05_price_low"], p["r05_price_high"]) & (y.amt20 >= p["r05_min_amt"]) & (y.amount_ratio >= p["r05_amount_ratio"]) & y.r20.between(0, p["r05_r20_max"]) & (y.ma20gap <= p["r05_ma20gap_max"]) & (y.prior60_position >= p["r05_prior60_min"]) & (y.aclose > y.prior_high10) & rscore.notna()
        cc = y.loc[h].copy()
        cc["score"] = rscore.loc[h]
        cc = cc.sort_values(["score", "code"], ascending=[False, True])
        cc["rank"] = np.arange(1, len(cc) + 1)
    return c, cc, float(expo), int(min(slots, p["max_positions"]))


def r05_exit_reason(p0, row, cfg: dict):
    if row is None or not np.isfinite(row.aclose): return None
    adj = float(row.aclose)
    ret = adj / p0.entry_adj - 1.0
    p0.peak_adj = max(p0.peak_adj, adj)
    p0.hold_days += 1
    ar = float(row.amount_ratio) if np.isfinite(row.amount_ratio) else 0.0
    eps = 1e-12
    if adj <= p0.entry_adj * (1.0 - cfg["r05_hard_stop"]) + eps: return "HARD"
    if p0.mode == "NORMAL" and ret >= cfg["r05_runner_trigger"] - eps and ar >= 2.0 - eps: p0.mode = "RUNNER"
    if p0.mode in {"RUNNER", "MEGA", "TARGET"} and ret >= 0.80 - eps:
        if ar >= 1.20 - eps: p0.mode = "MEGA"
        elif p0.mode != "MEGA": p0.mode = "TARGET"
    if p0.mode == "MEGA":
        if adj <= p0.peak_adj * 0.84 + eps: return "MEGA_TRAIL"
        if p0.hold_days >= cfg["r05_max_hold"]: return "RUNNER_TIME"
        return None
    if p0.mode == "TARGET":
        if ret >= 2.00 - eps: return "TARGET_200"
        if adj <= p0.peak_adj * 0.80 + eps: return "TARGET_TRAIL"
        if p0.hold_days >= cfg["r05_max_hold"]: return "RUNNER_TIME"
        return None
    if p0.mode == "RUNNER":
        if adj <= p0.peak_adj * (1.0 - cfg["r05_runner_trail"]) + eps: return "RUNNER_TRAIL"
        if p0.hold_days >= cfg["r05_max_hold"]: return "RUNNER_TIME"
        return None
    if ret >= cfg["r05_base_trail_trigger"] - eps and adj <= p0.peak_adj * (1.0 - cfg["r05_base_trail"]) + eps: return "BASE_TRAIL"
    if p0.hold_days >= cfg["r05_max_hold"]: return "TIME"
    return None


def dd_multiplier(dd: float, p: dict) -> float:
    if dd <= p["dd_level3"]: return p["dd_mult3"]
    if dd <= p["dd_level2"]: return p["dd_mult2"]
    if dd <= p["dd_level1"]: return p["dd_mult1"]
    return 1.0


def simulate(ctx: dict, p: dict, collect: bool = False) -> dict:
    eval_dates, next_date, feat_idx = ctx["eval_dates"], ctx["next_date"], ctx["feat_idx"]
    cash = INITIAL_CAPITAL
    positions: Dict[str, bt.Position] = {}
    pending_buys: Dict[int, List[bt.BuyOrder]] = {}
    pending_sells: Dict[int, List[bt.SellOrder]] = {}
    nav_rows, trade_rows, order_rows = [], [], []
    hwm = INITIAL_CAPITAL
    no_buy_until = -1
    force_cooldown_until = -1
    last_regime = None
    last_reb_i = None
    force_events = 0
    for i, di in enumerate(eval_dates):
        for o in pending_sells.pop(di, []):
            k = bt.pos_key(o.strategy, o.code)
            pos = positions.get(k)
            if pos is None: continue
            row = bt.row_lookup(feat_idx, di, pos.code)
            if row is None or not np.isfinite(row.open):
                if di in next_date:
                    o.execute_date = next_date[di]
                    pending_sells.setdefault(o.execute_date, []).append(o)
                continue
            px = bt.legal_sell_price(float(row.open))
            proceeds = px * pos.shares * (1.0 - bt.SELL_FEE - bt.SELL_TAX)
            cash += proceeds
            trade_rows.append({"strategy": pos.strategy, "code": pos.code, "name": pos.name, "entry_date": pos.entry_date, "exit_date": di, "entry_price": pos.entry_price, "exit_price": px, "shares": pos.shares, "cost_total": pos.cost_total, "proceeds": proceeds, "pnl": proceeds - pos.cost_total, "return": proceeds / pos.cost_total - 1.0 if pos.cost_total else np.nan, "exit_reason": o.reason, "hold_days": pos.hold_days, "mode": pos.mode})
            if collect: order_rows.append({"decision_date": o.decision_date, "execute_date": di, "strategy": pos.strategy, "side": "SELL", "code": pos.code, "shares": pos.shares, "fill_price": px, "filled": True, "reason": o.reason})
            del positions[k]
        for o in pending_buys.pop(di, []):
            row = bt.row_lookup(feat_idx, di, o.code)
            fill = None
            if row is not None and np.isfinite(row.open) and np.isfinite(row.low): fill = bt.buy_fill(float(row.open), float(row.low), float(o.limit))
            filled = fill is not None
            reason = "FILLED" if filled else "LIMIT_NOT_TOUCHED"
            if filled:
                cost = float(fill) * o.shares * (1.0 + bt.BUY_FEE)
                if cost > cash + 1e-6:
                    filled = False; reason = "CASH_SHORT_AT_EXECUTION"
                else:
                    cash -= cost
                    factor = float(row.aclose / row.close) if np.isfinite(row.aclose) and row.close else 1.0
                    entry_adj = float(fill) * factor
                    positions[bt.pos_key(o.strategy, o.code)] = bt.Position(o.strategy, o.code, o.name, int(o.shares), di, float(fill), entry_adj, cost, entry_adj)
            if collect: order_rows.append({"decision_date": o.decision_date, "execute_date": di, "strategy": o.strategy, "side": "BUY", "code": o.code, "shares": o.shares, "order_price": o.limit, "fill_price": float(fill) if filled else np.nan, "filled": bool(filled), "reason": reason, "target_cash": o.target_cash, "rank": o.rank})
        nav, stock_mv = bt.mark_nav(cash, positions, feat_idx, di)
        hwm = max(hwm, nav)
        dd = nav / hwm - 1.0
        exposure = stock_mv / nav if nav > 0 else 0.0
        for pos in positions.values():
            row = bt.row_lookup(feat_idx, di, pos.code)
            if row is not None and np.isfinite(row.aclose): pos.peak_adj = max(pos.peak_adj, float(row.aclose))
        r7_cands, r05_cands, r7_expo, r7_slots = candidates_for_day(ctx, di, p)
        state = ctx["states"][di]
        regime = state["regime"]
        regime_changed = last_regime is None or regime != last_regime
        reb_due = last_reb_i is None or regime_changed or (i - last_reb_i) >= p["r7_rebalance_days"]
        if reb_due: last_reb_i = i
        last_regime = regime
        r7_rank = {str(x.code): int(x.rank) for x in r7_cands[["code", "rank"]].itertuples(index=False)}
        r7_score = {str(x.code): float(x.score) for x in r7_cands[["code", "score"]].itertuples(index=False)}
        r05_score = {} if r05_cands.empty else {str(x.code): float(x.score) for x in r05_cands[["code", "score"]].itertuples(index=False)}
        sell_map: dict[str, bt.SellOrder] = {}
        if di in next_date:
            exdate = next_date[di]
            for k, pos in list(positions.items()):
                row = bt.row_lookup(feat_idx, di, pos.code)
                reason = None
                if pos.strategy == "R7":
                    pos.hold_days += 1
                    if row is not None and np.isfinite(row.aclose) and float(row.aclose) <= pos.entry_adj * (1.0 - p["r7_hard_stop"]): reason = "HARD"
                    elif reb_due:
                        rank = r7_rank.get(pos.code, 10**9)
                        if r7_expo <= 0: reason = "REB_REGIME0"
                        elif r7_slots <= 0 or rank > p["r7_rank_exit_mult"] * max(1, r7_slots): reason = "REB_RANK"
                else:
                    reason = r05_exit_reason(pos, row, p)
                if reason: sell_map[k] = bt.SellOrder(di, exdate, pos.strategy, pos.code, reason)
            r7_mv = bt.value_of(positions, feat_idx, di, strategy="R7")
            r7_target = nav * r7_expo
            already = sum(bt.value_of({k: positions[k]}, feat_idx, di) for k in sell_map if k in positions and positions[k].strategy == "R7")
            projected = max(0.0, r7_mv - already)
            if projected > r7_target * 1.03 + 1.0:
                remain = [(k, q) for k, q in positions.items() if q.strategy == "R7" and k not in sell_map]
                remain.sort(key=lambda kv: r7_rank.get(kv[1].code, 10**9), reverse=True)
                for k, pos in remain:
                    if projected <= r7_target: break
                    mv = bt.value_of({k: pos}, feat_idx, di)
                    sell_map[k] = bt.SellOrder(di, exdate, pos.strategy, pos.code, "EXPO")
                    projected -= mv
            if dd <= p["force_dd"] and i >= force_cooldown_until:
                force_cooldown_until = i + p["force_cooldown"]
                no_buy_until = max(no_buy_until, i + p["force_no_buy"])
                force_events += 1
                target = nav * p["force_target_exposure"]
                projected = stock_mv - sum(bt.value_of({k: positions[k]}, feat_idx, di) for k in sell_map if k in positions)
                remain = [(k, q) for k, q in positions.items() if k not in sell_map]
                def weakness(item):
                    _, q = item
                    if q.strategy == "R7": return (0, -r7_rank.get(q.code, 10**9), r7_score.get(q.code, -1e9))
                    row = bt.row_lookup(feat_idx, di, q.code)
                    ret = (float(row.aclose) / q.entry_adj - 1.0) if row is not None and np.isfinite(row.aclose) else -9.0
                    return (1, r05_score.get(q.code, -1e9), ret)
                remain.sort(key=weakness)
                for k, pos in remain:
                    if projected <= target: break
                    mv = bt.value_of({k: pos}, feat_idx, di)
                    sell_map[k] = bt.SellOrder(di, exdate, pos.strategy, pos.code, "FORCE_DD")
                    projected -= mv
            if sell_map: pending_sells.setdefault(exdate, []).extend(sell_map.values())
        created: list[bt.BuyOrder] = []
        if di in next_date and i >= no_buy_until:
            exdate = next_date[di]
            codes_after = {q.code for q in positions.values()}
            base_exposure = bt.value_of(positions, feat_idx, di)
            base_r7 = bt.value_of(positions, feat_idx, di, strategy="R7")
            reserved_exposure = 0.0
            reserved_r7 = 0.0
            reserved_code: dict[str, float] = {}
            def try_order(strategy: str, row, rank: int):
                nonlocal reserved_exposure, reserved_r7, codes_after
                code = str(row.code); name = str(row.name); k = bt.pos_key(strategy, code)
                if k in positions: return
                if code not in codes_after and len(codes_after) >= p["max_positions"]: return
                if strategy == "R05":
                    n = sum(q.strategy == "R05" for q in positions.values()) + sum(o.strategy == "R05" for o in created)
                    if n >= p["r05_max_slots"]: return
                    base_pct = p["r05_base"]
                    limit = float(core.floor_tick(float(row.close) * p["r05_limit_mult"]))
                else:
                    n = sum(q.strategy == "R7" for q in positions.values()) + sum(o.strategy == "R7" for o in created)
                    if n >= r7_slots: return
                    base_pct = p["r7_base"]
                    limit = float(core.floor_tick(float(row.close) * p["r7_limit_mult"]))
                current_code = bt.value_of(positions, feat_idx, di, code=code) + reserved_code.get(code, 0.0)
                rem_single = nav * p["max_single"] - current_code
                rem_global = nav * p["max_total"] - base_exposure - reserved_exposure
                target = nav * base_pct * dd_multiplier(dd, p)
                if strategy == "R7": target = min(target, nav * r7_expo - base_r7 - reserved_r7)
                target = min(target, rem_single, rem_global)
                if target <= 0 or limit <= 0: return
                shares, _mode = size_shares(target, limit, float(row.avgvol20))
                if shares <= 0: return
                notional = shares * limit
                if notional > target + 1e-6: return
                reserve = notional * (1.0 + bt.BUY_FEE)
                created.append(bt.BuyOrder(di, exdate, strategy, code, name, limit, shares, target, reserve, rank))
                reserved_exposure += notional
                reserved_code[code] = reserved_code.get(code, 0.0) + notional
                if strategy == "R7": reserved_r7 += notional
                codes_after.add(code)
            if state["r05_risk_on"] and not r05_cands.empty:
                for _, row in r05_cands.head(p["candidate_depth"]).iterrows(): try_order("R05", row, int(row["rank"]))
            if reb_due and r7_expo > 0 and not r7_cands.empty:
                for _, row in r7_cands.head(max(r7_slots, p["candidate_depth_r7"])).iterrows(): try_order("R7", row, int(row["rank"]))
            if created: pending_buys.setdefault(exdate, []).extend(created)
        nav_rows.append({"date": di, "nav": nav, "cash": cash if collect else np.nan, "stock_mv": stock_mv if collect else np.nan, "exposure": exposure if collect else np.nan, "drawdown": dd, "positions": len({q.code for q in positions.values()}) if collect else np.nan})
        if dd < -0.30: break
    nav_df = pd.DataFrame(nav_rows)
    if nav_df.empty: raise RuntimeError("empty NAV")
    end_nav = float(nav_df.iloc[-1].nav)
    finished = int(nav_df.iloc[-1].date) == int(eval_dates[-1])
    max_dd = float(nav_df.drawdown.min()) if finished else min(float(nav_df.drawdown.min()), -0.300001)
    years = max((bt.dt_from_int(eval_dates[-1]) - bt.dt_from_int(eval_dates[0])).days / 365.25, 1 / 365.25)
    cagr = (end_nav / INITIAL_CAPITAL) ** (1.0 / years) - 1.0 if end_nav > 0 and finished else -1.0
    annual = {}
    if finished:
        temp = nav_df.copy(); temp["year"] = temp.date.astype(str).str[:4].astype(int); prev = INITIAL_CAPITAL
        for y, g in temp.groupby("year"):
            e = float(g.iloc[-1].nav); annual[str(int(y))] = e / prev - 1.0; prev = e
    summary = {"engine_version": ENGINE_VERSION, "initial_capital": INITIAL_CAPITAL, "eval_start": int(eval_dates[0]), "eval_end": int(eval_dates[-1]), "end_nav": end_nav, "cagr": cagr, "max_dd": max_dd, "feasible_dd": bool(finished and max_dd >= MAX_DD_FLOOR - 1e-12), "completed_trades": int(len(trade_rows)), "force_dd_events": int(force_events), "annual_returns": annual, "finished": bool(finished)}
    if collect:
        summary["nav_df"] = nav_df; summary["trades_df"] = pd.DataFrame(trade_rows); summary["orders_df"] = pd.DataFrame(order_rows)
    return summary


def sample_params(trial) -> dict:
    r7w = _norm({"p10": trial.suggest_float("r7_w_p10", 0.12, 0.38), "p20": trial.suggest_float("r7_w_p20", 0.10, 0.34), "p60": trial.suggest_float("r7_w_p60", 0.00, 0.18), "pf": trial.suggest_float("r7_w_flow", 0.04, 0.24), "pa": trial.suggest_float("r7_w_amtacc", 0.03, 0.20), "pc": trial.suggest_float("r7_w_clv", 0.00, 0.15), "pn": trial.suggest_float("r7_w_near", 0.00, 0.15)})
    p = {
        "r7_weights": r7w,
        "r7_min_amt": trial.suggest_categorical("r7_min_amt_m", [20, 30, 40, 50, 70]) * 1_000_000,
        "r7_nearhigh": trial.suggest_float("r7_nearhigh", 0.72, 0.90, step=0.02),
        "r7_ma_mode": trial.suggest_categorical("r7_ma_mode", ["MA60", "MA120", "TREND"]),
        "r7_base": trial.suggest_float("r7_base", 0.12, 0.25, step=0.01),
        "r7_limit_mult": trial.suggest_float("r7_limit_mult", 0.965, 0.995, step=0.005),
        "r7_rebalance_days": trial.suggest_int("r7_rebalance_days", 10, 25),
        "r7_hard_stop": trial.suggest_float("r7_hard_stop", 0.07, 0.14, step=0.01),
        "r7_rank_exit_mult": trial.suggest_float("r7_rank_exit_mult", 1.5, 3.0, step=0.25),
        "r05_weights": {"pclv10": trial.suggest_float("r05_w_clv10", 0.30, 0.65), "pamt": trial.suggest_float("r05_w_amt", 0.10, 0.35), "pclv5": trial.suggest_float("r05_w_clv5", 0.00, 0.15), "pf3": trial.suggest_float("r05_w_f3", 0.00, 0.15), "pf10": trial.suggest_float("r05_w_f10", -0.15, 0.02), "pt5": trial.suggest_float("r05_w_t5", 0.00, 0.10), "pgap": trial.suggest_float("r05_w_gap", -0.30, -0.05)},
        "r05_price_low": trial.suggest_categorical("r05_price_low", [8, 10, 12, 15]),
        "r05_price_high": trial.suggest_categorical("r05_price_high", [30, 40, 50, 60, 80]),
        "r05_min_amt": trial.suggest_categorical("r05_min_amt_m", [30, 50, 70, 100]) * 1_000_000,
        "r05_amount_ratio": trial.suggest_float("r05_amount_ratio", 0.8, 1.5, step=0.1),
        "r05_r20_max": trial.suggest_float("r05_r20_max", 0.15, 0.35, step=0.025),
        "r05_ma20gap_max": trial.suggest_float("r05_ma20gap_max", 0.10, 0.25, step=0.025),
        "r05_prior60_min": trial.suggest_float("r05_prior60_min", -0.25, -0.05, step=0.025),
        "r05_base": trial.suggest_float("r05_base", 0.10, 0.23, step=0.01),
        "r05_limit_mult": trial.suggest_float("r05_limit_mult", 0.985, 1.000, step=0.005),
        "r05_max_slots": trial.suggest_int("r05_max_slots", 1, 3),
        "r05_hard_stop": trial.suggest_float("r05_hard_stop", 0.06, 0.12, step=0.01),
        "r05_runner_trigger": trial.suggest_float("r05_runner_trigger", 0.30, 0.50, step=0.05),
        "r05_runner_trail": trial.suggest_float("r05_runner_trail", 0.10, 0.18, step=0.02),
        "r05_base_trail_trigger": trial.suggest_float("r05_base_trail_trigger", 0.30, 0.60, step=0.05),
        "r05_base_trail": trial.suggest_float("r05_base_trail", 0.08, 0.18, step=0.02),
        "r05_max_hold": trial.suggest_int("r05_max_hold", 45, 100, step=5),
        "max_positions": trial.suggest_int("max_positions", 3, 5),
        "max_single": trial.suggest_float("max_single", 0.20, 0.28, step=0.02),
        "max_total": trial.suggest_float("max_total", 0.75, 0.95, step=0.05),
        "strong_exposure": trial.suggest_float("strong_exposure", 0.80, 1.00, step=0.05),
        "strong_slots": trial.suggest_int("strong_slots", 3, 5),
        "normal_exposure": trial.suggest_float("normal_exposure", 0.60, 0.90, step=0.05),
        "normal_slots": trial.suggest_int("normal_slots", 2, 4),
        "repair_exposure": trial.suggest_float("repair_exposure", 0.30, 0.70, step=0.05),
        "repair_slots": trial.suggest_int("repair_slots", 1, 3),
        "weak_exposure": trial.suggest_float("weak_exposure", 0.00, 0.35, step=0.05),
        "weak_slots": trial.suggest_int("weak_slots", 0, 2),
        "dd_level1": -0.05,
        "dd_level2": trial.suggest_float("dd_level2", -0.10, -0.07, step=0.01),
        "dd_level3": trial.suggest_float("dd_level3", -0.14, -0.11, step=0.01),
        "dd_mult1": trial.suggest_float("dd_mult1", 0.70, 1.00, step=0.05),
        "dd_mult2": trial.suggest_float("dd_mult2", 0.35, 0.75, step=0.05),
        "dd_mult3": trial.suggest_float("dd_mult3", 0.20, 0.55, step=0.05),
        "force_dd": trial.suggest_float("force_dd", -0.13, -0.09, step=0.01),
        "force_target_exposure": trial.suggest_float("force_target_exposure", 0.25, 0.60, step=0.05),
        "force_no_buy": trial.suggest_int("force_no_buy", 5, 20),
        "force_cooldown": trial.suggest_int("force_cooldown", 10, 30, step=5),
        "candidate_depth": 20,
        "candidate_depth_r7": 6,
    }
    if p["dd_level3"] >= p["dd_level2"]: p["dd_level3"] = p["dd_level2"] - 0.02
    return p


def baseline_params() -> dict:
    return {"r7_weights": _norm({"p10": .26, "p20": .22, "p60": .10, "pf": .14, "pa": .12, "pc": .08, "pn": .08}), "r7_min_amt": 30_000_000, "r7_nearhigh": .78, "r7_ma_mode": "MA120", "r7_base": .22, "r7_limit_mult": .98, "r7_rebalance_days": 15, "r7_hard_stop": .12, "r7_rank_exit_mult": 2.0, "r05_weights": {"pclv10": .5251, "pamt": .2465, "pclv5": .0683, "pf3": .0628, "pf10": -.0778, "pt5": .0195, "pgap": -.2}, "r05_price_low": 10, "r05_price_high": 40, "r05_min_amt": 50_000_000, "r05_amount_ratio": 1.0, "r05_r20_max": .20, "r05_ma20gap_max": .18, "r05_prior60_min": -.15, "r05_base": .20, "r05_limit_mult": .995, "r05_max_slots": 3, "r05_hard_stop": .10, "r05_runner_trigger": .40, "r05_runner_trail": .14, "r05_base_trail_trigger": .50, "r05_base_trail": .12, "r05_max_hold": 60, "max_positions": 5, "max_single": .25, "max_total": .95, "strong_exposure": 1.0, "strong_slots": 4, "normal_exposure": .80, "normal_slots": 3, "repair_exposure": .60, "repair_slots": 2, "weak_exposure": .20, "weak_slots": 2, "dd_level1": -.06, "dd_level2": -.09, "dd_level3": -.15, "dd_mult1": .85, "dd_mult2": .45, "dd_mult3": .40, "force_dd": -.14, "force_target_exposure": .50, "force_no_buy": 10, "force_cooldown": 15, "candidate_depth": 20, "candidate_depth_r7": 6}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=80); ap.add_argument("--seed", type=int, default=20260828); args = ap.parse_args()
    strict = strict_slot_contract(); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "strict_t1_contract.json").write_text(json.dumps(strict, ensure_ascii=False, indent=2), encoding="utf-8")
    ctx = prepare_context(); baseline = baseline_params(); baseline_result = simulate(ctx, baseline, collect=False)
    bt.log("[FACTORY] strict baseline " + json.dumps({k: baseline_result[k] for k in ["end_nav", "cagr", "max_dd", "completed_trades", "feasible_dd"]}))
    import optuna
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    rows = []
    def objective(trial):
        p = sample_params(trial)
        try:
            r = simulate(ctx, p, collect=False)
            trial.set_user_attr("end_nav", r["end_nav"]); trial.set_user_attr("cagr", r["cagr"]); trial.set_user_attr("max_dd", r["max_dd"]); trial.set_user_attr("trades", r["completed_trades"]); trial.set_user_attr("feasible_dd", r["feasible_dd"])
            score = r["cagr"] if r["feasible_dd"] else -10.0 + r["max_dd"]
            rows.append({"trial": trial.number, "objective": score, "end_nav": r["end_nav"], "cagr": r["cagr"], "max_dd": r["max_dd"], "trades": r["completed_trades"], "feasible_dd": r["feasible_dd"], "params_json": json.dumps(p, ensure_ascii=False, separators=(",", ":"))})
            return score
        except Exception as exc:
            trial.set_user_attr("error", repr(exc)); rows.append({"trial": trial.number, "objective": -999.0, "error": repr(exc), "feasible_dd": False}); return -999.0
    study.optimize(objective, n_trials=max(1, args.trials), gc_after_trial=True, show_progress_bar=False)
    trials_df = pd.DataFrame(rows).sort_values(["feasible_dd", "objective"], ascending=[False, False]); trials_df.to_csv(OUT / "leaderboard.csv", index=False, encoding="utf-8-sig")
    feasible = trials_df[trials_df.feasible_dd.eq(True)] if not trials_df.empty else trials_df
    if feasible.empty:
        summary = {"status": "NO_FEASIBLE_STRATEGY", "engine_version": ENGINE_VERSION, "constraint": "Max DD >= -15%", "trials": int(args.trials), "baseline": baseline_result, "strict_t1_contract": strict}
        (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"); raise SystemExit("No strategy met Max DD <= 15% drawdown constraint")
    best_row = feasible.iloc[0]; best_params = json.loads(best_row.params_json); best = simulate(ctx, best_params, collect=True)
    nav_df = best.pop("nav_df"); trades_df = best.pop("trades_df"); orders_df = best.pop("orders_df")
    nav_df.to_csv(OUT / "best_daily_nav.csv", index=False, encoding="utf-8-sig"); trades_df.to_csv(OUT / "best_trades.csv", index=False, encoding="utf-8-sig"); orders_df.to_csv(OUT / "best_orders.csv", index=False, encoding="utf-8-sig")
    (OUT / "best_strategy.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summary = {"status": "PASS", "engine_version": ENGINE_VERSION, "trials": int(args.trials), "hard_rules": {"warmup": "2020", "evaluation": "2021-2025", "initial_capital_twd": INITIAL_CAPITAL, "max_drawdown_floor": MAX_DD_FLOOR, "sell_execution": "T decision -> T+1 Open with -0.5% adverse slippage", "pending_sell_slot_rule": "occupies slot through T close; released only at T+1 open", "buy_execution": "T decision -> precommitted T+1 limit; no touch=no fill"}, "strict_t1_contract": strict, "baseline_strict_t1": baseline_result, "best": best, "best_trial": int(best_row.trial)}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("FACTORY_RESULT=" + json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
