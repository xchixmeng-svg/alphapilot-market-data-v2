#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaPilot R10 clean causal 2021-2025 validation.

This engine is deliberately forbidden from reading any historical order, trade,
exit-price, NAV, or Golden-Master fixture. It regenerates everything from raw
OHLCV + institutional history + the documented R10 rules.

Historical reference ledgers may be compared ONLY by a separate post-run audit.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import r10_fast_validation as fast

bt = fast.bt
core = fast.core

ENGINE_VERSION = "AlphaPilot-R10-TRUE-CAUSAL-v1"


def _hash_frame(df: pd.DataFrame, cols: list[str]) -> str:
    x = df[cols].copy().sort_values(cols[:2] if len(cols) >= 2 else cols).reset_index(drop=True)
    payload = x.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def size_shares_true(target_cash: float, limit: float, avg_vol20: float) -> Tuple[int, str]:
    """Documented R10 quantity rule with no outcome fixture dependency.

    - normal stocks: floor to whole 1,000-share board lots;
    - odd lots only when one board lot itself exceeds the effective target;
    - liquidity <= 2% of T-known 20D average volume;
    - never round above the effective target.
    """
    if target_cash <= 0 or limit <= 0 or not np.isfinite(avg_vol20):
        return 0, "NONE"
    one_lot = limit * 1000.0
    if one_lot > target_cash + 1e-9:
        shares = int(math.floor(target_cash / limit + 1e-12))
        liq = int(math.floor(float(avg_vol20) * bt.ADV_CAP + 1e-12))
        return max(0, min(shares, max(0, liq))), "HIGH_PRICE_ODDLOT"
    lots = int(math.floor(target_cash / one_lot + 1e-12))
    liq_lots = int(math.floor(float(avg_vol20) * bt.ADV_CAP / 1000.0 + 1e-12))
    return max(0, min(lots, max(0, liq_lots)) * 1000), "BOARD_LOT"


def _position_mv(p, feat_idx, di: int) -> float:
    r = bt.row_lookup(feat_idx, di, p.code)
    px = float(r.close) if r is not None and np.isfinite(r.close) else p.entry_price
    return float(p.shares) * px


def simulate_true() -> dict:
    # Explicitly restore the documented R10 controls. No benchmark-validation
    # profile is allowed to disable them.
    bt.ADV_CAP = fast.full._BASE_ADV_CAP
    bt.FORCE_DD = fast.full._BASE_FORCE_DD
    bt.dd_multiplier = fast.full._BASE_DD_MULTIPLIER
    bt.VERSION = ENGINE_VERSION

    cfg = bt.SCENARIOS["validation2021_2025"]
    raw = bt.load_scenario_ohlcv(cfg)
    feat, corp_events, bm = bt.build_features(raw)
    eval_start, eval_end = bt.intdate(cfg["eval_start"]), bt.intdate(cfg["eval_end"])
    eval_dates = sorted(int(x) for x in feat.date.unique() if eval_start <= int(x) <= eval_end)
    next_date = {eval_dates[i]: eval_dates[i + 1] for i in range(len(eval_dates) - 1)}
    feat_idx = feat.set_index(["date", "code"]).sort_index()

    ins = fast.fetch_institutional_fast(feat, eval_start, eval_end)
    by_date, r7_states, r7_cands_map, r05_states, r05_cands_map = fast.precompute_signals(
        feat, bm, ins, eval_dates
    )

    cash = float(bt.INITIAL_CAPITAL)
    positions: Dict[str, bt.Position] = {}
    pending_buys: Dict[int, List[bt.BuyOrder]] = {}
    pending_sells: Dict[int, List[bt.SellOrder]] = {}
    nav_rows: list[dict] = []
    order_rows: list[dict] = []
    trade_rows: list[dict] = []
    event_rows: list[dict] = []

    hwm = cash
    last_regime = None
    last_reb_i = None
    no_buy_until = -1
    force_cooldown_until = -1
    forced_count = 0

    for i, di in enumerate(eval_dates):
        # T+1 execution ordering is deterministic: all sells first, then buys.
        for o in pending_sells.pop(di, []):
            k = bt.pos_key(o.strategy, o.code)
            p = positions.get(k)
            if p is None:
                continue
            r = bt.row_lookup(feat_idx, di, p.code)
            if r is None or not np.isfinite(r.open):
                if di in next_date:
                    o.execute_date = next_date[di]
                    pending_sells.setdefault(o.execute_date, []).append(o)
                continue
            px = bt.legal_sell_price(float(r.open))
            gross = px * p.shares
            proceeds = gross * (1.0 - bt.SELL_FEE - bt.SELL_TAX)
            cash += proceeds
            pnl = proceeds - p.cost_total
            trade_rows.append({
                "strategy": p.strategy, "code": p.code, "name": p.name,
                "entry_date": p.entry_date, "exit_date": di,
                "entry_price": p.entry_price, "exit_price": px, "shares": p.shares,
                "cost_total": p.cost_total, "proceeds": proceeds, "pnl": pnl,
                "return": pnl / p.cost_total if p.cost_total else np.nan,
                "exit_reason": o.reason, "hold_days": p.hold_days, "mode": p.mode,
            })
            order_rows.append({
                "decision_date": o.decision_date, "execute_date": di,
                "strategy": p.strategy, "side": "SELL", "code": p.code, "name": p.name,
                "order_price": np.nan, "shares": p.shares, "filled": True,
                "fill_price": px, "reason": o.reason,
            })
            del positions[k]

        for o in pending_buys.pop(di, []):
            r = bt.row_lookup(feat_idx, di, o.code)
            fill = None
            if r is not None and np.isfinite(r.open) and np.isfinite(r.low):
                fill = bt.buy_fill(float(r.open), float(r.low), float(o.limit))
            filled = fill is not None
            reason = "FILLED" if filled else "LIMIT_NOT_TOUCHED"
            if filled:
                cost = float(fill) * o.shares * (1.0 + bt.BUY_FEE)
                if cost > cash + 1e-6:
                    filled = False
                    reason = "CASH_SHORT_AT_EXECUTION"
                else:
                    cash -= cost
                    factor = float(r.aclose / r.close) if np.isfinite(r.aclose) and r.close else 1.0
                    entry_adj = float(fill) * factor
                    positions[bt.pos_key(o.strategy, o.code)] = bt.Position(
                        o.strategy, o.code, o.name, int(o.shares), di,
                        float(fill), entry_adj, cost, entry_adj,
                    )
            order_rows.append({
                "decision_date": o.decision_date, "execute_date": di,
                "strategy": o.strategy, "side": "BUY", "code": o.code, "name": o.name,
                "order_price": o.limit, "shares": o.shares, "filled": bool(filled),
                "fill_price": float(fill) if filled else np.nan, "reason": reason,
                "target_cash": o.target_cash, "reserved_cash": o.reserved_cash, "rank": o.rank,
            })

        nav, stock_mv = bt.mark_nav(cash, positions, feat_idx, di)
        hwm = max(hwm, nav)
        dd = nav / hwm - 1.0
        exposure = stock_mv / nav if nav > 0 else 0.0

        # Peaks are updated using information known at T close only.
        for p in positions.values():
            r = bt.row_lookup(feat_idx, di, p.code)
            if r is not None and np.isfinite(r.aclose):
                p.peak_adj = max(p.peak_adj, float(r.aclose))

        r7_state = r7_states[di]
        r7_cands = r7_cands_map[di]
        r05_state = r05_states[di]
        r05_cands = r05_cands_map[di]
        regime = r7_state["regime"]
        regime_changed = last_regime is None or regime != last_regime
        reb_due = last_reb_i is None or regime_changed or (i - last_reb_i) >= 15
        if reb_due:
            last_reb_i = i
        last_regime = regime

        r7_rank = {str(x.code): int(x.r7_rank) for x in r7_cands[["code", "r7_rank"]].itertuples(index=False)}
        r7_score = {str(x.code): float(x.r7_score) for x in r7_cands[["code", "r7_score"]].itertuples(index=False)}
        r05_score = {} if r05_cands.empty else {
            str(x.code): float(x.r05_score) for x in r05_cands[["code", "r05_score"]].itertuples(index=False)
        }

        sell_map: dict[str, bt.SellOrder] = {}
        if di in next_date:
            exdate = next_date[di]
            for k, p in list(positions.items()):
                r = bt.row_lookup(feat_idx, di, p.code)
                reason = None
                if p.strategy == "R7":
                    p.hold_days += 1
                    if r is not None and np.isfinite(r.aclose) and float(r.aclose) <= p.entry_adj * 0.88 + 1e-12:
                        reason = "HARD"
                    elif reb_due:
                        n = int(r7_state["slots"])
                        rank = r7_rank.get(p.code, 10**9)
                        if float(r7_state["exposure"]) <= 0:
                            reason = "REB_REGIME0"
                        elif n <= 0 or rank > 2 * n:
                            reason = "REB_RANK"
                else:
                    reason = bt.r05_exit_reason(p, r)
                if reason:
                    sell_map[k] = bt.SellOrder(di, exdate, p.strategy, p.code, reason)

            # R7 regime exposure trim is based only on T-known close/rank state.
            exclude = set(sell_map)
            r7_mv = bt.value_of(positions, feat_idx, di, strategy="R7", exclude=exclude)
            r7_target = nav * float(r7_state["exposure"])
            if r7_mv > r7_target * 1.03 + 1.0:
                remain = [(k, p) for k, p in positions.items() if p.strategy == "R7" and k not in exclude]
                remain.sort(key=lambda kp: r7_rank.get(kp[1].code, 10**9), reverse=True)
                projected = r7_mv
                for k, p in remain:
                    if projected <= r7_target:
                        break
                    mv = _position_mv(p, feat_idx, di)
                    sell_map[k] = bt.SellOrder(di, exdate, p.strategy, p.code, "EXPO")
                    projected -= mv

            # Documented portfolio DD defense: T decision, T+1 execution.
            if dd <= bt.FORCE_DD and i >= force_cooldown_until:
                force_cooldown_until = i + bt.FORCE_COOLDOWN_DAYS
                no_buy_until = max(no_buy_until, i + bt.FORCE_NO_BUY_DAYS)
                forced_count += 1
                exclude = set(sell_map)
                projected = bt.value_of(positions, feat_idx, di, exclude=exclude)
                force_target = nav * bt.FORCE_TARGET_EXPOSURE
                remain = [(k, p) for k, p in positions.items() if k not in exclude]

                def weakness(item):
                    _, p = item
                    if p.strategy == "R7":
                        return (0, -r7_rank.get(p.code, 10**9), r7_score.get(p.code, -1e9))
                    rr = bt.row_lookup(feat_idx, di, p.code)
                    ret = (float(rr.aclose) / p.entry_adj - 1.0) if rr is not None and np.isfinite(rr.aclose) else -9.0
                    return (1, r05_score.get(p.code, -1e9), ret)

                remain.sort(key=weakness)
                for k, p in remain:
                    if projected <= force_target:
                        break
                    mv = _position_mv(p, feat_idx, di)
                    sell_map[k] = bt.SellOrder(di, exdate, p.strategy, p.code, "FORCE_DD")
                    projected -= mv
                event_rows.append({
                    "date": di, "event": "FORCE_DD", "dd": dd,
                    "target_exposure": bt.FORCE_TARGET_EXPOSURE,
                })

            if sell_map:
                pending_sells.setdefault(exdate, []).extend(sell_map.values())

        created: list[bt.BuyOrder] = []
        if di in next_date and i >= no_buy_until:
            exdate = next_date[di]
            sell_keys = set(sell_map)

            # Planned T+1 sells execute before buys, so they release T+1 slots
            # and projected exposure. Their unknown T+1 proceeds are NOT guessed;
            # actual affordability is checked after the sells execute.
            codes_after = {p.code for k, p in positions.items() if k not in sell_keys}
            base_exposure = bt.value_of(positions, feat_idx, di, exclude=sell_keys)
            base_r7 = bt.value_of(positions, feat_idx, di, strategy="R7", exclude=sell_keys)
            reserved_exposure = 0.0
            reserved_r7 = 0.0
            reserved_code: dict[str, float] = {}

            def try_order(strategy: str, row, rank: int):
                nonlocal reserved_exposure, reserved_r7, codes_after
                code = str(row.code)
                name0 = str(row.name)
                k = bt.pos_key(strategy, code)
                if k in positions and k not in sell_keys:
                    return
                if code not in codes_after and len(codes_after) >= bt.MAX_POSITIONS:
                    return

                if strategy == "R05":
                    n = sum(1 for kk, p in positions.items() if p.strategy == "R05" and kk not in sell_keys)
                    n += sum(1 for o in created if o.strategy == "R05")
                    if n >= bt.R05_MAX_SLOTS:
                        return
                    base_pct = bt.R05_BASE
                    limit = float(core.floor_tick(float(row.close) * 0.995))
                else:
                    n = sum(1 for kk, p in positions.items() if p.strategy == "R7" and kk not in sell_keys)
                    n += sum(1 for o in created if o.strategy == "R7")
                    if n >= int(r7_state["slots"]):
                        return
                    base_pct = bt.R7_BASE
                    limit = float(core.floor_tick(float(row.close) * 0.98))

                current_code = bt.value_of(positions, feat_idx, di, code=code, exclude=sell_keys)
                current_code += reserved_code.get(code, 0.0)
                rem_single = nav * bt.MAX_SINGLE - current_code
                rem_global = nav * bt.MAX_TOTAL - base_exposure - reserved_exposure
                base_target = nav * base_pct
                target = base_target * bt.dd_multiplier(dd)
                if strategy == "R7":
                    r7_cap = nav * float(r7_state["exposure"])
                    target = min(target, r7_cap - base_r7 - reserved_r7)
                target = min(target, rem_single, rem_global)
                if target <= 0:
                    return

                shares, share_mode = size_shares_true(target, limit, float(row.avgvol20))
                if shares <= 0:
                    return
                notional = shares * limit
                reserve = notional * (1.0 + bt.BUY_FEE)
                if notional > target + 1e-6 or notional > rem_single + 1e-6 or notional > rem_global + 1e-6:
                    raise RuntimeError(
                        f"sizing cap breach {di} {strategy} {code} mode={share_mode} "
                        f"notional={notional} target={target} single={rem_single} global={rem_global}"
                    )
                if strategy == "R7":
                    r7_cap = nav * float(r7_state["exposure"])
                    if base_r7 + reserved_r7 + notional > r7_cap + 1e-6:
                        raise RuntimeError(f"R7 exposure cap breach {di} {code}")

                created.append(bt.BuyOrder(
                    di, exdate, strategy, code, name0, limit, shares, target, reserve, rank
                ))
                reserved_exposure += notional
                reserved_code[code] = reserved_code.get(code, 0.0) + notional
                if strategy == "R7":
                    reserved_r7 += notional
                codes_after.add(code)

            if bool(r05_state.get("risk_on")) and not r05_cands.empty:
                for _, row in r05_cands.head(20).iterrows():
                    try_order("R05", row, int(row.r05_rank))
            if reb_due and float(r7_state["exposure"]) > 0 and not r7_cands.empty:
                for _, row in r7_cands.head(max(0, int(r7_state["slots"]))).iterrows():
                    try_order("R7", row, int(row.r7_rank))
            if created:
                pending_buys.setdefault(exdate, []).extend(created)

        nav_rows.append({
            "date": di, "nav": nav, "cash": cash, "stock_mv": stock_mv,
            "exposure": exposure, "drawdown": dd,
            "positions": len({p.code for p in positions.values()}),
            "r7_positions": sum(p.strategy == "R7" for p in positions.values()),
            "r05_positions": sum(p.strategy == "R05" for p in positions.values()),
            "r7_regime": regime, "r7_regime_exposure": r7_state["exposure"],
            "r7_rebalance_due": reb_due,
            "r05_risk_on": bool(r05_state.get("risk_on", False)),
            "dd_multiplier": bt.dd_multiplier(dd), "no_buy_active": i < no_buy_until,
        })
        if (i + 1) % 100 == 0 or i + 1 == len(eval_dates):
            bt.log(f"[TRUE-SIM] {i+1}/{len(eval_dates)} date={di} nav={nav:,.0f} trades={len(trade_rows)}")

    nav_df = pd.DataFrame(nav_rows)
    trades_df = pd.DataFrame(trade_rows)
    orders_df = pd.DataFrame(order_rows)
    events_df = pd.DataFrame(event_rows)
    end_nav = float(nav_df.iloc[-1].nav)
    years = max((bt.dt_from_int(eval_dates[-1]) - bt.dt_from_int(eval_dates[0])).days / 365.25, 1 / 365.25)
    cagr = (end_nav / bt.INITIAL_CAPITAL) ** (1.0 / years) - 1.0
    max_dd = float(nav_df.drawdown.min())
    buy_orders = int((orders_df.side == "BUY").sum()) if not orders_df.empty else 0
    fills = int(((orders_df.side == "BUY") & orders_df.filled.eq(True)).sum()) if not orders_df.empty else 0

    nav_df["year"] = nav_df.date.astype(str).str[:4].astype(int)
    annual = {}
    prev = bt.INITIAL_CAPITAL
    for y, g in nav_df.groupby("year"):
        e = float(g.iloc[-1].nav)
        annual[str(y)] = e / prev - 1.0
        prev = e

    provenance = {
        "engine_version": ENGINE_VERSION,
        "fixture_inputs_used": False,
        "ohlcv_source_template": bt.RELEASE,
        "ohlcv_years": cfg["years"],
        "ohlcv_rows": int(len(raw)),
        "ohlcv_sha256": _hash_frame(raw, ["date", "code", "open", "high", "low", "close", "volume"]),
        "institutional_rows": int(len(ins)),
        "institutional_eval_dates": int(ins[ins.date.between(eval_start, eval_end)].date.nunique()),
        "institutional_sha256": _hash_frame(
            ins.fillna(0), ["date", "code", "foreign_net", "trust_net", "dealer_net", "Foreign3D", "Foreign10D", "Trust5D"]
        ),
        "causality": "T-close decision; T+1 sells before T+1 buys; raw T+1 Open/Low execution only",
    }

    result = {
        "status": "PASS", "engine_version": ENGINE_VERSION,
        "scenario": "validation2021_2025", "initial_nav": bt.INITIAL_CAPITAL,
        "end_nav": end_nav, "total_return": end_nav / bt.INITIAL_CAPITAL - 1.0,
        "cagr": cagr, "max_dd": max_dd,
        "completed_trades": int(len(trades_df)), "orders": buy_orders, "fills": fills,
        "fill_rate": fills / buy_orders if buy_orders else 0.0,
        "min_cash": float(nav_df.cash.min()), "avg_exposure": float(nav_df.exposure.mean()),
        "max_exposure": float(nav_df.exposure.max()), "max_positions": int(nav_df.positions.max()),
        "force_dd_events": int(forced_count), "corporate_action_continuity_events": int(corp_events),
        "annual_returns": annual, "fixture_inputs_used": False,
        "future_data_used": False,
    }

    out = bt.OUT_ROOT / "latest" / "true_validation2021_2025"
    out.mkdir(parents=True, exist_ok=True)
    nav_df.to_csv(out / "daily_nav.csv", index=False, encoding="utf-8-sig")
    trades_df.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    orders_df.to_csv(out / "orders.csv", index=False, encoding="utf-8-sig")
    events_df.to_csv(out / "risk_events.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out / "data_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    bt.log(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    simulate_true()
