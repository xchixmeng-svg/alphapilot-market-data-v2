#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaPilot R10-MAX 0.5% locked-rule stress backtest.

Runs reproducible stress-period backtests using the LOCKED R10-MAX 0.5%
portfolio rules. All signals are formed at T close and execute T+1 only.

Critical data boundary: TWSE stock-level daily institutional T86 does not
cover 2008. The GFC scenario therefore disables R0.5 rather than fabricating
institutional features and is labelled PARTIAL_R10_R7_ONLY. 2020+ scenarios
can run FULL_R10.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import build_r10_scan as core  # noqa: E402

VERSION = "AlphaPilot-R10-MAX-0p5-Stress-v1.0"
INITIAL_CAPITAL = 1_300_000.0
BUY_FEE = 0.000855
SELL_FEE = 0.000855
SELL_TAX = 0.003
SELL_SLIPPAGE = 0.005
MAX_POSITIONS = 5
MAX_SINGLE = 0.25
MAX_TOTAL = 0.95
R7_BASE = 0.22
R05_BASE = 0.20
R05_MAX_SLOTS = 3
ADV_CAP = 0.02
FORCE_DD = -0.14
FORCE_TARGET_EXPOSURE = 0.50
FORCE_COOLDOWN_DAYS = 15
FORCE_NO_BUY_DAYS = 10

RELEASE = (
    "https://github.com/yukishirotsubasa/tw-stock-data-release/"
    "releases/download/daily-close-csv/yearly_{year}.zip"
)
OUT_ROOT = ROOT / "stress_results"
CACHE_ROOT = ROOT / ".stress_cache"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 AlphaPilot-R10-Stress/1.0", "Accept": "*/*"})

SCENARIOS = {
    "gfc2008": {
        "label": "2008 Global Financial Crisis + 2009 recovery",
        "warmup_start": "2007-01-01",
        "eval_start": "2008-01-02",
        "eval_end": "2009-12-31",
        "years": [2007, 2008, 2009],
        "r05": False,
        "mode": "PARTIAL_R10_R7_ONLY",
        "reason": "TWSE stock-level daily institutional T86 data are unavailable for 2008; R0.5 is disabled rather than fabricated.",
    },
    "covid2020": {
        "label": "2020 COVID crash + V recovery",
        "warmup_start": "2019-01-01",
        "eval_start": "2020-01-02",
        "eval_end": "2020-12-31",
        "years": [2019, 2020],
        "r05": True,
        "mode": "FULL_R10",
        "reason": "Full R7 + R0.5 + R10 portfolio layer.",
    },
    "bear2022": {
        "label": "2022 rate-hike / bear market",
        "warmup_start": "2021-01-01",
        "eval_start": "2022-01-03",
        "eval_end": "2022-12-30",
        "years": [2021, 2022],
        "r05": True,
        "mode": "FULL_R10",
        "reason": "Full R7 + R0.5 + R10 portfolio layer.",
    },
    "validation2021_2025": {
        "label": "Locked-sample regression validation 2021-2025",
        "warmup_start": "2020-01-01",
        "eval_start": "2021-01-04",
        "eval_end": "2025-12-31",
        "years": [2020, 2021, 2022, 2023, 2024, 2025],
        "r05": True,
        "mode": "FULL_R10_VALIDATION",
        "reason": "Regression mode against locked benchmark.",
    },
}

# Historical reference retained only for forensic comparison. It is NOT a
# valid causal regression target: the inherited R0.5 ledger used exit-day
# intraday stop/trailing information to select the exit date, then repriced that
# same exit at the already-passed opening price (confirmed 15/15 hard stops and
# 2/2 trailing exits).
LEGACY_CONTAMINATED_BENCHMARK = {
    "end_nav": 9_888_538.413551485,
    "cagr": 0.5019270634416155,
    "max_dd": -0.12258760312884043,
    "completed_trades": 241,
    "causal": False,
}

# Zero-look-ahead reference produced by the documented T-close -> T+1 engine
# after repairing buy-side 0.5% adverse execution. This is the active regression
# target; it must be reproduced before long historical/capital-path results are
# accepted.
CAUSAL_BENCHMARK = {
    "end_nav": 4_020_109.243251493,
    "cagr": 0.2539714041310821,
    "max_dd": -0.1813892621615033,
    "completed_trades": 217,
    "causal": True,
}

# Backward-compatible alias. All validation deltas now point to the causal gate.
LOCKED_BENCHMARK = CAUSAL_BENCHMARK

@dataclass
class Position:
    strategy: str
    code: str
    name: str
    shares: int
    entry_date: int
    entry_price: float
    entry_adj: float
    cost_total: float
    peak_adj: float
    mode: str = "NORMAL"
    hold_days: int = 0

@dataclass
class BuyOrder:
    decision_date: int
    execute_date: int
    strategy: str
    code: str
    name: str
    limit: float
    shares: int
    target_cash: float
    reserved_cash: float
    rank: int

@dataclass
class SellOrder:
    decision_date: int
    execute_date: int
    strategy: str
    code: str
    reason: str

def log(msg: str) -> None:
    print(msg, flush=True)

def intdate(s: str | date) -> int:
    if isinstance(s, date):
        return int(s.strftime("%Y%m%d"))
    return int(pd.Timestamp(s).strftime("%Y%m%d"))

def dt_from_int(v: int) -> date:
    return datetime.strptime(str(int(v)), "%Y%m%d").date()

def request_bytes(url: str, tries: int = 5) -> bytes:
    last = None
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=(20, 120))
            r.raise_for_status()
            if not r.content:
                raise RuntimeError("empty response")
            return r.content
        except Exception as exc:
            last = exc
            if i + 1 < tries:
                time.sleep(min(20, 2 ** i))
    raise RuntimeError(f"download failed {url}: {last}")

def normalize_ohlcv(q: pd.DataFrame) -> pd.DataFrame:
    need = ["date", "code", "name", "volume", "open", "high", "low", "close"]
    miss = [c for c in need if c not in q.columns]
    if miss:
        raise RuntimeError(f"OHLCV missing columns: {miss}")
    q = q[need].copy()
    q["code"] = q["code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4)
    q["name"] = q["name"].astype(str).str.strip()
    raw_date = q["date"].astype(str).str.strip()
    parsed_date = pd.to_datetime(raw_date, errors="coerce")
    digits = raw_date.str.replace(r"\D", "", regex=True)
    q["date"] = np.where(
        digits.str.len().eq(8),
        pd.to_numeric(digits, errors="coerce"),
        pd.to_numeric(parsed_date.dt.strftime("%Y%m%d"), errors="coerce"),
    )
    for c in ["date", "volume", "open", "high", "low", "close"]:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q.dropna(subset=["date", "code", "open", "high", "low", "close"])
    q["date"] = q["date"].astype(int)
    q["volume"] = q["volume"].fillna(0).astype(int)
    q = q.drop_duplicates(["date", "code"], keep="last")
    return q.sort_values(["code", "date"]).reset_index(drop=True)

def load_year(year: int) -> pd.DataFrame:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = CACHE_ROOT / f"yearly_{year}.csv"
    if cache.exists():
        return normalize_ohlcv(pd.read_csv(cache, dtype={"code": str}))
    url = RELEASE.format(year=year)
    log(f"[OHLCV] download yearly_{year}.zip")
    blob = request_bytes(url)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"yearly_{year}.zip contains no csv")
        q = pd.read_csv(zf.open(names[0]), dtype={"code": str})
    q = normalize_ohlcv(q)
    q.to_csv(cache, index=False, encoding="utf-8-sig")
    return q

def load_scenario_ohlcv(cfg: dict) -> pd.DataFrame:
    q = pd.concat([load_year(y) for y in cfg["years"]], ignore_index=True)
    lo, hi = intdate(cfg["warmup_start"]), intdate(cfg["eval_end"])
    q = q[(q.date >= lo) & (q.date <= hi)].copy()
    if q.empty or "0050" not in set(q.code):
        raise RuntimeError("scenario OHLCV/0050 missing")
    return q

def build_features(raw: pd.DataFrame) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    adj, events = core.adjust(raw)
    feat = core.features(adj)
    # 20D average volume is part of locked 2% ADV sizing.
    feat["avgvol20"] = (
        feat.groupby("code", sort=False).volume.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    )
    bm = feat[feat.code.eq("0050")][["date", "aclose"]].dropna().copy()
    bm = bm.rename(columns={"aclose": "mkt"}).drop_duplicates("date").sort_values("date")
    return feat, events, bm

def build_inst_features(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date","market","code","foreign_net","trust_net","dealer_net","Foreign3D","Foreign10D","Trust5D"])
    q = pd.DataFrame(rows).drop_duplicates(["date", "market", "code"], keep="last").sort_values(["market", "code", "date"])
    for c in ("foreign_net", "trust_net", "dealer_net"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    g = q.groupby(["market", "code"], sort=False)
    q["Foreign3D"] = g.foreign_net.transform(lambda s: s.rolling(3, min_periods=3).sum())
    q["Foreign10D"] = g.foreign_net.transform(lambda s: s.rolling(10, min_periods=10).sum())
    q["Trust5D"] = g.trust_net.transform(lambda s: s.rolling(5, min_periods=5).sum())
    return q

def fetch_institutional(feat: pd.DataFrame, eval_start: int, eval_end: int) -> pd.DataFrame:
    min_dt = dt_from_int(eval_start) - timedelta(days=50)
    dates = [int(x) for x in sorted(feat.date.unique()) if min_dt <= dt_from_int(int(x)) <= dt_from_int(eval_end)]
    rows: List[dict] = []
    failures = []
    for n, di in enumerate(dates, 1):
        d = dt_from_int(di)
        a = b = []
        try:
            a = core.twse_inst(d)
        except Exception as exc:
            failures.append(f"{di} TWSE {exc}")
        time.sleep(0.05)
        try:
            b = core.tpex_inst(d)
        except Exception as exc:
            failures.append(f"{di} TPEX {exc}")
        rows.extend(a); rows.extend(b)
        if n % 25 == 0:
            log(f"[INST] {n}/{len(dates)} dates rows={len(rows)} failures={len(failures)}")
        time.sleep(0.05)
    ins = build_inst_features(rows)
    coverage = ins[ins.date.between(eval_start, eval_end)].date.nunique()
    target_dates = len([d for d in dates if eval_start <= d <= eval_end])
    ratio = coverage / target_dates if target_dates else 0.0
    log(f"[INST] coverage={coverage}/{target_dates} ({ratio:.1%})")
    if ratio < 0.90:
        raise RuntimeError(f"institutional coverage too low {ratio:.1%}: {'; '.join(failures[:8])}")
    return ins

def dd_multiplier(dd: float) -> float:
    if dd <= -0.15: return 0.40
    if dd <= -0.09: return 0.45
    if dd <= -0.06: return 0.85
    return 1.0

def legal_sell_price(open_price: float) -> float:
    return float(core.floor_tick(float(open_price) * (1.0 - SELL_SLIPPAGE)))

def _tick_size(price: float) -> float:
    # Taiwan equity tick schedule used for adverse buy-fill rounding.
    if price < 10: return 0.01
    if price < 50: return 0.05
    if price < 100: return 0.10
    if price < 500: return 0.50
    if price < 1000: return 1.00
    return 5.00

def _ceil_tick(price: float) -> float:
    # Golden Master first rounds the adverse execution estimate to cents,
    # then rounds UP to the next legal Taiwan tick.
    rounded = round(float(price), 2)
    step = _tick_size(rounded)
    q = math.ceil((rounded - 1e-12) / step) * step
    return float(round(q, 10))

def buy_fill(open_price: float, low_price: float, limit: float) -> Optional[float]:
    # Golden Master execution: if T+1 opens through/below the precommitted
    # limit, model 0.5% adverse buy slippage, cents-round, legal-tick round UP,
    # but never pay above the locked limit. If only the intraday low touches
    # the limit, fill exactly at the locked limit. No touch => cancel/no chase.
    if open_price <= limit:
        return float(min(limit, _ceil_tick(float(open_price) * (1.0 + SELL_SLIPPAGE))))
    if low_price <= limit: return float(limit)
    return None

def pos_key(strategy: str, code: str) -> str:
    return f"{strategy}:{code}"

def size_shares(target_cash: float, base_target_cash: float, limit: float, avg_vol20: float) -> Tuple[int, str]:
    if target_cash <= 0 or base_target_cash <= 0 or limit <= 0 or not np.isfinite(avg_vol20):
        return 0, "NONE"
    per_share_cost = limit * (1.0 + BUY_FEE)
    one_lot_cost = per_share_cost * 1000.0

    # HIGH-PRICE EXCEPTION ONLY: one full board lot is already larger than the
    # strategy's normal 22%/20% base allocation. Only here may odd lots exist.
    if one_lot_cost > base_target_cash + 1e-9:
        mode = "HIGH_PRICE_ODDLOT"
        shares = max(1, int(math.floor(target_cash / per_share_cost + 0.5)))
        liq = int(math.floor(float(avg_vol20) * ADV_CAP))
        shares = min(shares, max(0, liq))
        return max(0, int(shares)), mode

    # Normal-price stocks: whole board lots only. Use the nearest board lot;
    # even a sub-lot computed target becomes one full lot, never 578/867 shares.
    mode = "BOARD_LOT"
    raw_lots = target_cash / one_lot_cost
    lots = max(1, int(math.floor(raw_lots + 0.5)))
    liq_lots = int(math.floor(float(avg_vol20) * ADV_CAP / 1000.0))
    lots = min(lots, max(0, liq_lots))
    return max(0, lots * 1000), mode

def row_lookup(feat_idx, di: int, code: str):
    try:
        r = feat_idx.loc[(di, code)]
        return r.iloc[-1] if isinstance(r, pd.DataFrame) else r
    except KeyError:
        return None

def mark_nav(cash: float, positions: Dict[str, Position], feat_idx, di: int) -> Tuple[float, float]:
    mv = 0.0
    for p in positions.values():
        r = row_lookup(feat_idx, di, p.code)
        px = float(r.close) if r is not None and np.isfinite(r.close) else p.entry_price
        mv += p.shares * px
    return cash + mv, mv

def value_of(positions, feat_idx, di, strategy=None, code=None, exclude=None) -> float:
    exclude = exclude or set()
    total = 0.0
    for k, p in positions.items():
        if k in exclude: continue
        if strategy is not None and p.strategy != strategy: continue
        if code is not None and p.code != code: continue
        r = row_lookup(feat_idx, di, p.code)
        px = float(r.close) if r is not None else p.entry_price
        total += p.shares * px
    return total

def r05_exit_reason(p: Position, r) -> Optional[str]:
    if r is None or not np.isfinite(r.aclose): return None
    adj = float(r.aclose); ret = adj / p.entry_adj - 1.0
    p.peak_adj = max(p.peak_adj, adj); p.hold_days += 1
    ar = float(r.amount_ratio) if np.isfinite(r.amount_ratio) else 0.0
    eps = 1e-12
    if adj <= p.entry_adj * 0.90 + eps: return "HARD"
    if p.mode == "NORMAL" and ret >= 0.40 - eps and ar >= 2.0 - eps: p.mode = "RUNNER"
    if p.mode in {"RUNNER", "MEGA", "TARGET"} and ret >= 0.80 - eps:
        if ar >= 1.20 - eps: p.mode = "MEGA"
        elif p.mode != "MEGA": p.mode = "TARGET"
    if p.mode == "MEGA":
        if adj <= p.peak_adj * 0.84 + eps: return "MEGA_TRAIL"
        if p.hold_days >= 120: return "RUNNER_TIME"
        return None
    if p.mode == "TARGET":
        if ret >= 2.00 - eps: return "TARGET_200"
        if adj <= p.peak_adj * 0.80 + eps: return "TARGET_TRAIL"
        if p.hold_days >= 120: return "RUNNER_TIME"
        return None
    if p.mode == "RUNNER":
        if adj <= p.peak_adj * 0.86 + eps: return "RUNNER_TRAIL"
        if p.hold_days >= 120: return "RUNNER_TIME"
        return None
    if ret >= 0.50 - eps and adj <= p.peak_adj * 0.88 + eps: return "BASE_TRAIL"
    if p.hold_days >= 60: return "TIME"
    return None

def make_summary_md(results: List[dict]) -> str:
    lines = [
        "# AlphaPilot R10-MAX 0.5% Stress Backtest", "", f"Engine: `{VERSION}`", "",
        "## Critical interpretation", "",
        "- `FULL_R10` = R7 + R0.5 + locked R10 portfolio overlay.",
        "- `PARTIAL_R10_R7_ONLY` = R0.5 disabled because the required official daily institutional history does not exist; never present it as a full R10 result.", "",
        "| Scenario | Mode | End NAV | Return | CAGR | Max DD | Trades | Orders/Fills |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if r.get("status") != "PASS":
            lines.append(f"| {r.get('scenario')} | {r.get('mode','')} | ERROR | | | | | {r.get('error','')[:80]} |")
        else:
            lines.append("| {scenario} | {mode} | {end_nav:,.0f} | {total_return:.2%} | {cagr:.2%} | {max_dd:.2%} | {completed_trades} | {orders}/{fills} |".format(**r))
    lines += [
        "", "## Locked rules used", "",
        "- NT$1,300,000 common pool; no 62/38 sleeves.",
        "- R7 base 22% NAV; R0.5 base 20% NAV.",
        "- Max 5 stocks, 25% single-name, 95% normal total exposure.",
        "- DD throttle: -6%=>85%, -9%=>45%, -15%=>40%.",
        "- DD<=-14%: reduce toward 50%, 10 trading days no new buys, 15-day defensive cooldown.",
        "- T close locks T+1 buy limit/quantity; no touch=no fill/no chase.",
        "- T close sell decision; T+1 open with 0.5% adverse slippage.",
        "- Fee 0.0855% each side; sell tax 0.3%; board-lot first; 2% of T-known 20D ADV liquidity cap.", "",
        "## Regression benchmark", "",
        "Locked 2021-2025 benchmark: end NAV 9,888,538; CAGR 50.19%; Max DD -12.26%; 241 completed trades.",
        "Run `validation2021_2025` before treating reconstructed stress figures as official-equivalent.",
    ]
    return "\n".join(lines) + "\n"

def simulate(name: str, cfg: dict) -> dict:
    log(f"\n===== {name}: {cfg['label']} =====")
    raw = load_scenario_ohlcv(cfg)
    feat, corp_events, bm = build_features(raw)
    eval_start, eval_end = intdate(cfg["eval_start"]), intdate(cfg["eval_end"])
    eval_dates = sorted(int(x) for x in feat.date.unique() if eval_start <= x <= eval_end)
    if len(eval_dates) < 100: raise RuntimeError(f"too few evaluation dates: {len(eval_dates)}")
    next_date = {eval_dates[i]: eval_dates[i+1] for i in range(len(eval_dates)-1)}
    feat_idx = feat.set_index(["date", "code"]).sort_index()
    ins = fetch_institutional(feat, eval_start, eval_end) if cfg["r05"] else None

    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    pending_buys: Dict[int, List[BuyOrder]] = {}
    pending_sells: Dict[int, List[SellOrder]] = {}
    nav_rows, order_rows, trade_rows, event_rows = [], [], [], []
    hwm = INITIAL_CAPITAL; min_cash = cash; max_positions_seen = 0
    last_regime = None; last_reb_i = None; no_buy_until = -1; force_cooldown_until = -1; forced_count = 0

    for i, di in enumerate(eval_dates):
        # T+1 sells first.
        for o in pending_sells.pop(di, []):
            k = pos_key(o.strategy, o.code); p = positions.get(k)
            if p is None: continue
            r = row_lookup(feat_idx, di, p.code)
            if r is None or not np.isfinite(r.open):
                if di in next_date:
                    o.execute_date = next_date[di]; pending_sells.setdefault(o.execute_date, []).append(o)
                continue
            px = legal_sell_price(float(r.open)); gross = px * p.shares
            proceeds = gross * (1.0 - SELL_FEE - SELL_TAX); cash += proceeds
            pnl = proceeds - p.cost_total
            trade_rows.append({"strategy":p.strategy,"code":p.code,"name":p.name,"entry_date":p.entry_date,"exit_date":di,"entry_price":p.entry_price,"exit_price":px,"shares":p.shares,"cost_total":p.cost_total,"proceeds":proceeds,"pnl":pnl,"return":pnl/p.cost_total if p.cost_total else np.nan,"exit_reason":o.reason,"hold_days":p.hold_days,"mode":p.mode})
            order_rows.append({"decision_date":o.decision_date,"execute_date":di,"strategy":p.strategy,"side":"SELL","code":p.code,"name":p.name,"order_price":np.nan,"shares":p.shares,"filled":True,"fill_price":px,"reason":o.reason})
            del positions[k]

        # T+1 buys: fixed T order, no chasing.
        for o in pending_buys.pop(di, []):
            r = row_lookup(feat_idx, di, o.code); fill = None
            if r is not None: fill = buy_fill(float(r.open), float(r.low), float(o.limit))
            filled = fill is not None; reason = "FILLED" if filled else "LIMIT_NOT_TOUCHED"
            if filled:
                cost = float(fill) * o.shares * (1.0 + BUY_FEE)
                if cost > cash + 1e-6:
                    filled = False; reason = "CASH_SHORT_AT_EXECUTION"
                else:
                    cash -= cost
                    factor = float(r.aclose / r.close) if np.isfinite(r.aclose) and r.close else 1.0
                    entry_adj = float(fill) * factor
                    positions[pos_key(o.strategy, o.code)] = Position(o.strategy,o.code,o.name,int(o.shares),di,float(fill),entry_adj,cost,entry_adj)
            order_rows.append({"decision_date":o.decision_date,"execute_date":di,"strategy":o.strategy,"side":"BUY","code":o.code,"name":o.name,"order_price":o.limit,"shares":o.shares,"filled":bool(filled),"fill_price":float(fill) if filled else np.nan,"reason":reason,"target_cash":o.target_cash,"reserved_cash":o.reserved_cash,"rank":o.rank})

        nav, stock_mv = mark_nav(cash, positions, feat_idx, di)
        hwm = max(hwm, nav); dd = nav / hwm - 1.0; exposure = stock_mv / nav if nav > 0 else 0.0
        min_cash = min(min_cash, cash); max_positions_seen = max(max_positions_seen, len({p.code for p in positions.values()}))
        for p in positions.values():
            r = row_lookup(feat_idx, di, p.code)
            if r is not None and np.isfinite(r.aclose): p.peak_adj = max(p.peak_adj, float(r.aclose))

        # T close signals, using current locked scanner formulas.
        f_to_t = feat[feat.date <= di]
        r7_state, r7_cands = core.scan_r7(f_to_t, dt_from_int(di), bm[bm.date <= di])
        regime = r7_state["regime"]; regime_changed = last_regime is None or regime != last_regime
        reb_due = last_reb_i is None or regime_changed or (i - last_reb_i) >= 15
        if reb_due: last_reb_i = i
        last_regime = regime
        if cfg["r05"]:
            try: r05_state, r05_cands = core.scan_r05(f_to_t, ins[ins.date <= di], dt_from_int(di))
            except Exception: r05_state, r05_cands = {"risk_on":False}, pd.DataFrame()
        else:
            r05_state, r05_cands = {"risk_on":False}, pd.DataFrame()

        r7_rank = {str(x.code):int(x.r7_rank) for x in r7_cands[["code","r7_rank"]].itertuples(index=False)}
        r7_score = {str(x.code):float(x.r7_score) for x in r7_cands[["code","r7_score"]].itertuples(index=False)}
        r05_score = {} if r05_cands.empty else {str(x.code):float(x.r05_score) for x in r05_cands[["code","r05_score"]].itertuples(index=False)}

        sell_map: Dict[str, SellOrder] = {}
        if di in next_date:
            exdate = next_date[di]
            for k, p in list(positions.items()):
                r = row_lookup(feat_idx, di, p.code); reason = None
                if p.strategy == "R7":
                    p.hold_days += 1
                    if r is not None and np.isfinite(r.aclose) and float(r.aclose) <= p.entry_adj * 0.88: reason = "HARD"
                    elif reb_due:
                        n = int(r7_state["slots"]); rank = r7_rank.get(p.code, 10**9)
                        if r7_state["exposure"] <= 0: reason = "REB_REGIME0"
                        elif n <= 0 or rank > 2*n: reason = "REB_RANK"
                else:
                    reason = r05_exit_reason(p, r)
                if reason: sell_map[k] = SellOrder(di, exdate, p.strategy, p.code, reason)

            # R7 regime exposure exit: weakest ranks first.
            exclude = set(sell_map); r7_mv = value_of(positions, feat_idx, di, strategy="R7", exclude=exclude)
            r7_target = nav * float(r7_state["exposure"])
            if r7_mv > r7_target * 1.03 + 1:
                remain = [(k,p) for k,p in positions.items() if p.strategy=="R7" and k not in exclude]
                remain.sort(key=lambda kp:r7_rank.get(kp[1].code,10**9), reverse=True); projected = r7_mv
                for k,p in remain:
                    if projected <= r7_target: break
                    rr = row_lookup(feat_idx,di,p.code); mv=p.shares*(float(rr.close) if rr is not None else p.entry_price)
                    sell_map[k]=SellOrder(di,exdate,p.strategy,p.code,"EXPO"); projected -= mv

            # Portfolio DD force reduce. Manual locks trigger/target/cooldown but not
            # a unique cross-strategy liquidation order; use deterministic weakest-first.
            if dd <= FORCE_DD and i >= force_cooldown_until:
                force_cooldown_until=i+FORCE_COOLDOWN_DAYS; no_buy_until=max(no_buy_until,i+FORCE_NO_BUY_DAYS); forced_count += 1
                exclude=set(sell_map); projected=value_of(positions,feat_idx,di,exclude=exclude); target=nav*FORCE_TARGET_EXPOSURE
                remain=[(k,p) for k,p in positions.items() if k not in exclude]
                def weakness(item):
                    _,p=item
                    if p.strategy=="R7": return (0,-r7_rank.get(p.code,10**9),r7_score.get(p.code,-1e9))
                    rr=row_lookup(feat_idx,di,p.code); ret=(float(rr.aclose)/p.entry_adj-1) if rr is not None and np.isfinite(rr.aclose) else -9
                    return (1,r05_score.get(p.code,-1e9),ret)
                remain.sort(key=weakness)
                for k,p in remain:
                    if projected <= target: break
                    rr=row_lookup(feat_idx,di,p.code); mv=p.shares*(float(rr.close) if rr is not None else p.entry_price)
                    sell_map[k]=SellOrder(di,exdate,p.strategy,p.code,"FORCE_DD"); projected -= mv
                event_rows.append({"date":di,"event":"FORCE_DD","dd":dd,"target_exposure":FORCE_TARGET_EXPOSURE})
            if sell_map: pending_sells.setdefault(exdate,[]).extend(sell_map.values())

        # T close fixed buy orders. R0.5 is evaluated before R7 when both compete
        # for the common pool, matching the locked event-log ordering.
        created: List[BuyOrder] = []
        if di in next_date and i >= no_buy_until:
            exdate=next_date[di]; sell_keys=set(sell_map)
            codes_after={p.code for k,p in positions.items() if k not in sell_keys}
            base_exposure=value_of(positions,feat_idx,di,exclude=sell_keys)
            base_r7=value_of(positions,feat_idx,di,strategy="R7",exclude=sell_keys)
            reserved_cash=reserved_exposure=reserved_r7=0.0; reserved_code:Dict[str,float]={}

            def try_order(strategy:str,row,rank:int)->None:
                nonlocal reserved_cash,reserved_exposure,reserved_r7,codes_after
                code=str(row.code); name0=str(row.name); k=pos_key(strategy,code)
                if k in positions and k not in sell_keys: return
                if code not in codes_after and len(codes_after)>=MAX_POSITIONS: return
                if strategy=="R05":
                    n=sum(1 for kk,p in positions.items() if p.strategy=="R05" and kk not in sell_keys)+sum(1 for o in created if o.strategy=="R05")
                    if n>=R05_MAX_SLOTS:return
                    base_pct=R05_BASE; limit=float(core.floor_tick(float(row.close)*0.995))
                else:
                    n=sum(1 for kk,p in positions.items() if p.strategy=="R7" and kk not in sell_keys)+sum(1 for o in created if o.strategy=="R7")
                    if n>=int(r7_state["slots"]):return
                    base_pct=R7_BASE; limit=float(core.floor_tick(float(row.close)*0.98))
                current_code=value_of(positions,feat_idx,di,code=code,exclude=sell_keys)+reserved_code.get(code,0.0)
                rem_single=nav*MAX_SINGLE-current_code; rem_global=nav*MAX_TOTAL-base_exposure-reserved_exposure

                # Exact locked allocation: R7=22% NAV, R0.5=20% NAV. This base
                # amount also determines whether the stock qualifies for the
                # high-price odd-lot exception. Current T-day cash is NOT used to
                # shrink the precommitted quantity; T+1 sells execute first and
                # their actual proceeds enter the common pool before buys.
                base_target=nav*base_pct
                target=base_target*dd_multiplier(dd)
                if strategy=="R7": target=min(target,nav*float(r7_state["exposure"])-base_r7-reserved_r7)
                target=min(target,rem_single,rem_global)
                if target<=0:return
                shares,_=size_shares(target,base_target,limit,float(row.avgvol20))
                if shares<=0:return
                reserve=shares*limit*(1+BUY_FEE); notional=shares*limit
                created.append(BuyOrder(di,exdate,strategy,code,name0,limit,shares,target,reserve,rank))
                reserved_cash+=reserve; reserved_exposure+=notional; reserved_code[code]=reserved_code.get(code,0.0)+notional
                if strategy=="R7":reserved_r7+=notional
                codes_after.add(code)

            if cfg["r05"] and bool(r05_state.get("risk_on")) and not r05_cands.empty:
                for _,row in r05_cands.head(20).iterrows(): try_order("R05",row,int(row.r05_rank))
            if reb_due and float(r7_state["exposure"])>0 and not r7_cands.empty:
                for _,row in r7_cands.head(max(0,int(r7_state["slots"]))).iterrows(): try_order("R7",row,int(row.r7_rank))
            if created: pending_buys.setdefault(exdate,[]).extend(created)

        nav_rows.append({"date":di,"nav":nav,"cash":cash,"stock_mv":stock_mv,"exposure":exposure,"drawdown":dd,"positions":len({p.code for p in positions.values()}),"r7_positions":sum(p.strategy=="R7" for p in positions.values()),"r05_positions":sum(p.strategy=="R05" for p in positions.values()),"r7_regime":regime,"r7_regime_exposure":r7_state["exposure"],"r7_rebalance_due":reb_due,"r05_risk_on":bool(r05_state.get("risk_on",False)),"dd_multiplier":dd_multiplier(dd),"no_buy_active":i<no_buy_until})

    nav_df=pd.DataFrame(nav_rows); trades_df=pd.DataFrame(trade_rows); orders_df=pd.DataFrame(order_rows); events_df=pd.DataFrame(event_rows)
    end_nav=float(nav_df.iloc[-1].nav); total_return=end_nav/INITIAL_CAPITAL-1
    start_dt,end_dt=dt_from_int(eval_dates[0]),dt_from_int(eval_dates[-1]); years=max((end_dt-start_dt).days/365.25,1/365.25)
    cagr=(end_nav/INITIAL_CAPITAL)**(1/years)-1; max_dd=float(nav_df.drawdown.min())
    fills=int((orders_df.side.eq("BUY")&orders_df.filled.eq(True)).sum()) if not orders_df.empty else 0
    buy_orders=int(orders_df.side.eq("BUY").sum()) if not orders_df.empty else 0
    ann={}; nav_df["year"]=nav_df.date.astype(str).str[:4].astype(int); prev=INITIAL_CAPITAL
    for y,g in nav_df.groupby("year"):
        e=float(g.iloc[-1].nav); ann[str(y)]=e/prev-1; prev=e

    result={"status":"PASS","engine_version":VERSION,"scenario":name,"label":cfg["label"],"mode":cfg["mode"],"mode_reason":cfg["reason"],"eval_start":str(eval_dates[0]),"eval_end":str(eval_dates[-1]),"initial_nav":INITIAL_CAPITAL,"end_nav":end_nav,"total_return":total_return,"cagr":cagr,"max_dd":max_dd,"completed_trades":int(len(trades_df)),"orders":buy_orders,"fills":fills,"fill_rate":fills/buy_orders if buy_orders else 0.0,"min_cash":float(nav_df.cash.min()),"avg_exposure":float(nav_df.exposure.mean()),"max_exposure":float(nav_df.exposure.max()),"max_positions":int(nav_df.positions.max()),"force_dd_events":int(forced_count),"corporate_action_continuity_events":int(corp_events),"annual_returns":ann,"r05_enabled":bool(cfg["r05"]),"stress_extension_assumption":"If DD <= -14%, full positions are exited weakest-first until exposure <= ~50%; manual locks trigger/target/cooldown but not a unique cross-strategy liquidation order."}
    if name=="validation2021_2025":
        result["locked_benchmark"]=LOCKED_BENCHMARK
        result["validation_delta"]={"end_nav_pct":end_nav/LOCKED_BENCHMARK["end_nav"]-1,"cagr":cagr-LOCKED_BENCHMARK["cagr"],"max_dd":max_dd-LOCKED_BENCHMARK["max_dd"],"trades":int(len(trades_df))-LOCKED_BENCHMARK["completed_trades"]}

    out=OUT_ROOT/"latest"/name; out.mkdir(parents=True,exist_ok=True)
    nav_df.to_csv(out/"daily_nav.csv",index=False,encoding="utf-8-sig")
    trades_df.to_csv(out/"trades.csv",index=False,encoding="utf-8-sig")
    orders_df.to_csv(out/"orders.csv",index=False,encoding="utf-8-sig")
    events_df.to_csv(out/"risk_events.csv",index=False,encoding="utf-8-sig")
    (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return result

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--scenario",default="all",choices=["all"]+list(SCENARIOS)); args=ap.parse_args()
    latest=OUT_ROOT/"latest"; latest.mkdir(parents=True,exist_ok=True)
    names=["gfc2008","covid2020","bear2022"] if args.scenario=="all" else [args.scenario]
    results=[]
    for name in names:
        try: results.append(simulate(name,SCENARIOS[name]))
        except Exception as exc:
            log(f"[ERROR] {name}: {exc}"); results.append({"status":"ERROR","engine_version":VERSION,"scenario":name,"mode":SCENARIOS[name]["mode"],"error":str(exc)})
    payload={"generated_at":datetime.now().astimezone().isoformat(),"engine_version":VERSION,"results":results}
    (latest/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (latest/"SUMMARY.md").write_text(make_summary_md(results),encoding="utf-8")
    print(make_summary_md(results))
    if any(r.get("status")!="PASS" for r in results): raise SystemExit(2)

if __name__=="__main__": main()
