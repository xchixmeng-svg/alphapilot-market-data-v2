"""Research-only R10 institutional accumulation episode study.

The frozen R10 engine is run separately. Features at signal close T use only
history through T. An episode is a consecutive run of strictly positive net
shares, broken by a zero, negative, or missing report. This definition is
fixed before looking at trade outcomes; it is a proxy, never actual cost.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def dates(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y%m%d").astype(int)
    return pd.to_datetime(s.astype(str), errors="coerce").dt.strftime("%Y%m%d").astype("Int64")


def weighted_quantile(v, w, q):
    order = np.argsort(v)
    v, w = np.asarray(v)[order], np.asarray(w)[order]
    return float(v[np.searchsorted(np.cumsum(w), q * np.sum(w), side="left")])


def episode(history, column):
    """Return last contiguous positive-net episode within a 20-session horizon."""
    h = history.tail(20)
    values = h[column].to_numpy(dtype=float)
    positive = np.isfinite(values) & (values > 0)
    indices = np.flatnonzero(positive)
    if len(indices) == 0:
        return None
    end = int(indices[-1])
    start = end
    while start > 0 and positive[start - 1]:
        start -= 1
    active = h.iloc[start:end + 1]
    weights = active[column].to_numpy(dtype=float)
    prices = active["typical_index"].to_numpy(dtype=float)
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        return None
    return {
        "age": int(len(h) - 1 - end),
        "length": int(end - start + 1),
        "start_date": int(active.iloc[0]["date"]),
        "end_date": int(active.iloc[-1]["date"]),
        "center_index": float(np.average(prices, weights=weights)),
        "low_index": weighted_quantile(prices, weights, .2),
        "high_index": weighted_quantile(prices, weights, .8),
        "net_shares": float(weights.sum()),
    }


def pf(pnl):
    p = np.asarray(pnl, dtype=float)
    loss = -p[p < 0].sum()
    return float(p[p > 0].sum() / loss) if loss else None


def metrics(x):
    if x.empty:
        return {"n": 0}
    out = {"n": len(x), "win_rate": float((x["return"] > 0).mean()),
           "avg_return": float(x["return"].mean()), "pf": pf(x["pnl"]),
           "avg_mfe": float(x["mfe"].mean()), "avg_mae": float(x["mae"].mean())}
    for horizon in (5, 10, 20, 30):
        v = x[f"forward_{horizon}d"].dropna()
        out[f"forward_{horizon}d_n"] = len(v)
        out[f"forward_{horizon}d_mean"] = float(v.mean()) if len(v) else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", default="formal_run")
    parser.add_argument("--institutional", default="data/history/2020-2025/institutional_2020_2025.parquet")
    parser.add_argument("--out-dir", default="formal_run/inst_episode_oos")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    formal = Path(args.formal_dir)
    trades = pd.read_csv(formal / "r10max_formal_trades.csv", dtype={"code": str})
    orders = pd.read_csv(formal / "r10max_formal_orders.csv", dtype={"code": str})
    px = pd.read_csv(formal / "ohlcv_causal_2020_2025.csv.gz", dtype={"code": str}, low_memory=False)
    inst = pd.read_parquet(args.institutional)
    for d in (trades, orders, px, inst):
        d["code"] = d["code"].astype(str).str.zfill(4)
    for d, cols in ((trades, ("entry_date", "exit_date")),
                    (orders, ("signal_date", "fill_date")),
                    (px, ("date",)), (inst, ("date",))):
        for col in cols:
            d[col] = pd.to_numeric(d[col], errors="coerce").astype("Int64") if pd.api.types.is_numeric_dtype(d[col]) else dates(d[col])
    assert len(trades) == 150 and (trades["pnl"] > 0).sum() == 71
    buys = orders[(orders.side == "BUY") & (orders.status == "FILLED")]
    trades = trades.merge(buys[["code", "strategy", "fill_date", "signal_date"]].rename(
        columns={"fill_date": "entry_date"}), on=["code", "strategy", "entry_date"],
        how="left", validate="many_to_one")
    if trades.signal_date.isna().any():
        raise RuntimeError("R10 filled buy and completed trade ledger did not match")

    px = px.sort_values(["code", "date"]).copy()
    px["typical_index"] = (px.ahigh + px.alow + px.aclose) / 3
    prev = px.groupby("code").aclose.shift()
    tr = pd.concat([px.ahigh - px.alow, (px.ahigh - prev).abs(),
                    (px.alow - prev).abs()], axis=1).max(axis=1)
    px["atr14_index"] = tr.groupby(px.code).transform(lambda s: s.rolling(14, min_periods=14).mean())
    cols = ["foreign_net", "trust_net", "dealer_net"]
    missing = set(cols) - set(inst.columns)
    if missing:
        raise RuntimeError(f"institutional columns missing: {sorted(missing)}")
    if inst.duplicated(["date", "code"]).any():
        raise RuntimeError("duplicate institutional date/code")
    px = px.merge(inst[["date", "code"] + cols], on=["date", "code"], how="left", validate="one_to_one")
    # Signed combined flow: a sale by one institution offsets a purchase by another.
    px["combined_net"] = px[cols].sum(axis=1, min_count=1)
    entities = {"foreign": "foreign_net", "trust": "trust_net",
                "dealer": "dealer_net", "combined": "combined_net"}
    groups = {code: g.reset_index(drop=True) for code, g in px.groupby("code")}
    rows = []
    for trade_id, trade in trades.reset_index(drop=True).iterrows():
        hist = groups.get(trade.code)
        if hist is None:
            raise RuntimeError(f"no price history for {trade.code}")
        sig, ent, ex = map(int, (trade.signal_date, trade.entry_date, trade.exit_date))
        at = hist.index[hist.date == sig]
        entry = hist.index[hist.date == ent]
        if len(at) != 1 or len(entry) != 1 or int(entry[0]) != int(at[0]) + 1:
            raise RuntimeError(f"signal/entry not consecutive stock sessions {trade.code} {sig} {ent}")
        t = hist.iloc[int(at[0])]
        e = hist.iloc[int(entry[0])]
        scale_entry = float(e.aclose / e.close)
        entry_index = float(trade.entry_raw_price) * scale_entry
        hold = hist[(hist.date >= ent) & (hist.date < ex)]
        outcome = {"trade_id": trade_id, "code": trade.code, "strategy": trade.strategy,
                   "signal_date": sig, "entry_date": ent, "exit_date": ex,
                   "return": float(trade["return"]), "pnl": float(trade.pnl),
                   "mfe": float(hold.ahigh.max() / entry_index - 1),
                   "mae": float(hold.alow.min() / entry_index - 1)}
        for horizon in (5, 10, 20, 30):
            index = int(entry[0]) + horizon - 1
            outcome[f"forward_{horizon}d"] = float(hist.iloc[index].aclose / entry_index - 1) if index < len(hist) else np.nan
        for entity, column in entities.items():
            ep = episode(hist.iloc[:int(at[0]) + 1], column)
            row = dict(outcome, entity=entity, available=ep is not None)
            if ep:
                row.update(ep)
                row["signal_distance_pct"] = float(t.aclose / ep["center_index"] - 1)
                row["above_zone_pct"] = float(t.aclose / ep["high_index"] - 1)
                row["above_zone_atr"] = float((t.aclose - ep["high_index"]) / t.atr14_index) if pd.notna(t.atr14_index) and t.atr14_index > 0 else np.nan
                row["signal_close_raw"] = float(t.close)
                row["episode_center_raw_t"] = float(ep["center_index"] * t.close / t.aclose)
            rows.append(row)
    feature = pd.DataFrame(rows)
    assert len(feature) == 150 * len(entities)
    feature.to_csv(out / "trade_features.csv", index=False)

    # Fixed chronological split. Only training trades determine the cutoffs.
    result = {"frozen_commit": "3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9",
              "sample": "2021-2023 train; 2024-2025 OOS, trades assigned by signal date",
              "episode_rule": "latest contiguous positive-net daily run in last 20 stock sessions, one zero/negative/missing report breaks it",
              "cost_proxy": "net-buy-weighted causal typical price, not actual institutional cost",
              "decision_time": "signal-day close T; actual buy on T+1; no entry fill in feature",
              "completed_trades": 150, "entities": {}}
    for entity, z in feature.groupby("entity"):
        train = z[(z.signal_date < 20240101) & z.available].copy()
        test = z[(z.signal_date >= 20240101) & z.available].copy()
        cutoff = float(train.above_zone_atr.quantile(.75)) if len(train) >= 20 else np.nan
        near = test[test.above_zone_atr <= cutoff] if np.isfinite(cutoff) else test.iloc[0:0]
        far = test[test.above_zone_atr > cutoff] if np.isfinite(cutoff) else test.iloc[0:0]
        result["entities"][entity] = {
            "coverage_all": int(z.available.sum()), "missing_all": int((~z.available).sum()),
            "train_n": len(train), "oos_n": len(test), "train_far_q75_atr": cutoff if np.isfinite(cutoff) else None,
            "oos_near_or_normal": metrics(near), "oos_far": metrics(far),
            "oos_missing": metrics(z[(z.signal_date >= 20240101) & ~z.available]),
            "oos_years": {str(y): {"near": metrics(near[near.signal_date // 10000 == y]),
                                  "far": metrics(far[far.signal_date // 10000 == y])} for y in (2024, 2025)},
            "oos_average_return_near_minus_far": float(near["return"].mean() - far["return"].mean()) if len(near) and len(far) else None,
            "sufficient_for_filter": bool(len(near) >= 15 and len(far) >= 15),
        }
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
