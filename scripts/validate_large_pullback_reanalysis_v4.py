"""Causal A/B test: reanalyse V3 large candidates after a 10%-12% pullback."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import validate_six_factor_strategies as core
import validate_three_layer_strategy_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "large_pullback_10_12_reanalysis"
OUT.mkdir(parents=True, exist_ok=True)


def build_pullback_signals(x: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Track raw signals, then rerun the large-layer thesis at confirmation."""
    signal = pd.Series(False, index=x.index)
    audit_rows = []
    ma60_5d=x.ma60.groupby(x.code,sort=False).shift(5)
    for code, indexes in x.groupby("code", sort=False).groups.items():
        ids = list(indexes)
        positions = {k: j for j, k in enumerate(ids)}
        for start_idx in (k for k in ids if bool(x.at[k, "large"])):
            start = positions[start_idx]
            peak = float(x.at[start_idx, "adj_close"])
            peak_date = x.at[start_idx, "date"]
            finished = False
            for age in range(1, 21):
                j = start + age
                if j >= len(ids): break
                k = ids[j]
                close = float(x.at[k, "adj_close"])
                if close > peak:
                    peak = close; peak_date = x.at[k, "date"]
                drawdown = close / peak - 1
                if drawdown < -.12 or close < float(x.at[k, "ma60"]):
                    audit_rows.append(dict(code=code, raw_signal_date=x.at[start_idx, "date"],
                        outcome="cancel_structure", end_date=x.at[k, "date"], peak=peak,
                        drawdown=drawdown))
                    break
                if not (-.12 <= drawdown <= -.10):
                    continue
                touch_idx = k
                q=j+3
                if q>=len(ids): break
                stable_ids=ids[j+1:q+1]; ck=ids[q]
                touch_close=float(x.at[touch_idx,"adj_close"])
                stable_closes=x.loc[stable_ids,"adj_close"].astype(float)
                stable_volumes=x.loc[stable_ids,"volume"].astype(float)
                cclose=float(x.at[ck,"adj_close"])
                tests={
                    "three_days_within_12pct":bool((stable_closes/peak-1>=-.12).all()),
                    "three_days_above_ma60":bool((stable_closes>=x.loc[stable_ids,"ma60"].astype(float)).all()),
                    "no_new_close_low_beyond_1pct":float(stable_closes.min())>=touch_close*.99,
                    "third_close_recovers_touch_close":cclose>=touch_close,
                    "three_day_volume_contracted":float(stable_volumes.mean())<=float(x.at[ck,"vol20"]),
                    "revenue_yoy_positive":float(x.at[ck,"yoy"])>0,
                    "revenue_acceleration_positive":float(x.at[ck,"accel"])>0,
                    "institutional_3d_nonnegative":float(x.at[ck,"inst3"])>=0,
                    "foreign_or_trust_3_of_5":float(x.at[ck,"foreign_pos5"])>=3 or float(x.at[ck,"trust_pos5"])>=3,
                    "rising_ma60":float(x.at[ck,"ma60"])>float(ma60_5d.at[ck]),
                    "market_breadth_present":float(x.at[ck,"market_money_pulse"])>=.08,
                }
                if all(tests.values()):
                    signal.at[ck] = True
                    audit_rows.append(dict(code=code, raw_signal_date=x.at[start_idx, "date"],
                        outcome="reanalysis_passed", end_date=x.at[ck, "date"], peak_date=peak_date,
                        peak=peak, pullback_date=x.at[touch_idx, "date"],
                        drawdown=float(x.at[touch_idx, "adj_close"]) / peak - 1,
                        rejection_reasons=""))
                    finished = True
                else:
                    audit_rows.append(dict(code=code,raw_signal_date=x.at[start_idx,"date"],
                        outcome="reanalysis_rejected",end_date=x.at[ck,"date"],peak_date=peak_date,
                        peak=peak,pullback_date=x.at[touch_idx,"date"],drawdown=drawdown,
                        rejection_reasons="|".join(name for name,passed in tests.items() if not passed)))
                if finished: break
            else:
                audit_rows.append(dict(code=code, raw_signal_date=x.at[start_idx, "date"],
                    outcome="expired_20d", end_date=x.at[ids[min(start+20, len(ids)-1)], "date"],
                    peak=peak))
    return signal, pd.DataFrame(audit_rows)


def excursions(x: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    paired = v3.completed_trade_rows(ledger)
    if paired.empty: return paired
    rows=[]
    prices=x.set_index(["code", "date"]).adj_close.sort_index()
    for r in paired.itertuples(index=False):
        path=prices.loc[(str(r.code), slice(pd.Timestamp(r.buy_date), pd.Timestamp(r.sell_date)))]
        if isinstance(path.index, pd.MultiIndex): path.index=path.index.get_level_values(-1)
        base=float(path.iloc[0])
        z=r._asdict(); z["mae"]=float((path/base-1).min()); z["mfe"]=float((path/base-1).max())
        rows.append(z)
    return pd.DataFrame(rows)


def yearly_rows(named_curves: dict[str, pd.Series]) -> pd.DataFrame:
    rows=[]
    for name, series in named_curves.items():
        for year,z in series.dropna().groupby(series.dropna().index.year):
            if 2021 <= year <= 2026 and len(z)>1:
                rows.append(dict(strategy=name,year=int(year),start_nav=float(z.iloc[0]),
                    end_nav=float(z.iloc[-1]),year_return=float(z.iloc[-1]/z.iloc[0]-1),
                    year_max_drawdown=float((z/z.cummax()-1).min())))
    return pd.DataFrame(rows)


def main():
    raw=core.load_ohlcv(); revenue=core.fetch_revenue(); x=v3.add_features(raw,revenue)
    original_large=x.large.copy()
    pullback, signal_audit=build_pullback_signals(x)
    signal_audit.to_csv(OUT/"pullback_signal_audit.csv",index=False)

    configs=[]
    for variant, large_signal in (("v3_original", original_large), ("pullback_10_12_reanalysed", pullback)):
        x["large"]=large_signal
        for layers,suffix in ((('large',),"large_layer"),
                              (("short","swing","large"),"combined_three_layer")):
            label=f"{variant}_{suffix}"
            result,curve,ledger,splits,rejected=v3.simulate(x,layers,label)
            paired=excursions(x,ledger)
            result.update(completed_trades=len(paired),wins=int(paired.win.sum()) if not paired.empty else 0,
                win_rate=float(paired.win.mean()) if not paired.empty else 0.0,
                average_mae=float(paired.mae.mean()) if not paired.empty else np.nan,
                average_mfe=float(paired.mfe.mean()) if not paired.empty else np.nan)
            configs.append((label,result,curve,ledger,paired,splits,rejected))
            ledger.to_csv(OUT/f"trades_{label}.csv",index=False)
            paired.to_csv(OUT/f"completed_trades_{label}.csv",index=False)
            rejected.to_csv(OUT/f"rejected_orders_{label}.csv",index=False)

    bm,bc=core.benchmark(x)
    results=[q[1] for q in configs]+[bm]
    pd.DataFrame(results).to_csv(OUT/"performance_summary.csv",index=False)
    curves={q[0]:q[2].nav for q in configs}; curves["0050_BH"]=bc
    pd.concat(curves,axis=1).to_csv(OUT/"equity_curves.csv")
    yearly_rows(curves).to_csv(OUT/"yearly_performance.csv",index=False)

    market_days=sorted(x.loc[x.code.eq("0050"),"date"].drop_duplicates())
    next_day={market_days[i]:market_days[i+1] for i in range(len(market_days)-1)}
    ledgers=[q[3] for q in configs]
    exact_t1=all(t.empty or all(next_day.get(pd.Timestamp(d))==pd.Timestamp(e)
        for d,e in zip(t.decision_date,t.execute_date)) for t in ledgers)
    checks={
        "coverage_2021_2026":set(range(2021,2027)).issubset(set(x.date.dt.year.unique())),
        "pullback_exactly_10_to_12pct":bool(signal_audit.query("outcome == 'reanalysis_passed'").drawdown.between(-.12,-.10).all()),
        "confirmation_is_after_pullback":bool((pd.to_datetime(signal_audit.query("outcome == 'reanalysis_passed'").end_date)>
            pd.to_datetime(signal_audit.query("outcome == 'reanalysis_passed'").pullback_date)).all()),
        "all_execution_exact_t1":exact_t1,
        "all_shares_100_step":all(t.empty or (t.shares.mod(100).eq(0)).all() for t in ledgers),
        "fees_and_taxes_present":all(t.empty or (t.commission.ge(0)&t.tax.ge(0)).all() for t in ledgers),
        "performance_finite":all(np.isfinite(r["final_nav"]) for r in results),
    }
    checks={k:bool(v) for k,v in checks.items()}
    (OUT/"contract_audit.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    if not all(checks.values()): raise RuntimeError(checks)
    report={"test":"large signal, 10%-12% pullback, three-day stabilization, full reanalysis, then T+1",
        "watch_days":20,"stabilization_days":3,"results":results,
        "signal_funnel":signal_audit.outcome.value_counts().to_dict(),"contract_audit":checks}
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
