#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "independent_buy_selector_v2_results"
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("v1core", ROOT / "scripts" / "backtest_independent_buy_selector_v1.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fit_models(train: pd.DataFrame):
    med = train[m.FEATURES].replace([np.inf,-np.inf], np.nan).median(numeric_only=True)
    X = train[m.FEATURES].replace([np.inf,-np.inf], np.nan).fillna(med).fillna(0.0)
    y_rel = train["y_quality"].astype(float)
    y_abs = train["persistent_positive"].astype(int)
    reg = HistGradientBoostingRegressor(learning_rate=0.04,max_iter=160,max_leaf_nodes=31,min_samples_leaf=60,l2_regularization=3.0,random_state=915)
    clf = HistGradientBoostingClassifier(learning_rate=0.04,max_iter=160,max_leaf_nodes=31,min_samples_leaf=60,l2_regularization=3.0,random_state=916)
    reg.fit(X,y_rel); clf.fit(X,y_abs)
    return reg, clf, med


def main():
    px, inst = m.load_data(); ds = m.build_dataset(px,inst)
    req = ["fwd20","fwd60","fwd120"]
    valid_abs = ds[req].notna().all(axis=1)
    ds["persistent_positive"] = np.nan
    ds.loc[valid_abs,"persistent_positive"] = ((ds.loc[valid_abs,"fwd20"]>0)&(ds.loc[valid_abs,"fwd60"]>0)&(ds.loc[valid_abs,"fwd120"]>0)).astype(int)
    dates=np.array(sorted(ds.date.unique()),dtype=np.int64); d2i={int(d):i for i,d in enumerate(dates)}
    picks=[]; rows=[]
    for year in range(2021,2026):
        test=ds[(ds.date>=year*10000+101)&(ds.date<=year*10000+1231)&ds.universe_ok].copy()
        first=int(test.date.min()); cutoff=int(dates[d2i[first]-m.PURGE])
        train=ds[(ds.date<cutoff)&ds.universe_ok&ds.y_quality.notna()&ds.persistent_positive.notna()].copy()
        if len(train)<500: raise RuntimeError(f"insufficient train {year}: {len(train)}")
        reg,clf,med=fit_models(train)
        X=test[m.FEATURES].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0.0)
        test["p_persistent"]=clf.predict_proba(X)[:,1]
        test["pred_relative_quality"]=reg.predict(X)
        test["ai_score"]=test["p_persistent"] + 1e-3*test["pred_relative_quality"]
        test["ai_rank"]=test.groupby("date")["ai_score"].rank(method="first",ascending=False)
        top=test[test.ai_rank<=10].copy(); top["test_year"]=year; top["train_cutoff"]=cutoff; picks.append(top)
        labeled=top[top[req].notna().all(axis=1)].copy()
        row={"test_year":year,"train_cutoff":cutoff,"train_rows":len(train),"selected_rows_labeled":len(labeled),"persistent_positive_rate":float(labeled.persistent_positive.mean())}
        universe=test[test[req].notna().all(axis=1)].copy(); row["universe_persistent_positive_rate"]=float(universe.persistent_positive.mean())
        for h in m.HORIZONS:
            medret=universe.groupby("date")[f"fwd{h}"].median()
            z=labeled[labeled[f"fwd{h}"].notna()].copy(); z[f"alpha{h}"]=z[f"fwd{h}"]-z.date.map(medret)
            row[f"mean_fwd{h}"]=float(z[f"fwd{h}"].mean()); row[f"mean_alpha{h}"]=float(z[f"alpha{h}"].mean())
        rows.append(row)
        print(f"V2 fold={year} train={len(train)} top={len(top)} persistent={row['persistent_positive_rate']:.4f}",flush=True)
    sel=pd.concat(picks,ignore_index=True); diag=pd.DataFrame(rows)
    keep=["date","code","name","ai_score","ai_rank","p_persistent","pred_relative_quality","test_year","train_cutoff"]+[f"fwd{h}" for h in m.HORIZONS]+["persistent_positive"]
    sel[keep].to_csv(OUT/"INDEPENDENT_BUY_SELECTOR_V2_TOP10_OOS.csv",index=False); diag.to_csv(OUT/"INDEPENDENT_BUY_SELECTOR_V2_YEARLY.csv",index=False)
    overall={"selected_rows":int(len(sel)),"unique_selected_codes":int(sel.code.nunique()),"persistent_positive_rate":float(sel.persistent_positive.dropna().mean())}
    oos=ds[(ds.date>=20210101)&(ds.date<=20251231)&ds.universe_ok&ds.persistent_positive.notna()].copy(); overall["universe_persistent_positive_rate"]=float(oos.persistent_positive.mean())
    positive_years={}
    for h in m.HORIZONS:
        medret=oos.groupby("date")[f"fwd{h}"].median(); z=sel[sel[f"fwd{h}"].notna()].copy(); z[f"alpha{h}"]=z[f"fwd{h}"]-z.date.map(medret)
        overall[f"mean_fwd{h}"]=float(z[f"fwd{h}"].mean()); overall[f"mean_alpha{h}"]=float(z[f"alpha{h}"].mean()); positive_years[str(h)]=int((diag[f"mean_fwd{h}"]>0).sum())
    overall["positive_absolute_years"]=positive_years
    gate={
      "absolute_positive_20_60_120":bool(all(overall[f"mean_fwd{h}"]>0 for h in (20,60,120))),
      "four_of_five_positive_years_20_60_120":bool(all(positive_years[str(h)]>=4 for h in (20,60,120))),
      "positive_alpha_all_horizons":bool(all(overall[f"mean_alpha{h}"]>0 for h in m.HORIZONS)),
      "persistent_positive_uplift":bool(overall["persistent_positive_rate"]>overall["universe_persistent_positive_rate"])
    }; gate["promising_selector_v2"]=all(gate.values()); gate["note"]="Research gate only; no exit or holding rule."
    (OUT/"INDEPENDENT_BUY_SELECTOR_V2_OVERALL.json").write_text(json.dumps(overall,indent=2),encoding="utf-8"); (OUT/"INDEPENDENT_BUY_SELECTOR_V2_GATE.json").write_text(json.dumps(gate,indent=2),encoding="utf-8")
    print(diag.to_string(index=False)); print(json.dumps(overall,indent=2)); print(json.dumps(gate,indent=2))

if __name__=="__main__": main()
