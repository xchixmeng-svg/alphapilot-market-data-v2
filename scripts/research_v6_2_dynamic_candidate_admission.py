#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
import backtest_ai_market_reasoning_v6_1_corp_safe as safe
import research_v6_5_cross_sectional_pareto as v65
import research_v6_3_first_passage as fp
OUT=ROOT/'v6_2_dynamic_candidate_results'; OUT.mkdir(exist_ok=True)
H=v65.HORIZONS; BARRIERS=(.05,.10,.20); SEED=926622

def fit_and_calibrate(train,feats,year):
    up,sa,n=v65.fit_heads(train,feats,SEED+year)
    # Calibration is based only on pre-evaluation support. It never sees evaluated-day ranks/counts.
    rng=np.random.default_rng(SEED+100+year); ix=np.flatnonzero(train.universe_ok.to_numpy())
    if len(ix)>180000: ix=rng.choice(ix,180000,replace=False)
    p=train.iloc[ix]; scores=[]
    for h in H:
        X=safe.model_frame(p,feats,float(h)); u=np.clip(up.predict(X),0,1); s=np.clip(sa.predict(X),0,1)
        scores.append(np.minimum(u,s))
    z=np.concatenate(scores)
    # Fixed preregistered support quantiles create absolute score gates, not daily Top-K.
    watch=float(np.quantile(z,.75)); cand=float(np.quantile(z,.90)); high=float(np.quantile(z,.97))
    return up,sa,n,{'watch':watch,'candidate':cand,'high_conviction':high,'calibration_rows':int(len(z))}

def targets(df,h):
    t=v65.path_targets(df,h); c=fp.first_passage_targets(df,h)
    # first-passage helper supplies reset-safe +10%; MFE supplies other barriers inside same segment.
    for b in BARRIERS: t[f'hit_{int(b*100)}']=(t.mfe>=b).astype(float)
    t['time_to_10']=c['up_time']; return t

def run(year):
    if year not in (2022,2023,2024): raise RuntimeError('development only 2022-2024; 2025/2026 sealed')
    panel,bf,_=safe.load_frozen_panel(require_lock=False); ds,feats=safe.add_safe_observable_transforms(panel,bf)
    train,test,cut=safe.year_context(ds,year); up,sa,n,gate=fit_and_calibrate(train,feats,year)
    rows=[]
    for h in H:
        t=targets(test,h); valid=test.universe_ok.to_numpy() & np.isfinite(t.mfe.to_numpy(float)) & np.isfinite(t.mae.to_numpy(float))
        e=test.loc[valid,['date','code','price_segment_id','close']].copy(); X=safe.model_frame(test.loc[valid],feats,float(h))
        e['horizon']=h; e['pred_upside']=np.clip(up.predict(X),0,1); e['pred_safety']=np.clip(sa.predict(X),0,1); e['confidence']=np.minimum(e.pred_upside,e.pred_safety)
        e['mfe']=t.loc[valid,'mfe'].to_numpy(float); e['mae']=t.loc[valid,'mae'].to_numpy(float); e['end_return']=t.loc[valid,'end_return'].to_numpy(float); e['time_to_10']=t.loc[valid,'time_to_10'].to_numpy(float)
        for b in BARRIERS:e[f'hit_{int(b*100)}']=t.loc[valid,f'hit_{int(b*100)}'].to_numpy(float)
        rows.append(e)
    long=pd.concat(rows,ignore_index=True)
    # Horizon is diagnostic only: retain strongest predicted path per stock/date; no forced holding rule.
    best=long.sort_values(['date','code','confidence'],ascending=[True,True,False]).drop_duplicates(['date','code'])
    best['admission']='REJECT'; best.loc[best.confidence>=gate['watch'],'admission']='WATCH'; best.loc[best.confidence>=gate['candidate'],'admission']='CANDIDATE'; best.loc[best.confidence>=gate['high_conviction'],'admission']='HIGH_CONVICTION'
    best['is_candidate']=best.admission.isin(['CANDIDATE','HIGH_CONVICTION'])
    # Rank is display-only and computed strictly after admission.
    best['display_rank']=np.nan; m=best.is_candidate; best.loc[m,'display_rank']=best.loc[m].groupby('date').confidence.rank(method='first',ascending=False)
    cand=best[m].copy(); daily=cand.groupby('date').size().rename('candidate_count').reindex(pd.Index(sorted(best.date.unique())),fill_value=0)
    base={}; quality={}
    for b in BARRIERS:
        c=f'hit_{int(b*100)}'; br=float(best[c].mean()); cr=float(cand[c].mean()) if len(cand) else None
        base[c]={'base_rate':br,'candidate_rate':cr,'lift':(cr/br if cr is not None and br>0 else None)}
    bins=pd.qcut(best.confidence,10,duplicates='drop'); cal=best.assign(bin=bins.astype(str)).groupby('bin',observed=True).agg(n=('code','size'),mean_confidence=('confidence','mean'),hit10=('hit_10','mean'),hit20=('hit_20','mean')).reset_index()
    quality={'year':year,'train_cutoff':cut,'fit_rows':n,'gates':gate,'eligible_stock_days':int(len(best)),'candidate_stock_days':int(len(cand)),'coverage':float(len(cand)/len(best)),'days':int(len(daily)),'zero_candidate_days':int((daily==0).sum()),'zero_candidate_day_rate':float((daily==0).mean()),'candidate_count':{'min':int(daily.min()),'median':float(daily.median()),'mean':float(daily.mean()),'max':int(daily.max())},'barriers':base,'candidate_mfe_mean':float(cand.mfe.mean()) if len(cand) else None,'candidate_mae_mean':float(cand.mae.mean()) if len(cand) else None,'median_time_to_10_when_hit':float(cand.loc[cand.hit_10==1,'time_to_10'].median()) if (cand.hit_10==1).any() else None,'corporate_action_boundary_rule':'price_segment_id only; inherited V6.1 safe labels','topk_used_for_admission':False,'2025_opened':False,'2026_opened':False}
    best.to_parquet(OUT/f'V6_2_{year}_ALL_DECISIONS.parquet',index=False); cand.to_csv(OUT/f'V6_2_{year}_CANDIDATES.csv',index=False,encoding='utf-8-sig'); daily.to_csv(OUT/f'V6_2_{year}_DAILY_COUNTS.csv',encoding='utf-8-sig'); cal.to_csv(OUT/f'V6_2_{year}_CALIBRATION.csv',index=False,encoding='utf-8-sig'); (OUT/f'V6_2_{year}_QUALITY.json').write_text(json.dumps(quality,indent=2,ensure_ascii=False)+'\n')
    print('[V6.2 QUALITY]',json.dumps(quality,ensure_ascii=False),flush=True)

def aggregate(p):
    qs=[json.loads((p/f'V6_2_{y}_QUALITY.json').read_text()) for y in (2022,2023,2024)]
    out={'status':'DEVELOPMENT_EVIDENCE','years':qs,'passes_abstention':all(q['zero_candidate_days']>0 for q in qs),'passes_variable_count':all(q['candidate_count']['max']>q['candidate_count']['min'] for q in qs),'passes_positive_lift_all_barriers':all(all(v['lift'] is not None and v['lift']>1 for v in q['barriers'].values()) for q in qs),'2025_opened':False,'2026_opened':False}
    out['qualified']=out['passes_abstention'] and out['passes_variable_count'] and out['passes_positive_lift_all_barriers']
    (OUT/'V6_2_DEVELOPMENT_GATE.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--year',type=int); a.add_argument('--aggregate-dir'); n=a.parse_args()
    if bool(n.year)==bool(n.aggregate_dir): raise SystemExit('specify exactly one')
    run(n.year) if n.year else aggregate(Path(n.aggregate_dir))
