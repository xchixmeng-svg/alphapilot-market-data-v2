#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
D=ROOT/'v17_v16_survivors_portfolio_out'
R10_CAGR=0.1311026701635103
R10_DD=-0.19515438907459803
R10_PNL_PF=1.8627309459518986


def pnl_pf(df):
    if df.empty or 'pnl' not in df: return 0.0
    g=float(df.loc[df.pnl>0,'pnl'].sum()); l=float(-df.loc[df.pnl<0,'pnl'].sum())
    return (99.0 if g>0 else 0.0) if l<=1e-12 else g/l


def main():
    s=pd.read_csv(D/'v17_summary.csv')
    rows=[]
    for r in s.itertuples(index=False):
        cfg=str(r.config)
        full=pd.read_csv(D/f'{cfg}__full_trades.csv')
        blind=pd.read_csv(D/f'{cfg}__blind_2025_trades.csv')
        full_pfp=pnl_pf(full); blind_pfp=pnl_pf(blind)
        row=r._asdict()
        row['full_return_pf']=row.pop('full_pf')
        row['blind_2025_return_pf']=row.pop('blind_2025_pf')
        row['full_pnl_pf']=full_pfp; row['blind_2025_pnl_pf']=blind_pfp
        row['beats_r10_pnl_pf']=bool(full_pfp>R10_PNL_PF)
        row['pareto_dominates_r10_cagr_dd_pnlpf']=bool(row['full_cagr']>=R10_CAGR and row['full_max_dd']>=R10_DD and full_pfp>=R10_PNL_PF)
        row['r10_dominates_candidate_cagr_dd_pnlpf']=bool(R10_CAGR>=row['full_cagr'] and R10_DD>=row['full_max_dd'] and R10_PNL_PF>=full_pfp)
        rows.append(row)
    z=pd.DataFrame(rows)
    z['full_calmar']=z.full_cagr/(-z.full_max_dd)
    z['corrected_rank_score']=pd.concat([
        z.full_cagr.rank(pct=True),z.full_calmar.rank(pct=True),z.full_pnl_pf.rank(pct=True),z.full_max_dd.rank(pct=True),z.blind_2025_return.rank(pct=True)
    ],axis=1).mean(axis=1)
    z=z.sort_values(['corrected_rank_score','full_cagr'],ascending=False)
    z.to_csv(D/'v17_pf_consistent_summary.csv',index=False)
    audit={'version':'v17b-pf-consistent-postaudit','note':'v17 full_pf/blind_2025_pf are return-based PF. Locked R10 PF is PnL-based. This post-audit adds apples-to-apples PnL PF and never compares mixed definitions.',
           'r10':{'cagr':R10_CAGR,'max_dd':R10_DD,'pnl_pf':R10_PNL_PF},
           'candidate_count':int(len(z)),'pareto_dominators':z[z.pareto_dominates_r10_cagr_dd_pnlpf].config.tolist(),
           'r10_dominates':z[z.r10_dominates_candidate_cagr_dd_pnlpf].config.tolist(),
           'best_corrected_rank':None if z.empty else z.iloc[0].config,
           'rows':z[['config','full_cagr','full_max_dd','full_pnl_pf','full_return_pf','blind_2025_return','blind_2025_pnl_pf','corrected_rank_score']].to_dict(orient='records')}
    (D/'v17_pf_consistent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(z[['config','full_cagr','full_max_dd','full_pnl_pf','full_return_pf','blind_2025_return','corrected_rank_score']].to_string(index=False))
    print(json.dumps(audit,ensure_ascii=False,indent=2,default=float))

if __name__=='__main__': main()
