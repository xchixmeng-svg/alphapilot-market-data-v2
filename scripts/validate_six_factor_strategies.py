"""Independent validation of Taiwan-stock factors A/B/C/D/E/A+E.

Causal rules:
- 2020 warm-up, evaluation 2021 through latest 2026 observation.
- Monthly revenue becomes usable on the 10th of the following month.
- Factor statistics aggregate equal-weight signal returns by decision date and
  use a Newey-West/HAC t statistic to avoid treating overlapping stock returns
  as independent observations.
- Trading uses T-close decisions and T+1 execution only.
"""
from __future__ import annotations
import io, json, math, re, time, zipfile
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"six_factor_validation"
OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/"data"/"history"/"2020-2025"
INITIAL=1_000_000.0
FEE=0.001425
TAX=0.003
SELL_ADVERSE=0.02
LOT=100
MAX_W=0.20

def get(url, tries=5):
    err=None
    for i in range(tries):
        try:
            with urlopen(Request(url,headers={"User-Agent":"Mozilla/5.0 AlphaPilot-independent-validation"}),timeout=120) as r:
                return r.read()
        except Exception as e:
            err=e; time.sleep(min(8,2**i))
    raise RuntimeError(f"GET failed {url}: {err}")

def load_ohlcv():
    fs=[pd.read_parquet(DATA/f"ohlcv_{y}.parquet") for y in range(2020,2026)]
    # Same archived weekly source already used by this repository's 2026 builder.
    rel=json.loads(get("https://api.github.com/repos/yukishirotsubasa/tw-stock-data-release/releases/tags/daily-close-csv"))
    assets=sorted((a for a in rel.get("assets",[]) if re.fullmatch(r"weekly_2026_W\\d+\\.zip",a["name"])),key=lambda x:x["name"])
    rows=[]
    for a in assets:
        with zipfile.ZipFile(io.BytesIO(get(a["browser_download_url"]))) as z:
            for n in z.namelist():
                if n.lower().endswith(".csv"):
                    q=pd.read_csv(z.open(n),dtype={"code":str})
                    q.columns=[str(c).lower() for c in q.columns]
                    rows.append(q)
    if rows:
        y=pd.concat(rows,ignore_index=True)
        y["date"]=pd.to_datetime(y["date"],errors="coerce")
        y=y[y.date.dt.year.eq(2026)]
        fs.append(y)
    d=pd.concat(fs,ignore_index=True)
    d.columns=[str(c).lower() for c in d.columns]
    d["code"]=d["code"].astype(str).str.replace(r"\\.0$","",regex=True).str.zfill(4)
    d["date"]=pd.to_datetime(d["date"],errors="coerce")
    for c in ["open","high","low","close","volume"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["date","code","open","high","low","close","volume"])
    d=d[(d.open>0)&(d.high>0)&(d.low>0)&(d.close>0)&(d.volume>=0)]
    d=d.sort_values(["code","date"]).drop_duplicates(["date","code"],keep="last")
    # Detect only unmistakable splits. Prices adjusted only for signal continuity.
    g=d.groupby("code",sort=False)
    ratio=d.open/g.close.shift()
    d["split_mult"]=1
    for m in (2,3,4,5,10):
        d.loc[(ratio-1/m).abs()<=0.03/m,"split_mult"]=m
    future=d.groupby("code",sort=False).split_mult.transform(lambda s:s.iloc[::-1].cumprod().iloc[::-1]/s)
    for c in ["open","high","low","close"]:
        d["adj_"+c]=d[c]/future
    # Corporate-action audit: every modeled split must preserve overnight
    # notional within 3%; any unexplained >35% discontinuity fails closed.
    prev_close=d.groupby("code",sort=False).close.shift()
    audit=d.loc[d.split_mult.gt(1),["date","code","name","open","close","split_mult"]].copy()
    audit["prev_close"]=prev_close.loc[audit.index]
    audit["notional_ratio"]=audit.open*audit.split_mult/audit.prev_close
    audit["invariant_pass"]=audit.notional_ratio.between(.97,1.03)
    audit.to_csv(OUT/"corporate_action_audit.csv",index=False)
    unexplained=(ratio.lt(.65)|ratio.gt(1.55)) & d.split_mult.eq(1) & prev_close.notna()
    suspects=d.loc[unexplained,["date","code","name","open","close"]].copy()
    suspects["prev_close"]=prev_close.loc[suspects.index]
    suspects["overnight_ratio"]=ratio.loc[suspects.index]
    suspects.to_csv(OUT/"unresolved_price_discontinuities.csv",index=False)
    if not audit.invariant_pass.all():
        raise RuntimeError("split notional invariant failed; refusing corrupted backtest")
    # Unknown discontinuities are never guessed into split factors. Quarantine
    # the entire affected security from signals/trading and retain the audit.
    # This is deliberately conservative: no false profit can enter the ledger.
    bad_codes=set(suspects.code.astype(str))
    d["ca_clean"]=~d.code.isin(bad_codes)
    print(f"[CA AUDIT] modeled_splits={len(audit)} quarantined_codes={len(bad_codes)}",flush=True)
    return d.reset_index(drop=True)

def _fetch_revenue_month(y,m,market):
    url=f"https://mops.twse.com.tw/nas/t21/{market}/t21sc03_{y-1911}_{m}_0.html"
    raw=get(url,tries=3).decode("big5","ignore")
    out=[]
    for t in pd.read_html(io.StringIO(raw)):
        if isinstance(t.columns,pd.MultiIndex):
            t.columns=["|".join(str(v) for v in c if str(v)!="nan") for c in t.columns]
        cols=list(map(str,t.columns))
        cc=next((c for c in cols if "公司代號" in c),None)
        nc=next((c for c in cols if "公司名稱" in c),None)
        rc=next((c for c in cols if "當月營收" in c and "去年" not in c),None)
        if not cc or not rc: continue
        for _,r in t.iterrows():
            code=str(r.get(cc,"")).strip().replace("=","").replace('"',"")
            if not re.fullmatch(r"\\d{4}",code): continue
            val=pd.to_numeric(str(r.get(rc,"")).replace(",",""),errors="coerce")
            if pd.notna(val): out.append((y,m,code,str(r.get(nc,"")),float(val)))
    if not out: raise RuntimeError(f"empty revenue {y}-{m:02d} {market}")
    return out

def fetch_revenue():
    cache=OUT/"monthly_revenue.csv"
    if cache.exists(): return pd.read_csv(cache,dtype={"code":str},parse_dates=["available_date"])
    tasks=[(y,m,market) for y in range(2019,2027)
           for m in range(1,(12 if y<2026 else 8)+1) for market in ("sii","otc")]
    out=[]; failures=[]
    # Bounded parallelism prevents the 150+ archive requests from taking hours.
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(_fetch_revenue_month,*t):t for t in tasks}
        for i,f in enumerate(as_completed(fut),1):
            t=fut[f]
            try: out.extend(f.result())
            except Exception as e: failures.append({"year":t[0],"month":t[1],"market":t[2],"error":str(e)})
            if i%20==0 or i==len(tasks):
                print(f"[REVENUE] {i}/{len(tasks)} rows={len(out)} failures={len(failures)}",flush=True)
    pd.DataFrame(failures).to_csv(OUT/"revenue_download_failures.csv",index=False)
    q=pd.DataFrame(out,columns=["year","month","code","name","revenue"]).drop_duplicates(["year","month","code"],keep="last")
    covered=len(q[["year","month"]].drop_duplicates())
    if covered<88 or len(q)<100000:
        raise RuntimeError(f"revenue coverage insufficient months={covered} rows={len(q)} failures={len(failures)}")
    prev={(r.code,r.year,r.month):r.revenue for r in q.itertuples()}
    q["yoy"]=[(r.revenue/prev.get((r.code,r.year-1,r.month))-1) if prev.get((r.code,r.year-1,r.month)) not in (None,0) else np.nan for r in q.itertuples()]
    q["period"]=pd.to_datetime(dict(year=q.year,month=q.month,day=1))
    q["available_date"]=q.period+pd.offsets.MonthBegin(1)+pd.Timedelta(days=9)
    q=q.sort_values(["code","available_date"])
    q["avg_yoy_90d"]=q.groupby("code").yoy.transform(lambda z:z.shift(1).rolling(3,min_periods=2).mean())
    q["accel"]=q.yoy-q.avg_yoy_90d
    q.to_csv(cache,index=False)
    return q

def features(d,rev):
    x=d.copy(); g=x.groupby("code",sort=False)
    x["ret5"]=g.adj_close.pct_change(5)
    x["hi60"]=g.adj_close.transform(lambda s:s.rolling(60,min_periods=60).max())
    x["near_high"]=x.adj_close/x.hi60
    value=x.close*x.volume
    x["adv5"]=value.groupby(x.code).transform(lambda s:s.rolling(5,min_periods=5).mean())
    x["adv20"]=value.groupby(x.code).transform(lambda s:s.rolling(20,min_periods=20).mean())
    x["vol_accel"]=x.adv5/x.adv20-1
    for n in (5,20,60,120,200):
        x[f"ma{n}"]=g.adj_close.transform(lambda s,n=n:s.rolling(n,min_periods=n).mean())
    x["prev_hi10"]=g.adj_close.transform(lambda s:s.rolling(10,min_periods=10).max().shift(1))
    x["breakout"]=x.adj_close/x.prev_hi10-1
    x["ret20"]=g.adj_close.pct_change(20)
    # Point-in-time asof join: latest revenue actually public by each trade date.
    r=rev[["code","available_date","yoy","accel"]].dropna(subset=["available_date"]).sort_values(["available_date","code"])
    parts=[]
    for code,z in x.groupby("code",sort=False):
        rr=r[r.code.eq(code)].sort_values("available_date")
        z=z.sort_values("date")
        if rr.empty: z["yoy"]=np.nan; z["accel"]=np.nan
        else:
            z=pd.merge_asof(z,rr[["available_date","yoy","accel"]],left_on="date",right_on="available_date",direction="backward")
        parts.append(z)
    x=pd.concat(parts,ignore_index=True).sort_values(["date","code"])
    # Base universe, including contemporaneous name when available.
    names=x["name"].fillna("") if "name" in x else pd.Series("",index=x.index)
    x["base"]=x.code.str.fullmatch(r"[1-9]\\d{3}") & ~names.str.contains("KY",case=False,na=False) & (x.adv20>=30e6) & x.ca_clean
    x["A"]=False
    for day,idx in x[x.base].groupby("date").groups.items():
        z=x.loc[idx].nlargest(40,"near_high")
        z=z.nlargest(max(1,math.ceil(len(z)/3)),"ret5")
        x.loc[z.index,"A"]=z.vol_accel.lt(1.0)
    x["B"]=x.base & x.close.between(10,50) & x.breakout.gt(0)
    x["D"]=x.base & (x.ma5>x.ma20)&(x.ma20>x.ma60)&(x.ma60>x.ma120)
    x["E"]=False
    for day,idx in x[x.base & x.yoy.gt(0) & x.accel.notna()].groupby("date").groups.items():
        z=x.loc[idx]; cut=z.accel.quantile(.85); x.loc[z.index,"E"]=z.accel.ge(cut)
    x["AE"]=x.A & x.E
    # C: A confirmation followed by a -5% to -15% pullback within 20 sessions.
    x["C"]=False
    for code,z in x.groupby("code",sort=False):
        confirm=[]; post_hi=np.nan
        for j,(ix,rw) in enumerate(z.iterrows()):
            confirm=[v for v in confirm if j-v[0]<=20]
            if rw.A: confirm.append((j,rw.adj_close)); post_hi=rw.adj_close
            if confirm:
                post_hi=max(post_hi,rw.adj_close)
                pb=rw.adj_close/post_hi-1
                if -.15<=pb<=-.05 and rw.base: x.at[ix,"C"]=True
            else: post_hi=np.nan
    return x

def hac_t(s,lag):
    a=np.asarray(pd.Series(s).dropna(),float); n=len(a)
    if n<20:return np.nan
    u=a-a.mean(); v=np.dot(u,u)/n
    for k in range(1,min(lag,n-1)+1):
        cov=np.dot(u[k:],u[:-k])/n; v+=2*(1-k/(lag+1))*cov
    return a.mean()/math.sqrt(max(v,1e-18)/n)

def signal_stats(x):
    rows=[]
    horizons=(10,20,40)
    for h in horizons:
        x[f"fwd{h}"]=x.groupby("code").adj_close.shift(-h)/x.adj_close-1
    periods={"train":("2021-01-01","2023-12-31"),"test":("2024-01-01","2025-12-31"),"blind2026":("2026-01-01","2026-12-31")}
    for strat in ("A","B","C","D","E","AE"):
        for h in horizons:
            for period,(lo,hi) in periods.items():
                q=x[x[strat]&x.date.between(lo,hi)].dropna(subset=[f"fwd{h}"])
                daily=q.groupby("date")[f"fwd{h}"].mean()
                rows.append({"strategy":strat,"horizon":h,"period":period,"signals":len(q),"signal_days":len(daily),
                             "mean_return":daily.mean(),"win_rate":(daily>0).mean(),"hac_t":hac_t(daily,h-1)})
    return pd.DataFrame(rows)

def score(z,strat):
    if strat=="A": return z.near_high.rank(pct=True)+z.ret5.rank(pct=True)
    if strat=="B": return z.breakout
    if strat=="C": return z.ret20
    if strat=="D": return z.ret20
    if strat=="E": return z.accel
    return z.near_high.rank(pct=True)+z.ret5.rank(pct=True)+z.accel.rank(pct=True)

def simulate(x,strat):
    days=sorted(x.date.unique()); by={d:z.set_index("code") for d,z in x.groupby("date")}
    cash=INITIAL; pos={}; pending_buys=[]; pending_sells=set(); trades=[]; vals=[]; peak=INITIAL; lock=0; cool=0
    idx=x[x.code.eq("0050")].set_index("date")
    for day in days:
        if day<pd.Timestamp("2021-01-01"): continue
        bars=by[day]
        for c in list(pos):
            if c in bars.index and int(bars.at[c,"split_mult"])>1:
                mult=int(bars.at[c,"split_mult"])
                old_sh=pos[c]["shares"]
                pos[c]["shares"]=old_sh*mult
                pos[c]["entry"]=pos[c]["entry"]/mult
                pos[c]["high"]=pos[c]["high"]/mult
                if pos[c]["shares"] != old_sh*mult:
                    raise RuntimeError(f"split share invariant failed {day} {c}")
        for c in list(pending_sells):
            if c in pos and c in bars.index:
                p=float(bars.at[c,"open"])*(1-SELL_ADVERSE); sh=pos[c]["shares"]
                cash+=sh*p-sh*float(bars.at[c,"open"])*(FEE+TAX)
                trades.append({"date":day,"side":"SELL","code":c,"shares":sh,"price":p,"reason":pos[c].get("exit_reason","exit")})
                del pos[c]
        pending_sells=set()
        for c,limit,target_cash in pending_buys:
            if c in pos or c not in bars.index: continue
            op,lo=float(bars.at[c,"open"]),float(bars.at[c,"low"])
            fill=op if op<=limit else (limit if lo<=limit else None)
            if fill is None: continue
            sh=int(min(target_cash,cash/(fill*(1+FEE)))//LOT*LOT)
            if sh>0:
                cash-=sh*fill+sh*op*FEE
                pos[c]={"shares":sh,"entry":fill,"high":float(bars.at[c,"close"]),"days":0}
                trades.append({"date":day,"side":"BUY","code":c,"shares":sh,"price":fill,"reason":"signal"})
        pending_buys=[]
        nav=cash+sum(p["shares"]*float(bars.at[c,"close"]) for c,p in pos.items() if c in bars.index)
        peak=max(peak,nav); dd=nav/peak-1; vals.append((day,nav))
        # Close decisions for T+1.
        for c,p in pos.items():
            if c not in bars.index: continue
            cl=float(bars.at[c,"close"]); p["days"]+=1; p["high"]=max(p["high"],cl)
            reason=None
            if cl/p["entry"]-1<=-.15: reason="stop_-15"
            elif cl/p["high"]-1<=-.15: reason="trailing_-15"
            if reason and p["days"]>=1: p["exit_reason"]=reason; pending_sells.add(c)
        if dd<=-.14 and cool==0:
            lock=10; cool=15
            expo=sum(p["shares"]*float(bars.at[c,"close"]) for c,p in pos.items() if c in bars.index)/nav
            if expo>.50:
                for c in sorted(pos,key=lambda k:pos[k]["days"],reverse=True):
                    pos[c]["exit_reason"]="dd_guard"; pending_sells.add(c)
                    expo-=pos[c]["shares"]*float(bars.at[c,"close"])/nav
                    if expo<=.50: break
        lock=max(0,lock-1); cool=max(0,cool-1)
        if lock: continue
        if day not in idx.index: continue
        mr=idx.loc[day]
        exposure=.85 if mr.adj_close>mr.ma60 and mr.adj_close>mr.ma200 else (.60 if mr.adj_close>mr.ma200 else (.35 if mr.adj_close>mr.ma60 else .15))
        free=5-len(pos)-len(pending_sells)
        if free<=0: continue
        q=bars[bars[strat] & ~bars.index.isin(pos)].copy()
        if q.empty: continue
        q["rankscore"]=score(q,strat)
        each=min(MAX_W,exposure/5)*nav
        pending_buys=[(c,float(q.at[c,"close"])*.98,each) for c in q.nlargest(free,"rankscore").index]
    curve=pd.Series(dict(vals)).sort_index()
    years=(curve.index[-1]-curve.index[0]).days/365.2425
    dd=curve/curve.cummax()-1
    buys=sum(t["side"]=="BUY" for t in trades); sells=sum(t["side"]=="SELL" for t in trades)
    return {"strategy":strat,"final_nav":curve.iloc[-1],"total_return":curve.iloc[-1]/INITIAL-1,
            "cagr":(curve.iloc[-1]/INITIAL)**(1/years)-1,"max_drawdown":dd.min(),
            "buys":buys,"completed_sells":sells},curve,pd.DataFrame(trades)

def benchmark(x):
    m=x[x.code.eq("0050")&x.date.ge("2021-01-01")].set_index("date").sort_index()
    p=float(m.iloc[0].open); sh=int(INITIAL/(p*(1+FEE))//LOT*LOT); cash=INITIAL-sh*p*(1+FEE)
    curve=cash+sh*m.close
    years=(curve.index[-1]-curve.index[0]).days/365.2425; dd=curve/curve.cummax()-1
    return {"strategy":"0050_BH","final_nav":curve.iloc[-1],"total_return":curve.iloc[-1]/INITIAL-1,
            "cagr":(curve.iloc[-1]/INITIAL)**(1/years)-1,"max_drawdown":dd.min(),"buys":1,"completed_sells":0},curve

def main():
    d=load_ohlcv(); rev=fetch_revenue(); x=features(d,rev)
    stats=signal_stats(x); stats.to_csv(OUT/"factor_statistics.csv",index=False)
    results=[]; curves={}
    for s in ("A","B","C","D","E","AE"):
        m,c,t=simulate(x,s); results.append(m); curves[s]=c; t.to_csv(OUT/f"trades_{s}.csv",index=False)
        print(s,m,flush=True)
    bm,bc=benchmark(x); results.append(bm); curves["0050_BH"]=bc
    pd.DataFrame(results).to_csv(OUT/"performance_summary.csv",index=False)
    pd.concat(curves,axis=1).to_csv(OUT/"equity_curves.csv")
    report={"method":"date-clustered forward returns with Newey-West HAC t; causal T+1 portfolio simulation",
            "period":{"warmup":"2020","train":"2021-2023","test":"2024-2025","blind":"2026 YTD"},
            "execution":"T close; T+1 buy limit 0.98*T close; sell T+1 open*0.98; full 0.1425% fees and 0.3% sell tax; 100-share step",
            "performance":results}
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2,default=str))

if __name__=="__main__": main()
