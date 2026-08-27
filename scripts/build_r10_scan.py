#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaPilot R10-MAX rolling market scanner. Public market data only."""
from __future__ import annotations
import io,json,math,re,time,zipfile
from datetime import date,datetime,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

TZ=ZoneInfo("Asia/Taipei")
VERSION="AlphaPilot-R10-Scanner-V1.0"
ROOT=Path("."); DATA=ROOT/"data"; STATE=DATA/"r10_state.json"; TFILE=ROOT/".alphapilot_trade_date"
S=requests.Session(); S.headers.update({"User-Agent":"Mozilla/5.0 AlphaPilot-R10-Scanner/1.0","Accept":"application/json,text/plain,*/*","Accept-Language":"zh-TW,zh;q=0.9,en;q=0.5"})
WEEKLY="https://github.com/yukishirotsubasa/tw-stock-data-release/releases/download/daily-close-csv/weekly_{year}_W{week:02d}.zip"
TWSE_MI="https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TWSE_T86="https://www.twse.com.tw/rwd/zh/fund/T86"
TWSE_TAI50="https://www.twse.com.tw/indicesReport/TAI50I"
TPEX_OHLCV="https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php"
TPEX_INST_OLD="https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
TPEX_INST_NEW="https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"
COLS=["date","code","name","volume","open","high","low","close"]

def log(x): print(x,flush=True)
def nk(x): return re.sub(r"[\s_\-()/（）％%]+","",str(x or "")).lower()
def num(x):
    if x is None:return np.nan
    s=str(x).strip().replace(",","").replace("+","").replace("−","-")
    if s in {"","--","---","----","null","None"}:return np.nan
    try:return float(s)
    except:return np.nan

def nint(x):
    z=num(x); return np.nan if not np.isfinite(z) else int(round(z))
def anydate(x):
    d=re.sub(r"\D","",str(x or ""))
    if len(d)==8:
        try:return datetime.strptime(d,"%Y%m%d").date()
        except:pass
    if len(d)==7:
        try:return datetime.strptime(f"{int(d[:3])+1911}{d[3:]}","%Y%m%d").date()
        except:pass
    return None
def roc(d): return f"{d.year-1911}/{d:%m/%d}"

def getj(url,params=None,tries=5):
    last=None
    for k in range(tries):
        try:
            r=S.get(url,params=params or {},timeout=(15,60)); r.raise_for_status()
            if not r.text.strip():raise RuntimeError("empty response")
            return r.json()
        except Exception as e:
            last=e
            if k<tries-1:time.sleep(min(12,2**k))
    raise RuntimeError(f"GET failed {url}: {last}")

def tables(payload):
    out=[]
    def walk(x):
        if isinstance(x,dict):
            f=x.get("fields"); d=x.get("data")
            if isinstance(f,list) and isinstance(d,list) and d:out.append((f,d,str(x.get("title",""))))
            aa=x.get("aaData")
            if isinstance(aa,list) and aa:out.append((f or [],aa,str(x.get("title",""))))
            for k,v in x.items():
                if str(k).startswith("fields") and isinstance(v,list):
                    dd=x.get("data"+str(k)[6:])
                    if isinstance(dd,list) and dd:out.append((v,dd,str(x.get("title",""))))
                if isinstance(v,(dict,list)):walk(v)
        elif isinstance(x,list):
            if x and isinstance(x[0],dict):out.append((list(x[0].keys()),x,"list"))
            for v in x[:10]:
                if isinstance(v,(dict,list)):walk(v)
    walk(payload); return out

def dicts(fields,rows):
    out=[]
    for r in rows:
        if isinstance(r,dict):out.append(r)
        elif isinstance(r,list) and fields:out.append({fields[i]:r[i] if i<len(r) else None for i in range(len(fields))})
    return out

def select(payload,tokens):
    best=None
    for f,r,t in tables(payload):
        j="|".join(map(str,f)); m=sum(x in j for x in tokens); score=m*100000+len(r)+(10000 if "代號" in j else 0)
        if best is None or score>best[0]:best=(score,m,f,r)
    if best is None or best[1]<max(1,len(tokens)-1):return None,None
    return best[2],best[3]

def pick(row,*names):
    m={nk(k):v for k,v in row.items()}
    for x in names:
        if nk(x) in m:return m[nk(x)]
    return None

def code(row):
    c=str(pick(row,"Code","SecuritiesCompanyCode","證券代號","代號","股票代號") or "").strip().replace('"',"").replace("=","")
    return c if re.fullmatch(r"\d{4}",c) else None

def norm_ohlcv(rows,d):
    out=[]
    for r in rows:
        c=code(r)
        if not c:continue
        op=num(pick(r,"OpeningPrice","Open","開盤價","開盤")); hi=num(pick(r,"HighestPrice","High","最高價","最高")); lo=num(pick(r,"LowestPrice","Low","最低價","最低")); cl=num(pick(r,"ClosingPrice","Close","收盤價","收盤")); vol=num(pick(r,"TradeVolume","TradingShares","成交股數","成交量"))
        if not all(np.isfinite(x) for x in (op,hi,lo,cl)):continue
        out.append({"date":int(d.strftime("%Y%m%d")),"code":c,"name":str(pick(r,"Name","CompanyName","證券名稱","名稱") or "").strip(),"volume":int(vol) if np.isfinite(vol) else 0,"open":op,"high":hi,"low":lo,"close":cl})
    return out

def twse_day(d):
    p=getj(TWSE_MI,{"date":d.strftime("%Y%m%d"),"type":"ALLBUT0999","response":"json"}); f,r=select(p,["證券代號","開盤價","最高價","最低價","收盤價"])
    return [] if not f else norm_ohlcv(dicts(f,r),d)
def tpex_day(d):
    p=getj(TPEX_OHLCV,{"l":"zh-tw","d":roc(d),"se":"EW","o":"json"}); f,r=select(p,["代號","開盤","最高","最低","收盤"])
    return [] if not f else norm_ohlcv(dicts(f,r),d)
def official_day(d):
    a=b=[]
    try:a=twse_day(d)
    except Exception as e:log(f"[warn] TWSE OHLCV {d}: {e}")
    time.sleep(.12)
    try:b=tpex_day(d)
    except Exception as e:log(f"[warn] TPEx OHLCV {d}: {e}")
    return pd.DataFrame(a+b,columns=COLS) if (a or b) else pd.DataFrame(columns=COLS)

def seed(target):
    parts=[]; y,w,_=target.isocalendar()
    if y!=target.year:raise RuntimeError("ISO year boundary unsupported")
    for wk in range(1,w):
        try:
            r=S.get(WEEKLY.format(year=target.year,week=wk),timeout=(15,60))
            if r.status_code==404:continue
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                names=[n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names:continue
                q=pd.read_csv(z.open(names[0]),dtype={"code":str})
            if any(c not in q.columns for c in COLS):continue
            parts.append(q[COLS].copy()); log(f"[seed] W{wk:02d}: {len(q)}")
        except Exception as e:log(f"[warn] weekly W{wk:02d}: {e}")
    if not parts:return pd.DataFrame(columns=COLS)
    q=pd.concat(parts,ignore_index=True); q["code"]=q.code.astype(str).str.strip().str.zfill(4); q["date"]=pd.to_numeric(q.date,errors="coerce"); q=q.dropna(subset=["date"]); q["date"]=q.date.astype(int); return q

def snapshot(target):
    parts=[]; base=DATA/target.isoformat()/"normalized"
    for fn in ("twse_ohlcv.csv","tpex_ohlcv.csv"):
        p=base/fn
        if not p.exists():continue
        q=pd.read_csv(p,dtype={"stock_id":str})
        if q.empty:continue
        parts.append(pd.DataFrame({"date":int(target.strftime("%Y%m%d")),"code":q.stock_id.astype(str).str.zfill(4),"name":q.name.astype(str),"volume":q.volume,"open":q.open,"high":q.high,"low":q.low,"close":q.close}))
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=COLS)

def history(target):
    h=seed(target)
    if h.empty or h.date.nunique()<120:raise RuntimeError(f"weekly seed only {h.date.nunique() if not h.empty else 0} days")
    last=datetime.strptime(str(int(h.date.max())),"%Y%m%d").date(); d=last+timedelta(days=1); parts=[h]
    while d<=target:
        if d.weekday()<5:
            q=official_day(d)
            if not q.empty:parts.append(q); log(f"[delta] {d}: {len(q)}")
            time.sleep(.12)
        d+=timedelta(days=1)
    s=snapshot(target)
    if not s.empty:parts.append(s)
    q=pd.concat(parts,ignore_index=True); q["code"]=q.code.astype(str).str.strip().str.zfill(4); q["name"]=q.name.astype(str).str.strip()
    for c in ("date","volume","open","high","low","close"):q[c]=pd.to_numeric(q[c],errors="coerce")
    q=q.dropna(subset=["date","code","open","high","low","close"]); q["date"]=q.date.astype(int); q=q.drop_duplicates(["date","code"],keep="last"); q=q[q.date<=int(target.strftime("%Y%m%d"))]
    return q.sort_values(["code","date"]).reset_index(drop=True)

def adjust(df):
    md=np.array(sorted(df.date.unique()),int); pm={md[i]:md[i-1] for i in range(1,len(md))}; groups=[]; events=0
    for c,q in df.groupby("code",sort=False):
        q=q.sort_values("date").copy(); factor=1.; prev=None; pd0=None; ao=[];ah=[];al=[];ac=[]
        for r in q.itertuples(index=False):
            op=float(r.open)*factor; hi=float(r.high)*factor; lo=float(r.low)*factor; cl=float(r.close)*factor
            if prev is not None and pd0 is not None and pm.get(int(r.date))==int(pd0) and op>0:
                ratio=op/prev
                if ratio<.885 or ratio>1.115:
                    step=prev/op; factor*=step; op*=step;hi*=step;lo*=step;cl*=step;events+=1
            ao.append(op);ah.append(hi);al.append(lo);ac.append(cl);prev=cl;pd0=int(r.date)
        q["aopen"]=ao;q["ahigh"]=ah;q["alow"]=al;q["aclose"]=ac;groups.append(q)
    return pd.concat(groups,ignore_index=True).sort_values(["code","date"]).reset_index(drop=True),events

def features(q):
    q=q.copy();q["amount"]=q.close.astype(float)*q.volume.astype(float);g=q.groupby("code",sort=False);q["ret1"]=g.aclose.pct_change(1,fill_method=None)
    for w in (10,20,60):q[f"r{w}"]=g.aclose.pct_change(w,fill_method=None)
    for w in (20,60,120):q[f"ma{w}"]=g.aclose.rolling(w,min_periods=w).mean().reset_index(level=0,drop=True)
    q["amt5"]=g.amount.rolling(5,min_periods=5).mean().reset_index(level=0,drop=True);q["amt20"]=g.amount.rolling(20,min_periods=20).mean().reset_index(level=0,drop=True);q["high60"]=g.aclose.rolling(60,min_periods=60).max().reset_index(level=0,drop=True)
    q["prior_high60"]=g.aclose.transform(lambda s:s.shift(1).rolling(60,min_periods=60).max());q["prior_high10"]=g.aclose.transform(lambda s:s.shift(1).rolling(10,min_periods=10).max())
    rng=(q.high-q.low).replace(0,np.nan);q["clv"]=((2*q.close-q.high-q.low)/rng).fillna(0).clip(-1,1);q["signed_amt"]=np.sign(q.ret1.fillna(0))*q.amount;q["clv_amt"]=q.clv*q.amount
    q["sumamt20"]=g.amount.rolling(20,min_periods=20).sum().reset_index(level=0,drop=True);q["flow20"]=g.signed_amt.rolling(20,min_periods=20).sum().reset_index(level=0,drop=True)/q.sumamt20;q["clvflow20"]=g.clv_amt.rolling(20,min_periods=20).sum().reset_index(level=0,drop=True)/q.sumamt20
    q["clvflow10"]=g.clv_amt.rolling(10,min_periods=10).sum().reset_index(level=0,drop=True)/g.amount.rolling(10,min_periods=10).sum().reset_index(level=0,drop=True);q["clvflow5"]=g.clv_amt.rolling(5,min_periods=5).sum().reset_index(level=0,drop=True)/g.amount.rolling(5,min_periods=5).sum().reset_index(level=0,drop=True)
    q["amtacc"]=q.amt5/q.amt20-1;q["amount_ratio"]=q.amt5/q.amt20;q["nearhigh"]=q.aclose/q.high60;q["ma20gap"]=q.aclose/q.ma20-1
    return q

def tai50(target):
    rows=[]
    for m in range(1,target.month+1):
        try:p=getj(TWSE_TAI50,{"date":date(target.year,m,1).strftime("%Y%m%d"),"response":"json"})
        except Exception as e:log(f"[warn] TAI50 {m}: {e}");continue
        f,d=select(p,["日期","臺灣50","報酬"])
        if not f:continue
        idt=next((i for i,x in enumerate(f) if nk(x)==nk("日期")),-1); ir=next((i for i,x in enumerate(f) if "50" in nk(x) and "報酬" in nk(x)),-1)
        if idt<0 or ir<0:continue
        for r in d:
            if isinstance(r,dict):
                dd=anydate(pick(r,"日期","Date")); val=next((num(v) for k,v in r.items() if "50" in nk(k) and "報酬" in nk(k)),np.nan)
            else:dd=anydate(r[idt] if idt<len(r) else None);val=num(r[ir] if ir<len(r) else None)
            if dd and dd<=target and np.isfinite(val):rows.append((int(dd.strftime("%Y%m%d")),float(val)))
        time.sleep(.08)
    if not rows:raise RuntimeError("Taiwan50 total-return unavailable")
    return pd.DataFrame(rows,columns=["date","mkt"]).drop_duplicates("date",keep="last").sort_values("date")

def pr(s):return s.rank(method="average",pct=True)
def common(x):return x.code.str.fullmatch(r"[1-9]\d{3}",na=False)&~x.name.str.contains("KY",case=False,na=False)

def scan_r7(f,target,bm):
    td=int(target.strftime("%Y%m%d")); br=[]
    for d in sorted(f.date.unique())[-160:]:
        x=f[f.date==d];e=x[(x.amt20>=30_000_000)&x.aclose.notna()]
        if not e.empty:br.append({"date":int(d),"breadth":float((e.aclose>e.ma60).mean()),"advance10":float((e.r10>0).mean())})
    b=pd.DataFrame(br);b["breadth_mean20"]=b.breadth.rolling(20,min_periods=10).mean()
    z=bm[bm.date<=td].copy();z["ma60"]=z.mkt.rolling(60,min_periods=60).mean();z["ma120"]=z.mkt.rolling(120,min_periods=120).mean();z["mr20"]=z.mkt.pct_change(20,fill_method=None);z["mr60"]=z.mkt.pct_change(60,fill_method=None)
    rr=z[z.date==td];bb=b[b.date==td]
    if rr.empty or bb.empty:raise RuntimeError("R7 target benchmark/breadth missing")
    r=rr.iloc[-1];v=bb.iloc[-1];m,ma60,ma120,mr20,mr60,breadth,adv,bmean=[float(x) for x in (r.mkt,r.ma60,r.ma120,r.mr20,r.mr60,v.breadth,v.advance10,v.breadth_mean20)]
    if mr20<=-.08 or (m<ma120 and mr60<0 and breadth<.40):reg,expo,slots="Bear",0.,0
    elif m<ma120*1.02 and mr20>0 and breadth>.42 and breadth>bmean:reg,expo,slots="Repair",.60,2
    elif m>ma60 and m>ma120 and mr20>0 and mr60>0 and breadth>=.60 and adv>=.52:reg,expo,slots="Strong Bull",1.,4
    elif m>ma120 and mr60>0 and breadth>=.45:reg,expo,slots="Normal Bull",.80,3
    elif m>ma120*.98 and breadth>=.38:reg,expo,slots="Weak",.20,2
    else:reg,expo,slots="Fallback/Bear",0.,0
    x=f[(f.date==td)&common(f)].copy();x["rel20"]=x.r20-mr20;x["rel60"]=x.r60-mr60
    for s,n in [("r10","p10"),("rel20","p20"),("rel60","p60"),("flow20","pf"),("amtacc","pa"),("clvflow20","pc"),("nearhigh","pn")]:x[n]=pr(x[s])
    x["r7_score"]=.26*x.p10+.22*x.p20+.10*x.p60+.14*x.pf+.12*x.pa+.08*x.pc+.08*x.pn
    c=x[(x.amt20>=30_000_000)&(x.aclose>x.ma120)&(x.nearhigh>=.78)&x.r7_score.notna()].sort_values(["r7_score","code"],ascending=[False,True]).copy();c["r7_rank"]=np.arange(1,len(c)+1)
    return {"regime":reg,"exposure":expo,"slots":slots,"mkt":m,"mkt_ma60":ma60,"mkt_ma120":ma120,"mr20":mr20,"mr60":mr60,"breadth60":breadth,"breadth_mean20":bmean,"advance10":adv},c

def match(row,inst,actions,reject=()):
    c=[]
    for k,v in row.items():
        z=nk(k)
        if any(z.startswith(nk(x)) for x in reject):continue
        if any(nk(x) in z for x in inst) and any(nk(x) in z for x in actions):c.append((-len(z),v))
    return sorted(c,reverse=True)[0][1] if c else None
def fval(r,a):
    v=match(r,["外陸資","外資及陸資"],a)
    return v if v is not None else match(r,["Foreign","外資"],a,["外資自營商"])
def tval(r,a):return match(r,["InvestmentTrust","Trust","投信"],a)
def dval(r,a):
    c=[]
    for k,v in r.items():
        z=nk(k)
        if z.startswith(nk("外資自營商")):continue
        if not(z.startswith(nk("自營商")) or "dealer" in z) or not any(nk(x) in z for x in a):continue
        score=-len(z)+(10000 if "自行買賣" not in z and "避險" not in z else 0);c.append((score,v))
    return sorted(c,reverse=True)[0][1] if c else None

def norm_inst(rows,d,market):
    out=[]
    for r in rows:
        c=code(r)
        if not c:continue
        fb,fs,fn=nint(fval(r,["Buy","買進"])),nint(fval(r,["Sell","賣出"])),nint(fval(r,["Net","Difference","買賣超","差額"]));tb,ts,tn=nint(tval(r,["Buy","買進"])),nint(tval(r,["Sell","賣出"])),nint(tval(r,["Net","Difference","買賣超","差額"]));db,ds,dn=nint(dval(r,["Buy","買進"])),nint(dval(r,["Sell","賣出"])),nint(dval(r,["Net","Difference","買賣超","差額"]))
        if not np.isfinite(fn) and np.isfinite(fb) and np.isfinite(fs):fn=int(fb-fs)
        if not np.isfinite(tn) and np.isfinite(tb) and np.isfinite(ts):tn=int(tb-ts)
        if not np.isfinite(dn) and np.isfinite(db) and np.isfinite(ds):dn=int(db-ds)
        out.append({"date":int(d.strftime("%Y%m%d")),"market":market,"code":c,"foreign_net":fn,"trust_net":tn,"dealer_net":dn})
    return out

def twse_inst(d):
    p=getj(TWSE_T86,{"date":d.strftime("%Y%m%d"),"selectType":"ALLBUT0999","response":"json"});f,r=select(p,["證券代號","外資","投信","自營商"]);return [] if not f else norm_inst(dicts(f,r),d,"TWSE")
def parse_tpex_inst(p,d):
    best=[]
    for f,r,_ in tables(p):
        z=norm_inst(dicts(f,r),d,"TPEX")
        if len(z)>len(best):best=z
    return best
def tpex_inst(d):
    opts=[(TPEX_INST_OLD,{"l":"zh-tw","o":"json","se":"EW","t":"D","d":roc(d)}),(TPEX_INST_NEW,{"date":d.strftime("%Y/%m/%d"),"response":"json","type":"Daily"}),(TPEX_INST_NEW,{"date":d.strftime("%Y/%m/%d"),"response":"json"})];err=[]
    for u,p in opts:
        try:
            z=parse_tpex_inst(getj(u,p,3),d)
            if len(z)>=50:return z
            err.append(f"{u} rows={len(z)}")
        except Exception as e:err.append(str(e))
    raise RuntimeError("; ".join(err))
def inst_history(target):
    rows=[];d=target-timedelta(days=35)
    while d<=target:
        if d.weekday()<5:
            a=b=[]
            try:a=twse_inst(d)
            except Exception as e:log(f"[warn] TWSE inst {d}: {e}")
            time.sleep(.1)
            try:b=tpex_inst(d)
            except Exception as e:log(f"[warn] TPEx inst {d}: {e}")
            if a or b:rows.extend(a+b);log(f"[inst] {d}: {len(a)}/{len(b)}")
            time.sleep(.1)
        d+=timedelta(days=1)
    if not rows:raise RuntimeError("institutional history empty")
    q=pd.DataFrame(rows).drop_duplicates(["date","market","code"],keep="last").sort_values(["market","code","date"])
    for c in ("foreign_net","trust_net","dealer_net"):q[c]=pd.to_numeric(q[c],errors="coerce")
    g=q.groupby(["market","code"],sort=False);q["Foreign3D"]=g.foreign_net.transform(lambda s:s.rolling(3,min_periods=3).sum());q["Foreign10D"]=g.foreign_net.transform(lambda s:s.rolling(10,min_periods=10).sum());q["Trust5D"]=g.trust_net.transform(lambda s:s.rolling(5,min_periods=5).sum());return q

def scan_r05(f,ins,target):
    td=int(target.strftime("%Y%m%d"));x=f[(f.date==td)&common(f)].copy();et=f[(f.code=="0050")&(f.date<=td)].sort_values("date").copy();et["m60"]=et.close.rolling(60,min_periods=60).mean();et["r20x"]=et.close.pct_change(20,fill_method=None);et["r60x"]=et.close.pct_change(60,fill_method=None)
    if len(et)<61:raise RuntimeError("0050 history insufficient")
    e=et.iloc[-1];risk=bool(e.close>e.m60 and e.r20x>0 and e.r60x>0);it=ins[ins.date==td].sort_values("market").drop_duplicates("code",keep="last")
    if it.empty:raise RuntimeError("target institutional missing")
    x=x.merge(it[["code","Foreign3D","Foreign10D","Trust5D"]],on="code",how="left")
    for s,n in [("clvflow10","pclv10"),("amount_ratio","pamt"),("clvflow5","pclv5"),("Foreign3D","pf3"),("Foreign10D","pf10"),("Trust5D","pt5"),("ma20gap","pgap")]:x[n]=pr(x[s])
    x["r05_score"]=.5251*x.pclv10+.2465*x.pamt+.0683*x.pclv5+.0628*x.pf3-.0778*x.pf10+.0195*x.pt5-.2*x.pgap;x["prior60_position"]=x.aclose/x.prior_high60-1
    h=(x.close.between(10,40))&(x.amt20>=50_000_000)&(x.amount_ratio>=1)&(x.r20.between(0,.20))&(x.ma20gap<=.18)&(x.prior60_position>=-.15)&(x.aclose>x.prior_high10)
    c=x[h&x.r05_score.notna()].sort_values(["r05_score","code"],ascending=[False,True]).copy();c["r05_rank"]=np.arange(1,len(c)+1)
    if not risk:c=c.iloc[0:0]
    return {"risk_on":risk,"0050_close":float(e.close),"0050_ma60":float(e.m60),"0050_ret20":float(e.r20x),"0050_ret60":float(e.r60x)},c

def tick(p):return .01 if p<10 else .05 if p<50 else .1 if p<100 else .5 if p<500 else 1. if p<1000 else 5.
def floor_tick(p):
    t=tick(float(p));return round(math.floor((float(p)+1e-10)/t)*t,4)
def limits(q,kind):
    q=q.copy()
    if not q.empty:q["t1_limit"]=(q.aclose*.98 if kind=="R7" else q.close*.995).map(floor_tick)
    return q

def state():
    try:return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    except:return {}
def days_since(f,s,target):
    try:a=int(date.fromisoformat(s).strftime("%Y%m%d"))
    except:return 999
    b=int(target.strftime("%Y%m%d"));return sum(1 for d in sorted(f.date.unique()) if a<d<=b)
def rebalance(st,f,target,reg):
    if not st or not st.get("last_r7_rebalance_date"):return True,"INITIAL_FORWARD_SCAN"
    if st.get("last_scan_date")==target.isoformat():return bool(st.get("last_r7_rebalance_date")==target.isoformat()),"IDEMPOTENT_RERUN"
    if st.get("last_regime")!=reg:return True,"REGIME_CHANGE"
    n=days_since(f,st.get("last_r7_rebalance_date"),target);return (True,f"15_TRADING_DAYS ({n})") if n>=15 else (False,f"NOT_DUE ({n}/15)")
def records(q,cols,n=10):
    if q.empty:return []
    return json.loads(q.head(n)[cols].replace([np.inf,-np.inf],np.nan).to_json(orient="records",force_ascii=False))

def main():
    if not TFILE.exists():raise RuntimeError("run fetch_today.py first")
    target=date.fromisoformat(TFILE.read_text(encoding="utf-8").strip());out=DATA/target.isoformat()/"r10_scan";out.mkdir(parents=True,exist_ok=True);log(f"[R10] T={target}")
    raw=history(target);feat,events=adjust(raw);feat=features(feat);bm=tai50(target);r7s,r7=scan_r7(feat,target,bm);r7=limits(r7,"R7");ins=inst_history(target)
    if ins.date.nunique()<10:raise RuntimeError(f"only {ins.date.nunique()} institutional dates")
    r05s,r05=scan_r05(feat,ins,target);r05=limits(r05,"R05");st=state();due,why=rebalance(st,feat,target,r7s["regime"])
    r7cols=["r7_rank","code","name","close","aclose","r7_score","r10","r20","r60","amt20","nearhigh","flow20","amtacc","clvflow20","t1_limit"]
    r05cols=["r05_rank","code","name","close","r05_score","r20","amount_ratio","ma20gap","prior60_position","clvflow10","clvflow5","Foreign3D","Foreign10D","Trust5D","t1_limit"]
    r7[[c for c in r7cols if c in r7.columns]].head(50).to_csv(out/"r7_candidates.csv",index=False,encoding="utf-8-sig");r05[[c for c in r05cols if c in r05.columns]].head(50).to_csv(out/"r05_candidates.csv",index=False,encoding="utf-8-sig")
    ms={"version":VERSION,"generated_at":datetime.now(TZ).isoformat(),"trade_date":target.isoformat(),"status":"PASS","privacy_boundary":"PUBLIC_MARKET_ONLY_NO_HOLDINGS_NO_NAV_NO_COST_BASIS","historical_ohlcv":{"unique_dates":int(raw.date.nunique()),"min_date":str(int(raw.date.min())),"max_date":str(int(raw.date.max())),"seed":"weekly public packages from official TWSE/TPEx + direct official current-week fill"},"corporate_action_method":{"method":"residual impossible-gap continuity pass","threshold":[.885,1.115],"consecutive_market_day_guard":True,"events_in_window":int(events),"archived_sidecar_present":False},"benchmark":{"name":"Taiwan 50 Total Return Index","source":"TWSE indicesReport/TAI50I","rows":int(len(bm))},"r7":{**r7s,"rebalance_due":bool(due),"rebalance_reason":why,"eligible_count":int(len(r7)),"top_candidates":records(r7,["r7_rank","code","name","close","r7_score","t1_limit"])},"r05":{**r05s,"institutional_trading_dates":int(ins.date.nunique()),"candidate_count":int(len(r05)),"top_candidates":records(r05,["r05_rank","code","name","close","r05_score","t1_limit"])},"warnings":[]}
    (out/"market_state.json").write_text(json.dumps(ms,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    new={"version":VERSION,"last_scan_date":target.isoformat(),"last_regime":r7s["regime"],"last_r7_rebalance_date":target.isoformat() if due else st.get("last_r7_rebalance_date"),"note":"Public market-strategy clock only. No user portfolio data."};STATE.write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding="utf-8")
    man={"version":VERSION,"trade_date":target.isoformat(),"status":"PASS","counts":{"ohlcv_rows":int(len(raw)),"ohlcv_dates":int(raw.date.nunique()),"institutional_rows":int(len(ins)),"institutional_dates":int(ins.date.nunique()),"r7_eligible":int(len(r7)),"r05_candidates":int(len(r05))}};(out/"scan_manifest.json").write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding="utf-8")
    log(json.dumps({"status":"PASS","T":target.isoformat(),"r7_regime":r7s["regime"],"r7_rebalance_due":due,"r7_eligible":len(r7),"r05_risk_on":r05s["risk_on"],"r05_candidates":len(r05)},ensure_ascii=False))
if __name__=="__main__":main()
