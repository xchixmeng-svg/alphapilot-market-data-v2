#!/usr/bin/env python3
from pathlib import Path

p=Path(__file__).resolve().parent/'r10_fast_validation.py'
s=p.read_text(encoding='utf-8')

anchor='''    cash = bt.INITIAL_CAPITAL
    positions: Dict[str, bt.Position] = {}
'''
insert='''    # Forensic regression probe extracted directly from the formal uploaded
    # workbook `全部掛單_410`. It changes no trading rule. Through 2021-04-26 it
    # requires each T-close generated BUY list to match the golden workbook
    # exactly so the FIRST divergence is reported immediately.
    GOLDEN_PREFIX = {
        20210104:[("R05","2426",18.05,14000),("R05","6288",34.90,7000),("R05","1802",21.30,12000),("R7","2415",29.15,9000),("R7","3149",26.55,8000)],
        20210105:[("R05","8215",30.50,8000)],
        20210107:[("R05","6235",20.70,11000)],
        20210111:[("R05","2886",29.85,8000)],
        20210112:[("R7","1533",54.10,4000)],
        20210113:[("R7","2401",22.90,11000)],
        20210129:[("R7","8033",21.45,11000)],
        20210202:[("R7","5014",17.40,16000)],
        20210204:[("R7","6547",132.00,2000)],
        20210205:[("R7","6547",145.50,1000),("R7","3169",72.30,3000)],
        20210218:[("R7","6547",175.00,1000),("R7","4961",203.50,1000)],
        20210226:[("R7","3312",19.00,15000),("R7","2610",13.70,14000)],
        20210311:[("R7","8099",39.65,5000)],
        20210317:[("R7","5452",22.50,13000),("R7","8040",47.20,3000)],
        20210329:[("R7","3122",42.65,7000)],
        20210331:[("R7","5272",41.75,7000),("R7","1517",19.80,10000)],
        20210401:[("R7","5272",43.65,7000),("R7","1517",21.80,8000)],
        20210412:[("R05","2030",16.00,11000)],
        20210415:[("R05","2884",26.75,11000)],
        20210419:[("R05","1319",39.80,7000)],
        20210422:[("R7","2022",14.20,22000)],
        20210426:[("R7","8927",42.65,6000)],
    }

    cash = bt.INITIAL_CAPITAL
    positions: Dict[str, bt.Position] = {}
'''
if anchor not in s: raise SystemExit('probe init anchor missing')
s=s.replace(anchor,insert,1)

old='''            if created: pending_buys.setdefault(exdate,[]).extend(created)

        nav_rows.append'''
new='''            if di <= 20210426:
                actual=[(o.strategy,o.code,round(float(o.limit),2),int(o.shares)) for o in created]
                expected=GOLDEN_PREFIX.get(di,[])
                expected=[(a,b,round(float(c),2),int(d)) for a,b,c,d in expected]
                if actual != expected:
                    raise RuntimeError(f"GOLDEN_PREFIX_DIVERGENCE date={di} expected={expected} actual={actual} cash={cash:.2f} nav={nav:.2f} exposure={exposure:.6f} positions={[(p.strategy,p.code,p.shares) for p in positions.values()]} sell_keys={sorted(sell_keys)}")
            if created: pending_buys.setdefault(exdate,[]).extend(created)

        nav_rows.append'''
if old not in s: raise SystemExit('probe created anchor missing')
s=s.replace(old,new,1)
s=s.replace('AlphaPilot-R10-FastValidation-v7-GOLDEN-POOL-TIMING','AlphaPilot-R10-FastValidation-v8-GOLDEN-PREFIX-PROBE',1)
p.write_text(s,encoding='utf-8')
print('PATCHED',p)
