import pandas as pd, numpy as np, bear as b, grid as g
e=pd.read_pickle('/home/claude/work/sysevents_mix.pkl'); G3=pd.read_pickle('/home/claude/work/g3_mix.pkl'); g.EV['混合']=(e,G3)
T=pd.read_pickle('/home/claude/work/new_trades.pkl')
C,O,PC,M60=g.C,g.O,g.PC,g.M60b
D=C.index; prev=lambda d: D[D.get_loc(d)-1]
ix=g.tx.close; ma=ix.rolling(60).mean(); BEAR=((ix<ma)&(ma<ma.shift(20))).to_dict()
lim=lambda d: 1.095 if d>=pd.Timestamp('2015-06-01') else 1.065
bad=[]; cnt={}
def flag(r,msg): bad.append((r.code,r.kind,str(r.ent.date()),str(r.out.date()),r.why,msg))
for r in T.itertuples():
    s=r.sig  # 進場訊號日
    # 進場價＝進場日開盤
    if abs(O.at[r.ent,r.code]-r.epx)>1e-6: flag(r,'進場價不是開盤價')
    if O.at[r.ent,r.code]>=PC.at[r.ent,r.code]*(lim(r.ent)-0.003): flag(r,'開盤漲停卻買到')
    if r.kind=='多頭':
        ok=(g.VAL20.at[s,r.code]>=3e7, g.N.at[s,r.code]>=250, g.TOH.at[s,r.code]>=-0.001, g.VAL5.at[s,r.code]>g.VAL20.at[s,r.code]*2.0, G3.get(s,0)>=2, not BEAR.get(s,False))
        names=['成交額','上市天數','一年新高','量增2倍','三燈兩盞','熊市指標']
        for k,v in zip(names,ok):
            if not v: flag(r,'進場條件不符：'+k)
    else:
        if e.get(s)!='抓底進場': flag(r,'抄底單不在抓底訊號隔天')
        if r.code not in g.candA(s): flag(r,'抄底股不在跌最深名單')
    # 出場價＝出場日開盤（停牌用前收）
    o=O.at[r.out,r.code]
    if o==o and abs(o-r.opx)>1e-6: flag(r,'出場價不是開盤價')
    xs=prev(r.out)
    cl=C.at[xs,r.code]; why=r.why
    if why=='虧10%停損' and not (cl<r.epx*0.85): flag(r,'停損條件不成立')
    if why=='跌破季線' and not (cl<M60.at[xs,r.code]): flag(r,'季線條件不成立')
    if why=='抓底股漲50%停利' and not (cl>=r.epx*1.5): flag(r,'停利條件不成立')
    if why=='三燈兩綠，抓底股換多頭' and not (G3.get(xs,0)>=2): flag(r,'三燈條件不成立')
    if why=='大盤出場訊號' and e.get(xs)!='出場': flag(r,'沒有出場訊號')
    if why=='帳戶回落20%煞車' and not (G3.get(xs,0)<3): flag(r,'煞車條件不成立')
    # 有沒有更早就該出場卻沒出
    hold=C.loc[r.ent:xs, r.code].iloc[:-1]
    if r.kind=='多頭':
        early=hold[(hold<r.epx*0.85)|(hold<M60.loc[hold.index, r.code])]
        if len(early): flag(r,f'更早 {early.index[0].date()} 就該出場')
    else:
        early=hold[hold>=r.epx*1.5]
        if len(early): flag(r,f'更早 {early.index[0].date()} 就該停利')
    cnt[why]=cnt.get(why,0)+1
print('交易',len(T),'出場原因',cnt)
B=pd.DataFrame(bad,columns=['代號','類型','買','賣','原因','問題'])
print('有問題的',len(B)); print(B.問題.str.split('：').str[0].str.replace(r'更早 .*', '更早該出場', regex=True).value_counts().to_string())
print(B.head(15).to_string(index=False))
B.to_csv('/home/claude/work/rulecheck_bad.csv',index=False,encoding='utf-8-sig')
