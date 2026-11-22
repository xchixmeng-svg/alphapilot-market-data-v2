"""選股核心（V2，2026-11-22 凍結）：線上 picks_daily.py 和歷史對帳共用這一份。

全部 N+1：第 N 天收盤後判斷（decide），第 N+1 天開盤成交（execute）。
  抓底：燈號抓底訊號 → 買離 250 日最高跌最深的 10 檔（20 日均成交額 ≥ 3,000 萬、上市 ≥ 120 天）
        收盤 ≥ 買價 ×1.5 停利；三燈亮兩盞或抓底股賣光 → 剩下的換多頭
        大盤收盤 < 抓底那天 ×0.8 → 抓底股全賣（災難停損），等三燈亮兩盞
  多頭：創 250 日新高、5 日均成交額 > 20 日均 ×2、20 日均成交額 ≥ 3,000 萬、上市 ≥ 250 天、
        最新已公布月營收年增 > 20%，依 60 日漲幅排序取前 40，有空位依序買
        收盤 < 買價 ×0.9 停損；收盤 < 60 日均線出場
        三燈 < 2 或熊市指標（大盤在季線下且季線 20 天前比較高）→ 不買新股
        帳戶從高點回落 20%：三燈 < 3 → 多頭股全賣、不買；三燈 = 3 → 高點重設
  共同：燈號出場訊號 → 全賣；每檔帳戶 9%，最多 10 檔，資金最多 90%；買賣各扣 0.3%
        開盤漲停（≥ 前收 ×1.092）買不到換下一檔；沒有開盤價就延後賣；連續 20 天沒價格用最後收盤價出清
"""
import math

SLOT, MAXN, TP, STOP, BRAKE, VOLR, DS = 0.09, 10, 0.50, -0.10, 0.20, 2.0, 0.20
CIN, COUT = 0.003, 0.003
CASH0 = 1_000_000


def new_state():
    return {"cash": CASH0, "pos": {}, "mode": "bull", "peak": CASH0, "ix0": None, "carry": [],
            "pend": {"sells": [], "buys": [], "sellall": False}, "why": {}, "last_date": ""}


def nav(st, px):
    """帳戶價值：現金＋持股×價格（沒有價格用買價）。px: dict code→價格"""
    return st["cash"] + sum(p["sh"] * (px.get(c) if px.get(c) == px.get(c) and px.get(c) is not None else p["epx"]) for c, p in st["pos"].items())


def execute(st, day, opn, lim_open, lim_ref, lim, last_px, days_since_px, fractional=True):
    """第 N+1 天開盤：先賣後買。opn: code→今天開盤（成交價）；lim_open／lim_ref: 判斷開盤漲停用的開盤價和前收
    （線上用原始價格，除權息日用參考價，不用到當天收盤）；lim: 漲停門檻倍數；
    last_px: code→最後一筆收盤；days_since_px: code→距最後一筆價格幾個交易日。回傳成交紀錄。"""
    done = []
    pend = st["pend"]
    todo = list(st["pos"]) if pend.get("sellall") else list(dict.fromkeys(st["carry"] + pend.get("sells", [])))
    st["carry"] = []
    for c in todo:
        if c not in st["pos"]:
            continue
        x = opn.get(c)
        note = ""
        if not (x == x and x):
            if days_since_px.get(c, 0) >= 20 and last_px.get(c):
                x = last_px[c]; note = "（停牌 20 天，用最後收盤價出清）"
            else:
                st["carry"].append(c); continue
        p = st["pos"].pop(c)
        st["cash"] += p["sh"] * x * (1 - COUT)
        why = "大盤出場訊號" if pend.get("sellall") and c not in st["why"] else st["why"].get(c, "大盤出場訊號")
        done.append(dict(date=day, side="賣", code=c, kind=p["kind"], sig=p.get("sig"), ent=p["ent"], epx=p["epx"], px=x, sh=p["sh"],
                         ret=x * (1 - COUT) / (p["epx"] * (1 + CIN)) - 1, why=why + note))
        st["why"].pop(c, None)
    if pend.get("buys"):
        # 賣出後的帳戶價值＝現金＋剩下持股（訊號日價格），和回測一樣
        val = nav(st, pend.get("px", {}))
        for b in pend["buys"]:
            c = b["code"]
            if len(st["pos"]) >= MAXN or st["cash"] - val * (1 - MAXN * SLOT) < val * 0.02:
                break
            if c in st["pos"]:
                continue
            x = opn.get(c); pc = lim_ref.get(c); xo = lim_open.get(c, x)
            if not (x == x and x) or (pc == pc and pc and xo == xo and xo >= pc * lim):
                continue
            amt = min(st["cash"] - val * (1 - MAXN * SLOT), val * SLOT)
            sh = amt / (x * (1 + CIN)) if fractional else math.floor(amt / (x * (1 + CIN)))
            if sh <= 0:
                continue
            st["pos"][c] = dict(sh=sh, kind=b["kind"], ent=day, epx=x, sig=pend.get("day"))
            st["cash"] -= sh * x * (1 + CIN)
            done.append(dict(date=day, side="買", code=c, kind=b["kind"], sig=pend.get("day"), ent=day, epx=x, px=x, sh=sh, ret=None, why=""))
    st["pend"] = {"sells": [], "buys": [], "sellall": False}
    return done


def decide(st, day, close, ma60, act, g, taiex, bear, candA, candM):
    """第 N 天收盤：依燈號事件、三燈、熊市指標決定明天的單。close/ma60: code→值；candA/candM: 函式，回傳排序好的代號清單。
    act: 'exit'／'bottom'／'back'／''（燈號這天的動作）。回傳 (sells, buys, sellall, room)"""
    pos = st["pos"]; why = st["why"]
    val = nav(st, close)
    sells, buys, sellall, room = [], [], False, 0
    log = None

    def fin():
        st["pend"] = {"sells": sells, "buys": buys, "sellall": sellall, "day": day, "px": dict(close)}
        st["last_date"] = day; st["value"] = val
        return sells, buys, sellall, room, log

    if act == "exit":
        why.clear(); sellall = True; st["mode"] = "cash"; log = "出場"
        return fin()
    if st["mode"] == "cash":
        if act == "bottom":
            st["mode"] = "bottom"; st["ix0"] = taiex; log = "抓底買進"
            buys = [dict(code=c, kind="抄底") for c in candA()]; room = MAXN - len(pos)
        elif act == "back":
            st["mode"] = "bull"; st["peak"] = val; log = "回場"
        return fin()
    if st["mode"] == "cashds":
        if g >= 2:
            st["mode"] = "bull"; st["peak"] = val; log = "災難停損後回場"
        else:
            return fin()
    if st["mode"] == "bottom" and DS and st.get("ix0") and taiex is not None and taiex < st["ix0"] * (1 - DS):
        for c in pos: why[c] = "抓底災難停損"
        sells = list(pos); st["mode"] = "cashds"; log = "抓底災難停損"
        return fin()
    if st["mode"] == "bottom":
        for c, p in pos.items():
            x = close.get(c)
            if p["kind"] == "抄底" and TP and x == x and x is not None and x >= p["epx"] * (1 + TP):
                sells.append(c); why[c] = "抓底股漲50%停利"
        left = [c for c, p in pos.items() if p["kind"] == "抄底" and c not in sells]
        if g >= 2 or not left:
            sells += left
            for c in left: why[c] = "三燈兩盞，抓底股換多頭"
            st["mode"] = "bull"; st["peak"] = val; log = "轉多頭"
        else:
            return fin()
    for c, p in pos.items():
        x = close.get(c)
        if p["kind"] == "多頭" and x == x and x is not None:
            m = ma60.get(c)
            if STOP and x < p["epx"] * (1 + STOP):
                sells.append(c); why[c] = f"收盤虧{int(-STOP * 100)}%停損"
            elif m == m and m is not None and x < m:
                sells.append(c); why[c] = "跌破季線"
    room = MAXN - len([c for c in pos if c not in sells])
    if g < 2:
        room = 0
    if bear:
        room = 0
    st["peak"] = max(st["peak"], val)
    if BRAKE and val < st["peak"] * (1 - BRAKE):
        if g < 3:
            br = [c for c, p in pos.items() if p["kind"] == "多頭" and c not in sells]
            sells += br; room = 0
            for c in br: why[c] = "帳戶回落20%煞車"
        else:
            st["peak"] = val
    if room > 0:
        buys = [dict(code=c, kind="多頭") for c in candM() if c not in pos][:room + 10]
    return fin()
