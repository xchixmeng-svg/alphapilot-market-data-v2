"""交易計畫結算（plan-v2）：依說明書第六、七章，模擬「T+1 開盤買進、不當沖」的真實操作。

買進：訊號日 T 的隔天（T+1）開盤價買進。以下三種情況放棄不買：
      開盤 ≤ 停損價；開盤 ≥ 目標價；一字漲停鎖死（T+1 最低價 ≥ T 日收盤 × 1.095）。
持有：買進當天記為第 1 天。買進當天不賣（不當沖）。
  第 2 天起：災難停損（收盤停損價再減 1 倍 ATR20）盤中觸發 → 開盤已低於則以開盤價，否則以災難價成交。
  每天收盤後檢查，觸發者「隔天開盤賣出」：
    1. 收盤 < 停損價（碰過 +10% 後停損上移到買價，稱保本出場）
    2. 碰過 +10% 後：收盤 < 持有期間最高收盤 − 3 × ATR20（移動停利）
    3. 第 N 天（主規則 10）收盤時，期間最高價從未達到買價 +5%（發動失敗）
    4. 第 30 天收盤時仍未碰過 +10%
    5. 第 60 天（安全上限）
  報酬扣手續費（買賣各 0.1425%）與證交稅（0.3%）。ATR20 取訊號日的值（不看未來）。價格未還原除權息。

三組停損規則平行結算（推播只依主規則，另兩組只記錄，3 個月後比較）：
  main：停損距離下限 1.5 倍 ATR、時間停損 10 天；loose：2.5 倍、15 天；tight：1.0 倍、7 天。
  各組停損價 = min(AI 原始停損價, 收盤 − 下限倍數 × ATR20)，移動停利一律 3 倍 ATR20。
"""
import math

PLAN_VERSION = "plan-v2"
COST = 0.001425 * 2 + 0.003
HIT, EARLY = 0.10, 0.05
TRAIL_ATR, DISASTER_ATR = 3.0, 1.0
TIME30, CAP60 = 30, 60
LIMIT_UP = 1.095
VARIANTS = {"main": (1.5, 10), "loose": (2.5, 15), "tight": (1.0, 7)}


def _f(x):
    try:
        x = float(x)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _one(ref_close, target, stop_v, atr, bars, time_days):
    out = {"status": "PENDING", "stop": round(stop_v, 2)}
    if not bars:
        return out
    d, o, h, l, c = bars[0]
    if None in (o, h, l, c):
        out.update(status="CANCELLED", reason="T+1 無成交資料（可能停牌）")
        return out
    if o <= stop_v:
        out.update(status="CANCELLED", reason=f"開盤 {o:g} 已低於停損 {stop_v:.2f}")
        return out
    if o >= target:
        out.update(status="CANCELLED", reason=f"開盤 {o:g} 已高於目標 {target:g}，不追價")
        return out
    if l >= ref_close * LIMIT_UP:
        out.update(status="CANCELLED", reason="一字漲停鎖死，買不到")
        return out
    buy = o
    disaster = stop_v - DISASTER_ATR * atr
    out.update(status="OPEN", buy_date=d, buy_price=round(buy, 2), disaster=round(disaster, 2))
    stop_level, armed, best_close, max_high = stop_v, False, None, buy
    hit_day = moved_day = None
    pending = None
    for k, (d, o, h, l, c) in enumerate(bars, start=1):
        if None in (o, h, l, c):
            continue
        if pending:
            return _close(out, d, o, pending, k, buy, max_high, hit_day)
        if k >= 2:  # 災難停損：券商觸價單，T+2 起生效
            if o <= disaster:
                return _close(out, d, o, "災難停損（開盤跳空）", k, buy, max(max_high, h), hit_day)
            if l <= disaster:
                return _close(out, d, disaster, "災難停損", k, buy, max(max_high, h), hit_day)
        max_high = max(max_high, h)
        if moved_day is None and max_high >= buy * (1 + EARLY):
            moved_day = k
        if not armed and h >= buy * (1 + HIT):
            armed, hit_day = True, k
            stop_level = max(stop_level, buy)
        if armed:
            best_close = c if best_close is None else max(best_close, c)
        if c < stop_level:
            pending = "保本出場" if armed and stop_level >= buy else "收盤跌破停損"
        elif armed and c < best_close - TRAIL_ATR * atr:
            pending = "移動停利"
        elif moved_day is None and k >= time_days:
            pending = f"{time_days} 天未漲到 +5%，發動失敗"
        elif not armed and k >= TIME30:
            pending = "30 天未達 +10%"
        elif k >= CAP60:
            pending = "持有滿 60 天"
        out.update(hold_days=k, last_close=c, unrealized=round((c / buy - 1) * 100, 2),
                   max_gain=round((max_high / buy - 1) * 100, 2), hit10_day=hit_day, trailing=armed,
                   stop_now=round(stop_level, 2), exit_pending=pending)
    return out


def _close(out, d, px, reason, k, buy, max_high, hit_day):
    gross = px / buy - 1
    out.update(status="CLOSED", sell_date=d, sell_price=round(px, 2), exit_reason=reason, hold_days=k,
               ret_gross=round(gross * 100, 2), ret_net=round((gross - COST) * 100, 2),
               max_gain=round((max(max_high, px) / buy - 1) * 100, 2), hit10_day=hit_day, exit_pending=None)
    return out


def simulate(rec, bars, atr20, ref_close=None):
    """rec：一筆有效推薦（含 target、stop；stop_ai 為 AI 原始停損）。bars：T 日之後的日 K [(date, o, h, l, c), ...]。
    回傳主規則結果，另附 variants（寬鬆、緊縮兩組）。"""
    target, atr = _f(rec.get("target")), _f(atr20)
    ref_close = _f(ref_close) or _f((rec.get("feat") or {}).get("close"))
    ai_stop = _f(rec.get("stop_ai")) or _f(rec.get("stop")) or _f(rec.get("exit"))
    if None in (target, atr, ref_close, ai_stop):
        return {"status": "NO_PLAN", "plan_version": PLAN_VERSION}
    res = {}
    for name, (k_atr, tdays) in VARIANTS.items():
        stop_v = min(ai_stop, ref_close - k_atr * atr)
        res[name] = _one(ref_close, target, stop_v, atr, bars, tdays)
    main = dict(res["main"])
    main["plan_version"] = PLAN_VERSION
    main["variants"] = {k: v for k, v in res.items() if k != "main"}
    return main
