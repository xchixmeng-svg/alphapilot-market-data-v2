  // ───────────────────────── V4 teaching indicators ─────────────────────────
  // Indicators are computed from the same causal OHLC history already visible to the learner.
  // They are teaching evidence, never a standalone buy/sell signal.
  function emaSeries(values, period) {
    period = Math.max(1, Math.floor(period || 1));
    var out = new Array(values.length).fill(null), alpha = 2 / (period + 1), prev = null;
    for (var i = 0; i < values.length; i++) {
      var x = values[i];
      if (!isNum(x)) { out[i] = prev; continue; }
      prev = prev === null ? x : alpha * x + (1 - alpha) * prev;
      out[i] = prev;
    }
    return out;
  }

  function macdSeries(close, fast, slow, signal) {
    fast = fast || 12; slow = slow || 26; signal = signal || 9;
    var ef = emaSeries(close, fast), es = emaSeries(close, slow), dif = new Array(close.length).fill(null);
    for (var i = 0; i < close.length; i++) if (isNum(ef[i]) && isNum(es[i])) dif[i] = ef[i] - es[i];
    var dea = emaSeries(dif, signal), hist = new Array(close.length).fill(null);
    for (i = 0; i < close.length; i++) if (isNum(dif[i]) && isNum(dea[i])) hist[i] = dif[i] - dea[i];
    return { dif: dif, dea: dea, hist: hist, fast: fast, slow: slow, signal: signal };
  }

  function kdSeries(high, low, close, period) {
    period = period || 9;
    var n = Math.min(high.length, low.length, close.length), rsv = new Array(n).fill(null), k = new Array(n).fill(null), d = new Array(n).fill(null), pk = 50, pd = 50;
    for (var i = 0; i < n; i++) {
      var s = Math.max(0, i - period + 1), hh = -Infinity, ll = Infinity;
      for (var j = s; j <= i; j++) { if (isNum(high[j])) hh = Math.max(hh, high[j]); if (isNum(low[j])) ll = Math.min(ll, low[j]); }
      if (!isNum(close[i]) || !isFinite(hh) || !isFinite(ll)) continue;
      var rv = hh === ll ? 50 : clamp((close[i] - ll) / (hh - ll) * 100, 0, 100);
      pk = (2 * pk + rv) / 3; pd = (2 * pd + pk) / 3;
      rsv[i] = rv; k[i] = pk; d[i] = pd;
    }
    return { rsv: rsv, k: k, d: d, period: period };
  }

  function rsiSeries(close, period) {
    period = period || 14;
    var n = close.length, out = new Array(n).fill(null);
    if (n <= period) return out;
    var gain = 0, loss = 0, i;
    for (i = 1; i <= period; i++) {
      if (!isNum(close[i]) || !isNum(close[i - 1])) continue;
      var ch = close[i] - close[i - 1]; if (ch > 0) gain += ch; else loss -= ch;
    }
    var ag = gain / period, al = loss / period;
    function value() { if (ag === 0 && al === 0) return 50; if (al === 0) return 100; if (ag === 0) return 0; var rs = ag / al; return 100 - 100 / (1 + rs); }
    out[period] = value();
    for (i = period + 1; i < n; i++) {
      if (!isNum(close[i]) || !isNum(close[i - 1])) { out[i] = out[i - 1]; continue; }
      var ch2 = close[i] - close[i - 1], g = Math.max(0, ch2), l = Math.max(0, -ch2);
      ag = (ag * (period - 1) + g) / period; al = (al * (period - 1) + l) / period; out[i] = value();
    }
    return out;
  }

  function teachingIndicators(R) {
    if (!R || !Array.isArray(R.c)) return null;
    return { macd: macdSeries(R.c, 12, 26, 9), kd: kdSeries(R.h || [], R.l || [], R.c, 9), rsi14: rsiSeries(R.c, 14) };
  }
