// Layer 4 — DAILY_SCENARIOS. Finds one real historical window for each required script in the training-era
// bank (pools A/B; the 2024–2025 C pool stays unseen for the blind exams) and writes build/scenarios.json.
// Detection may look at the future (that is how we know a window is a "false breakout"); the game never does:
// at checkpoint k the player and the rubric only see bars ≤ start+k (see app.02994 and the leakage audit).
import fs from 'node:fs'
import { E, dec, minA, maxA, avg, upTo, analyze } from './rubric.mjs'
const bank = JSON.parse(fs.readFileSync('build/bank.json', 'utf8'))
const DAYS = 7   // checkpoints D0..D+7 → 8 decisions
const PER = 6    // cases per script → 6 sets of 12 different historical cases

const SCRIPTS = [
  { id: 'uptrend_pullback', title: '上升趨勢正常回檔', lesson: '上升結構裡回到 MA20 附近、低點守住，是順勢的較好位置；回檔不等於轉弱。',
    test: (R, s) => { const A = analyze(R, s); return A.trend === 'up' && A.ma && R.c[s] >= A.ma * .98 && R.c[s] <= A.ma * 1.03 && R.c[s] < maxA(R.h, s - 10, s - 1) * .97 && maxA(R.h, s + 1, s + DAYS) > maxA(R.h, s - 10, s - 1) && minA(R.l, s + 1, s + DAYS) > A.sup } },
  { id: 'false_breakout', title: '假突破', lesson: '收盤站上前高後，很快又收回區間內：突破論點失效。先定好失策，失效就退出，不要硬抱。',
    test: (R, s) => { const ph = maxA(R.h, s - 20, s - 1); return R.c[s] > ph * 1.005 && minA(R.c, s + 1, s + 3) < ph * .995 && R.c[s + DAYS] < ph } },
  { id: 'retest_success', title: '突破後成功回測', lesson: '突破後回到原壓力附近，收盤守住，原壓力轉成支撐；回測守住的位置失策近、R:R 好。',
    test: (R, s) => { const ph = maxA(R.h, s - 20, s - 1); return R.c[s] > ph * 1.005 && minA(R.l, s + 1, s + 5) <= ph * 1.015 && minA(R.c, s + 1, s + DAYS) >= ph * .99 && maxA(R.h, s + 1, s + DAYS) > R.h[s] } },
  { id: 'breakout_fail', title: '突破後失敗', lesson: '突破後一路跌回區間下方：這就是為什麼進場前一定要先知道失策在哪裡。',
    test: (R, s) => { const ph = maxA(R.h, s - 20, s - 1); return R.c[s] > ph * 1.005 && R.c[s + DAYS] < ph * .96 && minA(R.c, s + 1, s + DAYS) < ph * .95 } },
  { id: 'oversold_bounce', title: '超賣反彈但沒有反轉', lesson: 'RSI 超賣後的反彈，在沒有較高低點、沒有突破前高之前，只是下降結構裡的反彈。',
    test: (R, s) => { const A = analyze(R, s), rsi = E.teachingIndicators(upTo(R, s)).rsi14[s]; return A.trend === 'down' && rsi < 32 && maxA(R.c, s + 1, s + 4) > R.c[s] * 1.03 && minA(R.l, s + 3, s + DAYS) < minA(R.l, s - 5, s) } },
  { id: 'true_reversal', title: '真正反轉', lesson: '止跌 → 較高低點 → 突破前高，三步到齊才是確認反轉；確認之後才找進場。',
    test: (R, s) => { const pl = minA(R.l, s - 30, s - 11), hl = minA(R.l, s - 10, s); return maxA(R.h, s - 45, s - 25) > R.c[s] * 1.12 && hl > pl * 1.02 && R.c[s] > maxA(R.h, s - 15, s - 3) && R.c[s + DAYS] > R.c[s] && minA(R.l, s + 1, s + DAYS) > hl } },
  { id: 'long_range', title: '長時間盤整', lesson: '高低點一直重疊的盤整區，中間位置沒有優勢；等突破或回到邊緣，大部分時間不交易。',
    test: (R, s) => { const hi = maxA(R.h, s - 40, s), lo = minA(R.l, s - 40, s); return (hi - lo) / R.c[s] < .1 && maxA(R.h, s + 1, s + DAYS) <= hi * 1.01 && minA(R.l, s + 1, s + DAYS) >= lo * .99 && Math.abs(R.c[s] / ((hi + lo) / 2) - 1) < .025 } },
  { id: 'pretty_indicators', title: '指標很好看但結構很差', lesson: 'KD 黃金交叉、MACD 柱體改善都很漂亮，但價格仍在下降結構、仍在 MA20 下方：結構優先。',
    test: (R, s) => { const A = analyze(R, s), kd = E.teachingIndicators(upTo(R, s)).kd, cross = (kd.k[s - 1] - kd.d[s - 1]) < 0 && (kd.k[s] - kd.d[s]) > 0; return A.trend === 'down' && A.ma && R.c[s] < A.ma && cross && minA(R.l, s + 1, s + DAYS) < R.l[s] * .97 } },
  { id: 'lowvol_breakout', title: '無量突破', lesson: '量縮的突破缺乏追價，常常走不遠；要等回測守住或量能補上再說。',
    test: (R, s) => { const ph = maxA(R.h, s - 20, s - 1), vr = R.v[s] / Math.max(1, avg(R.v, s - 20, s - 1)); return R.c[s] > ph * 1.005 && vr < .9 && R.c[s + 5] < R.c[s] } },
  { id: 'highvol_breakout', title: '有量突破', lesson: '量能明顯放大的突破，可信度高；但仍要先定失策，並注意別在離 MA20 太遠時追。',
    test: (R, s) => { const ph = maxA(R.h, s - 20, s - 1), vr = R.v[s] / Math.max(1, avg(R.v, s - 20, s - 1)); return R.c[s] > ph * 1.005 && vr >= 2 && R.c[s + DAYS] > R.c[s] * 1.05 } },
  { id: 'good_decision_stopped', title: '好決策最後停損', lesson: '條件到位、失策合理的進場，還是會被打到停損。照計畫停損就是好決策；結果賠錢 ≠ 決策錯誤。',
    test: (R, s) => { const A = analyze(R, s); return A.decision === 'trade' && minA(R.l, s + 1, s + DAYS) <= A.stop } },
  { id: 'bad_decision_won', title: '爛決策剛好賺錢', lesson: '離均線太遠、R:R 不划算時追進，這次剛好賺錢，下次就未必；結果賺錢 ≠ 決策正確。',
    test: (R, s) => { const A = analyze(R, s); return A.decision === 'notrade' && A.trend !== 'down' && A.dist != null && A.dist > .1 && R.c[s + DAYS] > R.c[s] * 1.06 } }
]

const rounds = bank.rounds.filter(r => r.pool === 'A' || r.pool === 'B').sort((a, b) => a.id < b.id ? -1 : 1)
const used = new Set(), hits = {}, blockers = []
const decoded = new Map(rounds.map(r => [r.id, dec(r)]))
for (let set = 0; set < PER; set++) {
  for (const sc of SCRIPTS) {
    let hit = null
    for (const r of rounds) {
      if (used.has(r.id)) continue
      const R = decoded.get(r.id), n = R.c.length
      for (let s = n - DAYS - 2; s >= 120 && !hit; s--) {
        if (!R.atr[s] || !R.ma20[s] || !R.ma60[s]) continue
        try { if (sc.test(R, s)) hit = { roundId: r.id, start: s } } catch (e) { }
      }
      if (hit) break
    }
    if (!hit) { blockers.push(sc.id + '#' + (set + 1)); continue }
    used.add(hit.roundId);
    (hits[set] = hits[set] || []).push({ sc, hit })
  }
}
const out = []
for (let set = 0; set < PER; set++) {
  // order inside a set is shuffled so the script cannot be guessed from the position
  const list = (hits[set] || []).slice(), rnd = E.mulberry32(E.hash32('SCEN|' + set))
  for (let k = list.length - 1; k > 0; k--) { const m = Math.floor(rnd() * (k + 1)); [list[k], list[m]] = [list[m], list[k]] }
  for (const { sc, hit } of list) out.push({ id: 'S' + String(out.length + 1).padStart(2, '0'), set: set + 1, script: sc.id, title: sc.title, lesson: sc.lesson, roundId: hit.roundId, start: hit.start, days: DAYS })
}
fs.writeFileSync('build/scenarios.json', JSON.stringify({ version: 2, days: DAYS, sets: PER, scripts: SCRIPTS.length, scenarios: out, blockers }))
console.log('SCENARIOS', out.length, 'in', PER, 'sets', blockers.length ? 'missing: ' + blockers.join(',') : '')
