// Node port of the game's chart rubric (app.02991 analyze/decide + app.02996 outcome/clearClass).
// Build tools use it to pick cases; CI re-checks the game computes the same class for every picked case.
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
export const E = require('../src/engine.js')

export const dec = r => ({ o: r.o.map(x => x / 100), h: r.h.map(x => x / 100), l: r.l.map(x => x / 100), c: r.c.map(x => x / 100), v: r.v.slice(),
  atr: r.atr.map(x => x == null ? null : x / 10000), ma20: r.m20.map(x => x == null ? null : x / 10000), ma60: r.m60.map(x => x == null ? null : x / 10000) })
export const minA = (a, s, e) => { let v = Infinity; for (let i = Math.max(0, s); i <= Math.min(a.length - 1, e); i++) v = Math.min(v, a[i]); return v }
export const maxA = (a, s, e) => { let v = -Infinity; for (let i = Math.max(0, s); i <= Math.min(a.length - 1, e); i++) v = Math.max(v, a[i]); return v }
export const avg = (a, s, e) => { let n = 0, t = 0; for (let i = Math.max(0, s); i <= Math.min(a.length - 1, e); i++) { t += a[i]; n++ } return n ? t / n : 0 }
export const upTo = (R, j) => Object.fromEntries(Object.entries(R).map(([k, v]) => [k, Array.isArray(v) ? v.slice(0, j + 1) : v]))

export function analyze(R, j) {
  const c = R.c[j], a = R.atr[j] || c * .02, ma = R.ma20[j], sup = minA(R.l, j - 20, j - 1), res = maxA(R.h, j - 20, j - 1)
  const hi1 = maxA(R.h, j - 24, j - 13), hi2 = maxA(R.h, j - 12, j), lo1 = minA(R.l, j - 24, j - 13), lo2 = minA(R.l, j - 12, j)
  const up = hi2 > hi1 * 1.005 && lo2 > lo1 * 1.005, down = hi2 < hi1 * .995 && lo2 < lo1 * .995, flat = Math.abs(hi2 / hi1 - 1) < .015 && Math.abs(lo2 / lo1 - 1) < .015
  const trend = up ? 'up' : down ? 'down' : flat ? 'range' : 'unclear', dist = ma ? c / ma - 1 : null
  const stop = E.roundToTick(Math.max(.01, sup - a * .45), 'down'), target = E.roundToTick(res > c * 1.005 ? res : maxA(R.h, j - 60, j - 1)), risk = c - stop, rr = risk > 0 ? (target - c) / risk : 0
  const pos = res > sup ? (c - sup) / (res - sup) : .5, loc = c > res * 1.002 ? 'above' : pos < .33 ? 'nearSup' : pos > .67 ? 'nearRes' : 'middle'
  const v20 = avg(R.v, j - 20, j - 1), vr = v20 ? R.v[j] / v20 : 1, breakout = c > sup && c > res * 1.002
  let decision
  if (trend === 'down' || rr < 1 || (dist != null && dist > .1)) decision = 'notrade'
  else if (trend === 'up' && ma && c > ma && (dist == null || dist < .08) && rr >= 1.5 && (loc !== 'nearRes' || (breakout && vr >= 1.5))) decision = 'trade'
  else decision = 'wait'
  return { c, a, ma, sup, res, hi1, hi2, lo1, lo2, trend, dist, stop, target, rr, loc, vr, breakout, decision }
}

// A case is only used for a graded decision when it is clearly inside its class (not on a threshold edge).
export function clearClass(A) {
  const near = (x, t, w) => x != null && Math.abs(x - t) < w
  if (A.decision === 'trade') return A.rr >= 1.6 && A.dist != null && A.dist < .07 ? 'trade' : null
  if (A.decision === 'notrade') {
    if (A.hi2 < A.hi1 * .98 && A.lo2 < A.lo1 * .98) return 'notrade'
    if (A.trend !== 'down' && A.rr < .8 && !near(A.dist, .1, .015)) return 'notrade'
    if (A.trend !== 'down' && A.dist != null && A.dist > .12) return 'notrade'
    return null
  }
  if (near(A.rr, 1, .15) || near(A.rr, 1.5, .15) || near(A.dist, .08, .01) || near(A.dist, .1, .015) || near(A.dist, 0, .005)) return null
  if (A.trend === 'up' && !(A.hi2 > A.hi1 * 1.015 && A.lo2 > A.lo1 * 1.015)) return null
  return 'wait'
}
export function notradeKind(A) { return A.trend === 'down' ? 'down' : A.dist != null && A.dist > .1 ? 'chase' : 'rr' }

// What would have happened to a long entered at the close of bar i with the structural stop and target.
export function outcome(R, i, A, H = 20) {
  const e = R.c[i]
  for (let k = 1; k <= H && i + k < R.c.length; k++) {
    const j = i + k
    if (R.l[j] <= A.stop) return { res: 'stop', k, ret: Math.min(R.o[j], A.stop) / e - 1 }
    if (R.h[j] >= A.target) return { res: 'target', k, ret: Math.max(R.o[j], A.target) / e - 1 }
  }
  const j = Math.min(R.c.length - 1, i + H)
  return { res: 'open', k: j - i, ret: R.c[j] / e - 1 }
}
