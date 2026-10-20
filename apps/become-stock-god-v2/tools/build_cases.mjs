// Picks the graded chart positions for the blind exam, the start / retest exams and the decision drills, so the
// correct answers are balanced (blind exam 20 / 40 / 40: no single answer scores above 40%) and every graded case sits clearly inside its class.
// Also measures how each class actually played out afterwards (shown to players as historical base rates).
// Pools never overlap: blind exam = primary C; start/retest = reserve C #1–10 (early / late windows);
// final exam = reserve C #11–20; decision drills = training rounds (A/B). Writes build/cases.json.
import fs from 'node:fs'
import { E, dec, analyze, clearClass, notradeKind, outcome } from './rubric.mjs'
const bank = JSON.parse(fs.readFileSync('build/bank.json', 'utf8'))
const LO = 130, HI = 340   // HI + 20 bars stay inside the round for the outcome reveal
const byId = id => bank.rounds.find(r => r.id === id)
const sortIds = rs => rs.map(r => r.id).sort()
const EXAM = sortIds(bank.rounds.filter(r => r.slot === 'primary' && r.pool === 'C'))
const RESERVE = sortIds(bank.rounds.filter(r => r.slot === 'reserve' && r.pool === 'C'))
const TRAIN = sortIds(bank.rounds.filter(r => r.slot === 'primary' && (r.pool === 'A' || r.pool === 'B')))
const TESTS = RESERVE.slice(0, 10), FINAL = RESERVE.slice(10)

function candidates(id, lo, hi) {
  const R = dec(byId(id)), out = { trade: [], wait: [], notrade: [] }
  for (let i = lo; i <= hi; i++) {
    if (!R.atr[i] || !R.ma20[i] || !R.ma60[i]) continue
    const A = analyze(R, i), k = clearClass(A)
    if (k) out[k].push({ i, kind: k === 'notrade' ? notradeKind(A) : null, out: outcome(R, i, A) })
  }
  return out
}
// assign classes to rounds: scarce class first, rounds with the fewest options first
function assign(ids, quota, lo, hi, salt) {
  const cand = new Map(ids.map(id => [id, candidates(id, lo, hi)])), left = new Set(ids), picked = [], short = []
  for (const cls of Object.keys(quota).sort((a, b) => quota[b] - quota[a]).sort((a, b) => (a === 'trade' ? -1 : b === 'trade' ? 1 : 0))) {
    let need = quota[cls], kinds = {}
    const order = [...left].filter(id => cand.get(id)[cls].length).sort((a, b) => cand.get(a)[cls].length - cand.get(b)[cls].length)
    for (const id of order) {
      if (!need) break
      let list = cand.get(id)[cls]
      if (cls === 'notrade') { const rare = ['down', 'chase', 'rr'].sort((a, b) => (kinds[a] || 0) - (kinds[b] || 0)).find(k => list.some(x => x.kind === k)); list = list.filter(x => x.kind === rare) }
      const c = list[Math.floor(E.hashUnit(salt + '|' + id + '|' + cls) * list.length)]
      if (c.kind) kinds[c.kind] = (kinds[c.kind] || 0) + 1
      picked.push({ r: id, i: c.i, cls, kind: c.kind || undefined }); left.delete(id); need--
    }
    if (need) short.push(cls + ' short by ' + need)
  }
  // scarce class could not be filled (the rubric rarely says "trade"): fill the rest with wait / no-trade alternately
  let alt = 0
  for (const id of [...left]) {
    if (picked.length >= Object.values(quota).reduce((a, b) => a + b, 0)) break
    const cls = ['wait', 'notrade'][alt % 2], list = cand.get(id)[cls].length ? cand.get(id)[cls] : cand.get(id)[cls === 'wait' ? 'notrade' : 'wait']
    if (!list.length) continue
    const c = list[Math.floor(E.hashUnit(salt + '|fill|' + id) * list.length)]
    picked.push({ r: id, i: c.i, cls: c.kind ? 'notrade' : (list === cand.get(id).wait ? 'wait' : 'notrade'), kind: c.kind || undefined }); left.delete(id); alt++
  }
  const rnd = E.mulberry32(E.hash32('ORDER|' + salt))
  for (let k = picked.length - 1; k > 0; k--) { const m = Math.floor(rnd() * (k + 1)); [picked[k], picked[m]] = [picked[m], picked[k]] }
  return { picked, short }
}

const mixed = assign(EXAM, { trade: 20, wait: 40, notrade: 40 }, LO, HI, 'MIXED')
const pre = assign(TESTS, { trade: 2, wait: 4, notrade: 4 }, LO, 200, 'PRE')
const post = assign(TESTS, { trade: 2, wait: 4, notrade: 4 }, 270, HI, 'POST')

// decision drills + historical base rates on the training rounds
const drills = { trade: [], wait: [], notrade: [] }, base = {}
for (const id of TRAIN) {
  const c = candidates(id, LO, HI)
  for (const cls of ['trade', 'wait', 'notrade']) {
    const b = base[cls] || (base[cls] = { n: 0, target: 0, stop: 0, open: 0, ret: 0 })
    c[cls].forEach(x => { b.n++; b[x.out.res]++; b.ret += x.out.ret })
    if (c[cls].length) { const x = c[cls][Math.floor(E.hashUnit('DRILL|' + id + '|' + cls) * c[cls].length)]; drills[cls].push({ r: id, i: x.i, cls, kind: x.kind || undefined }) }
  }
}
for (const k in base) { const b = base[k]; b.targetRate = +(b.target / b.n).toFixed(3); b.stopRate = +(b.stop / b.n).toFixed(3); b.avgRet = +(b.ret / b.n).toFixed(4); delete b.ret }
const notes = [...mixed.short.map(s => 'mixed ' + s), ...pre.short.map(s => 'pre ' + s), ...post.short.map(s => 'post ' + s)]
fs.writeFileSync('build/cases.json', JSON.stringify({ version: 1, window: [LO, HI], mixed: mixed.picked, pre: pre.picked, post: post.picked, finalPool: FINAL, testPool: TESTS, drills, base, notes }))
const count = l => l.reduce((o, x) => (o[x.cls] = (o[x.cls] || 0) + 1, o), {})
console.log('CASES mixed', JSON.stringify(count(mixed.picked)), 'pre', JSON.stringify(count(pre.picked)), 'post', JSON.stringify(count(post.picked)),
  'drills', Object.fromEntries(Object.entries(drills).map(([k, v]) => [k, v.length])), 'base', JSON.stringify(base), notes.length ? 'NOTES ' + notes.join('; ') : '')
