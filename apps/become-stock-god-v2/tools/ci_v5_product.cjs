// Behaviour tests for the whole learning product. Drives the real built page (dist/) in Chromium with touch:
// core course, training ground (do-it-yourself drills + review), blind exam (balanced answers, no leak, outcome
// reveal), daily scenarios, final exam, start test / retest, and the open-all review build.
// Usage: serve dist/ on BSG_URL (default http://127.0.0.1:4175/) and run `node tools/ci_v5_product.cjs`.
const { chromium } = require('playwright')
const APP_URL = process.env.BSG_URL || 'http://127.0.0.1:4175/'
const ALL = Array.from({ length: 63 }, (_, i) => i + 1)
const FMT = 'mixed-v6'
function save(over) { return Object.assign({ version: 3, created: new Date().toISOString(), completed: [], xp: 0, coins: 0, streak: 0, premium: true, missionStats: {}, practice: { idx: 0, correct: 0, total: 0, round: 1 }, exam: { idx: 0, correct: 0, total: 0, format: FMT }, lastMission: 1, seed: 'ci-seed' }, over || {}) }
const LAND = { width: 844, height: 390 }, ANDROID = { width: 412, height: 915 }
const ANDROID_UA = 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Mobile Safari/537.36'
const assert = (c, m) => { if (!c) throw new Error(m) }

;(async () => {
  const browser = await chromium.launch({ headless: true })
  const report = {}
  async function page(view, s, ua, openAll) {
    const ctx = await browser.newContext({ viewport: view, isMobile: true, hasTouch: true, deviceScaleFactor: 2, userAgent: ua })
    if (openAll) await ctx.addInitScript(() => { window.BSG_OPEN_ALL = true })
    const p = await ctx.newPage(), errs = []
    p.on('pageerror', e => errs.push('pageerror ' + e)); p.on('console', m => { if (m.type() === 'error') errs.push('console ' + m.text()) })
    await p.goto(APP_URL, { waitUntil: 'load' })
    await p.evaluate(s => { localStorage.setItem('bsg_v3_product', JSON.stringify(s)); sessionStorage.setItem('bsg_landing_entered', '1') }, s)
    await p.reload({ waitUntil: 'load' }); await p.waitForSelector('.hudstats', { timeout: 15000 })
    return { p, ctx, errs }
  }
  const act = (p, a) => p.evaluate(a => { const b = document.createElement('button'); b.dataset.act = a; document.body.appendChild(b); b.click(); b.remove() }, a)
  const setSave = (p, fn) => p.evaluate(src => { const s = JSON.parse(localStorage.getItem('bsg_v3_product')); new Function('s', src)(s); localStorage.setItem('bsg_v3_product', JSON.stringify(s)) }, fn)
  const reload = async p => { await p.reload(); await p.waitForSelector('.hudstats') }
  // answer the current V5 step correctly: choices by tapping, picks by setting the reference answer (real touch picks
  // are tested separately), info steps are just read
  async function stepCorrect(p, flowKind) {
    const f = await p.evaluate(k => { const f = window.BSG_V5.flow(); if (!f || (k && f.kind !== k)) return null; const st = f.steps[f.k]; return { kind: st.kind, c: st.correct, id: st.id } }, flowKind)
    if (!f) return null
    if (f.kind === 'choice') await p.locator('[data-act="v5Choice"][data-i="' + f.c + '"]').tap()
    else if (f.kind === 'price' || f.kind === 'candle') { await p.evaluate(() => { const f = window.BSG_V5.flow(), st = f.steps[f.k]; st.pick = st.target }); await p.locator('[data-act="v5ConfirmPick"]').tap() }
    await p.locator('[data-act="v5Next"]').tap()
    return f
  }

  // 1. Core course: 63 lessons, teaching order, unlocks follow the order, new lessons valid on many charts
  { const { p, ctx, errs } = await page(LAND, save())
    const r = await p.evaluate(() => { const t = window.BSGV3._test, CT = window.BSG_CONTENT, V = window.BSG_V5, order = V.order()
      if (CT.MISSIONS.length !== 63 || order.length !== 63 || new Set(order).size !== 63) throw new Error('course map')
      if (CT.FREE_THROUGH !== 15 || CT.PRODUCT_PRICE_TWD !== 149) throw new Error('commercial defaults changed')
      if (window.BSG_V6.openAll) throw new Error('production build must not be open-all')
      const pos = id => order.indexOf(id)
      for (const [id, after] of [[51, 1], [52, 11], [53, 15], [54, 18], [55, 18], [56, 26], [57, 28], [58, 33], [59, 41], [60, 43], [61, 44], [62, 48], [63, 49]]) if (!(pos(id) > pos(after))) throw new Error('M' + id + ' taught before M' + after)
      if (pos(50) !== 62) throw new Error('daily-reveal lesson must be the last core lesson')
      let checked = 0; for (let id = 51; id <= 63; id++) { const m = CT.MISSIONS[id - 1], suit = t.TRAIN.filter(r => t.suitable(id, t.getR(r))); if (suit.length < 4) throw new Error('M' + id + ' has too few historical examples: ' + suit.length)
        for (const rid of suit.slice(0, 40)) { t.ui.roundId = rid; t.ui.mission = m; const R = t.getR(rid); for (let ex = 1; ex <= 3; ex++) { const q = t.question(m, R, ex); if (!q || q.opts.length < 3 || new Set(q.opts).size !== q.opts.length || q.correct < 0 || q.correct >= q.opts.length || !t.qtext(m, R, ex)) throw new Error('M' + id + ' ex' + ex + ' invalid on ' + rid); checked++ } } }
      return { lessons: CT.MISSIONS.length, checked } })
    await setSave(p, 's.completed=[1,2,3,4,5]'); await reload(p)
    const nx = await p.evaluate(() => { const b = document.createElement('button'); b.dataset.act = 'continueNext'; document.body.appendChild(b); b.click(); return window.BSGV3._test.ui.mission.id })
    assert(nx === 51, 'next lesson after M05 should be M51, got ' + nx)
    await act(p, 'home'); await p.waitForTimeout(150)
    const home = await p.locator('.ckStats').innerText(); assert(/\/63/.test(home) && !/\/36/.test(home), 'home progress must count 63 lessons: ' + home)
    assert(!errs.length, errs.join(' | ')); report.core = r; await ctx.close() }

  // 2. Training ground: unlock by progress, do-it-yourself drills, 60 questions played, review queue
  { const { p, ctx, errs } = await page(LAND, save())
    assert(await p.evaluate(() => window.BSG_V5.drills.available().length) === 0, 'drills available before any lesson')
    const pools = {}
    for (const n of [6, 20, 40, 63]) {
      await setSave(p, 's.completed=' + JSON.stringify((await p.evaluate(() => window.BSG_V5.order())).slice(0, n))); await reload(p)
      pools[n] = await p.evaluate(n => { const V = window.BSG_V5, taught = new Set(V.order().slice(0, n)), av = V.drills.available(); const bad = av.filter(d => !taught.has(d.lesson)).map(d => d.id); if (bad.length) throw new Error('untaught drills in pool: ' + bad.join(',')); const fam = {}, kinds = {}; av.forEach(d => { fam[d.family] = (fam[d.family] || 0) + 1; const k = d.gen && d.gen.kind || (d.gen ? 'choice' : 'lesson'); kinds[k] = (kinds[k] || 0) + 1 }); return { n: av.length, fam, kinds } }, n) }
    assert(pools[6].n < pools[20].n && pools[20].n < pools[40].n && pools[40].n < pools[63].n, 'pool must grow with progress')
    assert(!(pools[20].fam.rsi || pools[20].fam.macd || pools[20].fam.kd), 'indicator drills before indicators were taught')
    for (const f of ['structure', 'levels', 'breakout', 'reversal', 'volume', 'ma', 'macd', 'kd', 'rsi', 'conflict', 'entry', 'notrade']) assert(pools[63].fam[f] >= 9, 'family ' + f + ' has only ' + pools[63].fam[f] + ' drill types')
    assert(pools[63].kinds.price >= 4 && pools[63].kinds.candle >= 2 && pools[63].kinds.decide === 1, 'do-it-yourself drills missing ' + JSON.stringify(pools[63].kinds))
    await act(p, 'startPractice'); await p.waitForTimeout(200)
    const seen = {}; let les = 0
    while ((await p.evaluate(() => window.BSGV3.state().practice.total)) < 60) {
      const isFlow = await p.evaluate(() => { const f = window.BSG_V5.flow(); return !!(f && f.kind === 'practice') })
      if (isFlow) { const f = await stepCorrect(p, 'practice'); seen[f.kind] = (seen[f.kind] || 0) + 1 }
      else { les++; const c = await p.evaluate(() => { const T = window.BSGV3._test, u = T.ui; return T.question(u.mission, T.getR(u.roundId), u.exercise).correct }); await p.locator('[data-act="answer"][data-i="' + c + '"]').click(); await p.locator('[data-act="afterFeedback"]').click() } }
    const st = await p.evaluate(() => ({ pr: window.BSGV3.state().practice, fams: Object.keys(window.BSG_V5.state().skills).length, rv: window.BSG_V6.review.count() }))
    assert(st.pr.total === 60 && st.pr.correct === 60 && les >= 5 && (seen.price || 0) + (seen.candle || 0) + (seen.info || 0) >= 15 && st.fams >= 8 && st.rv === 0, 'training ground play ' + JSON.stringify({ st, seen, les }))
    // a real finger tap on the chart draws the line in a drawing drill
    let tapped = false
    for (let n = 0; n < 80 && !tapped; n++) {
      const k = await p.evaluate(() => { const f = window.BSG_V5.flow(); return f && f.kind === 'practice' ? f.steps[f.k].kind : 'lesson' })
      if (k === 'price') { const box = await p.locator('#v3Chart canvas').boundingBox(); await p.touchscreen.tap(box.x + box.width * .45, box.y + box.height * .4); const pk = await p.evaluate(() => { const f = window.BSG_V5.flow(); return f.steps[f.k].pick }); assert(pk != null, 'touch did not draw a line'); tapped = true; await p.locator('[data-act="v5ConfirmPick"]').tap()
        const fb = await p.locator('.feedbackBox').innerText(); assert(/參考答案/.test(fb), 'reference answer not shown after drawing'); await p.locator('[data-act="v5Next"]').tap() }
      else if (k === 'lesson') { const c = await p.evaluate(() => { const T = window.BSGV3._test, u = T.ui; return T.question(u.mission, T.getR(u.roundId), u.exercise).correct }); await p.locator('[data-act="answer"][data-i="' + c + '"]').click(); await p.locator('[data-act="afterFeedback"]').click() }
      else await stepCorrect(p, 'practice') }
    assert(tapped, 'no drawing drill reached')
    // review: a missed skill comes back on a new chart after 5 questions, and leaves the queue after 3 correct reviews
    const rv = await p.evaluate(() => { const R = window.BSG_V6.review, t = window.BSGV3.state().practice.total; R.miss('f:levels'); const s = R.state()['f:levels']; return { due: s.due, t } })
    assert(rv.due === rv.t + 5, 'review gap ' + JSON.stringify(rv))
    let reviewSeen = 0
    for (let n = 0; n < 120 && (await p.evaluate(() => !!window.BSG_V6.review.state()['f:levels'])); n++) {
      const info = await p.evaluate(() => { const f = window.BSG_V5.flow(); if (f && f.kind === 'practice') { const st = f.steps[f.k]; return { flow: 1, head: st.head || '' } } return { flow: 0 } })
      if (info.flow) { if (/複習/.test(info.head)) reviewSeen++; await stepCorrect(p, 'practice') }
      else { const c = await p.evaluate(() => { const T = window.BSGV3._test, u = T.ui; return T.question(u.mission, T.getR(u.roundId), u.exercise).correct }); await p.locator('[data-act="answer"][data-i="' + c + '"]').click(); await p.locator('[data-act="afterFeedback"]').click() } }
    const left = await p.evaluate(() => window.BSG_V6.review.state()['f:levels'])
    assert(!left && reviewSeen >= 3, 'missed skill must come back 3 times and then clear ' + JSON.stringify({ left, reviewSeen }))
    assert(!errs.length, errs.join(' | ')); report.drills = { pools: Object.fromEntries(Object.entries(pools).map(([k, v]) => [k, v.n])), families: pools[63].fam, kinds: pools[63].kinds, played: { seen, les }, reviewSeen }; await ctx.close() }

  // 3. Blind exam: 100 distinct unseen charts, balanced answers, game rubric = build tool, no leak before the reveal
  { const { p, ctx, errs } = await page(ANDROID, save({ completed: ALL }), ANDROID_UA)
    const r = await p.evaluate(() => { const V = window.BSG_V5, V6 = window.BSG_V6, t = window.BSGV3._test, cases = V.mixed.cases, dec = { trade: 0, wait: 0, notrade: 0 }, panels = {}, steps = {}, leaks = []
      if (cases.length !== 100) throw new Error('blind exam needs 100 cases')
      const exam = new Set(t.EXAM); if (new Set(cases.map(c => c.r)).size !== 100 || cases.some(c => !exam.has(c.r))) throw new Error('blind exam must use 100 distinct EXAM charts')
      for (let k = 0; k < 100; k++) { const c = cases[k], R = t.getR(c.r), A = V.analyze(R, c.i)
        if (A.decision !== c.cls || V6.clearClass(A) !== c.cls) throw new Error('game rubric disagrees with build tool on case ' + k + ': ' + A.decision + ' vs ' + c.cls)
        dec[c.cls]++; const plan = V.mixed.plan(k, A); panels[plan.panel] = (panels[plan.panel] || 0) + 1; plan.steps.forEach(s => steps[s] = (steps[s] || 0) + 1) }
      return { dec, panels, steps } })
    for (const k of ['MACD', 'KD', 'RSI14', 'VOL']) assert(r.panels[k] >= 20, 'evidence panel ' + k + ' underused')
    const max = Math.max(r.dec.trade, r.dec.wait, r.dec.notrade); assert(max <= 45 && r.dec.trade >= 15, 'answers must be balanced (no single answer above 45%) ' + JSON.stringify(r.dec))
    assert(r.steps.drawSup === 100 && r.steps.drawRes === 100 && r.steps.drawStop === r.dec.trade + r.dec.wait && r.steps.whyNot === r.dec.notrade, 'do-it-yourself steps ' + JSON.stringify(r.steps))
    await p.evaluate(() => window.BSG_V5.start.exam()); await p.waitForTimeout(200)
    const forbidden = /RSI 題|MACD 題|KD 題|突破題|支撐題|這是.{0,6}題/; let realTap = false, reveals = 0
    while ((await p.evaluate(() => window.BSGV3.state().exam.idx)) < 5) {
      const f = await p.evaluate(() => { const f = window.BSG_V5.flow(), st = f.steps[f.k], c = window.BSG_V5.mixed.caseAt(window.BSGV3.state().exam.idx), d = st.chart(st), dec = f.steps.findIndex(s => s.id === 'decision'); return { id: st.id, kind: st.kind, k: f.k, dec, last: f.steps.length - 1, head: document.querySelector('.trainingHead').innerText + document.querySelector('.prompt').innerText, lastIdx: d.lastIdx, len: d.R.c.length, i: c.i } })
      assert(!forbidden.test(f.head), 'mixed exam names the topic: ' + f.head)
      if (f.kind === 'info') { assert(f.k === f.last && f.k > f.dec, 'outcome reveal must come last, after the decision'); assert(f.lastIdx === f.i + 20, 'reveal shows the next 20 days'); reveals++ }
      else assert(f.lastIdx === f.i && f.len === f.i + 1, 'future bars visible before the reveal at step ' + f.id)
      if (f.kind === 'price' && !realTap) { const box = await p.locator('#v3Chart canvas').boundingBox(); await p.touchscreen.tap(box.x + box.width * .4, box.y + box.height * .35); assert(await p.evaluate(() => { const f = window.BSG_V5.flow(); return f.steps[f.k].pick != null }), 'touch did not draw a line in the blind exam'); await p.locator('[data-act="v5ConfirmPick"]').tap(); await p.locator('[data-act="v5Next"]').tap(); realTap = true; continue }
      await stepCorrect(p, 'mixed') }
    const ex = await p.evaluate(() => window.BSGV3.state().exam), oc = await p.evaluate(() => window.BSG_V5.state().outcomes)
    assert(ex.idx === 5 && ex.total >= 35 && reveals === 5 && Object.values(oc).reduce((a, b) => a + b.n, 0) === 5, 'blind exam play ' + JSON.stringify({ ex, reveals, oc }))
    assert(!errs.length, errs.join(' | ')); report.mixed = { decisions: r.dec, panels: r.panels, steps: r.steps, played: ex }; await ctx.close() }

  // 4. Daily scenarios: gating, 72 cases (6 sets × 12 scripts), no future leak at any checkpoint, separate scoring
  { const { p, ctx, errs } = await page(LAND, save({ completed: ALL }))
    assert(!(await p.evaluate(() => window.BSG_V5.dailyUnlocked())), 'daily scenarios open before training/mixed basics')
    await setSave(p, "s.practice={idx:60,correct:50,total:60,round:1};s.exam={idx:20,correct:70,total:100,format:'" + FMT + "'}"); await reload(p)
    const r = await p.evaluate(() => { const D = window.BSG_V5.daily, need = ['uptrend_pullback', 'false_breakout', 'retest_success', 'breakout_fail', 'oversold_bounce', 'true_reversal', 'long_range', 'pretty_indicators', 'lowvol_breakout', 'highvol_breakout', 'good_decision_stopped', 'bad_decision_won']
      if (D.scenarios.length !== 72) throw new Error('expected 72 daily cases, got ' + D.scenarios.length)
      for (const n of need) if (D.scenarios.filter(s => s.script === n).length !== 6) throw new Error('script ' + n + ' must appear once per set')
      if (new Set(D.scenarios.map(s => s.roundId)).size !== 72) throw new Error('daily cases must use different charts')
      const leaks = []; for (const sc of D.scenarios) for (let k = 0; k <= sc.days; k++) { const pr = D.probe(sc.id, k); if (pr.len !== sc.start + k + 1 || pr.lastIdx !== sc.start + k || pr.arrays.some(n => n !== sc.start + k + 1)) leaks.push(sc.id + '@' + k) }
      if (leaks.length) throw new Error('future data leak: ' + leaks.join(','))
      for (const s of D.scenarios) { if (s.script === 'good_decision_stopped' && D.probe(s.id, 0).rubric !== 'trade') throw new Error(s.id + ' good-decision case must be a valid setup at D0'); if (s.script === 'bad_decision_won' && D.probe(s.id, 0).rubric !== 'notrade') throw new Error(s.id + ' bad-decision case must be a no-trade at D0') }
      const exam = new Set(window.BSGV3._test.EXAM); if (D.scenarios.some(s => exam.has(s.roundId))) throw new Error('daily scenario uses a blind-exam chart')
      return { scenarios: D.scenarios.length, checkpoints: D.scenarios.reduce((a, s) => a + s.days + 1, 0) } })
    await act(p, 'dailyOpen'); await p.waitForTimeout(150); assert(await p.locator('.v5SetHead').count() === 6 && await p.locator('.v5Scen').count() === 72, 'daily home must list 6 sets of cases')
    const g = await p.evaluate(() => window.BSG_V5.daily.scenarios.find(s => s.script === 'good_decision_stopped').id)
    await p.evaluate(id => window.BSG_V5.start.daily(id), g); await p.waitForTimeout(150)
    await p.locator('[data-act="dailyStopPick"]').tap(); await p.locator('[data-act="dailyEnter"][data-s="structure"]').tap()
    for (let n = 0; n < 10; n++) { if (await p.locator('[data-act="dailyHome"]').count()) break; if (await p.locator('[data-a="hold"]').count()) await p.locator('[data-a="hold"]').tap(); else await p.locator('[data-a="wait"]').tap() }
    const res = await p.evaluate(id => window.BSG_V5.state().daily[id], g)
    assert(res.done && res.pnl < 0 && res.decision >= 70 && res.log.some(l => l.action === 'stopped'), 'good decision / losing outcome not scored separately ' + JSON.stringify({ d: res.decision, p: res.pnl }))
    assert(!errs.length, errs.join(' | ')); report.daily = Object.assign(r, { goodDecisionStopped: { decision: res.decision, pnl: +(res.pnl * 100).toFixed(2) } }); await ctx.close() }

  // 5. Final exam: unlock after one set of daily cases, unseen pool, mixed interactions, report with 10 dimensions
  { const { p, ctx, errs } = await page(LAND, save({ completed: ALL, practice: { idx: 60, correct: 50, total: 60, round: 1 }, exam: { idx: 20, correct: 70, total: 100, format: FMT } }))
    assert(!(await p.evaluate(() => window.BSG_V5.final.unlocked())), 'final exam open before the daily scenarios')
    await setSave(p, "s.v5={daily:{}};window.BSG_V5.daily.scenarios.slice(0,12).forEach(sc=>s.v5.daily[sc.id]={done:true,decision:80,pnl:.01,log:[],scores:[1],stopScores:[1]})"); await reload(p)
    const pool = await p.evaluate(() => { const t = window.BSGV3._test, F = window.BSG_V5.final.pool, T = window.BSG_V6.cases.testPool; if (!window.BSG_V5.final.unlocked()) throw new Error('final exam must open after one set of daily cases'); if (F.length < 3) throw new Error('final pool too small'); if (F.some(r => t.TRAIN.includes(r) || t.EXAM.includes(r) || T.includes(r))) throw new Error('final exam chart was seen before'); return F.length })
    await p.evaluate(() => window.BSG_V5.start.final()); await p.waitForTimeout(200); const kinds = {}; let n = 0
    for (; n < 40; n++) { const f = await p.evaluate(() => { const f = window.BSG_V5.flow(); if (!f) return null; const st = f.steps[f.k]; return { id: st.id, kind: st.kind, c: st.correct, enter: (st.options || []).findIndex(o => o.label === '進場') } }); if (!f) break; kinds[f.kind] = (kinds[f.kind] || 0) + 1
      if (f.kind === 'choice') await p.locator('[data-act="v5Choice"][data-i="' + (f.id === 'decision' ? f.enter : f.c) + '"]').tap()
      else { const box = await p.locator('#v3Chart canvas').boundingBox(); await p.touchscreen.tap(box.x + box.width * .45, box.y + box.height * .55); assert(await p.evaluate(() => { const f = window.BSG_V5.flow(); return f.steps[f.k].pick != null }), 'touch did not pick on ' + f.id); await p.locator('[data-act="v5ConfirmPick"]').tap() }
      await p.locator('[data-act="v5Next"]').tap() }
    const fin = await p.evaluate(() => window.BSG_V5.state().final)
    assert(fin && fin.done && Object.keys(fin.dims).length === 10 && fin.report && fin.report.habits.length === 5, 'final report incomplete')
    assert(kinds.candle >= 2 && kinds.price >= 3 && kinds.choice >= 8, 'final exam interaction mix ' + JSON.stringify(kinds))
    assert(await p.locator('.v5Report .v5Dim').count() === 10, 'report screen')
    assert(!errs.length, errs.join(' | ')); report.final = { pool, steps: n, kinds, total: fin.total }; await ctx.close() }

  // 6. Start test and retest: unseen matched charts, same answer mix, retest after the final exam, score comparison
  { const { p, ctx, errs } = await page(ANDROID, save(), ANDROID_UA)
    const r = await p.evaluate(() => { const C = window.BSG_V6.cases, t = window.BSGV3._test, V = window.BSG_V5, cnt = l => l.reduce((o, x) => (o[x.cls] = (o[x.cls] || 0) + 1, o), {})
      if (C.pre.length !== 10 || C.post.length !== 10) throw new Error('tests need 10 charts each')
      const all = C.pre.concat(C.post); if (all.some(c => !C.testPool.includes(c.r) || t.TRAIN.includes(c.r) || t.EXAM.includes(c.r) || C.finalPool.includes(c.r))) throw new Error('test chart seen elsewhere')
      const norm = o => JSON.stringify(Object.keys(o).sort().map(k => [k, o[k]])); if (norm(cnt(C.pre)) !== norm(cnt(C.post))) throw new Error('start test and retest must have the same answer mix')
      for (const a of C.pre) for (const b of C.post) if (a.r === b.r && Math.abs(a.i - b.i) < 70) throw new Error('retest window overlaps the start test on ' + a.r)
      for (const c of all) { const A = V.analyze(t.getR(c.r), c.i); if (A.decision !== c.cls) throw new Error('test rubric mismatch on ' + c.r) }
      return { pre: cnt(C.pre), post: cnt(C.post) } })
    await act(p, 'practiceHome'); await p.waitForTimeout(150)
    assert(!(await p.locator('[data-act="startPre"]').isDisabled()) && await p.locator('[data-act="startPost"]').isDisabled(), 'start test open from day one; retest only after the final exam')
    await p.locator('[data-act="startPre"]').tap(); await p.waitForTimeout(150)
    for (let n = 0; n < 60 && (await p.evaluate(() => { const f = window.BSG_V5.flow(); return !!(f && f.kind === 'test') })); n++) await stepCorrect(p, 'test')
    const pre = await p.evaluate(() => window.BSG_V5.state().tests.pre)
    assert(pre.done && pre.score.total === 100 && await p.locator('.v5Cmp').count() >= 5, 'start test result ' + JSON.stringify(pre && pre.score))
    assert(!errs.length, errs.join(' | ')); report.tests = Object.assign(r, { preScore: pre.score.total }); await ctx.close() }

  // 7. Review build: window.BSG_OPEN_ALL opens every lesson and layer (production keeps the gates, checked above)
  { const { p, ctx, errs } = await page(LAND, save({ premium: false }), undefined, true)
    await act(p, 'practiceHome'); await p.waitForTimeout(150)
    const dis = await p.$$eval('.v5Layer button', b => b.filter(x => x.disabled).map(x => x.dataset.act))
    assert(!dis.length, 'open-all build still locks: ' + dis.join(','))
    const lk = await p.evaluate(() => { const t = window.BSG_V6; return window.BSG_V5.order().filter(id => !t.unlocked(id) || t.premiumNeeded(id)) })
    assert(!lk.length, 'open-all build still locks lessons ' + lk.join(','))
    assert(await p.locator('.v5TestMode').count() === 1, 'open-all build must say it is a review build')
    assert(!errs.length, errs.join(' | ')); report.openAll = { layers: 7 }; await ctx.close() }

  // 8. Crosshair on every chart: tap shows the price at the finger (legal tick) and the day; open/high/low/close only
  //    when they cannot give an answer away; drag moves it, tapping the same spot hides it; pick modes keep the tap.
  for (const [view, ua, name] of [[LAND, undefined, 'iphone-landscape'], [ANDROID, ANDROID_UA, 'android']]) {
    const { p, ctx, errs } = await page(view, save({ completed: ALL, practice: { idx: 60, correct: 50, total: 60, round: 1 }, exam: { idx: 20, correct: 70, total: 100, format: FMT } }), ua)
    const cross = () => p.evaluate(() => { const w = document.getElementById('v3Chart'), l = w && w.__bsgLayout; return l && l.cross ? Object.assign({ last: l.candles[l.candles.length - 1].idx }, l.cross) : null })
    const tap = async (fx, fy) => { const box = await p.locator('#v3Chart canvas').boundingBox(); await p.touchscreen.tap(box.x + box.width * fx, box.y + box.height * fy); await p.waitForTimeout(120) }
    const legal = v => p.evaluate(v => Math.abs(window.BSG.roundToTick(v) - v) < 1e-9, v)
    await p.evaluate(() => window.BSG_V5.start.exam()); await p.waitForTimeout(200)
    await tap(.45, .3); let c = await cross()
    assert(c && c.price > 0 && await legal(c.price) && !c.ohlc && !/開 /.test(c.text) && c.i <= c.last, name + ': crosshair before answering ' + JSON.stringify(c))
    await tap(.45, .3); assert(!(await cross()), name + ': tapping the same spot hides the crosshair')
    await tap(.3, .3); const a = await cross(); await tap(.75, .5); const b = await cross(); assert(a && b && b.i > a.i, name + ': tapping elsewhere moves the crosshair')
    const k = await p.evaluate(() => { const f = window.BSG_V5.flow(); return f.steps[f.k].correct }); await p.locator('[data-act="v5Choice"][data-i="' + k + '"]').tap(); await p.waitForTimeout(150)
    await tap(.5, .35); c = await cross(); assert(c && c.ohlc && /開 .*高 .*低 .*收 /.test(c.text), name + ': open/high/low/close after answering ' + JSON.stringify(c))
    await p.locator('[data-act="v5Next"]').tap(); await p.waitForTimeout(150)
    await tap(.5, .4); const pk = await p.evaluate(() => { const f = window.BSG_V5.flow(); return f.steps[f.k].pick })
    assert(pk != null && !(await cross()), name + ': drawing steps keep the tap for the line')
    // lesson quiz: price only until the answer is shown
    await p.evaluate(() => { const b = document.createElement('button'); b.dataset.act = 'continueNext'; document.body.appendChild(b); b.click() }); await p.waitForTimeout(200)
    const mb = await p.$('[data-act=beginQuestions]'); if (mb) { await mb.click(); await p.waitForTimeout(200) }
    const inQuiz = await p.evaluate(() => { const u = window.BSGV3._test.ui; return u.exercise > 0 && !u.feedback && !!document.querySelector('#v3Chart canvas') })
    if (inQuiz) { await tap(.4, .4); c = await cross(); assert(c && !c.ohlc && c.price > 0, name + ': lesson quiz must not list open/high/low/close ' + JSON.stringify(c)) }
    // day-by-day case: only past bars, so open/high/low/close are shown and never beyond today
    const sid = await p.evaluate(() => window.BSG_V5.daily.scenarios[0].id); await p.evaluate(id => window.BSG_V5.start.daily(id), sid); await p.waitForTimeout(200)
    await tap(.95, .4); c = await cross(); const sc = await p.evaluate(id => window.BSG_V5.daily.scenarios.find(s => s.id === id), sid)
    assert(c && c.ohlc && c.i <= sc.start && c.last === sc.start, name + ': daily crosshair ' + JSON.stringify(c))
    assert(!errs.length, errs.join(' | ')); (report.crosshair = report.crosshair || {})[name] = 'ok'; await ctx.close() }

  console.log('V5_PRODUCT_OK', JSON.stringify(report))
  await browser.close()
})().catch(e => { console.error(e); console.log('::error title=V5 behaviour test::' + String(e && e.stack || e).slice(0, 1500).replace(/%/g, '%25').replace(/\r?\n/g, '%0A')); process.exit(1) })
