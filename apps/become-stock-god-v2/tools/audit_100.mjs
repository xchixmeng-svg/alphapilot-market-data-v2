// 100-lesson acceptance audit. Drives the built page in Chromium and checks the curriculum contract:
// counts and display numbers, old-save migration (make-up lessons, Continue), module/total progress,
// every lesson enterable and completable through the real UI actions, navigation, persistence,
// phone layouts, case-data causality (no future leakage), feedback/takeaway on every case question,
// and graduation only after all 100. Writes audit/audit_100.json; exits non-zero on any failure.
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const url = process.env.BSG_AUDIT_URL || process.env.BSG_URL || 'http://127.0.0.1:4173/'
const out = path.resolve('audit'); fs.mkdirSync(path.join(out, 'a100'), { recursive: true })
const fails = [], report = {}
const check = (ok, msg) => { if (!ok) fails.push(msg); return ok }
const baseSave = over => Object.assign({ version: 3, created: new Date().toISOString(), completed: [], xp: 0, coins: 0, streak: 0, premium: true, missionStats: {}, practice: { idx: 0, correct: 0, total: 0, round: 1 }, exam: { idx: 0, correct: 0, total: 0 }, lastMission: 1, seed: 'audit-seed' }, over || {})

const browser = await chromium.launch({ headless: true })
async function open(view, save, openAll) {
  const ctx = await browser.newContext({ viewport: view, isMobile: true, hasTouch: true, deviceScaleFactor: 2 })
  if (openAll) await ctx.addInitScript(() => { window.BSG_OPEN_ALL = true })
  const p = await ctx.newPage(), errs = []
  p.on('pageerror', e => errs.push(String(e))); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()) })
  await p.goto(url, { waitUntil: 'load' })
  await p.evaluate(s => { if (s) localStorage.setItem('bsg_v3_product', JSON.stringify(s)); else localStorage.removeItem('bsg_v3_product'); sessionStorage.setItem('bsg_landing_entered', '1') }, save)
  await p.reload({ waitUntil: 'load' }); await p.waitForSelector('.hudbar', { timeout: 15000 })
  return { p, ctx, errs }
}
const act = (p, a, data) => p.evaluate(([a, data]) => { const b = document.createElement('button'); b.dataset.act = a; Object.assign(b.dataset, data || {}); document.body.appendChild(b); b.click(); b.remove() }, [a, data])
const state = p => p.evaluate(() => window.BSGV3.state())

try {
  // 1–4: counts, displayNo, permanent ids
  { const { p, ctx, errs } = await open({ width: 390, height: 844 }, baseSave())
    const r = await p.evaluate(() => { const CT = window.BSG_CONTENT, o = window.BSG100.order(); return { n: CT.MISSIONS.length, order: o, disp: o.map(id => CT.displayNo(id)), ids: CT.MISSIONS.map(m => m.id), mods: CT.WORLDS.length, titles: CT.MISSIONS.map(m => m.title), modsOf: CT.MISSIONS.map(m => m.moduleId) } })
    check(r.n === 100, '1 course total ' + r.n)
    check(r.disp.every((d, k) => d === k + 1), '2 displayNo must run 01..100 in order')
    check(r.ids.every((id, k) => id === k + 1) && r.ids.slice(0, 63).every(x => x <= 63), '3 permanent ids 1..63 kept')
    check(new Set(r.ids.slice(63)).size === 37 && r.ids.slice(63).every(x => x >= 64 && x <= 100), '4 new ids 64..100 unique')
    check(r.mods === 14, 'module count ' + r.mods)
    check(r.titles[49] === '逐日 K 線技術面實戰', 'lesson id 50 renamed')
    report.course = { lessons: r.n, modules: r.mods, firstFive: r.order.slice(0, 8), last: r.order.slice(-3) }
    check(!errs.length, 'console errors (course): ' + errs.join(' | ')); await ctx.close() }

  // 5–11: migration of an old 63-lesson player (completed by the old teaching order up to old M36)
  { const old = [1,2,3,4,5,51,6,7,8,9,10,11,52,12,13,14,15,53,16,17,18,54,55,19,20,21,22,23,24,25,26,56,27,28,57,29,30,31,32,33,58,34,35,36]
    const { p, ctx, errs } = await open({ width: 390, height: 844 }, baseSave({ completed: old.slice().sort((a, b) => a - b), lastMission: 37, xp: 5000 }))
    const s = await state(p)
    check(old.every(id => s.completed.includes(id)), '5 old completions kept')
    check(s.curriculumVersion === 2, 'curriculumVersion written')
    const r = await p.evaluate(() => { const B = window.BSG100, CT = window.BSG_CONTENT; return { next: B.next(), makeup: B.makeup(), frontier: B.frontier(), nextDisp: CT.displayNo(B.next()) } })
    // old player finished up to id 36 (displayNo 47). 37 and 38 moved earlier (09 and 07) -> make-up, not a reset.
    check(r.makeup.includes(37) && r.makeup.includes(38), '補課 must list moved lessons 37/38: ' + r.makeup)
    check(r.nextDisp > 40, '6 Continue must not drag the player back: next displayNo ' + r.nextDisp)
    await act(p, 'continueNext'); await p.waitForTimeout(200)
    const cur = await p.evaluate(() => window.BSGV3._test.ui.mission.id)
    check(cur === r.next, '6 Continue opens the next lesson ' + cur + ' vs ' + r.next)
    await act(p, 'overview'); await p.waitForTimeout(250)
    const ov = await p.evaluate(() => { const mods = [...document.querySelectorAll('.ckModule')].map(m => ({ id: m.id, prog: m.querySelector('.ckModProg')?.textContent, tiles: m.querySelectorAll('.ckTile').length, done: m.querySelectorAll('.ckTile.done').length })); return { mods, top: document.querySelector('.ckOvTop .ckRank')?.textContent, nums: [...document.querySelectorAll('.ckTileNum')].map(x => x.textContent), makeupTiles: document.querySelectorAll('.ckMakeup').length } })
    check(ov.mods.length === 14 && ov.mods.reduce((a, m) => a + m.tiles, 0) === 100, '7 overview lists 14 modules / 100 tiles')
    check(ov.mods.every(m => m.prog === m.done + ' / ' + m.tiles), '7/8 module progress matches done tiles ' + JSON.stringify(ov.mods.filter(m => m.prog !== m.done + ' / ' + m.tiles)))
    check(ov.top && ov.top.indexOf(old.length + ' / 100') >= 0, '9 total progress X / 100: ' + ov.top)
    check(ov.nums.length === 100 && ov.nums.every((x, k) => x === String(k + 1).padStart(2, '0') || x === '✓'), '2 tiles show displayNo only')
    check(ov.makeupTiles >= 2, 'make-up tiles shown')
    const doneTiles = ov.mods.reduce((a, m) => a + m.done, 0)
    check(doneTiles === old.length, '10/11 completed tiles = completed lessons (' + doneTiles + ')')
    await p.screenshot({ path: path.join(out, 'a100', 'migrated-overview.png') })
    check(!errs.length, 'console errors (migration): ' + errs.join(' | ')); await ctx.close() }

  // 12–16, 24–28: play all 100 lessons through the real actions, in teaching order, on a phone held upright
  { const { p, ctx, errs } = await open({ width: 390, height: 844 }, baseSave())
    const order = await p.evaluate(() => window.BSG100.order())
    const played = [], caseQ = { questions: 0, withWhy: 0 }
    for (const id of order) {
      const unlocked = await p.evaluate(id => window.BSG100.unlocked(id), id)
      if (!check(unlocked || id === 100, '12 lesson ' + id + ' should be unlocked in order')) break
      await act(p, 'mission', { id: String(id) }); await p.waitForTimeout(60)
      const info = await p.evaluate(() => { const m = window.BSGV3._test.ui.mission; return { id: m.id, type: m.type, title: m.title, head: document.querySelector('.trainingHead strong')?.textContent, tag: document.querySelector('.missionTag')?.textContent } })
      if (!check(info.id === id, '12 lesson ' + id + ' did not open (got ' + info.id + ')')) break
      check(info.head === info.title, '25 lesson title shown for ' + id)
      check(info.tag && info.tag.replace(/\D/g, '') === String(await p.evaluate(id => window.BSG_CONTENT.displayNo(id), id)).padStart(2, '0'), 'displayNo shown in lesson ' + id + ': ' + info.tag)
      if (info.type === 'concept') {
        const res = await p.evaluate(() => {
          const ACT = null, B = window.BSG100, out = { steps: 0, why: 0, takeaway: false }
          const click = (a, d) => { const b = document.createElement('button'); b.dataset.act = a; Object.assign(b.dataset, d || {}); document.body.appendChild(b); b.click(); b.remove() }
          let P = B.player(); if (!P) return { err: 'no player' }
          click('cpNext'); P = B.player()
          for (let guard = 0; guard < 80 && P.step < P.steps.length; guard++) {
            const st = P.steps[P.step], q = st.q
            if (q.info) { click('cpNext'); P = B.player(); continue }
            out.steps++; if (q.why || q.grade) out.why++
            let pick = q.correct
            if (q.grade) { pick = 0; for (let i = 0; i < q.options.length; i++) { const g = q.grade(i, P.cases[st.ci], P) || {}; const keys = Object.keys(g).filter(k => k[0] !== '_' && k !== 'tag'); if (keys.every(k => g[k] > 0)) { pick = i; break } } }
            if (!(pick >= 0 && pick < q.options.length)) return { err: 'answer index out of range at step ' + P.step }
            if (new Set(q.options).size !== q.options.length) return { err: 'duplicate options at step ' + P.step }
            click('cpAnswer', { i: String(pick) }); click('cpAfter'); P = B.player()
          }
          out.takeaway = !!document.querySelector('.prompt .question')?.textContent
          out.finishText = document.querySelector('[data-act="cpFinish"]')?.textContent
          click('cpFinish'); return out })
        if (!check(!res.err, 'lesson ' + id + ': ' + res.err)) break
        caseQ.questions += res.steps; caseQ.withWhy += res.why
        check(res.steps > 0, 'case lesson ' + id + ' has no questions')
        check(res.takeaway, '27 takeaway shown for ' + id)
      } else {
        const res = await p.evaluate(() => {
          const t = window.BSGV3._test, ui = t.ui, click = (a, d) => { const b = document.createElement('button'); b.dataset.act = a; Object.assign(b.dataset, d || {}); document.body.appendChild(b); b.click(); b.remove() }
          click('beginQuestions')
          for (let guard = 0; guard < 12 && window.BSGV3._test.ui.screen === 'training'; guard++) { const R = t.getR(ui.roundId), q = t.question(ui.mission, R, ui.exercise); click('answer', { i: String(q.correct) }); click('afterFeedback') }
          return { screen: ui.screen } })
        check(res.screen === 'mastery', '13 chart lesson ' + id + ' did not finish (screen ' + res.screen + ')')
      }
      await p.waitForTimeout(40)
      const s = await state(p)
      if (!check(s.completed.includes(id), '13 lesson ' + id + ' not recorded as completed')) break
      played.push(id)
      if (id !== 100) {
        const nx = await p.evaluate(() => window.BSG100.next())
        const want = order[order.indexOf(id) + 1]
        check(nx === want, '14 Next after ' + id + ' should be ' + want + ', got ' + nx)
        const grad = await p.evaluate(() => window.BSG100.graduated())
        check(!grad, '28 graduated too early at ' + id)
        if (id === 50) { const lock = await p.evaluate(() => window.BSG100.unlocked(100)); check(!lock, '100 must stay locked until the other 99 are done') }
      }
    }
    report.play = { played: played.length, caseQuestions: caseQ }
    check(played.length === 100, '12/13 all 100 lessons played (' + played.length + ')')
    check(caseQ.withWhy === caseQ.questions, '26 every case question has feedback')
    const end = await p.evaluate(() => ({ next: window.BSG100.next(), grad: window.BSG100.graduated(), h2: document.querySelector('.mastery h2')?.textContent }))
    check(end.grad, '28 graduation after 100'); check(end.next === 100, '15 no lesson 101 after the last one (next=' + end.next + ')')
    check(/畢業/.test(end.h2 || ''), '28 graduation shown: ' + end.h2)
    await p.reload({ waitUntil: 'load' }); await p.waitForSelector('.hudbar')
    const s2 = await state(p); check(s2.completed.length === 100, '16 progress kept after reload: ' + s2.completed.length)
    check(!errs.length, 'console errors (play-through): ' + errs.slice(0, 5).join(' | ')); await ctx.close() }

  // 17–21: layouts. Portrait list, landscape chart, long titles inside cards, nav never hides the last tile
  for (const [w, h, tag] of [[375, 667, 'portrait-SE'], [390, 844, 'portrait'], [844, 390, 'landscape']]) {
    const { p, ctx, errs } = await open({ width: w, height: h }, baseSave({ completed: Array.from({ length: 70 }, (_, i) => i + 1) }), true)
    await act(p, 'overview'); await p.waitForTimeout(300)
    const r = await p.evaluate(() => { const box = document.querySelector('#app>main'); const tiles = [...document.querySelectorAll('.ckTile')]; const over = tiles.filter(t => t.scrollWidth > t.clientWidth + 2 || t.getBoundingClientRect().right > innerWidth + 1).map(t => t.dataset.mid)
      if (box) box.scrollTop = box.scrollHeight; const last = tiles[tiles.length - 1].getBoundingClientRect(), nav = document.querySelector('.ckNav')?.getBoundingClientRect()
      return { over, lastBottom: last.bottom, navTop: nav ? nav.top : innerHeight, docScroll: document.documentElement.scrollHeight - innerHeight } })
    check(!r.over.length, '17/20 tiles clipped at ' + tag + ': ' + r.over.slice(0, 5))
    check(r.lastBottom <= r.navTop + 1, '21 bottom nav covers the last lesson at ' + tag)
    if (tag !== 'landscape') check(r.docScroll <= 1, 'page scrolls at ' + tag)
    await p.screenshot({ path: path.join(out, 'a100', tag + '-overview.png') })
    for (const id of [71, 98]) {
      await act(p, 'mission', { id: String(id) }); await p.waitForTimeout(150); await act(p, 'cpNext'); await p.waitForTimeout(150)
      const l = await p.evaluate(() => { const svg = document.querySelector('.cpScene svg'), b = svg && svg.getBoundingClientRect(), dock = document.querySelector('.lessonDock').getBoundingClientRect(); return { svg: b && { w: b.width, h: b.height, r: b.right }, dockBottom: dock.bottom, ih: innerHeight, iw: innerWidth, sw: document.documentElement.scrollWidth } })
      check(l.svg && l.svg.w > 100 && l.svg.h > 60 && l.svg.r <= l.iw + 1, '18 chart not visible/clipped in lesson ' + id + ' at ' + tag + ' ' + JSON.stringify(l.svg))
      check(l.dockBottom <= l.ih + 1 && l.sw <= l.iw + 1, '19 lesson dock off screen at ' + tag)
      await p.screenshot({ path: path.join(out, 'a100', tag + '-L' + id + '.png') })
      await act(p, 'overview'); await p.waitForTimeout(150)
    }
    check(!errs.length, 'console errors (' + tag + '): ' + errs.join(' | ')); await ctx.close()
  }

  // 22–24: case data causality — nothing visible after the decision date; reveal kept separate
  { const { p, ctx } = await open({ width: 390, height: 844 }, baseSave())
    const r = await p.evaluate(() => { const B = window.BSG100, bad = [], counts = {}
      const lastD = c => c && c.d && c.d.length ? c.d[c.d.length - 1] : null
      for (let id = 64; id <= 100; id++) { const data = B.cases(id); counts[id] = data.length
        data.forEach((x, k) => { const tag = id + '#' + k
          if (x.asof == null) return
          for (const key of ['line', 'stock', 'market', 'sector', 'index', 'share']) if (Array.isArray(x[key]) && x.d && x[key].length !== x.d.length) bad.push(tag + ' ' + key + ' length')
          if (x.d && x.d[x.d.length - 1] > x.asof) bad.push(tag + ' series runs past asof')
          if (x.stock && x.stock.d && lastD(x.stock) > x.asof) bad.push(tag + ' stock candles past asof')
          if (x.chart && x.chart.d && [70, 68, 79, 82].indexOf(id) < 0 && lastD(x.chart) > x.asof) bad.push(tag + ' chart past asof')
          if (x.news && x.news.time && [79].indexOf(id) < 0 && Number(String(x.news.time).slice(0, 10).replace(/-/g, '')) > x.asof) bad.push(tag + ' news after asof')
          ;(x.news && Array.isArray(x.news) ? x.news : []).forEach(n => { if (Number(String(n.time).slice(0, 10).replace(/-/g, '')) > x.asof) bad.push(tag + ' news after asof') })
          ;(x.fundQ || []).forEach(r => { if (r.avail > x.asof) bad.push(tag + ' quarter not yet public') })
          ;(x.fundM || []).forEach(r => { if (r.avail > x.asof) bad.push(tag + ' revenue not yet public') })
          ;(x.rows || []).forEach(r => { if (r.avail && r.avail > x.asof && !r.reveal) bad.push(tag + ' row not yet public ' + (r.q || r.ym)) })
          if (id === 70 && Math.abs(x.before - x.dividend - x.reference) > 0.06 && x.kind === '息') bad.push(tag + ' ex-dividend reference mismatch')
        }) }
      return { bad, counts } })
    report.cases = r.counts
    check(!r.bad.length, '22/23 causality: ' + r.bad.slice(0, 10).join('; '))
    await ctx.close() }

  // smoke: home -> overview -> lesson -> back -> continue -> reload
  { const { p, ctx, errs } = await open({ width: 390, height: 844 }, baseSave({ completed: [1, 2, 3] }))
    await p.locator('[data-act="overview"]').first().tap(); await p.waitForSelector('.ckOverview')
    await p.locator('.ckTile.next[data-act="mission"]').first().tap(); await p.waitForSelector('.trainingHead')
    await p.locator('.hudbar .iconbtn').first().tap(); await p.waitForTimeout(200)
    await p.locator('[data-act="continueNext"]').first().tap(); await p.waitForSelector('.trainingHead')
    const id = await p.evaluate(() => window.BSGV3._test.ui.mission.id)
    await p.reload({ waitUntil: 'load' }); await p.waitForSelector('.hudbar')
    const s = await state(p)
    check(id === 4 && s.completed.length === 3, 'smoke: continue/reload ' + id + ' ' + s.completed.length)
    check(!errs.length, 'console errors (smoke): ' + errs.join(' | ')); await ctx.close() }
} catch (e) { fails.push('crash: ' + (e && e.stack || e)) }
await browser.close()
fs.writeFileSync(path.join(out, 'audit_100.json'), JSON.stringify({ generatedAt: new Date().toISOString(), ok: !fails.length, fails, report }, null, 2))
if (fails.length) { console.error('AUDIT_100_FAIL ' + fails.length + '\n' + fails.slice(0, 40).join('\n')); process.exitCode = 1 }
else console.log('AUDIT_100_OK', JSON.stringify(report))
