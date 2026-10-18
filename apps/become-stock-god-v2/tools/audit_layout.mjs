// Layout audit (V3.6 → V4 portrait-first) — geometry assertions for every mission, every exercise, every phone viewport.
// Portrait is the primary layout: the chart keeps real height and the question may scroll below it.
// Reads the renderer's own layout record (wrap.__bsgLayout) instead of judging screenshots, so a
// label that sits next to the wrong candle or a candle hidden under a button fails the build.
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const url = process.env.BSG_AUDIT_URL || 'http://127.0.0.1:4173/'
const outDir = path.resolve('audit')
fs.mkdirSync(outDir, { recursive: true })
const VIEWS = (process.env.BSG_AUDIT_VIEWS || '360x780,375x812,390x844,393x852,430x932,844x390').split(',').map(v => v.split('x').map(Number))
const MISSIONS = (process.env.BSG_AUDIT_MISSIONS || Array.from({ length: 36 }, (_, i) => i + 1).join(',')).split(',').map(Number)
const SHOTS = new Set(['390x844:1', '390x844:4', '390x844:6', '390x844:10', '390x844:18', '390x844:29', '390x844:36', '360x780:6', '430x932:6', '844x390:6'])
const saveAll = { version: 3, created: new Date().toISOString(), completed: Array.from({ length: 36 }, (_, i) => i + 1), xp: 0, coins: 0, streak: 0, premium: true, missionStats: {}, practice: { idx: 0, correct: 0, total: 0 }, exam: { idx: 0, correct: 0, total: 0 }, lastMission: 1 }

const failures = [], rows = []
const ov = (a, b) => a.x0 < b.x1 - .5 && b.x0 < a.x1 - .5 && a.y0 < b.y1 - .5 && b.y0 < a.y1 - .5
const gap = (b, p) => Math.hypot(Math.max(0, b.x0 - p.x, p.x - b.x1), Math.max(0, b.y0 - p.y, p.y - b.y1))
function fail(tag, msg) { failures.push(tag + ' ' + msg) }

const browser = await chromium.launch({ headless: true })
try {
  for (const [w, h] of VIEWS) {
    const view = w + 'x' + h, portrait = h > w
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, isMobile: true, hasTouch: true, deviceScaleFactor: 2, reducedMotion: 'reduce' })
    const page = await ctx.newPage(), errors = []
    page.on('pageerror', e => errors.push('pageerror: ' + String(e)))
    page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()) })
    await page.goto(url, { waitUntil: 'load', timeout: 30000 })
    await page.addStyleTag({ content: '*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}' })
    await page.evaluate(s => localStorage.setItem('bsg_v3_product', JSON.stringify(s)), saveAll)
    await page.reload({ waitUntil: 'load' }); await page.waitForSelector('.hudstats', { timeout: 15000 })
    const enter=page.locator('[data-ui="enter"]')
    if(await enter.count()){
      await enter.click()
      await page.waitForSelector('#ckLanding',{state:'detached',timeout:5000})
    }
    await page.addStyleTag({ content: '*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}' })
    for (const mid of MISSIONS) {
      await page.evaluate(mid => { const b = document.createElement('button'); b.setAttribute('data-act', 'mission'); b.dataset.id = mid; b.hidden = true; document.getElementById('app').appendChild(b); b.click() }, mid)
      await page.waitForSelector('#v3Chart canvas', { timeout: 10000 })
      const seen = new Set()
      for (let ex = 0; ex <= 3; ex++) {
        const tag = `[${view} M${String(mid).padStart(2, '0')} ex${ex}]`
        if (ex === 1) await page.evaluate(() => document.querySelector('[data-act="beginQuestions"]')?.click())
        if (ex >= 2) {
          const ci = await page.evaluate(() => { const T = window.BSGV3._test, u = T.ui; return T.question(u.mission, T.getR(u.roundId), u.exercise).correct })
          await page.evaluate(ci => document.querySelector(`[data-act="answer"][data-i="${ci}"]`)?.click(), ci)
          await page.evaluate(() => document.querySelector('[data-act="afterFeedback"]')?.click())
        }
        await page.waitForSelector('#v3Chart canvas'); await page.waitForTimeout(160)
        const r = await page.evaluate(() => {
          scrollTo(0, 0)
          const T = window.BSGV3._test, u = T.ui, wrap = document.getElementById('v3Chart'), L = wrap.__bsgLayout, cv = wrap.querySelector('canvas').getBoundingClientRect()
          const rel = el => { if (!el || getComputedStyle(el).display === 'none') return null; const b = el.getBoundingClientRect(); return { x0: b.left - cv.left, y0: b.top - cv.top, x1: b.right - cv.left, y1: b.bottom - cv.top } }
          const ans = [...document.querySelectorAll('[data-act="answer"]')].map(b => b.getBoundingClientRect().bottom)
          const box = el => { if (!el) return null; const cs = getComputedStyle(el); if (cs.display === 'none' || cs.visibility === 'hidden') return null; const q = el.getBoundingClientRect(); return q.width && q.height ? { x0: q.left, y0: q.top + scrollY, x1: q.right, y1: q.bottom + scrollY } : null }
          const frame = box(document.querySelector('.chartFrame')), hudB = box(document.querySelector('.hudbar')), headB = box(document.querySelector('.trainingHead')), promptB = box(document.querySelector('.prompt')), answersB = box(document.querySelector('.answers'))
          const mascots = [...document.querySelectorAll('.ckChick,.coachPortrait,.coachOrb,.homeCoachPortrait,img')].map(box).filter(Boolean)
          const R = T.getR(u.roundId), X = window.BSG.ix, i = X(0)
          return { mission: u.mission.id, ex: u.exercise, round: u.roundId, L, ctl: rel(wrap.querySelector('.chartctl')), legend: rel(wrap.querySelector('.chartlegend')), canvasH: cv.height,
            iw: innerWidth, ih: innerHeight, scrollW: document.documentElement.scrollWidth, answersBottom: ans.length ? Math.max(...ans) : null, scrollY: scrollY, frame, hudB, headB, promptB, answersB, mascots,
            closes: [R.c[i - 4], R.c[i - 3], R.c[i - 2], R.c[i - 1], R.c[i]] }
        })
        if (r.mission !== mid || r.ex !== ex) { fail(tag, 'navigation landed on M' + r.mission + ' ex' + r.ex); continue }
        const L = r.L
        if (r.scrollW > r.iw + 1) fail(tag, 'horizontal overflow ' + r.scrollW)
        if (!portrait && r.canvasH < 200) fail(tag, 'landscape chart too short: ' + Math.round(r.canvasH) + 'px')
        if (portrait && r.canvasH < 400) fail(tag, 'portrait chart squashed: ' + Math.round(r.canvasH) + 'px tall')
        if (portrait && r.frame && r.frame.y1 - r.frame.y0 < (r.frame.x1 - r.frame.x0) * 1.05) fail(tag, 'portrait chart frame flatter than tall: ' + Math.round(r.frame.x1 - r.frame.x0) + '×' + Math.round(r.frame.y1 - r.frame.y0))
        const ovr = (a, b) => a && b && Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0) > 2 && Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) > 2
        if (portrait) for (const [nm, other] of [['HUD', r.hudB], ['lesson header', r.headB], ['question', r.promptB], ['answers', r.answersB]]) if (ovr(r.frame, other)) fail(tag, 'chart overlaps ' + nm)
        for (const mm of r.mascots) if (ovr(r.frame, mm)) fail(tag, 'mascot/coach artwork inside the chart frame')
        const P = L.plot
        for (const box of [r.ctl, r.legend].filter(Boolean)) if (box.y0 < P.y1 - .5 && box.x0 < P.x1) fail(tag, 'chart control/legend intrudes into plot area')
        for (const dd of L.dots) if (dd.x < P.x0 || dd.x > P.x1 || dd.y < P.y0 - .5 || dd.y > P.y1 + .5) fail(tag, `dot ${dd.t} outside plot`)
        const texts = L.labels.map(l => ({ t: l.t, ...l.box, kind: l.kind, dot: l.dot, leader: l.leader })).concat(L.lane.map(l => ({ t: l.t, ...l.box, kind: 'lane', dot: l.dot })))
        for (let a = 0; a < texts.length; a++) for (let b = a + 1; b < texts.length; b++) if (ov(texts[a], texts[b])) fail(tag, `labels overlap: ${texts[a].t} × ${texts[b].t}`)
        for (const t of texts) {
          if (t.x0 < -.5 || t.x1 > (t.kind === 'lane' ? L.W : P.x1) + .5) fail(tag, `label ${t.t} outside chart width`)
          if (t.kind === 'marker') {
            const g = gap(t, t.dot); if (g > 12 && !t.leader) fail(tag, `label ${t.t} is ${Math.round(g)}px from its own dot`)
            const cx = (t.x0 + t.x1) / 2, cy = (t.y0 + t.y1) / 2, own = Math.hypot(t.dot.x - cx, t.dot.y - cy)
            if (!t.leader) for (const o of L.dots) if (o.t !== t.t && Math.hypot(o.x - cx, o.y - cy) + 2 < own && gap(t, o) < g) fail(tag, `label ${t.t} reads as belonging to ${o.t}`)
            for (const o of L.dots) if (o.t !== t.t && gap(t, o) < 1) fail(tag, `label ${t.t} covers dot ${o.t}`)
          }
          if (t.kind === 'lane') { if (Math.abs((t.x0 + t.x1) / 2 - t.dot.x) > .6) fail(tag, `lane label ${t.t} not centred on its candle`); if (t.y0 < L.reserved.lane[0] - .5 || t.y1 > L.reserved.lane[1] + .5) fail(tag, `lane label ${t.t} outside lane`) }
        }
        for (let a = 0; a < L.badges.length; a++) for (let b = a + 1; b < L.badges.length; b++) { const A = L.badges[a], B = L.badges[b]; if (A.y0 < B.y1 - .5 && B.y0 < A.y1 - .5) fail(tag, `axis badges overlap: ${A.t} × ${B.t}`) }
        if (mid === 6) {
          if (L.lane.length < 2 || !L.lane.some(l => l.t === 'D0') || !L.lane.some(l => l.t === 'D-4')) fail(tag, 'M06 lane must show at least D-4 and D0')
          const c = r.closes, chg = c[4] / c[0] - 1; let u = 0, d = 0; for (let k = 1; k < 5; k++) { if (c[k] > c[k - 1]) u++; else if (c[k] < c[k - 1]) d++ }
          if (!((chg >= .025 && u >= 3) || (chg <= -.025 && d >= 3) || (Math.abs(chg) <= .007 && u >= 1 && d >= 1))) fail(tag, 'M06 chart is ambiguous: closes ' + c.join(','))
        }
        if (seen.has(r.round)) fail(tag, 'same chart reused inside this mission: ' + r.round); seen.add(r.round)
        if (SHOTS.has(view + ':' + mid) && ex === 2) await page.screenshot({ path: path.join(outDir, `layout-${view}-M${String(mid).padStart(2, '0')}.png`) })
        rows.push({ view, mission: mid, ex, round: r.round, canvasH: Math.round(r.canvasH), labels: texts.length, lane: L.lane.map(l => l.t) })
      }
    }
    if (errors.length) fail('[' + view + ']', errors.join(' | '))
    await ctx.close()
  }
} finally { await browser.close() }
fs.writeFileSync(path.join(outDir, 'audit_layout.json'), JSON.stringify({ generatedAt: new Date().toISOString(), views: VIEWS.map(v => v.join('x')), missions: MISSIONS.length, checks: rows.length, failures, rows }, null, 1))
if (failures.length) { console.error('AUDIT_LAYOUT_FAIL ' + failures.length + '\n' + failures.slice(0, 80).join('\n')); process.exit(1) }
console.log('AUDIT_LAYOUT_OK', { checks: rows.length, views: VIEWS.length, missions: MISSIONS.length })