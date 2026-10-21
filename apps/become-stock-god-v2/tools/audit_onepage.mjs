// One-page rule for phones held upright: the page itself must never scroll.
// Top bar and bottom tabs stay fixed; only the box between them may slide (long lists, a long lesson card).
// The home dashboard must fit without any inner scrolling, and a lesson must keep the chart and the
// question on the same screen. Fails on any vertical page scroll, so a later "just let it scroll" patch is caught.
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const url = process.env.BSG_AUDIT_URL || process.env.BSG_URL || 'http://127.0.0.1:4173/'
const outDir = path.resolve('audit')
fs.mkdirSync(path.join(outDir, 'onepage'), { recursive: true })
const SIZES = [[360, 780], [375, 667], [390, 664], [390, 844], [430, 932]]
const MISSIONS = [1, 4, 20, 40, 63]
const save = mid => {
  const completed = []; for (let i = 1; i < mid; i++) completed.push(i)
  return { version: 3, created: new Date().toISOString(), completed, xp: (mid - 1) * 120, coins: 0, streak: 0, premium: true, missionStats: {}, practice: { idx: 0, correct: 0, total: 0 }, exam: { idx: 0, correct: 0, total: 0 }, lastMission: mid }
}
const failures = [], rows = []
const browser = await chromium.launch({ headless: true })
try {
  for (const [w, h] of SIZES) {
    const tag = w + 'x' + h
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 })
    const page = await ctx.newPage(), errors = []
    page.on('pageerror', e => errors.push(String(e)))
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()) })
    const act = a => page.evaluate(a => { const x = document.createElement('button'); x.dataset.act = a; x.style.cssText = 'position:fixed;left:0;top:0;width:4px;height:4px'; document.body.appendChild(x); x.click(); x.remove() }, a)
    const check = async (name, opts = {}) => {
      await page.waitForTimeout(300)
      const m = await page.evaluate(() => {
        const d = document.documentElement, r = el => { if (!el) return null; const b = el.getBoundingClientRect(); return { top: Math.round(b.top), bottom: Math.round(b.bottom), h: Math.round(b.height) } }
        const content = document.querySelector('#app>main,#app>.main,#app>.paywall,#app>.mastery')
        return {
          ih: innerHeight, iw: innerWidth, sh: Math.max(d.scrollHeight, document.body.scrollHeight), sw: d.scrollWidth, scrollY: Math.round(scrollY),
          contentOverflow: content ? content.scrollHeight - content.clientHeight : 0,
          chart: r(document.querySelector('.chartFrame')), dock: r(document.querySelector('.lessonDock')), nav: r(document.querySelector('.ckNav')),
          cover: (() => { const l = document.getElementById('ckLanding'); if (!l) return null; const s = l.querySelector('.ckLandingStart'); return { start: r(s) } })()
        }
      })
      const fail = msg => failures.push(tag + ' ' + name + ': ' + msg)
      if (m.sh > m.ih + 1) fail('page scrolls vertically (' + m.sh + ' > ' + m.ih + ')')
      if (m.sw > m.iw + 1) fail('page scrolls sideways (' + m.sw + ' > ' + m.iw + ')')
      if (m.scrollY) fail('page scrolled to ' + m.scrollY)
      if (opts.noInner && m.contentOverflow > 1) fail('content needs inner scrolling (' + m.contentOverflow + 'px) — must fit one screen')
      if (opts.lesson) {
        if (!m.chart || m.chart.h < 140) fail('chart too small ' + JSON.stringify(m.chart))
        if (!m.dock || m.dock.bottom > m.ih + 1 || m.dock.top >= m.ih) fail('question dock not on screen ' + JSON.stringify(m.dock))
      }
      if (opts.cover && m.cover && (!m.cover.start || m.cover.start.bottom > m.ih + 1)) fail('start button off screen ' + JSON.stringify(m.cover))
      rows.push({ size: tag, screen: name, ...m })
      await page.screenshot({ path: path.join(outDir, 'onepage', tag + '-' + name + '.png') })
    }
    await page.goto(url, { waitUntil: 'load', timeout: 30000 })
    await page.waitForTimeout(1200)
    await check('cover', { cover: true })
    const enter = page.locator('[data-ui="enter"]')
    if (await enter.count()) { await enter.click(); await page.waitForTimeout(400) }
    await check('home', { noInner: true })
    for (const s of ['overview', 'practiceHome', 'dailyHome', 'testReport', 'paywall']) { await act(s); await check(s) }
    for (const mid of MISSIONS) {
      await page.evaluate(s => localStorage.setItem('bsg_v3_product', JSON.stringify(s)), save(mid))
      await page.reload({ waitUntil: 'load' }); await page.waitForTimeout(500)
      const en = page.locator('[data-ui="enter"]'); if (await en.count()) { await en.click(); await page.waitForTimeout(300) }
      await act('overview'); await page.waitForTimeout(300)
      const tile = page.locator('[data-act="mission"][data-id="' + mid + '"]').first()
      if (!await tile.count()) { failures.push(tag + ' M' + mid + ': mission tile missing'); continue }
      await tile.scrollIntoViewIfNeeded(); await tile.click(); await page.waitForTimeout(600)
      const id = 'M' + String(mid).padStart(2, '0')
      await check(id + '-intro', { lesson: true })
      const begin = page.locator('[data-act="beginQuestions"]:visible').first()
      if (await begin.count()) { await begin.click(); await page.waitForTimeout(500); await check(id + '-question', { lesson: true }) }
      const ans = page.locator('[data-act="answer"]:visible').first()
      if (await ans.count()) { await ans.click(); await page.waitForTimeout(500); await check(id + '-feedback', { lesson: true }) }
    }
    for (const e of errors) failures.push(tag + ' console: ' + e)
    await ctx.close()
  }
} finally { await browser.close() }
fs.writeFileSync(path.join(outDir, 'audit_onepage.json'), JSON.stringify({ generatedAt: new Date().toISOString(), sizes: SIZES.map(s => s.join('x')), failures, rows }, null, 2))
if (failures.length) { console.error('AUDIT_ONEPAGE_FAIL ' + failures.length + '\n' + failures.slice(0, 40).join('\n')); process.exitCode = 1 }
else console.log('AUDIT_ONEPAGE_OK', { sizes: SIZES.length, screens: rows.length })
