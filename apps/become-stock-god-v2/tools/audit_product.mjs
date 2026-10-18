// Product audit (V4 從菜雞到股神): save compatibility, brand, 36-mission overview, mixed-skill practice / blind exam.
// Runs portrait-first (390x844) against the built page, through the real UI where it matters.
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const url = process.env.BSG_AUDIT_URL || 'http://127.0.0.1:4173/'
const outDir = path.resolve('audit')
fs.mkdirSync(outDir, { recursive: true })
const KEY = 'bsg_v3_product'
const checks = [], failures = []
function check(name, ok, detail) { checks.push({ name, ok: !!ok, detail }); if (!ok) failures.push(name + (detail === undefined ? '' : ' :: ' + JSON.stringify(detail))) }
const legacy = (completed, extra) => Object.assign({ version: 3, created: '2026-10-01T00:00:00.000Z', completed, xp: completed.length * 120 + 35, coins: completed.length * 80 + 7, streak: 3, premium: true, missionStats: { 3: { wrong: 2, hints: 1, complete: true } }, practice: { idx: 12, correct: 9, total: 12 }, exam: { idx: 4, correct: 3, total: 4 }, lastMission: completed.length + 1 }, extra || {})

const browser = await chromium.launch({ headless: true })
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 })
const page = await ctx.newPage(), errors = []
page.on('pageerror', e => errors.push('pageerror: ' + String(e)))
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()) })
const dismissLanding = async () => {
  const enter = page.locator('[data-ui="enter"]')
  if (await enter.count()) {
    await enter.click()
    await page.waitForSelector('#ckLanding', { state: 'detached', timeout: 5000 })
  }
}
const setSave = async s => {
  await page.evaluate(([k, v]) => localStorage.setItem(k, JSON.stringify(v)), [KEY, s])
  await page.reload({ waitUntil: 'load' })
  await page.waitForSelector('.hudstats', { timeout: 15000 })
  await dismissLanding()
}
const state = () => page.evaluate(k => JSON.parse(localStorage.getItem(k)), KEY)
const visibleText = () => page.evaluate(() => document.body.innerText)
const ui = () => page.evaluate(() => { const u = window.BSGV3._test.ui; return { mode: u.mode, mission: u.mission && u.mission.id, skill: u.mission && u.mission.skill, title: u.mission && u.mission.title, round: u.roundId, ex: u.exercise } })
async function answerCorrect() { const ci = await page.evaluate(() => { const T = window.BSGV3._test, u = T.ui; return T.question(u.mission, T.getR(u.roundId), u.exercise).correct }); await page.click(`[data-act="answer"][data-i="${ci}"]`); await page.click('[data-act="afterFeedback"]') }

try {
  await page.goto(url, { waitUntil: 'load', timeout: 30000 })

  const old = legacy([1, 2, 3, 4, 5, 6, 7, 8, 9])
  await setSave(old)
  const s1 = await state()
  check('save key unchanged and old progress kept', s1 && s1.version === 3 && JSON.stringify(s1.completed) === JSON.stringify(old.completed) && s1.xp === old.xp && s1.coins === old.coins && s1.streak === old.streak && s1.premium === true, s1)
  check('practice/exam history kept', s1.practice.idx === 12 && s1.practice.correct === 9 && s1.exam.idx === 4 && s1.exam.correct === 3, { practice: s1.practice, exam: s1.exam })
  check('migration adds seed and practice round', !!s1.seed && s1.practice.round === 1, { seed: s1.seed, round: s1.practice.round })
  check('mission stats kept', s1.missionStats && s1.missionStats[3] && s1.missionStats[3].wrong === 2)

  const title = await page.title()
  check('document title is the new brand', title === '從菜雞到股神', title)
  let txt = await visibleText()
  check('home is the real dashboard after cover', !!(await page.$('.ckHome.ckDashboard')) && !(await page.$('#ckLanding')))
  check('home has the three CTAs', (await page.$('.ckHero [data-act="mission"]')) && (await page.$('.ckHero [data-act="overview"]')) && (await page.$('.ckHero [data-act="practiceHome"]')))
  check('home rank uses chick naming', txt.includes('K 線小雞'), txt.slice(0, 200))
  const oldBrand = /成為股神|NEURAL MARKET ACADEMY|V3 PRODUCT TEST/
  check('no old brand on home', !oldBrand.test(txt))

  await page.click('.ckHero [data-act="overview"]'); await page.waitForSelector('.ckOverview')
  const tiles = await page.evaluate(() => [...document.querySelectorAll('.ckOverview .ckTile')].map(t => ({ id: +t.dataset.mid, enabled: !t.disabled && t.dataset.act === 'mission', cls: t.className })))
  check('overview lists all 36 missions', tiles.length === 36, tiles.length)
  check('done + next missions are enabled', tiles.filter(t => t.id <= 10).every(t => t.enabled), tiles.filter(t => t.id <= 10 && !t.enabled))
  check('future missions are locked', tiles.filter(t => t.id > 10).every(t => !t.enabled), tiles.filter(t => t.id > 10 && t.enabled))
  check('exactly one next mission (M10)', tiles.filter(t => /\bnext\b/.test(t.cls)).map(t => t.id).join() === '10')
  txt = await visibleText(); check('no old brand on overview', !oldBrand.test(txt))
  await page.evaluate(() => { const b = document.createElement('button'); b.setAttribute('data-act', 'mission'); b.dataset.id = 14; b.hidden = true; document.body.appendChild(b); b.click() })
  await page.waitForTimeout(150)
  check('skipping to a locked mission is refused', (await ui()).mission !== 14 && !!(await page.$('.ckOverview')))
  const before = await state()
  await page.click('.ckOverview .ckTile[data-mid="5"]'); await page.waitForSelector('#v3Chart canvas')
  check('replay opens the completed mission', (await ui()).mission === 5)
  await page.click('[data-act="beginQuestions"]')
  for (let k = 0; k < 3; k++) await answerCorrect()
  await page.waitForSelector('.mastery')
  const after = await state()
  check('replay does not re-award first-completion XP / coins', after.xp === before.xp && after.coins === before.coins, { before: [before.xp, before.coins], after: [after.xp, after.coins] })
  check('replay does not change completed list', JSON.stringify(after.completed) === JSON.stringify(before.completed))
  txt = await visibleText(); check('result page uses new brand', !oldBrand.test(txt) && txt.includes('關完成'))
  check('result page shows the mascot', !!(await page.$('.mastery .ckChick')))

  await setSave(legacy(Array.from({ length: 36 }, (_, i) => i + 1), { practice: { idx: 0, correct: 0, total: 0, round: 1 }, exam: { idx: 0, correct: 0, total: 0 } }))
  const plans = await page.evaluate(() => { const T = window.BSGV3._test, S = T.save(); const p1 = T.skillPlan('practice'); S.practice.round = 2; const p2 = T.skillPlan('practice'); S.practice.round = 1; return { p1, p2, ex: T.skillPlan('exam') } })
  const distinct = a => new Set(a).size, consec = a => a.some((x, i) => i && x === a[i - 1])
  check('practice plan has 400 questions', plans.p1.length === 400)
  check('practice first 100 cover >= 24 skills', distinct(plans.p1.slice(0, 100)) >= 24, distinct(plans.p1.slice(0, 100)))
  check('practice 400 cover all 36 skills', distinct(plans.p1) === 36)
  check('practice is not fixed to M36', plans.p1.filter(x => x === 36).length < 40, plans.p1.filter(x => x === 36).length)
  check('practice never repeats a skill back-to-back', !consec(plans.p1))
  check('new practice round is reshuffled', JSON.stringify(plans.p1) !== JSON.stringify(plans.p2))
  check('exam plan has 100 questions', plans.ex.length === 100)
  check('exam covers >= 18 skills', distinct(plans.ex) >= 18, distinct(plans.ex))
  check('exam not all FULL_SCAN', plans.ex.some(x => x !== 36))
  check('exam never repeats a skill back-to-back', !consec(plans.ex))

  await page.click('.ckNav [data-act="practiceHome"]'); await page.waitForSelector('.ckArena')
  await page.click('[data-act="startPractice"]'); await page.waitForSelector('#v3Chart canvas')
  const TRAIN = new Set(await page.evaluate(() => window.BSGV3._test.TRAIN)), EXAM = new Set(await page.evaluate(() => window.BSGV3._test.EXAM))
  const seenSkills = new Set(); let leak = [], pool = [], verdict = []
  const VERDICT = /假突破|跌回區間|曾收上|守住回測|^突破$|^回測$|原壓力|支撐候選|壓力候選|區間上緣|區間下緣|失策點|第一壓力/
  async function inspectScan(expectPool, tag) {
    const u = await ui(); seenSkills.add(u.skill)
    const head = await page.evaluate(() => document.querySelector('.trainingHead').innerText)
    if (head.includes(u.title)) leak.push(tag + ' header shows ' + u.title)
    if (await page.$('.main .sidebar')) leak.push(tag + ' mission list visible')
    const coach = await page.evaluate(() => { const e = document.querySelector('.coachSpeechBody'); return e ? e.textContent : '' })
    if (coach && coach.includes(u.title)) leak.push(tag + ' coach names the mission')
    if (!expectPool.has(u.round)) pool.push(tag + ' ' + u.round)
    const L = await page.evaluate(() => { const l = document.getElementById('v3Chart').__bsgLayout; return l ? l.labels.map(x => x.t).concat(l.lane.map(x => x.t)) : [] })
    for (const t of L) if (VERDICT.test(t)) verdict.push(tag + ' M' + u.mission + ' label ' + t)
    return u
  }
  for (let k = 0; k < 40; k++) { await inspectScan(TRAIN, 'practice#' + k); await answerCorrect(); await page.waitForSelector('#v3Chart canvas') }
  check('practice draws many different skills (40 questions)', seenSkills.size >= 20, seenSkills.size)
  check('practice uses only the 2020-2023 pool', pool.length === 0, pool.slice(0, 5))
  check('practice does not reveal the mission name', leak.length === 0, leak.slice(0, 5))
  check('practice chart shows no verdict labels', verdict.length === 0, verdict.slice(0, 5))
  const sp = await state(); check('practice progress saved', sp.practice.idx === 40 && sp.practice.total === 40, sp.practice)

  await page.evaluate(k => { const s = JSON.parse(localStorage.getItem(k)); s.practice.idx = 399; localStorage.setItem(k, JSON.stringify(s)) }, KEY)
  await page.reload({ waitUntil: 'load' }); await page.waitForSelector('.hudstats')
  await page.evaluate(() => { const b = document.createElement('button'); b.setAttribute('data-act', 'startPractice'); b.hidden = true; document.body.appendChild(b); b.click() })
  await page.waitForSelector('#v3Chart canvas'); await answerCorrect(); await page.waitForSelector('.ckArena')
  const sr = await state(); check('after 400 questions a new round starts', sr.practice.round === 2 && sr.practice.idx === 0, sr.practice)

  seenSkills.clear(); leak = []; pool = []; verdict = []
  await page.click('[data-act="startExam"]'); await page.waitForSelector('#v3Chart canvas')
  let hints = 0
  for (let k = 0; k < 30; k++) { await inspectScan(EXAM, 'exam#' + k); hints += await page.evaluate(() => document.querySelectorAll('[data-act="hint"],.coachLine').length); await answerCorrect(); if (k < 29) await page.waitForSelector('#v3Chart canvas') }
  check('exam draws many different skills (30 questions)', seenSkills.size >= 15, seenSkills.size)
  check('exam uses only the isolated 2024-2025 pool', pool.length === 0, pool.slice(0, 5))
  check('exam shows no hint button or hint text', hints === 0, hints)
  check('exam does not reveal the mission name', leak.length === 0, leak.slice(0, 5))
  check('exam chart shows no verdict labels', verdict.length === 0, verdict.slice(0, 5))
  const se = await state(); check('exam score recorded', se.exam.idx === 30 && se.exam.total === 30 && se.exam.correct === 30, se.exam)
  const yrs = await page.evaluate(() => { const B = JSON.parse(document.getElementById('bank-data').textContent), T = window.BSGV3._test, by = {}; B.rounds.forEach(r => by[r.id] = r); const yr = id => +String(B.calendar[by[id].d0]).slice(0, 4); return { train: [...new Set(T.TRAIN.map(yr))].sort(), exam: [...new Set(T.EXAM.map(yr))].sort() } })
  check('training pool decision dates are 2020-2023', yrs.train.every(y => y <= 2023), yrs.train)
  check('exam pool decision dates are 2024-2025', yrs.exam.every(y => y >= 2024), yrs.exam)

  if (errors.length) check('no runtime errors', false, errors.slice(0, 5)); else check('no runtime errors', true)
} catch (e) { failures.push('audit crashed: ' + (e && e.stack || e)) }
finally { await browser.close() }
fs.writeFileSync(path.join(outDir, 'audit_product.json'), JSON.stringify({ generatedAt: new Date().toISOString(), viewport: '390x844', checks, failures }, null, 1))
if (failures.length) { console.error('AUDIT_PRODUCT_FAIL ' + failures.length + '\n' + failures.join('\n')); process.exit(1) }
console.log('AUDIT_PRODUCT_OK', { checks: checks.length })