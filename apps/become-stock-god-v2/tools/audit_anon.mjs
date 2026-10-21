// Anonymity audit: while a case question is open, no company name, stock code or calendar date may be on
// screen, and none may appear in the feedback either. Only the graduation reveal (lessons 98-100: the
// "揭曉：" step, the final reveal scene and the finish summary) may name the company and the date.
// Also checks that no provenance badge ("真實歷史資料" / "教學示意") is rendered anywhere.
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'
const url = process.env.BSG_AUDIT_URL || process.env.BSG_URL || 'http://127.0.0.1:4173/'
const browser = await chromium.launch({ headless: true })
const p = await (await browser.newContext()).newPage()
await p.addInitScript(() => { window.BSG_OPEN_ALL = true; sessionStorage.setItem('bsg_landing_entered', '1') })
await p.goto(url, { waitUntil: 'load' }); await p.waitForFunction(() => window.BSG100)
const res = await p.evaluate(() => {
  const B = window.BSG100, fails = [], DATE = [/(19|20)\d{6}/, /(19|20)\d\d[\/\-.]\d\d?/, /(19|20)\d\d\s*年/], BADGE = /真實歷史資料|真實資料|真實歷史|教學示意|教學計算/
  const strip = h => String(h || '').replace(/<[^>]+>/g, ' ').replace(/aria-label="[^"]*"/g, '')
  const tl = sc => { const o = []; (function w(x) { if (!x || typeof x !== 'object') return; if (x.type === 'timeline') (x.items || []).forEach(i => o.push(i.t + ' ' + i.label)); (x.parts || []).forEach(w) })(sc); return o.join(' ') }
  const html = sc => { const t = tl(sc), m = t.match(/\b\d{1,2}\/\d{1,2}\b/); return strip(B.renderScene(sc)) + (m ? ' 2000/' + m[0] : '') }
  let checked = 0
  for (let id = 64; id <= 100; id++) {
    const sp = B.spec(id), data = B.cases(sp.dataFrom || id), names = new Set()
    ;(function walk(o) { if (!o || typeof o !== 'object') return; if (typeof o.name === 'string' && o.name.length >= 2) names.add(o.name); if (typeof o.code === 'string') names.add(o.code); if (Array.isArray(o.names)) o.names.forEach(n => names.add(n)); (Array.isArray(o) ? o : Object.values(o)).forEach(walk) })(data)
    const bad = (t, where) => { checked++; for (const n of names) if (t.includes(n)) fails.push(`L${id} ${where}: name/code "${n}"`); for (const r of DATE) { const m = t.match(r); if (m) fails.push(`L${id} ${where}: date "${m[0]}"`) } const b = t.match(BADGE); if (b) fails.push(`L${id} ${where}: badge "${b[0]}"`) }
    const all = B.allCases(id)
    all.forEach((c, ci) => {
      bad(html(c.scene), `case${ci} scene`)
      ;(c.qs || []).forEach((q, qi) => {
        const txt = typeof q.text === 'function' ? q.text(c) : q.text, reveal = id >= 98 && /^揭曉：/.test(txt)
        if (reveal) return
        const w = `case${ci} q${qi}`
        bad(txt + ' ' + (q.options || []).join(' '), w + ' question')
        if (q.scene) bad(html(q.scene), w + ' scene')
        if (q.revealScene) bad(html(q.revealScene), w + ' revealScene')
        bad(String(typeof q.why === 'function' ? '' : q.why || '') + ' ' + Object.values(q.wrong || {}).join(' '), w + ' feedback')
      })
    })
    bad(sp.teach.map(t => t.join(' ')).join(' ') + ' ' + sp.takeaway, 'teach')
  }
  return { fails, checked }
})
await browser.close()
fs.mkdirSync('audit', { recursive: true }); fs.writeFileSync(path.join('audit', 'audit_anon.json'), JSON.stringify(res, null, 2))
console.log(`anon audit: ${res.checked} texts checked, ${res.fails.length} failures`)
res.fails.slice(0, 60).forEach(f => console.log('  ✗ ' + f))
process.exit(res.fails.length ? 1 : 0)
