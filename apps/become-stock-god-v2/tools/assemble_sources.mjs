import fs from 'node:fs'
import path from 'node:path'
const root = process.cwd()
const specs = {
  'engine':'src/engine.js',
  'app':'src/app.js',
  'content':'src/content.js',
  'chart':'src/chart.js',
  'style':'src/style.css',
  'build_bank':'tools/build_bank.mjs'
}
for (const [name, out] of Object.entries(specs)) {
  const dir = path.join(root, 'parts')
  const files = fs.readdirSync(dir).filter(f => f.startsWith(name + '.') && f.endsWith('.part')).sort()
  if (!files.length) throw new Error('missing parts for ' + name)
  let source = files.map(f => fs.readFileSync(path.join(dir,f),'utf8')).join('')

  // The in-chart coach orb competes with the candlesticks and can cover the lesson evidence.
  // The AI coach already has a dedicated right rail, so remove this legacy chart overlay at build time.
  if (name === 'app') {
    const coachOrb = '<div class="coachOrb">AI</div>'
    if (!source.includes(coachOrb)) throw new Error('coach orb signature changed')
    source = source.replace(coachOrb, '')
  }

  // The chart source historically enforced a 260px internal canvas even when the phone landscape
  // slot was shorter. Patch the assembled source so the canvas uses the real slot height, while
  // preserving a small 120px guard for transient zero-height layout states.
  if (name === 'chart') {
    const oldResize = 'H=Math.max(260,r.height);'
    const newResize = 'H=Math.max(120,r.height||120);'
    if (!source.includes(oldResize)) throw new Error('chart resize guard signature changed')
    source = source.replace(oldResize, newResize)

    const oldBody = 'body=Math.max(2,Math.min(bw*.66,13));'
    const newBody = 'body=Math.max(3,Math.min(bw*.78,22));'
    if (!source.includes(oldBody)) throw new Error('chart candle body signature changed')
    source = source.replace(oldBody, newBody)

    const oldWick = 'ctx.lineWidth=Math.max(1,Math.min(1.6,bw*.14));'
    const newWick = 'ctx.lineWidth=Math.max(1.4,Math.min(2.2,bw*.12));'
    if (!source.includes(oldWick)) throw new Error('chart wick width signature changed')
    source = source.replace(oldWick, newWick)

    const oldMarker = "ctx.fillText(m.label||'',Math.min(plotW-110,cx+8),py-10);"
    const newMarker = "var lx=cx+8+(m.dx||0),ly=py-10+(m.dy||0);lx=Math.max(4,Math.min(plotW-4,lx));ly=Math.max(padT+12,Math.min(priceH-5,ly));ctx.textAlign=m.align==='right'?'right':'left';ctx.fillText(m.label||'',lx,ly);ctx.textAlign='left';"
    if (!source.includes(oldMarker)) throw new Error('chart marker signature changed')
    source = source.replace(oldMarker, newMarker)
  }
  fs.mkdirSync(path.dirname(path.join(root, out)), {recursive:true})
  fs.writeFileSync(path.join(root, out), source)
  console.log('assembled', out, files.length, 'parts')
}
