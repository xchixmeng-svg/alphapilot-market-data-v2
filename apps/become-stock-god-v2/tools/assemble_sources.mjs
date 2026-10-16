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

  // V3.6: the former build-time chart patches (resize guard, candle body/wick width, marker
  // placement) now live directly in parts/chart.01.part, so the renderer can be read in one place.
  fs.mkdirSync(path.dirname(path.join(root, out)), {recursive:true})
  fs.writeFileSync(path.join(root, out), source)
  console.log('assembled', out, files.length, 'parts')
}
