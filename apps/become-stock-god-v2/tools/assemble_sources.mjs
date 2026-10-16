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
  fs.mkdirSync(path.dirname(path.join(root, out)), {recursive:true})
  fs.writeFileSync(path.join(root, out), files.map(f => fs.readFileSync(path.join(dir,f),'utf8')).join(''))
  console.log('assembled', out, files.length, 'parts')
}
