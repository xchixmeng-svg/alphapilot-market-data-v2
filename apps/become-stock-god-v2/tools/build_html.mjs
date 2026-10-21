// 從菜雞到股神 — assemble product HTML + verified visual assets + installable PWA bundle.
import fs from 'node:fs'
import path from 'node:path'
import sharp from 'sharp'
const root = process.cwd() + path.sep
const read = p => fs.readFileSync(path.join(root, p), 'utf8')
const bank = read('build/bank.json')
const scenarios = fs.existsSync(path.join(root, 'build/scenarios.json')) ? read('build/scenarios.json') : '{"version":1,"scenarios":[]}'
const cases = fs.existsSync(path.join(root, 'build/cases.json')) ? read('build/cases.json') : '{"version":1,"mixed":[]}'
const coach = read('assets/coach.b64').replace(/\s+/g,'')
if (!coach.startsWith('UklGR')) throw new Error('coach asset is not valid WebP base64')
const art = {}
for (const name of ['hero', 'chick_rookie', 'chick_scholar', 'chick_god', 'landing']) {
  art[name] = read('assets/' + name + '.b64').replace(/\s+/g, '')
  if (!art[name].startsWith('UklGR')) throw new Error(name + ' asset is not valid WebP base64')
}
// A truncated image still starts with "UklGR", so also check the RIFF length: a broken cover once shipped as a black screen.
for (const [name, b64] of Object.entries({ coach, ...art })) {
  const buf = Buffer.from(b64, 'base64')
  if (buf.length < 12 || buf.readUInt32LE(4) + 8 !== buf.length) throw new Error(name + ' asset is truncated (' + buf.length + ' of ' + (buf.length >= 8 ? buf.readUInt32LE(4) + 8 : '?') + ' bytes)')
}
// Images ship as their own cacheable files instead of base64 copies inside the page.
const imageFiles = { coach: 'coach.webp', hero: 'hero.webp', chick_rookie: 'chick-rookie.webp', chick_scholar: 'chick-scholar.webp', chick_god: 'chick-god.webp', landing: 'landing.webp' }
let style = read('src/style.css').replaceAll('__COACH_ASSET__', './' + imageFiles.coach)
for (const name of Object.keys(art)) style = style.replaceAll('__' + name.toUpperCase() + '_ASSET__', './' + imageFiles[name])
if (/__[A-Z_]+_ASSET__/.test(style)) throw new Error('unresolved asset placeholder in style.css')
const safe = s => s.replace(/<\/script/gi, '<\\/script')
const html = `<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,maximum-scale=1">
<meta name="theme-color" content="#030814">
<meta name="application-name" content="從菜雞到股神">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="從菜雞到股神">
<meta name="description" content="從完全不會看 K 線開始，50 關循序學會價格、結構、支撐壓力、量價、均線、MACD、KD、RSI、進場、失策與逐日實戰。">
<link rel="manifest" href="./manifest.webmanifest">
<link rel="icon" type="image/png" sizes="192x192" href="./icon-192.png">
<link rel="apple-touch-icon" sizes="180x180" href="./apple-touch-icon.png">
<title>從菜雞到股神</title>
<style>${style}</style>
</head>
<body>
<div id="app"></div>
<script id="bank-data" type="application/json">${safe(bank)}</script>
<script id="scenario-data" type="application/json">${safe(scenarios)}</script>
<script id="case-data" type="application/json">${safe(cases)}</script>
<script>${safe(read('src/engine.js'))}</script>
<script>${safe(read('src/content.js'))}</script>
<script>${safe(read('src/chart.js'))}</script>
<script>${safe(read('src/app.js'))}</script>
<script>if('serviceWorker' in navigator){window.addEventListener('load',function(){navigator.serviceWorker.register('./sw.js').catch(function(){});});}</script>
</body>
</html>`
const dist = path.join(root, 'dist')
fs.mkdirSync(dist, {recursive:true})
const god = Buffer.from(art.chick_god, 'base64')
const landing = Buffer.from(art.landing, 'base64')
fs.writeFileSync(path.join(dist, 'app-icon.webp'), god)
for (const [name, file] of Object.entries(imageFiles)) fs.writeFileSync(path.join(dist, file), Buffer.from(name === 'coach' ? coach : art[name], 'base64'))
if (fs.existsSync(path.join(root, 'assets/bgm.mp3'))) fs.copyFileSync(path.join(root, 'assets/bgm.mp3'), path.join(dist, 'bgm.mp3'))   // background music (Focus Flow)
await sharp(god).resize(192,192,{fit:'cover'}).png().toFile(path.join(dist,'icon-192.png'))
await sharp(god).resize(512,512,{fit:'cover'}).png().toFile(path.join(dist,'icon-512.png'))
await sharp(god).resize(180,180,{fit:'cover'}).png().toFile(path.join(dist,'apple-touch-icon.png'))
const manifest = {
  name:'從菜雞到股神',
  short_name:'菜雞到股神',
  description:'從完全不會看 K 線開始，循序學會價格、結構、量價、指標與逐日實戰。',
  start_url:'./',
  scope:'./',
  display:'standalone',
  orientation:'any',
  background_color:'#030814',
  theme_color:'#030814',
  lang:'zh-Hant',
  icons:[
    {src:'./icon-192.png',sizes:'192x192',type:'image/png',purpose:'any maskable'},
    {src:'./icon-512.png',sizes:'512x512',type:'image/png',purpose:'any maskable'}
  ]
}
fs.writeFileSync(path.join(dist,'manifest.webmanifest'),JSON.stringify(manifest,null,2))
fs.writeFileSync(path.join(dist,'sw.js'),`self.addEventListener('install',()=>self.skipWaiting());\nself.addEventListener('activate',event=>event.waitUntil(self.clients.claim()));\n`)
fs.writeFileSync(path.join(dist, 'from-chick-to-stock-god.html'), html)
fs.writeFileSync(path.join(dist, 'index.html'), html)
console.log('dist/index.html', (Buffer.byteLength(html)/1024/1024).toFixed(2)+' MB', 'PWA icons 180/192/512 ready')
