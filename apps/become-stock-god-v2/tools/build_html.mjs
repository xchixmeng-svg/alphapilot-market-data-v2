// 從菜雞到股神 — assemble one self-contained product HTML.
import fs from 'node:fs'
import path from 'node:path'
const root = process.cwd() + path.sep
const read = p => fs.readFileSync(path.join(root, p), 'utf8')
const bank = read('build/bank.json')
// Store the coach as text-base64 so GitHub/CI never corrupts the binary asset.
const coach = read('assets/coach.b64').replace(/\s+/g,'')
if (!coach.startsWith('UklGR')) throw new Error('coach asset is not valid WebP base64')
// Brand art is stored as base64 text, like the coach asset.
const art = {}
for (const name of ['hero', 'chick_rookie', 'chick_scholar', 'chick_god', 'landing']) {
  art[name] = read('assets/' + name + '.b64').replace(/\s+/g, '')
  if (!art[name].startsWith('UklGR')) throw new Error(name + ' asset is not valid WebP base64')
}
let style = read('src/style.css').replaceAll('__COACH_ASSET__', 'data:image/webp;base64,' + coach)
for (const [name, b64] of Object.entries(art)) style = style.replaceAll('__' + name.toUpperCase() + '_ASSET__', 'data:image/webp;base64,' + b64)
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
<meta name="description" content="新手也能學會看 K 線與結構：用真實台股歷史行情，36 關一步一步學會讀 K 線、結構、位置與風險。">
<link rel="icon" type="image/webp" href="./app-icon.webp">
<link rel="apple-touch-icon" href="./app-icon.webp">
<title>從菜雞到股神</title>
<style>${style}</style>
</head>
<body>
<div id="app"></div>
<script id="bank-data" type="application/json">${safe(bank)}</script>
<script>${safe(read('src/engine.js'))}</script>
<script>${safe(read('src/content.js'))}</script>
<script>${safe(read('src/chart.js'))}</script>
<script>${safe(read('src/app.js'))}</script>
</body>
</html>`
fs.mkdirSync(path.join(root, 'dist'), {recursive:true})
fs.writeFileSync(path.join(root, 'dist/app-icon.webp'), Buffer.from(art.chick_god, 'base64'))
fs.writeFileSync(path.join(root, 'dist/from-chick-to-stock-god.html'), html)
fs.writeFileSync(path.join(root, 'dist/index.html'), html)
console.log('dist/index.html', (Buffer.byteLength(html)/1024/1024).toFixed(2)+' MB')
