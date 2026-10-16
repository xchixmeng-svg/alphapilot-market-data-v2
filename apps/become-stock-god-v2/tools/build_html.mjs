// 成為股神 V2 — assemble one offline HTML from the local source tree.
import fs from 'node:fs'
import path from 'node:path'
const root = process.cwd() + path.sep
const read = p => fs.readFileSync(path.join(root, p), 'utf8')
const bank = read('build/bank.json')
const safe = s => s.replace(/<\/script/gi, '<\\/script')
const html = `<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,maximum-scale=1">
<meta name="theme-color" content="#071019">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>成為股神 V2｜K線駕訓班</title>
<style>${read('src/style.css')}</style>
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
fs.writeFileSync(path.join(root, 'dist/become-stock-god-v2-alpha.html'), html)
console.log('dist/become-stock-god-v2-alpha.html', (Buffer.byteLength(html)/1024/1024).toFixed(2)+' MB')
