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
function replaceOne(source, from, to, label) {
  if (!source.includes(from)) throw new Error('patch signature changed: ' + label)
  return source.replace(from, to)
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

  // V4 learning foundation: every value printed on the *price axis* must be a legal Taiwan-stock
  // tick. Price-choice lessons can additionally request neutral numeric guide lines through
  // data.priceGuides. This keeps exact prices readable without naming the answer.
  if (name === 'chart') {
    source = replaceOne(
      source,
      "function fmt(p){if(!isFinite(p))return '—';var d=p<50?2:(p<1000?1:0);return p.toFixed(d);}",
      "function fmt(p){if(!isFinite(p))return '—';var e=window.BSG,d=e&&e.priceDecimals?e.priceDecimals(p):(p<50?2:(p<1000?1:0));return p.toFixed(d);}",
      'legal price formatter'
    )
    source = replaceOne(
      source,
      "ctx.strokeStyle=COL.grid;ctx.lineWidth=1;ctx.font='600 11px ui-monospace,Menlo,monospace';ctx.fillStyle=COL.muted;ctx.textAlign='left';ctx.textBaseline='middle';for(var gy=0;gy<=4;gy++){var py=padT+gy*(priceH-padT)/4,pv=yi(py);ctx.beginPath();ctx.moveTo(0,Math.round(py)+.5);ctx.lineTo(plotW,Math.round(py)+.5);ctx.stroke();ctx.fillText(fmt(pv),plotW+7,py);}",
      "ctx.strokeStyle=COL.grid;ctx.lineWidth=1;ctx.font='600 11px ui-monospace,Menlo,monospace';ctx.fillStyle=COL.muted;ctx.textAlign='left';ctx.textBaseline='middle';var axisSeen={};for(var gy=0;gy<=4;gy++){var rawPy=padT+gy*(priceH-padT)/4,rawPv=yi(rawPy),pv=(window.BSG&&window.BSG.roundToTick)?window.BSG.roundToTick(rawPv):rawPv,pk=fmt(pv);if(axisSeen[pk])continue;axisSeen[pk]=1;var py=y(pv);ctx.beginPath();ctx.moveTo(0,Math.round(py)+.5);ctx.lineTo(plotW,Math.round(py)+.5);ctx.stroke();if(!d.hideAxisTicks)ctx.fillText(pk,plotW+7,py);}",
      'legal price axis ticks'
    )
    source = replaceOne(
      source,
      "var lns=(d.lines||[]).filter(function(ln){return ln.price>0;}).map(function(ln){return {ln:ln,py:y(ln.price),by:y(ln.price)};}).sort(function(a,b){return a.py-b.py;});",
      "var guideLines=[],guideSeen={};(d.priceGuides||[]).forEach(function(p){if(!isFinite(p)||p<=0)return;var k=fmt(p);if(guideSeen[k])return;guideSeen[k]=1;guideLines.push({price:p,color:'#7890a4',label:'',dash:true,guide:true});});var lns=(d.lines||[]).concat(guideLines).filter(function(ln){return ln.price>0;}).map(function(ln){return {ln:ln,py:y(ln.price),by:y(ln.price)};}).sort(function(a,b){return a.py-b.py;});",
      'exact-price guide lines'
    )
    source = replaceOne(
      source,
      "function up(e){var p=pointers[e.pointerId];delete pointers[e.pointerId];if(drag&&drag.type==='pan'&&!drag.moved&&p&&geo){var i=Math.round(geo.first+(p.x-(view.count-(geo.last-geo.first+1))*geo.bw)/geo.bw-.5);view.cross=(i===view.cross||i<geo.first||i>geo.last)?null:i;draw();}if(!Object.keys(pointers).length)drag=null;}",
      "function up(e){var p=pointers[e.pointerId];delete pointers[e.pointerId];if(drag&&drag.type==='pan'&&!drag.moved&&p&&geo){if(data().allowCross===false){view.cross=null;}else{var i=Math.round(geo.first+(p.x-(view.count-(geo.last-geo.first+1))*geo.bw)/geo.bw-.5);view.cross=(i===view.cross||i<geo.first||i>geo.last)?null:i;}draw();}if(!Object.keys(pointers).length)drag=null;}",
      'lesson crosshair control'
    )
  }

  // Raw lesson candles must themselves be valid Taiwan-stock prices. The old check only required
  // two decimal places, which still allowed impossible values such as 68.04 or 125.30.
  if (name === 'build_bank') {
    source = replaceOne(
      source,
      "function onPriceGrid(R) {\n  for (const k of ['o', 'h', 'l', 'c']) for (const x of R[k]) if (Math.abs(x * 100 - Math.round(x * 100)) > 1e-6) return false\n  return true\n}",
      "function onPriceGrid(R) {\n  for (const k of ['o', 'h', 'l', 'c']) for (const x of R[k]) {\n    if (!Number.isFinite(x) || x <= 0) return false\n    const legal = E.roundToTick(x)\n    if (Math.abs(x - legal) > 1e-6) return false\n  }\n  return true\n}",
      'bank legal Taiwan tick grid'
    )
  }

  // V3.6: the former build-time chart patches (resize guard, candle body/wick width, marker
  // placement) now live directly in parts/chart.01.part, so the renderer can be read in one place.
  fs.mkdirSync(path.dirname(path.join(root, out)), {recursive:true})
  fs.writeFileSync(path.join(root, out), source)
  console.log('assembled', out, files.length, 'parts')
}
