import fs from 'node:fs'
import vm from 'node:vm'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)

function assert(ok, msg) { if (!ok) throw new Error(msg) }
function near(a,b,eps=1e-8){return Math.abs(a-b)<=eps}

const E = require('../src/engine.js')
const chart = fs.readFileSync('src/chart.js','utf8')
const app = fs.readFileSync('src/app.js','utf8')
const contentSrc = fs.readFileSync('src/content.js','utf8')
const spec = fs.readFileSync('LEARNING_SYSTEM_V4.md','utf8')

// Taiwan stock tick ladder.
const tickCases = [
  [9.99,0.01], [10,0.05], [49.95,0.05], [50,0.1], [99.9,0.1],
  [100,0.5], [499.5,0.5], [500,1], [999,1], [1000,5], [1200,5]
]
for (const [p,t] of tickCases) assert(near(E.tickSize(p),t), `tickSize(${p}) != ${t}`)

function legal(p){const t=E.tickSize(p);return near(p/t,Math.round(p/t),1e-7)}
for (const p of [9.997,10.03,32.037,68.04,125.3,499.7,680.4,999.4,1202]) {
  const q=E.roundToTick(p)
  assert(legal(q), `roundToTick produced illegal price ${p} -> ${q}`)
}
assert(E.priceDecimals(32.05)===2,'32.xx should show two decimals')
assert(E.priceDecimals(68.1)===1,'68.x should show one decimal')
assert(E.priceDecimals(125.5)===1,'125.5 should show one decimal')
assert(E.priceDecimals(680)===0,'680 should show no decimal')
assert(E.priceDecimals(1200)===0,'1200 should show no decimal')

// Assembled renderer contract.
for (const marker of [
  'var axisSeen={}',
  'window.BSG.roundToTick(rawPv)',
  'd.priceGuides||[]',
  "guideLines.push({price:p,color:'#7890a4',label:'',dash:true,guide:true})",
  'if(!d.hideAxisTicks)',
  'if(data().allowCross===false)'
]) assert(chart.includes(marker), 'chart contract missing: '+marker)

// App pedagogy contract.
for (const marker of [
  'learning-foundation-v1',
  '先學會這件事：',
  'OHLC 看圖順序｜',
  "out.priceGuides=uniquePrices([R.h[i],o,c,R.l[i]])",
  "out.hideAxisTicks=true;out.allowCross=false",
  "window.BSG.roundToTick(Math.max(.01,sup-atr*.45),'down')"
]) assert(app.includes(marker), 'app learning contract missing: '+marker)

// M01 must teach before testing and must not use the old ambiguous open/close-edge labels in the
// V4 foundation path.
assert(app.includes('一根 K 棒，是由哪四種價格組成？'),'M01 concept check missing')
assert(app.includes('哪個是最高價？')&&app.includes('哪個是最低價？'),'M01 price checks missing')
assert(app.includes("label:'開盤價'")&&app.includes("label:'收盤價'"),'M01 teaching labels missing')

// Current production course map is still 36 missions while V4 is built incrementally.
const sandbox={window:{}};vm.createContext(sandbox);vm.runInContext(contentSrc,sandbox)
const CT=sandbox.window.BSG_CONTENT
assert(CT&&Array.isArray(CT.MISSIONS)&&CT.MISSIONS.length===36,'current course map != 36')
assert(CT.MISSIONS[0].skill==='OHLC','M01 must remain OHLC')
for (let i=0;i<CT.MISSIONS.length;i++) assert(CT.MISSIONS[i].id===i+1,'mission ids must stay sequential during migration')

// Locked roadmap must include the full beginner path requested for V4.
for (const marker of ['先教，才考','台股合法跳動單位','MACD / KD / RSI','止跌不等於反轉','逐日 K 線實戰','結果賺錢 ≠ 決策一定正確']) {
  assert(spec.includes(marker),'V4 roadmap missing: '+marker)
}

const report={
  ok:true,
  currentMissions:CT.MISSIONS.length,
  foundation:'learning-foundation-v1',
  tickCases:tickCases.length,
  legalAxis:true,
  exactPriceGuides:true,
  m01TeachBeforeTest:true,
  roadmapLocked:true
}
fs.mkdirSync('audit',{recursive:true})
fs.writeFileSync('audit/learning_contract.json',JSON.stringify(report,null,2))
console.log('LEARNING_CONTRACT_OK',report)
