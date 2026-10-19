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

const tickCases = [
  [9.99,0.01], [10,0.05], [49.95,0.05], [50,0.1], [99.9,0.1],
  [100,0.5], [499.5,0.5], [500,1], [999,1], [1000,5], [1200,5]
]
for (const [p,t] of tickCases) assert(near(E.tickSize(p),t), `tickSize(${p}) != ${t}`)
function legal(p){const t=E.tickSize(p);return near(p/t,Math.round(p/t),1e-7)}
for (const p of [9.997,10.03,32.037,68.04,125.3,499.7,680.4,999.4,1202]) {
  const q=E.roundToTick(p); assert(legal(q), `roundToTick produced illegal price ${p} -> ${q}`)
}
assert(E.priceDecimals(32.05)===2,'32.xx should show two decimals')
assert(E.priceDecimals(68.1)===1,'68.x should show one decimal')
assert(E.priceDecimals(125.5)===1,'125.5 should show one decimal')
assert(E.priceDecimals(680)===0,'680 should show no decimal')
assert(E.priceDecimals(1200)===0,'1200 should show no decimal')

function decodeLessonPrices(row){const f100=a=>a.map(x=>x/100);return {o:f100(row.o),h:f100(row.h),l:f100(row.l),c:f100(row.c)}}
const bank=JSON.parse(fs.readFileSync('build/bank.json','utf8'))
let checkedPrices=0
for(const row of bank.rounds){
  const R=decodeLessonPrices(row)
  for(const k of ['o','h','l','c'])for(const p of R[k]){assert(legal(p),`illegal bank price ${row.id||row.round_id||'?'} ${k}=${p}`);checkedPrices++}
}

const close=Array.from({length:80},(_,i)=>100+i*0.5)
const high=close.map(x=>x+1),low=close.map(x=>x-1)
const macd=E.macdSeries(close,12,26,9),kd=E.kdSeries(high,low,close,9),rsi=E.rsiSeries(close,14)
assert(macd.dif.length===80&&macd.dea.length===80&&macd.hist.length===80,'MACD series length mismatch')
assert(Number.isFinite(macd.dif[79])&&Number.isFinite(macd.dea[79])&&Number.isFinite(macd.hist[79]),'MACD tail not finite')
assert(kd.k.length===80&&kd.d.length===80&&kd.k[79]>=0&&kd.k[79]<=100&&kd.d[79]>=0&&kd.d[79]<=100,'KD out of range')
assert(rsi.length===80&&rsi[79]>99,'monotonic rise should produce high RSI')
const indicators=E.teachingIndicators({h:high,l:low,c:close})
assert(indicators&&indicators.macd&&indicators.kd&&indicators.rsi14,'teachingIndicators contract missing')

for (const marker of [
  'var axisSeen={}','window.BSG.roundToTick(rawPv)','d.priceGuides||[]',
  "guideLines.push({price:p,color:'#7890a4',label:'',dash:true,guide:true})",'if(!d.hideAxisTicks)','if(data().allowCross===false)'
]) assert(chart.includes(marker), 'chart contract missing: '+marker)

for (const marker of [
  'learning-foundation-v4','先學會這件事：','OHLC 看圖順序｜',
  'K 棒中間的「實體」代表哪一段？','台股紅 K、綠 K 是怎麼決定的？','上影線是哪一段？','K 棒顏色與「今天漲跌」分別比較什麼？','要開始看一段走勢，第一步應該比較什麼？',
  '什麼樣的位置比較像「重要波峰」？','「高點墊高」是什麼意思？','「低點墊高」是什麼意思？','哪一組最符合「下降結構」？','什麼叫做「盤整」？','下降結構要出現較明確改變，通常先看哪兩件事？',
  '什麼樣的地方比較像「支撐」？','什麼樣的地方比較像「壓力」？','同一個盤整區間裡，位置重要嗎？','「碰到壓力」和「有效突破」最大的差別是什麼？','什麼叫做「假突破」？','什麼叫「突破後回測」？',
  "window.BSG.roundToTick(Math.max(.01,sup-atr*.45),'down')"
]) assert(app.includes(marker), 'app learning contract missing: '+marker)

const introMarkers=[
  ['OHLC 看圖順序｜','M01'],['看實體｜','M02'],['看顏色｜','M03'],['看影線｜','M04'],['分兩件事｜','M05'],['從單根進到一段｜','M06'],
  ['先找波峰｜','M07'],['比較兩個高點｜','M08'],['比較兩個低點｜','M09'],['下降結構｜','M10'],['盤整也有結構｜','M11'],['反轉要證據｜','M12'],
  ['支撐是區域｜','M13'],['壓力是區域｜','M14'],['同區間，不同位置｜','M15'],['突破看收盤｜','M16'],['突破也會失敗｜','M17'],['回測是再驗證｜','M18']
]
for(const [s,id] of introMarkers)assert(app.includes(s),`${id} teaching copy missing`)
assert(app.includes("label:'開盤價'")&&app.includes("label:'收盤價'"),'M01 teaching labels missing')
assert(app.includes("label:'實體上緣'")&&app.includes("label:'實體下緣'"),'M04 body/wick teaching labels missing')
assert(app.includes("label:'昨收'")&&app.includes("label:'今開'")&&app.includes("label:'今收'"),'M05 comparison evidence missing')
assert(app.includes("label:'前段高'")&&app.includes("label:'後段高'")&&app.includes("label:'前段低'")&&app.includes("label:'後段低'"),'structure comparison evidence missing')
assert(app.includes("label:quiz?'前高基準':'前一重要高點'"),'M12 neutral practice reference missing')
assert(app.includes("label:'原壓力區'")&&app.includes("label:'突破時收盤'")&&app.includes("label:'前一段收盤'"),'M17/M18 neutral practice evidence missing')
assert(app.includes('if(mid>=1&&mid<=18)out.allowCross=false'),'M01-M18 must block tap-to-reveal')

const sandbox={window:{}};vm.createContext(sandbox);vm.runInContext(contentSrc,sandbox)
const CT=sandbox.window.BSG_CONTENT
assert(CT&&Array.isArray(CT.MISSIONS)&&CT.MISSIONS.length===36,'current course map != 36')
const skills=['OHLC','BODY','COLOR','WICK','DAYCHANGE','SEQUENCE','SWING_HIGH','HIGHER_HIGH','HIGHER_LOW','DOWNTREND','RANGE','STRUCTURE_SHIFT','SUPPORT','RESISTANCE','RANGE_EDGE','BREAKOUT','FALSE_BREAKOUT','RETEST']
for(let i=0;i<skills.length;i++)assert(CT.MISSIONS[i].skill===skills[i],`M${String(i+1).padStart(2,'0')} skill changed unexpectedly`)
for (let i=0;i<CT.MISSIONS.length;i++) assert(CT.MISSIONS[i].id===i+1,'mission ids must stay sequential during migration')

for (const marker of ['先教，才考','台股合法跳動單位','MACD / KD / RSI','止跌不等於反轉','逐日 K 線實戰','結果賺錢 ≠ 決策一定正確']) assert(spec.includes(marker),'V4 roadmap missing: '+marker)

const report={
  ok:true,currentMissions:CT.MISSIONS.length,foundation:'learning-foundation-v4',tickCases:tickCases.length,checkedPrices,
  legalAxis:true,exactPriceGuides:true,
  beginnerModule:['M01','M02','M03','M04','M05','M06'],
  structureModule:['M07','M08','M09','M10','M11','M12'],
  locationBreakoutModule:['M13','M14','M15','M16','M17','M18'],
  teachBeforeTest:true,indicators:['MACD','KD','RSI14'],roadmapLocked:true
}
fs.mkdirSync('audit',{recursive:true})
fs.writeFileSync('audit/learning_contract.json',JSON.stringify(report,null,2))
console.log('LEARNING_CONTRACT_OK',report)
