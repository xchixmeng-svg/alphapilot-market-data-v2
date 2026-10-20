import fs from 'node:fs'
import vm from 'node:vm'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)

function assert(ok,msg){if(!ok)throw new Error(msg)}
function near(a,b,eps=1e-8){return Math.abs(a-b)<=eps}

const E=require('../src/engine.js')
const chart=fs.readFileSync('src/chart.js','utf8')
const app=fs.readFileSync('src/app.js','utf8')
const contentSrc=fs.readFileSync('src/content.js','utf8')
const spec=fs.readFileSync('LEARNING_SYSTEM_V4.md','utf8')

const tickCases=[[9.99,.01],[10,.05],[49.95,.05],[50,.1],[99.9,.1],[100,.5],[499.5,.5],[500,1],[999,1],[1000,5],[1200,5]]
for(const [p,t] of tickCases)assert(near(E.tickSize(p),t),`tickSize(${p}) != ${t}`)
function legal(p){const t=E.tickSize(p);return near(p/t,Math.round(p/t),1e-7)}
for(const p of [9.997,10.03,32.037,68.04,125.3,499.7,680.4,999.4,1202]){const q=E.roundToTick(p);assert(legal(q),`roundToTick produced illegal price ${p} -> ${q}`)}
assert(E.priceDecimals(32.05)===2,'32.xx should show two decimals')
assert(E.priceDecimals(68.1)===1,'68.x should show one decimal')
assert(E.priceDecimals(680)===0,'680 should show no decimal')

function decodeLessonPrices(row){const f=a=>a.map(x=>x/100);return{o:f(row.o),h:f(row.h),l:f(row.l),c:f(row.c)}}
const bank=JSON.parse(fs.readFileSync('build/bank.json','utf8'))
let checkedPrices=0
for(const row of bank.rounds){const R=decodeLessonPrices(row);for(const k of ['o','h','l','c'])for(const p of R[k]){assert(legal(p),`illegal bank price ${row.id||'?'} ${k}=${p}`);checkedPrices++}}

const close=Array.from({length:80},(_,i)=>100+i*.5),high=close.map(x=>x+1),low=close.map(x=>x-1)
const macd=E.macdSeries(close,12,26,9),kd=E.kdSeries(high,low,close,9),rsi=E.rsiSeries(close,14)
assert(macd.dif.length===80&&Number.isFinite(macd.hist[79]),'MACD engine invalid')
assert(kd.k.length===80&&kd.k[79]>=0&&kd.k[79]<=100,'KD engine invalid')
assert(rsi.length===80&&rsi[79]>99,'RSI engine invalid')
assert(E.teachingIndicators({h:high,l:low,c:close}).rsi14,'indicator bundle missing')

for(const marker of [
  'var axisSeen={}','window.BSG.roundToTick(rawPv)','d.priceGuides||[]',"guideLines.push({price:p,color:'#7890a4',label:'',dash:true,guide:true})",
  'd.indicatorPanel','ip.thresholds||[]','d.pricePickMode','你的線','typeof opt.onPricePick'
])assert(chart.includes(marker),'chart contract missing: '+marker)

const lessonMarkers=[
  'OHLC 看圖順序｜','K 棒中間的「實體」代表哪一段？','台股紅 K、綠 K 是怎麼決定的？','上影線是哪一段？','K 棒顏色與「今天漲跌」分別比較什麼？','要開始看一段走勢，第一步應該比較什麼？',
  '什麼樣的位置比較像「重要波峰」？','「高點墊高」是什麼意思？','「低點墊高」是什麼意思？','哪一組最符合「下降結構」？','什麼叫做「盤整」？','下降結構要出現較明確改變，通常先看哪兩件事？',
  '什麼樣的地方比較像「支撐」？','什麼樣的地方比較像「壓力」？','同一個盤整區間裡，位置重要嗎？','「碰到壓力」和「有效突破」最大的差別是什麼？','什麼叫做「假突破」？','什麼叫「突破後回測」？',
  'MA20 到底是什麼？','要判斷價格在 MA20 上方或下方','要比較「離 MA20 多遠」','要看 MA20 自己往上','加入 MA60 後','ATR 主要在量什麼？',
  '要判斷今天是放量還是縮量','量價分析為什麼不能只看','什麼條件比較像「順勢回檔」','突破買點首先要確認','完整的「突破回測」順序','哪種位置最需要提醒自己「不要追」',
  '「失策點」真正要回答什麼？','規劃第一個目標／上方空間','R:R（風險報酬比）怎麼算','部位大小正確的計算順序','續抱或退出，最重要','完整掃描的順序',
  '時間週期真正改變的是什麼？','什麼叫做 K 線缺口？','MACD 最核心在描述什麼？','MACD 柱狀體代表什麼？','什麼叫 MACD 偏空背離？','KD 的 K、D 主要在描述什麼？','「KD 鈍化」在提醒什麼？','RSI 在量什麼？','價格結構和指標衝突時','多證據判斷應該怎麼做？','自己畫支撐／壓力時','下降結構的反轉證據','真正找進場點前','D0 未來全部遮住'
]
for(const marker of lessonMarkers)assert(app.includes(marker),'app learning contract missing: '+marker)

const sandbox={window:{}};vm.createContext(sandbox);vm.runInContext(contentSrc,sandbox)
const CT=sandbox.window.BSG_CONTENT
assert(CT&&Array.isArray(CT.MISSIONS)&&CT.MISSIONS.length===63,'current course map != 63 core lessons')
assert(Array.isArray(CT.ORDER)&&CT.ORDER.length===63&&new Set(CT.ORDER).size===63,'teaching order must list every core lesson once')
assert(CT.WORLDS.length===8,'current world map != 8')
const advanced=['TIMEFRAME','GAP','MACD_BASE','MACD_HIST','MACD_DIVERGENCE','KD_BASE','KD_STICKY','RSI','INDICATOR_CONFLICT','CONFLUENCE','DRAW_LEVEL','REVERSAL_CONFIRM','ENTRY_INVALIDATION','DAILY_REVEAL']
for(let i=0;i<CT.MISSIONS.length;i++)assert(CT.MISSIONS[i].id===i+1,'mission ids must stay sequential')
for(let i=0;i<advanced.length;i++)assert(CT.MISSIONS[36+i].skill===advanced[i],`M${37+i} advanced skill mismatch`)

for(const marker of ['先教，才考','台股合法跳動單位','MACD / KD / RSI','止跌不等於反轉','逐日 K 線實戰','結果賺錢 ≠ 決策一定正確'])assert(spec.includes(marker),'V4 roadmap missing: '+marker)
assert(app.includes('CT.MISSIONS.length'),'course length must be data driven')
assert(app.includes('window.BSG_V4_LAST_PICK'),'drawing practice callback missing')
assert(app.includes('reveal:{days:[0,1,2]}')&&app.includes('out.lastIdx=Math.min(i+revealDay,R.c.length-1)'),'daily reveal visual contract missing')

// Lesson schema contract: M37+ are data-driven lessons. Every one is registered through defineLesson and
// no app code may branch on a mission id from M37 upward — new missions must be content, not if-chains.
const lessonIds=[...app.matchAll(/defineLesson\(\{id:(\d+)/g)].map(m=>+m[1])
for(let id=37;id<=CT.MISSIONS.length;id++)assert(lessonIds.includes(id),`M${id} is not registered as a schema lesson`)
assert(new Set(lessonIds).size===lessonIds.length,'duplicate lesson registration')
const hardCoded=[...app.matchAll(/\b(?:id|mid|m\.id|mission\.id)\s*(?:===|==|!==|<=|>=|<|>)\s*(\d+)\b/g)].filter(m=>+m[1]>=37)
assert(!hardCoded.length,'mission-id branch for M37+ found (use the lesson schema): '+hardCoded.slice(0,5).map(m=>m[0]).join(', '))
for(const marker of ['window.BSG_LESSONS','function defineLesson(','interaction:{type:\'pricePick\'','Math.abs(pick-target)<=tol'])assert(app.includes(marker),'lesson schema contract missing: '+marker)

const report={
  ok:true,currentMissions:CT.MISSIONS.length,worlds:8,tickCases:tickCases.length,checkedPrices,legalAxis:true,exactPriceGuides:true,teachBeforeTest:true,
  modules:{foundation:'M01-M36',indicators:'M37-M46',integration:'M47-M50',coreGaps:'M51-M63'},layers:['core','drills','mixed','daily','final'],
  indicators:['MACD','KD','RSI14'],drawingPractice:true,dailyReveal:true,roadmapLocked:true,
  lessonSchema:{lessons:lessonIds.length,from:Math.min(...lessonIds),to:Math.max(...lessonIds),hardCodedM37Plus:0}
}
fs.mkdirSync('audit',{recursive:true});fs.writeFileSync('audit/learning_contract.json',JSON.stringify(report,null,2));console.log('LEARNING_CONTRACT_OK',report)
