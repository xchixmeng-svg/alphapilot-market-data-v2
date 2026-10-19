import fs from 'node:fs'
import vm from 'node:vm'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)

function assert(ok, msg) { if (!ok) throw new Error(msg) }
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

// Later indicator lessons already have a deterministic calculation engine even though the first
// 36-lesson rebuild deliberately focuses on price/structure/MA/volume/risk first.
const close=Array.from({length:80},(_,i)=>100+i*.5),high=close.map(x=>x+1),low=close.map(x=>x-1)
const macd=E.macdSeries(close,12,26,9),kd=E.kdSeries(high,low,close,9),rsi=E.rsiSeries(close,14)
assert(macd.dif.length===80&&Number.isFinite(macd.hist[79]),'MACD engine invalid')
assert(kd.k.length===80&&kd.k[79]>=0&&kd.k[79]<=100,'KD engine invalid')
assert(rsi.length===80&&rsi[79]>99,'RSI engine invalid')
assert(E.teachingIndicators({h:high,l:low,c:close}).rsi14,'indicator bundle missing')

for(const marker of ['var axisSeen={}','$PLACEHOLDER']){}
for(const marker of ['var axisSeen={}c'.slice(0,-1),'window.BSG.roundToTick(rawPv)','d.priceGuides||[]',"guideLines.push({price:p,color:'#7890a4',label:'',dash:true,guide:true})",'if(!d.hideAxisTicks)','if(data().allowCross===false)'])assert(chart.includes(marker),'chart contract missing: '+marker)

const lessonMarkers=[
  'learning-foundation-v7','先學會這件事：','OHLC 看圖順序｜',
  'K 棒中間的「實體」代表哪一段？','台股紅 K、綠 K 是怎麼決定的？','上影線是哪一段？','K 棒顏色與「今天漲跌」分別比較什麼？','要開始看一段走勢，第一步應該比較什麼？',
  '什麼樣的位置比較像「重要波峰」？','「高點墊高」是什麼意思？','「低點墊高」是什麼意思？','哪一組最符合「下降結構」？','什麼叫做「盤整」？','下降結構要出現較明確改變，通常先看哪兩件事？',
  '什麼樣的地方比較像「支撐」？','什麼樣的地方比較像「壓力」？','同一個盤整區間裡，位置重要嗎？','「碰到壓力」和「有效突破」最大的差別是什麼？','什麼叫做「假突破」？','什麼叫「突破後回測」？',
  'MA20 到底是什麼？','要判斷價格在 MA20 上方或下方','要比較「離 MA20 多遠」','要看 MA20 自己往上','加入 MA60 後','ATR 主要在量什麼？',
  '要判斷今天是放量還是縮量','量價分析為什麼不能只看','什麼條件比較像「順勢回檔」','突破買點首先要確認','完整的「突破回測」順序','哪種位置最需要提醒自己「不要追」',
  '「失策點」真正要回答什麼？','規劃第一個目標／上方空間','R:R（風險報酬比）怎麼算','部位大小正確的計算順序','續抱或退出，最重要','完整掃描的順序'
]
for(const marker of lessonMarkers)assert(app.includes(marker),'app learning contract missing: '+marker)

const introMarkers=[
  ['OHLC 看圖順序｜','M01'],['看實體｜','M02'],['看顏色｜','M03'],['看影線｜','M04'],['分兩件事｜','M05'],['從單根進到一段｜','M06'],
  ['先找波峰｜','M07'],['比較兩個高點｜','M08'],['比較兩個低點｜','M09'],['下降結構｜','M10'],['盤整也有結構｜','M11'],['反轉要證據｜','M12'],
  ['支撐是區域｜','M13'],['壓力是區域｜','M14'],['同區間，不同位置｜','M15'],['突破看收盤｜','M16'],['突破也會失敗｜','M17'],['回測是再驗證｜','M18'],
  ['MA20 是平均位置｜','M19'],['先看哪一側｜','M20'],['乖離要用百分比｜','M21'],['均線也有方向｜','M22'],['短中期一起看｜','M23'],['ATR 是波動尺｜','M24'],
  ['先建立量的基準｜','M25'],['量要跟價格一起讀｜','M26'],['回檔不是接刀｜','M27'],['突破先問「突破什麼」｜','M28'],['回測是第二次驗證｜','M29'],['會放棄也是能力｜','M30'],
  ['失策不是「我不想再賠」｜','M31'],['先看市場給多少空間｜','M32'],['先算值不值得｜','M33'],['先決定最多能賠多少｜','M34'],['續抱看理由，不看日曆｜','M35'],['最後把流程固定下來｜','M36']
]
for(const [s,id] of introMarkers)assert(app.includes(s),`${id} teaching copy missing`)

const sandbox={window:{}};vm.createContext(sandbox);vm.runInContext(contentSrc,sandbox)
const CT=sandbox.window.BSG_CONTENT
assert(CT&&Array.isArray(CT.MISSIONS)&&CT.MISSIONS.length===36,'current course map != 36')
const skills=['OHLC','BODY','COLOR','WICK','DAYCHANGE','SEQUENCE','SWING_HIGH','HIGHER_HIGH','HIGHER_LOW','DOWNTREND','RANGE','STRUCTURE_SHIFT','SUPPORT','RESISTANCE','RANGE_EDGE','BREAKOUT','FALSE_BREAKOUT','RETEST','MA20_ID','MA20_SIDE','MA20_DISTANCE','MA20_SLOPE','MA_STACK','ATR','VOLUME','PRICE_VOLUME','PULLBACK','BREAKOUT_SETUP','RETEST_SETUP','NO_CHASE','INVALIDATION','TARGET','RR','POSITION','HOLD_EXIT','FULL_SCAN']
assert(skills.length===36,'audit skill map length != 36')
for(let i=0;i<36;i++){assert(CT.MISSIONS[i].id===i+1,'mission ids must stay sequential');assert(CT.MISSIONS[i].skill===skills[i],`M${String(i+1).padStart(2,'0')} skill changed unexpectedly`)}

for(const marker of ['先教，才考','台股合法跳動單位','MACD / KD / RSI','止跌不等於反轉','逐日 K 線實戰','結果賺錢 ≠ 決策一定正確'])assert(spec.includes(marker),'V4 roadmap missing: '+marker)

const report={
  ok:true,currentMissions:36,foundation:'learning-foundation-v7',tickCases:tickCases.length,checkedPrices,legalAxis:true,exactPriceGuides:true,teachBeforeTest:true,
  modules:{beginner:'M01-M06',structure:'M07-M12',locationBreakout:'M13-M18',maAtr:'M19-M24',volumeSetups:'M25-M30',riskDecision:'M31-M36'},
  indicators:['MACD','KD','RSI14'],roadmapLocked:true
}
fs.mkdirSync('audit',{recursive:true});fs.writeFileSync('audit/learning_contract.json',JSON.stringify(report,null,2));console.log('LEARNING_CONTRACT_OK',report)
