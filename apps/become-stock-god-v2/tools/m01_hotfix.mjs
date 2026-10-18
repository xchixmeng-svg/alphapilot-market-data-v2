import fs from 'node:fs'

const base = 'https://from-chick-to-stock-god-preview.vercel.app'
const outDir = 'hotfix-dist'
fs.mkdirSync(outDir, { recursive: true })

async function get(path, binary = false) {
  const r = await fetch(base + path, { redirect: 'follow' })
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`)
  return binary ? Buffer.from(await r.arrayBuffer()) : await r.text()
}

let s = await get('/')
function rep(a, b, label) {
  if (!s.includes(a)) throw new Error(`missing patch target: ${label}`)
  s = s.replace(a, b)
}

rep(
  "M(1,1,'一天四個價格','開盤、最高、最低、收盤','OHLC','先知道一根 K 棒到底由哪四個價格組成。','開＝第一筆附近；高＝當日最高；低＝當日最低；收＝最後成交附近。','不要先猜漲跌，先把四個價格認對。','candle'),",
  "M(1,1,'一天四個價格','開盤、最高、最低、收盤','OHLC','先把一根 K 棒的開、高、低、收四個位置看懂。','一根 K 棒有四個價格：開盤、最高、最低、收盤。最高／最低在影線端點；開盤／收盤在實體兩端。','示範會把四個位置標清楚；開始練習後只留價格，不顯示答案名稱。','candle'),",
  'mission copy'
)

rep(
  "function suitable(mid,R){var q=criterion(mid,R),i=q.i,k,stop,target;if(mid===6)",
  "function suitable(mid,R){var q=criterion(mid,R),i=q.i,k,stop,target;if(mid===1)return R.h[i]>Math.max(R.o[i],R.c[i])&&R.l[i]<Math.min(R.o[i],R.c[i])&&Math.abs(R.o[i]-R.c[i])>0.0001;if(mid===6)",
  'M01 sample suitability'
)

rep(
  "case'OHLC': if(ex%3===1)return sh(['開盤 '+px(q.o),'最高 '+px(R.h[i]),'收盤 '+px(q.c)],2);if(ex%3===2)return sh(['最低 '+px(R.l[i]),'開盤 '+px(q.o),'最高 '+px(R.h[i])],0);return sh(['最高 '+px(R.h[i]),'收盤 '+px(q.c),'最低 '+px(R.l[i])],0);",
  "case'OHLC': if(ex%3===1)return sh(['開盤、最高、最低、收盤','昨收、今開、均價、成交量','支撐、壓力、MA20、ATR'],0);if(ex%3===2)return sh([px(R.h[i]),px(Math.max(q.o,q.c)),px(R.l[i])],0);return sh([px(R.l[i]),px(Math.min(q.o,q.c)),px(R.h[i])],0);",
  'M01 choices'
)

rep(
  "case'OHLC':return ex%3===1?'D0 這一天，哪個是「收盤」？':ex%3===2?'D0 這一天，哪個是「最低」？':'D0 這一天，哪個是「最高」？';",
  "case'OHLC':return ex%3===1?'一根 K 棒，是由哪四種價格組成？':ex%3===2?'看 D0 右側的四個價格，哪個是最高價？':'看 D0 右側的四個價格，哪個是最低價？';",
  'M01 question text'
)

rep(
  `function promptIntro(m){return '<div class="promptTop"><span class="qtag">MISSION '+String(m.id).padStart(2,'0')+'</span><span class="difficulty">LEARN</span></div><div class="question">'+esc(m.objective)+'</div><div class="explain"><strong style="color:var(--cyan)">只新增一件事：</strong> '+esc(m.rule)+'</div><div class="coachLine">AI 教練｜'+esc(m.anti)+'</div>';}`,
  `function promptIntro(m){var extra=m.id===1?'<div class="coachLine"><strong>OHLC 看圖順序｜</strong>開盤 Open → 最高 High → 最低 Low → 收盤 Close。<br>影線最上端＝最高、最下端＝最低；中間實體的兩端＝開盤與收盤。</div>':'';return '<div class="promptTop"><span class="qtag">MISSION '+String(m.id).padStart(2,'0')+'</span><span class="difficulty">LEARN</span></div><div class="question">'+esc(m.objective)+'</div><div class="explain"><strong style="color:var(--cyan)">先學會這件事：</strong> '+esc(m.rule)+'</div>'+extra+'<div class="coachLine">AI 教練｜'+esc(m.anti)+'</div>';}`,
  'M01 teaching block'
)

rep(
  "ctx.strokeStyle=COL.grid;ctx.lineWidth=1;ctx.font='600 11px ui-monospace,Menlo,monospace';ctx.fillStyle=COL.muted;ctx.textAlign='left';ctx.textBaseline='middle';for(var gy=0;gy<=4;gy++){var py=padT+gy*(priceH-padT)/4,pv=yi(py);ctx.beginPath();ctx.moveTo(0,Math.round(py)+.5);ctx.lineTo(plotW,Math.round(py)+.5);ctx.stroke();ctx.fillText(fmt(pv),plotW+7,py);}",
  "ctx.strokeStyle=COL.grid;ctx.lineWidth=1;ctx.font='600 11px ui-monospace,Menlo,monospace';ctx.fillStyle=COL.muted;ctx.textAlign='left';ctx.textBaseline='middle';for(var gy=0;gy<=4;gy++){var py=padT+gy*(priceH-padT)/4,pv=yi(py);ctx.beginPath();ctx.moveTo(0,Math.round(py)+.5);ctx.lineTo(plotW,Math.round(py)+.5);ctx.stroke();if(!d.hideAxisTicks)ctx.fillText(fmt(pv),plotW+7,py);}",
  'axis tick control'
)

rep(
  "function up(e){var p=pointers[e.pointerId];delete pointers[e.pointerId];if(drag&&drag.type==='pan'&&!drag.moved&&p&&geo){var i=Math.round(geo.first+(p.x-(view.count-(geo.last-geo.first+1))*geo.bw)/geo.bw-.5);view.cross=(i===view.cross||i<geo.first||i>geo.last)?null:i;draw();}if(!Object.keys(pointers).length)drag=null;}",
  "function up(e){var p=pointers[e.pointerId];delete pointers[e.pointerId];if(drag&&drag.type==='pan'&&!drag.moved&&p&&geo){if(data().allowCross===false){view.cross=null;}else{var i=Math.round(geo.first+(p.x-(view.count-(geo.last-geo.first+1))*geo.bw)/geo.bw-.5);view.cross=(i===view.cross||i<geo.first||i>geo.last)?null:i;}draw();}if(!Object.keys(pointers).length)drag=null;}",
  'disable M01 tap tooltip'
)

rep(
  "return{R:R,lastIdx:last,d0Idx:i,showMA:showMA,showVolume:showVolume,zones:zones,markers:markers,lines:lines,focusCandles:focus,focusVolume:(m.id===25||m.id===26)?i:null,labelFor:function(k){return 'D'+(k-i);}};",
  "return{R:R,lastIdx:last,d0Idx:i,showMA:showMA,showVolume:showVolume,zones:zones,markers:markers,lines:lines,focusCandles:focus,focusVolume:(m.id===25||m.id===26)?i:null,allowCross:m.id!==1,hideAxisTicks:m.id===1,labelFor:function(k){return 'D'+(k-i);}};",
  'M01 chart controls'
)

rep(
`      case 1:
        out.focusCandles=[i];mark(i,R.h[i],'高','#ffd44a',8,-10);mark(i,Math.max(o,c),'開/收上緣','#39f5ff',34,8);mark(i,Math.min(o,c),'開/收下緣','#9876ff',34,22);mark(i,R.l[i],'低','#24d79b',8,18);break;`,
`      case 1:{
        out.focusCandles=[i];var a1=api(),teaching=!(a1&&a1.ui&&a1.ui.exercise>0);
        out.hideAxisTicks=true;out.allowCross=false;
        if(teaching){
          mark(i,R.h[i],'最高價','#ffd44a',8,-10);mark(i,o,'開盤價','#9876ff',34,8);mark(i,c,'收盤價','#39f5ff',34,22);mark(i,R.l[i],'最低價','#24d79b',8,18);
          line(R.h[i],'#ffd44a',R.h[i].toFixed(2));line(o,'#9876ff',o.toFixed(2));line(c,'#39f5ff',c.toFixed(2));line(R.l[i],'#24d79b',R.l[i].toFixed(2));
        }else{
          line(R.h[i],'#7890a4',R.h[i].toFixed(2));line(o,'#7890a4',o.toFixed(2));line(c,'#7890a4',c.toFixed(2));line(R.l[i],'#7890a4',R.l[i].toFixed(2));
        }
        break;}`,
  'M01 visual contract'
)

fs.writeFileSync(`${outDir}/index.html`, s)
for (const name of ['manifest.webmanifest', 'icon-192.png', 'icon-512.png', 'apple-touch-icon.png']) {
  fs.writeFileSync(`${outDir}/${name}`, await get('/' + name, true))
}

console.log(`M01 hotfix ready: ${Buffer.byteLength(s)} bytes`)
