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
function sortParts(name, files) {
  function key(f) {
    if (name === 'app' && f === 'app.030.part') return 'app.029z1.part'
    if (name === 'app' && f === 'app.031.part') return 'app.029z2.part'
    return f
  }
  return files.sort((a,b) => key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0)
}
for (const [name, out] of Object.entries(specs)) {
  const dir = path.join(root, 'parts')
  const files = sortParts(name, fs.readdirSync(dir).filter(f => f.startsWith(name + '.') && f.endsWith('.part')))
  if (!files.length) throw new Error('missing parts for ' + name)
  let source = files.map(f => fs.readFileSync(path.join(dir,f),'utf8')).join('')

  if (name === 'engine') {
    const inc = fs.readFileSync(path.join(root,'tools','v4_indicator_engine.inc.js'),'utf8').trimEnd()
    source = replaceOne(source, '\n  var api = {', '\n'+inc+'\n\n  var api = {', 'indicator engine api boundary')
  }

  if (name === 'app') {
    const coachOrb = '<div class="coachOrb">AI</div>'
    if (!source.includes(coachOrb)) throw new Error('coach orb signature changed')
    source = source.replace(coachOrb, '')

    source = replaceOne(source,
      "s.completed=s.completed.map(Number).filter(function(x){return x>=1&&x<=36;})",
      "s.completed=s.completed.map(Number).filter(function(x){return x>=1&&x<=CT.MISSIONS.length;})",
      'save migration course length')
    source = replaceOne(source,
      "function nextMission(){for(var i=1;i<=CT.MISSIONS.length;i++)if(!completed(i))return i;return 36;}",
      "function nextMission(){for(var i=1;i<=CT.MISSIONS.length;i++)if(!completed(i))return i;return CT.MISSIONS.length;}",
      'next mission course length')
    source = replaceOne(source,
      "function rankName(){var n=S.completed.length;return n>=36?'股神候補':RANKS[Math.floor(n/6)];}",
      "function rankName(){var n=S.completed.length;if(n>=CT.MISSIONS.length)return'股神候補';if(n>=36)return'進階分析雞';return RANKS[Math.min(RANKS.length-1,Math.floor(n/6))];}",
      'rank course length')
    source = replaceOne(source,
      "function chickStage(){var n=S.completed.length;return n>=36?'god':n>=12?'scholar':'rookie';}",
      "function chickStage(){var n=S.completed.length;return n>=CT.MISSIONS.length?'god':n>=12?'scholar':'rookie';}",
      'chick course length')

    source = source.replaceAll('done>=36','done>=CT.MISSIONS.length')
    source = source.replaceAll('S.completed.length>=36','S.completed.length>=CT.MISSIONS.length')
    source = source.replaceAll('S.completed.length<36','S.completed.length<CT.MISSIONS.length')
    source = source.replaceAll('for(k=1;k<=36;k++)','for(k=1;k<=CT.MISSIONS.length;k++)')
    source = source.replaceAll('CT.MISSIONS[35]','CT.MISSIONS[CT.MISSIONS.length-1]')
    source = source.replaceAll('Math.min(36,m.id+1)','Math.min(CT.MISSIONS.length,m.id+1)')
    source = source.replaceAll("m.id<36?m.id+1:36","m.id<CT.MISSIONS.length?m.id+1:CT.MISSIONS.length")
    source = source.replaceAll("m.id<36?'<button","m.id<CT.MISSIONS.length?'<button")
    source = source.replaceAll('n>=36?', 'n>=CT.MISSIONS.length?')
    source = source.replaceAll('done/36*100','done/CT.MISSIONS.length*100')
    source = source.replaceAll("'<small>/36</small>'","'<small>/'+CT.MISSIONS.length+'</small>'")
    source = source.replaceAll("<small>/36</small>","<small>/'+CT.MISSIONS.length+'</small>")
    source = source.replaceAll("'+done+' / 36<","'+done+' / '+CT.MISSIONS.length+'<")
    source = source.replaceAll("'<span class=\"ckMuted\">'+done+' / 36</span>'","'<span class=\"ckMuted\">'+done+' / '+CT.MISSIONS.length+'</span>'")
    source = source.replaceAll("'<div class=\"ckRank\">已完成 '+done+' / 36</div>'","'<div class=\"ckRank\">已完成 '+done+' / '+CT.MISSIONS.length+'</div>'")
    source = source.replaceAll("'目前 '+S.completed.length+' / 36。</span>'","'目前 '+S.completed.length+' / '+CT.MISSIONS.length+'。</span>'")
    source = source.replaceAll("toast('完成 36 關後解鎖')","toast('完成全部課程後解鎖')")
    source = source.replaceAll('36 關總覽','完整課程')
    source = source.replaceAll('36 關全部完成','完整課程全部完成')
    source = source.replaceAll('完成 36 關解鎖','完成全部課程解鎖')
    source = source.replaceAll('36 種題型','完整題型')
    source = source.replaceAll('先完成 36 關才會開放','先完成全部課程才會開放')
    source = source.replaceAll('免費試玩到第 6 關','前 15 關免費')
    source = source.replaceAll('完整版：36 關能力鏈、400 題實戰、100 題盲測','完整版：完整能力鏈、指標整合、畫線、逐日實戰、400 題實戰、100 題盲測')
    source = source.replaceAll('<div class="payFeature">✓ 工具：MA20 / MA60 / ATR</div>','<div class="payFeature">✓ 工具：MA20 / MA60 / ATR / MACD / KD / RSI</div>')
    source = source.replaceAll('<div class="payFeature">✓ 決策：失策、R:R、部位、出場</div>','<div class="payFeature">✓ 決策：畫線、反轉、進場、失策、R:R、部位、出場</div><div class="payFeature">✓ 畢業：未來遮住的逐日 K 線實戰</div>')
  }

  if (name === 'chart') {
    source = replaceOne(
      source,
      "function fmt(p){if(!isFinite(p))return '—';var d=p<50?2:(p<1000?1:0);return p.toFixed(d);}",
      "function fmt(p){if(!isFinite(p))return '—';var e=window.BSG,d=e&&e.priceDecimals?e.priceDecimals(p):(p<50?2:(p<1000?1:0));return p.toFixed(d);}",
      'legal price formatter')
    source = replaceOne(
      source,
      "ctx.strokeStyle=COL.grid;ctx.lineWidth=1;ctx.font='600 11px ui-monospace,Menlo,monospace';ctx.fillStyle=COL.muted;ctx.textAlign='left';ctx.textBaseline='middle';for(var gy=0;gy<=4;gy++){var py=padT+gy*(priceH-padT)/4,pv=yi(py);ctx.beginPath();ctx.moveTo(0,Math.round(py)+.5);ctx.lineTo(plotW,Math.round(py)+.5);ctx.stroke();ctx.fillText(fmt(pv),plotW+7,py);}",
      "ctx.strokeStyle=COL.grid;ctx.lineWidth=1;ctx.font='600 11px ui-monospace,Menlo,monospace';ctx.fillStyle=COL.muted;ctx.textAlign='left';ctx.textBaseline='middle';var axisSeen={};for(var gy=0;gy<=4;gy++){var rawPy=padT+gy*(priceH-padT)/4,rawPv=yi(rawPy),pv=(window.BSG&&window.BSG.roundToTick)?window.BSG.roundToTick(rawPv):rawPv,pk=fmt(pv);if(axisSeen[pk])continue;axisSeen[pk]=1;var py=y(pv);ctx.beginPath();ctx.moveTo(0,Math.round(py)+.5);ctx.lineTo(plotW,Math.round(py)+.5);ctx.stroke();if(!d.hideAxisTicks)ctx.fillText(pk,plotW+7,py);}",
      'legal price axis ticks')
    source = replaceOne(source,
      "if(showVol){var vh=vmax?((R.v[k]||0)/vmax)*volH:0;",
      "if(showVol&&!d.indicatorPanel){var vh=vmax?((R.v[k]||0)/vmax)*volH:0;",
      'reserve lower panel for indicators')
    source = replaceOne(
      source,
      "  var mode=d.showMA===true?'both':d.showMA;",
      `  if(d.indicatorPanel&&showVol&&volH>0){
    var ip=d.indicatorPanel,series=ip.series||[],vals=[],z;
    series.forEach(function(s){for(z=first;z<=last;z++)if(Number.isFinite(s.values&&s.values[z]))vals.push(s.values[z]);});
    if(ip.hist)for(z=first;z<=last;z++)if(Number.isFinite(ip.hist[z]))vals.push(ip.hist[z]);
    var imin=ip.range?ip.range[0]:(vals.length?Math.min.apply(null,vals):0),imax=ip.range?ip.range[1]:(vals.length?Math.max.apply(null,vals):1);
    if(!ip.range){imin=Math.min(imin,0);imax=Math.max(imax,0);if(imax===imin){imax+=1;imin-=1;}var ipp=(imax-imin)*.08;imax+=ipp;imin-=ipp;}
    function iy(v){return volTop+(imax-v)/(imax-imin)*volH;}
    ctx.save();ctx.fillStyle='rgba(4,12,24,.72)';ctx.fillRect(0,volTop-5,plotW,volH+5);ctx.font='800 10px -apple-system,"PingFang TC",sans-serif';ctx.fillStyle=COL.muted;ctx.textAlign='left';ctx.fillText(ip.type||'IND',6,volTop+8);
    (ip.thresholds||[]).forEach(function(t){var yy=iy(t);ctx.strokeStyle='rgba(120,150,175,.25)';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(0,yy);ctx.lineTo(plotW,yy);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=COL.muted;ctx.fillText(String(t),plotW-26,yy-7);});
    if(ip.hist){var zero=iy(0);for(z=first;z<=last;z++){var hv=ip.hist[z];if(!Number.isFinite(hv))continue;var xx=x(z),yy=iy(hv),bw2=Math.max(2,Math.min(bw*.55,10));ctx.fillStyle=hv>=0?'rgba(255,70,95,.55)':'rgba(36,215,155,.55)';ctx.fillRect(xx-bw2/2,Math.min(zero,yy),bw2,Math.max(1,Math.abs(yy-zero)));}}
    series.forEach(function(s,si){ctx.strokeStyle=s.color||[COL.cyan,COL.ma20,COL.violet][si%3];ctx.lineWidth=1.6;ctx.beginPath();var on=false;for(z=first;z<=last;z++){var v=s.values&&s.values[z];if(!Number.isFinite(v)){on=false;continue;}var xx=x(z),yy=iy(v);if(!on){ctx.moveTo(xx,yy);on=true;}else ctx.lineTo(xx,yy);}ctx.stroke();var lv=s.values&&s.values[last];if(Number.isFinite(lv)){ctx.fillStyle=s.color||COL.cyan;ctx.fillText((s.name||'')+' '+(Math.abs(lv)>=100?lv.toFixed(0):lv.toFixed(1)),6+si*94,volTop+20);}});
    if(ip.summary&&ip.summary.length){ctx.font='800 9px -apple-system,"PingFang TC",sans-serif';ctx.fillStyle='#c5d9e6';var sy=volTop+Math.min(volH-7,34);ip.summary.forEach(function(s,si){ctx.fillText(s,6+(si%2)*Math.max(150,plotW*.47),sy+Math.floor(si/2)*12);});}
    ctx.restore();
  }
  var mode=d.showMA===true?'both':d.showMA;`,
      'indicator teaching panel')
    source = replaceOne(
      source,
      "var lns=(d.lines||[]).filter(function(ln){return ln.price>0;}).map(function(ln){return {ln:ln,py:y(ln.price),by:y(ln.price)};}).sort(function(a,b){return a.py-b.py;});",
      "var guideLines=[],guideSeen={};(d.priceGuides||[]).forEach(function(p){if(!isFinite(p)||p<=0)return;var k=fmt(p);if(guideSeen[k])return;guideSeen[k]=1;guideLines.push({price:p,color:'#7890a4',label:'',dash:true,guide:true});});if(d.pricePickMode&&Number.isFinite(view.pick))guideLines.push({price:view.pick,color:'#39f5ff',label:'你的線',dash:false,guide:true});var lns=(d.lines||[]).concat(guideLines).filter(function(ln){return ln.price>0;}).map(function(ln){return {ln:ln,py:y(ln.price),by:y(ln.price)};}).sort(function(a,b){return a.py-b.py;});",
      'exact-price guide lines and user line')
    source = replaceOne(
      source,
      "function up(e){var p=pointers[e.pointerId];delete pointers[e.pointerId];if(drag&&drag.type==='pan'&&!drag.moved&&p&&geo){var i=Math.round(geo.first+(p.x-(view.count-(geo.last-geo.first+1))*geo.bw)/geo.bw-.5);view.cross=(i===view.cross||i<geo.first||i>geo.last)?null:i;draw();}if(!Object.keys(pointers).length)drag=null;}",
      "function up(e){var p=pointers[e.pointerId];delete pointers[e.pointerId];if(drag&&drag.type==='pan'&&!drag.moved&&p&&geo){var dd=data();if(dd.candlePickMode){var ci=Math.round(geo.first+(p.x-(view.count-(geo.last-geo.first+1))*geo.bw)/geo.bw-.5);view.cross=null;if(ci>=geo.first&&ci<=geo.last&&typeof opt.onCandlePick==='function')opt.onCandlePick(ci);draw();}else if(dd.pricePickMode&&p.y>=0){var pp=geo.yi(Math.max(0,Math.min(geo.priceH,p.y))),eng=window.BSG;pp=eng&&eng.roundToTick?eng.roundToTick(pp):pp;view.pick=pp;view.cross=null;if(typeof opt.onPricePick==='function')opt.onPricePick(pp);draw();}else if(dd.allowCross===false){view.cross=null;draw();}else{var i=Math.round(geo.first+(p.x-(view.count-(geo.last-geo.first+1))*geo.bw)/geo.bw-.5);view.cross=(i===view.cross||i<geo.first||i>geo.last)?null:i;draw();}}if(!Object.keys(pointers).length)drag=null;}",
      'lesson crosshair and price-pick control')
    // Crosshair on every chart: tap shows a vertical + horizontal line, the price at the finger (legal tick) on
    // the price axis and the day label on top; drag moves it, tap again hides it. Open/high/low/close are only
    // listed when d.crossOHLC is true (teaching, after answering, past-only daily charts) so they never give an
    // answer away. Pick modes (draw a line / pick a candle) keep the tap for themselves.
    source = replaceOne(source,
      "else if(dd.allowCross===false){view.cross=null;draw();}else{var i=Math.round(geo.first+(p.x-(view.count-(geo.last-geo.first+1))*geo.bw)/geo.bw-.5);view.cross=(i===view.cross||i<geo.first||i>geo.last)?null:i;draw();}}if(!Object.keys(pointers).length)drag=null;}",
      "else{crossAt(p);}}else if(drag&&drag.type==='cross'&&!drag.moved){var cc=view.cross,far=p&&cc&&geo&&(Math.abs(p.x-geo.x(cc.i))>24||(cc.price!=null&&Math.abs(p.y-geo.y(cc.price))>24));if(far)crossAt(p);else{view.cross=null;draw();}}if(!Object.keys(pointers).length)drag=null;}\n function crossAt(p){if(!geo)return;var i=Math.round(geo.first+(p.x-(view.count-(geo.last-geo.first+1))*geo.bw)/geo.bw-.5);i=Math.max(geo.first,Math.min(geo.last,i));var pr=null;if(p.y>=TOP&&p.y<=geo.priceH){pr=geo.yi(p.y);var eng=window.BSG;if(eng&&eng.roundToTick)pr=eng.roundToTick(pr);}view.cross={i:i,price:pr};draw();}",
      'crosshair tap')
    source = replaceOne(source,
      "if(ids.length===1)drag={type:'pan',x0:p.x,right0:view.right,moved:false};",
      "if(ids.length===1){var d1=data();drag=view.cross&&d1&&!d1.pricePickMode&&!d1.candlePickMode?{type:'cross',x0:p.x,y0:p.y,moved:false}:{type:'pan',x0:p.x,right0:view.right,moved:false};}",
      'crosshair drag start')
    source = replaceOne(source,
      "if(drag.type==='pan'){var dx=p.x-drag.x0;",
      "if(drag.type==='cross'){if(Math.abs(p.x-drag.x0)>3||Math.abs(p.y-drag.y0)>3)drag.moved=true;if(drag.moved)crossAt(p);return;}if(drag.type==='pan'){var dx=p.x-drag.x0;",
      'crosshair drag move')
    source = replaceOne(source,
      "if(view.cross!==null&&view.cross>=first&&view.cross<=last){var ci=view.cross,cxp=x(ci);ctx.save();ctx.setLineDash([2,3]);ctx.strokeStyle='rgba(110,230,255,.55)';ctx.beginPath();ctx.moveTo(cxp,padT);ctx.lineTo(cxp,laneTop);ctx.stroke();ctx.restore();tip.style.display='block';tip.textContent=(d.labelFor?d.labelFor(ci):ci)+'  開 '+fmt(R.o[ci])+'  高 '+fmt(R.h[ci])+'  低 '+fmt(R.l[ci])+'  收 '+fmt(R.c[ci])+'  量 '+Math.round(R.v[ci]||0).toLocaleString('en-US');}else tip.style.display='none';",
      "if(view.cross&&typeof view.cross==='object'&&view.cross.i>=first&&view.cross.i<=last){var ci=view.cross.i,cxp=x(ci),cp=view.cross.price,dl=String(d.labelFor?d.labelFor(ci):ci);ctx.save();ctx.setLineDash([3,4]);ctx.strokeStyle='rgba(235,248,255,.6)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(Math.round(cxp)+.5,padT);ctx.lineTo(Math.round(cxp)+.5,laneTop);ctx.stroke();\n    if(cp!=null){var cyp=Math.round(y(cp))+.5;ctx.beginPath();ctx.moveTo(0,cyp);ctx.lineTo(plotW,cyp);ctx.stroke();ctx.setLineDash([]);var cl=fmt(cp);ctx.font='800 11px ui-monospace,Menlo,monospace';ctx.fillStyle='#eaf7ff';rr(ctx,plotW+2,cyp-9,Math.max(ctx.measureText(cl).width+10,axis-6),18,4);ctx.fill();ctx.fillStyle='#04101c';ctx.textAlign='left';ctx.textBaseline='middle';ctx.fillText(cl,plotW+7,cyp+.5);}\n    ctx.setLineDash([]);ctx.font='800 10px ui-monospace,Menlo,monospace';var dw=ctx.measureText(dl).width+10,dx0=Math.max(0,Math.min(plotW-dw,cxp-dw/2));ctx.fillStyle='#eaf7ff';rr(ctx,dx0,priceH-17,dw,15,4);ctx.fill();ctx.fillStyle='#04101c';ctx.textAlign='left';ctx.textBaseline='middle';ctx.fillText(dl,dx0+5,priceH-9);ctx.restore();\n    var ohlc=d.crossOHLC!==undefined?!!d.crossOHLC:d.allowCross!==false;tip.style.display='block';tip.textContent=ohlc?dl+'  開 '+fmt(R.o[ci])+'  高 '+fmt(R.h[ci])+'  低 '+fmt(R.l[ci])+'  收 '+fmt(R.c[ci])+'  量 '+Math.round(R.v[ci]||0).toLocaleString('en-US'):dl+(cp!=null?'  價位 '+fmt(cp):'')+'｜作答後可看開高低收';layout.cross={i:ci,price:cp,ohlc:ohlc,text:tip.textContent};}else tip.style.display='none';",
      'crosshair drawing')
  }

  if (name === 'build_bank') {
    source = replaceOne(
      source,
      "function onPriceGrid(R) {\n  for (const k of ['o', 'h', 'l', 'c']) for (const x of R[k]) if (Math.abs(x * 100 - Math.round(x * 100)) > 1e-6) return false\n  return true\n}",
      "function onPriceGrid(R) {\n  for (const k of ['o', 'h', 'l', 'c']) for (const x of R[k]) {\n    if (!Number.isFinite(x) || x <= 0) return false\n    const legal = E.roundToTick(x)\n    if (Math.abs(x - legal) > 1e-6) return false\n  }\n  return true\n}",
      'bank legal Taiwan tick grid')
  }

  // Readability pass (2026-10): every canvas label on the chart lessons is 2px larger than originally drawn.
  if (name === 'chart') source = source.replace(/(['"])(\d+) ?(\d+)px (ui-monospace|-apple-system)/g, (m, q, w, px, f) => q + w + ' ' + (+px + 2) + 'px ' + f)
  fs.mkdirSync(path.dirname(path.join(root, out)), {recursive:true})
  fs.writeFileSync(path.join(root, out), source)
  console.log('assembled', out, files.length, 'parts')
}
