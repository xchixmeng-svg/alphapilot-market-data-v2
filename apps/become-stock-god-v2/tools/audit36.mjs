import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const url = process.env.BSG_AUDIT_URL || 'http://127.0.0.1:4173/'
const outDir = path.resolve('audit')
fs.mkdirSync(outDir,{recursive:true})

const expected = {
  1:{ma:false,vol:false,z:0,l:0,m:4}, 2:{ma:false,vol:false,z:0,l:0,m:2}, 3:{ma:false,vol:false,z:0,l:0,m:2},
  4:{ma:false,vol:false,z:0,l:0,m:4}, 5:{ma:false,vol:false,z:0,l:1,m:2}, 6:{ma:false,vol:false,z:0,l:0,m:5},
  7:{ma:false,vol:false,z:0,l:0,m:1}, 8:{ma:false,vol:false,z:0,l:0,m:2}, 9:{ma:false,vol:false,z:0,l:0,m:2},
  10:{ma:false,vol:false,z:0,l:0,m:0},11:{ma:false,vol:false,z:0,l:0,m:0},12:{ma:false,vol:false,z:0,l:1,m:2},
  13:{ma:false,vol:false,z:1,l:0,m:0},14:{ma:false,vol:false,z:1,l:0,m:0},15:{ma:false,vol:false,z:2,l:0,m:0},
  16:{ma:false,vol:false,z:1,l:1,m:0},17:{ma:false,vol:false,z:1,l:1,m:2},18:{ma:false,vol:false,z:1,l:1,m:2},
  19:{ma:'ma20',vol:false,z:0,l:0,m:1},20:{ma:'ma20',vol:false,z:0,l:0,m:2},21:{ma:'ma20',vol:false,z:0,l:0,m:2},
  22:{ma:'ma20',vol:false,z:0,l:0,m:2},23:{ma:'both',vol:false,z:0,l:0,m:3},24:{ma:false,vol:false,z:0,l:0,m:0},
  25:{ma:false,vol:true,z:0,l:0,m:0},26:{ma:false,vol:true,z:0,l:0,m:0},27:{ma:'ma20',vol:false,z:1,l:0,m:0},
  28:{ma:false,vol:false,z:1,l:1,m:0},29:{ma:false,vol:false,z:1,l:1,m:2},30:{ma:'ma20',vol:false,z:1,l:0,m:0},
  31:{ma:false,vol:false,z:1,l:1,m:0},32:{ma:false,vol:false,z:0,l:1,m:1},33:{ma:false,vol:false,z:0,l:2,m:1},
  34:{ma:false,vol:false,z:0,l:0,m:0},35:{ma:false,vol:false,z:0,l:1,m:2},36:{ma:'both',vol:true,z:2,l:0,m:0}
}

function assert(cond,msg){if(!cond)throw new Error(msg)}

const browser = await chromium.launch({headless:true})
const page = await browser.newPage({viewport:{width:844,height:390},isMobile:true,hasTouch:true,deviceScaleFactor:2})
const runtimeErrors=[]
page.on('pageerror',e=>runtimeErrors.push('pageerror: '+String(e)))
page.on('console',m=>{if(m.type()==='error')runtimeErrors.push('console: '+m.text())})

const report={generatedAt:new Date().toISOString(),viewport:'844x390',logic:[],visual:[],audio:null}
try{
  await page.goto(url,{waitUntil:'load',timeout:30000})
  await page.waitForSelector('.hudstats',{timeout:15000})
  const enter=page.locator('[data-ui="enter"]')
  if(await enter.count()){
    await enter.click()
    await page.waitForSelector('#ckLanding',{state:'detached',timeout:5000})
  }
  await page.waitForTimeout(180)
  report.audio=await page.evaluate(()=>window.BSG_AUDIO_MIXER&&window.BSG_AUDIO_MIXER.state())
  assert(report.audio,'audio mixer missing')
  assert(report.audio.music==='focus-flow-file','wrong BGM profile: '+JSON.stringify(report.audio))
  assert(report.audio.minHz>=220,'BGM minimum frequency below 220 Hz')

  const logic=await page.evaluate((expected)=>{
    function maxA(a,s,e){let v=-Infinity;for(let i=Math.max(0,s);i<=Math.min(a.length-1,e);i++)v=Math.max(v,a[i]);return v}
    function arg(a,s,e,max){let bi=Math.max(0,s),bv=max?-Infinity:Infinity;for(let i=Math.max(0,s);i<=Math.min(a.length-1,e);i++){if((max&&a[i]>bv)||(!max&&a[i]<bv)){bv=a[i];bi=i}}return bi}
    function nz(x){return x===false?false:x||false}
    const t=window.BSGV3&&window.BSGV3._test,ct=window.BSG_CONTENT,vis=window.BSG_V3_VISUALS
    if(!t||!ct||!vis)throw new Error('test hooks missing')
    const rows=[]
    for(let mid=1;mid<=36;mid++){
      const m=ct.MISSIONS[mid-1], exRows=[]
      for(let ex=1;ex<=3;ex++){
        t.setupMission(mid,ex,'mission')
        const rid=t.ui.roundId,R=t.getR(rid),q=t.question(m,R,ex),qt=t.qtext(m,R,ex),c=t.criterion(mid,R)
        if(!R)throw new Error('M'+mid+' missing round')
        if(!t.suitable(mid,R))throw new Error('M'+mid+' ex'+ex+' roundFor returned unsuitable sample')
        if(!q||!Array.isArray(q.opts)||q.opts.length!==3||!Number.isInteger(q.correct)||q.correct<0||q.correct>2)throw new Error('M'+mid+' ex'+ex+' invalid question contract')
        if(typeof qt!=='string'||!qt.trim())throw new Error('M'+mid+' ex'+ex+' empty qtext')
        if(new Set(q.opts.map(String)).size!==3)throw new Error('M'+mid+' ex'+ex+' duplicate options')
        if(mid===17){const r=maxA(R.h,c.i-22,c.i-3);let broke=false;for(let k=c.i-2;k<c.i;k++)if(R.c[k]>r*1.002)broke=true;if(!broke||!(R.c[c.i]<r))throw new Error('M17 false-breakout sample lacks close breakout then failure')}
        if(mid===29){const r=maxA(R.h,c.i-25,c.i-6);let broke=false;for(let k=c.i-5;k<c.i;k++)if(R.c[k]>r*1.005)broke=true;if(!broke||R.c[c.i]<r*.995)throw new Error('M29 sample is not a held retest')}
        if(mid===32||mid===33){const st=Math.max(.01,c.sup-c.a*.45),tg=c.res>c.c?c.res:maxA(R.h,c.i-60,c.i-1);if(!(st<c.c&&tg>c.c*1.005))throw new Error('M'+mid+' invalid stop/target geometry')}
        exRows.push({ex,round:rid,correct:q.correct})
      }
      t.setupMission(mid,1,'mission')
      const rid=t.ui.roundId,R=t.getR(rid),base=t.chartData(m,R),v=vis.transform(base,mid),c=t.criterion(mid,R),e=expected[mid]
      if(nz(v.showMA)!==nz(e.ma))throw new Error('M'+mid+' showMA mismatch '+v.showMA+' vs '+e.ma)
      if(!!v.showVolume!==!!e.vol)throw new Error('M'+mid+' volume layer mismatch')
      if((v.zones||[]).length!==e.z)throw new Error('M'+mid+' zones '+(v.zones||[]).length+' vs '+e.z)
      if((v.lines||[]).length!==e.l)throw new Error('M'+mid+' lines '+(v.lines||[]).length+' vs '+e.l)
      if((v.markers||[]).length!==e.m)throw new Error('M'+mid+' markers '+(v.markers||[]).length+' vs '+e.m)
      if(mid===7){const ix=arg(R.h,c.i-20,c.i-1,true),mk=v.markers[0];if(mk.idx!==ix||Math.abs(mk.price-R.h[ix])>1e-8)throw new Error('M07 marker not on true high')}
      if(mid===8){const a=arg(R.h,c.i-24,c.i-13,true),b=arg(R.h,c.i-12,c.i,true);if(v.markers[0].idx!==a||v.markers[1].idx!==b)throw new Error('M08 high markers wrong')}
      if(mid===9){const a=arg(R.l,c.i-24,c.i-13,false),b=arg(R.l,c.i-12,c.i,false);if(v.markers[0].idx!==a||v.markers[1].idx!==b)throw new Error('M09 low markers wrong')}
      if(mid===17){const r=maxA(R.h,c.i-22,c.i-3),z=v.zones[0],center=(z.lo+z.hi)/2;if(Math.abs(center-r)>Math.max(1e-8,Math.abs(r)*1e-7))throw new Error('M17 visual pressure != judging pressure')}
      if(mid===18||mid===29){const r=maxA(R.h,c.i-25,c.i-6),z=v.zones[0],center=(z.lo+z.hi)/2;if(Math.abs(center-r)>Math.max(1e-8,Math.abs(r)*1e-7))throw new Error('M'+mid+' visual pressure != judging pressure')}
      if(mid===22){const old=R.ma20[c.i-5],now=R.ma20[c.i];if(Math.abs(v.markers[0].price-old)>1e-8||Math.abs(v.markers[1].price-now)>1e-8)throw new Error('M22 MA20 comparison points wrong')}
      rows.push({mission:mid,skill:m.skill,exercises:exRows,layers:{ma:v.showMA||false,volume:!!v.showVolume,zones:(v.zones||[]).length,lines:(v.lines||[]).length,markers:(v.markers||[]).length}})
    }
    return rows
  },expected)
  report.logic=logic

  for(let mid=1;mid<=36;mid++){
    await page.evaluate((mid)=>{
      const completed=[];for(let i=1;i<mid;i++)completed.push(i)
      const save={version:3,created:new Date().toISOString(),completed,xp:(mid-1)*120,coins:0,streak:0,premium:true,missionStats:{},practice:{idx:0,correct:0,total:0},exam:{idx:0,correct:0,total:0},lastMission:mid}
      localStorage.setItem('bsg_v3_product',JSON.stringify(save))
    },mid)
    await page.reload({waitUntil:'load',timeout:30000})
    const launch=page.locator('[data-act="mission"][data-id="'+mid+'"]').first()
    await launch.waitFor({state:'visible',timeout:10000})
    await launch.click()
    await page.waitForSelector('#v3Chart canvas',{timeout:10000})
    await page.waitForTimeout(120)
    const metrics=await page.evaluate(()=>{
      const r=el=>{const x=el.getBoundingClientRect();return {x:x.x,y:x.y,w:x.width,h:x.height,right:x.right,bottom:x.bottom}}
      const wrap=document.querySelector('.chartwrap'),canvas=wrap&&wrap.querySelector('canvas'),frame=document.querySelector('.chartFrame'),dock=document.querySelector('.lessonDock'),training=document.querySelector('.training'),main=document.querySelector('.main')
      return {iw:innerWidth,ih:innerHeight,wrap:r(wrap),canvas:r(canvas),frame:r(frame),dock:r(dock),training:r(training),main:r(main),tag:document.querySelector('.missionTag')?.textContent||''}
    })
    assert(metrics.tag==='M'+String(mid).padStart(2,'0'),'M'+mid+' wrong rendered mission '+metrics.tag)
    assert(Math.abs(metrics.canvas.h-metrics.wrap.h)<=2.5,'M'+mid+' canvas height '+metrics.canvas.h+' != wrap '+metrics.wrap.h)
    assert(metrics.canvas.bottom<=metrics.wrap.bottom+2,'M'+mid+' canvas clips below wrap')
    { const f=metrics.frame,d=metrics.dock,ix=Math.min(f.right,d.right)-Math.max(f.x,d.x),iy=Math.min(f.bottom,d.bottom)-Math.max(f.y,d.y); assert(!(ix>3&&iy>3),'M'+mid+' chart overlaps lesson dock') }
    assert(metrics.dock.bottom<=metrics.ih+2,'M'+mid+' lesson dock below viewport')
    assert(metrics.training.bottom<=metrics.ih+2,'M'+mid+' training below viewport')
    assert(metrics.main.right<=metrics.iw+2,'M'+mid+' main exceeds viewport width')
    await page.screenshot({path:path.join(outDir,'M'+String(mid).padStart(2,'0')+'.png'),fullPage:false})
    report.visual.push({mission:mid,...metrics})
  }

  if(runtimeErrors.length)throw new Error(runtimeErrors.join('\n'))
  fs.writeFileSync(path.join(outDir,'audit36.json'),JSON.stringify(report,null,2))
  console.log('AUDIT36_OK',{missions:report.logic.length,screenshots:report.visual.length,audio:report.audio})
} catch(err) {
  report.errors=[String(err&&err.stack||err),...runtimeErrors]
  fs.writeFileSync(path.join(outDir,'audit36.json'),JSON.stringify(report,null,2))
  console.error(err)
  process.exitCode=1
} finally {
  await browser.close()
}