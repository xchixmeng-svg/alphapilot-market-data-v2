import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const url = process.env.BSG_AUDIT_URL || 'http://127.0.0.1:4173/'
const outDir = path.resolve('audit')
fs.mkdirSync(outDir,{recursive:true})
function assert(cond,msg){if(!cond)throw new Error(msg)}
function saveFor(mid){
  const completed=[]; for(let i=1;i<mid;i++)completed.push(i)
  return {version:3,created:new Date().toISOString(),completed,xp:(mid-1)*120,coins:0,streak:0,premium:true,missionStats:{},practice:{idx:0,correct:0,total:0},exam:{idx:0,correct:0,total:0},lastMission:mid}
}

const browser = await chromium.launch({headless:true})
const page = await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true,deviceScaleFactor:2})
const errors=[]
page.on('pageerror',e=>errors.push('pageerror: '+String(e)))
page.on('console',m=>{if(m.type()==='error')errors.push('console: '+m.text())})
try{
  await page.goto(url,{waitUntil:'load',timeout:30000})
  await page.waitForSelector('.hudstats',{timeout:15000})
  const home = await page.evaluate(()=>{
    const el=document.querySelector('.homeCoach')
    const cs=el?getComputedStyle(el):null
    return {iw:innerWidth,scrollW:document.documentElement.scrollWidth,coachDisplay:cs&&cs.display}
  })
  assert(home.scrollW<=home.iw+2,'portrait home horizontal overflow: '+JSON.stringify(home))
  assert(!home.coachDisplay||home.coachDisplay==='none','portrait home coach artwork still visible')
  await page.screenshot({path:path.join(outDir,'portrait-home.png'),fullPage:false})

  const rows=[]
  for(const mid of [4,36]){
    await page.evaluate((save)=>localStorage.setItem('bsg_v3_product',JSON.stringify(save)),saveFor(mid))
    await page.reload({waitUntil:'load',timeout:30000})
    const launch=page.locator('[data-act="mission"][data-id="'+mid+'"]').first()
    await launch.waitFor({state:'visible',timeout:10000})
    await launch.click()
    await page.waitForSelector('#v3Chart canvas',{timeout:10000})
    await page.waitForTimeout(180)
    const m=await page.evaluate(()=>{
      const rect=el=>{const r=el.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height,right:r.right,bottom:r.bottom}}
      const rail=document.querySelector('.coachRail'),portrait=document.querySelector('.coachPortrait'),frame=document.querySelector('.chartFrame'),training=document.querySelector('.training'),prompt=document.querySelector('.prompt')
      const rs=rail?getComputedStyle(rail):null,ps=portrait?getComputedStyle(portrait):null
      return {iw:innerWidth,ih:innerHeight,scrollW:document.documentElement.scrollWidth,railDisplay:rs&&rs.display,portraitDisplay:ps&&ps.display,frame:rect(frame),training:rect(training),prompt:rect(prompt),tag:document.querySelector('.missionTag')?.textContent||''}
    })
    assert(m.tag==='M'+String(mid).padStart(2,'0'),'portrait M'+mid+' wrong mission '+m.tag)
    assert(!m.railDisplay||m.railDisplay==='none','portrait M'+mid+' coach rail still visible')
    assert(!m.portraitDisplay||m.portraitDisplay==='none','portrait M'+mid+' coach image still visible')
    assert(m.scrollW<=m.iw+2,'portrait M'+mid+' horizontal overflow '+m.scrollW+' > '+m.iw)
    assert(m.frame.x>=-1&&m.frame.right<=m.iw+1,'portrait M'+mid+' chart frame outside viewport '+JSON.stringify(m.frame))
    assert(m.training.x>=-1&&m.training.right<=m.iw+1,'portrait M'+mid+' training outside viewport '+JSON.stringify(m.training))
    assert(m.frame.w>=m.iw-24,'portrait M'+mid+' chart did not reclaim viewport width: '+m.frame.w+' / '+m.iw)
    await page.screenshot({path:path.join(outDir,'portrait-M'+String(mid).padStart(2,'0')+'.png'),fullPage:false})
    rows.push({mission:mid,...m})
  }
  if(errors.length)throw new Error(errors.join('\n'))
  fs.writeFileSync(path.join(outDir,'audit_portrait.json'),JSON.stringify({generatedAt:new Date().toISOString(),viewport:'390x844',home,missions:rows},null,2))
  console.log('AUDIT_PORTRAIT_OK',{viewport:'390x844',missions:rows.map(x=>x.mission)})
} catch(err){
  fs.writeFileSync(path.join(outDir,'audit_portrait.json'),JSON.stringify({generatedAt:new Date().toISOString(),viewport:'390x844',errors:[String(err&&err.stack||err),...errors]},null,2))
  console.error(err)
  process.exitCode=1
} finally {
  await browser.close()
}
