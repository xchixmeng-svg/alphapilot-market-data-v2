#!/usr/bin/env python3
"""Probe the official MOPS announcement corpus for the known 4194 43-1 filing.

The list page can classify a row as Article 43-1 while the actor name lives only
in the row's detail action.  Emit the exact matching row/link markup so the next
backfill stage can follow the detail contract instead of guessing form fields.
"""
from __future__ import annotations
import html, json, re, urllib.parse, urllib.request, http.cookiejar

HOST="https://mopsov.twse.com.tw"
PAGE=HOST+"/mops/web/t146sb10"
ACTION=HOST+"/mops/web/ajax_t146sb10"
CONTROL={"stock_code":"4194","company":"禾生技","actor":"隆欣投資","roc_date":"1151012"}
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36"

def fetch(fields):
    jar=http.cookiejar.CookieJar(); op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    h={"User-Agent":UA,"Accept-Language":"zh-TW,zh;q=0.9,en;q=0.6"}
    try: op.open(urllib.request.Request(PAGE,headers=h),timeout=20).read()
    except Exception: pass
    req=urllib.request.Request(ACTION,data=urllib.parse.urlencode(fields).encode(),headers={**h,"Content-Type":"application/x-www-form-urlencoded","Referer":PAGE,"Origin":HOST},method="POST")
    with op.open(req,timeout=30) as r:return r.read().decode("utf-8","replace")

def compact_text(s):return re.sub(r"\s+","",html.unescape(re.sub(r"<[^>]+>"," ",s)))

def main():
    # Run only the variants already proven by CI to return an Article-43-1 result.
    base={"step":"1","firstin":"ture","off":"1","keyword4":"","code1":"4194","TYPEK2":"","checkbtn":"","queryName":"co_id_1","inpuType":"co_id","scope":"1","co_id_1":"4194","co_id_2":"4194","selecttype":"0","date":"4","yymmdd1":"1151012","yymmdd2":"1151012","sort":"1","noticeDate":"1"}
    for typek in ("pub","rotc","otc","sii","all",""):
      for notice_kind in ("1","4"):
        f={**base,"typek":typek,"TYPEK":typek,"noticeKind":notice_kind}
        try:text=fetch(f)
        except Exception as e:
            print(json.dumps({"typek":typek,"noticeKind":notice_kind,"error":repr(e)},ensure_ascii=False));continue
        compact=compact_text(text); has_code="4194" in text; has43="43條之1" in compact or "第43條之1" in compact; has_actor="隆欣投資" in text
        # Dump only small markup neighborhoods around the code / Article 43-1 token.
        snippets=[]
        for needle in ("4194","43條之1","第43條之1"):
            for m in list(re.finditer(re.escape(needle),text))[:2]:
                snippets.append(text[max(0,m.start()-900):min(len(text),m.end()+1400)])
        onclicks=[]
        for s in snippets:
            onclicks += re.findall(r'onclick=["\']([^"\']+)["\']',s,re.I)
            onclicks += re.findall(r'href=["\']([^"\']+)["\']',s,re.I)
        row={"typek":typek,"noticeKind":notice_kind,"bytes":len(text),"has4194":has_code,"has43":has43,"hasActor":has_actor,"links":list(dict.fromkeys(onclicks))[:12],"snippets":[re.sub(r"\s+"," ",x)[:2200] for x in snippets[:3]]}
        print("PROBE_ROW",json.dumps(row,ensure_ascii=False,sort_keys=True))
        # List-level identity is enough to prove the correct corpus/date/category;
        # actor identity remains a mandatory detail-page check before research output.
        if has_code and has43:
            print("POSITIVE_CONTROL_LIST_OK",json.dumps({"control":CONTROL,"fields":f,"actor_detail_required":not has_actor},ensure_ascii=False,sort_keys=True))
            return 0
    print("POSITIVE_CONTROL_FAILED",json.dumps(CONTROL,ensure_ascii=False));return 3
if __name__=="__main__":raise SystemExit(main())
