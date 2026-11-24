// AlphaPilot 早報 Worker（v25.0：加探底燈號早上推播）：不用 AI 模型，Cloudflare 免費方案。
//   07:00 早報（週一～週六；週六是美股週五收盤）、09:05 開盤核對、13:35 收盤快報、週日 20:00 下週預覽
//   盤中與盤後事件監控：觀察名單（入選、決賽、系統部位、你的持股）的重大訊息、注意股／處置股
//   只用一個「每 5 分鐘」排程觸發，程式內依台灣時間分派；錯過的工作 30 分鐘內補跑，同一天不重複。
// 設定方法見 SETUP.md。環境變數（Cloudflare 的 Variables and Secrets）：
//   BOT_TOKEN（secret）、CHAT_ID、FAMILY_CHAT_ID（可空）、SECRET（secret，和電腦 secrets.json 的 brief_secret 相同）
//   HOLIDAYS（可空，額外的台股休市日，逗號分隔 YYYYMMDD）
// KV 綁定名稱：BRIEF

// 台股 2026 休市日（證交所公告）。每年底請把隔年的休市日加到 HOLIDAYS 環境變數，或更新這裡。
const TW_HOLIDAYS = new Set(["20260101", "20260212", "20260213", "20260216", "20260217", "20260218", "20260219", "20260220",
  "20260227", "20260403", "20260406", "20260501", "20260619", "20260925", "20260928", "20261009", "20261026", "20261225"]);

export const CRON = "*/5 * * * *";            // v24.6：唯一的排程觸發（免費方案每帳號最多 5 個）

export const SYMBOLS = [
  ["^SOX", "費半"], ["^IXIC", "那指"], ["^GSPC", "標普"], ["TSM", "台積電 ADR"],
  ["^TNX", "美10年債"], ["TWD=X", "美元/台幣"], ["^VIX", "VIX"],
];
const NEWS = [
  ["policy", "川普 OR 關稅 OR 出口管制 OR 聯準會 OR 鮑爾"],
  ["market", "台股 OR 台積電 OR 半導體 OR 輝達"],
];

// ---------------------------------------------------------------- 時間
export function tw(now = Date.now()) {
  const d = new Date(now + 8 * 3600e3);
  const ymd = d.toISOString().slice(0, 10).replaceAll("-", "");
  return { ymd, wd: d.getUTCDay(), md: `${d.getUTCMonth() + 1}/${d.getUTCDate()}`, hm: d.toISOString().slice(11, 16), date: d };
}
const WD = "日一二三四五六";
export function isHoliday(ymd, env = {}) {
  const extra = new Set(String(env.HOLIDAYS || "").split(",").map((s) => s.trim()).filter(Boolean));
  const d = new Date(`${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6)}T00:00:00Z`);
  return d.getUTCDay() === 0 || d.getUTCDay() === 6 || TW_HOLIDAYS.has(ymd) || extra.has(ymd);
}
function addDays(ymd, k) {
  const d = new Date(`${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6)}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + k);
  return d.toISOString().slice(0, 10).replaceAll("-", "");
}
export function prevTradingDay(ymd, env = {}) {
  let d = addDays(ymd, -1);
  while (isHoliday(d, env)) d = addDays(d, -1);
  return d;
}
// 美股最近一個應有收盤的日子（美東日期）＝台灣日期的前一個平日
export function expectedUsDate(twYmd) {
  let d = addDays(twYmd, -1);
  for (;;) {
    const wd = new Date(`${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6)}T00:00:00Z`).getUTCDay();
    if (wd !== 0 && wd !== 6) return d;
    d = addDays(d, -1);
  }
}

// ---------------------------------------------------------------- 國際數字（Yahoo Finance 圖表介面）
export function parseChart(js, nowSec = Date.now() / 1000) {
  const r = js?.chart?.result?.[0];
  if (!r) return null;
  const ts = r.timestamp || [], cl = r.indicators?.quote?.[0]?.close || [], off = r.meta?.gmtoffset || 0;
  const pts = ts.map((t, i) => [t, cl[i]]).filter(([, c]) => c != null);
  if (pts.length < 2) return null;
  const [t1, c1] = pts[pts.length - 1], [, c0] = pts[pts.length - 2];
  const day = (t) => new Date((t + off) * 1000).toISOString().slice(0, 10).replaceAll("-", "");
  const reg = r.meta?.currentTradingPeriod?.regular;  // 美股交易中：最後一根是盤中價，不是收盤
  const live = !!(reg && reg.start && reg.end && t1 >= reg.start - 86400 && nowSec >= reg.start && nowSec < reg.end);
  return { last: c1, prev: c0, chg: c1 / c0 - 1, date: day(t1), live, series: pts.map(([t, c]) => [day(t), c]) };
}
// 台股上次收盤（refTw 13:30）之後的累計變化：基準＝美東日期早於 refTw 的最後一個收盤（台股收盤時已知的最新美股收盤）
export function cumSince(qt, refTw) {
  const s = qt?.series || [];
  const base = [...s].reverse().find(([d]) => d < refTw);
  const n = s.filter(([d]) => d >= refTw).length;
  return base && n ? { chg: qt.last / base[1] - 1, delta: qt.last - base[1], sessions: n, baseDate: base[0] } : null;
}
async function fetchQuote(sym) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(sym)}?range=10d&interval=1d`;
  const r = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0" } });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return parseChart(await r.json());
}
const sgn = (x, d = 2) => (x >= 0 ? "+" : "") + x.toFixed(d);
export function formatMarkets(q, twYmd, tsmcClose, refTw) {
  const lines = [], notes = [];
  const exp = expectedUsDate(twYmd);
  const us = ["^SOX", "^IXIC", "^GSPC"].map((s) => q[s]).find(Boolean);
  if (us && us.date < exp) notes.push(`美股最近收盤是 ${+us.date.slice(4, 6)}/${+us.date.slice(6)}（美股休市或資料未更新），以下不是昨晚的新數字`);
  const pc = (s, lab) => (q[s] ? `${lab} ${sgn(q[s].chg * 100)}%` : `${lab} 取得失敗`);
  lines.push([pc("^SOX", "費半"), pc("^IXIC", "那指"), pc("^GSPC", "標普")].join("｜"));
  if (q["TSM"]) {
    let t = `台積電 ADR ${sgn(q["TSM"].chg * 100)}%`;
    if (q["TWD=X"] && tsmcClose) t += `（換算溢價 ${sgn((q["TSM"].last * q["TWD=X"].last / 5 / tsmcClose - 1) * 100, 1)}%，對比台股最近收盤 ${tsmcClose}）`;
    lines.push(t);
  } else lines.push("台積電 ADR 取得失敗");
  const t = q["^TNX"] ? `美10年債 ${q["^TNX"].last.toFixed(2)}%（${sgn((q["^TNX"].last - q["^TNX"].prev) * 100, 0)}bp）` : "美10年債 取得失敗";
  const fx = q["TWD=X"] ? `美元/台幣 ${q["TWD=X"].last.toFixed(2)}（${sgn(q["TWD=X"].chg * 100)}%）` : "美元/台幣 取得失敗";
  const v = q["^VIX"] ? `VIX ${q["^VIX"].last.toFixed(1)}（${sgn(q["^VIX"].last - q["^VIX"].prev, 1)}）` : "VIX 取得失敗";
  lines.push([t, fx, v].join("｜"));
  // v24.8：台股休市期間美股交易了 2 天以上（連假），加上「自台股上次收盤累計」
  const ref = refTw || null, cum = {};
  let sessions = 0;
  if (ref) for (const s of Object.keys(q)) { const c = cumSince(q[s], ref); if (c) { cum[s] = c; sessions = Math.max(sessions, c.sessions); } }
  const gap = [];
  if (sessions >= 2) {
    const cp = (s, lab) => (cum[s] ? `${lab} ${sgn(cum[s].chg * 100)}%` : `${lab} —`);
    gap.push([cp("^SOX", "費半"), cp("^IXIC", "那指"), cp("^GSPC", "標普")].join("｜"));
    gap.push(cp("TSM", "台積電 ADR") + (q["TSM"] && q["TWD=X"] && tsmcClose
      ? `（ADR 換算每股 ${(q["TSM"].last * q["TWD=X"].last / 5).toFixed(0)} 元；台股最近收盤 ${tsmcClose}。ADR 長期有溢價，看累計漲跌的方向比看價差準）` : ""));
    const lvl = (s, d) => `${(q[s].last - cum[s].delta).toFixed(d)}→${q[s].last.toFixed(d)}`;
    gap.push([cum["^TNX"] ? `美10年債 ${lvl("^TNX", 2)}%（${sgn(cum["^TNX"].delta * 100, 0)}bp）` : "美10年債 —",
      cum["TWD=X"] ? `美元/台幣 ${lvl("TWD=X", 2)}（${sgn(cum["TWD=X"].chg * 100)}%，${cum["TWD=X"].chg > 0 ? "台幣貶" : "台幣升"}）` : "美元/台幣 —",
      cum["^VIX"] ? `VIX ${sgn(cum["^VIX"].delta, 1)}` : "VIX —"].join("｜"));
  }
  const live = ["^SOX", "^IXIC", "^GSPC"].some((k) => q[k]?.live);
  if (live) notes.push("美股還在交易中，最近一天的數字是盤中價，不是收盤（排程 07:00 發送時已收盤，不會有這個狀況）");
  return { lines, notes, usDate: us?.date, live, gap, gapSessions: sessions, refTw: ref };
}

// ---------------------------------------------------------------- 新聞（Google 新聞 RSS：只取標題與連結）
const ENT = { "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&apos;": "'" };
const unent = (s) => s.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1").replace(/&(amp|lt|gt|quot|#39|apos);/g, (m) => ENT[m]).trim();
export function parseRss(xml) {
  const out = [];
  for (const m of xml.matchAll(/<item>([\s\S]*?)<\/item>/g)) {
    const it = m[1], g = (tag) => { const x = it.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`)); return x ? unent(x[1]) : ""; };
    let title = g("title"), source = g("source") || g("News:Source"), link = g("link");
    if (source && title.endsWith(` - ${source}`)) title = title.slice(0, -(source.length + 3));
    const u = link.match(/[?&]url=([^&]+)/);  // Bing 的轉址連結 → 原文網址
    if (/bing\.com/.test(link) && u) { try { link = decodeURIComponent(u[1]); } catch (e) { /* 保留原連結 */ } }
    out.push({ title, link, source, t: Date.parse(g("pubDate")) || 0 });
  }
  return out;
}
export const NEWS_SOURCES = {
  google: (q) => `https://news.google.com/rss/search?q=${encodeURIComponent(q + " when:3d")}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant`,
  bing: (q) => `https://www.bing.com/news/search?q=${encodeURIComponent(q)}&format=rss&setlang=zh-TW&cc=TW&qft=sortbydate%3d%221%22`,
};
const UA = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36", "Accept-Language": "zh-TW,zh;q=0.9" };
// 依序嘗試各來源，第一個成功的就用；全部失敗才丟出錯誤（附每個來源的原因）
async function fetchNews(q) {
  const errs = [];
  for (const [name, url] of Object.entries(NEWS_SOURCES)) {
    try {
      const r = await fetch(url(q), { headers: UA });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const items = parseRss(await r.text());
      if (!items.length) throw new Error("沒有項目");
      return items.map((it) => ({ ...it, via: name }));
    } catch (e) { errs.push(`${name} ${e.message}`); }
  }
  throw new Error(errs.join("、"));
}
// 鉅亨網台股新聞（市場新聞的最後備援；公開 JSON 介面，格式未經官方保證）
export function parseCnyes(js) {
  return (js?.items?.data || []).map((x) => ({ title: String(x.title || ""), link: `https://news.cnyes.com/news/id/${x.newsId}`,
    source: "鉅亨網", t: (x.publishAt || 0) * 1000, via: "cnyes" }));
}
async function fetchCnyes() {
  const r = await fetch("https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?limit=30", { headers: UA });
  if (!r.ok) throw new Error(`cnyes HTTP ${r.status}`);
  const items = parseCnyes(await r.json());
  if (!items.length) throw new Error("cnyes 沒有項目");
  return items;
}
const norm = (s) => s.replace(/[\s「」『』"'、，,。.!！?？:：|｜\-－—]/g, "").slice(0, 40);
export function pickNews(lists, since, max) {
  const seen = new Set(), out = [];
  for (const it of lists.flat().sort((a, b) => b.t - a.t)) {
    const k = norm(it.title);
    if (!it.title || it.t < since || seen.has(k)) continue;
    seen.add(k);
    out.push(it);
    if (out.length >= max) break;
  }
  return out;
}
// 標題裡出現入選股或持股名稱的新聞；who: [{code, name, tag}]
export function relatedNews(items, who) {
  const out = [];
  for (const it of items) {
    const hit = who.find((w) => (w.name && it.title.includes(w.name)) || it.title.includes(w.code));
    if (hit) out.push({ ...it, tag: `${hit.tag}｜${hit.name}` });
  }
  return out;
}
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const li = (it) => `• <a href="${esc(it.link)}">${esc(it.title)}</a>${it.source ? `（${esc(it.source)}）` : ""}`;

// ---------------------------------------------------------------- 早報
export function composeBrief({ now, env, markets, news, picks, forFamily }) {
  const T = tw(now);
  const hol = isHoliday(T.ymd, env);
  const who = [...(picks?.picks || []).map((p) => ({ code: p.code, name: p.name, tag: "入選" })),
    ...(forFamily ? [] : (picks?.holdings || []).map((h) => ({ code: h.code, name: h.name, tag: "持股" })))];
  const gap = !markets.error && markets.gap?.length;
  const ref = markets.refTw, refMd = ref ? `${+ref.slice(4, 6)}/${+ref.slice(6)}` : "";
  const closedDays = ref ? (() => { let n = 0, d = addDays(ref, 1); while (d < T.ymd) { if (isHoliday(d, env)) n++; d = addDays(d, 1); } return n + (hol ? 1 : 0); })() : 0;
  const o = [gap && !hol ? `🌅 <b>連假後開盤特刊｜${T.md}（${WD[T.wd]}）</b>`
    : `🌅 <b>早安｜${T.md}（${WD[T.wd]}）</b>` + (hol ? "　今日台股休市" : "")];
  if (gap) {
    o.push(`台股上次收盤 ${refMd}，之後美股交易了 ${markets.gapSessions} 天${closedDays ? `（台股休市 ${closedDays} 天）` : ""}`, "",
      `<b>📊 自台股 ${refMd} 收盤以來累計</b>`, ...markets.gap);
  }
  if (markets.error) o.push(`⚠ 國際數字取得失敗：${esc(markets.error)}`);
  else {
    o.push(...(gap ? [""] : []), `<b>美股${gap ? "最近一天" : ""}</b>${markets.usDate ? `（${+markets.usDate.slice(4, 6)}/${+markets.usDate.slice(6)} ${markets.live ? "盤中" : "收盤"}）` : ""}`, ...markets.lines);
    for (const n of markets.notes) o.push(`⚠ ${esc(n)}`);
  }
  if (gap && picks?.picks?.length) {  // 休市期間，入選股逐檔的新聞與公告
    const evs = (picks._events || []).filter((e) => !forFamily || !e.mineOnly);
    o.push("", `<b>🛒 ${hol ? "下個交易日" : "今天"}要買的股票：${refMd} 收盤後的消息</b>`);
    for (const p of picks.picks) {
      const nw = relatedNews(news.all || [], [{ code: p.code, name: p.name, tag: "入選" }]).slice(0, 2);
      const ev = evs.filter((e) => e.code === p.code);
      const flag = ev.length ? "⚠️ " : nw.length ? "📰 " : "▫️ ";
      o.push(`${flag}<b>${esc(p.name || "")} ${p.code}</b>` + (ev.length || nw.length ? "" : "：沒有新聞或公告"));
      for (const e of ev) o.push(`　${esc(e.title)}`);
      for (const it of nw) o.push("　" + li(it).slice(2));
    }
    o.push("<i>⚠️＝有公告，開盤前請先看；📰＝有新聞。</i>");
  }
  const rel = relatedNews(news.all || [], gap ? who.filter((w) => w.tag !== "入選") : who).slice(0, 6);
  const used = new Set(rel.map((it) => norm(it.title)));  // 已列在「有關」的不再重複出現在下面
  const rest = (xs) => (xs || []).filter((it) => !used.has(norm(it.title)));
  if (gap && picks?.picks?.length) { if (rel.length) o.push("", "<b>📌 和你的持股有關</b>", ...rel.map((it) => `[${esc(it.tag)}] ` + li(it).slice(2))); }
  else if (who.length) {
    o.push("", `<b>📌 和${forFamily ? "入選股" : "你的入選股、持股"}有關</b>`);
    o.push(...(rel.length ? rel.map((it) => `[${esc(it.tag)}] ` + li(it).slice(2)) : ["沒有標題提到這些股票的新聞"]));
  }
  const pol = rest(news.policy), mkt = rest(news.market).filter((it) => !pol.some((x) => norm(x.title) === norm(it.title)));
  if (pol.length) o.push("", "<b>🏛 政策動態</b>", ...pol.map(li));
  if (mkt.length) o.push("", "<b>📰 市場新聞</b>", ...mkt.map(li));
  if (news.error) o.push("", `⚠ 新聞取得失敗：${esc(news.error)}`);
  if (!forFamily && picks && picks.date && picks.date < prevTradingDay(T.ymd, env))
    o.push("", `⚠ 昨晚的入選資料沒有同步（最後一次是 ${+picks.date.slice(4, 6)}/${+picks.date.slice(6)}），相關新聞與開盤核對可能不完整`);
  const via = [...new Set([...(news.policy || []), ...(news.market || []), ...rel].map((x) => ({ google: "Google 新聞", bing: "Bing 新聞", cnyes: "鉅亨網" })[x.via]).filter(Boolean))];
  o.push("", `<i>標題與連結${via.length ? "來自" + via.join("、") : "由新聞網站提供"}，內容以原文為準。</i>`);
  let text = o.join("\n");
  while (text.length > 4000 && o.length > 5) { o.splice(o.length - 3, 1); text = o.join("\n"); }
  return text;
}

async function gatherBrief(env, now, picks) {
  const q = {};
  let markets;
  try {
    await Promise.all(SYMBOLS.map(async ([s]) => { try { q[s] = await fetchQuote(s); } catch (e) { q[s] = null; } }));
    markets = Object.values(q).some(Boolean) ? formatMarkets(q, tw(now).ymd, picks?.tsmc_close, prevTradingDay(tw(now).ymd, env)) : { error: "全部來源都沒有回應" };
  } catch (e) { markets = { error: String(e) }; }
  const since = Date.parse(`${(() => { const p = prevTradingDay(tw(now).ymd, env); return `${p.slice(0, 4)}-${p.slice(4, 6)}-${p.slice(6)}`; })()}T13:30:00+08:00`);
  const news = {}, errs = [];
  const names = [...(picks?.picks || []), ...(picks?.holdings || [])].map((x) => `"${x.name || x.code}"`).slice(0, 12);
  const [pol, mkt, rel] = await Promise.allSettled([
    fetchNews(NEWS[0][1]),
    fetchNews(NEWS[1][1]).catch((e) => fetchCnyes().catch((e2) => { throw new Error(`${e.message}、${e2.message}`); })),
    names.length ? fetchNews(names.join(" OR ")) : Promise.resolve([]),
  ]);
  const ok = (r, lab) => { if (r.status === "fulfilled") return r.value; errs.push(`${lab}（${r.reason?.message || r.reason}）`); return []; };
  const P = ok(pol, "政策"), Mk = ok(mkt, "市場"), R = ok(rel, "相關");
  news.policy = pickNews([P], since, 5);
  news.market = pickNews([Mk], since, 6);
  news.all = pickNews([R, Mk, P], since, 60);
  if (errs.length) news.error = errs.join("；");
  return { markets, news };
}

// ---------------------------------------------------------------- 開盤核對（證交所即時報價）
export function evalOpen(p, q) {
  const num = (x) => (x == null || x === "-" || x === "" ? null : parseFloat(x));
  const o = num(q?.o), y = num(q?.y), u = num(q?.u);
  const head = `${p.name || ""} ${p.code}`;
  if (o == null) return { icon: "❔", line: `❔ ${head}｜尚未開出或查無報價` };
  const gap = y ? ` (${sgn((o / y - 1) * 100, 1)}%)` : "";
  const tail = `｜昨收 ${y ?? "—"}｜開盤 ${o}${gap}` + (p.buy_max != null ? `｜買進上限 ${p.buy_max}` : `｜停損 ${p.stop}｜目標 ${p.target}`);
  if (p.stop != null && o <= p.stop) return { icon: "⛔", line: `⛔ ${head}${tail}\n　開盤 ≤ 停損，照規則不買` };
  // v1.0.2：和電腦結算同一條規則（plan-v3）：開盤 ≥ 昨收 + 1 倍 ATR 就不追
  if (p.buy_max != null && o >= p.buy_max) return { icon: "⛔", line: `⛔ ${head}${tail}\n　開盤跳空超過 1 倍 ATR，照規則不買（不要追）` };
  if (p.target != null && o >= p.target) return { icon: "⛔", line: `⛔ ${head}${tail}\n　開盤 ≥ 目標，照規則不買` };
  if (u != null && o >= u) return { icon: "⚠️", line: `⚠️ ${head}${tail}\n　開盤漲停，整天鎖死就買不到` };
  return { icon: "✅", line: `✅ ${head}${tail}` };
}
export function mktPrefixes(m) {
  const s = String(m || "");
  if (/tpex|otc|櫃/i.test(s)) return ["otc"];
  if (/twse|tse|市/i.test(s)) return ["tse"];
  return ["tse", "otc"];
}
async function fetchMis(picks) {
  const ex = picks.flatMap((p) => mktPrefixes(p.market).map((m) => `${m}_${p.code}.tw`)).join("|");
  const home = await fetch("https://mis.twse.com.tw/stock/index.jsp", { headers: { "User-Agent": "Mozilla/5.0" } });
  const cookie = (home.headers.get("set-cookie") || "").split(";")[0];
  const r = await fetch(`https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=${encodeURIComponent(ex)}&json=1&delay=0&_=${Date.now()}`,
    { headers: { "User-Agent": "Mozilla/5.0", Referer: "https://mis.twse.com.tw/stock/index.jsp", Cookie: cookie } });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const js = await r.json();
  const m = {};
  for (const x of js.msgArray || []) if (x.c && (!m[x.c] || x.o !== "-")) m[x.c] = x;
  return m;
}
export function composeOpen({ now, picks, quotes, forFamily, error }) {
  const T = tw(now);
  const o = [`🔔 <b>開盤核對｜${T.md}（${WD[T.wd]}）${T.hm}</b>`, `昨晚（${+picks.date.slice(4, 6)}/${+picks.date.slice(6)}）入選 ${picks.picks.length} 檔，依交易計畫檢查開盤價：`, ""];
  if (error) o.push(`⚠ 即時報價取得失敗：${esc(error)}`);
  else for (const p of picks.picks) {
    const q = quotes[p.code];
    o.push(q && q.d && q.d !== T.ymd ? `❔ ${p.name} ${p.code}｜查無今日報價（最近是 ${q.d}）` : evalOpen(p, q).line);
  }
  o.push("", "✅ 在計畫範圍內　⛔ 照規則不買　⚠️ 注意");
  if (forFamily) o.push("請用自己的看盤 App 核對開盤價，數字不同請在群組說一聲。");
  return o.join("\n");
}

// ---------------------------------------------------------------- v24.6：觀察名單
// 電腦每晚同步：picks（昨晚入選）、finalists（決賽未入選）、positions（系統模擬部位）、holdings（你的持股，只給你）
export function watchList(picks) {
  // v24.9：只盯昨晚入選與你的持股（系統模擬部位、決賽股不再監控）
  const w = {};
  for (const p of picks?.picks || []) w[p.code] = { ...p, tag: "入選" };
  for (const h of picks?.holdings || []) if (!w[h.code]) w[h.code] = { ...h, tag: "你的持股", mineOnly: true };
  return w;
}
// CPU：免費方案每次 10 毫秒。實測 1.5MB 公告 JSON 解析約 2 毫秒（V8 的 JSON.parse 比逐一字串搜尋還快），所以直接解析。
const pickKey = (r, keys) => { for (const k of keys) { const v = r?.[k]; if (v != null && String(v).trim() !== "") return String(v).trim(); } return ""; };
const codeOf = (r, watch) => {
  const c = pickKey(r, ["公司代號", "SecuritiesCompanyCode", "Code", "證券代號", "有價證券代號"]);
  if (c) return watch[c] ? c : null;  // 有已知的代號欄位就只看它（省 CPU）；欄位名稱不認得才逐欄找
  for (const v of Object.values(r || {})) { const t = String(v ?? "").trim().split(/\s+/)[0]; if (watch[t]) return t; }
  return null;
};
export const hash = (s) => { let h = 2166136261; for (const ch of String(s)) { h ^= ch.codePointAt(0); h = Math.imul(h, 16777619); } return (h >>> 0).toString(36); };

// ---------------------------------------------------------------- v24.6：重大訊息（公開資訊觀測站當日公告）
// 上櫃（櫃買中心）的三個端點在 9/27 實測全部把 Cloudflare 的連線轉址到 /errors（擋連線），
// 所以雲端只抓上市；上櫃改由電腦每晚抓（ap/extras.py tpex_event_messages）。/api/diag 仍會測上櫃，方便日後確認是否解封。
export const MOPS = [["上市", "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"]];
export const TPEX_DIAG = [["重大訊息", "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O"],
  ["注意", "https://www.tpex.org.tw/openapi/v1/tpex_trading_warning_information"], ["處置", "https://www.tpex.org.tw/openapi/v1/tpex_disposal_information"]];
export function classify(title) {
  if (/停止買賣|暫停交易|停止交易|恢復交易|恢復買賣|終止上市|終止上櫃|下市|下櫃/.test(title)) return ["🔴", "交易狀態"];
  if (/減資|面額|分割|合併|股份轉換|公開收購|私募|增資/.test(title)) return ["🟠", "公司行為"];
  if (/財測|財務預測|訴訟|停工|停產|火災|爆炸|處分|取得.*資產|會計師|經營權|董事長|總經理|自結|虧損|違約|退票|跳票|裁罰|重整|破產|查核|更正/.test(title)) return ["⚠️", "重大訊息"];
  return ["📢", "公告"];
}
export function parseMops(text, watch) {
  const out = [];
  for (const r of JSON.parse(text)) {
    const code = codeOf(r, watch);
    if (!code) continue;
    const title = pickKey(r, ["主旨", "Subject", "主旨 "]) || pickKey(r, ["說明", "Description"]).slice(0, 80);
    const when = `${pickKey(r, ["發言日期", "SpokesDate", "出表日期"])} ${pickKey(r, ["發言時間", "SpokesTime"])}`.trim();
    out.push({ code, title, when, cat: classify(title) });
  }
  return out;
}
export function composeMops(ev, w) {
  const [icon, cat] = ev.cat;
  return `${icon} <b>${cat}｜${esc(w.name || "")} ${ev.code}</b>［${esc(w.tag)}］\n${esc(ev.title)}` +
    (ev.when ? `\n<i>公開資訊觀測站 ${esc(ev.when)}</i>` : "");
}

// ---------------------------------------------------------------- v24.6：注意股、處置股（盤後公告）
// 端點名稱來自官方 OpenAPI；上櫃「注意」的端點名稱尚未實測，/api/diag 會顯示每個端點是否可用。
export const WARN = [
  ["上市", "處置", "https://openapi.twse.com.tw/v1/announcement/punish"],
];
export function parseWarn(text, watch, kind) {
  const out = [];
  for (const r of JSON.parse(text)) {
    const code = codeOf(r, watch);
    if (!code) continue;
    // 欄位名稱：上市是英文（TradingInfoForAttention、ReasonsOfDisposition、DispositionPeriod、DispositionMeasures，9/27 實測），
    // 上櫃可能是中文，兩種都收
    const detail = Object.entries(r).filter(([k, v]) => /期間|期限|原因|條款|措施|內容|處置|注意|累計|起|迄|Attention|Reason|Period|Measure|Condition|Content|Detail/i.test(k)
      && !/^(Code|Name|Number|Date|ClosingPrice|PE)$/i.test(k) && String(v ?? "").trim())
      .map(([k, v]) => `${k}：${String(v).trim()}`).join("｜").slice(0, 200);
    out.push({ code, kind, detail });
  }
  return out;
}
export function composeWarn(ev, w) {
  const icon = ev.kind === "處置" ? "🔴" : "⚠️";
  const note = ev.kind === "處置" ? "處置期間改為分盤撮合，可能要預收款券；plan-v2「開盤價買進」的假設會失真。" : "連續被列注意，可能進一步被處置。";
  return `${icon} <b>${ev.kind}股｜${esc(w.name || "")} ${ev.code}</b>［${esc(w.tag)}］\n${esc(ev.detail || "（公告沒有細節欄位）")}\n${note}`;
}

// 事件監控共用：抓、比對、去重（KV 只在真的有事件時讀寫）、推播；失敗記在 err:日期:來源，隔天早報告訴你
async function recordErr(env, ymd, src, msg) {
  const k = `err:${ymd}:${src}`;
  if (!(await env.BRIEF.get(k))) await env.BRIEF.put(k, String(msg).slice(0, 200), { expirationTtl: 3 * 86400 });
}
export const SEVERE = /處置|停止買賣|暫停交易|停止交易|恢復交易|恢復買賣|終止上市|終止上櫃|減資/;
async function sendEvent(env, key, text, w, ttlDays, dry) {
  if (!SEVERE.test(text)) return false;  // v24.9：只推最嚴重的（處置、停止交易、減資），其他不推
  if (await env.BRIEF.get(key)) return false;
  if (!dry) {
    await tg(env, env.CHAT_ID, text);  // 家人群組不收事件
    await env.BRIEF.put(key, "1", { expirationTtl: ttlDays * 86400 });
    try {  // 事件紀錄（最近 40 則、保留 10 天）：連假後開盤特刊用來列出休市期間的公告
      const log = JSON.parse((await env.BRIEF.get("evlog")) || "[]");
      log.push({ t: Date.now(), code: w.code, title: text.split("\n").slice(0, 2).join("：").replace(/<[^>]+>/g, "").slice(0, 120), mineOnly: !!w.mineOnly });
      await env.BRIEF.put("evlog", JSON.stringify(log.slice(-40)), { expirationTtl: 10 * 86400 });
    } catch (e) { /* 不影響推播 */ }
  }
  return true;
}
export async function runEvents(env, now = Date.now(), dry = false, which = "mops") {
  const T = tw(now);
  const picks = await getPicks(env);
  const watch = watchList(picks);
  if (!Object.keys(watch).length) return { skipped: "沒有觀察名單" };
  const out = [];
  const srcs = which === "mops" ? MOPS.map(([m, u]) => [`重大訊息${m}`, u, null]) : WARN.map(([m, k, u]) => [`${k}${m}`, u, k]);
  for (const [name, url, kind] of srcs) {
    let text;
    try {
      // 不跟隨轉址：櫃買中心擋連線時會無限轉址到 /errors，每次轉址都算一個對外請求，會把免費方案的 50 個額度用光
      const r = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0", Accept: "application/json" }, redirect: "manual" });
      if (r.status >= 300 && r.status < 400) throw new Error(`被轉址到 ${(r.headers.get("location") || "").slice(0, 60)}（來源擋連線）`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      text = await r.text();
    } catch (e) { await recordErr(env, T.ymd, name, e.message || e); continue; }
    let evs;
    try { evs = kind ? parseWarn(text, watch, kind) : parseMops(text, watch); } catch (e) { await recordErr(env, T.ymd, name, `格式不符：${e.message}`); continue; }
    for (const ev of evs) {
      const w = watch[ev.code];
      const key = kind ? `ev:${kind}:${ev.code}:${kind === "注意" ? T.ymd : hash(ev.detail)}` : `ev:mops:${ev.code}:${hash(ev.when + ev.title)}`;
      const text2 = kind ? composeWarn(ev, w) : composeMops(ev, w);
      if (await sendEvent(env, key, text2, w, kind === "處置" ? 40 : 7, dry)) out.push(text2);
    }
  }
  return { sent: out };
}

// ---------------------------------------------------------------- v24.6：13:35 收盤快報（系統部位＋今天開盤買進的入選股）
const num = (x) => (x == null || x === "-" || x === "" ? null : parseFloat(x));
const pctTxt = (a, b, d = 1) => (a != null && b ? `${sgn((a / b - 1) * 100, d)}%` : "—");
export function closeLine(p, q, isNew) {
  const z = num(q?.z) ?? num(q?.pz), y = num(q?.y), h = num(q?.h), l = num(q?.l), o = num(q?.o);
  const head = `${esc(p.name || "")} ${p.code}`;
  if (z == null) return `❔ ${head}｜查無收盤價`;
  let buy = p.buy, note = [];
  if (isNew) {
    const ev = evalOpen(p, q);
    if (ev.icon !== "✅") return `⏸ ${head}｜今天照規則沒買（${ev.line.split("\n")[1]?.trim() || "開盤不在計畫範圍"}）`;
    buy = o;
  }
  const stop = p.stop;
  if (buy && h != null && h >= buy * 1.1 && !p.trailing) note.push("🎯 盤中碰到 +10%");
  if (!isNew && p.disaster != null && l != null && l <= p.disaster) note.push("🚨 盤中觸及災難停損價");
  if (stop != null && z < stop) note.push("⚠ 收盤低於停損，照規則明天開盤賣");
  const tag = isNew ? "今天買進" : `第 ${(p.hold_days || 0) + 1} 天`;
  return `${note.length ? "❗" : "▪️"} ${head}｜${tag}｜收 ${z}（${pctTxt(z, y)}）｜報酬 ${pctTxt(z, buy)}｜距停損 ${stop ? pctTxt(stop, z) : "—"}` +
    (note.length ? `\n　${note.join("；")}` : "");
}
export function composeClose({ now, picks, quotes, forFamily, error }) {
  const T = tw(now);
  const o = [`🔔 <b>收盤快報｜${T.md}（${WD[T.wd]}）</b>`];
  if (error) { o.push(`⚠ 即時報價取得失敗：${esc(error)}`); return o.join("\n"); }
  const pos = picks.positions || [], nw = picks.picks || [];
  if (nw.length) o.push("", `<b>昨晚入選（今天開盤買進）</b>`, ...nw.map((p) => closeLine(p, quotes[p.code], true)));
  if (pos.length) o.push("", `<b>系統部位</b>`, ...pos.map((p) => closeLine(p, quotes[p.code], false)));
  if (!forFamily && picks.holdings?.length) o.push("", "<b>你的持股</b>", ...picks.holdings.map((h) => {
    const q = quotes[h.code], z = num(q?.z) ?? num(q?.pz);
    return `▪️ ${esc(h.name || "")} ${h.code}｜收 ${z ?? "—"}（${pctTxt(z, num(q?.y))}）`;
  }));
  o.push("", "<i>以證交所即時報價的最後成交價計算；是否出場以晚上的結算為準。</i>");
  return o.join("\n");
}
export async function runClose(env, now = Date.now(), dry = false) {
  const T = tw(now);
  if (isHoliday(T.ymd, env)) return { skipped: "休市" };
  const picks = await getPicks(env);
  if (!picks || (!picks.picks?.length && !picks.positions?.length)) return { skipped: "沒有入選也沒有系統部位" };
  if (picks.date !== prevTradingDay(T.ymd, env)) return { skipped: `同步資料不是前一個交易日（${picks.date}）` };
  const all = [...(picks.picks || []), ...(picks.positions || []), ...(picks.holdings || [])];
  const uniq = [...new Map(all.map((p) => [p.code, p])).values()];
  let quotes = {}, error = null;
  try { quotes = await fetchMis(uniq); } catch (e) { error = String(e); }
  const mine = composeClose({ now, picks, quotes, forFamily: false, error });
  const fam = composeClose({ now, picks, quotes, forFamily: true, error });
  if (!dry) { await tg(env, env.CHAT_ID, mine); await tg(env, env.FAMILY_CHAT_ID, fam); }
  return { mine, fam };
}

// ---------------------------------------------------------------- v24.6：週日 20:00 下週預覽
// FOMC 會議日期（美國時間，聯準會公布）。決議在第二天美東 14:00，台灣時間隔天清晨。每年底更新。
export const FOMC = [["20261027", "20261028"], ["20261208", "20261209"], ["20270126", "20270127"], ["20270316", "20270317"],
  ["20270427", "20270428"], ["20270608", "20270609"]];
const mdOf = (ymd) => `${+ymd.slice(4, 6)}/${+ymd.slice(6)}`;
const wdOf = (ymd) => WD[new Date(`${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6)}T00:00:00Z`).getUTCDay()];
export function composePreview({ now, env, picks, forFamily }) {
  const T = tw(now);
  const mon = addDays(T.ymd, 1), fri = addDays(T.ymd, 5);
  const days = [0, 1, 2, 3, 4].map((k) => addDays(mon, k));
  const open = days.filter((d) => !isHoliday(d, env)), closed = days.filter((d) => isHoliday(d, env));
  const o = [`🗓 <b>下週預覽｜${mdOf(mon)}～${mdOf(fri)}</b>`, "",
    `台股交易日：${open.length ? open.map((d) => `${mdOf(d)}（${wdOf(d)}）`).join("、") : "無"}` + (closed.length ? `｜休市 ${closed.map(mdOf).join("、")}` : "")];
  const ev = [];
  if (days.some((d) => +d.slice(6) <= 10)) ev.push("上市櫃公司公布上月營收（期限每月 10 日）；觀察名單有異常會推播");
  for (const [a, b] of FOMC) if (b >= mon && a <= fri) ev.push(`FOMC ${mdOf(a)}～${mdOf(b)}（美國時間），決議約台灣 ${mdOf(addDays(b, 1))} 清晨公布`);
  const firstOfMonth = open.find((d) => { let x = addDays(d, -1); while (x.slice(0, 6) === d.slice(0, 6)) { if (!isHoliday(x, env)) return false; x = addDays(x, -1); } return true; });
  if (firstOfMonth) ev.push(`${mdOf(firstOfMonth)} 晚上：上個月的機率準確度月報`);
  o.push("", "<b>已知事件</b>", ...(ev.length ? ev.map((x) => `• ${x}`) : ["• 沒有已排定的大事件"]));
  const pos = picks?.positions || [];
  o.push("", `<b>系統部位</b>（${picks?.date ? mdOf(picks.date) : "—"} 收盤）${pos.length} 檔`);
  for (const p of pos) {
    const r = p.last_close && p.buy ? `${sgn((p.last_close / p.buy - 1) * 100, 1)}%` : "—";
    const nx = p.exit_pending ? `｜⚠ 週一開盤賣：${esc(p.exit_pending)}` : p.trailing ? "｜移動停利中" :
      (p.max_gain ?? 0) < 5 && (p.hold_days ?? 0) < 10 ? `｜發動檢查剩 ${10 - (p.hold_days ?? 0)} 天` : "";
    o.push(`▪️ ${esc(p.name || "")} ${p.code}｜第 ${p.hold_days ?? "—"} 天｜${r}${nx}`);
  }
  if (!pos.length) o.push("目前沒有");
  if (!forFamily && picks?.holdings?.length) o.push("", `你的持股 ${picks.holdings.length} 檔：${picks.holdings.map((h) => esc(h.name || h.code)).join("、")}`);
  o.push("", "<i>除權息日程、美國 CPI／非農等數據日期沒有穩定的免費來源，這裡不列。</i>");
  return o.join("\n");
}
export async function runPreview(env, now = Date.now(), dry = false) {
  const picks = await getPicks(env);
  const mine = composePreview({ now, env, picks, forFamily: false });
  const fam = composePreview({ now, env, picks, forFamily: true });
  if (!dry) { await tg(env, env.CHAT_ID, mine); await tg(env, env.FAMILY_CHAT_ID, fam); }
  return { mine, fam };
}

// ---------------------------------------------------------------- Telegram、KV、排程
async function tg(env, chat, text) {
  if (!chat) return;
  await fetch(`https://api.telegram.org/bot${env.BOT_TOKEN}/sendMessage`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ chat_id: chat, text, parse_mode: "HTML", disable_web_page_preview: true }),
  });
}
const RAW = "https://raw.githubusercontent.com/xchixmeng-svg/alphapilot-market-data-v2/main/data/extra/";
async function getJson(path, now) {
  const r = await fetch(RAW + path + "?t=" + (now || Date.now()));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
// 11/22：選股改在 GitHub 算（data/extra/picks/latest.json）；你的實際持股仍用電腦同步到 KV 的那份
async function getPicks(env) {
  let kv = null;
  try { const s = await env.BRIEF.get("picks"); kv = s ? JSON.parse(s) : null; } catch (e) { kv = null; }
  try {
    const g = await getJson("picks/latest.json");
    return { date: g.date, picks: g.picks || [], holdings: kv?.holdings || [], tsmc_close: g.tsmc_close ?? kv?.tsmc_close, _gh: g };
  } catch (e) { return kv; }
}

export async function runBrief(env, now = Date.now(), dry = false) {
  const picks = await getPicks(env);
  if (picks) {  // 上次台股收盤之後的事件（連假後開盤特刊用）
    const since = Date.parse(`${(() => { const p = prevTradingDay(tw(now).ymd, env); return `${p.slice(0, 4)}-${p.slice(4, 6)}-${p.slice(6)}`; })()}T13:30:00+08:00`);
    try { picks._events = JSON.parse((await env.BRIEF.get("evlog")) || "[]").filter((e) => e.t >= since); } catch (e) { picks._events = []; }
  }
  const { markets, news } = await gatherBrief(env, now, picks);
  let mine = composeBrief({ now, env, markets, news, picks, forFamily: false });
  try {  // 昨天事件監控的來源失敗（只告訴你；家人版不列）
    const y = addDays(tw(now).ymd, -1);
    const ks = await env.BRIEF.list({ prefix: `err:${y}:` });
    const errs = [];
    for (const k of ks.keys || []) errs.push(`${k.name.split(":")[2]}（${String(await env.BRIEF.get(k.name)).replace(/被轉址到 \S+（來源擋連線）/, "來源擋連線")}）`);
    if (errs.length) mine += `\n\n⚠ 昨天事件監控有來源失敗：${esc(errs.join("、"))}`;
  } catch (e) { /* 不影響早報 */ }
  // 11/22：燈號放在早報最上面，07:00 一起發（只發給你）
  const T0 = tw(now); let lightsHead = "", lightsFresh = false, lightsDate = "";
  try {
    const j = await getJson("lights/latest.json", now);
    if (j.text) {
      lightsFresh = j.stage === "morning" && String(j.morning_at || "").slice(0, 10).replaceAll("-", "") === T0.ymd;
      lightsDate = j.date;
      lightsHead = esc(j.text) + (!lightsFresh && !isHoliday(T0.ymd, env) ? "\n（早上版還沒算好，先附昨晚的燈號；算好後會補發一次）" : "");
    }
  } catch (e) { lightsHead = `⚠ 燈號讀取失敗：${esc(String(e.message || e))}`; }
  const gp = picks?._gh;
  if (gp && gp.date === prevTradingDay(T0.ymd, env) && gp.notify)
    lightsHead += `\n\n🛒 昨晚選股有買賣單（買 ${(gp.picks || []).length} 檔、賣 ${gp.sellall ? "全部" : (gp.sells || []).length + " 檔"}），開盤照單執行`;
  const full = lightsHead ? lightsHead + "\n\n━━━━━━━━━━\n\n" + mine : mine;
  const fam = composeBrief({ now, env, markets, news, picks, forFamily: true });
  if (!dry) {
    if (full.length <= 4096) await tg(env, env.CHAT_ID, full);
    else { await tg(env, env.CHAT_ID, lightsHead); await tg(env, env.CHAT_ID, mine); }
    if (lightsFresh && lightsDate) await env.BRIEF.put("lights:" + lightsDate, "1", { expirationTtl: 7 * 86400 });
    await tg(env, env.FAMILY_CHAT_ID, fam);
    await env.BRIEF.put(`brief:${tw(now).ymd}`, JSON.stringify({ date: tw(now).ymd, text: fam }), { expirationTtl: 60 * 86400 });
  }
  return { mine: full, fam };
}

export async function runOpen(env, now = Date.now(), dry = false) {
  const T = tw(now);
  if (isHoliday(T.ymd, env)) return { skipped: "休市" };
  const picks = await getPicks(env);
  if (!picks || !picks.picks?.length) return { skipped: "昨晚沒有入選" };
  if (picks.date !== prevTradingDay(T.ymd, env)) return { skipped: `入選資料不是前一個交易日（${picks.date}）` };
  let quotes = {}, error = null;
  try { quotes = await fetchMis(picks.picks); } catch (e) { error = String(e); }
  const mine = composeOpen({ now, picks, quotes, forFamily: false, error });
  const fam = composeOpen({ now, picks, quotes, forFamily: true, error });
  if (!dry) await tg(env, env.CHAT_ID, mine);  // v24.9：開盤核對只給你，家人群組不收
  return { mine, fam };
}

// ---------------------------------------------------------------- v24.6：排程分派
// 定時工作：到點（或 30 分鐘內補跑）且今天還沒做過才做；「做過」記在 KV done:工作:日期（每天幾次寫入）。
// 舊的兩個排程（UTC 23:00、01:05）也照樣對應到 07:00 早報、09:05 開盤核對。

// ---------------------------------------------------------------- 資料準時叫醒（2026-11-19 加入）
// GitHub Actions 自己的排程常延遲好幾小時，所以 17:30 由這裡叫 GitHub「現在立刻抓」（repository_dispatch，通常 1 分鐘內開始）；
// 18:00 先檢查今天的資料（價量 manifest + 雲端快照 status）是否已進 repo，沒有才再叫一次。
// 需要 Secret：GH_TOKEN（GitHub fine-grained token，只給 alphapilot-market-data-v2 的 Contents: Read and write）
const DATA_REPO = "xchixmeng-svg/alphapilot-market-data-v2";
async function dataReady(ymd) {
  const dash = ymd.slice(0, 4) + "-" + ymd.slice(4, 6) + "-" + ymd.slice(6);
  const raw = "https://raw.githubusercontent.com/" + DATA_REPO + "/main/data";
  const a = await fetch(raw + "/" + dash + "/manifest.json", { cf: { cacheTtl: 0 } });
  const b = await fetch(raw + "/extra/_status/" + ymd + ".json", { cf: { cacheTtl: 0 } });
  return a.ok && b.ok;
}
export async function runDataWake(env, now = Date.now(), dry = false) {
  const t = tw(now);
  if (isHoliday(t.ymd, env)) return { skipped: t.ymd + " 休市，不抓資料" };
  if (t.hm >= "17:50" && (await dataReady(t.ymd))) return { skipped: t.ymd + " 資料已到齊，不用再叫" };
  if (dry) return { skipped: "（測試）會叫 GitHub 開始抓 " + t.ymd };
  // 沒設定金鑰也沒關係：資料 repo 的「17:32 鬧鐘」排程（data_wake_1730.yml）會用 GitHub 內建金鑰自己叫醒，這裡只是加速備援
  if (!env.GH_TOKEN) return { skipped: "沒有 GH_TOKEN，交給 GitHub 的 17:32 鬧鐘" };
  const r = await fetch("https://api.github.com/repos/" + DATA_REPO + "/dispatches", {
    method: "POST",
    headers: { authorization: "Bearer " + env.GH_TOKEN, accept: "application/vnd.github+json",
               "user-agent": "alphapilot-brief-worker", "content-type": "application/json" },
    body: JSON.stringify({ event_type: "fetch-now", client_payload: { date: t.ymd, at: t.hm } }),
  });
  if (r.status !== 204) throw new Error("GitHub 回應 " + r.status + "：" + (await r.text()).slice(0, 200));
  return { skipped: "已叫 GitHub 開始抓 " + t.ymd + "（" + t.hm + "）" };
}

export const JOBS = [
  { id: "brief", at: "07:00", when: (T, env) => T.wd >= 1 && T.wd <= 6, run: (e, n) => runBrief(e, n) },
  { id: "open", at: "09:05", when: (T, env) => !isHoliday(T.ymd, env), run: (e, n) => runOpen(e, n) },
  // 11/21：wake1/wake2 停用（沒有 GH_TOKEN 必定失敗；GitHub 的 data_wake_1730 已用內建金鑰負責叫醒）
  // v24.9：收盤快報、週日預覽停用（只記錄的資訊不推播）；/api/run 仍可手動查看
];
const mins = (hm) => +hm.slice(0, 2) * 60 + +hm.slice(3, 5);
export function plan(now, env = {}) {
  const T = tw(now), m = mins(T.hm);
  const due = JOBS.filter((j) => j.when(T, env) && m >= mins(j.at) && m < mins(j.at) + 30).map((j) => j.id);
  // 重大訊息：每天 07:30～21:30 每 10 分鐘；注意／處置：交易日 15:30～20:00 每 30 分鐘
  if (m >= mins("07:30") && m <= mins("21:30") && m % 10 === 0) due.push("mops");
  if (!isHoliday(T.ymd, env) && m >= mins("15:30") && m <= mins("20:00") && m % 30 === 0) due.push("warn");
  return due;
}
export 
// ---------- 探底燈號早上推播（v25.0）：GitHub 早上 07:05 產生 morning 版，07:50～11:00 之間推一次給自己（不推家族群組）
async function lightsPush(env, now) {
  try {
    const T = tw(now);
    if (isHoliday(T.ymd, env) || T.hm < "07:00" || T.hm > "11:00") return;
    if (!(await env.BRIEF.get("done:brief:" + T.ymd))) return;   // 11/22：07:00 早報先發（燈號在早報最上面）；早上版晚到才在這裡補發
    const r = await fetch("https://raw.githubusercontent.com/xchixmeng-svg/alphapilot-market-data-v2/main/data/extra/lights/latest.json?t=" + now);
    if (!r.ok) return;
    const j = await r.json();
    if (j.stage !== "morning" || !j.text || !j.morning_at) return;
    if (j.morning_at.slice(0, 10).replaceAll("-", "") !== T.ymd) return;   // 今天早上產生的才推
    const k = "lights:" + j.date;
    if (await env.BRIEF.get(k)) return;
    await env.BRIEF.put(k, "1", { expirationTtl: 7 * 86400 });
    await tg(env, env.CHAT_ID, esc(j.text));
  } catch (e) { /* 下一個 5 分鐘再試 */ }
}



// 11/23：GitHub 排程整段停擺時的後手。交易日 18:30、19:30、20:30、21:30、22:30 檢查今天資料，沒進來就叫 GitHub 立刻抓（需要 GH_TOKEN）
async function autoWake(env, now) {
  try {
    const T = tw(now);
    if (isHoliday(T.ymd, env) || !env.GH_TOKEN) return;
    const hh = +T.hm.slice(0, 2), mm = +T.hm.slice(3, 5);
    if (hh < 18 || hh > 22 || mm < 30 || mm >= 45) return;
    const k = `autowake:${T.ymd}:${hh}`;
    if (await env.BRIEF.get(k)) return;
    await env.BRIEF.put(k, "1", { expirationTtl: 2 * 86400 });
    if (await dataReady(T.ymd)) return;
    await ghWake(env, now, "auto");
    await env.BRIEF.put("refresh:last", JSON.stringify({ at: now, by: "auto" }), { expirationTtl: 7 * 86400 });
  } catch (e) { await tg(env, env.CHAT_ID, `⚠️ 自動補抓資料失敗：${esc(String(e.message || e))}`); }
}
// 11/24：Cloudflare 當鬧鐘。GitHub 自己的排程常延遲好幾小時，所以每個固定時間由這裡直接叫 GitHub 立刻跑（通常 1 分鐘內開始）。
// GitHub 原本的排程保留當備援；同一件事跑兩次沒關係（程式會覆蓋成同一個結果）。只在台股交易日叫。
const CLOCK = [
  { at: "06:30", type: "lights-morning",   what: "早上燈號（含美股資料）" },
  { at: "06:50", type: "lights-morning",   what: "早上燈號（備援）" },
  { at: "13:10", type: "lights-afternoon", what: "下午燈號" },
  { at: "13:40", type: "lights-afternoon", what: "下午燈號（備援）" },
  { at: "17:30", type: "twd-now",          what: "台幣收盤價" },
  { at: "17:35", type: "fetch-now",        what: "晚上抓資料" },
  { at: "21:45", type: "lights-night",     what: "晚上燈號（融資公布後）" },
  { at: "22:15", type: "lights-night",     what: "晚上燈號（備援）" },
];
async function clockWake(env, now) {
  const T = tw(now);
  if (!env.GH_TOKEN || isHoliday(T.ymd, env) || T.wd < 1 || T.wd > 5) return;
  const m = mins(T.hm);
  for (const c of CLOCK) {
    const d = m - mins(c.at);
    if (d < 0 || d >= 15) continue;                       // 到點後 15 分鐘內（5 分鐘排程至少碰到 3 次）
    const k = `clock:${T.ymd}:${c.at}`;
    if (await env.BRIEF.get(k)) continue;
    try {
      await ghDispatch(env, c.type, { date: T.ymd, at: T.hm, by: "clock" });
      await env.BRIEF.put(k, "1", { expirationTtl: 2 * 86400 });
      if (c.type === "fetch-now") await env.BRIEF.put("refresh:last", JSON.stringify({ at: now, by: "clock" }), { expirationTtl: 7 * 86400 });
    } catch (e) {
      if (d >= 10) {                                       // 最後一次機會也失敗才通知，避免一直吵
        await env.BRIEF.put(k, "fail", { expirationTtl: 2 * 86400 });
        await tg(env, env.CHAT_ID, `⚠️ ${c.at} 叫 GitHub 跑「${c.what}」失敗：${esc(String(e.message || e))}\n可以到 App 系統頁按「立即抓資料」，或跟 Claude 說`);
      }
    }
  }
}
async function ghDispatch(env, type, payload) {
  const r = await fetch("https://api.github.com/repos/" + DATA_REPO + "/dispatches", {
    method: "POST",
    headers: { authorization: "Bearer " + env.GH_TOKEN, accept: "application/vnd.github+json",
               "user-agent": "alphapilot-brief-worker", "content-type": "application/json" },
    body: JSON.stringify({ event_type: type, client_payload: payload }),
  });
  if (r.status !== 204) throw new Error("GitHub 回應 " + r.status + "：" + (await r.text()).slice(0, 200));
}
async function ghWake(env, now, by) {
  const t = tw(now);
  const r = await fetch("https://api.github.com/repos/" + DATA_REPO + "/dispatches", {
    method: "POST",
    headers: { authorization: "Bearer " + env.GH_TOKEN, accept: "application/vnd.github+json",
               "user-agent": "alphapilot-brief-worker", "content-type": "application/json" },
    body: JSON.stringify({ event_type: "fetch-now", client_payload: { date: t.ymd, at: t.hm, by } }),
  });
  if (r.status !== 204) throw new Error("GitHub 回應 " + r.status + "：" + (await r.text()).slice(0, 200));
}
// 11/22：晚上選股推播（GitHub 21:37 後算好；有買賣單才推；內容變了會再推一次「更新」）
// 11/24：晚上結論分兩次。18:30 起資料一算好就先推「初步結論」（明天進不進場、還差哪盞燈）；
// 21:30 融資公布、21:45 重算後，只有結論改變（燈數、狀態、要買賣的股票）才再推一次，沒變就不吵。
async function picksPush(env, now) {
  try {
    const T = tw(now);
    if (T.hm < "18:30" || T.hm > "23:55" || isHoliday(T.ymd, env) || T.wd < 1 || T.wd > 5) return;
    const L = await getJson("lights/latest.json", now);
    if (!L || L.date !== T.ymd || L.stage !== "data" || !L.text) return;
    let P = null;
    try { P = await getJson("picks/latest.json", now); } catch (e) { P = null; }
    if (P && P.date !== T.ymd) P = null;
    if (!P && T.hm < "19:30") return;                       // 選股通常和燈號同時好；19:30 前再等一下
    const orders = P && P.notify && P.text ? P.text : "";
    const key = JSON.stringify([L.key && L.key.state, L.key && L.key.action, !!L.signal, L.lamps_on, L.green, hash(orders)]);
    const final = !!L.margin_ok;
    const k = "eve:" + T.ymd;
    let st = null; try { st = JSON.parse((await env.BRIEF.get(k)) || "null"); } catch (e) { st = null; }
    if (st && (st.final || (!final && st.key === key))) return;
    if (st && final && st.key === key) { await env.BRIEF.put(k, JSON.stringify({ key, final: true }), { expirationTtl: 7 * 86400 }); return; }
    const lines = L.text.split("\n");
    const head = []; for (const x of lines) { if (!x.trim()) break; head.push(x); }
    const miss = lines.find((x) => x.startsWith("還差"));
    const md = T.ymd.slice(4, 6).replace(/^0/, "") + "/" + T.ymd.slice(6).replace(/^0/, "");
    const title = !st ? (final ? `🌙 今晚結論 ${md}` : `🌙 今晚結論 ${md}（初步）`) : `🌙 今晚結論 ${md}（融資公布後有變）`;
    const msg = [title, ...head.filter((x) => !x.startsWith("📊"))];
    if (miss) msg.push(miss);
    msg.push(orders ? "\n" + orders : (P ? "\n明天沒有要買賣的" : ""));
    if (!final) msg.push("\n（融資 21:30 公布，21:45 重算；結論有變才會再通知你）");
    await tg(env, env.CHAT_ID, esc(msg.filter((x) => x !== "").join("\n")));
    await env.BRIEF.put(k, JSON.stringify({ key, final }), { expirationTtl: 7 * 86400 });
  } catch (e) { /* 下一個 5 分鐘再試 */ }
}

async function dispatch(env, now) {
  await lightsPush(env, now);
  await picksPush(env, now);
  await clockWake(env, now);
  await autoWake(env, now);
  const T = tw(now), done = [];
  for (const id of plan(now, env)) {
    if (id === "mops" || id === "warn") { await runEvents(env, now, false, id); done.push(id); continue; }
    const k = `done:${id}:${T.ymd}`;
    if (await env.BRIEF.get(k)) continue;
    await env.BRIEF.put(k, "1", { expirationTtl: 3 * 86400 });  // 先標記再做：失敗時不會每 5 分鐘重發一次
    const j = JOBS.find((x) => x.id === id);
    try { await j.run(env, now); } catch (e) { await tg(env, env.CHAT_ID, `⚠️ <b>Worker 工作 ${id} 失敗</b>\n${esc(String(e))}`); }
    done.push(id);
  }
  return done;
}

const json = (o, s = 200) => new Response(JSON.stringify(o), { status: s, headers: { "content-type": "application/json; charset=utf-8" } });

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const secret = req.headers.get("X-AP-Secret") || url.searchParams.get("secret");
    if (url.pathname === "/health") return json({ ok: true });
    if (url.pathname === "/api/refresh" || url.pathname === "/api/refresh-status") {
      const cors = { "access-control-allow-origin": "https://alphapilot-app.pages.dev", "access-control-allow-methods": "GET,POST,OPTIONS", "access-control-allow-headers": "content-type" };
      const cj = (o, s2 = 200) => new Response(JSON.stringify(o), { status: s2, headers: { "content-type": "application/json; charset=utf-8", ...cors } });
      if (req.method === "OPTIONS") return new Response(null, { headers: cors });
      const nowT = Date.now(), T = tw(nowT);
      let last = null; try { last = JSON.parse((await env.BRIEF.get("refresh:last")) || "null"); } catch (e) { last = null; }
      const ready = await dataReady(T.ymd).catch(() => false);
      const base = { today: T.ymd, holiday: isHoliday(T.ymd, env), data_ready: ready, last, can: !!env.GH_TOKEN };
      if (url.pathname === "/api/refresh-status" || req.method !== "POST") return cj(base);
      if (!env.GH_TOKEN) return cj({ ...base, ok: false, msg: "Worker 還沒有 GH_TOKEN，暫時只能到 GitHub 手動按 Run workflow" });
      if (last && nowT - last.at < 10 * 60 * 1000) return cj({ ...base, ok: false, msg: `10 分鐘內已經叫過（${tw(last.at).hm}），稍等再按` });
      try { await ghWake(env, nowT, "app"); } catch (e) { return cj({ ...base, ok: false, msg: String(e.message || e) }); }
      const rec = { at: nowT, by: "app" };
      await env.BRIEF.put("refresh:last", JSON.stringify(rec), { expirationTtl: 7 * 86400 });
      return cj({ ...base, last: rec, ok: true, msg: `已叫 GitHub 開始抓（${T.hm}），約 3～5 分鐘後完成` });
    }
    if (!env.SECRET || secret !== env.SECRET) return json({ error: "unauthorized" }, 401);
    if (url.pathname === "/api/picks" && req.method === "POST") {
      const body = await req.json();
      await env.BRIEF.put("picks", JSON.stringify(body));
      return json({ ok: true, picks: (body.picks || []).length });
    }
    if (url.pathname === "/api/briefs") {
      const ks = await env.BRIEF.list({ prefix: "brief:" });
      const out = [];
      for (const k of ks.keys.slice(-14)) { const v = await env.BRIEF.get(k.name); if (v) out.push(JSON.parse(v)); }
      return json({ briefs: out });
    }
    if (url.pathname === "/api/diag") {  // 瀏覽器打開 /api/diag?secret=... 看每個外部來源是否可用
      const out = {};
      const probe = async (name, fn) => { const t0 = Date.now(); try { out[name] = { ok: true, detail: await fn(), ms: Date.now() - t0 }; } catch (e) { out[name] = { ok: false, error: String(e.message || e), ms: Date.now() - t0 }; } };
      await probe("yahoo_quote", async () => { const q = await fetchQuote("^SOX"); if (!q) throw new Error("格式不符"); return `費半 ${q.last}（${q.date}）`; });
      for (const [name, u] of Object.entries(NEWS_SOURCES)) await probe(`news_${name}`, async () => {
        const r = await fetch(u("台積電"), { headers: UA }); if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const it = parseRss(await r.text()); if (!it.length) throw new Error("沒有項目"); return `${it.length} 則，例：${it[0].title}`; });
      await probe("news_cnyes", async () => { const it = await fetchCnyes(); return `${it.length} 則，例：${it[0].title}`; });
      await probe("twse_mis", async () => { const m = await fetchMis([{ code: "2330", market: "TWSE" }]); const x = m["2330"]; if (!x) throw new Error("沒有資料"); return `2330 日期 ${x.d} 開盤 ${x.o} 昨收 ${x.y}`; });
      for (const [m, u] of MOPS) await probe(`mops_${m}`, async () => { const r = await fetch(u, { headers: { "User-Agent": "Mozilla/5.0" }, redirect: "manual" }); if (r.status >= 300 && r.status < 400) throw new Error(`被轉址到 ${r.headers.get("location")}（來源擋連線）`); if (!r.ok) throw new Error(`HTTP ${r.status}`); const t = await r.text(); const js = JSON.parse(t); return `${js.length} 筆；欄位：${Object.keys(js[0] || {}).join("、")}`; });
      for (const [m, k, u] of WARN) await probe(`${k}_${m}`, async () => { const r = await fetch(u, { headers: { "User-Agent": "Mozilla/5.0" }, redirect: "manual" }); if (r.status >= 300 && r.status < 400) throw new Error(`被轉址到 ${r.headers.get("location")}（來源擋連線）`); if (!r.ok) throw new Error(`HTTP ${r.status}`); const t = await r.text(); const js = JSON.parse(t); return `${js.length} 筆；欄位：${Object.keys(js[0] || {}).join("、")}`; });
      for (const [k, u] of TPEX_DIAG) await probe(`上櫃_${k}（電腦負責，僅供參考）`, async () => { const r = await fetch(u, { headers: { "User-Agent": "Mozilla/5.0" }, redirect: "manual" }); if (r.status >= 300 && r.status < 400) throw new Error("仍被轉址（擋連線）"); if (!r.ok) throw new Error(`HTTP ${r.status}`); return `可連線，${JSON.parse(await r.text()).length} 筆`; });
      await probe("kv", async () => { const p = await getPicks(env); return p ? `入選資料 ${p.date}，${(p.picks || []).length} 檔` : "尚未收到電腦同步的入選資料"; });
      return json(out);
    }
    if (url.pathname === "/api/run") {  // 手動測試：/api/run?job=brief&dry=1&secret=...
      const dry = url.searchParams.get("dry") !== "0";
      const job = url.searchParams.get("job") || "brief", nowT = Date.now();
      const r = job === "open" ? await runOpen(env, nowT, dry) : job === "close" ? await runClose(env, nowT, dry)
        : job === "preview" ? await runPreview(env, nowT, dry) : job === "mops" || job === "warn" ? await runEvents(env, nowT, dry, job)
        : await runBrief(env, nowT, dry);
      const body = r.mine || r.skipped || (r.sent ? (r.sent.length ? r.sent.join("\n\n---\n\n") : "目前沒有觀察名單的新事件") : "");
      return new Response(body, { headers: { "content-type": "text/plain; charset=utf-8" } });
    }
    return json({ error: "not found" }, 404);
  },
  async scheduled(event, env, ctx) {
    const when = event.scheduledTime || Date.now();
    ctx.waitUntil(dispatch(env, when).catch((e) => tg(env, env.CHAT_ID, `⚠️ <b>早報 Worker 失敗</b>\n${esc(String(e))}`)));
  },
};
