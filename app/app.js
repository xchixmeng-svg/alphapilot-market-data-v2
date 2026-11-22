"use strict";
/* AlphaPilot V2（2026-11-22 凍結版）：只讀 data/ 底下雲端每天產生的 JSON，不在手機上推導任何交易狀態。
   四頁：行情雷達（燈號＋大盤＋資金動態）／今日決策（明天的 V2 買賣單）／我的持股（V2 模擬帳戶＋正常範圍＋策略對照）／搜尋（個股＋法人）。
   燈號一律用 🟡 亮、⚪ 沒亮、⛔ 警戒，不靠紅綠色。 */
const D = "data/";
const $ = (s, e = document) => e.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const n2 = x => x == null || x !== x ? "—" : (+x).toLocaleString("zh-TW", { maximumFractionDigits: 2 });
const n0 = x => x == null || x !== x ? "—" : Math.round(+x).toLocaleString("zh-TW");
const sg = x => x == null || x !== x ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toLocaleString("zh-TW", { maximumFractionDigits: 2 })}`;
const s0 = x => x == null || x !== x ? "—" : `${x >= 0 ? "+" : "−"}${Math.round(Math.abs(x)).toLocaleString("zh-TW")}`;
const pc = (x, d = 1) => x == null || x !== x ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(d)}%`;
const cls = x => x == null ? "" : x >= 0 ? "up" : "down";
const wan = x => x == null ? "—" : `${(x / 1e4).toLocaleString("zh-TW", { maximumFractionDigits: 1 })} 萬`;
const md = d => { d = String(d || "").replace(/-/g, ""); return d.length >= 8 ? `${d.slice(4, 6)}/${d.slice(6, 8)}` : "—"; };
const ZH = { DORMANT: "沉寂", EMERGING: "萌芽", CONFIRMING: "確認", ADVANCING: "主升", OVERHEATED: "過熱", FADING: "衰退",
  bull: "多頭選股中", bottom: "抓底持股中", cash: "空手", cashds: "空手（跌破防災線）", wait: "等底價", "多頭": "多頭股", "抓底": "抓底股", "抄底": "抓底股" };
const zh = s => ZH[s] || s || "—";
const tag = s => `<span class="tag ${esc(s || "NONE")}">${esc(zh(s))}</span>`;
const go = h => `data-go="${esc(h)}" tabindex="0" role="link"`;
const det = (id, title, body, open) => `<details class="card" data-k="${id}" ${open ? "open" : ""}><summary style="font-weight:650;cursor:pointer">${title}</summary><div style="margin-top:10px">${body}</div></details>`;
const lines = t => `<div style="white-space:pre-wrap;line-height:1.7;font-size:15px">${esc(t || "")}</div>`;

const cache = {};
async function get(p) {
  if (cache[p]) return cache[p];
  const r = await fetch(D + p, { cache: "no-cache" });
  if (!r.ok) throw new Error(r.status === 404 ? `NOT_FOUND:${p}` : `${p}：HTTP ${r.status}`);
  return (cache[p] = await r.json());
}
const tryget = p => get(p).catch(() => null);

function setState(text, kind) { const e = $("#dstate"); e.className = `dstate ${kind || ""}`; e.innerHTML = `<span class="dot"></span>${esc(text)}`; }
async function headerState() {
  const B = await tryget("build.json");
  if (!B) return setState("還沒有資料", "warn");
  const ok = B.inst_ok && B.market_status === "OK";
  setState(`${md(B.date)} 收盤資料・${ok ? "完成" : "法人未到"}`, ok ? "ok" : "warn");
}

/* ---------- 路由：#/<頁>/<參數> ---------- */
const PAGES = { radar, today, mine, find, stock, ind: industryPage, sys, bt };
let lastHash = "";
async function route() {
  let [p, ...rest] = (location.hash.slice(2) || "").split("/");
  if (p === "m" || p === "s") { p = rest.shift(); }          // 舊網址相容
  if (!PAGES[p]) { location.replace("#/radar"); return; }
  const arg = decodeURIComponent(rest.join("/") || "");
  document.querySelectorAll("nav button").forEach(b => b.setAttribute("aria-current", b.dataset.r === p || (p === "stock" && b.dataset.r === "find") || (p === "ind" && b.dataset.r === "radar") || (p === "bt" && b.dataset.r === "mine") ? "page" : "false"));
  const v = $("#view"), nav = location.hash !== lastHash;
  lastHash = location.hash;
  if (nav) v.innerHTML = '<div class="sk"></div><div class="sk"></div><div class="sk"></div>';
  headerState();
  try { await PAGES[p](v, arg); restoreDetails(); }
  catch (e) { const nf = String(e.message).startsWith("NOT_FOUND"); v.innerHTML = `<div class="note ${nf ? "warn" : "err"}"><div style="font-weight:650">${nf ? "這裡還沒有資料" : "讀取失敗"}</div><div class="sub" style="margin:4px 0 12px">${nf ? "雲端今天還沒產生這一頁。" : esc(e.message)}</div><button class="btn" onclick="location.reload()">重試</button></div>`; }
  if (nav) window.scrollTo(0, 0);
}
function restoreDetails() {
  document.querySelectorAll("details[data-k]").forEach(d => {
    const k = "ap_d_" + d.dataset.k, s = sessionStorage.getItem(k);
    if (s != null) d.open = s === "1";
    d.ontoggle = () => sessionStorage.setItem(k, d.open ? "1" : "0");
  });
}
window.addEventListener("hashchange", route);
document.querySelectorAll("nav button").forEach(b => b.onclick = () => location.hash = `#/${b.dataset.r}`);
document.addEventListener("click", e => { const t = e.target.closest("[data-go]"); if (t) location.hash = t.dataset.go; });
document.addEventListener("keydown", e => { const t = e.target.closest("[data-go]"); if (t && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); location.hash = t.dataset.go; } });
$("#b-menu").onclick = () => location.hash = "#/sys";

/* =====================================================================================
   行情雷達：燈號 → 大盤 → 資金動態 → 產業資金
   ===================================================================================== */
function spark(y, h = 46) {
  y = (y || []).filter(v => v != null);
  if (y.length < 2) return "";
  const lo = Math.min(...y), hi = Math.max(...y), w = 300, k = (hi - lo) || 1;
  const pts = y.map((v, i) => `${(i / (y.length - 1) * w).toFixed(1)},${(h - 4 - (v - lo) / k * (h - 8)).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:${h}px;margin-top:8px" preserveAspectRatio="none"><polyline fill="none" stroke="var(--text)" stroke-width="2" points="${pts}"/></svg>`;
}
function lightsHead(L) {
  const t = (L.text || "").split("\n").filter(Boolean);
  return `<div style="font-size:20px;font-weight:700;line-height:1.5">${esc(t[1] || "")}</div><div style="font-size:18px;font-weight:650;margin-top:2px">${esc(t[2] || "")}</div>
    <div class="sub" style="margin-top:6px">${esc(t.slice(3, 5).join("　"))}</div>`;
}
const KL = [["foreign", "外資"], ["trust", "投信"], ["dealer", "自營商"], ["all3", "合計"]];
const stk = s => s == null ? "—" : s === 0 ? "—" : `${s > 0 ? "買" : "賣"} ${Math.abs(s)} 日`;
function instGrid(r, keys = KL) {
  return `<div class="wrap"><table class="tbl"><thead><tr><th>億元</th><th>今日</th><th>5 日</th><th>20 日</th><th>連續</th></tr></thead><tbody>
    ${keys.filter(([k]) => (r || {})[k]).map(([k, lab]) => { const x = r[k]; return `<tr><td>${lab}</td><td class="num ${cls(x.today)}">${s0(x.today)}</td><td class="num ${cls(x.d5)}">${s0(x.d5)}</td><td class="num ${cls(x.d20)}">${s0(x.d20)}</td><td class="num">${stk(x.streak)}</td></tr>`; }).join("")}</tbody></table></div>`;
}
async function radar(v) {
  const [V, X, F] = await Promise.all([tryget("v2.json"), tryget("market/index.json"), tryget("market/flows.json")]);
  const L = (V || {}).lights || {};
  const be = L.bottom_est || {}, lp = L.lamps || {}, on = x => x ? "🟡" : "⚪";
  const keyc = `<div class="card"><dl class="kv num"><dt>三燈（錢／人／價）</dt><dd>${on(lp.money)} ${on(lp.people)} ${on(lp.price)}　亮 ${L.green ?? "—"}／3</dd>
      <dt>抓底條件</dt><dd>${L.lamps_on ?? "—"}／6</dd><dt>事件類型</dt><dd>${esc(L.event_type || "—")}</dd>
      <dt>上次出場</dt><dd>${L.exit_date ? `${md(L.exit_date)}　${n0(L.exit_px)}` : "—"}</dd>
      <dt>估計底價區</dt><dd>${be.lo ? `${n0(be.lo)}～${n0(be.hi)}（中心 ${n0(be.est)}）` : "—"}</dd>
      <dt>美股 VIX</dt><dd>${n2(L.vix)}（≥ 25 算恐慌）</dd><dt>融資維持率</dt><dd>${L.maint_ratio ? n0(L.maint_ratio) + "%" : "—"}（歷史真底多在 120～131%）</dd></dl></div>`;
  const light = L.text ? `<div class="card">${lightsHead(L)}</div>${keyc}${det("lt", "燈號全文（每一盞的規則與現況）", lines(L.text), false)}`
    : `<div class="note warn">燈號還沒產生</div>`;
  const t = (X || {}).taiex || {}, b = (X || {}).breadth || {};
  const idx = X ? `<div class="card"><div class="sub">加權指數｜${md(X.date)} 收盤</div>
      <div class="row" style="align-items:baseline;gap:10px"><span class="big num">${n2(t.close)}</span><span class="num ${cls(t.change)}" style="font-size:18px;font-weight:650">${sg(t.change)}（${pc(t.change_pct, 2)}）</span></div>
      <div class="sub num" style="margin-top:4px">成交金額 ${X.turnover ? n0(X.turnover / 1e8) + " 億" : "—"}｜上漲 ${b.up ?? "—"}（漲停 ${b.up_limit ?? "—"}）・下跌 ${b.down ?? "—"}（跌停 ${b.down_limit ?? "—"}）・持平 ${b.flat ?? "—"}</div>
      ${spark((X.series || {}).close)}<div class="meta">近 60 個交易日</div></div>` : `<div class="note warn">大盤資料還沒產生</div>`;
  v.innerHTML = `<h1>行情雷達</h1><h2>大盤燈號（V2）</h2>${light}<h2>大盤</h2>${idx}` + flowBlock(F);
}
function flowBlock(F) {
  if (!F) return `<h2>資金動態</h2><div class="note warn">資金動態還沒產生</div>`;
  if (!F.inst_ok) return `<h2>資金動態</h2><div class="note warn">${esc((F.notes || []).join("；") || "今天的法人資料還沒進來")}</div>`;
  const off = F.official ? `<div class="sub" style="margin-bottom:6px">官方金額（${esc(F.official.scope)}，買進金額－賣出金額）</div>${instGrid(F.official, KL.slice(0, 3))}` : "";
  const rows = (F.industries || []).map(r => `<div class="card tap" ${go(`#/ind/${encodeURIComponent(r.industry)}`)}>
      <div class="row"><span style="font-size:18px;font-weight:650">${esc(r.industry)}</span>${tag(r.state)}<span class="sp"></span><span class="sub num">${r.n} 檔</span><span class="chev">›</span></div>
      <div class="sub num" style="margin-top:6px">法人合計（億）今日 ${s0(r.all3.today)}・5 日 ${s0(r.all3.d5)}・20 日 ${s0(r.all3.d20)}・連${stk(r.all3.streak)}</div>
      <div class="sub">5 日主要由${esc(r.driver_5d)}推動｜成交占比 ${(r.share * 100).toFixed(1)}%（較 20 日平均 ${sg(+(r.share_chg * 100).toFixed(2))} 個百分點）</div></div>`).join("");
  return `<h2>資金動態</h2><div class="card">${off}<div class="sub" style="margin:10px 0 6px">全市場估算（上市＋上櫃個股，股數 × 收盤）</div>${instGrid(F.market)}</div>
    <h2>產業資金（估算）</h2><div class="sub">官方沒有產業別金額：這裡是「個股法人買賣超股數 × 收盤價」加總，依 5 日合計排序；標籤是產業階段。</div>${rows}`;
}
async function industryPage(v, name) {
  const F = await get("market/flows.json");
  const r = (F.industries || []).find(x => x.industry === name);
  if (!r) { v.innerHTML = `<div class="note err">找不到產業 ${esc(name)}</div>`; return; }
  v.innerHTML = `<h1>${esc(name)}</h1><div class="sub"><a ${go("#/radar")}>‹ 行情雷達</a></div><div class="row" style="margin-top:8px">${tag(r.state)}<span class="sub">${r.n} 檔｜前一天：${esc(zh(r.prev_state))}</span></div>
    <div class="card"><div class="sub" style="margin-bottom:6px">${esc(Object.values(r.phrases).join("；"))}</div>${instGrid(r)}</div>
    <div class="card"><div class="sub">階段理由</div><div>${esc(r.reason)}</div><ul class="plain">${(r.why_now || []).map(x => `<li>${esc(x)}</li>`).join("")}</ul>
    ${(r.risks || []).length ? `<div class="sub" style="margin-top:8px">風險</div><ul class="plain">${r.risks.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}</div>`;
}

/* =====================================================================================
   今日決策：明天開盤的 V2 買賣單
   ===================================================================================== */
async function today(v) {
  const V = await get("v2.json"), P = V.picks || {}, L = V.lights || {};
  const mode = P.mode || (V.account || {}).mode || L.state;
  const sells = P.sells || [], buys = P.picks || [];
  const head = `<div class="card"><div class="sub">${md(P.date || V.date)} 收盤後產生・明天開盤執行</div>
      <div style="font-size:20px;font-weight:700;margin-top:4px">系統狀態：${esc(zh(mode))}</div>
      ${P.sellall ? `<div class="note warn" style="margin-top:8px">⛔ 明天開盤全部多頭股賣出（煞車）</div>` : ""}
      ${(P.alarms || []).map(a => `<div class="note warn" style="margin-top:8px">⛔ ${esc(a)}</div>`).join("")}</div>`;
  const sl = sells.length ? `<div class="list">${sells.map(s => `<div class="li tap" ${go(`#/stock/${s.code}`)}><div class="main"><div class="nm">${esc(s.code)} ${esc(s.name || "")}</div><div class="why">${esc(s.why)}</div></div><div class="rt"><b>開盤賣</b></div><span class="chev">›</span></div>`).join("")}</div>` : `<div class="note empty">明天沒有要賣的</div>`;
  const bl = buys.length ? `<div class="list">${buys.map(b => `<div class="li tap" ${go(`#/stock/${b.code}`)}><div class="main"><div class="nm">${esc(b.code)} ${esc(b.name)}</div><div class="why">${esc(zh(b.kind))}・開盤價高於 ${n2(b.buy_max)}（漲停鎖住）就不買</div></div><div class="rt"><b>開盤買</b><div class="meta">每檔 9%</div></div><span class="chev">›</span></div>`).join("")}</div>` : `<div class="note empty">明天沒有要買的</div>`;
  v.innerHTML = `<h1>今日決策</h1>${head}<h2>明天要賣</h2>${sl}<h2>明天要買</h2>${bl}
    ${P.text ? det("pt", "推播全文", lines(P.text), false) : `<div class="note warn" style="margin-top:12px">選股程式還沒產生今天的結果（每個交易日晚上執行）。</div>`}
    ${det("rule", "V2 規則（2026-11-22 凍結）", lines(RULES), false)}`;
}
const RULES = `【燈號】大盤 3 燈（錢、人、價）；出場燈亮就全部出場。
【抓底】6 盞全亮才進場：台幣守住、恐慌過、VIX ≥25、融資或法人燈、切割線、底價區。急跌型（一年高點後 35 天內跌超過 12%）不用等底價區。
【抓底買法】離 250 日高點最深的 10 檔（20 日均成交值 ≥ 3,000 萬、上市 ≥ 120 天），漲 50% 停利；大盤跌破抓底日 × 0.8 全部停損。
【多頭買法】3 燈亮 2 盞以上、熊市指標沒亮才買：創 250 日新高、5 日均量 > 20 日均量 2 倍、20 日均成交值 ≥ 3,000 萬、上市 ≥ 250 天、最新月營收年增 > 20%，依 60 日漲幅排前段。
【多頭賣法】收盤跌破買價 × 0.9，或跌破自己的季線。
【煞車】帳戶比高點少 20% 而且燈號不到 3 盞，多頭股全部賣出；3 盞全亮時重設高點。
【部位】每檔 9%、最多 10 檔；手續費＋稅每邊 0.3%。今天收盤決定，明天開盤成交；開盤漲停（≥ 前收 × 1.092）不買。`;

/* =====================================================================================
   我的持股：V2 模擬帳戶＋正常範圍＋策略對照
   ===================================================================================== */
function eqChart(eq) {
  const e = Object.entries(eq || {}).sort();
  if (e.length < 2) return `<div class="meta">累積兩天以上才有走勢圖</div>`;
  return spark(e.map(x => x[1]), 70) + `<div class="meta">${md(e[0][0])}～${md(e.at(-1)[0])}</div>`;
}
async function mine(v) {
  const V = await get("v2.json"), A = V.account || {}, P = V.picks || {};
  const val = A.value ?? P.value;
  const pos = A.pos || [];
  const hold = pos.length ? `<div class="list">${pos.map(p => `<div class="li tap" ${go(`#/stock/${p.code}`)}><div class="main"><div class="nm">${esc(p.code)} ${esc((P.holdings || []).find(h => h.code === p.code)?.name || "")}</div>
      <div class="why">${esc(zh(p.kind))}・${md(p.ent)} 買 ${n2(p.epx)}・${n0(p.sh)} 股・市值 ${wan(p.mv)}</div>
      <div class="why">${p.kind === "多頭" ? `出場線：收盤 < ${n2(p.stop)}（買價 −10%）或 < 季線 ${n2(p.ma60)}` : `停利 ${n2(p.tp)}（+50%）；燈號轉多頭或大盤跌破防災線也會賣`}</div></div>
      <div class="rt"><b class="num">${n2(p.close)}</b><div class="num">${pc(p.ret)}</div></div><span class="chev">›</span></div>`).join("")}</div>` : `<div class="note empty">目前空手</div>`;
  const checks = (P.checks || []).length ? `<div class="card">${P.checks.map(c => `<div style="margin:6px 0">${esc(c)}</div>`).join("")}<div class="meta" style="margin-top:8px">🟡 在歷史正常範圍內・⛔ 超過警戒線（策略可能壞了，要停下來檢查）</div></div>`
    : `<div class="note empty">帳戶開始跑之後才會有</div>`;
  const tr = (V.trades || []).slice().reverse().slice(0, 30);
  const trades = tr.length ? `<div class="wrap"><table class="tbl"><thead><tr><th>日期</th><th>動作</th><th>股票</th><th>價格</th><th>報酬</th></tr></thead><tbody>
      ${tr.map(t => `<tr><td class="num">${md(t.date)}</td><td>${esc(t.side)}・${esc(zh(t.kind))}</td><td>${esc(t.code)} ${esc(t.name || "")}</td><td class="num">${n2(t.px)}</td><td class="num">${t.ret == null || t.ret !== t.ret ? "—" : pc(t.ret)}</td></tr>`).join("")}</tbody></table></div>` : `<div class="note empty">還沒有成交</div>`;
  const cmp = (V.compare || []).length ? `<div class="wrap"><table class="tbl"><thead><tr><th>策略</th><th>市值</th><th>報酬</th><th>離高點</th></tr></thead><tbody>
      ${V.compare.map(c => `<tr><td>${esc(c.name)}</td><td class="num">${wan(c.value)}</td><td class="num">${pc(c.ret)}</td><td class="num">${pc(c.dd)}</td></tr>`).join("")}</tbody></table></div>
      <div class="meta">同樣從 ${md(V.compare[0].since)} 開始、100 萬。以後的新策略會以影子帳戶加在這裡一起比。</div>` : `<div class="note empty">帳戶開始跑之後才會有</div>`;
  const ks = A.stats || {};
  const stat = Object.keys(ks).length ? `<div class="wrap"><table class="tbl"><thead><tr><th></th><th>筆數</th><th>勝率</th><th>贏的平均</th><th>輸的平均</th><th>最大虧</th></tr></thead><tbody>
      ${Object.entries(ks).map(([k, x]) => `<tr><td>${esc(zh(k))}</td><td class="num">${x.n}</td><td class="num">${(x.win * 100).toFixed(0)}%</td><td class="num">${pc(x.avg_win)}</td><td class="num">${pc(x.avg_loss)}</td><td class="num">${pc(x.max_loss)}</td></tr>`).join("")}</tbody></table></div><div class="meta">報酬是那一檔 9% 部位自己的漲跌（含手續費），不是占總資金的比例。</div>` : `<div class="note empty">還沒有賣出的成交</div>`;
  v.innerHTML = `<h1>我的持股</h1><div class="sub">V2 模擬帳戶（100 萬起，照規則自動買賣，不是真實下單）</div>
    <div class="card"><div class="sub">帳戶市值</div><div class="big num">${wan(val)}</div><div class="sub num">${val ? pc(val / 1e6 - 1) : "—"}｜現金 ${wan(A.cash)}｜高點 ${wan(A.peak)}（離高點 ${A.peak ? pc(val / A.peak - 1) : "—"}）｜${esc(zh(A.mode || P.mode))}</div>${eqChart(A.eq)}</div>
    <h2>持股</h2>${hold}<h2>正常範圍檢查</h2>${checks}<h2>實際成交統計</h2>${stat}<h2>策略對照</h2>${cmp}<h2>最近成交</h2>${trades}
    <h2>V2 回測（2008～2025）</h2><div class="list"><div class="li tap" ${go("#/bt")}><div class="main"><div class="nm">逐年績效・逐次事件・746 筆交易明細</div><div class="why">100 萬 → 9,862 萬，最大回撤 −28.5%</div></div><span class="chev">›</span></div></div>`;
}

/* =====================================================================================
   搜尋＋個股
   ===================================================================================== */
let IDX = null;
async function find(v) {
  v.innerHTML = `<h1>搜尋</h1><input id="q" class="search" placeholder="股票代號或公司名稱" autocomplete="off" inputmode="search" enterkeyhint="search" aria-label="搜尋股票">
    <div id="hits"></div>`;
  const q = $("#q");
  q.value = sessionStorage.getItem("ap_q") || "";
  IDX = IDX || (await tryget("search/index.json")) || [];
  const hint = `<div class="sub" style="margin:12px 2px">輸入代號（例：2330）或名稱（例：台積）。搜尋範圍是全台股。</div>`;
  const run = () => {
    const s = q.value.trim();
    sessionStorage.setItem("ap_q", s);
    if (!s) { $("#hits").innerHTML = hint; return; }
    const h = IDX.filter(r => r[0].startsWith(s) || (r[1] || "").includes(s)).slice(0, 30);
    $("#hits").innerHTML = h.length ? `<div class="list">${h.map(r => `<div class="li tap" ${go(`#/stock/${r[0]}`)}><div class="main"><div class="nm">${esc(r[0])} ${esc(r[1])}</div><div class="why">${esc(r[2] || "")}</div></div><span class="chev">›</span></div>`).join("")}</div>`
      : `<div class="note empty">找不到「${esc(s)}」</div>`;
  };
  q.oninput = run; run();
  if (!q.value) q.focus();
}
function chartBars(b) {
  if (!b || !b.c || b.c.length < 2) return `<div class="empty">沒有走勢資料</div>`;
  const N = b.c.length, { o, h, l, c, t, ma20, ma60 } = b;
  const all = h.concat(l).filter(x => x != null), hi = Math.max(...all), lo = Math.min(...all);
  const W = 640, H = 300, P = 70, cw = (W - P) / N, y = x => 10 + (hi - x) / (hi - lo || 1) * (H - 40);
  let g = "";
  for (let i = 0; i < N; i++) {   // 色盲友善：漲＝空心、跌＝實心
    const x = i * cw + cw / 2, up = c[i] >= o[i], top = y(Math.max(o[i], c[i])), hh = Math.max(1.5, Math.abs(y(o[i]) - y(c[i])));
    g += `<line x1="${x}" x2="${x}" y1="${y(h[i])}" y2="${y(l[i])}" stroke="var(--mute)" stroke-width="1.2"/><rect x="${x - cw * .34}" y="${top}" width="${cw * .68}" height="${hh}" fill="${up ? "var(--bg)" : "var(--mute)"}" stroke="var(--text)" stroke-width="1"/>`;
  }
  const poly = (arr, col, dash) => { const p = (arr || []).map((v, i) => v == null ? null : `${i * cw + cw / 2},${y(v)}`).filter(Boolean); return p.length > 1 ? `<polyline points="${p.join(" ")}" fill="none" stroke="${col}" stroke-width="2" ${dash ? 'stroke-dasharray="6 4"' : ""}/>` : ""; };
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="日 K 線">${g}${poly(ma20, "#C9A86A")}${poly(ma60, "#8BB8FF", 1)}
    <text x="${W - P + 6}" y="26">${n2(hi)}</text><text x="${W - P + 6}" y="${H - 22}">${n2(lo)}</text></svg>
    <div class="meta">${md(t[0])}～${md(t[N - 1])} 日線　空心＝收漲、實心＝收跌　<span style="color:#C9A86A">— 月線</span>　<span style="color:#8BB8FF">- - 季線</span></div>`;
}
const lots = x => x == null ? "—" : `<span class="${cls(x)}">${sg(Math.round(x / 1000))}</span> 張`;
function instTable(b) {
  if (!b) return `<div class="sub">沒有法人資料</div>`;
  return `<div class="wrap"><table class="tbl"><thead><tr><th></th><th>今日</th><th>5 日</th><th>20 日</th></tr></thead><tbody>
    <tr><td>外資</td><td class="num">${lots(b.f_today)}</td><td class="num">${lots(b.f_5)}</td><td class="num">${lots(b.f_20)}</td></tr>
    <tr><td>投信</td><td class="num">${lots(b.t_today)}</td><td class="num">${lots(b.t_5)}</td><td class="num">${lots(b.t_20)}</td></tr>
    <tr><td>自營</td><td class="num">${lots(b.d_today)}</td><td class="num">${lots(b.d_5)}</td><td class="num">${lots(b.d_20)}</td></tr></tbody></table></div>
    <div class="meta">外資連續 ${stk(b.f_streak)}・投信連續 ${stk(b.t_streak)}</div>`;
}
function chk(S) {
  const k = S.chk || {}, ok = x => x == null ? "—" : x ? "🟡" : "⚪";
  const rows = [["創 250 日新高（差 0.1% 內）", k.vs_h250 == null ? null : k.vs_h250 >= -0.001, `離高點 ${pc(k.vs_h250)}`],
    ["5 日均量 > 20 日均量 2 倍", k.vol_ratio == null ? null : k.vol_ratio > 2, `${n2(k.vol_ratio)} 倍`],
    ["20 日均成交值 ≥ 3,000 萬", k.val20 == null ? null : k.val20 >= 3e7, k.val20 >= 1e8 ? `${n2(k.val20 / 1e8)} 億` : `${n0((k.val20 || 0) / 1e4)} 萬`],
    ["最新月營收年增 > 20%", S.rev_yoy == null ? null : S.rev_yoy > 20, S.rev_yoy == null ? "沒有資料" : `${sg(S.rev_yoy)}%`]];
  return rows.map(r => `<div style="margin:6px 0">${ok(r[1])} ${esc(r[0])}<span class="sub">　${esc(r[2])}</span></div>`).join("") +
    `<div class="meta">近 60 日漲幅 ${pc(k.ret60)}（候選依這個排名）。另需上市滿 250 天、大盤燈號 ≥ 2 盞且熊市指標沒亮。</div>`;
}
async function stock(v, code) {
  const S = await get(`stocks/${code}.json`), b = S.bars || {}, c = (b.c || []).at(-1), p = (b.c || []).at(-2);
  v.innerHTML = `<h1>${esc(S.code)} ${esc(S.name)}</h1><div class="sub"><a ${go("#/find")}>‹ 搜尋</a>　${esc(S.industry || "")}</div>
    ${S.v2 ? `<div class="note" style="margin-top:8px">🟡 ${esc(S.v2)}</div>` : ""}
    <div class="card"><div class="row" style="align-items:baseline;gap:10px"><span class="big num">${n2(c)}</span><span class="num ${cls(c - p)}" style="font-size:18px">${p ? pc(c / p - 1, 2) : ""}</span></div>
    <dl class="kv num" style="margin-top:8px"><dt>月線／季線</dt><dd>${n2((b.ma20 || []).at(-1))}／${n2((b.ma60 || []).at(-1))}</dd><dt>最新月營收年增</dt><dd>${S.rev_yoy == null ? "—" : sg(S.rev_yoy) + "%"}</dd></dl></div>
    <h2>V2 多頭買進條件（以今天收盤看）</h2><div class="card">${chk(S)}</div>
    <div class="card">${chartBars(b)}</div><h2>法人買賣超</h2><div class="card">${instTable(S.inst)}</div>`;
}

async function bt(v) {
  const B = await get("backtest.json"), C = B.trade_cols, ix = k => C.indexOf(k);
  const yr = `<div class="wrap"><table class="tbl"><thead><tr><th>年</th><th>年底市值</th><th>報酬</th><th>大盤</th><th>年內最深</th><th>筆數</th></tr></thead><tbody>
    ${B.years.map(y => `<tr><td class="num">${y.year}</td><td class="num">${wan(y.end)}</td><td class="num">${pc(y.ret, 0)}</td><td class="num">${pc(y.taiex, 0)}</td><td class="num">${pc(y.dd, 0)}</td><td class="num">${y.trades}</td></tr>`).join("")}</tbody></table></div>`;
  const st = `<div class="wrap"><table class="tbl"><thead><tr><th></th><th>筆數</th><th>勝率</th><th>贏的平均</th><th>輸的平均</th><th>最大賺</th><th>最大虧</th><th>持有天數中位</th></tr></thead><tbody>
    ${Object.entries(B.stats).map(([k, x]) => `<tr><td>${esc(zh(k))}</td><td class="num">${x.n}</td><td class="num">${(x.win * 100).toFixed(0)}%</td><td class="num">${pc(x.avg_win)}</td><td class="num">${pc(x.avg_loss)}</td><td class="num">${pc(x.max_win, 0)}</td><td class="num">${pc(x.max_loss)}</td><td class="num">${x.med_days}</td></tr>`).join("")}</tbody></table></div><div class="meta">每筆報酬是那 9% 部位自己的漲跌，含手續費。</div>`;
  const ev = `<div class="wrap"><table class="tbl"><thead><tr><th>出場日</th><th>出場點</th><th>之後最低</th><th>結果</th><th>回場點</th><th>比出場價</th><th>離底</th></tr></thead><tbody>
    ${B.events.slice().reverse().map(e => `<tr><td class="num">${e.exit.slice(0, 4)}/${md(e.exit)}</td><td class="num">${n0(e.exit_px)}</td><td class="num">${e.low ? `${n0(e.low)}（${pc(e.low_pct, 0)}）` : "—"}</td><td>${esc(e.next)}${e.next_date ? " " + md(e.next_date) : ""}</td><td class="num">${n0(e.next_px)}</td><td class="num">${pc(e.vs_exit, 0)}</td><td class="num">${pc(e.above_low, 0)}</td></tr>`).join("")}</tbody></table></div>
    <div class="meta">「比出場價」：回場時大盤比出場時高或低多少（負數＝出場有躲掉跌幅）。「離底」：回場點比這段最低點高多少。</div>`;
  const rows = B.trades.slice().reverse();
  const tl = q => rows.filter(r => !q || String(r[ix("code")]).startsWith(q) || String(r[ix("ent")]).startsWith(q)).slice(0, 200).map(r => `<tr><td class="num">${r[ix("code")]}</td><td>${esc(zh(r[ix("kind")]))}</td><td class="num">${r[ix("ent")].slice(0, 4)}/${md(r[ix("ent")])} ${n2(r[ix("epx")])}</td><td class="num">${md(r[ix("out")])} ${n2(r[ix("opx")])}</td><td class="num">${pc(r[ix("ret")])}</td><td>${esc(r[ix("why")])}</td></tr>`).join("");
  v.innerHTML = `<h1>V2 回測</h1><div class="sub"><a ${go("#/mine")}>‹ 我的持股</a>　${esc(B.period)}，100 萬起，今天決定、明天開盤成交，含手續費與稅，不含配息</div>
    <div class="card"><dl class="kv num"><dt>期末</dt><dd>${wan(B.end)}</dd><dt>年化</dt><dd>${pc(B.cagr)}</dd><dt>最大回撤</dt><dd>${pc(B.max_dd)}</dd></dl></div>
    <h2>逐年</h2>${yr}<h2>多頭股 vs 抓底股</h2>${st}<h2>每一次出場事件</h2>${ev}
    <h2>交易明細</h2><input id="tq" class="search" placeholder="輸入代號或年份（例：2330、2020）" inputmode="search"><div class="wrap"><table class="tbl"><thead><tr><th>代號</th><th>類型</th><th>買</th><th>賣</th><th>報酬</th><th>原因</th></tr></thead><tbody id="tb">${tl("")}</tbody></table></div><div class="meta">最多顯示 200 筆，用搜尋找其他的。</div>`;
  $("#tq").oninput = e => $("#tb").innerHTML = tl(e.target.value.trim());
}
async function sys(v) {
  const B = await tryget("build.json");
  v.innerHTML = `<h1>系統</h1><div class="card"><dl class="kv num"><dt>資料日</dt><dd>${md((B || {}).date)}</dd><dt>產生時間</dt><dd>${esc((B || {}).built_at || "—")}</dd>
    <dt>個股</dt><dd>${n0((B || {}).stocks)} 檔</dd><dt>法人資料</dt><dd>${(B || {}).inst_ok ? "已到" : "未到"}</dd><dt>策略</dt><dd>V2（2026-11-22 凍結）</dd></dl></div>
    <div class="card sub">全部資料在雲端（GitHub Actions）每天收盤後產生，家裡電腦不用開。燈號 15:00 後、選股 21:00 後融資公布再更新一次。</div>`;
}

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
route();
