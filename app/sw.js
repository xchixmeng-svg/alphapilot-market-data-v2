/* quiet_shadow */ /* 離線時顯示最後一次成功同步的資料（規格 32.12）；資料一律先網路、失敗才用快取。 */
const SHELL = "ap-shell-v9", DATA = "ap-data-v9";
self.addEventListener("install", e => e.waitUntil(caches.open(SHELL).then(c => c.addAll(["./", "index.html", "app.js", "manifest.json", "icon.svg"]))));
self.addEventListener("activate", e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => ![SHELL, DATA].includes(k)).map(k => caches.delete(k))))));
self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.origin !== location.origin || u.pathname.startsWith("/api/")) return;   // 記帳 API 不快取
  if (u.pathname.includes("/data/")) {
    e.respondWith(fetch(e.request).then(r => { if (r.ok) { const cp = r.clone(); caches.open(DATA).then(c => c.put(e.request, cp)); } return r; })
      .catch(() => caches.match(e.request).then(r => r || new Response(JSON.stringify({ error: "offline" }), { status: 503 }))));
    return;
  }
  e.respondWith(fetch(e.request).then(r => { if (r.ok) { const cp = r.clone(); caches.open(SHELL).then(c => c.put(e.request, cp)); } return r; }).catch(() => caches.match(e.request)));
});
