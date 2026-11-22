// AlphaPilot App（Cloudflare Pages 進階模式）：1) 全站登入（密碼 = Pages secret APP_PASSWORD，沒設就全部拒絕）
// 2) /data/* 直接讀 GitHub site-data 分支（雲端每天產生，網站不用重新部署）  3) 其他檔案照常回傳
const COOKIE = "ap_auth";
const SRC = "https://raw.githubusercontent.com/xchixmeng-svg/alphapilot-market-data-v2/site-data/";
async function sign(pw) {
  const k = await crypto.subtle.importKey("raw", new TextEncoder().encode(pw), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const s = await crypto.subtle.sign("HMAC", k, new TextEncoder().encode("alphapilot-v1"));
  return [...new Uint8Array(s)].map(b => b.toString(16).padStart(2, "0")).join("");
}
const page = msg => new Response(`<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><body style="background:#0E1116;color:#E6E8EB;font-family:system-ui;display:grid;place-items:center;height:100vh;margin:0">
<form method=post action="/__login" style="display:grid;gap:10px;width:260px"><b style="color:#C9A86A">AlphaPilot</b><input name=pw type=password placeholder="密碼" autofocus style="padding:12px;border-radius:8px;border:1px solid #29303A;background:#161B22;color:#E6E8EB;font-size:16px">
<button style="padding:12px;border-radius:8px;border:0;background:#C9A86A;font-size:16px">登入</button><small style="color:#F85149">${msg || ""}</small></form>`, { status: 401, headers: { "content-type": "text/html;charset=utf-8" } });

export default {
  async fetch(request, env) {
    const pw = env.APP_PASSWORD;
    if (!pw) return new Response("APP_PASSWORD 未設定，網站關閉", { status: 503 });
    const url = new URL(request.url), good = await sign(pw);
    if (url.pathname === "/__login" && request.method === "POST") {
      const f = await request.formData();
      if ((f.get("pw") || "") !== pw) return page("密碼錯誤");
      return new Response(null, { status: 302, headers: { location: "/", "set-cookie": `${COOKIE}=${good}; Path=/; Max-Age=34560000; HttpOnly; Secure; SameSite=Lax` } });
    }
    const c = (request.headers.get("cookie") || "").split(/;\s*/).find(x => x.startsWith(COOKIE + "="));
    if (!c || c.slice(COOKIE.length + 1) !== good) return page("");
    if (url.pathname.startsWith("/data/")) {
      const p = url.pathname.slice(6);
      if (!/^[\w\-./]+\.json$/.test(p) || p.includes("..")) return new Response("bad path", { status: 400 });
      const r = await fetch(SRC + p, { cf: { cacheTtl: 120, cacheEverything: true } });
      if (!r.ok) return new Response(JSON.stringify({ error: "NOT_FOUND" }), { status: 404, headers: { "content-type": "application/json" } });
      return new Response(r.body, { headers: { "content-type": "application/json;charset=utf-8", "cache-control": "no-cache" } });
    }
    return env.ASSETS.fetch(request);
  }
};
