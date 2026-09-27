// Service worker: makes the page installable and keeps icon glyphs on the device.
// Icons never change for a given name, so they are served cache-first; the page is
// network-first (updates arrive, but it still opens offline); the API is never cached.
const ICONS = "icons-v1";
const SHELL = "shell-v1";
const scope = new URL(self.registration.scope);
const iconsPath = new URL("icons/", scope).pathname;

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== scope.origin || e.request.method !== "GET") return;
  if (url.pathname.startsWith(iconsPath)) {
    e.respondWith(caches.open(ICONS).then(async (cache) => {
      const hit = await cache.match(e.request);
      if (hit) return hit;
      const resp = await fetch(e.request);
      if (resp.ok) cache.put(e.request, resp.clone());
      return resp;
    }));
  } else if (url.pathname === scope.pathname || url.pathname === scope.pathname + "index.html") {
    e.respondWith(fetch(e.request).then((resp) => {
      const copy = resp.clone();
      caches.open(SHELL).then((cache) => cache.put(e.request, copy));
      return resp;
    }).catch(() => caches.match(e.request)));
  }
});
