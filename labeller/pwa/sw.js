// Service worker: makes the page installable and keeps icon glyphs on the device.
// Icons never change for a given name, so they are served cache-first, and so are the
// icon review's sprite sheets (their file names carry the build); the pages are
// network-first (updates arrive, but they still open offline); the API is never cached.
const ICONS = "icons-v1";
const SPRITES = "sprites-v1";
const SHELL = "shell-v1";
const scope = new URL(self.registration.scope);
const iconsPath = new URL("icons/", scope).pathname;
const spritesPath = new URL("sprites/", scope).pathname;
const pages = ["", "index.html", "curate"].map((p) => scope.pathname + p);

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

async function cacheFirst(name, request) {
  const cache = await caches.open(name);
  const hit = await cache.match(request);
  if (hit) return hit;
  const resp = await fetch(request);
  if (resp.ok) {
    cache.put(request, resp.clone());
    if (name === SPRITES) dropOtherBuilds(cache, request.url);
  }
  return resp;
}

// Sheets are named <build>-<n>.png; a rebuilt bundle makes the previous ones dead weight.
async function dropOtherBuilds(cache, url) {
  const build = url.slice(0, url.lastIndexOf("-") + 1);
  for (const req of await cache.keys()) {
    if (!req.url.startsWith(build)) cache.delete(req);
  }
}

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== scope.origin || e.request.method !== "GET") return;
  if (url.pathname.startsWith(iconsPath)) {
    e.respondWith(cacheFirst(ICONS, e.request));
  } else if (url.pathname.startsWith(spritesPath)) {
    e.respondWith(cacheFirst(SPRITES, e.request));
  } else if (pages.includes(url.pathname)) {
    e.respondWith(fetch(e.request).then((resp) => {
      const copy = resp.clone();
      caches.open(SHELL).then((cache) => cache.put(e.request, copy));
      return resp;
    }).catch(() => caches.match(e.request)));
  }
});
