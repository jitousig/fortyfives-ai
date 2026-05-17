/* Fortyfives service worker.
 * Network-first for same-origin GETs so frequent Render redeploys never
 * pin stale assets; cache is purely an offline fallback. WebSocket and
 * non-GET requests are left untouched. */

const CACHE = 'fortyfives-v3';
const SHELL = [
  '/',
  '/static/index.html',
  '/static/style.css?v=3',
  '/static/game.js?v=3',
  '/static/manifest.webmanifest',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
];

self.addEventListener('install', (e) => {
  self.skipWaiting();
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL)).catch(() => {})
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  const url = new URL(req.url);

  // Only handle same-origin GETs. Never touch the WebSocket.
  if (req.method !== 'GET' || url.origin !== self.location.origin) return;
  if (url.pathname === '/ws') return;

  e.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.status === 200 && res.type === 'basic') {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
        }
        return res;
      })
      .catch(() =>
        caches.match(req).then((hit) => hit || caches.match('/'))
      )
  );
});
