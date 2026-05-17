# Mobile-Native UI Plan

Tracking doc for making the Fortyfives web GUI mobile-native. Survives the
ephemeral web-session container via git. Update the checkboxes as milestones land.

## Direction (decided)

- One responsive codebase — no separate iOS/Android builds.
- Portrait-first, purpose-built mobile layout.
- Installable PWA (standalone, home-screen icon, offline app shell).
- Desktop layout preserved behind a breakpoint; existing `game.js` element IDs
  kept so render logic is untouched.

## Target mobile layout (portrait)

- Fixed top status bar: N/S vs E/W score, phase, trump, turn, connection dot.
- Opponents strip: West / North / East chips (card-back count, active highlight,
  dealer/partner tags).
- Center: trick — 4 played-card slots in a compact cross + trump + "Trick n/5".
- Your hand: overlapping fan that fits any width / card count; playable cards
  lift and are always highlighted (no hover dependency).
- Bottom action bar: full-width buttons, >=44px tap targets, sticky.
- Log + score detail: collapsed into a slide-up drawer.

## Native-feel details

- `100dvh` + safe-area insets, `viewport-fit=cover`.
- No tap delay / double-tap zoom, no text selection, `:active` press feedback.
- PWA: manifest, app icons (incl. maskable + apple-touch-icon), iOS meta,
  service worker with versioned cache + `skipWaiting`/`clientsClaim` and
  network-first HTML (so frequent Render redeploys don't pin stale assets).

## Milestones

- [ ] **M1 Foundation** — viewport/meta, `dvh`+safe-area, touch tweaks,
  breakpoint scaffold. Desktop unchanged; phone stops clipping.
- [ ] **M2 Portrait layout** — opponents strip, trick center, fanned hand,
  bottom action bar.
- [ ] **M3 Log/score drawer** + status-bar polish.
- [ ] **M4 PWA** — manifest, icons, service worker, iOS meta; FastAPI route so
  `sw.js` is served at root scope.
- [ ] **M5 Polish & test pass** — tap targets, animations, edge cases (discard
  hand of 6+, long log, game-over, reconnect after Render redeploy).

## Files in scope

- `web/static/index.html` — region restructure, meta/manifest/iOS links,
  drawer elements (IDs preserved).
- `web/static/style.css` — mobile breakpoint layout; desktop styles kept.
- `web/static/game.js` — drawer toggle, service-worker registration.
- New: `web/static/manifest.webmanifest`, `web/static/sw.js`,
  `web/static/icons/`.
- `web/server.py` — serve `sw.js` at root scope + cache headers.

## Testing notes

Server can be booted in-container to verify it serves and to eyeball a narrow
viewport, but real touch / iOS Safari / Android Chrome verification happens on
the physical phone via the Render URL after each milestone's push.

## Deploy loop

Each milestone: commit -> push to `claude/mobile-web-development-DkFI3` ->
Render auto-deploys -> verify on phone.
