'use strict';

// ── Constants ──
const SUIT_SYM = { S: '♠', H: '♥', D: '♦', C: '♣' };
const BID_LABEL = { 0: 'Pass', 1: '20', 2: '25', 3: '30', 4: 'Hold' };
const BID_VALUES = { 0: null, 1: 20, 2: 25, 3: 30, 4: null };
const PHASE = { AUCTION: 1, DECLARATION: 2, DISCARD: 3, GAMEPLAY: 4 };
// Screen positions around the table, index 0 = bottom (always "you").
const POS = ['south', 'west', 'north', 'east'];
const REL_NAMES = ['You', 'West', 'North', 'East'];

// Rotate an absolute seat to the local player's view: your seat is
// always at the bottom ('south'). Solo (your_seat 0 / undefined) → the
// identity, so single-player rendering is byte-identical to before.
function _you() {
  return (state && state.your_seat != null) ? state.your_seat : 0;
}
function posOf(seat) { return POS[(seat - _you() + 4) % 4]; }
function nameOf(seat) { return REL_NAMES[(seat - _you() + 4) % 4]; }
// Actual player/bot name for a seat (multiplayer); solo has no
// seat_names so this falls back to the rotated compass label →
// single-player display is unchanged.
function playerLabel(seat) {
  const n = state && state.seat_names;
  return (n && n[String(seat)]) || nameOf(seat);
}

let ws = null;
let state = null;
let roomCode = null;        // null = solo; set = multiplayer room
let lobby = null;           // last lobby payload (multiplayer only)
let playerName = null;      // multiplayer display name
let curScreen = 'landing';  // 'landing' | 'lobby' | 'game'
let discardSelections = new Set();
let trickAnimTimer = null;

// ── Connection ──

function _wsPath() {
  return roomCode ? `/ws/${roomCode}` : '/ws';
}

function _tokKey() { return 'ff_tok_' + (roomCode || ''); }

// Set true on a FATAL room error (room gone / stale token on a
// cold-start resume) so ws.onclose stops the 2s reconnect loop instead
// of hammering a dead room forever.
let abandon = false;

// Remember which room we have a live seat in, so a cold start (app
// killed & relaunched) can OFFER to resume it. Token itself is under
// _tokKey(); these point at it.
function _rememberActiveRoom() {
  if (!roomCode) return;
  try {
    localStorage.setItem('ff_active_room', roomCode);
    localStorage.setItem('ff_active_name', playerName || '');
  } catch (_) {}
}

function _forgetActiveRoom() {
  try {
    if (roomCode) localStorage.removeItem(_tokKey());
    const saved = localStorage.getItem('ff_active_room');
    if (saved) localStorage.removeItem('ff_tok_' + saved);
    localStorage.removeItem('ff_active_room');
    localStorage.removeItem('ff_active_name');
  } catch (_) {}
}

// Landing-screen "Resume game ABCD" affordance: visible only when a
// paused game is saved on this device.
function renderResume() {
  const row = document.getElementById('resume-row');
  if (!row) return;
  let code = '';
  try { code = localStorage.getItem('ff_active_room') || ''; } catch (_) {}
  if (code && localStorage.getItem('ff_tok_' + code)) {
    document.getElementById('resume-code').textContent = code;
    row.style.display = '';
  } else {
    row.style.display = 'none';
  }
}

function resumeGame() {
  let code = '';
  try { code = localStorage.getItem('ff_active_room') || ''; } catch (_) {}
  if (!code) { renderResume(); return; }
  abandon = false;
  roomCode = code;
  playerName = (localStorage.getItem('ff_active_name') || 'Player');
  connect();                 // onopen rejoins via ff_tok_<code>
  showScreen('lobby');       // server pushes state → game, or lobby
}

function forgetSavedGame() {
  const saved = (() => {
    try { return localStorage.getItem('ff_active_room'); } catch (_) { return null; }
  })();
  if (saved) { try { localStorage.removeItem('ff_tok_' + saved); } catch (_) {} }
  try {
    localStorage.removeItem('ff_active_room');
    localStorage.removeItem('ff_active_name');
  } catch (_) {}
  renderResume();
}

// In-game "Leave": forget the game ON THIS DEVICE and go back to the
// landing screen. The seat stays reserved server-side (PR-C: the game
// waits for that player) — this just stops THIS device auto-returning.
function leaveGame() {
  abandon = true;
  _forgetActiveRoom();
  try { if (ws) ws.close(); } catch (_) {}
  roomCode = null; playerName = null; state = null; lobby = null;
  showScreen('landing');
  renderResume();
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}${_wsPath()}`);

  ws.onopen = () => {
    setConn(true);
    if (!roomCode) return;            // solo: server pushes state
    // Prefer reconnecting to our existing seat (survives phone-lock /
    // network blips — the server keeps the seat & game paused). Fall
    // back to a fresh lobby join if we have no token yet.
    const tok = localStorage.getItem(_tokKey());
    if (tok) {
      ws.send(JSON.stringify({ type: 'rejoin', token: tok }));
    } else if (playerName) {
      ws.send(JSON.stringify({ type: 'join', name: playerName }));
    }
  };
  ws.onclose = () => {
    setConn(false);
    if (abandon) return;          // dead room / left — stop retrying
    setTimeout(connect, 2000);
  };
  ws.onerror = () => setConn(false);
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'seat') {
      // Persist the reconnect token so a phone-lock/refresh rejoins
      // the same seat instead of starting over, and mark this as the
      // saved game so a cold start can offer to resume it.
      try { localStorage.setItem(_tokKey(), data.token); } catch (_) {}
      _rememberActiveRoom();
      return;
    }
    if (data.type === 'lobby') {
      lobby = data;
      updateWaiting(data.waiting_for);
      showScreen(data.started ? 'game' : 'lobby');
      if (!data.started) renderLobby();
      return;
    }
    if (data.type === 'state') {
      showScreen('game');
      if (!state || state.phase !== data.phase || data.game_over) {
        discardSelections.clear();
      }
      // Cancel any pending animation when a non-animating state arrives
      if (!data.trick_animating && trickAnimTimer !== null) {
        clearTimeout(trickAnimTimer);
        trickAnimTimer = null;
      }
      state = data;
      updateWaiting(data.waiting_for);
      render();
      if (data.trick_animating && data.last_trick_winner != null) {
        const winner = data.last_trick_winner;
        trickAnimTimer = setTimeout(() => {
          trickAnimTimer = null;
          startTrickAnimation(winner);
        }, 2500);
      }
    } else if (data.type === 'error') {
      console.warn('Server error:', data.error);
      if (data.error === 'Room not found') {
        // The room is gone (server restarted / reaped after a week).
        // Don't loop-reconnect into a dead room — forget it and go
        // back to landing.
        abandon = true;
        _forgetActiveRoom();
        try { if (ws) ws.close(); } catch (_) {}
        roomCode = null;
        showScreen('landing');
        renderResume();
        alert('That game is no longer available — it ended or expired.');
      } else if (data.error === 'reconnect_failed') {
        // Stale token but the room still exists — drop the token and
        // re-enter the lobby fresh.
        try { localStorage.removeItem(_tokKey()); } catch (_) {}
        if (roomCode && playerName) {
          ws.send(JSON.stringify({ type: 'join', name: playerName }));
        }
      }
    }
  };
}

// "Waiting for <names> to reconnect…" banner — shown whenever a
// claimed seat is disconnected (the game is paused, not frozen).
function updateWaiting(list) {
  const el = document.getElementById('waiting-banner');
  if (!el) return;
  if (list && list.length) {
    el.textContent = `Waiting for ${list.join(', ')} to reconnect…`;
    el.style.display = '';
  } else {
    el.style.display = 'none';
  }
}

function setConn(ok) {
  const el = document.getElementById('conn-status');
  el.className = `conn-dot ${ok ? 'connected' : 'disconnected'}`;
  el.title = ok ? 'Connected' : 'Disconnected';
}

function sendAction(action) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'action', action }));
  }
}

function newGame() {
  discardSelections.clear();
  document.getElementById('overlay').classList.remove('visible');
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'new_game' }));
  }
}

// ── Screens: landing → (lobby) → game ──

function showScreen(name) {
  curScreen = name;
  const set = (id, on) => {
    const el = document.getElementById(id);
    if (el) el.style.display = on ? '' : 'none';
  };
  set('landing', name === 'landing');
  set('lobby', name === 'lobby');
  set('app', name === 'game');
  // "Leave" only matters in a multiplayer game (solo just uses New Game).
  set('leave-btn', name === 'game' && !!roomCode);
}

function _send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function playSolo() {
  abandon = false;
  roomCode = null;            // → connects to /ws (unchanged solo flow)
  connect();
  showScreen('game');
}

async function createRoom() {
  try {
    const r = await fetch('/room', { method: 'POST' });
    const { code } = await r.json();
    _enterRoom(code);
  } catch (e) {
    alert('Could not create room. Try again.');
  }
}

function joinRoom() {
  const code = (document.getElementById('join-code').value || '')
    .trim().toUpperCase();
  if (code.length < 3) { alert('Enter a room code.'); return; }
  _enterRoom(code);
}

function _enterRoom(code) {
  abandon = false;
  roomCode = code;
  playerName = (document.getElementById('player-name')
    && document.getElementById('player-name').value || '').trim()
    || 'Player';
  connect();
  showScreen('lobby');
  // join announced in ws.onopen once the socket is open
}

function claimSeat(seat) { _send({ type: 'claim_seat', seat }); }
function leaveSeat() { _send({ type: 'leave_seat' }); }
function startGame() { _send({ type: 'start' }); }

function copyRoomLink() {
  const link = `${location.origin}/?room=${roomCode}`;
  if (navigator.clipboard) navigator.clipboard.writeText(link);
}

function renderLobby() {
  if (!lobby) return;
  const codeEl = document.getElementById('lobby-code');
  if (codeEl) codeEl.textContent = lobby.code;
  const grid = document.getElementById('seat-grid');
  if (grid) {
    grid.innerHTML = lobby.seats.map(s => {
      const taken = s.claimed_by != null;
      const mine = s.is_you;
      const who = taken ? s.claimed_by : '<em>open</em>';
      const cls = ['seat-card', `pship-${s.partnership.toLowerCase()}`,
        mine ? 'mine' : '', taken && !mine ? 'taken' : ''].filter(Boolean).join(' ');
      const act = mine
        ? `<button onclick="leaveSeat()">Leave</button>`
        : (taken ? '' : `<button onclick="claimSeat(${s.seat})">Sit here</button>`);
      return `<div class="${cls}">
        <div class="seat-pos">${s.name} <span class="pship">${s.partnership}</span></div>
        <div class="seat-who">${who}</div>${act}</div>`;
    }).join('');
  }
  const startBtn = document.getElementById('lobby-start');
  if (startBtn) startBtn.disabled = !lobby.can_start;
}

// ── Card helpers ──

function cardStr(cs) {
  // cs like "AS", "TH", "KD"
  const rank = cs[0] === 'T' ? '10' : cs[0];
  const suit = cs[1];
  return { rank, suit, sym: SUIT_SYM[suit], red: suit === 'H' || suit === 'D' };
}

function cardHTML(cs, playable, action) {
  const { rank, sym, red } = cardStr(cs);
  const cls = ['card', red ? 'red' : '', playable ? 'playable' : ''].filter(Boolean).join(' ');
  const click = playable ? `onclick="sendAction(${action})"` : '';
  return `<div class="${cls}" ${click} title="${rank}${sym}">
    <div class="card-corner tl">${rank}<br>${sym}</div>
    <div class="card-mid">${sym}</div>
    <div class="card-corner br">${rank}<br>${sym}</div>
  </div>`;
}

function cardBackHTML() {
  return '<div class="card back"></div>';
}

function emptySlotHTML() {
  return '<div class="trick-slot-empty"></div>';
}

function toggleDiscardSelection(index) {
  if (discardSelections.has(index)) {
    discardSelections.delete(index);
  } else {
    discardSelections.add(index);
  }
  renderHands();
  renderActions();
}

function startTrickAnimation(winnerIdx) {
  const slotIds = ['trick-south', 'trick-west', 'trick-north', 'trick-east'];
  const winnerEl = document.getElementById(`${posOf(winnerIdx)}-hand`);
  if (!winnerEl) return;

  const wr = winnerEl.getBoundingClientRect();
  const targetX = wr.left + wr.width / 2;
  const targetY = wr.top + wr.height / 2;

  slotIds.forEach(slotId => {
    const slot = document.getElementById(slotId);
    if (!slot) return;
    const card = slot.querySelector('.card');
    if (!card) return;

    const cr = card.getBoundingClientRect();
    const dx = targetX - (cr.left + cr.width / 2);
    const dy = targetY - (cr.top + cr.height / 2);

    card.animate(
      [
        { transform: 'translate(0,0) scale(1)', opacity: 1 },
        { transform: `translate(${dx}px,${dy}px) scale(0.25)`, opacity: 0 },
      ],
      { duration: 1200, easing: 'ease-in', fill: 'forwards' }
    );
  });
}

function confirmDiscards() {
  const sorted = [...discardSelections].sort((a, b) => b - a);
  const willHave = state.hand.length - sorted.length;
  discardSelections.clear();
  for (const idx of sorted) {
    sendAction(idx);
  }
  if (willHave <= 5) {
    sendAction(16); // DISCARD_DONE
  }
}

// ── Main render ──

function render() {
  if (!state) return;
  renderPlayerZones();
  renderHands();
  renderTrick();
  renderTrumpBadge();
  renderInfo();
  renderSeatBids();
  renderActions();
  renderLog();

  if (state.game_over) {
    showGameOver();
  } else if (state.hand_over) {
    showHandOver();
  } else {
    document.getElementById('hand-overlay').classList.remove('visible');
  }
}

function fmtTrumpCard(cs) {
  if (!cs) return '';
  const sym = { S: '♠', H: '♥', D: '♦', C: '♣' };
  const rank = cs.slice(0, -1).replace('T', '10');
  return rank + (sym[cs.slice(-1)] || cs.slice(-1));
}

function handTeamSection(label, tricks, raw, delta, isBidTeam,
                         bidValue, bidMade, bidSuccessVal, highTrumpCard) {
  const fmt = (n) => (n > 0 ? `+${n}` : `${n}`);
  const rows = [
    `<div class="hr-row"><span>${tricks} trick` +
    `${tricks !== 1 ? 's' : ''} × 5</span><span>+${tricks * 5} pts</span></div>`,
  ];
  if (highTrumpCard) {
    rows.push(
      `<div class="hr-row highlight"><span>High trump ` +
      `${fmtTrumpCard(highTrumpCard)}</span><span>+5 pts</span></div>`);
  }
  rows.push(
    `<div class="hr-row total"><span>Earned</span><span>${raw} pts</span></div>`);
  let verdict = '';
  if (isBidTeam) {
    if (bidMade) {
      const bonus = (bidSuccessVal && bidSuccessVal !== bidValue)
        ? ` <small>(${bidValue}-for-${bidSuccessVal})</small>` : '';
      verdict = `<div class="hr-verdict made">✓ Made bid of ${bidValue}` +
        ` → ${fmt(delta)} pts${bonus}</div>`;
    } else {
      verdict = `<div class="hr-verdict fail">✗ Missed bid of ${bidValue}` +
        ` → ${fmt(delta)} pts</div>`;
    }
  } else {
    verdict = `<div class="hr-verdict">${fmt(delta)} pts this hand</div>`;
  }
  return `<div class="hr-team"><div class="hr-team-label">${label}</div>` +
    `<div class="hr-rows">${rows.join('')}</div>${verdict}</div>`;
}

function showHandOver() {
  const s = state.hand_summary || {};
  const fmt = (n) => (n > 0 ? `+${n}` : `${n}`);
  const msg = document.getElementById('hand-overlay-msg');

  // Rich breakdown when the enriched payload is present; otherwise
  // fall back to the legacy one-line summary.
  if (s.ns_tricks !== undefined && s.bid_team) {
    const nsBid = s.bid_team === 'ns';
    const nsSec = handTeamSection(
      'North / South', s.ns_tricks ?? 0, s.ns_raw ?? 0, s.d_ns ?? 0,
      nsBid, s.bid_value, s.bid_made, s.bid_success_val,
      s.high_trump_team === 'ns' ? s.high_trump_card : null);
    const ewSec = handTeamSection(
      'East / West', s.ew_tricks ?? 0, s.ew_raw ?? 0, s.d_ew ?? 0,
      !nsBid, s.bid_value, s.bid_made, s.bid_success_val,
      s.high_trump_team === 'ew' ? s.high_trump_card : null);
    msg.innerHTML =
      `<div class="hr-grid">${nsSec}${ewSec}</div>` +
      `<div class="ho-totals">Totals &nbsp; N/S ${s.ns ?? 0} ` +
      `&nbsp;·&nbsp; E/W ${s.ew ?? 0} &nbsp;<small>(first to 125)` +
      `</small></div>`;
  } else {
    msg.innerHTML =
      `This hand &nbsp;—&nbsp; <strong>N/S ${fmt(s.d_ns ?? 0)}</strong>` +
      ` &nbsp;·&nbsp; <strong>E/W ${fmt(s.d_ew ?? 0)}</strong><br>` +
      `<span class="ho-totals">Totals &nbsp; N/S ${s.ns ?? 0} ` +
      `&nbsp;·&nbsp; E/W ${s.ew ?? 0} &nbsp;<small>(first to 125)` +
      `</small></span>`;
  }
  document.getElementById('hand-overlay').classList.add('visible');
}

function continueHand() {
  document.getElementById('hand-overlay').classList.remove('visible');
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'continue_hand' }));
  }
}

// ── Player zone highlights ──

function renderPlayerZones() {
  const cp = state.current_player;
  for (let seat = 0; seat < 4; seat++) {
    const el = document.getElementById(`${posOf(seat)}-zone`);
    if (el) {
      el.classList.toggle('active-player', seat === cp && !state.game_over);
    }
  }

  // Your dealer tag (the south/bottom position is always you)
  const dt = document.getElementById('dealer-tag');
  if (dt) dt.style.display = state.dealer === _you() ? 'inline' : 'none';

  // The three non-you positions (west/north/east, relative to you)
  const labelBase = { west: 'West', north: 'North', east: 'East' };
  for (let seat = 0; seat < 4; seat++) {
    if (seat === _you()) continue;
    const pos = posOf(seat);
    const el = document.getElementById(`${pos}-label`);
    if (!el) continue;
    const dealerMark = seat === state.dealer
      ? ' <span class="dealer-tag">Dealer</span>' : '';
    const partnerMark = pos === 'north'
      ? ' <span class="partner-tag">Partner</span>' : '';
    el.innerHTML = playerLabel(seat) + dealerMark + partnerMark;
  }
}

// ── Hands ──

function renderHands() {
  // Other players: face-down cards, rotated to your view
  for (let seat = 0; seat < 4; seat++) {
    if (seat === _you()) continue;
    renderBackHand(`${posOf(seat)}-hand`, state.hand_counts[String(seat)]);
  }

  // Your hand is always the bottom (south) position
  const el = document.getElementById(`${posOf(_you())}-hand`);
  el.innerHTML = '';
  const ph = state.phase;
  const legal = state.legal_actions || [];
  const myTurn = state.is_human_turn;

  state.hand.forEach((cs, i) => {
    if (ph === PHASE.DISCARD && myTurn && legal.includes(i)) {
      const selected = discardSelections.has(i);
      const { rank, sym, red } = cardStr(cs);
      const cls = ['card', red ? 'red' : '', 'playable', selected ? 'selected-discard' : ''].filter(Boolean).join(' ');
      el.innerHTML += `<div class="${cls}" onclick="toggleDiscardSelection(${i})" title="${rank}${sym}">
        <div class="card-corner tl">${rank}<br>${sym}</div>
        <div class="card-mid">${sym}</div>
        <div class="card-corner br">${rank}<br>${sym}</div>
      </div>`;
    } else {
      const playable = myTurn && ph === PHASE.GAMEPLAY && legal.includes(i);
      el.innerHTML += cardHTML(cs, playable, i);
    }
  });
}

function renderBackHand(id, count) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = '';
  for (let i = 0; i < count; i++) {
    el.innerHTML += cardBackHTML();
  }
}

// ── Trick ──

function renderTrick() {
  const trick = state.current_trick || [null, null, null, null];
  // Each seat's played card goes to its rotated screen slot.
  for (let seat = 0; seat < 4; seat++) {
    const el = document.getElementById(`trick-${posOf(seat)}`);
    if (!el) continue;
    const cs = trick[seat];
    el.innerHTML = cs ? cardHTML(cs, false, -1) : emptySlotHTML();
  }
}

function renderTrumpBadge() {
  const badge = document.getElementById('trump-badge');
  const num = document.getElementById('trick-num');
  if (state.trump_display) {
    const isRed = state.trump_suit === 'H' || state.trump_suit === 'D';
    badge.innerHTML = `<span class="${isRed ? 'red-text' : ''}">${state.trump_display}</span>`;
  } else {
    badge.innerHTML = '';
  }
  if (state.phase === PHASE.GAMEPLAY) {
    // While the trick-winning animation plays, the engine has already
    // advanced trick_count — hold the displayed number on the trick
    // being animated; it advances once the next (non-animating) state
    // arrives. (Was a solo bug too.)
    const shown = state.trick_animating
      ? state.trick_count
      : state.trick_count + 1;
    num.textContent = `Trick ${Math.min(Math.max(shown, 1), 5)}/5`;
  } else {
    num.textContent = '';
  }
}

// ── Info panel ──

function renderInfo() {
  document.getElementById('phase-label').textContent = state.phase_name || '';

  const turnEl = document.getElementById('turn-display');
  if (state.game_over) {
    turnEl.textContent = 'Game over';
  } else if (state.is_human_turn) {
    turnEl.textContent = 'Your turn ▶';
    turnEl.style.color = '#fdd835';
  } else {
    turnEl.textContent = `${playerLabel(state.current_player)}'s turn…`;
    turnEl.style.color = '';
  }

  document.getElementById('score-ns').textContent = state.points.ns;
  document.getElementById('score-ew').textContent = state.points.ew;
  // Partnership member names (NS = seats 0&2, EW = 1&3 — absolute, not
  // rotated). Multiplayer → real names via seat_names; solo → falls
  // back to "You & North" / "West & East" (unchanged).
  const tns = document.getElementById('team-ns');
  const tew = document.getElementById('team-ew');
  if (tns) tns.textContent = `(${playerLabel(0)} & ${playerLabel(2)})`;
  if (tew) tew.textContent = `(${playerLabel(1)} & ${playerLabel(3)})`;

  renderBidBlock();
  renderTricksBlock();
}

function renderBidBlock() {
  const el = document.getElementById('bid-block');
  if (state.phase !== PHASE.AUCTION && state.phase !== PHASE.DECLARATION) {
    // Persistent through the whole hand: who won the bid, the value,
    // and trump (the user wants this visible the entire hand).
    if (state.highest_bidder !== null && state.highest_bidder !== undefined) {
      const name = playerLabel(state.highest_bidder);
      const val = state.highest_bid_value || '?';
      const sixty = val === 30 ? ' <span class="bb-bonus">(30→60)</span>' : '';
      const tr = state.trump_display
        ? ` · Trump <strong>${state.trump_display}</strong>` : '';
      el.innerHTML =
        `Bid: <strong>${name}</strong> won at <strong>${val}</strong>${sixty}${tr}`;
    } else {
      el.innerHTML = '';
    }
    return;
  }

  const bids = state.bids || {};
  const passed = state.passed || [];
  const dealer = state.dealer;
  const lines = [];

  for (let i = 0; i < 4; i++) {
    const name = playerLabel(i);
    const isDealer = i === dealer;
    const hasPassed = passed[i];
    const bid = bids[String(i)];

    let status;
    if (hasPassed) {
      status = '<span style="opacity:0.45">Pass</span>';
    } else if (bid && BID_LABEL[bid]) {
      status = `<strong>${bid === 4 ? 'Hold' : BID_LABEL[bid]}</strong>`;
    } else {
      status = '—';
    }

    const d = isDealer ? ' <span style="opacity:0.5">(D)</span>' : '';
    lines.push(`${name}${d}: ${status}`);
  }
  el.innerHTML = lines.join('<br>');
}

// Per-seat bid pill shown next to each player during the auction.
function renderSeatBids() {
  const show = state.phase === PHASE.AUCTION || state.phase === PHASE.DECLARATION;
  const bids = state.bids || {};
  const passed = state.passed || [];
  for (let p = 0; p < 4; p++) {
    // The chip element lives in the fixed zone; rotate seat p to its
    // on-screen chip so pills line up with the rotated board (#2).
    // Solo (your_seat 0) → identity → unchanged.
    const cid = (p - _you() + 4) % 4;
    const el = document.getElementById(`bid-chip-${cid}`);
    if (!el) continue;
    if (!show) {
      // After the auction: keep ONLY the winning bidder's pill on
      // their seat for the whole hand; clear everyone else.
      if (p === state.highest_bidder && state.highest_bid != null) {
        el.textContent = state.highest_bid === 4
          ? 'Hold'
          : (BID_LABEL[state.highest_bid] || state.highest_bid_value || '');
        el.className = 'bid-chip active';
      } else {
        el.textContent = '';
        el.className = 'bid-chip';
      }
      continue;
    }
    const bid = bids[String(p)];
    if (passed[p]) {
      el.textContent = 'Pass';
      el.className = 'bid-chip pass';
    } else if (bid && BID_LABEL[bid]) {
      el.textContent = bid === 4 ? 'Hold' : BID_LABEL[bid];
      el.className = 'bid-chip active';
    } else {
      el.textContent = '';
      el.className = 'bid-chip';
    }
  }
}

function renderTricksBlock() {
  const el = document.getElementById('tricks-block');
  if (state.phase !== PHASE.GAMEPLAY) { el.innerHTML = ''; return; }
  const tw = state.tricks_won || [0, 0, 0, 0];
  el.innerHTML = `Tricks — You:&nbsp;${tw[0]}&nbsp; W:&nbsp;${tw[1]}&nbsp; N:&nbsp;${tw[2]}&nbsp; E:&nbsp;${tw[3]}`;
}

// ── Action buttons ──

function renderActions() {
  const el = document.getElementById('action-area');
  if (!el) return;

  if (!state.is_human_turn || state.game_over) {
    el.innerHTML = state.game_over
      ? ''
      : '<span class="phase-prompt">Waiting…</span>';
    return;
  }

  const legal = state.legal_actions || [];

  if (state.phase === PHASE.AUCTION) {
    const defs = [
      { a: 0, label: 'Pass',   cls: 'btn-pass' },
      { a: 1, label: 'Bid 20', cls: 'btn-bid'  },
      { a: 2, label: 'Bid 25', cls: 'btn-bid'  },
      { a: 3, label: 'Bid 30', cls: 'btn-bid'  },
      { a: 4, label: 'Hold',   cls: 'btn-hold' },
    ];
    el.innerHTML = defs
      .filter(d => legal.includes(d.a))
      .map(d => `<button class="action-btn ${d.cls}" onclick="sendAction(${d.a})">${d.label}</button>`)
      .join('');

  } else if (state.phase === PHASE.DECLARATION) {
    const suits = [
      { a: 0, label: '♠ Spades',   cls: 'btn-spades'   },
      { a: 1, label: '♥ Hearts',   cls: 'btn-hearts'   },
      { a: 2, label: '♦ Diamonds', cls: 'btn-diamonds' },
      { a: 3, label: '♣ Clubs',    cls: 'btn-clubs'    },
    ];
    el.innerHTML =
      '<span class="phase-prompt">Declare trump:</span>' +
      suits.map(s =>
        `<button class="action-btn ${s.cls}" onclick="sendAction(${s.a})">${s.label}</button>`
      ).join('');

  } else if (state.phase === PHASE.DISCARD) {
    if (discardSelections.size > 0) {
      const willHave = state.hand.length - discardSelections.size;
      const label = willHave <= 5
        ? `Discard ${discardSelections.size} & Done`
        : `Discard ${discardSelections.size} Selected`;
      el.innerHTML = `<button class="action-btn btn-discard-confirm" onclick="confirmDiscards()">${label}</button>`;
    } else if (legal.includes(16)) {
      el.innerHTML = `<button class="action-btn btn-done" onclick="sendAction(16)">Done Discarding</button>`;
    } else {
      el.innerHTML = '<span class="phase-prompt">Select cards to discard</span>';
    }

  } else if (state.phase === PHASE.GAMEPLAY) {
    el.innerHTML = '<span class="phase-prompt">Click a highlighted card to play</span>';
  }
}

// ── Log ──

// The broadcast log is built server-side with absolute compass labels
// (South/West/North/East = seats 0/1/2/3). In multiplayer, swap them
// for the real lobby names so the log reads "Alice bids 20". Solo has
// no seat_names → returned unchanged (single-player log byte-identical).
// Single regex pass so an injected name can't be re-matched as another
// compass word; escHtml still runs AFTER, so names stay XSS-safe.
const _ABS_SEAT = { South: 0, West: 1, North: 2, East: 3 };
function _personalizeLog(line) {
  const names = state && state.seat_names;
  if (!names) return line;
  return line.replace(/\b(South|West|North|East)\b/g,
    (w) => names[String(_ABS_SEAT[w])] || w);
}

function renderLog() {
  const el = document.getElementById('game-log');
  const log = state.log || [];
  el.innerHTML = log.map((line, i) => {
    const isLatest = i === log.length - 1;
    const isSep = line.startsWith('Hand over') || line.startsWith('===');
    const cls = isLatest ? 'latest' : isSep ? 'hand-sep' : '';
    const gameEnd = line.startsWith('===');
    return `<div class="log-entry ${gameEnd ? 'game-end' : cls}">${escHtml(_personalizeLog(line))}</div>`;
  }).join('');
  el.scrollTop = el.scrollHeight;
}

// ── Game over ──

function showGameOver() {
  const ns = state.points.ns;
  const ew = state.points.ew;
  const winner = ns >= 125 ? 'NS wins! (You & North)' : 'EW wins! (West & East)';
  document.getElementById('overlay-title').textContent = winner;
  document.getElementById('overlay-msg').textContent = `Final score — N/S: ${ns}  •  E/W: ${ew}`;
  document.getElementById('overlay').classList.add('visible');
}

// ── Util ──

function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Mobile drawer ──
// The log + New Game button live in the right panel on desktop. On mobile
// they're relocated into a slide-up drawer (DOM moved, IDs preserved so
// render() is unaffected).

const _mq = window.matchMedia('(max-width: 820px)');

function syncResponsiveLayout() {
  const body = document.getElementById('drawer-body');
  const panel = document.getElementById('right-panel');
  const drawerItems = [
    document.getElementById('log-wrap'),
    document.getElementById('new-game-btn'),
  ];
  if (_mq.matches) {
    for (const el of drawerItems) {
      if (el && el.parentElement !== body) body.appendChild(el);
    }
  } else {
    for (const el of drawerItems) {
      if (el && el.parentElement === body) panel.appendChild(el);
    }
    closeDrawer();
  }
}

function toggleDrawer() {
  const open = document.getElementById('drawer').classList.toggle('open');
  document.getElementById('drawer-backdrop').classList.toggle('visible', open);
}

function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('drawer-backdrop').classList.remove('visible');
}

_mq.addEventListener('change', syncResponsiveLayout);

// ── Boot ──
window.addEventListener('load', () => {
  syncResponsiveLayout();
  // Multiplayer app: start on the landing screen instead of
  // auto-connecting. A ?room=CODE deep link jumps straight to that
  // room's lobby. Solo connects only when "Play Solo" is clicked
  // (then the game flow is exactly as before).
  const params = new URLSearchParams(location.search);
  const deepRoom = (params.get('room') || '').trim().toUpperCase();
  if (deepRoom) {
    showScreen('lobby');
    document.getElementById('join-code').value = deepRoom;
    // name still entered on landing; prefill then join
    _enterRoom(deepRoom);
  } else {
    // Cold start with no deep link: stay on landing, but OFFER to
    // resume a saved paused game (PR-D) — never auto-rejoin, so the
    // player can always choose to start fresh instead.
    showScreen('landing');
    renderResume();
  }
  initInstall();
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
    // A Render redeploy activates a new worker; reload once so the
    // client picks up fresh assets and a clean WebSocket.
    let reloaded = false;
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      if (reloaded) return;
      reloaded = true;
      window.location.reload();
    });
  }
});

// ── PWA install affordance ──
let _installEvent = null;

function _isStandalone() {
  return (window.matchMedia
            && window.matchMedia('(display-mode: standalone)').matches)
         || window.navigator.standalone === true;
}
function _installDismissed() {
  try { return localStorage.getItem('fortyfivesInstallDismissed') === '1'; }
  catch (e) { return false; }
}
function _markInstallDone() {
  try { localStorage.setItem('fortyfivesInstallDismissed', '1'); } catch (e) {}
}
function _hideInstallBar() {
  const bar = document.getElementById('install-bar');
  if (bar) bar.hidden = true;
}
function _showAndroidInstall() {
  if (_isStandalone() || _installDismissed()) return;
  const bar = document.getElementById('install-bar');
  const btn = document.getElementById('install-btn');
  const txt = document.getElementById('install-text');
  if (!bar || !btn || !txt) return;
  txt.textContent = 'Install Fortyfives for full-screen play';
  btn.hidden = false;
  bar.hidden = false;
}

function initInstall() {
  const bar = document.getElementById('install-bar');
  if (!bar || _isStandalone() || _installDismissed()) return;

  document.getElementById('install-dismiss').addEventListener('click', () => {
    _hideInstallBar();
    _markInstallDone();
  });

  document.getElementById('install-btn').addEventListener('click', async () => {
    if (!_installEvent) return;
    _installEvent.prompt();
    let outcome = 'dismissed';
    try { ({ outcome } = await _installEvent.userChoice); } catch (e) {}
    _installEvent = null;
    _hideInstallBar();
    if (outcome === 'accepted') _markInstallDone();
  });

  window.addEventListener('appinstalled', () => {
    _hideInstallBar();
    _markInstallDone();
  });

  // Android / desktop Chrome may have fired the event before init ran
  if (_installEvent) {
    _showAndroidInstall();
    return;
  }

  // iOS Safari never fires beforeinstallprompt — show a one-time manual
  // hint (Add to Home Screen lives in the Share sheet there).
  const ua = navigator.userAgent || '';
  const iOS = /iPad|iPhone|iPod/.test(ua)
    || (navigator.maxTouchPoints > 1 && /Macintosh/.test(ua));
  const inApp = /CriOS|FxiOS|EdgiOS|GSA|FBAN|FBAV|Instagram|Line/i.test(ua);
  if (iOS && !inApp) {
    document.getElementById('install-btn').hidden = true;
    document.getElementById('install-text').innerHTML =
      'Install: tap <strong>Share</strong>, then <strong>Add to Home Screen</strong>';
    bar.hidden = false;
  }
}

// Capture the install event as early as possible (it can fire before load)
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  _installEvent = e;
  _showAndroidInstall();
});
