'use strict';

// ── Constants ──
const SUIT_SYM = { S: '♠', H: '♥', D: '♦', C: '♣' };
const BID_LABEL = { 0: 'Pass', 1: '20', 2: '25', 3: '30', 4: 'Hold' };
const BID_VALUES = { 0: null, 1: 20, 2: 25, 3: 30, 4: null };
const PHASE = { AUCTION: 1, DECLARATION: 2, DISCARD: 3, GAMEPLAY: 4 };
const PLAYER_NAMES = ['You', 'West', 'North', 'East'];
// Player 0 = South (human), 1 = West, 2 = North, 3 = East

let ws = null;
let state = null;
let discardSelections = new Set();
let trickAnimTimer = null;

// ── Connection ──

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => setConn(true);
  ws.onclose = () => { setConn(false); setTimeout(connect, 2000); };
  ws.onerror = () => setConn(false);
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'state') {
      if (!state || state.phase !== data.phase || data.game_over) {
        discardSelections.clear();
      }
      // Cancel any pending animation when a non-animating state arrives
      if (!data.trick_animating && trickAnimTimer !== null) {
        clearTimeout(trickAnimTimer);
        trickAnimTimer = null;
      }
      state = data;
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
    }
  };
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
  const handIds = ['south-hand', 'west-hand', 'north-hand', 'east-hand'];
  const winnerEl = document.getElementById(handIds[winnerIdx]);
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
  renderActions();
  renderLog();

  if (state.game_over) {
    showGameOver();
  }
}

// ── Player zone highlights ──

function renderPlayerZones() {
  const cp = state.current_player;
  const zones = { 0: 'south-zone', 1: 'west-zone', 2: 'north-zone', 3: 'east-zone' };
  for (const [p, id] of Object.entries(zones)) {
    const el = document.getElementById(id);
    if (el) {
      el.classList.toggle('active-player', parseInt(p) === cp && !state.game_over);
    }
  }

  // Dealer tags — show for whichever player is dealer
  document.getElementById('dealer-tag').style.display = state.dealer === 0 ? 'inline' : 'none';

  // Update other player labels with dealer indicator
  const labelIds = { 1: 'west-label', 2: 'north-label', 3: 'east-label' };
  const labelBase = { 1: 'West', 2: 'North', 3: 'East' };
  for (const [p, id] of Object.entries(labelIds)) {
    const el = document.getElementById(id);
    if (!el) continue;
    const dealerMark = parseInt(p) === state.dealer
      ? ' <span class="dealer-tag">Dealer</span>' : '';
    const partnerMark = parseInt(p) === 2
      ? ' <span class="partner-tag">Partner</span>' : '';
    el.innerHTML = labelBase[p] + dealerMark + partnerMark;
  }
}

// ── Hands ──

function renderHands() {
  // Other players: face-down cards
  renderBackHand('north-hand', state.hand_counts['2']);
  renderBackHand('west-hand', state.hand_counts['1']);
  renderBackHand('east-hand', state.hand_counts['3']);

  // Human hand
  const el = document.getElementById('south-hand');
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
  // Player 0 = south, 1 = west, 2 = north, 3 = east
  const slots = { 'trick-south': 0, 'trick-west': 1, 'trick-north': 2, 'trick-east': 3 };

  for (const [slotId, playerIdx] of Object.entries(slots)) {
    const el = document.getElementById(slotId);
    if (!el) continue;
    const cs = trick[playerIdx];
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
    num.textContent = `Trick ${state.trick_count + 1}/5`;
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
    turnEl.textContent = `${PLAYER_NAMES[state.current_player]}'s turn…`;
    turnEl.style.color = '';
  }

  document.getElementById('score-ns').textContent = state.points.ns;
  document.getElementById('score-ew').textContent = state.points.ew;

  renderBidBlock();
  renderTricksBlock();
}

function renderBidBlock() {
  const el = document.getElementById('bid-block');
  if (state.phase !== PHASE.AUCTION && state.phase !== PHASE.DECLARATION) {
    // Show final bid result during play phases
    if (state.highest_bidder !== null && state.highest_bidder !== undefined) {
      const name = PLAYER_NAMES[state.highest_bidder];
      const val = state.highest_bid_value || '?';
      el.innerHTML = `Bid: ${name} at ${val}`;
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
    const name = PLAYER_NAMES[i];
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

function renderLog() {
  const el = document.getElementById('game-log');
  const log = state.log || [];
  el.innerHTML = log.map((line, i) => {
    const isLatest = i === log.length - 1;
    const isSep = line.startsWith('Hand over') || line.startsWith('===');
    const cls = isLatest ? 'latest' : isSep ? 'hand-sep' : '';
    const gameEnd = line.startsWith('===');
    return `<div class="log-entry ${gameEnd ? 'game-end' : cls}">${escHtml(line)}</div>`;
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

function syncDrawerContents() {
  const body = document.getElementById('drawer-body');
  const panel = document.getElementById('right-panel');
  const items = [
    document.getElementById('log-wrap'),
    document.getElementById('new-game-btn'),
  ];
  if (_mq.matches) {
    for (const el of items) {
      if (el && el.parentElement !== body) body.appendChild(el);
    }
  } else {
    for (const el of items) {
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

_mq.addEventListener('change', syncDrawerContents);

// ── Boot ──
window.addEventListener('load', () => {
  syncDrawerContents();
  connect();
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  }
});
