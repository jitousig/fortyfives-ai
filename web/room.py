"""
Room: one game shared by N connections, each bound to a seat.

PR-A: per-seat-filtered broadcast plumbing + solo rooms (unchanged).
PR-B1: real multiplayer rooms — a lobby (join / claim a seat / start),
bots fill unclaimed seats, and a SINGLE shared turn-driver serialized
by an asyncio.Lock so concurrent connections can't corrupt the game.

Hidden-information boundary still lives in
GameSession.serialize_state_for (regression test:
tests/test_state_no_leak.py + tests/test_multiplayer_room.py).

Solo `/ws` is untouched: create_solo / reset_session / the inline
solo driver in server.py keep PR-A's exact behaviour. The multiplayer
path (`/ws/{code}`) uses advance_and_broadcast below.
"""

from __future__ import annotations  # py3.9: allow `X | None` hints

import asyncio
import time
import uuid

from web.game_session import GameSession, card_to_str

# code -> Room. In-memory only (lost on server restart; acceptable for
# v1 — see RESEARCH.md / the multiplayer plan).
rooms: dict[str, "Room"] = {}

# A paused multiplayer room (claimed seats, nobody connected) is kept in
# memory so players can come back — but not forever. Reaped after this
# many seconds with zero live connections (server restart also clears
# all rooms regardless). Product decision: 1 week.
IDLE_ROOM_TTL_SECS = 7 * 24 * 60 * 60
REAP_INTERVAL_SECS = 60 * 60  # sweep hourly

TRICK_DISPLAY_SECS = 4.0
# Final-trick hold before the hand-summary popup. MUST outlast the
# client's fly-to-winner animation (game.js: 2.5s lead-in + 1.2s
# flight = 3.7s) PLUS a ~5s read pause → 9.0 (margin for jitter).
# Keep in sync with server.py HAND_END_TRICK_SECS / game.js. (#4)
HAND_END_TRICK_SECS = 9.0
SEAT_PARTNERSHIP = {0: "NS", 1: "EW", 2: "NS", 3: "EW"}
SEAT_NAME = {0: "South", 1: "West", 2: "North", 3: "East"}


def _new_code() -> str:
    # Short, human-readable, unambiguous (no 0/O/1/I/L).
    import random
    alpha = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    while True:
        c = "".join(random.choice(alpha) for _ in range(4))
        if c not in rooms:
            return c


class Room:
    def __init__(self, code: str, solo: bool):
        self.code = code
        self.solo = solo
        self.session = GameSession(human_player=0)
        self.started = solo  # solo starts immediately; rooms wait in lobby
        # Created lazily on first use so Room() is safe to construct
        # outside a running loop (py3.9 / tests); the server only
        # touches .lock from within the event loop.
        self._lock = None
        # Live sockets only: ws -> seat (int) or None (lobby/spectator).
        self.conns: dict[object, object] = {}
        # Join-time names, transient per socket (copied into a seat on
        # claim).
        self.names: dict[object, str] = {}
        # PERSISTENT claims, independent of any socket: seat -> {name,
        # token}. A claimed seat survives disconnect (PR-C: no bot ever
        # takes a human seat; the game pauses and waits for rejoin).
        self.seats: dict[int, dict] = {}
        # monotonic ts of when the room last went to zero live sockets
        # (None while anyone is connected). Drives the idle reaper so a
        # forever-paused room can't leak memory (PR-D).
        self.empty_since: float | None = None

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()  # binds to the running loop
        return self._lock

    # ---- construction -------------------------------------------------
    @classmethod
    def create_solo(cls) -> "Room":
        """Today's one-player room (seat 0; bots fill 1-3) — PR-A."""
        code = uuid.uuid4().hex[:8]
        room = cls(code, solo=True)
        rooms[code] = room
        return room

    @classmethod
    def create_room(cls) -> "Room":
        """A multiplayer room: starts in a lobby; humans claim seats;
        unclaimed seats become bots when the game starts."""
        code = _new_code()
        room = cls(code, solo=False)
        rooms[code] = room
        return room

    # ---- connection / lobby ------------------------------------------
    def add_connection(self, ws, seat) -> None:
        self.conns[ws] = seat
        self.empty_since = None  # someone's here → not idle

    def join(self, ws, name: str) -> None:
        self.conns[ws] = None
        self.names[ws] = (name or "Player").strip()[:20] or "Player"
        self.empty_since = None

    def _ws_at_seat(self, seat):
        for w, s in self.conns.items():
            if s == seat:
                return w
        return None

    def seat_connected(self, seat) -> bool:
        """A claimed seat with a live socket bound to it."""
        return self._ws_at_seat(seat) is not None

    def waiting_seats(self) -> list:
        """Claimed seats whose player is currently disconnected — the
        game pauses on these (no bot takeover)."""
        return [s for s in sorted(self.seats) if not self.seat_connected(s)]

    def claim_seat(self, ws, seat) -> dict | None:
        if self.started:
            return {"error": "Game already started"}
        if not isinstance(seat, int) or not (0 <= seat < 4):
            return {"error": "Invalid seat"}
        if seat in self.seats and self.conns.get(ws) != seat:
            return {"error": f"Seat {SEAT_NAME[seat]} is taken"}
        # Release any seat this socket already held (moving seats).
        prev = self.conns.get(ws)
        if isinstance(prev, int) and prev in self.seats:
            self.seats.pop(prev, None)
        token = uuid.uuid4().hex
        self.seats[seat] = {
            "name": self.names.get(ws, "Player"),
            "token": token,
        }
        self.conns[ws] = seat
        return {"token": token, "seat": seat}

    def leave_seat(self, ws) -> None:
        if self.started:
            return  # mid-game: a leave is a disconnect → seat reserved
        seat = self.conns.get(ws)
        if isinstance(seat, int):
            self.seats.pop(seat, None)
        self.conns[ws] = None

    def rejoin(self, ws, token):
        """Re-bind a socket to the seat that owns `token` (server-side
        validation — never trust a client-claimed seat). Returns the
        seat, or None if the token is unknown (stale/lost)."""
        for s, info in self.seats.items():
            if info.get("token") == token:
                self.conns[ws] = s
                self.empty_since = None  # player's back → cancel reap
                return s
        return None

    def claimed_seats(self) -> set:
        return set(self.seats.keys())

    def start(self, ws) -> dict | None:
        if self.started:
            return {"error": "Already started"}
        if not self.seats:
            return {"error": "Claim a seat before starting"}
        # Humans = claimed seats (permanent). Only seats nobody claimed
        # AT START become bots; a claimed seat is never botted.
        self.session.human_seats = set(self.seats.keys())
        self.started = True
        return None

    def remove_connection(self, ws) -> None:
        # Drop the live socket ONLY. The seat claim (name+token)
        # persists so the game pauses and the player can rejoin
        # indefinitely. No bot takeover; human_seats untouched.
        self.conns.pop(ws, None)
        self.names.pop(ws, None)
        # GC when no live socket remains AND nothing is worth keeping:
        #  - solo: transient, drop it (unchanged from pre-PR-C);
        #  - multiplayer with NO claimed seats: an abandoned lobby.
        # A multiplayer room with claimed seats SURVIVES an all-
        # disconnect so players can rejoin (documented v1 in-memory
        # limitation; the game is paused, not lost).
        if not self.conns and (self.solo or not self.seats):
            rooms.pop(self.code, None)
        elif not self.conns and self.empty_since is None:
            # Survives (claimed seats, game paused) but now has zero
            # live sockets — start the idle clock for the reaper (PR-D).
            self.empty_since = time.monotonic()

    def reset_session(self) -> None:
        """Fresh game, keeping claimed seats (rematch)."""
        self.session = GameSession(human_player=0)
        if self.solo:
            return
        if self.seats:
            self.session.human_seats = set(self.seats.keys())
            self.started = True
        else:
            self.started = False

    # ---- serialization ------------------------------------------------
    def lobby_state_for(self, ws) -> dict:
        my_seat = self.conns.get(ws)
        seats = []
        for s in range(4):
            claimed = s in self.seats
            seats.append({
                "seat": s,
                "name": SEAT_NAME[s],
                "partnership": SEAT_PARTNERSHIP[s],
                "claimed_by": self.seats[s]["name"] if claimed else None,
                "connected": self.seat_connected(s),
                "is_you": my_seat == s,
            })
        return {
            "type": "lobby",
            "code": self.code,
            "started": self.started,
            "your_seat": my_seat if isinstance(my_seat, int) else None,
            "seats": seats,
            "can_start": (not self.started) and bool(self.seats),
            "waiting_for": [self.seats[s]["name"]
                            for s in self.waiting_seats()],
        }

    def _seat_names(self) -> dict:
        """seat -> display name: the claiming human, else 'Bot'."""
        return {str(s): (self.seats[s]["name"] if s in self.seats
                         else "Bot") for s in range(4)}

    def _game_state(self, seat) -> dict:
        st = self.session.serialize_state_for(seat)
        if not self.solo:
            st["seat_names"] = self._seat_names()
            # Non-empty => game is paused waiting on these (disconnected)
            # players; the client shows a "waiting for …" banner instead
            # of a silent freeze. The turn driver already won't advance a
            # claimed human seat, so this IS the pause.
            st["waiting_for"] = [self.seats[s]["name"]
                                 for s in self.waiting_seats()]
        return st

    def _frozen(self, seat) -> dict:
        """Per-seat state with the last completed trick frozen for
        animation (multiplayer-safe; mirrors server._trick_complete_state
        but per recipient seat)."""
        st = self._game_state(seat)
        g = self.session.game
        if getattr(g, "last_completed_trick", None):
            st["current_trick"] = [
                card_to_str(c) if c is not None else None
                for c in g.last_completed_trick
            ]
        st["trick_animating"] = True
        st["last_trick_winner"] = getattr(
            g, "last_completed_trick_winner", None)
        return st

    # ---- broadcast ----------------------------------------------------
    async def broadcast(self, builder=None) -> None:
        """Send each connection its OWN seat's filtered view (or the
        lobby view if it hasn't been seated / game not started)."""
        dead = []
        for ws in list(self.conns.keys()):
            seat = self.conns.get(ws)
            seated = isinstance(seat, int)
            try:
                if not self.started or not seated:
                    # In lobby, OR an unseated connection in a running
                    # game (spectator-with-no-seat): NEVER send a hand —
                    # only the lobby view. Hidden-info safe.
                    await ws.send_json(self.lobby_state_for(ws))
                elif builder is not None:
                    await ws.send_json(builder(seat))
                else:
                    await ws.send_json(self._game_state(seat))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove_connection(ws)

    async def broadcast_lobby(self) -> None:
        dead = []
        for ws in list(self.conns.keys()):
            try:
                await ws.send_json(self.lobby_state_for(ws))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove_connection(ws)

    # ---- shared turn driver (multiplayer path) -----------------------
    async def advance_and_broadcast(self, delay: float = 0.5) -> None:
        """Run bot seats one turn at a time (with trick-freeze + pacing),
        broadcasting per-seat state, until a human seat must act or the
        hand/game pauses. MUST be called holding self.lock."""
        s = self.session
        safety = 0
        await self.broadcast()
        while (not s.game_over
               and not s.hand_over
               and s.game.current_player_id not in s.human_seats
               and safety < 200):
            pre = s.game.total_tricks_completed
            more = s.run_ai_turn()
            safety += 1
            if s.game.total_tricks_completed > pre:
                await self.broadcast(self._frozen)
                await asyncio.sleep(
                    HAND_END_TRICK_SECS if s.hand_over
                    else TRICK_DISPLAY_SECS)
                await self.broadcast()
            else:
                await asyncio.sleep(delay)
                await self.broadcast()
            if not more:
                break
        await self.broadcast()


# ---- idle-room reaper (PR-D) -----------------------------------------
def reap_idle_rooms(ttl: float = IDLE_ROOM_TTL_SECS) -> list:
    """Drop multiplayer rooms with zero live sockets for longer than
    `ttl` seconds. Solo rooms and any room with a live connection are
    untouched. Returns the reaped codes. Pure/synchronous so it's
    trivially unit-testable (the loop below just schedules it)."""
    now = time.monotonic()
    reaped = []
    for code, room in list(rooms.items()):
        if room.solo or room.conns or room.empty_since is None:
            continue
        if now - room.empty_since >= ttl:
            rooms.pop(code, None)
            reaped.append(code)
    return reaped


async def idle_room_reaper(
        interval: float = REAP_INTERVAL_SECS,
        ttl: float = IDLE_ROOM_TTL_SECS) -> None:
    """Background sweep started on server startup; never returns."""
    while True:
        await asyncio.sleep(interval)
        try:
            reap_idle_rooms(ttl)
        except Exception:
            pass  # a reaper crash must never take down the server
