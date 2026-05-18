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
import uuid

from web.game_session import GameSession, card_to_str

# code -> Room. In-memory only (lost on server restart; acceptable for
# v1 — see RESEARCH.md / the multiplayer plan).
rooms: dict[str, "Room"] = {}

TRICK_DISPLAY_SECS = 4.0
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
        # ws -> seat (int) or None (in lobby, not yet seated)
        self.conns: dict[object, object] = {}
        self.names: dict[object, str] = {}

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

    def join(self, ws, name: str) -> None:
        self.conns[ws] = None
        self.names[ws] = (name or "Player").strip()[:20] or "Player"

    def _seat_taken_by(self, seat):
        for w, s in self.conns.items():
            if s == seat:
                return w
        return None

    def claim_seat(self, ws, seat) -> dict | None:
        if self.started:
            return {"error": "Game already started"}
        if not isinstance(seat, int) or not (0 <= seat < 4):
            return {"error": "Invalid seat"}
        holder = self._seat_taken_by(seat)
        if holder is not None and holder is not ws:
            return {"error": f"Seat {SEAT_NAME[seat]} is taken"}
        self.conns[ws] = seat
        return None

    def leave_seat(self, ws) -> None:
        if ws in self.conns:
            self.conns[ws] = None

    def claimed_seats(self) -> set:
        return {s for s in self.conns.values() if isinstance(s, int)}

    def start(self, ws) -> dict | None:
        if self.started:
            return {"error": "Already started"}
        seats = self.claimed_seats()
        if not seats:
            return {"error": "Claim a seat before starting"}
        # Humans = claimed seats; every other seat is a bot.
        self.session.human_seats = set(seats)
        self.started = True
        return None

    def remove_connection(self, ws) -> None:
        seat = self.conns.pop(ws, None)
        self.names.pop(ws, None)
        # If a seated human leaves mid-game, hand the seat to a bot so
        # the game never stalls (full reconnect-by-token is PR-C).
        if self.started and isinstance(seat, int):
            self.session.human_seats.discard(seat)
        if not self.conns:
            rooms.pop(self.code, None)

    def reset_session(self) -> None:
        """Fresh game, keeping connections/seats. For a room, re-arm the
        lobby unless seats are still claimed (rematch keeps seats)."""
        self.session = GameSession(human_player=0)
        if self.solo:
            return
        seats = self.claimed_seats()
        if seats:
            self.session.human_seats = set(seats)
            self.started = True
        else:
            self.started = False

    # ---- serialization ------------------------------------------------
    def lobby_state_for(self, ws) -> dict:
        my_seat = self.conns.get(ws)
        seats = []
        for s in range(4):
            holder = self._seat_taken_by(s)
            seats.append({
                "seat": s,
                "name": SEAT_NAME[s],
                "partnership": SEAT_PARTNERSHIP[s],
                "claimed_by": self.names.get(holder) if holder else None,
                "is_you": holder is ws,
            })
        return {
            "type": "lobby",
            "code": self.code,
            "started": self.started,
            "your_seat": my_seat if isinstance(my_seat, int) else None,
            "seats": seats,
            "can_start": (not self.started) and bool(self.claimed_seats()),
        }

    def _frozen(self, seat) -> dict:
        """Per-seat state with the last completed trick frozen for
        animation (multiplayer-safe; mirrors server._trick_complete_state
        but per recipient seat)."""
        st = self.session.serialize_state_for(seat)
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
                    await ws.send_json(
                        self.session.serialize_state_for(seat))
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
                await asyncio.sleep(TRICK_DISPLAY_SECS)
                await self.broadcast()
            else:
                await asyncio.sleep(delay)
                await self.broadcast()
            if not more:
                break
        await self.broadcast()
