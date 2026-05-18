"""
Room: one game shared by N connections, each bound to a seat.

PR-A scope = the *plumbing*: a Room owns one GameSession and a
{websocket: seat} map, and broadcasts per-seat-filtered state to every
connection (each sees only its own hand — the hidden-information
boundary lives in GameSession.serialize_state_for). Single-player is
just a Room with one connection on seat 0; behaviour is unchanged.

The lobby / join-by-code / seat-claiming UX is PR-B; the global
`rooms` registry is here so PR-B can build on it without re-plumbing.
"""

import uuid

from web.game_session import GameSession

# code -> Room. In-memory only (lost on server restart; acceptable for
# v1 — see RESEARCH.md / the multiplayer plan).
rooms: dict[str, "Room"] = {}


class Room:
    def __init__(self, code: str):
        self.code = code
        self.session = GameSession(human_player=0)
        # websocket -> seat index it controls
        self.conns: dict[object, int] = {}

    @classmethod
    def create_solo(cls) -> "Room":
        """A private one-player room (seat 0; bots fill 1-3) — today's
        single-player experience, now expressed via the Room model."""
        code = uuid.uuid4().hex[:8]
        room = cls(code)
        rooms[code] = room
        return room

    def add_connection(self, ws, seat: int) -> None:
        self.conns[ws] = seat

    def remove_connection(self, ws) -> None:
        self.conns.pop(ws, None)
        if not self.conns:
            rooms.pop(self.code, None)

    def reset_session(self) -> None:
        """Start a fresh game in this room, keeping connections/seats."""
        self.session = GameSession(human_player=0)

    async def broadcast(self, builder=None) -> None:
        """Send each connection its OWN seat's filtered view.

        builder(seat) -> dict; defaults to the per-seat state. A failing
        socket is dropped rather than breaking the broadcast to others.
        """
        if builder is None:
            builder = self.session.serialize_state_for
        dead = []
        for ws, seat in list(self.conns.items()):
            try:
                await ws.send_json(builder(seat))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove_connection(ws)
