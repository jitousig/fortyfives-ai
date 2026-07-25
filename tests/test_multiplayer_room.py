"""
PR-B1: multiplayer Room logic — lobby/seat machine, bots fill unclaimed
seats, disconnect frees a seat, and (CANARY) per-seat no-leak still
holds with MULTIPLE human connections sharing one game.

Headless: a FakeWS records broadcast payloads; no ASGI/browser. Driven
at the freshly-dealt auction state (no cards played) so the only place
any card may appear in a seat's payload is that seat's own hand.
"""

import asyncio
import os
import sys
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "web"))

import time  # noqa: E402

import web.room as roommod  # noqa: E402
from web.room import Room, rooms, reap_idle_rooms  # noqa: E402
from game_session import card_to_str  # noqa: E402


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, d):
        self.sent.append(d)


def _strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _strings(v)


class TestMultiplayerRoom(unittest.TestCase):
    def tearDown(self):
        # don't leak test rooms into the module registry
        for c in [c for c, r in list(rooms.items()) if not r.solo]:
            rooms.pop(c, None)

    def test_lobby_claim_partnership_and_start(self):
        room = Room.create_room()
        self.assertFalse(room.solo)
        self.assertFalse(room.started)
        self.assertIn(room.code, rooms)

        a, b = FakeWS(), FakeWS()
        room.join(a, "Alice")
        room.join(b, "Bob")
        # claim_seat now returns a reconnect token (PR-C) on success.
        ra = room.claim_seat(a, 0)                     # Alice: South (NS)
        rb = room.claim_seat(b, 2)                     # Bob:   North (NS)
        self.assertEqual(ra["seat"], 0)
        self.assertTrue(ra["token"])
        self.assertEqual(rb["seat"], 2)
        self.assertNotEqual(ra["token"], rb["token"])
        # seat already taken → error, claim refused.
        c = FakeWS(); room.join(c, "Cara")
        self.assertIn("error", room.claim_seat(c, 0))

        ls = room.lobby_state_for(a)
        self.assertEqual(ls["type"], "lobby")
        s0 = next(s for s in ls["seats"] if s["seat"] == 0)
        s2 = next(s for s in ls["seats"] if s["seat"] == 2)
        self.assertEqual(s0["partnership"], "NS")
        self.assertEqual(s2["partnership"], "NS")
        self.assertEqual(s0["claimed_by"], "Alice")
        self.assertTrue(s0["is_you"])

        self.assertIsNone(room.start(a))
        self.assertTrue(room.started)
        # Humans = exactly the claimed seats; 1 & 3 are bots.
        self.assertEqual(room.session.human_seats, {0, 2})
        for bot in (1, 3):
            self.assertNotIn(bot, room.session.human_seats)
        self.assertIsNotNone(room.start(a))  # can't start twice

    def test_no_leak_multi_human(self):
        room = Room.create_room()
        a, b = FakeWS(), FakeWS()
        room.join(a, "A"); room.join(b, "B")
        room.claim_seat(a, 0)
        room.claim_seat(b, 1)            # opposite partnership (EW)
        room.start(a)

        roommod.TRICK_DISPLAY_SECS = 0   # keep test fast
        asyncio.run(room.broadcast())

        g = room.session.game
        hands = g.hands
        cards = {s: {card_to_str(c) for c in (
            hands.get(s, []) if isinstance(hands, dict)
            else (hands[s] if s < len(hands) else []))} for s in range(4)}

        for ws, seat in ((a, 0), (b, 1)):
            payload = ws.sent[-1]
            self.assertIn(payload.get("type"), ("state", "lobby"))
            if payload["type"] != "state":
                continue
            self.assertEqual(set(payload["hand"]), cards[seat])
            forbidden = set()
            for o in range(4):
                if o != seat:
                    forbidden |= cards[o]
            forbidden -= cards[seat]
            leak = forbidden.intersection(_strings(payload))
            self.assertFalse(
                leak, f"LEAK to seat {seat}: {sorted(leak)}")

    def test_disconnect_keeps_seat_human_no_bot(self):
        """PR-C: a claimed seat is NEVER converted to a bot on
        disconnect. The seat stays human and the game pauses on it."""
        room = Room.create_room()
        a, b = FakeWS(), FakeWS()
        room.join(a, "Alice"); room.join(b, "Bob")
        room.claim_seat(a, 0); room.claim_seat(b, 2)
        room.start(a)
        self.assertEqual(room.session.human_seats, {0, 2})

        room.remove_connection(a)        # Alice's phone locks mid-game
        # Seat 0 stays human (no bot takeover); game pauses on it.
        self.assertEqual(room.session.human_seats, {0, 2})
        self.assertFalse(room.seat_connected(0))
        self.assertTrue(room.seat_connected(2))
        self.assertEqual(room.waiting_seats(), [0])
        self.assertIn(room.code, rooms)  # room kept alive for reconnect

        room.remove_connection(b)        # everyone gone, game in progress
        # Room SURVIVES an all-disconnect while a game is running so
        # players can come back (documented v1 in-memory limitation).
        self.assertIn(room.code, rooms)
        self.assertEqual(room.waiting_seats(), [0, 2])

    def test_rejoin_by_token_restores_seat_and_resumes(self):
        room = Room.create_room()
        a, b = FakeWS(), FakeWS()
        room.join(a, "Alice"); room.join(b, "Bob")
        tok = room.claim_seat(a, 0)["token"]
        room.claim_seat(b, 2)
        room.start(a)

        room.remove_connection(a)
        self.assertEqual(room.waiting_seats(), [0])

        a2 = FakeWS()                    # Alice reopens the app
        seat = room.rejoin(a2, tok)
        self.assertEqual(seat, 0)
        self.assertTrue(room.seat_connected(0))
        self.assertEqual(room.waiting_seats(), [])  # game un-pauses
        self.assertEqual(room.session.human_seats, {0, 2})

        # A bad/stale token is refused (server falls back to lobby).
        self.assertIsNone(room.rejoin(FakeWS(), "deadbeef"))

    def test_rejoin_no_leak(self):
        """No-leak canary still holds after a reconnect."""
        room = Room.create_room()
        a, b = FakeWS(), FakeWS()
        room.join(a, "A"); room.join(b, "B")
        tok = room.claim_seat(a, 0)["token"]
        room.claim_seat(b, 1)
        room.start(a)
        room.remove_connection(a)
        a2 = FakeWS()
        room.rejoin(a2, tok)

        roommod.TRICK_DISPLAY_SECS = 0
        asyncio.run(room.broadcast())

        g = room.session.game
        hands = g.hands
        cards = {s: {card_to_str(c) for c in (
            hands.get(s, []) if isinstance(hands, dict)
            else (hands[s] if s < len(hands) else []))} for s in range(4)}
        payload = a2.sent[-1]
        if payload.get("type") == "state":
            self.assertEqual(set(payload["hand"]), cards[0])
            forbidden = (cards[1] | cards[2] | cards[3]) - cards[0]
            self.assertFalse(
                forbidden.intersection(_strings(payload)))

    def test_solo_unaffected_by_reconnect_model(self):
        room = Room.create_solo()
        ws = FakeWS()
        room.add_connection(ws, 0)
        self.assertTrue(room.solo)
        self.assertTrue(room.started)
        room.remove_connection(ws)       # solo: no seats dict → GC'd
        self.assertNotIn(room.code, rooms)

    def test_advance_drives_bots_no_leak(self):
        room = Room.create_room()
        a = FakeWS()
        room.join(a, "Solo-ish"); room.claim_seat(a, 0); room.start(a)
        roommod.TRICK_DISPLAY_SECS = 0
        # Bots (seats 1,2,3) act until it's seat 0's turn or a pause.
        asyncio.run(room.advance_and_broadcast(delay=0))
        last = a.sent[-1]
        self.assertIn(last.get("type"), ("state", "lobby"))
        if last["type"] == "state":
            g = room.session.game
            hands = g.hands
            mine = {card_to_str(c) for c in (
                hands.get(0, []) if isinstance(hands, dict)
                else (hands[0] if hands else []))}
            others = set()
            for o in (1, 2, 3):
                others |= {card_to_str(c) for c in (
                    hands.get(o, []) if isinstance(hands, dict)
                    else (hands[o] if o < len(hands) else []))}
            others -= mine
            self.assertFalse(others.intersection(_strings(last)))


class ScriptedWS:
    """Drives the real server handler dependency-free: replays scripted
    client messages, then signals disconnect."""

    def __init__(self, script):
        self._script = list(script)
        self.sent = []

    async def accept(self):
        pass

    async def close(self):
        pass

    async def send_json(self, d):
        self.sent.append(d)

    async def receive_json(self):
        from fastapi import WebSocketDisconnect
        if not self._script:
            raise WebSocketDisconnect(1000)
        return self._script.pop(0)


class TestRoomWsHandlerSmoke(unittest.TestCase):
    def tearDown(self):
        for c in [c for c, r in list(rooms.items()) if not r.solo]:
            rooms.pop(c, None)

    def test_handler_lobby_flow_and_disconnect_cleanup(self):
        import asyncio as aio
        from web.server import room_ws

        code = Room.create_room().code
        ws = ScriptedWS([
            {"type": "join", "name": "Zoe"},
            {"type": "claim_seat", "seat": 1},
            {"type": "leave_seat"},
        ])
        aio.run(room_ws(ws, code))

        # Got the initial lobby + a lobby update after each message.
        lobbies = [m for m in ws.sent if m.get("type") == "lobby"]
        self.assertGreaterEqual(len(lobbies), 3)
        after_claim = lobbies[2]  # connect, join, claim
        s1 = next(s for s in after_claim["seats"] if s["seat"] == 1)
        self.assertEqual(s1["claimed_by"], "Zoe")
        self.assertEqual(s1["partnership"], "EW")
        self.assertTrue(s1["is_you"])
        # Disconnect cleanup ran: last connection gone → room GC'd.
        self.assertNotIn(code, rooms)

    def test_handler_rejoin_by_token_resumes(self):
        """End-to-end through the real /ws handler: claim → start →
        drop → reconnect with the issued token resumes the same seat."""
        import asyncio as aio
        from web.server import room_ws

        roommod.TRICK_DISPLAY_SECS = 0
        room = Room.create_room()
        code = room.code

        ws1 = ScriptedWS([
            {"type": "join", "name": "Zoe"},
            {"type": "claim_seat", "seat": 0},
            {"type": "start"},
        ])
        aio.run(room_ws(ws1, code))
        seat_msg = next(m for m in ws1.sent if m.get("type") == "seat")
        token = seat_msg["token"]
        self.assertEqual(seat_msg["seat"], 0)
        # Seat persisted across the disconnect; game paused on it.
        self.assertIn(code, rooms)
        self.assertEqual(room.waiting_seats(), [0])

        ws2 = ScriptedWS([{"type": "rejoin", "token": token}])
        aio.run(room_ws(ws2, code))
        # Reconnect bound seat 0 and resumed (no reconnect_failed).
        self.assertFalse(
            any(m.get("error") == "reconnect_failed" for m in ws2.sent))
        self.assertTrue(
            any(m.get("type") == "state" for m in ws2.sent))

    def test_handler_rejoin_bad_token_falls_back(self):
        import asyncio as aio
        from web.server import room_ws
        code = Room.create_room().code
        ws = ScriptedWS([{"type": "rejoin", "token": "nope"}])
        aio.run(room_ws(ws, code))
        self.assertTrue(
            any(m.get("error") == "reconnect_failed" for m in ws.sent))
        self.assertTrue(any(m.get("type") == "lobby" for m in ws.sent))

    def test_unknown_room_rejected(self):
        import asyncio as aio
        from web.server import room_ws
        ws = ScriptedWS([])
        aio.run(room_ws(ws, "ZZZZ"))
        self.assertTrue(any(m.get("type") == "error" for m in ws.sent))


class TestIdleRoomReaper(unittest.TestCase):
    """PR-D: a forever-paused multiplayer room must not leak memory —
    it's reaped after IDLE_ROOM_TTL_SECS with zero live sockets."""

    def tearDown(self):
        for c in [c for c, r in list(rooms.items()) if not r.solo]:
            rooms.pop(c, None)

    def test_idle_claimed_room_is_reaped(self):
        room = Room.create_room()
        a = FakeWS()
        room.add_connection(a, None)
        room.join(a, "Alice")
        room.claim_seat(a, 0)
        room.start(a)
        self.assertIsNone(room.empty_since)  # connected → not idle

        room.remove_connection(a)            # everyone gone, game paused
        self.assertIn(room.code, rooms)      # survives for reconnect
        self.assertIsNotNone(room.empty_since)

        # Not idle long enough → kept.
        self.assertEqual(reap_idle_rooms(ttl=10_000), [])
        self.assertIn(room.code, rooms)

        # Backdate the idle clock past the TTL → reaped.
        room.empty_since = time.monotonic() - 20_000
        self.assertIn(room.code, reap_idle_rooms(ttl=10_000))
        self.assertNotIn(room.code, rooms)

    def test_reaper_spares_connected_and_solo(self):
        live = Room.create_room()
        a = FakeWS()
        live.add_connection(a, None); live.join(a, "A")
        live.claim_seat(a, 0); live.start(a)
        # Force a stale idle ts, but a live socket is still attached →
        # the reaper must skip it (guard is `room.conns`, not the ts).
        live.empty_since = time.monotonic() - 99_999
        reaped = reap_idle_rooms(ttl=0)
        self.assertNotIn(live.code, reaped)
        self.assertIn(live.code, rooms)

        solo = Room.create_solo()
        solo.empty_since = time.monotonic() - 99_999
        reap_idle_rooms(ttl=0)
        self.assertIn(solo.code, rooms)      # solo never reaped here
        rooms.pop(solo.code, None)

    def test_reconnect_clears_idle_clock(self):
        room = Room.create_room()
        a = FakeWS()
        room.add_connection(a, None); room.join(a, "A")
        tok = room.claim_seat(a, 0)["token"]
        room.start(a)
        room.remove_connection(a)
        self.assertIsNotNone(room.empty_since)

        room.rejoin(FakeWS(), tok)           # player comes back
        self.assertIsNone(room.empty_since)  # reap clock cancelled
        self.assertEqual(reap_idle_rooms(ttl=0), [])
        self.assertIn(room.code, rooms)


if __name__ == "__main__":
    unittest.main()
