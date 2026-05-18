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

import web.room as roommod  # noqa: E402
from web.room import Room, rooms  # noqa: E402
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
        self.assertIsNone(room.claim_seat(a, 0))      # Alice: South (NS)
        self.assertIsNone(room.claim_seat(b, 2))      # Bob:   North (NS)
        # seat already taken
        c = FakeWS(); room.join(c, "Cara")
        self.assertIsNotNone(room.claim_seat(c, 0))

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

    def test_disconnect_frees_seat_and_bot_takes_over(self):
        room = Room.create_room()
        a, b = FakeWS(), FakeWS()
        room.join(a, "A"); room.join(b, "B")
        room.claim_seat(a, 0); room.claim_seat(b, 2)
        room.start(a)
        self.assertEqual(room.session.human_seats, {0, 2})

        room.remove_connection(a)        # Alice drops mid-game
        self.assertNotIn(0, room.session.human_seats)  # seat 0 -> bot
        self.assertEqual(room.session.human_seats, {2})
        self.assertIn(room.code, rooms)  # room still alive (Bob present)

        room.remove_connection(b)        # last player leaves
        self.assertNotIn(room.code, rooms)  # room GC'd

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

    def test_unknown_room_rejected(self):
        import asyncio as aio
        from web.server import room_ws
        ws = ScriptedWS([])
        aio.run(room_ws(ws, "ZZZZ"))
        self.assertTrue(any(m.get("type") == "error" for m in ws.sent))


if __name__ == "__main__":
    unittest.main()
