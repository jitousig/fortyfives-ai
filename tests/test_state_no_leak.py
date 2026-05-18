"""
CANARY: the per-seat serialized state must never leak another seat's
cards. This is the hidden-information boundary for multiplayer — a leak
here means every client can see opponents' hands. Treat a failure like
the eval canary: stop everything.

Tested at the freshly-dealt auction state (no cards played yet), so the
ONLY place any card may legitimately appear in a seat's payload is that
seat's own `hand`. Also checks the action-authority boundary and
single-player parity.
"""

import os
import sys
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "web"))

from game_session import GameSession, card_to_str  # noqa: E402


def _all_strings(obj):
    """Yield every string anywhere in a nested dict/list structure."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_strings(v)


class TestStateNoLeak(unittest.TestCase):
    def test_no_other_seat_cards_in_per_seat_state(self):
        gs = GameSession(human_player=0)
        g = gs.game
        n = g.num_players
        hands = g.hands
        per_seat_cards = {
            s: {card_to_str(c) for c in (
                hands.get(s, []) if isinstance(hands, dict)
                else (hands[s] if s < len(hands) else []))}
            for s in range(n)
        }

        for s in range(n):
            state = gs.serialize_state_for(s)

            # 1. The seat sees exactly its own hand.
            self.assertEqual(set(state["hand"]), per_seat_cards[s],
                             f"seat {s} hand wrong")

            # 2. No other seat's card appears ANYWHERE in the payload
            #    (no cards are played yet, so nothing public can contain
            #    them legitimately).
            forbidden = set()
            for other in range(n):
                if other != s:
                    forbidden |= per_seat_cards[other]
            forbidden -= per_seat_cards[s]  # ignore genuine dups if any
            leaked = forbidden.intersection(_all_strings(state))
            self.assertFalse(
                leaked,
                f"LEAK: seat {s} payload exposes other seats' cards: "
                f"{sorted(leaked)}")

    def test_action_authority_boundary(self):
        gs = GameSession(human_player=0)
        g = gs.game
        bot_seat = next(s for s in range(g.num_players)
                        if s not in gs.human_seats)
        # A bot seat is never given legal actions, even on its turn.
        st = gs.serialize_state_for(bot_seat)
        self.assertEqual(st["legal_actions"], [])
        self.assertFalse(st["is_human_turn"])
        # And the server cannot submit an action for a non-human seat.
        err = gs.take_seat_action(bot_seat, 0)
        self.assertIsNotNone(err)
        self.assertIn("not human-controlled", err["error"])

    def test_single_player_parity(self):
        # Back-compat: serialize_state() == serialize_state_for(solo seat)
        gs = GameSession(human_player=0)
        self.assertEqual(gs.serialize_state(), gs.serialize_state_for(0))


if __name__ == "__main__":
    unittest.main()
