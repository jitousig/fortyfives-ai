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

from game_session import (  # noqa: E402
    GameSession, card_to_str, format_card, PLAYER_NAMES,
    PHASE_AUCTION, PHASE_DECLARATION, PHASE_DISCARD,
)


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

    def test_discards_are_face_down_in_broadcast_log(self):
        """Discards in 45s are face down: the shared log (broadcast
        identically to every seat) must NEVER name a discarded card —
        only an aggregate count. Regression for the prod-blocker leak."""
        gs = GameSession(human_player=0)
        gs.human_seats = {0, 1, 2, 3}  # drive every seat deterministically
        g = gs.game

        # Drive auction→declaration→discard: first actor bids 20, the
        # rest pass, winner declares the first legal trump.
        guard = 0
        while g.phase != PHASE_DISCARD and not gs.game_over and guard < 80:
            guard += 1
            p = g.current_player_id
            legal = g.get_legal_actions()
            if g.phase == PHASE_AUCTION:
                a = 1 if 1 in legal else (0 if 0 in legal else legal[0])
            else:  # declaration (or anything unexpected): first legal
                a = legal[0]
            self.assertIsNone(gs.take_seat_action(p, a),
                              f"unexpected error driving phase {g.phase}")
        self.assertEqual(g.phase, PHASE_DISCARD,
                         "test setup never reached the discard phase")

        # The seat on turn discards (down to legal, then Done),
        # recording every card it threw.
        discarder = g.current_player_id
        thrown = []
        guard = 0
        while (g.phase == PHASE_DISCARD
               and g.current_player_id == discarder and guard < 20):
            guard += 1
            legal = g.get_legal_actions()
            non_done = [x for x in legal if x != 16]
            if 16 in legal and thrown:
                action = 16  # DISCARD_DONE once we're allowed to stop
            elif non_done:
                action = non_done[0]
                hand = gs._get_hand(discarder)
                card = hand[action]
                thrown.append((card_to_str(card), format_card(card)))
            else:
                action = 16
            self.assertIsNone(gs.take_seat_action(discarder, action))

        self.assertTrue(thrown, "discarder never discarded a card")

        # 1. No discarded card appears, in ANY representation, as a
        #    SUBSTRING anywhere in ANY seat's broadcast payload. The log
        #    embeds cards in a sentence, so an exact-token check is not
        #    enough — the leak is a substring of a log line.
        forbidden = set()
        for s, f in thrown:
            forbidden.add(s)
            forbidden.add(f)
        for o in range(4):
            payload = gs.serialize_state_for(o)
            blob = "".join(_all_strings(payload))
            leaked = sorted(t for t in forbidden if t in blob)
            self.assertFalse(
                leaked,
                f"LEAK: seat {o} sees discarded card(s) {leaked} "
                f"(discards must be face down)")

        # 2. The log still tells everyone HOW MANY were discarded.
        log_text = "\n".join(gs.serialize_state_for(0)["log"])
        self.assertIn(
            f"{PLAYER_NAMES[discarder]}: discards {len(thrown)} card",
            log_text,
            f"expected an aggregate discard-count line; log:\n{log_text}")

    def test_single_player_parity(self):
        # Back-compat: serialize_state() == serialize_state_for(solo seat)
        gs = GameSession(human_player=0)
        self.assertEqual(gs.serialize_state(), gs.serialize_state_for(0))


if __name__ == "__main__":
    unittest.main()
