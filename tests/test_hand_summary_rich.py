"""
Backend test for the enriched hand-summary payload.

GameSession installs an end_hand() capture wrapper so the popup payload
(state['hand_summary']) carries the rich per-hand breakdown — tricks,
high-trump, bid made/missed, raw points — which the engine resets after
each hand. This verifies the captured data is present and internally
consistent. (The popup's JS/CSS rendering is not browser-tested here.)

Agents are forced to rule-based for all phases so the test is fast and
agent-independent — the capture wrapper fires on end_hand regardless of
which agent plays.
"""

import os
import sys
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "web"))

from game_session import GameSession  # noqa: E402


class TestHandSummaryRich(unittest.TestCase):
    def test_hand_summary_has_consistent_rich_fields(self):
        gs = GameSession(human_player=0)
        # Fast + agent-independent: rule-based for every phase.
        gs._pimc_agent = gs._rule_agent

        steps = 0
        while not gs.hand_over and not gs.game_over and steps < 800:
            steps += 1
            g = gs.game
            if g.current_player_id == gs.human_player:
                legal = g.get_legal_actions()  # raw game ids
                if not legal:
                    break
                gs.take_human_action(legal[0])
            else:
                gs.run_ai_turn()

        self.assertTrue(gs.hand_over,
                        f"hand did not complete within {steps} steps")

        s = gs.serialize_state()["hand_summary"]
        self.assertIsNotNone(s)

        # Legacy keys preserved (backward compatible).
        for k in ("ns", "ew", "d_ns", "d_ew"):
            self.assertIsInstance(s[k], int, f"{k} must be int")

        # Rich keys present + internally consistent.
        self.assertIn(s["bid_team"], ("ns", "ew"),
                      "a played hand has a bidding team")
        self.assertIsInstance(s["bid_made"], bool)
        self.assertEqual(s["ns_tricks"] + s["ew_tricks"], 5,
                         "a hand is exactly 5 tricks")
        self.assertIn(s["ns_raw"] + s["ew_raw"], (25, 30),
                      "raw hand points = 25 (no high trump) or 30")
        self.assertTrue(0 <= s["ns_raw"] <= 30)
        self.assertTrue(0 <= s["ew_raw"] <= 30)
        self.assertIn(s["trump_suit"], ("S", "H", "D", "C"))
        if s["high_trump_card"] is not None:
            self.assertIn(s["high_trump_team"], ("ns", "ew"))
        else:
            self.assertEqual(s["ns_raw"] + s["ew_raw"], 25,
                             "no high trump => no +5 bonus")
        # Bidding-team delta reflects make/miss; raw is non-negative.
        self.assertIsInstance(s["bid_value"], int)


if __name__ == "__main__":
    unittest.main()
