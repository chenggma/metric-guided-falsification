import unittest

from falsify.actors import ActorState
from falsify.fault import ATTACKER_CRASH, THIRD_PARTY, TRIVIAL, coarse_crash_label


def _actor(x, y, vx, vy, heading=0.0):
    return ActorState(x, y, vx, vy, heading, 5.0, 2.0)


class TestCoarseCrashLabel(unittest.TestCase):
    def test_attacker_rear_strike_is_trivial(self):
        ego = _actor(0.0, 0.0, 10.0, 0.0)
        att = _actor(-5.0, 0.0, 18.0, 0.0)  # behind, closing fast
        self.assertEqual(coarse_crash_label(ego, att, True), TRIVIAL)

    def test_ego_rear_ends_braking_attacker_is_nontrivial(self):
        ego = _actor(0.0, 0.0, 15.0, 0.0)
        att = _actor(6.0, 0.0, 4.0, 0.0)  # ahead, slower (cut-in-and-brake)
        self.assertEqual(coarse_crash_label(ego, att, True), ATTACKER_CRASH)

    def test_attacker_side_ram_is_trivial(self):
        ego = _actor(0.0, 0.0, 20.0, 0.0)
        att = _actor(1.0, 3.5, 20.0, -3.0)  # beside, moving laterally into ego
        self.assertEqual(coarse_crash_label(ego, att, True), TRIVIAL)

    def test_ego_lane_change_into_attacker_is_nontrivial(self):
        ego = _actor(0.0, 0.0, 20.0, 3.0)  # ego moving laterally toward attacker
        att = _actor(1.0, 3.5, 20.0, 0.0)
        self.assertEqual(coarse_crash_label(ego, att, True), ATTACKER_CRASH)

    def test_attacker_not_crashed_is_third_party(self):
        ego = _actor(0.0, 0.0, 15.0, 0.0)
        att = _actor(6.0, 0.0, 4.0, 0.0)
        self.assertEqual(coarse_crash_label(ego, att, False), THIRD_PARTY)

    def test_attacker_far_away_is_third_party(self):
        ego = _actor(0.0, 0.0, 15.0, 0.0)
        att = _actor(40.0, 0.0, 4.0, 0.0)  # crashed elsewhere with someone else
        self.assertEqual(coarse_crash_label(ego, att, True), THIRD_PARTY)


if __name__ == "__main__":
    unittest.main()
