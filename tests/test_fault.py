import unittest

from falsify.actors import ActorState
from falsify.fault import (
    ATTACKER_CRASH,
    LOW_SPEED,
    THIRD_PARTY,
    TRIVIAL,
    coarse_crash_label,
    closing_speed,
)


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
        att = _actor(1.0, 3.5, 20.0, -6.0)  # beside, cutting laterally into ego
        self.assertEqual(coarse_crash_label(ego, att, True), TRIVIAL)

    def test_ego_lane_change_into_attacker_is_nontrivial(self):
        ego = _actor(0.0, 0.0, 20.0, 6.0)  # ego moving laterally toward attacker
        att = _actor(1.0, 3.5, 20.0, 0.0)
        self.assertEqual(coarse_crash_label(ego, att, True), ATTACKER_CRASH)


class TestSeverityGate(unittest.TestCase):
    """The gate added after M2 seeds 3-4: a parking-speed contact is not a
    falsification, whoever is at fault."""

    def test_creeping_contact_is_low_speed_not_a_failure(self):
        # The observed exploit: ego ground down to a crawl, attacker nudges it.
        ego = _actor(0.0, 0.0, 2.1, 0.0)
        att = _actor(6.0, 0.0, 0.0, 0.0)
        self.assertEqual(coarse_crash_label(ego, att, True), LOW_SPEED)

    def test_severity_gate_precedes_fault(self):
        # Even a textbook attacker rear-strike is LOW_SPEED below the floor:
        # the gate is checked first, so no low-speed contact can pay.
        ego = _actor(0.0, 0.0, 2.0, 0.0)
        att = _actor(-5.0, 0.0, 4.0, 0.0)
        self.assertEqual(coarse_crash_label(ego, att, True), LOW_SPEED)

    def test_highway_cut_in_and_brake_survives_the_gate(self):
        # The genuine failure the sparse arm found: ego 17 m/s, closing ~9.
        ego = _actor(0.0, 0.0, 17.2, 0.0)
        att = _actor(6.0, 0.0, 7.5, 0.0)
        self.assertGreaterEqual(closing_speed(ego, att), 5.0)
        self.assertEqual(coarse_crash_label(ego, att, True), ATTACKER_CRASH)

    def test_closing_speed_is_relative_speed_magnitude(self):
        ego = _actor(0.0, 0.0, 10.0, 0.0)
        att = _actor(5.0, 0.0, 10.0, 4.0)  # same longitudinal speed, lateral 4
        self.assertAlmostEqual(closing_speed(ego, att), 4.0)

    def test_boundary_is_inclusive(self):
        ego = _actor(0.0, 0.0, 0.0, 0.0)
        att = _actor(6.0, 0.0, -5.0, 0.0)  # exactly 5.0 m/s closing
        self.assertNotEqual(coarse_crash_label(ego, att, True), LOW_SPEED)

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
