import unittest

import numpy as np

from falsify.stats import bootstrap_ci, holm, mannwhitney_exact, rank_sum_u


class TestMannWhitney(unittest.TestCase):
    def test_u_hand_case(self):
        # a strictly above b => U = n*m
        self.assertEqual(rank_sum_u([4, 5, 6], [1, 2, 3]), 9.0)
        self.assertEqual(rank_sum_u([1, 2, 3], [4, 5, 6]), 0.0)

    def test_ties_average_ranks(self):
        # identical samples => U = n*m/2
        self.assertEqual(rank_sum_u([1, 1], [1, 1]), 2.0)

    def test_complete_separation_5v5_hits_the_floor(self):
        # The design's best case: 2/252 is the smallest attainable p here.
        u, p = mannwhitney_exact([10, 11, 12, 13, 14], [0, 1, 2, 3, 4])
        self.assertEqual(u, 25.0)
        self.assertAlmostEqual(p, 2 / 252, places=6)

    def test_identical_samples_give_p_one(self):
        _, p = mannwhitney_exact([1, 1, 1], [1, 1, 1])
        self.assertAlmostEqual(p, 1.0)

    def test_symmetric_in_argument_order(self):
        a, b = [3, 1, 4, 1, 5], [2, 7, 1, 8, 2]
        self.assertAlmostEqual(mannwhitney_exact(a, b)[1], mannwhitney_exact(b, a)[1])

    def test_guards_against_huge_enumeration(self):
        with self.assertRaises(ValueError):
            mannwhitney_exact(list(range(30)), list(range(30)))


class TestBootstrap(unittest.TestCase):
    def test_point_estimate_is_the_statistic(self):
        est, lo, hi = bootstrap_ci([1, 2, 3, 4, 5])
        self.assertEqual(est, 3.0)
        self.assertLessEqual(lo, est)
        self.assertLessEqual(est, hi)

    def test_constant_sample_has_degenerate_ci(self):
        est, lo, hi = bootstrap_ci([7.0] * 5)
        self.assertEqual((est, lo, hi), (7.0, 7.0, 7.0))

    def test_deterministic_across_calls(self):
        self.assertEqual(bootstrap_ci([1, 5, 2, 8, 3]), bootstrap_ci([1, 5, 2, 8, 3]))

    def test_empty_returns_nan(self):
        self.assertTrue(all(np.isnan(x) for x in bootstrap_ci([])))


class TestHolm(unittest.TestCase):
    def test_scales_smallest_p_by_k(self):
        out = holm([("a", 0.01), ("b", 0.04), ("c", 0.5)])
        self.assertEqual(out[0][0], "a")
        self.assertAlmostEqual(out[0][2], 0.03)  # 3 * 0.01

    def test_adjusted_p_is_monotone(self):
        out = holm([("a", 0.02), ("b", 0.021), ("c", 0.9)])
        adj = [o[2] for o in out]
        self.assertEqual(adj, sorted(adj))

    def test_rejection_flag_at_005(self):
        out = dict((o[0], o[3]) for o in holm([("a", 0.001), ("b", 0.9)]))
        self.assertTrue(out["a"])
        self.assertFalse(out["b"])


if __name__ == "__main__":
    unittest.main()
