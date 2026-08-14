import unittest

from falsify.calibrate import PercentileNormalizer


class TestPercentileNormalizer(unittest.TestCase):
    def test_maps_into_unit_interval(self):
        n = PercentileNormalizer.fit(range(100), n_points=101)
        for raw in (-1e9, 0, 42, 99, 1e9):
            self.assertGreaterEqual(n(raw), 0.0)
            self.assertLessEqual(n(raw), 1.0)

    def test_monotone(self):
        n = PercentileNormalizer.fit([0, 1, 2, 5, 10, 50, 100], n_points=11)
        xs = [-5, 0, 0.5, 1, 3, 7, 20, 60, 100, 200]
        ys = [n(x) for x in xs]
        self.assertEqual(ys, sorted(ys))

    def test_below_min_is_zero_above_max_is_one(self):
        n = PercentileNormalizer.fit([10, 20, 30], n_points=3)
        self.assertEqual(n(5), 0.0)
        self.assertEqual(n(31), 1.0)

    def test_median_maps_near_half(self):
        n = PercentileNormalizer.fit(range(1001), n_points=1001)
        self.assertAlmostEqual(n(500), 0.5, delta=0.01)

    def test_modal_floor_value_maps_to_zero(self):
        # "no conflict" ties with most of the calibration mass and must earn 0
        n = PercentileNormalizer.fit([0.0] * 80 + [1.0, 2.0] * 10, n_points=101)
        self.assertEqual(n(0.0), 0.0)

    def test_json_roundtrip(self):
        n = PercentileNormalizer.fit([1.5, 2.5, 3.5, 9.0], n_points=5)
        m = PercentileNormalizer.from_json(n.to_json())
        for x in (0, 2, 3, 8, 10):
            self.assertEqual(n(x), m(x))

    def test_rejects_decreasing_grid(self):
        with self.assertRaises(ValueError):
            PercentileNormalizer([3, 2, 1])


if __name__ == "__main__":
    unittest.main()
