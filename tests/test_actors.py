import math
import unittest

from falsify.actors import ActorState, foes_within, from_vehicle


class _Stub:
    LENGTH = 5.0
    WIDTH = 2.0

    def __init__(self, x, y, heading, speed):
        self.position = [x, y]
        self.heading = heading
        self.speed = speed


class TestFromVehicle(unittest.TestCase):
    def test_velocity_decomposition(self):
        s = from_vehicle(_Stub(10.0, -3.0, math.pi / 2, 8.0))
        self.assertAlmostEqual(s.x, 10.0)
        self.assertAlmostEqual(s.y, -3.0)
        self.assertAlmostEqual(s.vx, 0.0, places=12)
        self.assertAlmostEqual(s.vy, 8.0)
        self.assertAlmostEqual(s.speed, 8.0)
        self.assertEqual(s.length, 5.0)
        self.assertEqual(s.width, 2.0)

    def test_heading_zero_moves_along_x(self):
        s = from_vehicle(_Stub(0.0, 0.0, 0.0, 12.5))
        self.assertAlmostEqual(s.vx, 12.5)
        self.assertAlmostEqual(s.vy, 0.0, places=12)


class TestFoesWithin(unittest.TestCase):
    def test_radius_filter(self):
        ego = ActorState(0, 0, 0, 0, 0, 5, 2)
        near = ActorState(30, 0, 0, 0, 0, 5, 2)
        boundary = ActorState(60, 0, 0, 0, 0, 5, 2)
        far = ActorState(60.1, 0, 0, 0, 0, 5, 2)
        got = foes_within(ego, [near, boundary, far], 60.0)
        self.assertEqual(got, [near, boundary])


if __name__ == "__main__":
    unittest.main()
