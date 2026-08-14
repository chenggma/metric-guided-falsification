import unittest

from falsify.actors import ActorState
from falsify.metrics import (
    TTS_FLOOR,
    inv_ttc,
    neg_tts_margin,
    pora_score,
    sparse,
)


def _ego(speed=10.0):
    return ActorState(0.0, 0.0, speed, 0.0, 0.0, 5.0, 2.0)


def _foe(x, vx):
    return ActorState(x, 0.0, vx, 0.0, 0.0, 5.0, 2.0)


class TestSparse(unittest.TestCase):
    def test_always_zero(self):
        self.assertEqual(sparse(_ego(), []), 0.0)
        self.assertEqual(sparse(_ego(), [_foe(10.0, -5.0)]), 0.0)


class TestInvTtc(unittest.TestCase):
    def test_no_foes(self):
        self.assertEqual(inv_ttc(_ego(), []), 0.0)

    def test_approaching_positive(self):
        self.assertGreater(inv_ttc(_ego(10.0), [_foe(50.0, 0.0)]), 0.0)

    def test_receding_zero(self):
        # foe ahead and faster: no conflict, TTC infinite
        self.assertEqual(inv_ttc(_ego(10.0), [_foe(50.0, 20.0)]), 0.0)

    def test_closer_is_riskier(self):
        far = inv_ttc(_ego(10.0), [_foe(80.0, 0.0)])
        near = inv_ttc(_ego(10.0), [_foe(20.0, 0.0)])
        self.assertGreater(near, far)


class TestNegTtsMargin(unittest.TestCase):
    def test_no_foes_floor(self):
        self.assertEqual(neg_tts_margin(_ego(), []), TTS_FLOOR)

    def test_receding_floor(self):
        self.assertEqual(neg_tts_margin(_ego(10.0), [_foe(50.0, 20.0)]), TTS_FLOOR)

    def test_closer_is_riskier(self):
        far = neg_tts_margin(_ego(10.0), [_foe(80.0, 0.0)])
        near = neg_tts_margin(_ego(10.0), [_foe(20.0, 0.0)])
        self.assertGreater(near, far)


class TestPora(unittest.TestCase):
    def test_no_foes(self):
        self.assertEqual(pora_score(_ego(), []), 0.0)

    def test_approaching_beats_receding(self):
        approaching = pora_score(_ego(10.0), [_foe(30.0, -10.0)])
        receding = pora_score(_ego(10.0), [_foe(30.0, 25.0)])
        self.assertGreater(approaching, receding)

    def test_nonnegative(self):
        self.assertGreaterEqual(pora_score(_ego(10.0), [_foe(30.0, -10.0)]), 0.0)


if __name__ == "__main__":
    unittest.main()
