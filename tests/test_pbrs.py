import unittest

import numpy as np

from falsify.env import AdversarialEnv, IDMEgoSUT


class TestPBRS(unittest.TestCase):
    def test_constant_potential_pays_gamma_minus_one(self):
        # With Φ ≡ c, every PBRS step must pay exactly γ·c - c: a constant
        # potential is unfarmable by construction.
        c, gamma = 0.8, 0.99
        env = AdversarialEnv(
            sut=IDMEgoSUT(),
            reward_metric="pora",
            normalizer=lambda raw: c,
            shaping="pbrs",
            gamma=gamma,
            effort_lambda=0.0,
        )
        env.reset(seed=1)
        for _ in range(3):
            _, r, term, trunc, info = env.step(np.zeros(2))
            if term or trunc or info["outcome"] is not None:
                break
            self.assertAlmostEqual(r, gamma * c - c, places=10)
        env.close()

    def test_reset_initializes_potential(self):
        # First step's shaped term must use Φ(s0), not 0 - otherwise every
        # episode starts with a spurious +γ·Φ(s1) windfall.
        env = AdversarialEnv(
            sut=IDMEgoSUT(),
            reward_metric="pora",
            normalizer=lambda raw: 0.5,
            shaping="pbrs",
            gamma=1.0,
            effort_lambda=0.0,
        )
        env.reset(seed=1)
        _, r, term, trunc, info = env.step(np.zeros(2))
        if not (term or trunc) and info["outcome"] is None:
            self.assertAlmostEqual(r, 0.0, places=10)  # 1.0*0.5 - 0.5
        env.close()

    def test_rejects_unknown_shaping(self):
        with self.assertRaises(ValueError):
            AdversarialEnv(sut=IDMEgoSUT(), reward_metric="pora", shaping="magic")

    def test_raw_mode_unchanged(self):
        env = AdversarialEnv(
            sut=IDMEgoSUT(),
            reward_metric="pora",
            normalizer=lambda raw: 0.25,
            shaping="raw",
            effort_lambda=0.0,
        )
        env.reset(seed=1)
        _, r, term, trunc, info = env.step(np.zeros(2))
        if not (term or trunc) and info["outcome"] is None:
            self.assertAlmostEqual(r, 0.25)
        env.close()


if __name__ == "__main__":
    unittest.main()
