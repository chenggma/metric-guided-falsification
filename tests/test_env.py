import unittest

import numpy as np

from falsify.env import AdversarialEnv, IDMEgoSUT
from falsify.env.adversary_env import ACCEL_MAX, LAT_ACCEL_MAX, STEER_ABS_MAX


def _env(**kw):
    return AdversarialEnv(sut=IDMEgoSUT(), reward_metric="sparse", **kw)


class TestReset(unittest.TestCase):
    def test_obs_shape_matches_space(self):
        env = _env()
        obs, info = env.reset(seed=1)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertIn("attacker_index", info)
        env.close()

    def test_same_seed_same_initial_obs(self):
        env = _env()
        a, _ = env.reset(seed=7)
        b, _ = env.reset(seed=7)
        np.testing.assert_array_equal(a, b)
        env.close()

    def test_ego_is_idm_and_attacker_is_kinematic(self):
        from highway_env.vehicle.behavior import IDMVehicle
        from highway_env.vehicle.kinematics import Vehicle

        env = _env()
        env.reset(seed=1)
        e = env.inner.unwrapped
        self.assertIsInstance(e.vehicle, IDMVehicle)
        self.assertIs(type(env.attacker), Vehicle)  # exactly kinematics, no IDM brain
        self.assertIn(env.attacker, e.road.vehicles)
        env.close()


class TestStep(unittest.TestCase):
    def test_attacker_obeys_commands(self):
        env = _env()
        env.reset(seed=1)
        v0 = env.attacker.speed
        for _ in range(5):
            env.step(np.array([1.0, 0.0]))  # full throttle, straight
        accelerated = env.attacker.speed
        env.reset(seed=1)
        for _ in range(5):
            env.step(np.array([-1.0, 0.0]))  # full braking
        braked = env.attacker.speed
        self.assertGreater(accelerated, v0)
        self.assertLess(braked, accelerated)
        env.close()

    def test_steer_bound_is_speed_dependent(self):
        env = _env()
        env.reset(seed=1)
        v = max(env.attacker.speed, 1.0)
        bound = env._steer_bound()
        self.assertLessEqual(bound, STEER_ABS_MAX)
        lat_acc = v * v * np.tan(bound) / env.attacker.LENGTH
        self.assertLessEqual(lat_acc, LAT_ACCEL_MAX + 1e-6)
        env.close()

    def test_info_carries_all_metrics(self):
        env = _env()
        env.reset(seed=1)
        _, _, _, _, info = env.step(np.zeros(2))
        for k in ("sparse", "inv_ttc", "neg_tts_margin", "pora"):
            self.assertIn(k, info["metrics"])
        self.assertEqual(info["risk_raw"], info["metrics"]["sparse"])
        env.close()

    def test_episode_terminates(self):
        env = _env()
        env.reset(seed=2)
        rng = np.random.default_rng(0)
        for i in range(200):
            _, _, term, trunc, info = env.step(rng.uniform(-1, 1, 2))
            if term or trunc:
                break
        self.assertTrue(term or trunc)
        self.assertIsNotNone(info["outcome"])
        env.close()

    def test_trivial_ram_gets_no_bonus(self):
        # Hold full throttle straight at whatever is ahead; if the attacker
        # rear-strikes the ego, the step reward must not contain the bonus.
        env = _env(effort_lambda=0.0)
        for seed in range(1, 30):
            env.reset(seed=seed)
            for _ in range(150):
                _, r, term, trunc, info = env.step(np.array([1.0, 0.0]))
                if term or trunc:
                    break
            if info["outcome"] == "ego_attacker_crash_trivial":
                self.assertLess(r, env.crash_bonus / 2)
                env.close()
                return
        env.close()
        self.skipTest("no trivial ram found in 30 seeds")


class TestRewardArms(unittest.TestCase):
    def test_unknown_metric_rejected(self):
        with self.assertRaises(ValueError):
            AdversarialEnv(sut=IDMEgoSUT(), reward_metric="nope")

    def test_normalizer_applies_to_reward_arm(self):
        env = AdversarialEnv(
            sut=IDMEgoSUT(),
            reward_metric="pora",
            normalizer=lambda raw: 0.25,
            effort_lambda=0.0,
        )
        env.reset(seed=1)
        _, r, term, trunc, info = env.step(np.zeros(2))
        if not (term or trunc):
            self.assertAlmostEqual(r, 0.25)
        self.assertEqual(info["risk_shaped"], 0.25)
        env.close()


if __name__ == "__main__":
    unittest.main()
