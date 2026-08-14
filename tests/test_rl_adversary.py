import json
import os
import tempfile
import unittest

try:
    import stable_baselines3  # noqa: F401

    HAVE_SB3 = True
except ImportError:  # CI is torch-free by design
    HAVE_SB3 = False

from falsify.search.rl_adversary import make_env


class TestMakeEnv(unittest.TestCase):
    def test_dense_arm_requires_calibration(self):
        with self.assertRaises(ValueError):
            make_env("pora", master_seed=1, calibration_path=None)

    def test_sparse_arm_needs_no_calibration(self):
        env = make_env("sparse", master_seed=1)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)
        env.close()


@unittest.skipUnless(HAVE_SB3, "stable-baselines3 not installed")
class TestTrainSmoke(unittest.TestCase):
    def test_short_sparse_training_run(self):
        from falsify.search.rl_adversary import train

        with tempfile.TemporaryDirectory() as d:
            path = train("sparse", seed=0, total_steps=256, outdir=d)
            self.assertTrue(os.path.exists(path))
            with open(os.path.join(d, "config.json")) as f:
                cfg = json.load(f)
            self.assertEqual(cfg["arm"], "sparse")
            self.assertGreaterEqual(cfg["env_steps"], 256)
            with open(os.path.join(d, "train_episodes.jsonl")) as f:
                self.assertGreater(len(f.readlines()), 0)


if __name__ == "__main__":
    unittest.main()
