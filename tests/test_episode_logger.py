import json
import os
import tempfile
import unittest

import gymnasium as gym
import numpy as np

from falsify.search.episode_logger import SeededEpisodeLogger


class _TinyEnv(gym.Env):
    """3-step episodes; records the reset seed it was given."""

    action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
    observation_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def __init__(self):
        self.seen_seeds = []

    def reset(self, *, seed=None, options=None):
        self.seen_seeds.append(seed)
        self.t = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.t += 1
        info = {"metrics": {"m": float(self.t)}, "outcome": None}
        term = self.t >= 3
        if term:
            info["outcome"] = "timeout"
        return np.zeros(1, np.float32), 1.0, term, False, info


class TestSeededEpisodeLogger(unittest.TestCase):
    def test_draws_deterministic_seed_stream(self):
        a, b = _TinyEnv(), _TinyEnv()
        wa = SeededEpisodeLogger(a, master_seed=42)
        wb = SeededEpisodeLogger(b, master_seed=42)
        for w in (wa, wb):
            for _ in range(3):
                w.reset()
        self.assertEqual(a.seen_seeds, b.seen_seeds)
        self.assertEqual(len(set(a.seen_seeds)), 3)

    def test_explicit_seed_passes_through(self):
        env = _TinyEnv()
        w = SeededEpisodeLogger(env, master_seed=0)
        w.reset(seed=777)
        self.assertEqual(env.seen_seeds, [777])
        self.assertEqual(w._current["seed"], 777)

    def test_record_written_on_done(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "eps.jsonl")
            w = SeededEpisodeLogger(_TinyEnv(), master_seed=1, jsonl_path=path)
            for _ in range(2):
                w.reset()
                done = False
                while not done:
                    _, _, term, trunc, _ = w.step(np.zeros(1))
                    done = term or trunc
            with open(path) as f:
                records = [json.loads(line) for line in f]
        self.assertEqual(len(records), 2)
        r = records[0]
        self.assertEqual(r["steps"], 3)
        self.assertEqual(r["return"], 3.0)
        self.assertEqual(r["outcome"], "timeout")
        self.assertEqual(r["max_metrics"], {"m": 3.0})
        self.assertEqual(records[1]["cumulative_steps"], 6)

    def test_budget_accounting(self):
        w = SeededEpisodeLogger(_TinyEnv(), master_seed=1)
        for _ in range(4):
            w.reset()
            for _ in range(3):
                w.step(np.zeros(1))
        self.assertEqual(w.total_steps, 12)
        self.assertEqual(w.episodes, 4)


if __name__ == "__main__":
    unittest.main()
