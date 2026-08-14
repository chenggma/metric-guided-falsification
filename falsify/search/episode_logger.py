"""Seeded episode logging: every episode is a replayable (seed, policy) pair.

SB3 (and any driver) resets the env without a seed between episodes, which
would leave discovered failures unreplayable. This wrapper draws each
episode's scenario seed from its own seeded stream, passes it to the inner
reset, and writes one JSONL record per episode (seed, outcome, steps,
return, per-metric maxima, crash snapshot). Replaying a logged failure is
`env.reset(seed=record["seed"])` plus the saved policy.
"""

import json
from typing import Optional

import gymnasium as gym
import numpy as np


class SeededEpisodeLogger(gym.Wrapper):
    def __init__(self, env, master_seed: int, jsonl_path: Optional[str] = None):
        super().__init__(env)
        self.master_seed = int(master_seed)
        self.seed_rng = np.random.default_rng(self.master_seed)
        self.jsonl_path = jsonl_path
        self.total_steps = 0  # budget accounting across the whole run
        self.episodes = 0
        self._current: Optional[dict] = None

    def reset(self, *, seed=None, options=None):
        ep_seed = int(self.seed_rng.integers(0, 2**31 - 1)) if seed is None else int(seed)
        obs, info = self.env.reset(seed=ep_seed, options=options)
        self._current = {
            "seed": ep_seed,
            "steps": 0,
            "return": 0.0,
            "max_metrics": {},
            "outcome": None,
            "crash_snapshot": None,
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        c = self._current
        c["steps"] += 1
        c["return"] += float(reward)
        self.total_steps += 1
        for k, v in info.get("metrics", {}).items():
            if k not in c["max_metrics"] or v > c["max_metrics"][k]:
                c["max_metrics"][k] = float(v)
        if info.get("outcome") is not None:
            c["outcome"] = info["outcome"]
        if "crash_snapshot" in info:
            c["crash_snapshot"] = info["crash_snapshot"]
        if terminated or truncated:
            if c["outcome"] is None:
                c["outcome"] = "timeout"
            c["cumulative_steps"] = self.total_steps
            self.episodes += 1
            if self.jsonl_path:
                with open(self.jsonl_path, "a") as f:
                    f.write(json.dumps(c) + "\n")
        return obs, reward, terminated, truncated, info
