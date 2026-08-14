"""Random-OU baseline: the floor every search method must beat.

Ornstein-Uhlenbeck noise over the adversary action space - temporally
correlated, so the attacker's behavior is sustained maneuvers rather than
white-noise jitter (which would be a strawman baseline).
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


class OUPolicy:
    """dx = theta * (mu - x) dt + sigma dW, clipped to the action box."""

    def __init__(self, theta: float = 0.15, sigma: float = 0.4, dt: float = 0.2,
                 rng: Optional[np.random.Generator] = None):
        self.theta, self.sigma, self.dt = theta, sigma, dt
        self.rng = rng or np.random.default_rng()
        self.x = np.zeros(2)

    def reset(self) -> None:
        self.x = np.zeros(2)

    def act(self, obs=None) -> np.ndarray:
        dx = -self.theta * self.x * self.dt + self.sigma * np.sqrt(self.dt) * self.rng.standard_normal(2)
        self.x = np.clip(self.x + dx, -1.0, 1.0)
        return self.x.copy()


@dataclass
class EpisodeRecord:
    seed: int
    outcome: str
    steps: int
    return_: float
    max_metrics: dict = field(default_factory=dict)
    crash_snapshot: Optional[dict] = None

    def to_json(self) -> str:
        d = self.__dict__.copy()
        d["return"] = d.pop("return_")
        return json.dumps(d)


def run_episodes(env, policy, seeds, jsonl_path: Optional[str] = None) -> List[EpisodeRecord]:
    """Roll `policy` for one episode per seed; log outcome + per-metric maxima."""
    records = []
    sink = open(jsonl_path, "a") if jsonl_path else None
    try:
        for seed in seeds:
            policy.reset()
            obs, _ = env.reset(seed=int(seed))
            done, steps, total = False, 0, 0.0
            max_m: dict = {}
            outcome, snapshot = "timeout", None
            while not done:
                obs, r, term, trunc, info = env.step(policy.act(obs))
                total += r
                steps += 1
                for k, v in info["metrics"].items():
                    if k not in max_m or v > max_m[k]:
                        max_m[k] = v
                if info["outcome"] is not None:
                    outcome = info["outcome"]
                snapshot = info.get("crash_snapshot", snapshot)
                done = term or trunc
            rec = EpisodeRecord(
                seed=int(seed),
                outcome=outcome,
                steps=steps,
                return_=float(total),
                max_metrics={k: float(v) for k, v in max_m.items()},
                crash_snapshot=snapshot,
            )
            records.append(rec)
            if sink:
                sink.write(rec.to_json() + "\n")
    finally:
        if sink:
            sink.close()
    return records
