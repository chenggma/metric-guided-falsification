"""Fit the per-metric percentile normalizers (DESIGN.md §5).

Runs seeded random-OU episodes, collects every step's raw metric values,
fits one PercentileNormalizer per dense metric, and writes the committed
calibration artifact with full provenance.

Usage: python scripts/calibrate.py [--episodes 100] [--master-seed 20260814]
"""

import argparse
import json
import os

import numpy as np

import gymnasium
import highway_env
import pora

from falsify.calibrate import PercentileNormalizer
from falsify.env import AdversarialEnv, IDMEgoSUT
from falsify.env.adversary_env import DEFAULT_CONFIG
from falsify.search import OUPolicy
from falsify.search.episode_logger import SeededEpisodeLogger

DENSE_METRICS = ("inv_ttc", "neg_tts_margin", "pora")


def main(episodes: int, master_seed: int, out: str) -> None:
    env = SeededEpisodeLogger(
        AdversarialEnv(sut=IDMEgoSUT(), reward_metric="sparse"),
        master_seed=master_seed,
    )
    policy = OUPolicy()
    samples = {m: [] for m in DENSE_METRICS}

    for _ in range(episodes):
        obs, _ = env.reset()
        policy.reset(seed=env._current["seed"])
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy.act(obs))
            for m in DENSE_METRICS:
                samples[m].append(info["metrics"][m])
            done = term or trunc

    blob = {
        "provenance": {
            "master_seed": master_seed,
            "episodes": episodes,
            "steps": env.total_steps,
            "policy": "OUPolicy(theta=0.15, sigma=0.4, dt=0.2), per-episode reseed",
            "env_config": DEFAULT_CONFIG,
            "versions": {
                "gymnasium": gymnasium.__version__,
                "highway_env": highway_env.__version__,
                "numpy": np.__version__,
                "pora_replication": getattr(pora, "__version__", "0.1.0"),
            },
        },
        "normalizers": {
            m: PercentileNormalizer.fit(vals).q for m, vals in samples.items()
        },
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(blob, f)
    print(f"{episodes} episodes, {env.total_steps} steps -> {out}")
    for m in DENSE_METRICS:
        v = sorted(samples[m])
        print(f"  {m:>15}: n={len(v)} min={v[0]:.3g} p50={v[len(v)//2]:.3g} max={v[-1]:.3g}")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--master-seed", type=int, default=20260814)
    ap.add_argument("--out", default="results/calibration/normalizers.json")
    a = ap.parse_args()
    main(a.episodes, a.master_seed, a.out)
