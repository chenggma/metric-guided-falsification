"""End-to-end smoke: one seeded random-OU episode with all metrics logged.

Usage: python scripts/smoke.py [n_episodes]
"""

import sys
import time

import numpy as np

from falsify.env import AdversarialEnv, IDMEgoSUT
from falsify.search import OUPolicy, run_episodes


def main(n: int = 3) -> None:
    env = AdversarialEnv(sut=IDMEgoSUT(), reward_metric="sparse")
    policy = OUPolicy(rng=np.random.default_rng(0))
    t0 = time.time()
    records = run_episodes(env, policy, seeds=range(1, n + 1))
    dt = time.time() - t0
    for r in records:
        print(
            f"seed={r.seed:>3} outcome={r.outcome:<22} steps={r.steps:>4} "
            f"max: " + " ".join(f"{k}={v:.3f}" for k, v in sorted(r.max_metrics.items()))
        )
    steps = sum(r.steps for r in records)
    print(f"{len(records)} episodes, {steps} adversary steps in {dt:.1f}s "
          f"({steps / dt:.1f} steps/s incl. all-metric logging)")
    env.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
