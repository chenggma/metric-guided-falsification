"""Evaluate a trained adversary (or the random-OU baseline) on held-out seeds.

Every method is evaluated on the SAME seed stream (same --master-seed), so
outcome differences are attributable to the policy, not the scenarios.

Usage:
  python scripts/eval_policy.py --model results/pilot/pora_s1/model.zip \
      --arm pora --calibration results/calibration/normalizers.json \
      --episodes 200 --master-seed 90001 --out results/pilot/pora_s1/eval.jsonl
  python scripts/eval_policy.py --random --episodes 200 --master-seed 90001 \
      --out results/pilot/random/eval.jsonl
"""

import argparse
import collections
import json

from falsify.search import OUPolicy
from falsify.search.rl_adversary import make_env


def main(a) -> None:
    env = make_env(
        a.arm,
        master_seed=a.master_seed,
        jsonl_path=a.out,
        calibration_path=a.calibration,
    )
    if a.random:
        policy = OUPolicy()

        def act(obs):
            return policy.act(obs)
    else:
        from stable_baselines3 import PPO

        model = PPO.load(a.model, device="cpu")

        def act(obs):
            return model.predict(obs, deterministic=True)[0]

    outcomes = collections.Counter()
    for _ in range(a.episodes):
        obs, _ = env.reset()
        if a.random:
            policy.reset(seed=env._current["seed"])
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
        outcomes[env._current["outcome"]] += 1

    total_steps = env.total_steps
    summary = {
        "policy": "random-OU" if a.random else a.model,
        "arm": a.arm,
        "episodes": a.episodes,
        "steps": total_steps,
        "outcomes": dict(outcomes),
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        with open(a.out.replace(".jsonl", "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--arm", default="sparse",
                    choices=["sparse", "inv_ttc", "neg_tts_margin", "pora"])
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--master-seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if bool(a.model) == a.random:
        ap.error("exactly one of --model / --random")
    main(a)
