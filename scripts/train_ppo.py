"""Train one PPO adversary arm.

Usage:
  python scripts/train_ppo.py --arm pora --seed 1 --steps 100000 \
      --calibration results/calibration/normalizers.json \
      --outdir results/pilot/pora_s1
"""

import argparse

from falsify.search.rl_adversary import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["sparse", "inv_ttc", "neg_tts_margin", "pora"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--shaping", default="raw", choices=["raw", "pbrs"])
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    path = train(a.arm, a.seed, a.steps, a.outdir, calibration_path=a.calibration,
                 shaping=a.shaping)
    print(f"saved {path}")
