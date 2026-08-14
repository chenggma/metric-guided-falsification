"""PPO adversary training: one reward arm, one seed, fixed hyperparameters.

Hyperparameters are IDENTICAL across arms (DESIGN.md §5-§6): the only
difference between runs is the reward metric and the training seed. SB3 is
imported lazily so the core package (and CI) stays torch-free.
"""

import json
import os
import platform
from typing import Dict, Optional

from ..calibrate import PercentileNormalizer
from ..env import AdversarialEnv, IDMEgoSUT
from ..metrics import METRICS
from .episode_logger import SeededEpisodeLogger

PPO_KWARGS = dict(
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.0,
)


def load_normalizers(path: str) -> Dict[str, PercentileNormalizer]:
    with open(path) as f:
        blob = json.load(f)
    return {
        name: PercentileNormalizer(q)
        for name, q in blob["normalizers"].items()
    }


def make_env(
    arm: str,
    master_seed: int,
    jsonl_path: Optional[str] = None,
    calibration_path: Optional[str] = None,
    log_all_metrics: bool = True,
    shaping: str = "raw",
) -> SeededEpisodeLogger:
    """Adversarial env for one reward arm, wrapped for seeded logging.

    The sparse arm takes no normalizer (its shaped term is identically 0);
    every dense arm REQUIRES a calibration artifact - training with raw
    metric scales would reintroduce the reward-scale confound.

    `log_all_metrics=False` computes only the reward arm's metric per step
    (training-speed mode: PORA at ~10 ms/call dominates wall clock if every
    arm logs it). Evaluation and failure replay keep the default True.
    """
    normalizer = None
    if arm != "sparse":
        if calibration_path is None:
            raise ValueError(
                f"arm {arm!r} needs a calibration artifact (DESIGN.md §5); "
                "run scripts/calibrate.py first"
            )
        normalizer = load_normalizers(calibration_path)[arm]
    env = AdversarialEnv(
        sut=IDMEgoSUT(),
        reward_metric=arm,
        normalizer=normalizer,
        log_all_metrics=log_all_metrics,
        shaping=shaping,
        gamma=PPO_KWARGS["gamma"],  # PBRS invariance requires matching discounts
    )
    return SeededEpisodeLogger(env, master_seed=master_seed, jsonl_path=jsonl_path)


def train(
    arm: str,
    seed: int,
    total_steps: int,
    outdir: str,
    calibration_path: Optional[str] = None,
    shaping: str = "raw",
) -> str:
    """Train one PPO adversary; returns the model path."""
    from stable_baselines3 import PPO  # lazy: keeps core import torch-free

    if arm not in METRICS:
        raise ValueError(f"unknown arm {arm!r}")
    os.makedirs(outdir, exist_ok=True)
    jsonl = os.path.join(outdir, "train_episodes.jsonl")
    # Truncate: the logger appends, so a retry after an interrupted run would
    # otherwise splice the dead run's episodes onto the new one.
    open(jsonl, "w").close()
    env = make_env(arm, master_seed=seed, jsonl_path=jsonl,
                   calibration_path=calibration_path, log_all_metrics=False,
                   shaping=shaping)

    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **PPO_KWARGS)
    model.learn(total_timesteps=total_steps)

    model_path = os.path.join(outdir, "model.zip")
    model.save(model_path)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(
            {
                "arm": arm,
                "shaping": shaping,
                "seed": seed,
                "total_steps": total_steps,
                "env_episodes": env.episodes,
                "env_steps": env.total_steps,
                "ppo_kwargs": PPO_KWARGS,
                "calibration_path": calibration_path,
                "python": platform.python_version(),
            },
            f,
            indent=2,
        )
    return model_path
