"""Percentile normalization of raw metric scores (DESIGN.md §5).

Raw metric scales differ wildly (inv TTC in [0, 10], PORA in [0, ~1], TTS
margin unbounded below). To compare PPO runs across reward arms without a
reward-scale confound, each metric is passed through its own empirical CDF,
estimated once from seeded random-adversary calibration rollouts and then
frozen. After calibration, every arm's shaped reward is distributionally
Uniform[0, 1] on the calibration distribution; the only thing that differs
across arms is WHICH states receive the high percentiles.
"""

import json
from bisect import bisect_right
from typing import Sequence


class PercentileNormalizer:
    """Monotone map raw score -> [0, 1] via an empirical quantile grid."""

    def __init__(self, quantiles: Sequence[float]):
        if len(quantiles) < 2:
            raise ValueError("need at least 2 quantile points")
        if any(b < a for a, b in zip(quantiles, quantiles[1:])):
            raise ValueError("quantile grid must be non-decreasing")
        self.q = list(map(float, quantiles))

    @classmethod
    def fit(cls, samples: Sequence[float], n_points: int = 1001) -> "PercentileNormalizer":
        xs = sorted(float(s) for s in samples)
        if len(xs) < 2:
            raise ValueError("need at least 2 samples")
        grid = [
            xs[min(len(xs) - 1, round(i * (len(xs) - 1) / (n_points - 1)))]
            for i in range(n_points)
        ]
        return cls(grid)

    def __call__(self, raw: float) -> float:
        """Fraction of the quantile grid strictly below `raw` (ties averaged
        by grid multiplicity), clipped to [0, 1]."""
        lo = bisect_right(self.q, float(raw))
        return lo / len(self.q)

    def to_json(self) -> str:
        return json.dumps({"quantiles": self.q})

    @classmethod
    def from_json(cls, text: str) -> "PercentileNormalizer":
        return cls(json.loads(text)["quantiles"])
