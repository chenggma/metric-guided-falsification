"""Small-sample statistics, numpy-only (no scipy dependency).

With 5 seeds per arm the asymptotic tests are inappropriate, so the
Mann-Whitney null distribution is enumerated exactly rather than
approximated. Note the floor this imposes: for n = m = 5 the smallest
attainable two-sided p is 2/252 = 0.0079, so no pairwise contrast in M2 can
be significant at alpha = 0.001 no matter how separated the arms are. That
is a property of the seed budget and is reported alongside the p-values.
"""

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np


def rank_sum_u(a: Sequence[float], b: Sequence[float]) -> float:
    """Mann-Whitney U for sample `a` against `b` (ties get average ranks)."""
    a, b = list(a), list(b)
    pooled = sorted(a + b)
    ranks: Dict[float, float] = {}
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed average rank over the tie block
        for v in pooled[i:j + 1]:
            ranks[v] = avg
        i = j + 1
    r_a = sum(ranks[v] for v in a)
    return r_a - len(a) * (len(a) + 1) / 2.0


def mannwhitney_exact(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Exact two-sided Mann-Whitney p by enumerating all label assignments.

    Returns (U, p). Enumeration is C(n+m, n) large - fine for the 5-vs-5
    design (252) and guarded against accidental use on big samples.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan"), float("nan")
    total = n + m
    from math import comb

    if comb(total, n) > 200_000:
        raise ValueError(f"exact enumeration too large for n={n}, m={m}")

    observed = rank_sum_u(a, b)
    pooled = list(a) + list(b)
    centre = n * m / 2.0
    extreme = 0
    count = 0
    for idx in itertools.combinations(range(total), n):
        left = [pooled[i] for i in idx]
        right = [pooled[i] for i in range(total) if i not in set(idx)]
        u = rank_sum_u(left, right)
        count += 1
        if abs(u - centre) >= abs(observed - centre) - 1e-12:
            extreme += 1
    return observed, extreme / count


def bootstrap_ci(
    values: Sequence[float],
    statistic=np.median,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260814,
) -> Tuple[float, float, float]:
    """(point estimate, lo, hi) percentile bootstrap over the given values.

    Resampling is over SEEDS, which is the unit of independent replication;
    resampling episodes instead would understate run-to-run variance.
    """
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True)
    stats = statistic(draws, axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(v)), float(lo), float(hi)


def holm(pairs: List[Tuple[str, float]]) -> List[Tuple[str, float, float, bool]]:
    """Holm-Bonferroni over pre-registered contrasts.

    Returns (label, raw p, adjusted p, reject at 0.05) sorted by raw p.
    Adjusted p-values are made monotone, as the procedure requires.
    """
    ordered = sorted(pairs, key=lambda kv: kv[1])
    k = len(ordered)
    out, running = [], 0.0
    for i, (label, p) in enumerate(ordered):
        adj = min(1.0, max(running, (k - i) * p))
        running = adj
        out.append((label, p, adj, adj < 0.05))
    return out
