# metric-guided-falsification

[![tests](https://github.com/chenggma/metric-guided-falsification/actions/workflows/tests.yml/badge.svg)](https://github.com/chenggma/metric-guided-falsification/actions/workflows/tests.yml)

RL-based falsification (adaptive stress testing) trains an adversary to
drive a system under test into failure - and the adversarial reward is
almost always a hand-picked risk proxy whose choice is rarely defended.
This project asks one narrow empirical question:

> When the *only* thing that changes between falsification runs is the risk
> metric used as the dense adversarial reward - PORA, inverse TTC, TTS
> margin, or nothing (sparse) - do the discovered failures differ in how
> fast they are found, how diverse they are, and whether failures found
> under metric A are even visible to metric B?

Third repository in a sequence:
[pora-replication](https://github.com/chenggma/pora-replication)
(independent implementation of the PORA metric) →
[risk-metric-bench](https://github.com/chenggma/risk-metric-bench)
(passive scoring: PORA wins AUROC but its fixed-FPR alarm fires latest) →
**this** (the metric becomes the optimization target of an active
adversary).

**Status: M0 - environment, metrics, baselines, and tests. No experimental
results yet; nothing here should be cited as a finding.** The full
experimental contract - hypotheses registered before any run, reward
design, confound controls, evaluation plan, milestones - is in
[DESIGN.md](DESIGN.md).

## What exists

* `falsify/env/` - a Gymnasium env whose **agent is the attacker**: one NPC
  near the ego loses its IDM brain and is driven by bounded continuous
  `(acceleration, steering)`; the ego is driven by the SUT (currently
  IDM+MOBIL; a frozen learned ego is milestone M3). Steering authority is
  bounded by lateral acceleration (4 m/s² at current speed), not by a fixed
  angle - a fixed 0.3 rad at highway speed is a super-human input that
  leaves the road in under a second.
* `falsify/metrics.py` - the same three metric adapters as
  risk-metric-bench, same defaults, plus the `sparse` control arm. Every
  step logs **all** metrics regardless of the reward arm, so cross-metric
  transfer analysis needs no re-simulation.
* `falsify/fault.py` - coarse online crash labeling. The crash bonus is
  paid only for non-trivial ego crashes (attacker not the striking party);
  otherwise every reward arm converges on "ram the ego."
* `falsify/calibrate.py` - per-metric percentile normalization, the control
  for the reward-scale confound (DESIGN.md §5).
* `falsify/search/` - random-OU baseline (temporally correlated noise, the
  floor every method must beat). CMA-ES and PPO arms are milestone M1.

## Try it

```bash
pip install -e .
python scripts/smoke.py 8    # seeded random-OU episodes, all metrics logged
```

## Tests

```bash
python -m unittest discover -s tests
```

36 unit tests (kinematic extraction, metric adapters with analytic cases,
fault-labeling geometry, normalizer properties, env contract incl.
same-seed determinism and speed-dependent steering bound); CI on Python
3.10 and 3.12.

MIT license.
