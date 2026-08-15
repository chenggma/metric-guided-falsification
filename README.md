# metric-guided-falsification

[![tests](https://github.com/chenggma/metric-guided-falsification/actions/workflows/tests.yml/badge.svg)](https://github.com/chenggma/metric-guided-falsification/actions/workflows/tests.yml)

RL-based falsification (adaptive stress testing) trains an adversary to
drive a system under test into failure, and its reward is typically
augmented with a dense safety-metric term - an established technique
reported to improve discovery (Corso et al. 2019; Cai et al., T-IV 2025).
What that work does not do is hold the pipeline fixed and vary *which*
metric supplies the term. This project asks one narrow empirical question:

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

**Status: M2 complete at 5 seeds/arm; extending to 7 for statistical power.**
Headline ([results/M2.md](results/M2.md)): with a properly gated outcome,
*dense shaping is necessary but only the potential-based form delivers it*.
The sparse-reward adversary found zero severe failures on all five seeds -
worse than random noise - while every raw-shaped arm collapsed into reward
farming (84-95% of late episodes end in timeout, hovering for shaped reward
and never finishing the attack) and every PBRS arm found failures on 4 of 5
seeds, PORA best (median 13/200). Three contrasts hit the exact floor of
the design (p = 0.0079, perfect separation) - which also exposed that the
registered correction scheme could never have rejected at n = 5, an
arithmetic error owned in DESIGN.md rather than buried.

Getting there required retracting two things: the project's framing (dense
shaping *is* defended in prior work - [RELATED_WORK.md](RELATED_WORK.md))
and a leaky outcome gate that let adversaries bank 131 parking-speed
"failures" per run ([results/PILOT.md](results/PILOT.md), falsify/fault.py).

<details><summary>Earlier: M1 pilot (1 seed per arm - a go/no-go, not a finding)</summary>
The pilot passed its go/no-go (the sparse-reward PPO adversary finds
non-trivial failures at ~24x the random baseline on held-out seeds, all of
them an interpretable cut-in-and-brake weakness of the IDM ego) and
*inverted* the registered H1: dense metric shaping induced reward farming
- hover near the ego, collect shaped risk, never finish the attack - with
a clean dose-response (PORA, the densest signal, fell to zero non-trivial
failures; sparse was unharmed). Numbers, mechanism evidence, and the
resulting M2 re-registration (potential-based shaping arms) are in
[results/PILOT.md](results/PILOT.md). M2 later showed both of these
readings were artifacts of the leaky outcome gate.
</details>

The full experimental contract - hypotheses registered before each run,
reward design, confound controls, and every amendment - is in
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
