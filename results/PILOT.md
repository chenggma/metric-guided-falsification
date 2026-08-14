# Pilot result (M1): dense risk-metric shaping suppressed falsification

**Scope guard: 1 seed per arm, 100k env steps each, SUT-A (IDM+MOBIL) only,
scenario family F1 only. This is the pilot the design calls for, not the
experiment. It answers the go/no-go and it re-registers hypotheses for M2;
nothing here is a paper claim yet.**

Setup: PPO adversary per DESIGN.md §5-§6 (identical hyperparameters across
arms; only the reward metric differs). Held-out evaluation: 200 episodes,
deterministic policy, same seed stream (master 90001) for every arm and for
the random-OU baseline. Non-trivial = ego crash where the attacker was not
the striking party (coarse label, falsify/fault.py).

## Held-out evaluation (200 episodes each, identical seeds)

| Arm | Non-trivial failures | per 10k steps | Timeout share | Mean ep len |
|---|---|---|---|---|
| **sparse** | **15/200** | **49.4** | 0% | 15.2 |
| inv_ttc | 5/200 | 2.2 | 91.5% | 114.5 |
| neg_tts_margin | 0/200 | 0.0 | 92.0% | 115.3 |
| pora | 0/200 | 0.0 | 97.5% | 119.4 |
| random-OU | 1/200 | 2.1 | 0% | 23.5 |

Go/no-go: **passed** - the sparse arm beats random-OU ~24x per step.
But the ranking is the *opposite* of registered H1: the denser the shaped
signal, the worse the falsifier.

## Mechanism: reward farming (a local optimum the dense gradient locks in)

Last quarter of training, per arm:

| Arm | Timeout share | Timeout-episode mean return | Non-trivial /10k |
|---|---|---|---|
| sparse | 0% | - | 32.3 |
| inv_ttc | 77.7% | 20.9 | 5.2 |
| neg_tts_margin | 85.8% | 35.2 | 0.8 |
| pora | 93.3% | 63.1 | 0.0 |

Three observations pin the mechanism:

1. **Dense arms farm instead of crashing.** A pora-arm timeout episode
   collects ~63 of shaped reward by hovering near the ego with risk held
   high for the full 120 steps. The accessible crash (a trivial ram) pays 0
   *and* ends the episode, so the shaped signal actively rewards not
   finishing the attack.
2. **Farming is locally, not globally, optimal.** A non-trivial crash at
   step ~12 (the sparse arm's typical time-to-crash) is worth
   0.99¹²·150 ≈ 133 discounted. Farming pays less, under discounting that
   matches: at the pora arm's late-training rate (63.1 undiscounted over
   121 steps), spreading that rate across the episode gives at most
   ≈ 36.7 discounted - and less than that in fact, since the shaped rate
   is not uniform. The same holds for the other arms (TTS ≤ 20.5,
   inv_ttc ≤ 12.2). Even compared *undiscounted*, farming (63.1) loses to
   the discounted crash (133). The dense gradient nevertheless pulls the
   policy into the hover basin and keeps it there.
3. **Dose-response.** Farmability tracks how much of the state space the
   metric scores above zero (calibration medians: pora 0.018 > 0 almost
   everywhere; inv_ttc exactly 0 until velocities conflict; TTS at its
   floor without a conflict). The suppression ranks in the same order:
   pora worst (0 non-trivial, 97.5% timeouts), then TTS, then inv_ttc,
   sparse unharmed. The dense arms even *regressed*: each found non-trivial
   failures in early training (first at 3.0-4.8k steps) and then learned
   away from them (pora: 0.00/10k in the last quarter).

## What the sparse arm found

A consistent, interpretable SUT weakness: the attacker overtakes, angles
across the ego's lane (relative heading ≈ -0.4 to -0.7 rad) and brakes
(attacker 7-8 m/s vs ego 14-21 m/s); the IDM ego rear-ends it within
~2.5 s - cut-in-and-brake. IDM reacts to the front vehicle in its own lane,
and a cutting-in vehicle registers too late. Every failure replays from its
logged (seed, policy) pair.

## Implications carried to M2 (registered before any M2 run)

* Add potential-based shaping arms (Ng et al. 1999): r = γ·Φ(s') - Φ(s)
  with Φ the normalized metric. PBRS provably preserves the optimal policy,
  so it removes the farming optimum by construction. M2 then tests:
  **H1-revised** - under PBRS, denser metrics help (the guidance value
  survives once farming is impossible); under raw shaping, density inverts
  into harm (as observed here).
* 5 seeds per arm; the pilot's 1 seed cannot distinguish "PORA farms" from
  "this run farmed."
* Report the farming diagnostics (timeout share, timeout-episode return)
  as first-class outcomes alongside failure counts.

Regenerate: `scripts/calibrate.py`, then per arm `scripts/train_ppo.py
--arm ARM --seed 1 --steps 100000 [--calibration
results/calibration/normalizers.json] --outdir ...`, then
`scripts/eval_policy.py --episodes 200 --master-seed 90001 ...`, then
`scripts/analyze_pilot.py`. Wall clock on an 8 GB Apple M3: ~20 min per
arm sequential (do not run arms concurrently on 8 GB; 12x degradation).
