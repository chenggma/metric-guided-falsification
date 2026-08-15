# Design: does the choice of risk metric as adversarial reward shape which failures get found?

Status: M0 (scaffolding). This document is the experiment's contract; results
never change the design retroactively without a logged amendment at the bottom.

## 1. Research question

RL-based falsification (adaptive stress testing, Koren et al. 2018; survey:
Corso et al., JAIR 2021) trains an adversary to drive a system under test (SUT)
into failure. Its reward is typically augmented with a dense safety-metric
term - an established, argued technique reported to *improve* discovery
(Corso et al. 2019 with RSS and trajectory-dissimilarity terms; Cai et al.,
T-IV 2025, with RSS + DTW). What that literature does not do is hold the
pipeline fixed and vary *which* metric supplies the dense term. We ask,
empirically:

> When the *only* thing that changes between falsification runs is the risk
> metric used as the dense adversarial reward, do the discovered failures
> differ systematically - in how fast they are found, how diverse they are,
> what kind of crashes they are, and whether failures found under metric A
> are even visible to metric B?

This extends [risk-metric-bench](https://github.com/chenggma/risk-metric-bench),
which found that discrimination (AUROC) and warning timeliness are different
questions (PORA won AUROC at 0.739 but its 5%-FPR alarm fired latest). The
natural follow-up: when the metric stops being a passive scorer and becomes
the optimization target of an active adversary, do those differences amplify
into different *failure-finding* behavior?

Hypotheses (registered before any full run):

* **H1 (efficiency).** Denser, earlier-rising signals make better rewards.
  PORA's occupancy overlap grows continuously as the attacker closes in, while
  inverse TTC is exactly zero until velocities actually conflict. Prediction:
  PORA-rewarded adversaries reach first non-trivial failure in fewer steps
  than inv-TTC-rewarded ones; both beat sparse (crash-only) reward.
* **H2 (bias).** Each metric's adversary over-samples failure modes that the
  metric scores highly and under-samples its blind spots; the cross-metric
  transfer matrix is asymmetric.
* **H3 (diversity).** The sparse arm, when it finds failures at all, finds a
  *wider* spread of crash geometries than any shaped arm (shaping is a prior,
  priors concentrate search).

Any of these being wrong is a publishable finding; the design does not depend
on the hypotheses holding.

## 2. System under test

* **SUT-A (scripted):** IDM + MOBIL ego (highway-env `IDMVehicle` swapped in
  as the ego). Deterministic given the scenario; no learned component.
* **SUT-B (learned):** a PPO ego trained on the standard `highway-v0` task
  (SB3, fixed seed, frozen checkpoint committed to the repo). Trained once,
  never retrained; all arms attack the same frozen weights.

Two SUTs so claims read "holds for a rule-based and a learned policy," not
"quirk of one controller."

## 3. Scenario families

* **F1:** `highway-v0` - straight multi-lane highway, IDM traffic.
* **F2 (M3):** `intersection-v0` - unprotected turns; echoes the right-angle
  conflicts in risk-metric-bench.

Milestones M0-M2 are F1 only.

## 4. Threat model

* One designated **attacker** NPC, chosen at reset by a deterministic rule
  (nearest vehicle to the ego within 60 m; tie-break by road ordering). Its
  IDM brain is removed; a kinematics-only vehicle takes its place.
* Attacker action, applied at 5 Hz: continuous `(acceleration, steering)`.
  Longitudinal bound `[-6, +3] m/s²` (emergency braking to brisk throttle).
  Steering is bounded by **lateral acceleration**, not a fixed angle: the
  per-step steering limit is the angle producing 4 m/s² of lateral
  acceleration at the attacker's current speed (bicycle model), with an
  absolute 0.3 rad cap that binds only at low speed. A fixed angle bound is
  the wrong threat model - 0.3 rad at 25 m/s is a >30 m/s² super-human jerk
  that leaves the road within a second (observed in smoke testing).
* All other NPCs stay IDM. Initial conditions come from the environment seed,
  not from the adversary (episode-level scenario search is out of scope; the
  adversary's power is purely behavioral).

## 5. Reward

Per policy step `t`:

```
r_t = shaped_t - λ · effort_t + B · 1[ego crash at t]
```

* `shaped_t = N_m(metric(ego_t, foes_t))` - the arm's metric, computed from
  the **ego's** perspective (risk experienced by the SUT), passed through a
  percentile normalizer `N_m` (below). Metric arms:
  `sparse` (shaped ≡ 0, control), `inv_ttc`, `neg_tts_margin`, `pora` -
  the same three adapters as risk-metric-bench, same defaults, so numbers are
  comparable across repositories.
* `effort_t = (|a|/a_max + |δ|/δ_max) / 2` - plausibility pressure: the
  adversary pays for aggressive control, so "just ram the ego" is not free.
  Same `λ` for every arm.
* `B` - crash bonus, same for every arm, paid **only for non-trivial ego
  crashes**: a coarse online fault label (falsify/fault.py) withholds the
  bonus when the attacker is the striking party (rear strike from behind,
  or side ram with the attacker supplying the lateral closing speed). If any
  crash paid, every arm would converge on ramming the ego and the experiment
  would measure nothing - M0 smoke runs already showed 2/8 *random* episodes
  ending in an attacker rear-strike. Episode ends on any ego crash, attacker
  crash (failed attack, no bonus), attacker leaving the road (failed attack),
  or timeout.
* `B` sizing: crashing ends the episode and forfeits future dense reward, so
  a too-small bonus makes *avoiding* the crash optimal for shaped arms
  ("hover and farm risk"). With shaped ∈ [0,1] and ≤120 policy steps, the
  maximum forfeitable dense return is ~120; default `B = 150` exceeds it, so
  finishing the attack is always preferred. Same `B` for every arm.

**Reward-scale confound and its control.** Raw metric scales differ wildly
(inv TTC ∈ [0,10], PORA ∈ [0,~1], TTS margin unbounded). Comparing PPO runs
across arms would confound metric *content* with reward *scale*. Control:
`N_m` maps each metric through its own empirical CDF, estimated from a fixed
set of calibration rollouts under a seeded random (Ornstein-Uhlenbeck)
adversary. After calibration every arm's shaped reward is distributionally
Uniform[0,1] on the calibration distribution; what differs is *which states*
get the high percentiles - exactly the variable under study. Calibration
seeds and the percentile tables are committed.

## 6. Search methods

* **random-OU:** Ornstein-Uhlenbeck noise over the action space (temporally
  correlated; white noise is a strawman). The floor every method must beat.
* **CMA-ES:** episode-return objective over a piecewise-constant
  parameterization of the attacker's action sequence (k segments × 2 dims).
  The "falsification as black-box optimization" tradition.
* **PPO** (SB3): the RL adversary. ≥5 seeds per arm per SUT.

All methods consume the same budget accounting (environment steps, including
calibration), so "found more failures" is never an artifact of more compute.

## 7. Evaluation

1. **Efficiency:** env steps to first non-trivial failure; non-trivial
   failures per 100k steps. Median ± bootstrap CI over seeds; Mann-Whitney
   between arms.
2. **Triviality / fault labeling (evaluation layer, not reward):** each ego
   crash is labeled from its geometry - attacker rear-ends ego → trivial
   (attacker's fault); ego rear-ends slowed attacker, side-swipe during ego
   lane change, ego crashes third party while evading → non-trivial. Both raw
   and non-trivial counts are always reported; headline numbers use
   non-trivial.
3. **Diversity:** featurize each failure (impact speed differential, relative
   heading at impact, ego maneuver state, attacker-relative position bin,
   metric values 1 s before impact), cluster with a fixed pipeline, report
   discovered-cluster counts vs budget (coverage curves).
4. **Cross-metric transfer matrix:** evaluation episodes and replayed
   failures record *all three* metrics regardless of the reward arm, so for
   failures found by arm `i` we read off metric `j`'s trajectory max.
   Asymmetries = blind spots. (Training steps record only the reward arm's
   metric, for speed; any training-time failure is replayable exactly from
   its (seed, policy) pair, so nothing is lost.)
5. **Timeliness echo:** per-failure alarm lead time for each metric at the
   5%-FPR threshold (thresholds recalibrated on this environment's negative
   samples, methodology identical to risk-metric-bench).
6. **Cross-SUT replay (M3):** attacker policies trained against SUT-A run
   against SUT-B and vice versa - do metric-shaped attackers transfer, or
   did they overfit the controller?

## 8. Statistical discipline

* ≥5 PPO seeds per (arm × SUT); seeds committed; every run regenerable from
  config + seed.
* Equal env-step budget per arm, calibration included.
* No metric hyperparameter tuning after the first full run; the bench
  defaults are frozen (they were chosen in risk-metric-bench before this
  project existed, which is the cleanest pre-registration we can get).
* Negative results reported. If all arms find the same failures at the same
  rate, that is the paper ("reward metric choice does not matter" would be
  genuinely useful to know).

## 9. Milestones

* **M0:** env wrapper + metric adapters + random-OU baseline + unit tests +
  CI. Smoke: one seeded episode end-to-end with all metrics logged.
* **M1:** calibration artifact, PPO training loop, pilot (1 seed × 4 arms,
  SUT-A). Go/no-go: PPO beats random-OU on ≥1 arm.
* **M2:** full SUT-A run (5 seeds × 4 arms + baselines), evaluation layer,
  figures. → arXiv preprint.
* **M3:** SUT-B, cross-SUT replay, F2 intersection family.
* **M4:** paper (IEEE IV 2027 format; deadline 2026-11-15; backup ITSC 2027,
  2027-03-01).

## 10. Known limitations (declared up front)

* highway-env's kinematics are simplified (no tire model); crash geometry
  labels are coarse. Claims are about *search behavior of metric-shaped
  adversaries*, not about absolute safety of any real system.
* PORA here is PORA-under-constant-velocity-Gaussian-occupancy (same caveat
  as risk-metric-bench); results do not grade the original paper's learned
  predictor.
* One attacker vehicle; multi-vehicle coordinated attacks out of scope.
* The effort penalty is a plausibility *pressure*, not a validated
  naturalistic driver model.

Positioning, prior work, and the framing this project had to retract are in
[RELATED_WORK.md](RELATED_WORK.md).

## Amendments

* 2026-08-14 (M2 halted at 24/35, gate corrected, batch restarted): the
  crash bonus was gated on fault only, and seeds 3-4 found the hole. Raw
  arms learned to grind the ego down to ~2 m/s and nudge it: the ego is the
  striking party, so every fault rule is satisfied and the full bonus pays.
  One seed produced 131 such "failures" - more than any genuine result in
  the experiment - at a median closing speed of 2.5 m/s with the ego at
  2.1 m/s. These are parking-speed contacts, not falsifications.
  **This is a second reward-hacking mode, distinct from the farming mode
  the pilot found**: farming games the *shaped* term by never finishing;
  creeping games the *bonus* term by finishing cheaply. Both are plausible
  behaviour, so neither is caught by plausibility constraints.
  Fix: a severity floor (closing speed >= 5.0 m/s) checked *before* fault,
  in the reward and in the analysis alike (falsify/fault.py). The threshold
  is physical and sits in the empirical valley between the two modes
  (n = 504 labelled failures: a mass at 2-4 m/s, a separate spread above
  6 m/s); conclusions are unchanged at 8 m/s.
  **It was nevertheless chosen after seeing data.** Consequences, recorded
  rather than absorbed: (a) the 24 completed runs trained against the
  farmable gate and are preserved unmodified at `results/m2_oldgate/` as
  the evidence for this finding; (b) they cannot serve as the raw-vs-PBRS
  test, because the raw arms were *paid* to creep, which confounds their
  poor severe-failure counts; (c) M2 restarts from zero with the corrected
  gate, all 7 arms x 5 seeds. Hypotheses are unchanged.

* 2026-08-14 (M0, pre-results): two changes driven by smoke testing, made
  before any training run existed. (1) Fixed-angle steering bound replaced
  by a lateral-acceleration bound (fixed 0.3 rad at highway speed is a
  super-human >30 m/s² input; random attackers left the road in <1 s).
  (2) Crash bonus gated on the coarse non-triviality label and raised to
  150 (see §5 for both rationales).
* 2026-08-14 (M1, pre-results): training steps log only the reward arm's
  metric; all-metric logging stays on for calibration, evaluation, and
  failure replay (§7.4). Motive: wall clock on the 8 GB pilot machine -
  four concurrent all-metric trainings degraded 12x from memory contention;
  arms now run sequentially. No analysis is lost (failures replay exactly).
* 2026-08-14 (literature check, pre-M2 results): §1's premise that the
  reward metric choice "is rarely defended" was **false** and is retracted -
  dense safety-metric augmentation is argued and evaluated in Corso et al.
  2019 and Cai et al. 2025, both reporting gains. The contribution is
  restated as a boundary condition on an accepted technique rather than a
  neglected gap; PBRS is explicitly not claimed as novel. See
  RELATED_WORK.md. No experimental design change follows - the arms, seeds,
  and analysis are unchanged; only the claims are.
* 2026-08-14 (post-pilot, pre-M2): pilot (1 seed/arm, results/PILOT.md)
  passed the go/no-go (sparse beats random-OU ~24x) but inverted H1: dense
  shaping induced reward farming (hover-and-collect; a local optimum the
  dense gradient locks in), with a dose-response - pora worst, sparse
  unharmed. M2 design change, registered before any M2 run: add
  potential-based shaping arms (r = γ·Φ(s')-Φ(s), Ng et al. 1999) for each
  dense metric. **H1-revised:** under PBRS density helps; under raw shaping
  density harms. H2/H3 unchanged, now evaluated under both shaping forms.
  Farming diagnostics (timeout share, timeout-episode return) become
  first-class reported outcomes. Arms: sparse + 3 raw + 3 PBRS = 7, ≥5
  seeds each.
