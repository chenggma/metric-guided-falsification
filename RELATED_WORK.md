# Related work and honest positioning

Checked 2026-08-14, before M2 results existed, because the pilot's finding
only matters if it is not already published. It is not - but the framing
this project started with **was wrong** and is corrected here.

## The claim this project can no longer make

DESIGN.md §1 originally asserted that the adversarial reward in falsification
"is almost always a hand-picked risk proxy and the choice is rarely
defended." That is false. Dense safety-metric shaping is an explicitly
argued, evaluated technique with published gains:

* **Corso, Du, Driggs-Campbell, Kochenderfer (ITSC 2019),
  [arXiv:1908.01046](https://arxiv.org/abs/1908.01046) - "Adaptive Stress
  Testing with Reward Augmentation."** Identifies a reward pathology in AST:
  because the objective rewards *likely* actions (the goal being likely
  failures), the agent can learn to take likely actions and never find a
  failure at all. Their fix is **reward augmentation** - adding dense shaped
  terms, including an RSS-based safety term and a trajectory-dissimilarity
  term. They report augmentation *improves* failure discovery and
  expressiveness. They do **not** report any case where adding a dense
  shaping term reduced failures found.
* **Cai et al. (IEEE T-IV 2025), "Adversarial Stress Test for Autonomous
  Vehicle via Series Reinforcement Learning Tasks with Reward Shaping"**
  ([IEEE Xplore](https://ieeexplore.ieee.org/document/10571558/), code:
  [caixxuan/AST-SRL](https://github.com/caixxuan/AST-SRL)). Shapes the
  adversarial reward with RSS and Dynamic Time Warping to steer successive
  agents; reports gains in both vulnerability-revealing collisions and
  scenario diversity.

So the honest framing is not "nobody thought about the reward." It is:
**dense safety-metric shaping is established practice reported to help, and
this project finds a regime where it inverts.** That is a boundary
condition on an accepted technique - a more defensible contribution than a
claimed gap, and it must be written that way.

## Reward hacking is known here - but a different mode

The adversarial-scenario-generation literature does acknowledge reward
hacking. The mode it names is **implausible behavior**: agents "discover
unrealistic or physically implausible strategies to maximize the reward,
such as erratic movements or violating basic traffic rules, unless the
reward function is meticulously engineered." Defenses are plausibility
constraints - naturalistic priors, feasibility guidance (e.g. FREA,
[arXiv:2406.02983](https://arxiv.org/abs/2406.02983)), effort penalties.

The pilot's mode is not that one. Our attacker's control authority is
already bounded and effort-penalized, so it never went erratic. It went
**plausibly inert**: it hovers beside the ego at high metric value,
collecting shaped reward for the full episode and *declining to complete
the attack*, because finishing ends the episode and forfeits the stream.
The pathology is not implausible action; it is plausible non-action. A
plausibility constraint cannot catch it - the behavior is perfectly
plausible, it just is not a test.

## PBRS is not our invention

Potential-based reward shaping (Ng, Harada, Russell 1999) is standard, and
its use in adversarial/falsification RL is already noted in that literature
(e.g. the falsification-based RARL line,
[arXiv:2007.00691](https://arxiv.org/abs/2007.00691), observes that proper
potential-based shaping preserves optimality). M2 therefore does **not**
claim PBRS as novel. It uses PBRS as the theoretically-forced control: if
the pathology is farming, a shaping form that provably cannot create new
optima must remove it. PBRS is the diagnostic instrument, not the
contribution.

## What is left that is ours

1. A **controlled comparison** in which the risk metric is the only
   manipulated variable, with the reward-scale confound explicitly removed
   by per-metric percentile normalization (DESIGN.md §5). Prior work
   compares *its* augmented reward against an unaugmented baseline; it does
   not hold the pipeline fixed and vary the metric identity.
2. A **density dose-response**: suppression ordered by how much of the
   state space each metric scores above zero (pilot: pora worst, then TTS,
   then inverse TTC, sparse unharmed). An ordering is a stronger claim than
   a single negative result and is what makes the effect diagnosable in
   other people's setups.
3. A **named failure mode with a mechanism and an arithmetic check** -
   plausible-inert farming, shown to be a local optimum the dense gradient
   locks in even though the crash pays more (results/PILOT.md).
4. A **shaping-form interaction**: whether density helps or harms depends
   on the shaping form (raw vs potential-based). This is the M2 test; it is
   registered before the run and reported either way.

## Venue implication

Both foils are ITS-community venues (ITSC 2019, T-IV 2025), which supports
IEEE IV 2027 / ITSC 2027 as the target rather than an ML venue. The paper
must cite Corso 2019 and Cai 2025 in the first two paragraphs and state
plainly that it is testing the boundary of their technique, not displacing
it.
