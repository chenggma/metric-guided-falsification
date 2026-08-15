"""Coarse online crash labeling: was the attacker the striking party?

If ANY ego crash paid the bonus, every reward arm would converge on "ram the
ego" and the experiment would collapse into measuring nothing (observed in
M0 smoke runs: 2/8 random episodes already end in an attacker rear-strike).
The bonus is therefore gated on this label; the offline classifier in eval/
refines it for reporting.

The rules are deliberately coarse and geometry-only, and are written for the
straight-highway family F1 (road x-axis ~ lane direction). They are NOT a
liability model:

* rear strike - attacker behind the ego (in the ego's frame) and closing:
  attacker's fault -> trivial.
* side ram - longitudinal overlap, and the attacker's own lateral speed
  toward the ego exceeds the ego's lateral speed toward the attacker:
  attacker steered into the ego -> trivial.
* everything else (ego rear-ends a braking attacker after a cut-in, ego
  side-swipes during its own lane change, ego crashes a third party while
  reacting) -> non-trivial: the SUT had the last clear chance.

**Severity gate (added 2026-08-14, after M2 seeds 3-4 exposed the hole).**
Fault alone is not enough. Raw-shaped arms discovered that grinding the ego
down to a crawl and nudging it satisfies every fault rule above: the ego is
the striking party, so the crash paid the full bonus. Those "failures" were
parking-speed contacts - median closing speed 2.5 m/s with the ego at
2.1 m/s - and one seed produced 131 of them, outscoring every genuine
result in the experiment. Pooling all labelled failures shows the two modes
plainly (n=504: a mass at 2-4 m/s and a separate spread above 6 m/s), so
the gate is set in the valley between them at

    SEVERITY_MIN_CLOSING = 5.0 m/s  (18 km/h)

A contact below that is a fender-bender, not a safety-critical event, and
must not be counted as falsifying the SUT. The threshold is physical, not
tuned: 5 m/s sits in the empirical valley, and the M2 conclusions are
unchanged at 8 m/s (reported as a sensitivity row, results/M2.md). It was
nonetheless chosen AFTER seeing data, which is recorded here and in
DESIGN.md rather than quietly absorbed.
"""

import math

from .actors import ActorState

TRIVIAL = "ego_attacker_crash_trivial"
ATTACKER_CRASH = "ego_attacker_crash"
THIRD_PARTY = "ego_third_party_crash"
LOW_SPEED = "ego_attacker_crash_low_speed"

SEVERITY_MIN_CLOSING = 5.0


def closing_speed(ego: ActorState, other: ActorState) -> float:
    """Magnitude of the relative velocity - the severity proxy."""
    return math.hypot(other.vx - ego.vx, other.vy - ego.vy)


def coarse_crash_label(ego: ActorState, attacker: ActorState, attacker_crashed: bool) -> str:
    """Label an ego crash. `attacker_crashed` is highway-env's collision flag
    for the attacker at the same step.

    A crash below SEVERITY_MIN_CLOSING is LOW_SPEED regardless of fault: it
    is a parking-speed contact, and paying the bonus for it lets the
    adversary farm the crash gate instead of falsifying the SUT.
    """
    if closing_speed(ego, attacker) < SEVERITY_MIN_CLOSING:
        return LOW_SPEED

    gap = math.hypot(attacker.x - ego.x, attacker.y - ego.y)
    touching = gap <= (ego.length + attacker.length)
    if not (attacker_crashed and touching):
        return THIRD_PARTY

    ch, sh = math.cos(ego.heading_rad), math.sin(ego.heading_rad)
    dx = ch * (attacker.x - ego.x) + sh * (attacker.y - ego.y)
    dvx = ch * (attacker.vx - ego.vx) + sh * (attacker.vy - ego.vy)

    overlap = 0.5 * (ego.length + attacker.length)
    if dx < 0 and dvx > 0:  # attacker behind and closing
        return TRIVIAL

    if abs(dx) < overlap:  # side contact: compare lateral speeds toward each other
        toward_ego = attacker.vy * math.copysign(1.0, ego.y - attacker.y)
        toward_att = ego.vy * math.copysign(1.0, attacker.y - ego.y)
        if toward_ego > max(toward_att, 0.0):
            return TRIVIAL

    return ATTACKER_CRASH
