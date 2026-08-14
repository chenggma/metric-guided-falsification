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
"""

import math

from .actors import ActorState

TRIVIAL = "ego_attacker_crash_trivial"
ATTACKER_CRASH = "ego_attacker_crash"
THIRD_PARTY = "ego_third_party_crash"


def coarse_crash_label(ego: ActorState, attacker: ActorState, attacker_crashed: bool) -> str:
    """Label an ego crash. `attacker_crashed` is highway-env's collision flag
    for the attacker at the same step."""
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
