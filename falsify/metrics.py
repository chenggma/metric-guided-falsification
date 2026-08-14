"""Metric adapters: (ego, foes at one instant) -> scalar risk score.

Same three adapters, same defaults, same "higher = riskier" convention as
risk-metric-bench (https://github.com/chenggma/risk-metric-bench), so scores
are comparable across the two projects. The `sparse` arm is this project's
control: identically zero, leaving only the crash bonus in the reward.

All adapters see the SAME information: current positions, velocities, and
dimensions. PORA's occupancy is the constant-velocity-Gaussian source from
pora-replication - the same kinematic assumption TTC makes, so no metric
gets privileged foresight.
"""

import math
from typing import Sequence

from pora import (
    SafetyBox,
    VehicleDims,
    constant_velocity_gaussian,
    min_ttc,
    pora_horizon,
    to_av_frame,
    tts_margin,
)
from pora.occupancy_sources import FoeState

from .actors import ActorState

TTC_CAP = 10.0  # scores are capped at 1/TTC_min with TTC_min = 0.1 s
TTS_FLOOR = -1e6


class _Kinematic:
    """Adapter view of ActorState for the pora baselines."""

    def __init__(self, a: ActorState):
        self.x, self.y = a.x, a.y
        self.vx, self.vy = a.vx, a.vy
        self.dims = VehicleDims(a.length, a.width)


def sparse(ego: ActorState, foes: Sequence[ActorState]) -> float:
    """Control arm: no dense signal at all."""
    return 0.0


def inv_ttc(ego: ActorState, foes: Sequence[ActorState]) -> float:
    """1 / TTC, capped at 1/0.1; 0 when no conflict."""
    if not foes:
        return 0.0
    t = min_ttc(_Kinematic(ego), [_Kinematic(f) for f in foes])
    if math.isinf(t):
        return 0.0
    return min(TTC_CAP, 1.0 / max(t, 1.0 / TTC_CAP))


def neg_tts_margin(ego: ActorState, foes: Sequence[ActorState]) -> float:
    """Negated time-to-stop margin; TTS_FLOOR when no conflict exists."""
    if not foes:
        return TTS_FLOOR
    t = min_ttc(_Kinematic(ego), [_Kinematic(f) for f in foes])
    m = tts_margin(ego.speed, t)
    if math.isinf(m):
        return TTS_FLOOR
    return -m


def pora_score(
    ego: ActorState,
    foes: Sequence[ActorState],
    horizon_s: float = 2.5,
    dt: float = 0.5,
    beta: float = 1.0,
    resolution: float = 0.5,
    extent: float = 40.0,
) -> float:
    """PORA scalar for a constant-velocity ego plan.

    Defaults are frozen to the risk-metric-bench values; see DESIGN.md §8
    (no metric hyperparameter tuning after the first full run).
    """
    if not foes:
        return 0.0

    foe_states = [
        FoeState(f.x, f.y, f.vx, f.vy, VehicleDims(f.length, f.width))
        for f in foes
    ]
    fleet_max_l = max(f.length for f in foes)
    fleet_min_w = min(f.width for f in foes)

    n = int(round(2.0 * extent / resolution)) + 1
    origin = (ego.x - extent, ego.y - extent)

    steps = int(round(horizon_s / dt)) + 1
    grids, boxes = [], []
    for k in range(steps):
        t = k * dt
        global_grid = constant_velocity_gaussian(
            foe_states,
            lead_time=t,
            origin=origin,
            shape=(n, n),
            resolution=resolution,
        )
        box = SafetyBox(
            av=VehicleDims(ego.length, ego.width),
            fleet_max_length=fleet_max_l,
            fleet_min_width=fleet_min_w,
            speed_ms=ego.speed,
        )
        half_len = box.phi_length / 2.0 + 1.0
        half_wid = box.phi_width / 2.0 + 1.0
        grids.append(
            to_av_frame(
                global_grid,
                ego.x + ego.vx * t,
                ego.y + ego.vy * t,
                ego.heading_rad,
                half_len,
                half_wid,
                resolution,
            )
        )
        boxes.append(box)
    return pora_horizon(grids, boxes, beta=beta).scalar


METRICS = {
    "sparse": sparse,
    "inv_ttc": inv_ttc,
    "neg_tts_margin": neg_tts_margin,
    "pora": pora_score,
}
