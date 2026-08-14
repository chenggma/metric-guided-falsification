"""Instantaneous actor states, extracted from highway-env vehicles.

The metric adapters consume plain kinematic snapshots so that nothing in the
metric layer depends on highway-env internals (and so the adapters stay
directly comparable to the SUMO-fed ones in risk-metric-bench).
"""

import math
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class ActorState:
    x: float
    y: float
    vx: float
    vy: float
    heading_rad: float
    length: float
    width: float

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


def from_vehicle(v) -> ActorState:
    """Snapshot a highway-env vehicle (any RoadObject with position/heading/speed)."""
    x = float(v.position[0])
    y = float(v.position[1])
    heading = float(v.heading)
    speed = float(v.speed)
    length = float(getattr(v, "LENGTH", 5.0))
    width = float(getattr(v, "WIDTH", 2.0))
    return ActorState(
        x=x,
        y=y,
        vx=speed * math.cos(heading),
        vy=speed * math.sin(heading),
        heading_rad=heading,
        length=length,
        width=width,
    )


def foes_within(ego: ActorState, others: Sequence[ActorState], radius: float) -> List[ActorState]:
    """Actors within `radius` meters of the ego (the metric neighborhood)."""
    return [
        o
        for o in others
        if math.hypot(o.x - ego.x, o.y - ego.y) <= radius
    ]
