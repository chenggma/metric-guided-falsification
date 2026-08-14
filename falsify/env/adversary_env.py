"""The falsification environment: an RL adversary controls one NPC.

Gymnasium env whose *agent is the attacker*, not the ego. Each episode:

1. The inner highway-env resets from the episode seed; the SUT installs its
   ego (see sut.py).
2. The vehicle nearest the ego (within ATTACK_RADIUS) becomes the attacker:
   its IDM brain is removed and a kinematics-only ``Vehicle`` takes its
   place, repeating whatever (acceleration, steering) command the adversary
   last issued.
3. Per step, the adversary's bounded continuous action is written to the
   attacker, the SUT acts, the sim advances one policy step, and the reward
   is `shaped_risk(ego) - λ·effort + B·1[ego crash]` (DESIGN.md §5).

Every step's info dict carries ALL metrics (not just the reward arm's), so
cross-metric transfer analysis needs no re-simulation.
"""

import math
from typing import Callable, Dict, Optional

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401  (registers env ids)
from highway_env.vehicle.kinematics import Vehicle

from ..actors import ActorState, foes_within, from_vehicle
from ..fault import TRIVIAL, coarse_crash_label
from ..metrics import METRICS
from .sut import SUT

ACCEL_MIN, ACCEL_MAX = -6.0, 3.0  # m/s^2: emergency braking to brisk throttle
LAT_ACCEL_MAX = 4.0  # m/s^2: aggressive-but-human lateral authority
STEER_ABS_MAX = 0.3  # rad, absolute cap (binds only at low speed)
ATTACK_RADIUS = 60.0  # m, attacker selection + metric neighborhood (bench value)

DEFAULT_CONFIG = {
    "lanes_count": 4,
    "vehicles_count": 30,
    "duration": 24,  # seconds
    "policy_frequency": 5,  # adversary control rate, Hz
    "simulation_frequency": 15,
}

Normalizer = Callable[[float], float]


class AdversarialEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        sut: SUT,
        reward_metric: str = "sparse",
        env_id: str = "highway-v0",
        config: Optional[dict] = None,
        n_neighbors: int = 4,
        effort_lambda: float = 0.05,
        crash_bonus: float = 150.0,
        normalizer: Optional[Normalizer] = None,
        log_all_metrics: bool = True,
    ):
        super().__init__()
        if reward_metric not in METRICS:
            raise ValueError(f"unknown metric {reward_metric!r}")
        self.sut = sut
        self.reward_metric = reward_metric
        self.effort_lambda = effort_lambda
        self.crash_bonus = crash_bonus
        self.normalizer = normalizer
        self.log_all_metrics = log_all_metrics
        self.n_neighbors = n_neighbors

        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config or {})
        self.inner = gym.make(env_id, config=cfg)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(7 * (2 + n_neighbors),), dtype=np.float32
        )

        self.attacker: Optional[Vehicle] = None
        self._prev_cmd = np.zeros(2)
        self._crash_paid = False

    # -- lifecycle ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ego_obs, _ = self.inner.reset(seed=seed)
        self.sut.install(self.inner)
        self._select_attacker()
        self._prev_cmd = np.zeros(2)
        self._crash_paid = False
        return self._obs(), {"attacker_index": self._attacker_index}

    def _select_attacker(self) -> None:
        e = self.inner.unwrapped
        ego = e.vehicle
        others = [v for v in e.road.vehicles if v is not ego]
        if not others:
            raise RuntimeError("scenario has no NPC to designate as attacker")
        dists = [float(np.linalg.norm(v.position - ego.position)) for v in others]
        order = int(np.argmin(dists))
        chosen = others[order]
        # Kinematics-only replacement: repeats the adversary's last command.
        attacker = Vehicle.create_from(chosen)
        e.road.vehicles[e.road.vehicles.index(chosen)] = attacker
        self.attacker = attacker
        self._attacker_index = e.road.vehicles.index(attacker)

    # -- stepping ----------------------------------------------------------

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        accel = ACCEL_MIN + (a[0] + 1.0) * 0.5 * (ACCEL_MAX - ACCEL_MIN)
        steer = float(a[1]) * self._steer_bound()
        self.attacker.act({"acceleration": float(accel), "steering": steer})

        ego_action = self.sut.act(self._ego_obs)
        self._ego_obs, _, term, trunc, _ = self.inner.step(ego_action)

        e = self.inner.unwrapped
        ego = e.vehicle
        ego_s = from_vehicle(ego)
        foe_s = [
            from_vehicle(v) for v in e.road.vehicles if v is not ego
        ]
        foes = foes_within(ego_s, foe_s, ATTACK_RADIUS)

        metrics: Dict[str, float] = {}
        if self.log_all_metrics:
            for name, fn in METRICS.items():
                metrics[name] = fn(ego_s, foes)
            raw = metrics[self.reward_metric]
        else:
            raw = METRICS[self.reward_metric](ego_s, foes)
            metrics[self.reward_metric] = raw

        shaped = self.normalizer(raw) if self.normalizer else raw
        # |a[1]| is already the fraction of the speed-dependent authority used
        effort = 0.5 * (abs(accel) / max(-ACCEL_MIN, ACCEL_MAX) + abs(float(a[1])))

        outcome = None
        reward = shaped - self.effort_lambda * effort
        if ego.crashed and not self._crash_paid:
            self._crash_paid = True
            att_s = from_vehicle(self.attacker)
            outcome = coarse_crash_label(ego_s, att_s, bool(self.attacker.crashed))
            if outcome != TRIVIAL:  # no bonus for ramming the ego (DESIGN.md §5)
                reward += self.crash_bonus
        elif self.attacker.crashed and not ego.crashed:
            outcome = "attacker_crash"
        elif not getattr(self.attacker, "on_road", True):
            outcome = "attacker_offroad"

        terminated = bool(term or ego.crashed or outcome in ("attacker_crash", "attacker_offroad"))
        truncated = bool(trunc and not terminated)
        if truncated:
            outcome = "timeout"

        info = {
            "metrics": metrics,
            "risk_raw": raw,
            "risk_shaped": float(shaped),
            "effort": float(effort),
            "attacker_cmd": (float(accel), steer),
            "outcome": outcome,
        }
        if outcome is not None and outcome.startswith("ego_"):
            info["crash_snapshot"] = self._crash_snapshot(ego_s)

        self._prev_cmd = np.array([accel / max(-ACCEL_MIN, ACCEL_MAX), float(a[1])])
        return self._obs(), float(reward), terminated, truncated, info

    def _steer_bound(self) -> float:
        """Steering angle whose lateral acceleration at the attacker's current
        speed is LAT_ACCEL_MAX (bicycle-model small-angle), capped for low
        speed. Keeps the threat model human-plausible at any speed."""
        v = max(float(self.attacker.speed), 1.0)
        length = float(getattr(self.attacker, "LENGTH", 5.0))
        return min(STEER_ABS_MAX, math.atan(length * LAT_ACCEL_MAX / (v * v)))

    def _crash_snapshot(self, ego_s: ActorState) -> dict:
        att_s = from_vehicle(self.attacker)
        rel_heading = (att_s.heading_rad - ego_s.heading_rad + math.pi) % (2 * math.pi) - math.pi
        return {
            "ego_speed": ego_s.speed,
            "attacker_speed": att_s.speed,
            "closing_speed": math.hypot(att_s.vx - ego_s.vx, att_s.vy - ego_s.vy),
            "rel_heading_rad": rel_heading,
            "ego_xy": (ego_s.x, ego_s.y),
            "attacker_xy": (att_s.x, att_s.y),
        }

    # -- observation -------------------------------------------------------

    def _obs(self) -> np.ndarray:
        e = self.inner.unwrapped
        ego = e.vehicle
        att_s = from_vehicle(self.attacker)

        blocks = [
            np.array(
                [
                    1.0,
                    att_s.y / 16.0,
                    att_s.speed / 40.0,
                    math.cos(att_s.heading_rad),
                    math.sin(att_s.heading_rad),
                    self._prev_cmd[0],
                    self._prev_cmd[1],
                ]
            ),
            self._rel_block(att_s, from_vehicle(ego)),
        ]

        others = [
            v for v in e.road.vehicles if v is not ego and v is not self.attacker
        ]
        others_s = sorted(
            (from_vehicle(v) for v in others),
            key=lambda s: math.hypot(s.x - att_s.x, s.y - att_s.y),
        )[: self.n_neighbors]
        for s in others_s:
            blocks.append(self._rel_block(att_s, s))
        while len(blocks) < 2 + self.n_neighbors:
            blocks.append(np.zeros(7))
        return np.concatenate(blocks).astype(np.float32)

    @staticmethod
    def _rel_block(ref: ActorState, s: ActorState) -> np.ndarray:
        return np.array(
            [
                1.0,
                (s.x - ref.x) / ATTACK_RADIUS,
                (s.y - ref.y) / 16.0,
                (s.vx - ref.vx) / 20.0,
                (s.vy - ref.vy) / 10.0,
                math.cos(s.heading_rad),
                math.sin(s.heading_rad),
            ]
        )

    def close(self):
        self.inner.close()
