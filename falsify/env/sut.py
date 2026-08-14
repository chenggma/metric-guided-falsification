"""Systems under test: the ego policies the adversary attacks.

A SUT owns the ego. `install` is called after every inner-env reset and may
swap the ego vehicle object; `act` supplies the per-step action passed to the
inner highway-env (which routes it to the ego).
"""

import abc

from highway_env.vehicle.behavior import IDMVehicle


class SUT(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def install(self, env) -> None:
        """Configure the ego inside a freshly reset highway-env instance."""

    @abc.abstractmethod
    def act(self, ego_obs):
        """Action for the inner env's action space, given the ego observation."""


class IDMEgoSUT(SUT):
    """SUT-A: IDM + MOBIL ego. Deterministic given the scenario.

    The default MDP ego is swapped for an ``IDMVehicle``, which ignores
    external actions and drives itself (highway-env behavior.py: "no action
    is supported"). ``act`` therefore returns IDLE, which is discarded.
    """

    name = "idm"
    IDLE = 1  # DiscreteMetaAction index; ignored by IDMVehicle

    def install(self, env) -> None:
        e = env.unwrapped
        old = e.vehicle
        new = IDMVehicle.create_from(old)
        e.road.vehicles[e.road.vehicles.index(old)] = new
        e.controlled_vehicles[0] = new

    def act(self, ego_obs):
        return self.IDLE
