"""Metric-guided falsification of driving policies.

Research question: when an RL adversary hunts for failures of a driving
policy, does the risk metric used as its dense reward shape which failures
get found? See DESIGN.md for the full experimental contract.
"""

from .actors import ActorState, from_vehicle
from .metrics import METRICS, inv_ttc, neg_tts_margin, pora_score, sparse

__all__ = [
    "ActorState",
    "from_vehicle",
    "METRICS",
    "inv_ttc",
    "neg_tts_margin",
    "pora_score",
    "sparse",
]
