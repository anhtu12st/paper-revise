"""
RBS-QA: Reasoning Belief State for Question Answering

A belief state tracking system for non-monotonic reasoning in long-context QA.
Builds upon GMM-XLNet to enable dynamic belief revision across document segments.
"""

from .belief_state import BeliefStateTracker, BeliefState, SpanCandidate
from .config import RBSTrainingConfig

__all__ = [
    "BeliefStateTracker",
    "BeliefState",
    "SpanCandidate",
    "RBSTrainingConfig",
]

__version__ = "0.1.0"