"""
GMM model components for memory-augmented XLNet.
"""

from .gating_network import MemoryGatingNetwork
from .memory_mixture import GatedMemoryMixture

__all__ = ["GatedMemoryMixture", "MemoryGatingNetwork"]
