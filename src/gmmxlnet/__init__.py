"""
GMMXLNet: Gated Memory Mixture for XLNet-based Question Answering.

This module provides multi-expert memory-augmented XLNet models for long-context QA.
"""

from gmmxlnet.models.memory_mixture import GatedMemoryMixture

__version__ = "0.1.0"

__all__ = ["GatedMemoryMixture"]
