"""
Gated Memory Mixture: Multi-expert memory bank management for GMM-XLNet.

This module provides the core infrastructure for managing k independent memory expert banks,
each with configurable initialization strategies.
"""

from typing import Literal

import torch
import torch.nn as nn

InitStrategy = Literal["learned", "zeros", "uniform", "orthogonal"]


class GatedMemoryMixture(nn.Module):
    """
    Multi-expert memory bank manager for GMM-XLNet.

    Maintains k parallel memory expert banks with independent state tracking and initialization.
    Each expert maintains shape (batch_size, memory_slots, hidden_dim).

    Args:
        num_experts: Number of memory experts (k). Must be power of 2 in [2, 8] (i.e., 2, 4, or 8).
        memory_slots: Number of memory slots per expert.
        hidden_dim: Hidden dimension size for each memory slot.
        init_strategies: Initialization strategy per expert. Either a single strategy
            applied to all experts, or a list of k strategies (one per expert).
            Supported strategies: "learned", "zeros", "uniform", "orthogonal".

    Raises:
        ValueError: If num_experts not power of 2 in [2, 8] or invalid initialization strategies.

    Example:
        >>> gmm = GatedMemoryMixture(
        ...     num_experts=4,
        ...     memory_slots=16,
        ...     hidden_dim=768,
        ...     init_strategies="orthogonal"
        ... )
        >>> expert_0_state = gmm.get_expert_state(0)
    """

    def __init__(
        self,
        num_experts: int,
        memory_slots: int,
        hidden_dim: int,
        init_strategies: InitStrategy | list[InitStrategy] = "learned",
    ):
        super().__init__()

        # Validate configuration: num_experts must be power of 2 in [2, 8]
        valid_expert_counts = {2, 4, 8}
        if num_experts not in valid_expert_counts:
            raise ValueError(
                f"num_experts must be power of 2 in [2, 8] (i.e., 2, 4, or 8), got {num_experts}. "
                f"GMM architecture requires power-of-2 expert counts for efficient routing."
            )

        self.num_experts = num_experts
        self.memory_slots = memory_slots
        self.hidden_dim = hidden_dim

        # Normalize init_strategies to list
        if isinstance(init_strategies, str):
            self._init_strategies = [init_strategies] * num_experts
        else:
            if len(init_strategies) != num_experts:
                raise ValueError(f"init_strategies must have length {num_experts}, got {len(init_strategies)}")
            self._init_strategies = list(init_strategies)

        # Validate all strategies
        valid_strategies = {"learned", "zeros", "uniform", "orthogonal"}
        for i, strategy in enumerate(self._init_strategies):
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Invalid init_strategy '{strategy}' for expert {i}. Must be one of {valid_strategies}"
                )

        # Initialize expert parameters
        self._expert_params = nn.ParameterList()
        for expert_idx in range(num_experts):
            strategy = self._init_strategies[expert_idx]
            expert_param = self._create_expert_param(strategy)
            self._expert_params.append(expert_param)

    def _create_expert_param(self, strategy: InitStrategy) -> nn.Parameter:
        """
        Create an expert parameter tensor based on initialization strategy.

        Args:
            strategy: Initialization strategy ("learned", "zeros", "uniform", "orthogonal").

        Returns:
            Initialized parameter tensor of shape (memory_slots, hidden_dim).
        """
        if strategy == "learned":
            # Small random initialization
            param = nn.Parameter(torch.randn(self.memory_slots, self.hidden_dim) * 0.02)
        elif strategy == "zeros":
            # Zero initialization
            param = nn.Parameter(torch.zeros(self.memory_slots, self.hidden_dim))
        elif strategy == "uniform":
            # Uniform random initialization
            param = nn.Parameter(torch.rand(self.memory_slots, self.hidden_dim) * 0.1)
        elif strategy == "orthogonal":
            # Orthogonal initialization for linear independence
            param = nn.Parameter(torch.empty(self.memory_slots, self.hidden_dim))
            nn.init.orthogonal_(param)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return param

    def get_expert_state(self, expert_idx: int) -> torch.Tensor:
        """
        Retrieve a specific expert's memory state.

        Args:
            expert_idx: Expert index (0-based, must be in [0, num_experts-1]).

        Returns:
            Expert memory tensor of shape (memory_slots, hidden_dim).

        Raises:
            IndexError: If expert_idx is out of bounds.
        """
        if not 0 <= expert_idx < self.num_experts:
            raise IndexError(
                f"expert_idx must be in [0, {self.num_experts - 1}], got {expert_idx}. "
                f"GMM has {self.num_experts} experts."
            )
        return self._expert_params[expert_idx]

    def set_expert_state(self, expert_idx: int, state: torch.Tensor) -> None:
        """
        Update a specific expert's memory state.

        Args:
            expert_idx: Expert index (0-based, must be in [0, num_experts-1]).
            state: New memory state of shape (memory_slots, hidden_dim).

        Raises:
            IndexError: If expert_idx is out of bounds.
            ValueError: If state shape doesn't match (memory_slots, hidden_dim).
        """
        if not 0 <= expert_idx < self.num_experts:
            raise IndexError(
                f"expert_idx must be in [0, {self.num_experts - 1}], got {expert_idx}. "
                f"GMM has {self.num_experts} experts."
            )

        expected_shape = (self.memory_slots, self.hidden_dim)
        if state.shape != expected_shape:
            raise ValueError(
                f"Expert state must have shape {expected_shape}, got {state.shape}. "
                f"All experts must maintain identical shape."
            )

        # Update the parameter data
        self._expert_params[expert_idx].data = state.data

    def reset_experts(self) -> None:
        """
        Re-initialize all expert memory banks to their original initialization state.

        This should be called when starting a new document to clear memory state.
        """
        for expert_idx in range(self.num_experts):
            strategy = self._init_strategies[expert_idx]
            new_param = self._create_expert_param(strategy)
            self._expert_params[expert_idx].data = new_param.data

    def get_all_experts(self) -> list[torch.Tensor]:
        """
        Batch access to all expert memory states.

        Returns:
            List of expert memory tensors, each of shape (memory_slots, hidden_dim).
        """
        return [self._expert_params[i] for i in range(self.num_experts)]

    def forward(self, expert_states: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        """
        Forward pass with shape validation.

        Args:
            expert_states: Optional list of external expert states to validate.
                If None, returns current internal expert states.

        Returns:
            List of validated expert memory tensors.

        Raises:
            ValueError: If expert_states has wrong length or shapes are mismatched.
        """
        if expert_states is None:
            # Return internal expert states
            return self.get_all_experts()

        # Validate external states
        if len(expert_states) != self.num_experts:
            raise ValueError(f"expert_states must contain {self.num_experts} experts, got {len(expert_states)}")

        # Validate each expert's shape
        for expert_idx, state in enumerate(expert_states):
            # Extract shape (handle batched case)
            if state.dim() == 2:
                # Shape: (memory_slots, hidden_dim)
                actual_shape = state.shape
                expected_shape = (self.memory_slots, self.hidden_dim)
            elif state.dim() == 3:
                # Shape: (batch_size, memory_slots, hidden_dim)
                actual_shape = state.shape[1:]
                expected_shape = (self.memory_slots, self.hidden_dim)
            else:
                raise ValueError(
                    f"Expert {expert_idx} state must be 2D (memory_slots, hidden_dim) "
                    f"or 3D (batch_size, memory_slots, hidden_dim), got {state.dim()}D tensor"
                )

            if actual_shape != expected_shape:
                raise ValueError(
                    f"Expert {expert_idx} state has shape mismatch. "
                    f"Expected {expected_shape}, got {actual_shape}. "
                    f"All experts must maintain identical shape."
                )

        return expert_states

    def __repr__(self) -> str:
        """String representation of GatedMemoryMixture."""
        strategies_str = ", ".join(self._init_strategies)
        return (
            f"GatedMemoryMixture(num_experts={self.num_experts}, "
            f"memory_slots={self.memory_slots}, "
            f"hidden_dim={self.hidden_dim}, "
            f"init_strategies=[{strategies_str}])"
        )
