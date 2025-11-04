"""
Aggregated Memory Reader for GMM-XLNet.

Implements weighted aggregation of expert memories for read operations,
allowing memory read tokens to access collective knowledge from all specialized experts.
"""

from typing import Literal

import torch
import torch.nn as nn

from .gating_network import MemoryGatingNetwork

RoutingMode = Literal["write-based", "read-based"]


class AggregatedMemoryReader(nn.Module):
    """
    Aggregates memory from multiple experts for read operations.

    Implements weighted aggregation: M_context = Σ(p_j · M_j) for j=1 to k
    Supports two routing modes:
    1. Write-based: Reuse routing probabilities from write operation (more efficient)
    2. Read-based: Compute new routing based on query/read context (more adaptive)

    Args:
        hidden_dim: Hidden dimension size for memory slots (e.g., 768 for XLNet-base)
        num_experts: Number of memory experts (k ∈ [2, 8])
        routing_mode: Routing mode for aggregation
            - "write-based": Reuse cached routing from write operation (default)
            - "read-based": Compute new routing for read queries
        temperature: Softmax temperature for read-based routing (only used when routing_mode="read-based")
        pooling_method: Pooling method for read-based routing (only used when routing_mode="read-based")

    Raises:
        ValueError: If num_experts not in [2, 8] or invalid routing_mode

    Example:
        >>> # Write-based routing (reuse write routing probs)
        >>> reader = AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")
        >>> expert_states = [torch.randn(8, 16, 768) for _ in range(4)]  # k=4 experts
        >>> routing_probs = torch.softmax(torch.randn(8, 4), dim=-1)  # (batch=8, k=4)
        >>> aggregated = reader(expert_states, routing_probs=routing_probs)
        >>> aggregated.shape
        torch.Size([8, 16, 768])

        >>> # Read-based routing (compute new routing)
        >>> reader = AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="read-based")
        >>> read_hiddens = torch.randn(8, 16, 768)  # Read query context
        >>> aggregated = reader(expert_states, read_hiddens=read_hiddens)
        >>> aggregated.shape
        torch.Size([8, 16, 768])
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        routing_mode: RoutingMode = "write-based",
        temperature: float = 1.0,
        pooling_method: Literal["mean", "max", "attention"] = "mean",
    ):
        super().__init__()

        # Validate configuration
        valid_expert_counts = {2, 4, 8}
        if num_experts not in valid_expert_counts:
            raise ValueError(
                f"num_experts must be power of 2 in [2, 8] (i.e., 2, 4, or 8), got {num_experts}. "
                f"GMM architecture requires power-of-2 expert counts for efficient routing."
            )
        if routing_mode not in ["write-based", "read-based"]:
            raise ValueError(
                f"routing_mode must be 'write-based' or 'read-based', got '{routing_mode}'. "
                f"Use 'write-based' to reuse write routing (efficient) or "
                f"'read-based' for adaptive read-specific routing."
            )

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.routing_mode = routing_mode

        # Initialize read-specific gating network if using read-based routing
        self.read_gating_network: MemoryGatingNetwork | None = None
        if routing_mode == "read-based":
            self.read_gating_network = MemoryGatingNetwork(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                temperature=temperature,
                pooling_method=pooling_method,
            )

    def forward(
        self,
        expert_states: list[torch.Tensor],
        routing_probs: torch.Tensor | None = None,
        read_hiddens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Aggregate expert memories using weighted sum.

        Args:
            expert_states: List of k expert memory tensors, each of shape
                (batch_size, memory_slots, hidden_dim)
            routing_probs: Routing probabilities from write operation (batch_size, num_experts)
                Required if routing_mode="write-based"
            read_hiddens: Read query hidden states (batch_size, memory_slots, hidden_dim)
                Required if routing_mode="read-based"

        Returns:
            Aggregated memory context: M_context = Σ(p_j · M_j)
                Shape: (batch_size, memory_slots, hidden_dim)

        Raises:
            ValueError: If required inputs are missing or shapes are invalid
        """
        # Validate inputs based on routing mode
        self._validate_inputs(expert_states, routing_probs, read_hiddens)

        # Get routing probabilities
        if self.routing_mode == "write-based":
            # Reuse cached routing from write operation
            assert routing_probs is not None, "routing_probs required for write-based mode"
            probs = routing_probs
        else:
            # Compute new routing for read queries
            assert read_hiddens is not None, "read_hiddens required for read-based mode"
            probs = self.compute_read_routing(read_hiddens)

        # Compute weighted aggregation
        aggregated_memory = self._aggregate_experts(expert_states, probs)

        return aggregated_memory

    def compute_read_routing(self, read_hiddens: torch.Tensor) -> torch.Tensor:
        """
        Compute read-specific routing probabilities.

        Args:
            read_hiddens: Read query hidden states
                Shape: (batch_size, memory_slots, hidden_dim)

        Returns:
            Routing probabilities for read operation
                Shape: (batch_size, num_experts)

        Raises:
            RuntimeError: If routing_mode is not "read-based"
        """
        if self.routing_mode != "read-based":
            raise RuntimeError(
                f"compute_read_routing() only available in read-based mode, current mode is '{self.routing_mode}'"
            )

        if self.read_gating_network is None:
            raise RuntimeError(
                "Read gating network not initialized. This should never happen if "
                "routing_mode='read-based' was set correctly in __init__."
            )

        # Compute routing using read-specific gating network
        routing_result = self.read_gating_network(read_hiddens)
        routing_probs = torch.as_tensor(routing_result[0])
        return routing_probs

    def _aggregate_experts(
        self,
        expert_states: list[torch.Tensor],
        routing_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform efficient batched aggregation of expert memories.

        Uses batched operations to compute: M_context = Σ(p_j · M_j)

        Args:
            expert_states: List of k expert memory tensors
                Each: (batch_size, memory_slots, hidden_dim)
            routing_probs: Routing probabilities (batch_size, num_experts)

        Returns:
            Aggregated memory (batch_size, memory_slots, hidden_dim)
        """
        # Stack experts for efficient batch operation
        # expert_stack: (batch_size, num_experts, memory_slots, hidden_dim)
        expert_stack = torch.stack(expert_states, dim=1)

        # Reshape routing for broadcasting
        # routing_expanded: (batch_size, num_experts, 1, 1)
        routing_expanded = routing_probs.unsqueeze(-1).unsqueeze(-1)

        # Weighted sum (single operation)
        # (batch, k, mem, hidden) * (batch, k, 1, 1) → (batch, k, mem, hidden)
        weighted_experts = expert_stack * routing_expanded

        # Sum over experts dimension
        # (batch, k, mem, hidden) → (batch, mem, hidden)
        aggregated_memory = weighted_experts.sum(dim=1)

        # Validate output shape
        batch_size = expert_states[0].shape[0]
        memory_slots = expert_states[0].shape[1]
        expected_shape = (batch_size, memory_slots, self.hidden_dim)
        if aggregated_memory.shape != expected_shape:
            raise RuntimeError(
                f"Aggregated memory shape mismatch. Expected {expected_shape}, got {aggregated_memory.shape}. "
                f"This is an internal error in aggregation logic."
            )

        return aggregated_memory

    def replace_read_embeddings(
        self,
        sequence_output: torch.Tensor,
        aggregated_memory: torch.Tensor,
        read_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace [MEM_READ] token embeddings with aggregated memory.

        Args:
            sequence_output: XLNet sequence output
                Shape: (batch_size, sequence_length, hidden_dim)
            aggregated_memory: Aggregated memory from experts
                Shape: (batch_size, memory_slots, hidden_dim)
            read_positions: Positions of [MEM_READ] tokens in sequence
                Shape: (batch_size, num_read_tokens) or (batch_size,)
                Values are token indices in [0, sequence_length)

        Returns:
            Modified sequence output with replaced embeddings
                Shape: (batch_size, sequence_length, hidden_dim)

        Raises:
            ValueError: If shapes are incompatible or positions are out of bounds
        """
        batch_size, sequence_length, hidden_dim = sequence_output.shape

        # Validate aggregated_memory shape
        if aggregated_memory.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: sequence_output has batch_size={batch_size}, "
                f"aggregated_memory has batch_size={aggregated_memory.shape[0]}"
            )
        if aggregated_memory.shape[2] != hidden_dim:
            raise ValueError(
                f"Hidden dimension mismatch: sequence_output has hidden_dim={hidden_dim}, "
                f"aggregated_memory has hidden_dim={aggregated_memory.shape[2]}"
            )

        memory_slots = aggregated_memory.shape[1]

        # Validate read_positions
        if read_positions.dim() == 1:
            # Single position per batch item: (batch_size,)
            num_read_tokens = 1
            read_positions = read_positions.unsqueeze(1)  # (batch_size, 1)
        elif read_positions.dim() == 2:
            # Multiple positions per batch item: (batch_size, num_read_tokens)
            num_read_tokens = read_positions.shape[1]
        else:
            raise ValueError(
                f"read_positions must be 1D (batch_size,) or 2D (batch_size, num_read_tokens), "
                f"got {read_positions.dim()}D tensor"
            )

        # Validate memory_slots matches num_read_tokens
        if memory_slots != num_read_tokens:
            raise ValueError(
                f"Number of memory slots ({memory_slots}) must match number of read positions ({num_read_tokens}). "
                f"Each [MEM_READ] token should correspond to one memory slot."
            )

        # Validate position bounds
        if torch.any(read_positions < 0) or torch.any(read_positions >= sequence_length):
            raise ValueError(
                f"read_positions must be in [0, {sequence_length}), "
                f"got min={read_positions.min().item()}, max={read_positions.max().item()}"
            )

        # Clone sequence_output to avoid in-place modification
        modified_output = sequence_output.clone()

        # Replace embeddings at read positions
        for batch_idx in range(batch_size):
            for mem_idx in range(num_read_tokens):
                pos = int(read_positions[batch_idx, mem_idx].item())
                modified_output[batch_idx, pos, :] = aggregated_memory[batch_idx, mem_idx, :]

        return modified_output

    def _validate_inputs(
        self,
        expert_states: list[torch.Tensor],
        routing_probs: torch.Tensor | None,
        read_hiddens: torch.Tensor | None,
    ) -> None:
        """
        Validate inputs based on routing mode.

        Args:
            expert_states: List of expert memory tensors
            routing_probs: Optional routing probabilities
            read_hiddens: Optional read query hidden states

        Raises:
            ValueError: If required inputs are missing or shapes are invalid
        """
        # Validate expert_states
        if not isinstance(expert_states, list):
            raise ValueError(f"expert_states must be a list, got {type(expert_states).__name__}")
        if len(expert_states) != self.num_experts:
            raise ValueError(
                f"expert_states must contain {self.num_experts} experts, got {len(expert_states)}. "
                f"GMM configuration has {self.num_experts} experts (routing_mode={self.routing_mode})."
            )

        # Validate routing mode requirements
        if self.routing_mode == "write-based":
            if routing_probs is None:
                raise ValueError(
                    "routing_probs required for write-based routing mode. "
                    "Provide routing probabilities from write operation."
                )
            # Validate routing_probs shape
            if routing_probs.dim() != 2:
                raise ValueError(
                    f"routing_probs must be 2D (batch_size, num_experts), got {routing_probs.dim()}D tensor"
                )
            batch_size = expert_states[0].shape[0]
            if routing_probs.shape != (batch_size, self.num_experts):
                raise ValueError(
                    f"routing_probs must have shape ({batch_size}, {self.num_experts}), "
                    f"got {routing_probs.shape}. Batch size or expert count mismatch."
                )
        else:  # read-based
            if read_hiddens is None:
                raise ValueError(
                    "read_hiddens required for read-based routing mode. "
                    "Provide read query hidden states for routing computation."
                )
            # Validate read_hiddens shape
            if read_hiddens.dim() != 3:
                raise ValueError(
                    f"read_hiddens must be 3D (batch_size, memory_slots, hidden_dim), got {read_hiddens.dim()}D tensor"
                )
            batch_size, memory_slots, hidden_dim = read_hiddens.shape
            if hidden_dim != self.hidden_dim:
                raise ValueError(
                    f"read_hiddens hidden_dim must be {self.hidden_dim}, got {hidden_dim}. Configuration mismatch."
                )

        # Validate all expert states have consistent shapes
        batch_size = expert_states[0].shape[0]
        memory_slots = expert_states[0].shape[1]
        expected_shape = (batch_size, memory_slots, self.hidden_dim)

        for expert_idx, expert_state in enumerate(expert_states):
            if expert_state.dim() != 3:
                raise ValueError(
                    f"Expert {expert_idx} state must be 3D (batch_size, memory_slots, hidden_dim), "
                    f"got {expert_state.dim()}D tensor"
                )
            if expert_state.shape != expected_shape:
                raise ValueError(
                    f"Expert {expert_idx} state has shape {expert_state.shape}, "
                    f"expected {expected_shape}. All experts must maintain identical shape."
                )

    def __repr__(self) -> str:
        """String representation of AggregatedMemoryReader."""
        return (
            f"AggregatedMemoryReader(hidden_dim={self.hidden_dim}, "
            f"num_experts={self.num_experts}, "
            f"routing_mode='{self.routing_mode}')"
        )
