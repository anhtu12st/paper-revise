"""
Expert Memory Updater with Routing Modulation for GMM-XLNet.

Implements expert-specific gated LSTM-style updates modulated by routing probabilities.
Each expert's memory is selectively updated based on routing decisions while maintaining
stable learning through memory protection mechanisms.
"""

import torch
import torch.nn as nn


class ExpertUpdater(nn.Module):
    """
    Applies routing-modulated gated updates to memory experts.

    Implements per-expert gated LSTM-style updates where gate activations are modulated
    by routing probabilities from the gating network. This allows experts to be selectively
    updated based on routing decisions:

    M_j^(i) = (p_j · g_j) ⊙ u_j + (1 - p_j · g_j) ⊙ M_j^(i-1)

    where:
    - M_j^(i): Expert j's memory at time i
    - p_j: Routing probability for expert j (from MemoryGatingNetwork)
    - g_j: Learned LSTM-style gate for expert j
    - u_j: Learned memory update candidate for expert j

    Memory protection: When p_j ≈ 0, modulated gate ≈ 0, so M_j_next ≈ M_j_prev (memory preserved).
    When p_j ≈ 1, update behaves like standard LSTM gate.

    Args:
        hidden_dim: Hidden dimension size for memory slots (e.g., 768 for XLNet-base)
        num_experts: Number of memory experts (k ∈ [2, 8])

    Example:
        >>> updater = ExpertUpdater(hidden_dim=768, num_experts=4)
        >>> expert_states = [torch.randn(8, 16, 768) for _ in range(4)]  # k=4 experts
        >>> write_hiddens = torch.randn(8, 16, 768)  # (batch, memory_slots, hidden)
        >>> routing_probs = torch.softmax(torch.randn(8, 4), dim=-1)  # (batch, k)
        >>> updated_states = updater(expert_states, write_hiddens, routing_probs)
        >>> len(updated_states)
        4
        >>> updated_states[0].shape
        torch.Size([8, 16, 768])
    """

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()

        # Validate configuration
        valid_expert_counts = {2, 4, 8}
        if num_experts not in valid_expert_counts:
            raise ValueError(
                f"num_experts must be power of 2 in [2, 8] (i.e., 2, 4, or 8), got {num_experts}. "
                f"GMM architecture requires power-of-2 expert counts for efficient routing."
            )

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # Per-expert gate networks (W_g): sigmoid(W_g @ [M_j, h_write])
        # Input: concatenated [M_j_prev, write_hiddens] of size 2*hidden_dim
        # Output: gate values of size hidden_dim
        self.gate_networks = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_experts)])

        # Per-expert update networks (W_u): tanh(W_u @ [M_j, h_write])
        # Input: concatenated [M_j_prev, write_hiddens] of size 2*hidden_dim
        # Output: update candidate of size hidden_dim
        self.update_networks = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_experts)])

    def forward(
        self,
        expert_states: list[torch.Tensor],
        write_hiddens: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Apply routing-modulated gated updates to all experts.

        Args:
            expert_states: List of k expert memory tensors, each of shape
                (batch_size, memory_slots, hidden_dim)
            write_hiddens: Proposed memory update from XLNet hidden states
                Shape: (batch_size, memory_slots, hidden_dim)
            routing_probs: Routing probabilities from gating network
                Shape: (batch_size, num_experts)

        Returns:
            List of k updated expert memory tensors, each of shape
                (batch_size, memory_slots, hidden_dim)

        Raises:
            ValueError: If shapes are mismatched or input validation fails
        """
        # Validate inputs
        self._validate_shapes(expert_states, write_hiddens, routing_probs)

        updated_experts = []

        for j in range(self.num_experts):
            # Step 1: Get previous expert state
            M_j_prev = expert_states[j]  # (batch, memory_slots, hidden_dim)

            # Step 2: Compute per-expert LSTM-style gates
            g_j, u_j = self.compute_expert_gates(M_j_prev, write_hiddens, j)

            # Step 3: Extract and reshape routing probability for this expert
            p_j = routing_probs[:, j]  # (batch,)
            p_j = p_j.view(-1, 1, 1)  # Broadcast: (batch, 1, 1)

            # Step 4: Modulate gate with routing probability
            modulated_gate = self.apply_routing_modulation(g_j, p_j)

            # Step 5: Apply gated update with routing modulation
            # M_j_next = (p_j · g_j) ⊙ u_j + (1 - p_j · g_j) ⊙ M_j_prev
            M_j_next = modulated_gate * u_j + (1 - modulated_gate) * M_j_prev

            updated_experts.append(M_j_next)

        return updated_experts

    def compute_expert_gates(
        self,
        expert_state: torch.Tensor,
        write_hidden: torch.Tensor,
        expert_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute LSTM-style gate and update candidate for a single expert.

        Args:
            expert_state: Previous memory state for this expert
                Shape: (batch_size, memory_slots, hidden_dim)
            write_hidden: Proposed memory update
                Shape: (batch_size, memory_slots, hidden_dim)
            expert_idx: Index of the expert (0-based, in [0, num_experts-1])

        Returns:
            Tuple of (gate, update):
                - gate (g_j): Gating values in [0, 1]
                    Shape: (batch_size, memory_slots, hidden_dim)
                - update (u_j): Update candidate in [-1, 1]
                    Shape: (batch_size, memory_slots, hidden_dim)

        Raises:
            IndexError: If expert_idx is out of bounds
        """
        if not 0 <= expert_idx < self.num_experts:
            raise IndexError(
                f"expert_idx must be in [0, {self.num_experts - 1}], got {expert_idx}. "
                f"GMM has {self.num_experts} experts."
            )

        # Concatenate previous state with proposed update
        combined = torch.cat([expert_state, write_hidden], dim=-1)  # (batch, memory_slots, 2*hidden_dim)

        # Compute gate: g_j = sigmoid(W_g[combined])
        g_j = torch.sigmoid(self.gate_networks[expert_idx](combined))  # (batch, memory_slots, hidden_dim)

        # Compute update candidate: u_j = tanh(W_u[combined])
        u_j = torch.tanh(self.update_networks[expert_idx](combined))  # (batch, memory_slots, hidden_dim)

        return g_j, u_j

    def apply_routing_modulation(
        self,
        gates: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Modulate gate activations with routing probabilities.

        Args:
            gates: Gate values from expert gate network
                Shape: (batch_size, memory_slots, hidden_dim)
            routing_probs: Routing probability for this expert (already reshaped)
                Shape: (batch_size, 1, 1)

        Returns:
            Modulated gates: Element-wise product p_j * g_j
                Shape: (batch_size, memory_slots, hidden_dim)
                Values in [0, 1] range

        Raises:
            ValueError: If modulated gate values are outside [0, 1] range
        """
        # Element-wise multiply: modulated_gate = p_j * g_j
        modulated_gate = routing_probs * gates  # Broadcasting: (batch, 1, 1) * (batch, mem, hidden)

        # Validate modulated gate values are in [0, 1] range
        if torch.any(modulated_gate < 0) or torch.any(modulated_gate > 1):
            raise ValueError(
                f"Modulated gate values must be in [0, 1] range. "
                f"Got min={modulated_gate.min().item():.4f}, max={modulated_gate.max().item():.4f}. "
                f"Check that routing_probs and gates are properly normalized."
            )

        return modulated_gate

    def _validate_shapes(
        self,
        expert_states: list[torch.Tensor],
        write_hiddens: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> None:
        """
        Validate input tensor shapes for consistency.

        Args:
            expert_states: List of expert memory tensors
            write_hiddens: Proposed memory update tensor
            routing_probs: Routing probability tensor

        Raises:
            ValueError: If any shape validation fails
        """
        # Validate expert_states is a list of correct length
        if not isinstance(expert_states, list):
            raise ValueError(f"expert_states must be a list, got {type(expert_states).__name__}")
        if len(expert_states) != self.num_experts:
            raise ValueError(
                f"expert_states must contain {self.num_experts} experts, got {len(expert_states)}. "
                f"GMM configuration mismatch."
            )

        # Validate write_hiddens shape
        if write_hiddens.dim() != 3:
            raise ValueError(
                f"write_hiddens must be 3D (batch_size, memory_slots, hidden_dim), got {write_hiddens.dim()}D tensor"
            )
        batch_size, memory_slots, hidden_dim = write_hiddens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"write_hiddens hidden_dim must be {self.hidden_dim}, got {hidden_dim}. Configuration mismatch."
            )

        # Validate routing_probs shape
        if routing_probs.dim() != 2:
            raise ValueError(f"routing_probs must be 2D (batch_size, num_experts), got {routing_probs.dim()}D tensor")
        if routing_probs.shape != (batch_size, self.num_experts):
            raise ValueError(
                f"routing_probs must have shape ({batch_size}, {self.num_experts}), "
                f"got {routing_probs.shape}. Batch size or expert count mismatch."
            )

        # Validate each expert state shape
        expected_shape = (batch_size, memory_slots, hidden_dim)
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
        """String representation of ExpertUpdater."""
        return f"ExpertUpdater(hidden_dim={self.hidden_dim}, num_experts={self.num_experts})"
