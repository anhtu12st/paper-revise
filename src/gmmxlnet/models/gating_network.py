"""
Memory Gating Network (Router) for GMM-augmented XLNet.

Implements content-based routing to compute probability distributions over memory experts.
Routes memory write proposals to appropriate experts based on information content.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryGatingNetwork(nn.Module):
    """
    Learnable gating network that routes memory updates to appropriate experts.

    Computes routing probabilities over k memory experts using content-based addressing:
    1. Pool memory write proposals (batch, memory_slots, hidden_dim) → (batch, hidden_dim)
    2. Linear projection: logits = W_gate @ pooled_hiddens → (batch, num_experts)
    3. Temperature-scaled softmax: probs = softmax(logits / temperature)

    Args:
        hidden_dim: Dimension of input hidden states (e.g., 768 for XLNet-base)
        num_experts: Number of memory experts (k ∈ [2, 8])
        temperature: Softmax temperature for routing sharpness (must be > 0)
            - temperature > 1.0: softer routing (more uniform distribution)
            - temperature = 1.0: standard softmax
            - temperature < 1.0: sharper routing (more peaked distribution)
        pooling_method: Method for pooling memory write tokens
            - "mean": Mean-pooling over memory slots (default)
            - "max": Max-pooling over memory slots
            - "attention": Attention-weighted pooling (not yet implemented)

    Raises:
        ValueError: If temperature <= 0 or num_experts not in [2, 8]

    Example:
        >>> router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)
        >>> hiddens = torch.randn(8, 16, 768)  # (batch=8, mem_slots=16, hidden=768)
        >>> probs, logits, entropy = router(hiddens)
        >>> probs.shape  # (8, 4) - routing probabilities per batch item
        torch.Size([8, 4])
        >>> torch.allclose(probs.sum(dim=-1), torch.ones(8))  # Probabilities sum to 1.0
        True
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        temperature: float = 1.0,
        pooling_method: Literal["mean", "max", "attention"] = "mean",
    ):
        super().__init__()

        # Validate configuration
        if temperature <= 0:
            raise ValueError(
                f"Temperature must be > 0, got {temperature}. "
                f"Temperature controls routing sharpness: higher = softer, lower = sharper."
            )
        if not (2 <= num_experts <= 8):
            raise ValueError(
                f"num_experts must be in [2, 8] for GMM routing, got {num_experts}. "
                f"Use powers of 2 (2, 4, 8) for optimal load balancing."
            )
        if pooling_method not in ["mean", "max", "attention"]:
            raise ValueError(
                f"pooling_method must be 'mean', 'max', or 'attention', got {pooling_method}"
            )
        if pooling_method == "attention":
            raise NotImplementedError(
                "Attention-weighted pooling not yet implemented. Use 'mean' or 'max'."
            )

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.temperature = temperature
        self.pooling_method = pooling_method

        # Learnable routing weight matrix W_gate ∈ R^(k × d)
        # Uses nn.Linear for efficient matrix multiplication and parameter management
        self.routing_projection = nn.Linear(hidden_dim, num_experts, bias=True)

        # Numerical stability constants
        self.epsilon = 1e-10
        self.logit_clamp_min = -10.0
        self.logit_clamp_max = 10.0

    def forward(
        self, memory_write_hiddens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing probabilities over memory experts.

        Args:
            memory_write_hiddens: Hidden states for memory write tokens
                Shape: (batch_size, memory_slots, hidden_dim)

        Returns:
            Tuple of (routing_probs, routing_logits, routing_entropy):
                - routing_probs: Probability distribution over experts (batch_size, num_experts)
                    Values in [0, 1], sum to 1.0 per batch item
                - routing_logits: Raw logits before softmax (batch_size, num_experts)
                - routing_entropy: Shannon entropy of routing distribution (batch_size,)
                    Higher entropy = more uniform routing (experts used equally)
                    Lower entropy = more specialized routing (one expert dominates)

        Raises:
            ValueError: If input shape is invalid
        """
        batch_size, memory_slots, hidden_dim = memory_write_hiddens.shape

        # Validate input shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}. "
                f"Routing network configuration mismatch."
            )

        # Step 1: Pool memory write tokens (batch, memory_slots, hidden_dim) → (batch, hidden_dim)
        pooled_hiddens = self._pool_hiddens(memory_write_hiddens)

        # Step 2: Handle all-zero inputs gracefully (return uniform distribution)
        # Check if pooled_hiddens are all zero (or very close to zero)
        is_zero_input = torch.all(torch.abs(pooled_hiddens) < self.epsilon, dim=-1)  # (batch_size,)

        # Step 3: Linear projection W_gate @ pooled_hiddens → (batch, num_experts)
        routing_logits = self.routing_projection(pooled_hiddens)

        # Step 4: Clamp logits for numerical stability (prevent NaN/Inf in softmax)
        routing_logits = torch.clamp(
            routing_logits, min=self.logit_clamp_min, max=self.logit_clamp_max
        )

        # Step 5: Temperature-scaled softmax
        routing_probs = F.softmax(routing_logits / self.temperature, dim=-1)

        # Step 6: Override with uniform distribution for all-zero inputs
        if is_zero_input.any():
            uniform_probs = torch.ones(self.num_experts, device=routing_probs.device) / self.num_experts
            routing_probs[is_zero_input] = uniform_probs

        # Step 7: Compute routing entropy (for analysis and optional regularization)
        routing_entropy = self._compute_entropy(routing_probs)

        return routing_probs, routing_logits, routing_entropy

    def _pool_hiddens(self, memory_write_hiddens: torch.Tensor) -> torch.Tensor:
        """
        Pool memory write token hidden states.

        Args:
            memory_write_hiddens: Shape (batch_size, memory_slots, hidden_dim)

        Returns:
            Pooled hiddens: Shape (batch_size, hidden_dim)
        """
        if self.pooling_method == "mean":
            # Mean-pooling over memory slots dimension
            return memory_write_hiddens.mean(dim=1)
        elif self.pooling_method == "max":
            # Max-pooling over memory slots dimension
            return memory_write_hiddens.max(dim=1)[0]  # max returns (values, indices)
        else:
            # Should never reach here due to __init__ validation
            raise NotImplementedError(f"Pooling method {self.pooling_method} not implemented")

    def _compute_entropy(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute Shannon entropy of routing distribution.

        Entropy measures routing specialization:
        - High entropy (→ log(k)): Uniform routing (all experts used equally)
        - Low entropy (→ 0): Specialized routing (one expert dominates)

        Args:
            routing_probs: Routing probabilities (batch_size, num_experts)

        Returns:
            Entropy per batch item (batch_size,)
        """
        # Shannon entropy: H(p) = -Σ p_i * log(p_i)
        # Add epsilon to prevent log(0)
        log_probs = torch.log(routing_probs + self.epsilon)
        entropy = -(routing_probs * log_probs).sum(dim=-1)
        return entropy

    def compute_routing_entropy(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """
        Public method to compute routing entropy (for external analysis).

        Args:
            routing_probs: Routing probabilities (batch_size, num_experts)

        Returns:
            Entropy per batch item (batch_size,)
        """
        return self._compute_entropy(routing_probs)

    def compute_load_balance_loss(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss (Switch Transformer style).

        Encourages balanced expert usage across batch to prevent routing collapse.
        Formula: load_balance_loss = k × Σ(f_i × P_i)
        where:
        - k = num_experts
        - f_i = fraction of batch routed to expert i (across batch)
        - P_i = mean routing probability to expert i (across batch)

        Typical weight: 0.01 (add to main loss as: total_loss + 0.01 * load_balance_loss)

        Args:
            routing_probs: Routing probabilities (batch_size, num_experts)

        Returns:
            Scalar load balance loss (higher when routing is imbalanced)

        Reference:
            Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
            https://arxiv.org/abs/2101.03961
        """
        batch_size = routing_probs.shape[0]

        # f_i: fraction of samples where expert i has highest probability
        # Approximate with argmax (hard routing)
        expert_assignments = routing_probs.argmax(dim=-1)  # (batch_size,)
        f_i = torch.zeros(self.num_experts, device=routing_probs.device)
        for i in range(self.num_experts):
            f_i[i] = (expert_assignments == i).float().sum() / batch_size

        # P_i: mean routing probability to expert i
        P_i = routing_probs.mean(dim=0)  # (num_experts,)

        # Load balance loss: k × Σ(f_i × P_i)
        load_balance_loss = self.num_experts * (f_i * P_i).sum()

        return load_balance_loss

    def get_routing_weights(self) -> torch.Tensor:
        """
        Access current routing weight matrix W_gate.

        Returns:
            W_gate: Shape (num_experts, hidden_dim)
        """
        return self.routing_projection.weight.data

    def get_routing_distribution(self, routing_probs: torch.Tensor) -> dict:
        """
        Analyze routing distribution for debugging and visualization.

        Args:
            routing_probs: Routing probabilities (batch_size, num_experts)

        Returns:
            Dictionary with routing statistics:
                - mean_probs: Mean probability per expert (num_experts,)
                - max_probs: Max probability per expert (num_experts,)
                - min_probs: Min probability per expert (num_experts,)
                - entropy_mean: Mean entropy across batch (scalar)
                - entropy_std: Std entropy across batch (scalar)
        """
        return {
            "mean_probs": routing_probs.mean(dim=0),  # Average routing per expert
            "max_probs": routing_probs.max(dim=0)[0],  # Peak routing per expert
            "min_probs": routing_probs.min(dim=0)[0],  # Min routing per expert
            "entropy_mean": self._compute_entropy(routing_probs).mean(),
            "entropy_std": self._compute_entropy(routing_probs).std(),
        }
