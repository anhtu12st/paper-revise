"""Enhanced Memory Modules for MA-XLNet.

This module implements differentiable memory mechanisms inspired by
Memory-Augmented Neural Networks (MANNs).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableMemory(nn.Module):
    """Differentiable memory with content-based and location-based addressing.

    This implementation provides:
    - Content-based addressing using similarity matching
    - Location-based addressing with shift operations
    - Memory sharpening for focused read/write
    - Usage tracking for memory slot allocation
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        num_heads: int = 1,
        sharpness: float = 1.0,
        enable_usage_tracking: bool = True,
        enable_temporal_links: bool = False,
    ):
        """Initialize differentiable memory.

        Args:
            num_slots: Number of memory slots
            slot_dim: Dimension of each memory slot
            num_heads: Number of read/write heads
            sharpness: Temperature for attention sharpening
            enable_usage_tracking: Track memory slot usage
            enable_temporal_links: Enable temporal link matrix
        """
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_heads = num_heads
        self.sharpness = sharpness
        self.enable_usage_tracking = enable_usage_tracking
        self.enable_temporal_links = enable_temporal_links

        # Initialize memory matrix
        self.register_buffer("memory", torch.zeros(num_slots, slot_dim))
        self.memory: torch.Tensor

        # Usage tracking
        if enable_usage_tracking:
            self.register_buffer("usage", torch.zeros(num_slots))
            self.usage: torch.Tensor

        # Temporal link matrix for sequential access
        if enable_temporal_links:
            self.register_buffer("temporal_links", torch.zeros(num_slots, num_slots))
            self.temporal_links: torch.Tensor

    def content_addressing(self, key: torch.Tensor, beta: torch.Tensor | None = None) -> torch.Tensor:
        """Compute content-based addressing weights.

        Args:
            key: Query key tensor (batch_size, num_heads, slot_dim)
            beta: Key strength (batch_size, num_heads, 1)

        Returns:
            Content-based weights (batch_size, num_heads, num_slots)
        """
        batch_size = key.size(0)

        # Expand memory for batch processing
        memory_expanded = self.memory.unsqueeze(0).unsqueeze(0)  # (1, 1, num_slots, slot_dim)
        memory_expanded = memory_expanded.expand(batch_size, self.num_heads, -1, -1)

        # Expand key
        key_expanded = key.unsqueeze(2)  # (batch_size, num_heads, 1, slot_dim)

        # Compute cosine similarity
        key_norm = F.normalize(key_expanded, p=2, dim=-1)
        memory_norm = F.normalize(memory_expanded, p=2, dim=-1)
        similarity = torch.matmul(key_norm, memory_norm.transpose(-2, -1)).squeeze(2)

        # Apply key strength (beta) if provided
        if beta is not None:
            similarity = similarity * beta
        else:
            similarity = similarity * self.sharpness

        # Softmax to get weights
        weights = F.softmax(similarity, dim=-1)

        return weights

    def read(self, weights: torch.Tensor) -> torch.Tensor:
        """Read from memory using attention weights.

        Args:
            weights: Attention weights (batch_size, num_heads, num_slots)

        Returns:
            Read vectors (batch_size, num_heads, slot_dim)
        """
        # Weighted sum of memory slots
        read_vectors = torch.matmul(weights, self.memory)
        return read_vectors

    def write(self, weights: torch.Tensor, write_vector: torch.Tensor, erase_vector: torch.Tensor | None = None):
        """Write to memory using attention weights.

        Args:
            weights: Attention weights (batch_size, num_heads, num_slots)
            write_vector: Vectors to write (batch_size, num_heads, slot_dim)
            erase_vector: Erase factors (batch_size, num_heads, slot_dim)
        """
        # Average across batch and heads for writing
        weights_avg = weights.mean(dim=[0, 1])  # (num_slots,)
        write_avg = write_vector.mean(dim=[0, 1])  # (slot_dim,)

        # Erase operation (if provided)
        if erase_vector is not None:
            erase_avg = erase_vector.mean(dim=[0, 1])  # (slot_dim,)
            erase_gate = 1 - torch.outer(weights_avg, erase_avg)
            self.memory = self.memory * erase_gate

        # Write operation
        self.memory = self.memory + torch.outer(weights_avg, write_avg)

        # Update usage if enabled
        if self.enable_usage_tracking:
            self.usage = 0.99 * self.usage + weights_avg

        # Update temporal links if enabled
        if self.enable_temporal_links:
            # Record transition from previous to current weights
            prev_weights_avg: torch.Tensor = self.prev_weights_avg if hasattr(self, "prev_weights_avg") else weights_avg
            self.temporal_links = 0.99 * self.temporal_links + torch.outer(prev_weights_avg, weights_avg)
            self.prev_weights_avg: torch.Tensor = weights_avg

    def get_usage_weights(self) -> torch.Tensor:
        """Get least-used memory slots for allocation.

        Returns:
            Allocation weights favoring unused slots (num_slots,)
        """
        if not self.enable_usage_tracking:
            return torch.ones_like(self.memory[:, 0]) / self.num_slots

        # Compute free gates (inverse of usage)
        free_gates = 1 - self.usage

        # Sort to find least used
        sorted_usage, indices = torch.sort(self.usage)

        # Create allocation weights
        allocation = torch.zeros_like(self.usage)
        allocation[indices[0]] = 1.0  # Allocate to least used

        result: torch.Tensor = allocation * free_gates
        return result

    def reset(self):
        """Reset memory to initial state."""
        self.memory.zero_()
        if self.enable_usage_tracking:
            self.usage.zero_()
        if self.enable_temporal_links:
            self.temporal_links.zero_()
            if hasattr(self, "prev_weights_avg"):
                delattr(self, "prev_weights_avg")


class MemoryController(nn.Module):
    """Unified controller for memory operations in MA-XLNet.

    Provides a high-level interface for:
    - Multi-head read/write operations
    - Memory state management
    - Visualization utilities
    """

    def __init__(
        self,
        input_dim: int,
        memory_slots: int,
        memory_dim: int,
        num_heads: int = 1,
        use_temporal_links: bool = False,
        use_usage_tracking: bool = True,
        sharpness: float = 1.0,
    ):
        """Initialize memory controller.

        Args:
            input_dim: Input dimension from model
            memory_slots: Number of memory slots
            memory_dim: Dimension of each memory slot
            num_heads: Number of read/write heads
            use_temporal_links: Enable temporal linking
            use_usage_tracking: Enable usage tracking
            sharpness: Attention sharpening factor
        """
        super().__init__()

        self.input_dim = input_dim
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.num_heads = num_heads

        # Differentiable memory
        self.memory = DifferentiableMemory(
            num_slots=memory_slots,
            slot_dim=memory_dim,
            num_heads=num_heads,
            sharpness=sharpness,
            enable_usage_tracking=use_usage_tracking,
            enable_temporal_links=use_temporal_links,
        )

        # Interface networks
        self.read_keys = nn.Linear(input_dim, num_heads * memory_dim)
        self.read_strengths = nn.Linear(input_dim, num_heads)

        self.write_keys = nn.Linear(input_dim, num_heads * memory_dim)
        self.write_strengths = nn.Linear(input_dim, num_heads)
        self.write_vectors = nn.Linear(input_dim, num_heads * memory_dim)
        self.erase_vectors = nn.Linear(input_dim, num_heads * memory_dim)

        # Gates
        self.write_gate = nn.Linear(input_dim, num_heads)
        self.read_gate = nn.Linear(input_dim, num_heads)

        # Output projection
        self.output_proj = nn.Linear(num_heads * memory_dim, input_dim)

    def forward(
        self,
        input_state: torch.Tensor,
        prev_read: torch.Tensor | None = None,
        prev_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through memory controller.

        Args:
            input_state: Input from model (batch_size, input_dim)
            prev_read: Previous read vectors (batch_size, num_heads, memory_dim)
            prev_weights: Previous attention weights

        Returns:
            output: Processed output (batch_size, input_dim)
            memory_info: Dictionary with memory operation details
        """
        batch_size = input_state.size(0)

        # Generate interface vectors
        read_keys = self.read_keys(input_state).view(batch_size, self.num_heads, self.memory_dim)
        read_strengths = F.softplus(self.read_strengths(input_state)).unsqueeze(-1)

        write_keys = self.write_keys(input_state).view(batch_size, self.num_heads, self.memory_dim)
        write_strengths = F.softplus(self.write_strengths(input_state)).unsqueeze(-1)
        write_vectors = self.write_vectors(input_state).view(batch_size, self.num_heads, self.memory_dim)
        erase_vectors = torch.sigmoid(self.erase_vectors(input_state)).view(batch_size, self.num_heads, self.memory_dim)

        # Gates
        write_gate = torch.sigmoid(self.write_gate(input_state)).unsqueeze(-1)
        read_gate = torch.sigmoid(self.read_gate(input_state)).unsqueeze(-1)

        # Write operation
        write_weights = self.memory.content_addressing(write_keys, write_strengths)

        # Apply write gate
        gated_write_weights = write_weights * write_gate
        gated_write_vectors = write_vectors * write_gate
        gated_erase_vectors = erase_vectors * write_gate

        self.memory.write(gated_write_weights, gated_write_vectors, gated_erase_vectors)

        # Read operation
        read_weights = self.memory.content_addressing(read_keys, read_strengths)

        # Apply read gate
        gated_read_weights = read_weights * read_gate

        read_vectors = self.memory.read(gated_read_weights)

        # Flatten read vectors and project to output
        read_flat = read_vectors.view(batch_size, -1)
        output = self.output_proj(read_flat)

        # Collect memory info for visualization
        memory_info = {
            "read_weights": gated_read_weights,
            "write_weights": gated_write_weights,
            "read_vectors": read_vectors,
            "write_vectors": gated_write_vectors,
            "memory_state": self.memory.memory.clone(),
        }

        if self.memory.enable_usage_tracking:
            memory_info["usage"] = self.memory.usage.clone()

        if self.memory.enable_temporal_links:
            memory_info["temporal_links"] = self.memory.temporal_links.clone()

        return output, memory_info

    def get_memory_state(self) -> torch.Tensor:
        """Get current memory state.

        Returns:
            Memory matrix (num_slots, memory_dim)
        """
        result: torch.Tensor = self.memory.memory.clone()
        return result

    def set_memory_state(self, state: torch.Tensor):
        """Set memory state.

        Args:
            state: Memory matrix (num_slots, memory_dim)
        """
        self.memory.memory.copy_(state)

    def reset_memory(self):
        """Reset memory to initial state."""
        self.memory.reset()

    def visualize_memory(self) -> dict[str, np.ndarray]:
        """Generate visualization data for memory state.

        Returns:
            Dictionary with visualization arrays
        """
        viz_data = {
            "memory": self.memory.memory.detach().cpu().numpy(),
        }

        if self.memory.enable_usage_tracking:
            viz_data["usage"] = self.memory.usage.detach().cpu().numpy()

        if self.memory.enable_temporal_links:
            viz_data["temporal_links"] = self.memory.temporal_links.detach().cpu().numpy()

        return viz_data
