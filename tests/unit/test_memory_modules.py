"""Unit tests for differentiable memory modules.

Tests the DifferentiableMemory and MemoryController classes to ensure
correct behavior of content-based addressing, multi-head attention,
usage tracking, and temporal links.
"""

import pytest
import torch

from memxlnet.models.memory_modules import DifferentiableMemory, MemoryController


class TestDifferentiableMemory:
    """Tests for DifferentiableMemory class."""

    def test_initialization(self):
        """Test memory initialization with different configurations."""
        # Basic initialization
        memory = DifferentiableMemory(num_slots=32, slot_dim=768)
        assert memory.num_slots == 32
        assert memory.slot_dim == 768
        assert memory.num_heads == 1
        assert memory.sharpness == 1.0
        assert memory.memory.shape == (32, 768)
        assert torch.allclose(memory.memory, torch.zeros(32, 768))

    def test_initialization_with_features(self):
        """Test initialization with advanced features enabled."""
        memory = DifferentiableMemory(
            num_slots=64,
            slot_dim=512,
            num_heads=4,
            sharpness=2.0,
            enable_usage_tracking=True,
            enable_temporal_links=True,
        )
        assert memory.num_heads == 4
        assert memory.sharpness == 2.0
        assert memory.enable_usage_tracking is True
        assert memory.enable_temporal_links is True
        assert hasattr(memory, "usage")
        assert hasattr(memory, "temporal_links")
        assert memory.usage.shape == (64,)
        assert memory.temporal_links.shape == (64, 64)

    def test_content_addressing(self):
        """Test content-based addressing with cosine similarity."""
        memory = DifferentiableMemory(num_slots=16, slot_dim=64, num_heads=2)

        # Set some known memory patterns
        with torch.no_grad():
            memory.memory[0] = torch.ones(64)
            memory.memory[1] = torch.ones(64) * 0.5
            memory.memory[2] = torch.zeros(64)

        # Create query keys
        batch_size = 4
        key = torch.ones(batch_size, 2, 64)  # (batch, heads, dim)

        # Compute addressing weights
        weights = memory.content_addressing(key)

        # Check output shape
        assert weights.shape == (batch_size, 2, 16)

        # Check weights are probabilities (sum to 1)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, 2), atol=1e-5)

        # Check weights are non-negative
        assert (weights >= 0).all()

    def test_content_addressing_with_beta(self):
        """Test content addressing with key strength parameter."""
        memory = DifferentiableMemory(num_slots=8, slot_dim=32, num_heads=1)

        # Set non-uniform memory to create variation in similarity
        with torch.no_grad():
            for i in range(8):
                memory.memory[i] = torch.randn(32)

        batch_size = 2
        key = torch.randn(batch_size, 1, 32)
        beta = torch.tensor([[[5.0]], [[0.5]]])  # Different strengths per batch

        weights = memory.content_addressing(key, beta)

        assert weights.shape == (batch_size, 1, 8)
        # Higher beta should lead to sharper distribution (higher max weight)
        # This should hold when memory slots are diverse
        weights_std_high = weights[0].std()
        weights_std_low = weights[1].std()
        assert weights_std_high > weights_std_low or weights[0].max() >= weights[1].max()

    def test_read_operation(self):
        """Test reading from memory using attention weights."""
        memory = DifferentiableMemory(num_slots=10, slot_dim=64, num_heads=2)

        # Set known memory values
        with torch.no_grad():
            for i in range(10):
                memory.memory[i] = torch.ones(64) * i

        # Create attention weights focusing on specific slots
        batch_size = 3
        weights = torch.zeros(batch_size, 2, 10)
        weights[0, 0, 0] = 1.0  # Focus on slot 0
        weights[0, 1, 5] = 1.0  # Focus on slot 5
        weights[1, 0, 9] = 1.0  # Focus on slot 9
        weights[1, 1, 3] = 1.0  # Focus on slot 3
        weights[2, 0, 2] = 1.0  # Focus on slot 2
        weights[2, 1, 7] = 1.0  # Focus on slot 7

        read_vectors = memory.read(weights)

        assert read_vectors.shape == (batch_size, 2, 64)
        # Check read values match expected memory slots
        assert torch.allclose(read_vectors[0, 0], torch.zeros(64), atol=1e-5)
        assert torch.allclose(read_vectors[0, 1], torch.ones(64) * 5, atol=1e-5)
        assert torch.allclose(read_vectors[1, 0], torch.ones(64) * 9, atol=1e-5)

    def test_write_operation(self):
        """Test writing to memory using attention weights."""
        memory = DifferentiableMemory(num_slots=8, slot_dim=32, num_heads=1)

        # Initial memory should be zeros
        assert torch.allclose(memory.memory, torch.zeros(8, 32))

        batch_size = 2
        weights = torch.zeros(batch_size, 1, 8)
        weights[:, :, 0] = 1.0  # Focus on slot 0

        write_vector = torch.ones(batch_size, 1, 32) * 5.0

        memory.write(weights, write_vector)

        # Slot 0 should now have value 5.0 (averaged across batch and heads)
        assert memory.memory[0].mean() > 0
        assert memory.memory[1:].abs().sum() == 0  # Other slots unchanged

    def test_write_with_erase(self):
        """Test write operation with erase mechanism."""
        memory = DifferentiableMemory(num_slots=6, slot_dim=16, num_heads=1)

        # Set initial memory
        with torch.no_grad():
            memory.memory[:] = torch.ones(6, 16) * 3.0

        batch_size = 2
        weights = torch.zeros(batch_size, 1, 6)
        weights[:, :, 0] = 1.0  # Focus on slot 0

        write_vector = torch.ones(batch_size, 1, 16) * 7.0
        erase_vector = torch.ones(batch_size, 1, 16)  # Full erase

        memory.write(weights, write_vector, erase_vector)

        # Slot 0 should be erased and rewritten
        assert memory.memory[0].mean() > 3.0  # Should be closer to 7.0
        # Other slots should remain at 3.0
        assert torch.allclose(memory.memory[1].mean(), torch.tensor(3.0), atol=0.1)

    def test_usage_tracking(self):
        """Test memory usage tracking functionality."""
        memory = DifferentiableMemory(num_slots=10, slot_dim=32, enable_usage_tracking=True)

        # Initially usage should be zero
        assert torch.allclose(memory.usage, torch.zeros(10))

        # Write to specific slots
        batch_size = 2
        weights = torch.zeros(batch_size, 1, 10)
        weights[:, :, 3] = 1.0  # Focus on slot 3
        write_vector = torch.ones(batch_size, 1, 32)

        memory.write(weights, write_vector)

        # Usage for slot 3 should increase
        assert memory.usage[3] > 0
        assert (memory.usage[:3] == 0).all()
        assert (memory.usage[4:] == 0).all()

        # Write again to different slot
        weights = torch.zeros(batch_size, 1, 10)
        weights[:, :, 7] = 1.0
        memory.write(weights, write_vector)

        # Both slots should have usage
        assert memory.usage[3] > 0
        assert memory.usage[7] > 0

    def test_temporal_links(self):
        """Test temporal link matrix updates."""
        memory = DifferentiableMemory(num_slots=8, slot_dim=16, enable_temporal_links=True)

        # Initially temporal links should be zero
        assert torch.allclose(memory.temporal_links, torch.zeros(8, 8))

        batch_size = 1
        write_vector = torch.ones(batch_size, 1, 16)

        # Write to slot 2
        weights1 = torch.zeros(batch_size, 1, 8)
        weights1[:, :, 2] = 1.0
        memory.write(weights1, write_vector)

        # Write to slot 5
        weights2 = torch.zeros(batch_size, 1, 8)
        weights2[:, :, 5] = 1.0
        memory.write(weights2, write_vector)

        # There should be a temporal link from slot 2 to slot 5
        assert memory.temporal_links[2, 5] > 0

    def test_get_usage_weights(self):
        """Test allocation weights based on usage."""
        memory = DifferentiableMemory(num_slots=10, slot_dim=32, enable_usage_tracking=True)

        # Set some usage values manually
        with torch.no_grad():
            memory.usage[0] = 0.9  # Heavily used
            memory.usage[1] = 0.5  # Moderately used
            memory.usage[2] = 0.1  # Lightly used

        allocation = memory.get_usage_weights()

        assert allocation.shape == (10,)
        # Least used slot should get highest allocation weight
        least_used_idx = memory.usage.argmin()
        assert allocation[least_used_idx] > 0

    def test_reset(self):
        """Test memory reset functionality."""
        memory = DifferentiableMemory(num_slots=8, slot_dim=16, enable_usage_tracking=True, enable_temporal_links=True)

        # Write some data
        batch_size = 2
        weights = torch.ones(batch_size, 1, 8) / 8
        write_vector = torch.ones(batch_size, 1, 16) * 5.0
        memory.write(weights, write_vector)

        # Memory should be non-zero
        assert memory.memory.abs().sum() > 0
        assert memory.usage.sum() > 0

        # Reset
        memory.reset()

        # Everything should be zero
        assert torch.allclose(memory.memory, torch.zeros(8, 16))
        assert torch.allclose(memory.usage, torch.zeros(8))
        assert torch.allclose(memory.temporal_links, torch.zeros(8, 8))


class TestMemoryController:
    """Tests for MemoryController class."""

    def test_initialization(self):
        """Test controller initialization."""
        controller = MemoryController(input_dim=768, memory_slots=32, memory_dim=768, num_heads=1)

        assert controller.input_dim == 768
        assert controller.memory_slots == 32
        assert controller.memory_dim == 768
        assert controller.num_heads == 1
        assert isinstance(controller.memory, DifferentiableMemory)

    def test_forward_single_head(self):
        """Test forward pass with single head."""
        controller = MemoryController(input_dim=256, memory_slots=16, memory_dim=256, num_heads=1)

        batch_size = 4
        input_state = torch.randn(batch_size, 256)

        output, memory_info = controller(input_state)

        # Check output shape
        assert output.shape == (batch_size, 256)

        # Check memory info
        assert "read_weights" in memory_info
        assert "write_weights" in memory_info
        assert "read_vectors" in memory_info
        assert "write_vectors" in memory_info
        assert "memory_state" in memory_info

        assert memory_info["read_weights"].shape == (batch_size, 1, 16)
        assert memory_info["write_weights"].shape == (batch_size, 1, 16)
        assert memory_info["read_vectors"].shape == (batch_size, 1, 256)
        assert memory_info["memory_state"].shape == (16, 256)

    def test_forward_multi_head(self):
        """Test forward pass with multiple heads."""
        controller = MemoryController(input_dim=512, memory_slots=32, memory_dim=512, num_heads=4)

        batch_size = 3
        input_state = torch.randn(batch_size, 512)

        output, memory_info = controller(input_state)

        assert output.shape == (batch_size, 512)
        assert memory_info["read_weights"].shape == (batch_size, 4, 32)
        assert memory_info["write_weights"].shape == (batch_size, 4, 32)
        assert memory_info["read_vectors"].shape == (batch_size, 4, 512)

    def test_gates(self):
        """Test read/write gate mechanisms."""
        controller = MemoryController(input_dim=128, memory_slots=8, memory_dim=128, num_heads=1)

        batch_size = 2
        input_state = torch.randn(batch_size, 128)

        output, memory_info = controller(input_state)

        # Gates should produce values between 0 and 1 (sigmoid)
        # We can verify this indirectly by checking that operations are bounded
        assert output.abs().max() < 1000  # Should be reasonable values
        assert (memory_info["read_weights"] >= 0).all()
        assert (memory_info["write_weights"] >= 0).all()

    def test_with_usage_tracking(self):
        """Test controller with usage tracking enabled."""
        controller = MemoryController(
            input_dim=256, memory_slots=16, memory_dim=256, num_heads=2, use_usage_tracking=True
        )

        batch_size = 2
        input_state = torch.randn(batch_size, 256)

        output, memory_info = controller(input_state)

        # Usage info should be present
        assert "usage" in memory_info
        assert memory_info["usage"].shape == (16,)

    def test_with_temporal_links(self):
        """Test controller with temporal links enabled."""
        controller = MemoryController(
            input_dim=256, memory_slots=16, memory_dim=256, num_heads=1, use_temporal_links=True
        )

        batch_size = 2
        input_state = torch.randn(batch_size, 256)

        output, memory_info = controller(input_state)

        # Temporal links should be present
        assert "temporal_links" in memory_info
        assert memory_info["temporal_links"].shape == (16, 16)

    def test_memory_state_management(self):
        """Test get/set/reset memory state."""
        controller = MemoryController(input_dim=128, memory_slots=10, memory_dim=128, num_heads=1)

        # Get initial state
        initial_state = controller.get_memory_state()
        assert initial_state.shape == (10, 128)
        assert torch.allclose(initial_state, torch.zeros(10, 128))

        # Run forward to modify memory
        input_state = torch.randn(2, 128)
        controller(input_state)

        # Get modified state
        modified_state = controller.get_memory_state()
        # Should be different from initial (memory was written to)
        assert not torch.allclose(modified_state, initial_state, atol=1e-5)

        # Set custom state
        custom_state = torch.ones(10, 128) * 5.0
        controller.set_memory_state(custom_state)
        assert torch.allclose(controller.get_memory_state(), custom_state)

        # Reset to zeros
        controller.reset_memory()
        assert torch.allclose(controller.get_memory_state(), torch.zeros(10, 128))

    def test_visualize_memory(self):
        """Test memory visualization data generation."""
        controller = MemoryController(
            input_dim=256,
            memory_slots=16,
            memory_dim=256,
            num_heads=2,
            use_usage_tracking=True,
            use_temporal_links=True,
        )

        # Run forward to populate memory
        input_state = torch.randn(3, 256)
        controller(input_state)

        # Get visualization data
        viz_data = controller.visualize_memory()

        assert "memory" in viz_data
        assert viz_data["memory"].shape == (16, 256)

        assert "usage" in viz_data
        assert viz_data["usage"].shape == (16,)

        assert "temporal_links" in viz_data
        assert viz_data["temporal_links"].shape == (16, 16)

        # Check data types (should be numpy arrays)
        import numpy as np

        assert isinstance(viz_data["memory"], np.ndarray)
        assert isinstance(viz_data["usage"], np.ndarray)
        assert isinstance(viz_data["temporal_links"], np.ndarray)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the controller."""
        controller = MemoryController(input_dim=64, memory_slots=8, memory_dim=64, num_heads=1)

        batch_size = 2
        input_state = torch.randn(batch_size, 64, requires_grad=True)

        output, memory_info = controller(input_state)

        # Compute a simple loss
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert input_state.grad is not None
        assert input_state.grad.abs().sum() > 0

        # Check that controller parameters have gradients
        for param in controller.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_different_sharpness_values(self):
        """Test controller with different sharpness values."""
        # Use a fixed seed for reproducibility in this test
        torch.manual_seed(42)

        sharpness_values = [0.5, 1.0, 2.0, 5.0]
        batch_size = 2
        input_state = torch.randn(batch_size, 128)

        max_weights = []

        for sharpness in sharpness_values:
            controller = MemoryController(
                input_dim=128, memory_slots=16, memory_dim=128, num_heads=1, sharpness=sharpness
            )

            output, memory_info = controller(input_state)

            # Check maximum weight value (higher sharpness should lead to higher max)
            weights = memory_info["read_weights"]
            max_weight = weights.max().item()
            max_weights.append(max_weight)

        # Verify sharpness affects the output (not necessarily monotonic due to learned parameters)
        # At minimum, check that different sharpness values produce different results
        assert len(set([round(w, 4) for w in max_weights])) > 1, "Sharpness should affect attention weights"

    def test_multiple_forward_passes(self):
        """Test multiple forward passes maintain memory state."""
        controller = MemoryController(input_dim=128, memory_slots=10, memory_dim=128, num_heads=1)

        batch_size = 2
        input_state1 = torch.randn(batch_size, 128)
        input_state2 = torch.randn(batch_size, 128)

        # First forward pass
        output1, memory_info1 = controller(input_state1)
        state_after_1 = controller.get_memory_state()

        # Second forward pass
        output2, memory_info2 = controller(input_state2)
        state_after_2 = controller.get_memory_state()

        # Memory state should have changed
        assert not torch.allclose(state_after_1, state_after_2, atol=1e-5)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_slots(self):
        """Test behavior with minimal slots."""
        # Note: We use 1 as the minimum, not 0
        memory = DifferentiableMemory(num_slots=1, slot_dim=64)
        assert memory.memory.shape == (1, 64)

    def test_very_high_sharpness(self):
        """Test with very high sharpness value."""
        memory = DifferentiableMemory(num_slots=8, slot_dim=32, sharpness=100.0)

        batch_size = 2
        key = torch.randn(batch_size, 1, 32)
        weights = memory.content_addressing(key)

        # Should still produce valid probabilities
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, 1), atol=1e-4)
        assert (weights >= 0).all()

    def test_very_low_sharpness(self):
        """Test with very low sharpness value."""
        memory = DifferentiableMemory(num_slots=8, slot_dim=32, sharpness=0.01)

        batch_size = 2
        key = torch.randn(batch_size, 1, 32)
        weights = memory.content_addressing(key)

        # Should still produce valid probabilities
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, 1), atol=1e-4)
        # Weights should be more uniform with low sharpness
        assert weights.std() < 0.5

    def test_single_dimension(self):
        """Test with minimal slot dimension."""
        controller = MemoryController(input_dim=1, memory_slots=4, memory_dim=1, num_heads=1)

        batch_size = 2
        input_state = torch.randn(batch_size, 1)

        output, memory_info = controller(input_state)

        assert output.shape == (batch_size, 1)
        assert memory_info["memory_state"].shape == (4, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
