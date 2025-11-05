"""
Unit tests for GMM memory initialization.

Tests that the GMM model's get_initial_memory() method correctly handles
different numbers of experts and returns the expected structure.
"""

import pytest
import torch

from transformers import XLNetForQuestionAnsweringSimple
from gmmxlnet.models import GMMXLNetForQA


class TestGMMMemoryInitialization:
    """Test GMM memory initialization functionality."""

    @pytest.fixture
    def base_model(self):
        """Create a base XLNet model for testing."""
        return XLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased")

    @pytest.mark.parametrize("num_experts", [2, 4, 8])
    def test_get_initial_memory_structure(self, base_model, num_experts):
        """Test get_initial_memory returns correct structure for different expert counts."""
        # Create GMM model
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=num_experts,
            memory_slots=16,
            routing_mode="write-based",
            use_gmm_memory=True,
        )

        batch_size = 4
        device = torch.device("cpu")

        # Test memory initialization
        memory_state = model.get_initial_memory(batch_size, device)

        # Should return a dictionary
        assert isinstance(memory_state, dict), f"Expected dict, got {type(memory_state)}"

        # Should have correct number of expert memories
        assert len(memory_state) == num_experts, f"Expected {num_experts} experts, got {len(memory_state)}"

        # Should have correct keys
        for expert_idx in range(num_experts):
            expected_key = f"expert_{expert_idx}"
            assert expected_key in memory_state, f"Missing key {expected_key} in memory_state"

        # Each expert memory should have correct shape
        for expert_idx in range(num_experts):
            expert_memory = memory_state[f"expert_{expert_idx}"]
            expected_shape = (batch_size, 16, 768)  # (batch, memory_slots, hidden_dim)
            assert expert_memory.shape == expected_shape, (
                f"Expert {expert_idx} memory shape mismatch: "
                f"expected {expected_shape}, got {expert_memory.shape}"
            )

        # Should be on correct device
        for expert_idx in range(num_experts):
            expert_memory = memory_state[f"expert_{expert_idx}"]
            assert expert_memory.device == device, f"Expert {expert_idx} not on correct device"

    def test_get_initial_memory_disabled(self, base_model):
        """Test get_initial_memory returns empty dict when GMM memory disabled."""
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            use_gmm_memory=False,  # Disabled
        )

        batch_size = 4
        device = torch.device("cpu")

        memory_state = model.get_initial_memory(batch_size, device)

        # Should return empty dict
        assert isinstance(memory_state, dict), f"Expected dict, got {type(memory_state)}"
        assert len(memory_state) == 0, f"Expected empty dict, got {len(memory_state)} items"

    def test_memory_consistency_across_calls(self, base_model):
        """Test memory initialization is consistent across multiple calls."""
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            memory_slots=16,
            use_gmm_memory=True,
        )

        batch_size = 2
        device = torch.device("cpu")

        # Get memory twice
        memory_state1 = model.get_initial_memory(batch_size, device)
        memory_state2 = model.get_initial_memory(batch_size, device)

        # Should be equal (but not the same objects)
        for expert_idx in range(model.num_experts):
            key = f"expert_{expert_idx}"
            assert torch.equal(memory_state1[key], memory_state2[key]), (
                f"Expert {expert_idx} memory not consistent across calls"
            )
            assert id(memory_state1[key]) != id(memory_state2[key]), (
                f"Expert {expert_idx} memory should be different objects"
            )

    def test_different_batch_sizes(self, base_model):
        """Test memory initialization works with different batch sizes."""
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            memory_slots=16,
            use_gmm_memory=True,
        )

        device = torch.device("cpu")

        # Test different batch sizes
        for batch_size in [1, 2, 8, 16]:
            memory_state = model.get_initial_memory(batch_size, device)

            # Check each expert memory has correct batch dimension
            for expert_idx in range(model.num_experts):
                expert_memory = memory_state[f"expert_{expert_idx}"]
                assert expert_memory.shape[0] == batch_size, (
                    f"Batch size mismatch for expert {expert_idx}: "
                    f"expected {batch_size}, got {expert_memory.shape[0]}"
                )

    def test_invalid_expert_configuration(self, base_model):
        """Test that invalid expert configurations raise appropriate errors."""
        # Test with invalid expert count (should raise ValueError)
        with pytest.raises(ValueError, match="num_experts must be power of 2"):
            GMMXLNetForQA(
                base_model=base_model,
                num_experts=3,  # Not a power of 2
                use_gmm_memory=True,
            )

    def test_memory_reset_functionality(self, base_model):
        """Test memory reset functionality works correctly."""
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            use_gmm_memory=True,
        )

        # Get initial memory state
        initial_memory = model.get_memory_state()
        assert len(initial_memory) == 4, "Should have 4 expert memories"

        # Reset memory
        model.reset_memory()

        # Get memory state after reset (should be same as initial for learned memory)
        reset_memory = model.get_memory_state()

        # Should have same structure
        assert len(reset_memory) == len(initial_memory), "Memory structure should be same after reset"
        for expert_idx in range(4):
            assert f"expert_{expert_idx}" in reset_memory

    def test_memory_state_get_set(self, base_model):
        """Test memory state getter and setter methods."""
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            memory_slots=16,
            use_gmm_memory=True,
        )

        # Get initial memory state
        memory_state = model.get_memory_state()
        assert isinstance(memory_state, dict)
        assert len(memory_state) == 4

        # Modify memory state
        modified_state = {}
        for expert_idx in range(4):
            key = f"expert_{expert_idx}"
            # Create new memory with random values
            new_memory = torch.randn(16, 768)  # (memory_slots, hidden_dim)
            modified_state[key] = new_memory

        # Set new memory state
        model.set_memory_state(modified_state)

        # Verify it was set correctly
        retrieved_state = model.get_memory_state()
        for expert_idx in range(4):
            key = f"expert_{expert_idx}"
            assert torch.equal(retrieved_state[key], modified_state[key]), (
                f"Expert {expert_idx} memory not set correctly"
            )

    def test_device_placement(self, base_model):
        """Test memory tensors are placed on correct device."""
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            memory_slots=16,
            use_gmm_memory=True,
        )

        batch_size = 2

        # Test CPU
        device_cpu = torch.device("cpu")
        memory_state_cpu = model.get_initial_memory(batch_size, device_cpu)
        for expert_idx in range(4):
            expert_memory = memory_state_cpu[f"expert_{expert_idx}"]
            assert expert_memory.device == device_cpu

        # Test CUDA if available
        if torch.cuda.is_available():
            device_cuda = torch.device("cuda")
            memory_state_cuda = model.get_initial_memory(batch_size, device_cuda)
            for expert_idx in range(4):
                expert_memory = memory_state_cuda[f"expert_{expert_idx}"]
                assert expert_memory.device.type == "cuda"

    def test_memory_shapes_consistency(self, base_model):
        """Test that memory shapes are consistent across experts."""
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            memory_slots=16,
            use_gmm_memory=True,
        )

        batch_size = 3
        device = torch.device("cpu")

        memory_state = model.get_initial_memory(batch_size, device)

        # All experts should have the same shape
        first_shape = None
        for expert_idx in range(4):
            expert_memory = memory_state[f"expert_{expert_idx}"]
            if first_shape is None:
                first_shape = expert_memory.shape
            else:
                assert expert_memory.shape == first_shape, (
                    f"Expert {expert_idx} shape {expert_memory.shape} "
                    f"differs from first expert shape {first_shape}"
                )

        # Expected shape: (batch_size, memory_slots, hidden_dim)
        expected_shape = (batch_size, 16, model.hidden_dim)
        assert first_shape == expected_shape, (
            f"Expected shape {expected_shape}, got {first_shape}"
        )