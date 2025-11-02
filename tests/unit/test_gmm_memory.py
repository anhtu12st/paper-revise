"""
Unit tests for GatedMemoryMixture class.

Tests cover:
- All initialization strategies (learned, zeros, uniform, orthogonal)
- Expert state access and modification
- Reset functionality
- Shape validation with various batch sizes
- Independent expert state management
"""

import pytest
import torch

from gmmxlnet.models import GatedMemoryMixture


@pytest.mark.unit
class TestGatedMemoryMixtureInitialization:
    """Test suite for GatedMemoryMixture initialization."""

    def test_valid_configuration(self):
        """Test valid GMM configuration."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )
        assert gmm.num_experts == 4
        assert gmm.memory_slots == 16
        assert gmm.hidden_dim == 768

    def test_invalid_num_experts_too_low(self):
        """Test that num_experts < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_experts must be power of 2 in \\[2, 8\\]"):
            GatedMemoryMixture(
                num_experts=1, memory_slots=16, hidden_dim=768, init_strategies="learned"
            )

    def test_invalid_num_experts_too_high(self):
        """Test that num_experts > 8 raises ValueError."""
        with pytest.raises(ValueError, match="num_experts must be power of 2 in \\[2, 8\\]"):
            GatedMemoryMixture(
                num_experts=9, memory_slots=16, hidden_dim=768, init_strategies="learned"
            )

    @pytest.mark.parametrize("num_experts", [3, 5, 6, 7])
    def test_invalid_non_power_of_2_experts(self, num_experts):
        """Test that non-power-of-2 expert counts raise ValueError."""
        with pytest.raises(ValueError, match="num_experts must be power of 2 in \\[2, 8\\]"):
            GatedMemoryMixture(
                num_experts=num_experts,
                memory_slots=16,
                hidden_dim=768,
                init_strategies="learned",
            )

    def test_invalid_init_strategy(self):
        """Test that invalid init_strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid init_strategy"):
            GatedMemoryMixture(
                num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="invalid"
            )

    def test_mismatched_init_strategies_length(self):
        """Test that init_strategies length must match num_experts."""
        with pytest.raises(ValueError, match="init_strategies must have length 4"):
            GatedMemoryMixture(
                num_experts=4,
                memory_slots=16,
                hidden_dim=768,
                init_strategies=["learned", "zeros"],  # Only 2 strategies for 4 experts
            )

    @pytest.mark.parametrize("num_experts", [2, 4, 8])
    def test_valid_expert_counts(self, num_experts):
        """Test that valid expert counts [2, 4, 8] work correctly."""
        gmm = GatedMemoryMixture(
            num_experts=num_experts,
            memory_slots=16,
            hidden_dim=768,
            init_strategies="learned",
        )
        assert gmm.num_experts == num_experts
        assert len(gmm.get_all_experts()) == num_experts


@pytest.mark.unit
class TestInitializationStrategies:
    """Test suite for different initialization strategies."""

    def test_learned_initialization(self):
        """Test learned initialization strategy."""
        gmm = GatedMemoryMixture(
            num_experts=2, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )
        expert_0 = gmm.get_expert_state(0)
        assert expert_0.shape == (16, 768)
        # Learned initialization should have non-zero values
        assert not torch.allclose(expert_0, torch.zeros_like(expert_0))

    def test_zeros_initialization(self):
        """Test zeros initialization strategy."""
        gmm = GatedMemoryMixture(
            num_experts=2, memory_slots=16, hidden_dim=768, init_strategies="zeros"
        )
        expert_0 = gmm.get_expert_state(0)
        assert expert_0.shape == (16, 768)
        # Zeros initialization should be all zeros
        assert torch.allclose(expert_0, torch.zeros_like(expert_0))

    def test_uniform_initialization(self):
        """Test uniform initialization strategy."""
        gmm = GatedMemoryMixture(
            num_experts=2, memory_slots=16, hidden_dim=768, init_strategies="uniform"
        )
        expert_0 = gmm.get_expert_state(0)
        assert expert_0.shape == (16, 768)
        # Uniform initialization should be in [0, 0.1]
        assert (expert_0 >= 0).all()
        assert (expert_0 <= 0.1).all()

    def test_orthogonal_initialization(self):
        """Test orthogonal initialization strategy."""
        gmm = GatedMemoryMixture(
            num_experts=2, memory_slots=16, hidden_dim=768, init_strategies="orthogonal"
        )
        expert_0 = gmm.get_expert_state(0)
        assert expert_0.shape == (16, 768)
        # Orthogonal initialization should have non-zero values
        assert not torch.allclose(expert_0, torch.zeros_like(expert_0))

    def test_mixed_initialization_strategies(self):
        """Test different strategies for different experts."""
        gmm = GatedMemoryMixture(
            num_experts=4,
            memory_slots=16,
            hidden_dim=768,
            init_strategies=["learned", "zeros", "uniform", "orthogonal"],
        )

        # Check expert 1 is zeros
        expert_1 = gmm.get_expert_state(1)
        assert torch.allclose(expert_1, torch.zeros_like(expert_1))

        # Check expert 2 is uniform (all values in [0, 0.1])
        expert_2 = gmm.get_expert_state(2)
        assert (expert_2 >= 0).all() and (expert_2 <= 0.1).all()

    def test_different_experts_have_different_states(self):
        """Test that different experts initialize with different values."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )

        expert_0 = gmm.get_expert_state(0)
        expert_1 = gmm.get_expert_state(1)
        expert_2 = gmm.get_expert_state(2)

        # Experts should have different random initializations
        assert not torch.allclose(expert_0, expert_1)
        assert not torch.allclose(expert_1, expert_2)
        assert not torch.allclose(expert_0, expert_2)


@pytest.mark.unit
class TestExpertStateManagement:
    """Test suite for expert state access and modification."""

    def test_get_expert_state_valid_index(self):
        """Test getting expert state with valid index."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )
        for expert_idx in range(4):
            state = gmm.get_expert_state(expert_idx)
            assert state.shape == (16, 768)

    def test_get_expert_state_invalid_negative_index(self):
        """Test that negative expert index raises IndexError."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )
        with pytest.raises(IndexError, match="expert_idx must be in \\[0, 3\\]"):
            gmm.get_expert_state(-1)

    def test_get_expert_state_invalid_high_index(self):
        """Test that expert index >= num_experts raises IndexError."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )
        with pytest.raises(IndexError, match="expert_idx must be in \\[0, 3\\]"):
            gmm.get_expert_state(4)

    def test_set_expert_state_valid(self):
        """Test setting expert state with valid tensor."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="zeros"
        )

        # Create new state
        new_state = torch.ones(16, 768)
        gmm.set_expert_state(0, new_state)

        # Verify state was updated
        retrieved_state = gmm.get_expert_state(0)
        assert torch.allclose(retrieved_state, new_state)

        # Verify other experts unchanged
        expert_1 = gmm.get_expert_state(1)
        assert torch.allclose(expert_1, torch.zeros_like(expert_1))

    def test_set_expert_state_invalid_index(self):
        """Test that invalid expert index raises IndexError."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )
        new_state = torch.ones(16, 768)

        with pytest.raises(IndexError, match="expert_idx must be in \\[0, 3\\]"):
            gmm.set_expert_state(4, new_state)

    def test_set_expert_state_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )
        wrong_shape_state = torch.ones(16, 512)  # Wrong hidden_dim

        with pytest.raises(ValueError, match="Expert state must have shape"):
            gmm.set_expert_state(0, wrong_shape_state)

    def test_independent_expert_updates(self):
        """Test that updating one expert doesn't affect others."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="zeros"
        )

        # Store original states
        original_states = [gmm.get_expert_state(i).clone() for i in range(4)]

        # Update expert 2
        new_state = torch.ones(16, 768) * 5.0
        gmm.set_expert_state(2, new_state)

        # Verify expert 2 changed
        assert torch.allclose(gmm.get_expert_state(2), new_state)

        # Verify other experts unchanged
        for i in [0, 1, 3]:
            assert torch.allclose(gmm.get_expert_state(i), original_states[i])

    def test_get_all_experts(self):
        """Test batch access to all expert states."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )

        all_experts = gmm.get_all_experts()
        assert len(all_experts) == 4
        for expert_state in all_experts:
            assert expert_state.shape == (16, 768)

        # Verify same as individual access
        for i in range(4):
            assert torch.equal(all_experts[i], gmm.get_expert_state(i))


@pytest.mark.unit
class TestResetFunctionality:
    """Test suite for expert reset functionality."""

    def test_reset_experts_learned(self):
        """Test reset with learned initialization."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )

        # Modify expert states
        for i in range(4):
            gmm.set_expert_state(i, torch.ones(16, 768) * (i + 1))

        # Reset
        gmm.reset_experts()

        # Verify all experts reset (non-zero for learned)
        for i in range(4):
            expert_state = gmm.get_expert_state(i)
            assert expert_state.shape == (16, 768)
            assert not torch.allclose(expert_state, torch.ones(16, 768) * (i + 1))

    def test_reset_experts_zeros(self):
        """Test reset with zeros initialization."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="zeros"
        )

        # Modify expert states
        for i in range(4):
            gmm.set_expert_state(i, torch.ones(16, 768) * (i + 1))

        # Reset
        gmm.reset_experts()

        # Verify all experts are zeros
        for i in range(4):
            expert_state = gmm.get_expert_state(i)
            assert torch.allclose(expert_state, torch.zeros_like(expert_state))

    def test_reset_experts_mixed_strategies(self):
        """Test reset with mixed initialization strategies."""
        gmm = GatedMemoryMixture(
            num_experts=4,
            memory_slots=16,
            hidden_dim=768,
            init_strategies=["learned", "zeros", "uniform", "orthogonal"],
        )

        # Modify all expert states to same value
        for i in range(4):
            gmm.set_expert_state(i, torch.ones(16, 768) * 99.0)

        # Reset
        gmm.reset_experts()

        # Expert 1 should be zeros
        expert_1 = gmm.get_expert_state(1)
        assert torch.allclose(expert_1, torch.zeros_like(expert_1))

        # Expert 2 should be uniform [0, 0.1]
        expert_2 = gmm.get_expert_state(2)
        assert (expert_2 >= 0).all() and (expert_2 <= 0.1).all()

        # All should not be 99.0 anymore
        for i in range(4):
            expert_state = gmm.get_expert_state(i)
            assert not torch.allclose(expert_state, torch.ones(16, 768) * 99.0)


@pytest.mark.unit
class TestShapeValidation:
    """Test suite for shape validation with various batch sizes."""

    def test_forward_no_input_returns_internal_states(self):
        """Test forward() with no input returns internal expert states."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )

        output = gmm.forward()
        assert len(output) == 4
        for expert_state in output:
            assert expert_state.shape == (16, 768)

    def test_forward_valid_2d_states(self):
        """Test forward() with valid 2D expert states."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )

        expert_states = [torch.randn(16, 768) for _ in range(4)]
        output = gmm.forward(expert_states)
        assert len(output) == 4

    def test_forward_valid_3d_batched_states(self):
        """Test forward() with valid 3D batched expert states."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )

        # Batch size 8
        expert_states = [torch.randn(8, 16, 768) for _ in range(4)]
        output = gmm.forward(expert_states)
        assert len(output) == 4

    def test_forward_wrong_expert_count(self):
        """Test forward() raises ValueError for wrong expert count."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )

        expert_states = [torch.randn(16, 768) for _ in range(3)]  # Only 3 experts
        with pytest.raises(ValueError, match="expert_states must contain 4 experts"):
            gmm.forward(expert_states)

    def test_forward_wrong_memory_slots(self):
        """Test forward() raises ValueError for wrong memory_slots."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )

        expert_states = [torch.randn(8, 768) for _ in range(4)]  # Wrong memory_slots
        with pytest.raises(ValueError, match="Expert 0 state has shape mismatch"):
            gmm.forward(expert_states)

    def test_forward_wrong_hidden_dim(self):
        """Test forward() raises ValueError for wrong hidden_dim."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )

        expert_states = [torch.randn(16, 512) for _ in range(4)]  # Wrong hidden_dim
        with pytest.raises(ValueError, match="Expert 0 state has shape mismatch"):
            gmm.forward(expert_states)

    def test_forward_invalid_dimensions(self):
        """Test forward() raises ValueError for invalid tensor dimensions."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )

        expert_states = [torch.randn(768) for _ in range(4)]  # 1D tensors
        with pytest.raises(ValueError, match="must be 2D .* or 3D"):
            gmm.forward(expert_states)

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_forward_various_batch_sizes(self, batch_size):
        """Test forward() with various batch sizes."""
        gmm = GatedMemoryMixture(
            num_experts=4, memory_slots=16, hidden_dim=768, init_strategies="learned"
        )

        expert_states = [torch.randn(batch_size, 16, 768) for _ in range(4)]
        output = gmm.forward(expert_states)
        assert len(output) == 4


@pytest.mark.unit
class TestStringRepresentation:
    """Test suite for string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        gmm = GatedMemoryMixture(
            num_experts=4,
            memory_slots=16,
            hidden_dim=768,
            init_strategies=["learned", "zeros", "uniform", "orthogonal"],
        )

        repr_str = repr(gmm)
        assert "GatedMemoryMixture" in repr_str
        assert "num_experts=4" in repr_str
        assert "memory_slots=16" in repr_str
        assert "hidden_dim=768" in repr_str
        assert "learned" in repr_str
        assert "zeros" in repr_str
        assert "uniform" in repr_str
        assert "orthogonal" in repr_str
