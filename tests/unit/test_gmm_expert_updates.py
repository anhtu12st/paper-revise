"""
Unit tests for ExpertUpdater with routing modulation.

Tests cover:
- Per-expert gate computation (AC: 1, 2)
- Routing modulation application (AC: 3)
- Gradient flow verification (AC: 4)
- Memory protection mechanism (AC: 5)
- Shape validation and edge cases (AC: 6)
"""

import pytest
import torch

from src.gmmxlnet.models.expert_updates import ExpertUpdater


class TestExpertUpdaterInitialization:
    """Test ExpertUpdater initialization and configuration validation."""

    def test_valid_initialization(self):
        """Test successful initialization with valid configurations."""
        # Test all valid expert counts
        for num_experts in [2, 4, 8]:
            updater = ExpertUpdater(hidden_dim=768, num_experts=num_experts)
            assert updater.num_experts == num_experts
            assert updater.hidden_dim == 768
            assert len(updater.gate_networks) == num_experts
            assert len(updater.update_networks) == num_experts

    def test_invalid_expert_count(self):
        """Test initialization fails with invalid expert counts."""
        invalid_counts = [1, 3, 5, 7, 9, 16]
        for num_experts in invalid_counts:
            with pytest.raises(ValueError, match="must be power of 2 in \\[2, 8\\]"):
                ExpertUpdater(hidden_dim=768, num_experts=num_experts)

    def test_network_parameters_tracked(self):
        """Test that per-expert networks are properly tracked."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        # Check that all networks are nn.Modules
        for i in range(4):
            assert isinstance(updater.gate_networks[i], torch.nn.Linear)
            assert isinstance(updater.update_networks[i], torch.nn.Linear)

        # Check that parameters are tracked
        params = list(updater.parameters())
        # Each expert has 2 networks (gate + update), each with weight + bias = 4 params per expert
        # For 4 experts: 4 experts * 2 networks * 2 params = 16 params
        expected_params = 4 * 2 * 2
        assert len(params) == expected_params


class TestExpertGateComputation:
    """Test per-expert gate computation (AC: 1, 2)."""

    def test_compute_expert_gates_shapes(self):
        """Test that compute_expert_gates returns correct shapes."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        batch_size, memory_slots = 8, 16
        expert_state = torch.randn(batch_size, memory_slots, 768)
        write_hidden = torch.randn(batch_size, memory_slots, 768)

        g_j, u_j = updater.compute_expert_gates(expert_state, write_hidden, expert_idx=0)

        expected_shape = (batch_size, memory_slots, 768)
        assert g_j.shape == expected_shape
        assert u_j.shape == expected_shape

    def test_gate_value_ranges(self):
        """Test that gate values are in [0, 1] and updates in [-1, 1]."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        expert_state = torch.randn(8, 16, 768)
        write_hidden = torch.randn(8, 16, 768)

        g_j, u_j = updater.compute_expert_gates(expert_state, write_hidden, expert_idx=2)

        # Gates should be in [0, 1] due to sigmoid
        assert torch.all(g_j >= 0) and torch.all(g_j <= 1)

        # Updates should be in [-1, 1] due to tanh
        assert torch.all(u_j >= -1) and torch.all(u_j <= 1)

    def test_different_experts_different_gates(self):
        """Test that different experts compute different gates (separate parameters)."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        expert_state = torch.randn(8, 16, 768)
        write_hidden = torch.randn(8, 16, 768)

        # Compute gates for different experts
        g_0, u_0 = updater.compute_expert_gates(expert_state, write_hidden, expert_idx=0)
        g_1, u_1 = updater.compute_expert_gates(expert_state, write_hidden, expert_idx=1)

        # Different experts should produce different outputs (separate parameters)
        assert not torch.allclose(g_0, g_1)
        assert not torch.allclose(u_0, u_1)

    def test_invalid_expert_index(self):
        """Test that invalid expert index raises IndexError."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        expert_state = torch.randn(8, 16, 768)
        write_hidden = torch.randn(8, 16, 768)

        # Test out-of-bounds indices
        with pytest.raises(IndexError, match="expert_idx must be in \\[0, 3\\]"):
            updater.compute_expert_gates(expert_state, write_hidden, expert_idx=4)

        with pytest.raises(IndexError, match="expert_idx must be in \\[0, 3\\]"):
            updater.compute_expert_gates(expert_state, write_hidden, expert_idx=-1)


class TestRoutingModulation:
    """Test routing modulation application (AC: 3)."""

    def test_routing_modulation_shapes(self):
        """Test that routing modulation produces correct output shape."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        gates = torch.rand(8, 16, 768)  # Values in [0, 1]
        routing_probs = torch.rand(8, 1, 1)  # Single expert probability, reshaped

        modulated = updater.apply_routing_modulation(gates, routing_probs)

        assert modulated.shape == gates.shape

    def test_routing_modulation_value_range(self):
        """Test that modulated gates remain in [0, 1] range."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        gates = torch.rand(8, 16, 768)
        routing_probs = torch.rand(8, 1, 1)

        modulated = updater.apply_routing_modulation(gates, routing_probs)

        # Modulated gates should still be in [0, 1]
        assert torch.all(modulated >= 0) and torch.all(modulated <= 1)

    def test_routing_modulation_zero_probability(self):
        """Test that p_j=0 results in zero modulated gate."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        gates = torch.rand(8, 16, 768)
        routing_probs = torch.zeros(8, 1, 1)  # p_j = 0

        modulated = updater.apply_routing_modulation(gates, routing_probs)

        # When p_j=0, modulated gate should be all zeros
        assert torch.allclose(modulated, torch.zeros_like(modulated))

    def test_routing_modulation_one_probability(self):
        """Test that p_j=1 leaves gates unchanged."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        gates = torch.rand(8, 16, 768)
        routing_probs = torch.ones(8, 1, 1)  # p_j = 1

        modulated = updater.apply_routing_modulation(gates, routing_probs)

        # When p_j=1, modulated gate should equal original gate
        assert torch.allclose(modulated, gates)

    def test_routing_modulation_broadcast(self):
        """Test that routing probability broadcasts correctly."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        batch_size, memory_slots, hidden_dim = 8, 16, 768
        gates = torch.ones(batch_size, memory_slots, hidden_dim) * 0.5
        routing_probs = torch.tensor([0.2]).view(1, 1, 1).expand(batch_size, 1, 1)

        modulated = updater.apply_routing_modulation(gates, routing_probs)

        # Each element should be 0.5 * 0.2 = 0.1
        expected = torch.ones_like(gates) * 0.1
        assert torch.allclose(modulated, expected, atol=1e-6)


class TestMemoryProtection:
    """Test memory protection mechanism (AC: 5)."""

    def test_zero_routing_preserves_memory(self):
        """Test that p_j ≈ 0 preserves expert memory (no update)."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        batch_size, memory_slots, hidden_dim = 8, 16, 768
        expert_states = [torch.randn(batch_size, memory_slots, hidden_dim) for _ in range(4)]
        write_hiddens = torch.randn(batch_size, memory_slots, hidden_dim)

        # All routing probabilities ≈ 0 for expert 0, others get equal share
        routing_probs = torch.tensor([0.001, 0.333, 0.333, 0.333]).unsqueeze(0).expand(batch_size, 4)

        original_expert_0 = expert_states[0].clone()
        updated_states = updater(expert_states, write_hiddens, routing_probs)

        # Expert 0 should be nearly unchanged (memory protected)
        assert torch.allclose(updated_states[0], original_expert_0, atol=1e-2)

    def test_one_routing_applies_full_update(self):
        """Test that p_j ≈ 1 applies standard LSTM update behavior."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        batch_size, memory_slots, hidden_dim = 8, 16, 768
        expert_states = [torch.randn(batch_size, memory_slots, hidden_dim) for _ in range(4)]
        write_hiddens = torch.randn(batch_size, memory_slots, hidden_dim)

        # Expert 0 gets nearly all routing probability
        routing_probs = torch.tensor([0.97, 0.01, 0.01, 0.01]).unsqueeze(0).expand(batch_size, 4)

        original_expert_0 = expert_states[0].clone()
        updated_states = updater(expert_states, write_hiddens, routing_probs)

        # Expert 0 should be significantly changed (update applied)
        # Mean absolute difference should be substantial
        diff = torch.abs(updated_states[0] - original_expert_0).mean()
        assert diff > 0.01  # Expect noticeable change

    def test_partial_routing_partial_update(self):
        """Test that p_j = 0.5 applies partial update."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        batch_size, memory_slots, hidden_dim = 8, 16, 768
        expert_states = [torch.randn(batch_size, memory_slots, hidden_dim) for _ in range(4)]
        write_hiddens = torch.randn(batch_size, memory_slots, hidden_dim)

        # All experts get equal routing (0.25 each)
        routing_probs = torch.ones(batch_size, 4) * 0.25

        original_expert_0 = expert_states[0].clone()
        updated_states = updater(expert_states, write_hiddens, routing_probs)

        # Expert should be partially updated (not fully preserved, not fully changed)
        diff = torch.abs(updated_states[0] - original_expert_0).mean()
        assert 0.001 < diff < 1.0  # Some change, but not extreme


class TestGradientFlow:
    """Test gradient flow through routing probabilities (AC: 4)."""

    def test_gradients_flow_through_routing_probs(self):
        """Test that gradients flow from loss to routing probabilities."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        batch_size, memory_slots, hidden_dim = 8, 16, 768
        expert_states = [torch.randn(batch_size, memory_slots, hidden_dim) for _ in range(4)]
        write_hiddens = torch.randn(batch_size, memory_slots, hidden_dim)

        # Create leaf tensor for routing logits, then softmax
        routing_logits = torch.randn(batch_size, 4, requires_grad=True)
        routing_probs = torch.softmax(routing_logits, dim=-1)

        # Forward pass
        updated_states = updater(expert_states, write_hiddens, routing_probs)

        # Create dummy loss from updated states
        loss = sum(state.sum() for state in updated_states)

        # Backward pass
        loss.backward()

        # Verify routing_logits (leaf tensor) has gradients
        assert routing_logits.grad is not None
        assert not torch.all(routing_logits.grad == 0)  # Gradients should be non-zero

    def test_gradients_flow_to_gate_networks(self):
        """Test that gradients flow to gate network parameters."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        batch_size, memory_slots, hidden_dim = 8, 16, 768
        expert_states = [torch.randn(batch_size, memory_slots, hidden_dim) for _ in range(4)]
        write_hiddens = torch.randn(batch_size, memory_slots, hidden_dim)
        routing_probs = torch.softmax(torch.randn(batch_size, 4), dim=-1)

        # Forward pass
        updated_states = updater(expert_states, write_hiddens, routing_probs)

        # Create dummy loss
        loss = sum(state.sum() for state in updated_states)

        # Backward pass
        loss.backward()

        # Verify all gate network parameters have gradients
        for i in range(4):
            gate_weight = updater.gate_networks[i].weight
            gate_bias = updater.gate_networks[i].bias
            update_weight = updater.update_networks[i].weight
            update_bias = updater.update_networks[i].bias

            assert gate_weight.grad is not None
            assert gate_bias.grad is not None
            assert update_weight.grad is not None
            assert update_bias.grad is not None

    def test_no_detach_in_forward_pass(self):
        """Test that forward pass doesn't detach tensors (preserves gradient flow)."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        batch_size, memory_slots, hidden_dim = 4, 8, 768
        expert_states = [torch.randn(batch_size, memory_slots, hidden_dim, requires_grad=True) for _ in range(4)]
        write_hiddens = torch.randn(batch_size, memory_slots, hidden_dim, requires_grad=True)
        routing_probs = torch.softmax(torch.randn(batch_size, 4, requires_grad=True), dim=-1)

        # Forward pass
        updated_states = updater(expert_states, write_hiddens, routing_probs)

        # All outputs should require grad
        for state in updated_states:
            assert state.requires_grad


class TestShapeValidation:
    """Test shape validation and error handling (AC: 6)."""

    def test_validate_expert_states_not_list(self):
        """Test that non-list expert_states raises ValueError."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        expert_states = torch.randn(4, 8, 16, 768)  # Tensor instead of list
        write_hiddens = torch.randn(8, 16, 768)
        routing_probs = torch.rand(8, 4)

        with pytest.raises(ValueError, match="expert_states must be a list"):
            updater(expert_states, write_hiddens, routing_probs)

    def test_validate_wrong_expert_count(self):
        """Test that wrong number of experts raises ValueError."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        expert_states = [torch.randn(8, 16, 768) for _ in range(3)]  # Only 3 experts
        write_hiddens = torch.randn(8, 16, 768)
        routing_probs = torch.rand(8, 4)

        with pytest.raises(ValueError, match="expert_states must contain 4 experts"):
            updater(expert_states, write_hiddens, routing_probs)

    def test_validate_write_hiddens_dimension(self):
        """Test that wrong write_hiddens dimension raises ValueError."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        expert_states = [torch.randn(8, 16, 768) for _ in range(4)]
        write_hiddens = torch.randn(8, 768)  # Missing memory_slots dimension
        routing_probs = torch.rand(8, 4)

        with pytest.raises(ValueError, match="write_hiddens must be 3D"):
            updater(expert_states, write_hiddens, routing_probs)

    def test_validate_routing_probs_dimension(self):
        """Test that wrong routing_probs dimension raises ValueError."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        expert_states = [torch.randn(8, 16, 768) for _ in range(4)]
        write_hiddens = torch.randn(8, 16, 768)
        routing_probs = torch.rand(8)  # Missing num_experts dimension

        with pytest.raises(ValueError, match="routing_probs must be 2D"):
            updater(expert_states, write_hiddens, routing_probs)

    def test_validate_expert_state_shape_mismatch(self):
        """Test that mismatched expert state shapes raise ValueError."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        expert_states = [
            torch.randn(8, 16, 768),
            torch.randn(8, 16, 768),
            torch.randn(8, 8, 768),  # Wrong memory_slots
            torch.randn(8, 16, 768),
        ]
        write_hiddens = torch.randn(8, 16, 768)
        routing_probs = torch.rand(8, 4)

        with pytest.raises(ValueError, match="Expert 2 state has shape"):
            updater(expert_states, write_hiddens, routing_probs)

    def test_validate_batch_size_mismatch(self):
        """Test that batch size mismatch raises ValueError."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        expert_states = [torch.randn(8, 16, 768) for _ in range(4)]
        write_hiddens = torch.randn(8, 16, 768)
        routing_probs = torch.rand(4, 4)  # Wrong batch size

        with pytest.raises(ValueError, match="routing_probs must have shape"):
            updater(expert_states, write_hiddens, routing_probs)


class TestUpdateConsistency:
    """Test update consistency across different expert counts."""

    @pytest.mark.parametrize("num_experts", [2, 4, 8])
    def test_update_consistency_across_expert_counts(self, num_experts):
        """Test that updates work correctly for all valid expert counts."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=num_experts)

        batch_size, memory_slots, hidden_dim = 8, 16, 768
        expert_states = [torch.randn(batch_size, memory_slots, hidden_dim) for _ in range(num_experts)]
        write_hiddens = torch.randn(batch_size, memory_slots, hidden_dim)
        routing_probs = torch.softmax(torch.randn(batch_size, num_experts), dim=-1)

        # Forward pass should succeed
        updated_states = updater(expert_states, write_hiddens, routing_probs)

        # Verify output
        assert len(updated_states) == num_experts
        for state in updated_states:
            assert state.shape == (batch_size, memory_slots, hidden_dim)

    def test_batch_size_one(self):
        """Test edge case: batch_size=1."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        batch_size, memory_slots, hidden_dim = 1, 16, 768
        expert_states = [torch.randn(batch_size, memory_slots, hidden_dim) for _ in range(4)]
        write_hiddens = torch.randn(batch_size, memory_slots, hidden_dim)
        routing_probs = torch.softmax(torch.randn(batch_size, 4), dim=-1)

        updated_states = updater(expert_states, write_hiddens, routing_probs)

        assert len(updated_states) == 4
        for state in updated_states:
            assert state.shape == (batch_size, memory_slots, hidden_dim)

    def test_all_routing_probs_zero(self):
        """Test edge case: all routing probabilities = 0 (all experts frozen)."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        batch_size, memory_slots, hidden_dim = 8, 16, 768
        expert_states = [torch.randn(batch_size, memory_slots, hidden_dim) for _ in range(4)]
        write_hiddens = torch.randn(batch_size, memory_slots, hidden_dim)

        # All zeros (technically invalid probability distribution, but tests memory protection)
        routing_probs = torch.zeros(batch_size, 4)

        original_states = [state.clone() for state in expert_states]
        updated_states = updater(expert_states, write_hiddens, routing_probs)

        # All experts should be unchanged
        for original, updated in zip(original_states, updated_states):
            assert torch.allclose(original, updated, atol=1e-6)

    def test_single_expert_routing(self):
        """Test edge case: single expert gets all probability, rest get 0."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)

        batch_size, memory_slots, hidden_dim = 8, 16, 768
        expert_states = [torch.randn(batch_size, memory_slots, hidden_dim) for _ in range(4)]
        write_hiddens = torch.randn(batch_size, memory_slots, hidden_dim)

        # Expert 2 gets all probability
        routing_probs = torch.tensor([0.0, 0.0, 1.0, 0.0]).unsqueeze(0).expand(batch_size, 4)

        original_states = [state.clone() for state in expert_states]
        updated_states = updater(expert_states, write_hiddens, routing_probs)

        # Experts 0, 1, 3 should be nearly unchanged
        assert torch.allclose(updated_states[0], original_states[0], atol=1e-6)
        assert torch.allclose(updated_states[1], original_states[1], atol=1e-6)
        assert torch.allclose(updated_states[3], original_states[3], atol=1e-6)

        # Expert 2 should be updated
        diff = torch.abs(updated_states[2] - original_states[2]).mean()
        assert diff > 0.01


class TestReprMethod:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        updater = ExpertUpdater(hidden_dim=768, num_experts=4)
        repr_str = repr(updater)

        assert "ExpertUpdater" in repr_str
        assert "hidden_dim=768" in repr_str
        assert "num_experts=4" in repr_str
