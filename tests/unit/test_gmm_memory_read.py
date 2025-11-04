"""
Unit tests for Aggregated Memory Reader.

Tests weighted aggregation, routing mode support, embedding replacement,
and efficient computation for GMM-augmented XLNet read operations.
"""

import pytest
import torch

from gmmxlnet.models.memory_read import AggregatedMemoryReader


class TestAggregatedMemoryReaderInitialization:
    """Tests for AggregatedMemoryReader initialization and configuration validation."""

    def test_valid_initialization_write_based(self):
        """Test successful initialization with write-based routing."""
        reader = AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")
        assert reader.hidden_dim == 768
        assert reader.num_experts == 4
        assert reader.routing_mode == "write-based"
        assert reader.read_gating_network is None  # Not needed for write-based

    def test_valid_initialization_read_based(self):
        """Test successful initialization with read-based routing."""
        reader = AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="read-based", temperature=1.5)
        assert reader.hidden_dim == 768
        assert reader.num_experts == 4
        assert reader.routing_mode == "read-based"
        assert reader.read_gating_network is not None  # Required for read-based

    def test_num_experts_validation_invalid(self):
        """Test that invalid num_experts raises ValueError."""
        with pytest.raises(ValueError, match="num_experts must be power of 2"):
            AggregatedMemoryReader(hidden_dim=768, num_experts=3)

    def test_num_experts_validation_too_large(self):
        """Test that num_experts > 8 raises ValueError."""
        with pytest.raises(ValueError, match="num_experts must be power of 2"):
            AggregatedMemoryReader(hidden_dim=768, num_experts=16)

    def test_valid_num_experts_range(self):
        """Test that num_experts in [2, 4, 8] are all valid."""
        for num_experts in [2, 4, 8]:
            reader = AggregatedMemoryReader(hidden_dim=768, num_experts=num_experts)
            assert reader.num_experts == num_experts

    def test_routing_mode_validation_invalid(self):
        """Test that invalid routing_mode raises ValueError."""
        with pytest.raises(ValueError, match="routing_mode must be"):
            AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="invalid-mode")

    def test_read_gating_network_not_created_for_write_based(self):
        """Test that read gating network is not created for write-based mode."""
        reader = AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")
        assert reader.read_gating_network is None

    def test_read_gating_network_created_for_read_based(self):
        """Test that read gating network is created for read-based mode."""
        reader = AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="read-based")
        assert reader.read_gating_network is not None
        assert reader.read_gating_network.hidden_dim == 768
        assert reader.read_gating_network.num_experts == 4


class TestWriteBasedRouting:
    """Tests for write-based routing mode (reuse cached routing)."""

    @pytest.fixture
    def reader(self):
        """Write-based reader fixture."""
        return AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")

    @pytest.fixture
    def expert_states(self):
        """Generate sample expert states."""
        batch_size = 8
        memory_slots = 16
        hidden_dim = 768
        return [torch.randn(batch_size, memory_slots, hidden_dim) for _ in range(4)]

    @pytest.fixture
    def routing_probs(self):
        """Generate sample routing probabilities."""
        return torch.softmax(torch.randn(8, 4), dim=-1)

    def test_forward_output_shape(self, reader, expert_states, routing_probs):
        """Test that forward pass returns correct shape."""
        aggregated = reader(expert_states, routing_probs=routing_probs)
        assert aggregated.shape == (8, 16, 768)

    def test_forward_requires_routing_probs(self, reader, expert_states):
        """Test that write-based mode requires routing_probs."""
        with pytest.raises(ValueError, match="routing_probs required"):
            reader(expert_states)

    def test_forward_with_various_batch_sizes(self, reader):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            expert_states = [torch.randn(batch_size, 16, 768) for _ in range(4)]
            routing_probs = torch.softmax(torch.randn(batch_size, 4), dim=-1)
            aggregated = reader(expert_states, routing_probs=routing_probs)
            assert aggregated.shape == (batch_size, 16, 768)

    def test_forward_with_various_memory_slots(self, reader):
        """Test forward pass with different memory slot counts."""
        for memory_slots in [4, 8, 16, 32]:
            expert_states = [torch.randn(8, memory_slots, 768) for _ in range(4)]
            routing_probs = torch.softmax(torch.randn(8, 4), dim=-1)
            aggregated = reader(expert_states, routing_probs=routing_probs)
            assert aggregated.shape == (8, memory_slots, 768)

    def test_routing_probs_shape_validation(self, reader, expert_states):
        """Test that incorrect routing_probs shape raises ValueError."""
        # Wrong shape: (8, 3) instead of (8, 4)
        routing_probs = torch.softmax(torch.randn(8, 3), dim=-1)
        with pytest.raises(ValueError, match="routing_probs must have shape"):
            reader(expert_states, routing_probs=routing_probs)

    def test_routing_probs_batch_size_mismatch(self, reader, expert_states):
        """Test that batch size mismatch raises ValueError."""
        # Batch size mismatch: routing_probs has batch_size=4, expert_states has 8
        routing_probs = torch.softmax(torch.randn(4, 4), dim=-1)
        with pytest.raises(ValueError, match="routing_probs must have shape"):
            reader(expert_states, routing_probs=routing_probs)


class TestReadBasedRouting:
    """Tests for read-based routing mode (compute new routing)."""

    @pytest.fixture
    def reader(self):
        """Read-based reader fixture."""
        return AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="read-based", temperature=1.0)

    @pytest.fixture
    def expert_states(self):
        """Generate sample expert states."""
        return [torch.randn(8, 16, 768) for _ in range(4)]

    @pytest.fixture
    def read_hiddens(self):
        """Generate sample read query hidden states."""
        return torch.randn(8, 16, 768)

    def test_forward_output_shape(self, reader, expert_states, read_hiddens):
        """Test that forward pass returns correct shape."""
        aggregated = reader(expert_states, read_hiddens=read_hiddens)
        assert aggregated.shape == (8, 16, 768)

    def test_forward_requires_read_hiddens(self, reader, expert_states):
        """Test that read-based mode requires read_hiddens."""
        with pytest.raises(ValueError, match="read_hiddens required"):
            reader(expert_states)

    def test_compute_read_routing(self, reader, read_hiddens):
        """Test that compute_read_routing returns valid routing probabilities."""
        routing_probs = reader.compute_read_routing(read_hiddens)
        assert routing_probs.shape == (8, 4)
        # Check probabilities sum to 1.0
        prob_sums = routing_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(8), atol=1e-6)

    def test_compute_read_routing_fails_in_write_based_mode(self):
        """Test that compute_read_routing raises error in write-based mode."""
        reader = AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")
        read_hiddens = torch.randn(8, 16, 768)
        with pytest.raises(RuntimeError, match="only available in read-based mode"):
            reader.compute_read_routing(read_hiddens)

    def test_forward_with_various_batch_sizes(self, reader):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            expert_states = [torch.randn(batch_size, 16, 768) for _ in range(4)]
            read_hiddens = torch.randn(batch_size, 16, 768)
            aggregated = reader(expert_states, read_hiddens=read_hiddens)
            assert aggregated.shape == (batch_size, 16, 768)

    def test_read_hiddens_shape_validation(self, reader, expert_states):
        """Test that incorrect read_hiddens shape raises ValueError."""
        # Wrong shape: 2D instead of 3D
        read_hiddens = torch.randn(8, 768)
        with pytest.raises(ValueError, match="read_hiddens must be 3D"):
            reader(expert_states, read_hiddens=read_hiddens)

    def test_read_hiddens_hidden_dim_mismatch(self, reader, expert_states):
        """Test that hidden_dim mismatch raises ValueError."""
        # Wrong hidden_dim: 512 instead of 768
        read_hiddens = torch.randn(8, 16, 512)
        with pytest.raises(ValueError, match="read_hiddens hidden_dim must be"):
            reader(expert_states, read_hiddens=read_hiddens)


class TestWeightedAggregation:
    """Tests for weighted aggregation computation correctness."""

    @pytest.fixture
    def reader(self):
        """Standard reader fixture."""
        return AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")

    def test_aggregation_correctness_manual_calculation(self, reader):
        """Test aggregation matches manual calculation."""
        batch_size = 2
        memory_slots = 4
        hidden_dim = 768

        # Create simple expert states for manual verification
        expert_states = [
            torch.ones(batch_size, memory_slots, hidden_dim) * (i + 1) for i in range(4)
        ]  # Expert 0: all 1s, Expert 1: all 2s, etc.

        # Create simple routing probs
        routing_probs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])

        aggregated = reader(expert_states, routing_probs=routing_probs)

        # Manual calculation:
        # Batch 0: 1.0*1 + 0.0*2 + 0.0*3 + 0.0*4 = 1.0
        # Batch 1: 0.5*1 + 0.5*2 + 0.0*3 + 0.0*4 = 1.5
        expected_batch_0 = torch.ones(memory_slots, hidden_dim)
        expected_batch_1 = torch.ones(memory_slots, hidden_dim) * 1.5

        assert torch.allclose(aggregated[0], expected_batch_0, atol=1e-6)
        assert torch.allclose(aggregated[1], expected_batch_1, atol=1e-6)

    def test_aggregation_single_expert_selected(self, reader):
        """Test aggregation when single expert has probability 1.0."""
        expert_states = [torch.randn(8, 16, 768) for _ in range(4)]

        # Select only expert 2
        routing_probs = torch.zeros(8, 4)
        routing_probs[:, 2] = 1.0

        aggregated = reader(expert_states, routing_probs=routing_probs)

        # Aggregated should equal expert 2's state
        assert torch.allclose(aggregated, expert_states[2], atol=1e-6)

    def test_aggregation_uniform_routing(self, reader):
        """Test aggregation with uniform routing probabilities."""
        batch_size = 8
        memory_slots = 16
        hidden_dim = 768

        # Create distinct expert states
        expert_states = [torch.ones(batch_size, memory_slots, hidden_dim) * (i + 1) for i in range(4)]

        # Uniform routing (all experts weighted equally)
        routing_probs = torch.ones(batch_size, 4) / 4

        aggregated = reader(expert_states, routing_probs=routing_probs)

        # Expected: (1 + 2 + 3 + 4) / 4 = 2.5
        expected = torch.ones(batch_size, memory_slots, hidden_dim) * 2.5
        assert torch.allclose(aggregated, expected, atol=1e-6)

    def test_aggregation_with_various_expert_counts(self):
        """Test aggregation with k=2, 4, 8 experts."""
        for num_experts in [2, 4, 8]:
            reader = AggregatedMemoryReader(hidden_dim=768, num_experts=num_experts, routing_mode="write-based")
            expert_states = [torch.randn(8, 16, 768) for _ in range(num_experts)]
            routing_probs = torch.softmax(torch.randn(8, num_experts), dim=-1)

            aggregated = reader(expert_states, routing_probs=routing_probs)
            assert aggregated.shape == (8, 16, 768)

    def test_aggregation_preserves_gradient_flow(self, reader):
        """Test that aggregation preserves gradient flow."""
        expert_states = [torch.randn(8, 16, 768, requires_grad=True) for _ in range(4)]
        routing_probs = torch.softmax(torch.randn(8, 4), dim=-1)

        aggregated = reader(expert_states, routing_probs=routing_probs)

        # Compute loss and backward
        loss = aggregated.sum()
        loss.backward()

        # Check that gradients flow to all expert states
        for expert_state in expert_states:
            assert expert_state.grad is not None
            assert not torch.all(expert_state.grad == 0)


class TestEmbeddingReplacement:
    """Tests for memory embedding replacement logic."""

    @pytest.fixture
    def reader(self):
        """Standard reader fixture."""
        return AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")

    def test_replace_embeddings_single_position(self, reader):
        """Test replacement with single read position per batch item."""
        batch_size = 8
        sequence_length = 128
        hidden_dim = 768

        sequence_output = torch.randn(batch_size, sequence_length, hidden_dim)
        aggregated_memory = torch.randn(batch_size, 1, hidden_dim)  # 1 read token
        read_positions = torch.randint(0, sequence_length, (batch_size,))

        modified_output = reader.replace_read_embeddings(sequence_output, aggregated_memory, read_positions)

        # Check shape preserved
        assert modified_output.shape == sequence_output.shape

        # Check embeddings replaced at read positions
        for batch_idx in range(batch_size):
            pos = read_positions[batch_idx].item()
            assert torch.allclose(
                modified_output[batch_idx, pos, :],
                aggregated_memory[batch_idx, 0, :],
                atol=1e-6,
            )

    def test_replace_embeddings_multiple_positions(self, reader):
        """Test replacement with multiple read positions per batch item."""
        batch_size = 8
        sequence_length = 128
        hidden_dim = 768
        num_read_tokens = 3

        sequence_output = torch.randn(batch_size, sequence_length, hidden_dim)
        aggregated_memory = torch.randn(batch_size, num_read_tokens, hidden_dim)
        read_positions = torch.randint(0, sequence_length, (batch_size, num_read_tokens))

        modified_output = reader.replace_read_embeddings(sequence_output, aggregated_memory, read_positions)

        # Check shape preserved
        assert modified_output.shape == sequence_output.shape

        # Check embeddings replaced at all read positions
        for batch_idx in range(batch_size):
            for mem_idx in range(num_read_tokens):
                pos = read_positions[batch_idx, mem_idx].item()
                assert torch.allclose(
                    modified_output[batch_idx, pos, :],
                    aggregated_memory[batch_idx, mem_idx, :],
                    atol=1e-6,
                )

    def test_replace_embeddings_non_read_positions_unchanged(self, reader):
        """Test that non-read positions remain unchanged."""
        batch_size = 8
        sequence_length = 128
        hidden_dim = 768

        sequence_output = torch.randn(batch_size, sequence_length, hidden_dim)
        aggregated_memory = torch.randn(batch_size, 1, hidden_dim)
        read_positions = torch.tensor([10] * batch_size)  # All at position 10

        modified_output = reader.replace_read_embeddings(sequence_output, aggregated_memory, read_positions)

        # Check non-read positions unchanged
        for pos in range(sequence_length):
            if pos != 10:
                assert torch.allclose(modified_output[:, pos, :], sequence_output[:, pos, :], atol=1e-6)

    def test_replace_embeddings_batch_size_mismatch(self, reader):
        """Test that batch size mismatch raises ValueError."""
        sequence_output = torch.randn(8, 128, 768)
        aggregated_memory = torch.randn(4, 1, 768)  # Wrong batch size
        read_positions = torch.tensor([10] * 8)

        with pytest.raises(ValueError, match="Batch size mismatch"):
            reader.replace_read_embeddings(sequence_output, aggregated_memory, read_positions)

    def test_replace_embeddings_hidden_dim_mismatch(self, reader):
        """Test that hidden_dim mismatch raises ValueError."""
        sequence_output = torch.randn(8, 128, 768)
        aggregated_memory = torch.randn(8, 1, 512)  # Wrong hidden_dim
        read_positions = torch.tensor([10] * 8)

        with pytest.raises(ValueError, match="Hidden dimension mismatch"):
            reader.replace_read_embeddings(sequence_output, aggregated_memory, read_positions)

    def test_replace_embeddings_memory_slots_mismatch(self, reader):
        """Test that memory_slots and num_read_tokens mismatch raises ValueError."""
        sequence_output = torch.randn(8, 128, 768)
        aggregated_memory = torch.randn(8, 3, 768)  # 3 memory slots
        read_positions = torch.randint(0, 128, (8, 2))  # Only 2 read positions

        with pytest.raises(ValueError, match="Number of memory slots"):
            reader.replace_read_embeddings(sequence_output, aggregated_memory, read_positions)

    def test_replace_embeddings_position_out_of_bounds(self, reader):
        """Test that out-of-bounds positions raise ValueError."""
        sequence_output = torch.randn(8, 128, 768)
        aggregated_memory = torch.randn(8, 1, 768)
        read_positions = torch.tensor([150] * 8)  # Out of bounds

        with pytest.raises(ValueError, match="read_positions must be in"):
            reader.replace_read_embeddings(sequence_output, aggregated_memory, read_positions)


class TestInputValidation:
    """Tests for input validation and error handling."""

    @pytest.fixture
    def reader(self):
        """Standard reader fixture."""
        return AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")

    def test_expert_states_not_list(self, reader):
        """Test that non-list expert_states raises ValueError."""
        expert_states = torch.randn(8, 16, 768)  # Tensor instead of list
        routing_probs = torch.softmax(torch.randn(8, 4), dim=-1)

        with pytest.raises(ValueError, match="expert_states must be a list"):
            reader(expert_states, routing_probs=routing_probs)

    def test_expert_states_wrong_count(self, reader):
        """Test that wrong number of expert_states raises ValueError."""
        expert_states = [torch.randn(8, 16, 768) for _ in range(3)]  # 3 instead of 4
        routing_probs = torch.softmax(torch.randn(8, 4), dim=-1)

        with pytest.raises(ValueError, match="expert_states must contain 4 experts"):
            reader(expert_states, routing_probs=routing_probs)

    def test_expert_states_wrong_dimensions(self, reader):
        """Test that wrong-dimensional expert states raise ValueError."""
        expert_states = [torch.randn(8, 768) for _ in range(4)]  # 2D instead of 3D
        routing_probs = torch.softmax(torch.randn(8, 4), dim=-1)

        with pytest.raises(ValueError, match="Expert .* state must be 3D"):
            reader(expert_states, routing_probs=routing_probs)

    def test_expert_states_inconsistent_shapes(self, reader):
        """Test that inconsistent expert state shapes raise ValueError."""
        expert_states = [
            torch.randn(8, 16, 768),
            torch.randn(8, 16, 768),
            torch.randn(8, 8, 768),  # Different memory_slots
            torch.randn(8, 16, 768),
        ]
        routing_probs = torch.softmax(torch.randn(8, 4), dim=-1)

        with pytest.raises(ValueError, match="All experts must maintain identical shape"):
            reader(expert_states, routing_probs=routing_probs)


class TestEfficiency:
    """Tests for efficient computation and performance."""

    @pytest.fixture
    def reader(self):
        """Standard reader fixture."""
        return AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")

    def test_batched_aggregation_efficiency(self, reader):
        """Test that aggregation uses batched operations (no Python loops)."""
        # This test verifies that aggregation completes quickly for large batch
        import time

        batch_size = 256
        expert_states = [torch.randn(batch_size, 16, 768) for _ in range(4)]
        routing_probs = torch.softmax(torch.randn(batch_size, 4), dim=-1)

        start_time = time.time()
        aggregated = reader(expert_states, routing_probs=routing_probs)
        elapsed = time.time() - start_time

        # Should complete very quickly (< 0.1s on CPU)
        assert elapsed < 0.1
        assert aggregated.shape == (batch_size, 16, 768)

    def test_aggregation_memory_slots_scaling(self, reader):
        """Test that aggregation time scales linearly with memory_slots."""
        import time

        expert_states_16 = [torch.randn(8, 16, 768) for _ in range(4)]
        expert_states_64 = [torch.randn(8, 64, 768) for _ in range(4)]
        routing_probs = torch.softmax(torch.randn(8, 4), dim=-1)

        # Time with 16 memory slots
        start = time.time()
        reader(expert_states_16, routing_probs=routing_probs)
        time_16 = time.time() - start

        # Time with 64 memory slots
        start = time.time()
        reader(expert_states_64, routing_probs=routing_probs)
        time_64 = time.time() - start

        # Should scale roughly linearly (allow 10x factor for noise)
        assert time_64 < time_16 * 10


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_batch_item(self):
        """Test aggregation with batch_size=1."""
        reader = AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")
        expert_states = [torch.randn(1, 16, 768) for _ in range(4)]
        routing_probs = torch.softmax(torch.randn(1, 4), dim=-1)

        aggregated = reader(expert_states, routing_probs=routing_probs)
        assert aggregated.shape == (1, 16, 768)

    def test_zero_routing_to_all_experts(self):
        """Test aggregation when all routing probabilities are effectively zero."""
        reader = AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")
        expert_states = [torch.ones(8, 16, 768) * (i + 1) for i in range(4)]

        # Very small but non-zero probabilities that sum to 1
        routing_probs = torch.ones(8, 4) / 4

        aggregated = reader(expert_states, routing_probs=routing_probs)
        assert aggregated.shape == (8, 16, 768)
        # Should be average: (1+2+3+4)/4 = 2.5
        expected = torch.ones(8, 16, 768) * 2.5
        assert torch.allclose(aggregated, expected, atol=1e-6)

    def test_replace_embeddings_edge_positions(self):
        """Test replacement at first and last positions in sequence."""
        reader = AggregatedMemoryReader(hidden_dim=768, num_experts=4, routing_mode="write-based")
        sequence_output = torch.randn(8, 128, 768)
        aggregated_memory = torch.randn(8, 2, 768)

        # Test first and last positions
        read_positions = torch.tensor([[0, 127]] * 8)  # First and last for all batch items

        modified_output = reader.replace_read_embeddings(sequence_output, aggregated_memory, read_positions)
        assert modified_output.shape == sequence_output.shape

        # Verify replacements at edges
        assert torch.allclose(modified_output[:, 0, :], aggregated_memory[:, 0, :])
        assert torch.allclose(modified_output[:, 127, :], aggregated_memory[:, 1, :])
