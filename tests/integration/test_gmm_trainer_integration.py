"""
Integration tests for GMM trainer functionality.

Tests that the GMM model integrates correctly with the training infrastructure
and that memory initialization works in the trainer context.
"""

import pytest
import torch

from transformers import XLNetForQuestionAnsweringSimple
from gmmxlnet.models import GMMXLNetForQA


class MockBatch:
    """Mock batch object for testing."""

    def __init__(self, batch_size=2, seq_len=64):
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Create mock data
        self.example_ids = [f"doc_{i}" for i in range(batch_size)]
        self.input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        self.attention_mask = torch.ones(batch_size, seq_len)
        self.token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        self.start_positions = torch.randint(0, seq_len, (batch_size,))
        self.end_positions = torch.randint(0, seq_len, (batch_size,))
        self.document_mask = torch.ones(batch_size, dtype=torch.bool)  # All active


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self):
        self.example_to_segments = {"doc_0": [0, 1], "doc_1": [0, 1]}


class TestGMMTrainerIntegration:
    """Test GMM model integration with trainer infrastructure."""

    @pytest.fixture
    def base_model(self):
        """Create a base XLNet model for testing."""
        return XLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased")

    def test_gmm_model_forward_pass(self, base_model):
        """Test GMM model can perform forward pass with memory state."""
        # Create GMM model
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            memory_slots=16,
            use_gmm_memory=True,
        )

        # Create mock batch
        batch = MockBatch(batch_size=2, seq_len=128)
        device = torch.device("cpu")

        # Move tensors to device
        batch.input_ids = batch.input_ids.to(device)
        batch.attention_mask = batch.attention_mask.to(device)
        batch.token_type_ids = batch.token_type_ids.to(device)
        batch.start_positions = batch.start_positions.to(device)
        batch.end_positions = batch.end_positions.to(device)

        # Get initial memory state
        memory_state = model.get_initial_memory(batch.batch_size, device)

        # Perform forward pass
        with torch.no_grad():  # No gradients for test
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                token_type_ids=batch.token_type_ids,
                start_positions=batch.start_positions,
                end_positions=batch.end_positions,
                memory_state=memory_state,
                mem_read_ids=None,  # Not testing memory token functionality
                mem_write_ids=None,
            )

        # Check outputs
        assert "loss" in outputs or outputs["loss"] is not None
        assert "start_logits" in outputs
        assert "end_logits" in outputs
        assert "new_memory_state" in outputs

        # Check logits shapes
        assert outputs["start_logits"].shape == (batch.batch_size, batch.seq_len)
        assert outputs["end_logits"].shape == (batch.batch_size, batch.seq_len)

        # Check new memory state structure
        new_memory = outputs["new_memory_state"]
        assert isinstance(new_memory, dict)
        assert len(new_memory) == 4  # 4 experts
        for expert_idx in range(4):
            key = f"expert_{expert_idx}"
            assert key in new_memory
            assert new_memory[key].shape == (batch.batch_size, 16, 768)  # (batch, slots, hidden)

    def test_gmm_memory_bank_operations(self, base_model):
        """Test memory bank operations work correctly with GMM models."""
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            memory_slots=16,
            use_gmm_memory=True,
        )

        device = torch.device("cpu")
        batch_size = 2

        # Simulate memory bank operations
        memory_bank = {}

        # Test new document initialization
        ex_id = "test_doc_1"
        initial_memory = model.get_initial_memory(1, device)
        memory_bank[ex_id] = initial_memory

        # Verify memory bank stores correct structure
        assert ex_id in memory_bank
        stored_memory = memory_bank[ex_id]
        assert isinstance(stored_memory, dict)
        assert len(stored_memory) == 4
        for expert_idx in range(4):
            assert f"expert_{expert_idx}" in stored_memory

        # Test memory retrieval and modification
        retrieved_memory = memory_bank[ex_id]
        for expert_idx in range(4):
            key = f"expert_{expert_idx}"
            original_shape = retrieved_memory[key].shape
            # Simulate memory update (e.g., from training)
            updated_memory = retrieved_memory[key] + 0.01 * torch.randn_like(retrieved_memory[key])
            retrieved_memory[key] = updated_memory
            assert retrieved_memory[key].shape == original_shape

    def test_batch_memory_state_construction(self, base_model):
        """Test batch memory state construction works correctly."""
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            memory_slots=16,
            use_gmm_memory=True,
        )

        device = torch.device("cpu")
        batch_size = 3

        # Mock memory bank with existing and new documents
        memory_bank = {
            "existing_doc": model.get_initial_memory(1, device)
        }

        # Create mock batch with mixed new/continuing documents
        batch = MockBatch(batch_size=batch_size, seq_len=64)
        example_ids = ["new_doc_1", "existing_doc", "new_doc_2"]  # Mixed
        document_mask = torch.tensor([True, True, True])

        # Build memory state for batch (simulate trainer logic)
        memory_state_batch = {}
        for expert_idx in range(model.num_experts):
            expert_memories = []

            for ex_id, active in zip(example_ids, document_mask.tolist()):
                if not active:
                    # Inactive document
                    expert_memory = model.get_initial_memory(1, device)[f"expert_{expert_idx}"]
                else:
                    # Check if document exists in memory bank
                    prev = memory_bank.get(ex_id)
                    if prev is None:
                        # New document: use initial memory
                        expert_memory = model.get_initial_memory(1, device)[f"expert_{expert_idx}"]
                    else:
                        # Existing document: use stored memory
                        expert_memory = prev[f"expert_{expert_idx}"]

                expert_memories.append(expert_memory.squeeze(0))  # Remove batch dim

            # Stack expert memories across batch
            memory_state_batch[f"expert_{expert_idx}"] = torch.stack(expert_memories, dim=0)

        # Verify batch memory state structure
        assert isinstance(memory_state_batch, dict)
        assert len(memory_state_batch) == 4  # 4 experts

        for expert_idx in range(4):
            key = f"expert_{expert_idx}"
            assert key in memory_state_batch
            # Shape should be (batch_size, memory_slots, hidden_dim)
            assert memory_state_batch[key].shape == (batch_size, 16, 768)

    def test_gmm_trainer_memory_compatibility(self, base_model):
        """Test that GMM model is compatible with trainer memory expectations."""
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            memory_slots=16,
            use_gmm_memory=True,
        )

        device = torch.device("cpu")

        # Test that get_initial_memory works in trainer context
        batch_size = 1
        initial_memory = model.get_initial_memory(batch_size, device)

        # Trainer expects to be able to access individual expert memories
        # This should work without KeyError
        try:
            for expert_idx in range(model.num_experts):
                expert_memory = initial_memory[f"expert_{expert_idx}"]
                assert expert_memory.shape[0] == batch_size
        except KeyError as e:
            pytest.fail(f"Trainer memory access failed: {e}")

        # Test that model can handle None memory_state (trainer behavior)
        with torch.no_grad():
            batch = MockBatch(batch_size=1, seq_len=32)
            outputs = model(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                token_type_ids=batch.token_type_ids.to(device),
                memory_state=None,  # Should trigger internal initialization
            )
            assert "new_memory_state" in outputs
            assert isinstance(outputs["new_memory_state"], dict)

    def test_memory_state_propagation(self, base_model):
        """Test memory state propagation across multiple forward passes."""
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=2,  # Use fewer experts for faster test
            memory_slots=8,  # Use fewer slots
            use_gmm_memory=True,
        )

        device = torch.device("cpu")
        batch_size = 1

        # Simulate multiple time steps for the same document
        batch = MockBatch(batch_size=batch_size, seq_len=32)
        batch.input_ids = batch.input_ids.to(device)
        batch.attention_mask = batch.attention_mask.to(device)
        batch.token_type_ids = batch.token_type_ids.to(device)

        # Initial memory state
        memory_state = model.get_initial_memory(batch_size, device)

        # Multiple forward passes (simulate time steps)
        for step in range(3):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    token_type_ids=batch.token_type_ids,
                    memory_state=memory_state,
                )

                # Update memory state for next step
                memory_state = outputs["new_memory_state"]

                # Verify memory state maintains structure
                assert isinstance(memory_state, dict)
                assert len(memory_state) == 2  # 2 experts
                for expert_idx in range(2):
                    key = f"expert_{expert_idx}"
                    assert key in memory_state
                    assert memory_state[key].shape == (batch_size, 8, 768)

    def test_gmm_vs_non_gmm_compatibility(self, base_model):
        """Test that GMM and non-GMM models can coexist."""
        # Create both GMM and non-GMM models
        gmm_model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            use_gmm_memory=True,
        )

        non_gmm_model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            use_gmm_memory=False,
        )

        device = torch.device("cpu")
        batch = MockBatch(batch_size=1, seq_len=32)
        batch.input_ids = batch.input_ids.to(device)
        batch.attention_mask = batch.attention_mask.to(device)

        # Test GMM model memory initialization
        gmm_memory = gmm_model.get_initial_memory(1, device)
        assert isinstance(gmm_memory, dict)
        assert len(gmm_memory) == 4

        # Test non-GMM model memory initialization
        non_gmm_memory = non_gmm_model.get_initial_memory(1, device)
        assert isinstance(non_gmm_memory, dict)
        assert len(non_gmm_memory) == 0  # Empty dict when disabled

        # Both should be able to perform forward pass
        with torch.no_grad():
            gmm_outputs = gmm_model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                memory_state=gmm_memory,
            )
            assert "new_memory_state" in gmm_outputs
            assert isinstance(gmm_outputs["new_memory_state"], dict)

            non_gmm_outputs = non_gmm_model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                memory_state=non_gmm_memory,
            )
            assert "new_memory_state" in non_gmm_outputs
            assert isinstance(non_gmm_outputs["new_memory_state"], dict)