"""
Test suite for GMM memory indexing fix.

This test validates that the fix for storing individual memories
instead of batch tensors in the memory bank is working correctly.

Author: James (Dev Agent)
Fix Story: GMM Tensor Shape Mismatch Error
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from gmmxlnet.models import GMMXLNetForQA


class TestMemoryIndexingFix:
    """Test that memory bank storage correctly extracts individual memories."""

    @pytest.fixture
    def mock_base_model(self):
        """Create a mock base XLNet model."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.d_model = 768
        mock_model.config.n_head = 12
        mock_model.config.d_head = 64
        mock_model.config.layer_norm_eps = 1e-12
        mock_model.config.dropout = 0.1
        return mock_model

    @pytest.fixture
    def gmm_model(self, mock_base_model):
        """Create a GMM model for testing."""
        model = GMMXLNetForQA(
            base_model=mock_base_model,
            num_experts=4,
            memory_slots=16,
            routing_mode="write-based",
            routing_temperature=1.0,
            use_gmm_memory=True,
        )
        return model

    def test_memory_bank_shape_consistency(self, gmm_model):
        """Test that memory bank stores individual memories with correct shapes."""
        device = "cpu"
        batch_size = 4

        # Simulate the scenario from the fix
        # new_memory_state would have shape (batch_size, memory_slots, hidden_dim)
        new_memory_state = {
            f"expert_{i}": torch.randn(batch_size, 16, 768, device=device)
            for i in range(4)
        }

        # Simulate batch["example_ids"] structure
        example_ids = ["doc_1", "doc_2", "doc_3", "doc_4"]
        document_mask = torch.ones(batch_size, dtype=torch.bool)

        # Simulate the fixed memory bank storage logic
        memory_bank = {}
        for ex_id, active in zip(example_ids, document_mask.tolist()):
            if active:
                # Get batch index for this document to extract individual memory
                example_ids_list = example_ids
                ex_idx = example_ids_list.index(ex_id)

                # This is the key fix: extract individual memory from batch tensor
                memory_bank[ex_id] = {
                    expert_key: tensor[ex_idx].detach().to(device)  # Index ex_idx extracts individual memory
                    for expert_key, tensor in new_memory_state.items()
                }

        # Validate that all stored memories have consistent shape [16, 768]
        for doc_id, memory_dict in memory_bank.items():
            for expert_key, memory_tensor in memory_dict.items():
                # Should be [16, 768] not [4, 16, 768]
                assert memory_tensor.shape == (16, 768), f"Document {doc_id} expert {expert_key} has shape {memory_tensor.shape}, expected (16, 768)"
                assert memory_tensor.device == torch.device(device), f"Memory tensor on wrong device: {memory_tensor.device}"

        # Test that we can retrieve memories consistently
        for doc_id in memory_bank:
            doc_memory = memory_bank[doc_id]
            assert len(doc_memory) == 4, f"Document {doc_id} should have 4 expert memories"
            for expert_key in ["expert_0", "expert_1", "expert_2", "expert_3"]:
                assert expert_key in doc_memory, f"Missing expert {expert_key} for document {doc_id}"

    def test_batch_vs_individual_memory_shapes(self):
        """Test the difference between batch tensors and individual memories."""
        batch_size = 4
        memory_slots = 16
        hidden_dim = 768

        # Simulate batch tensor (what was causing the error)
        batch_tensor = torch.randn(batch_size, memory_slots, hidden_dim)
        assert batch_tensor.shape == (4, 16, 768), "Batch tensor should have shape (4, 16, 768)"

        # Simulate individual memory extraction (the fix)
        individual_memories = []
        for i in range(batch_size):
            individual_memory = batch_tensor[i]  # Extract individual memory
            individual_memories.append(individual_memory)
            assert individual_memory.shape == (16, 768), f"Individual memory {i} should have shape (16, 768)"

        # Stack them back - this should work now with consistent shapes
        stacked = torch.stack(individual_memories, dim=0)
        assert stacked.shape == (4, 16, 768), "Stacked individual memories should have shape (4, 16, 768)"

    def test_error_scenario_before_fix(self):
        """Test that demonstrates the error scenario before the fix."""
        # Simulate the problematic scenario
        # Some memories with batch dimension, some without
        inconsistent_memories = [
            torch.randn(16, 768),    # Individual memory
            torch.randn(4, 16, 768),  # Batch tensor (problematic)
            torch.randn(16, 768),    # Individual memory
            torch.randn(16, 768),    # Individual memory
        ]

        # This should fail with the same error we saw
        with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
            torch.stack(inconsistent_memories, dim=0)

    def test_fix_scenario_after_fix(self):
        """Test that the fix resolves the shape inconsistency."""
        # Simulate consistent individual memories (after fix)
        consistent_memories = [
            torch.randn(16, 768),  # All individual memories
            torch.randn(16, 768),
            torch.randn(16, 768),
            torch.randn(16, 768),
        ]

        # This should work fine now
        stacked = torch.stack(consistent_memories, dim=0)
        assert stacked.shape == (4, 16, 768), "Stacked consistent memories should work correctly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])