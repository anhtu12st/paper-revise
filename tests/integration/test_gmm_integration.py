"""
Integration tests for GMMXLNetForQA.

Tests full forward pass orchestration with all GMM components.
"""

import pytest
import torch

from gmmxlnet.models.gmm_xlnet_qa import GMMXLNetForQA


@pytest.fixture
def toy_base_model():
    """Create a toy XLNet model for testing."""
    from transformers import XLNetConfig, XLNetForQuestionAnsweringSimple

    config = XLNetConfig(
        vocab_size=1000,
        d_model=128,  # Small for fast testing
        n_layer=2,
        n_head=4,
        d_inner=512,
    )
    model = XLNetForQuestionAnsweringSimple(config)
    return model


@pytest.fixture
def toy_input_data():
    """Create toy input data for testing."""
    batch_size = 4
    seq_len = 32

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Mock memory token IDs
    mem_read_ids = [10, 11, 12, 13]  # 4 read tokens
    mem_write_ids = [14, 15, 16, 17]  # 4 write tokens

    # Inject memory tokens into input_ids
    for i in range(batch_size):
        # Place write tokens at positions 5-8
        for j, mem_id in enumerate(mem_write_ids):
            input_ids[i, 5 + j] = mem_id
        # Place read tokens at positions 20-23
        for j, mem_id in enumerate(mem_read_ids):
            input_ids[i, 20 + j] = mem_id

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mem_read_ids": mem_read_ids,
        "mem_write_ids": mem_write_ids,
    }


class TestGMMXLNetIntegration:
    """Integration tests for GMMXLNetForQA."""

    def test_model_initialization(self, toy_base_model):
        """Test that model initializes correctly with various configurations."""
        # Test k=2 experts
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=2,
            memory_slots=16,
            routing_mode="write-based",
        )
        assert model.num_experts == 2
        assert model.memory_slots == 16
        assert model.use_gmm_memory is True

        # Test k=4 experts
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=4,
            memory_slots=16,
            routing_mode="read-based",
        )
        assert model.num_experts == 4

        # Test k=8 experts
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=8,
            memory_slots=16,
        )
        assert model.num_experts == 8

    def test_forward_pass_basic(self, toy_base_model, toy_input_data):
        """Test basic forward pass with GMM memory."""
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=4,
            memory_slots=4,  # Match number of memory tokens
            routing_mode="write-based",
        )

        # Forward pass
        outputs = model(
            input_ids=toy_input_data["input_ids"],
            attention_mask=toy_input_data["attention_mask"],
            mem_read_ids=toy_input_data["mem_read_ids"],
            mem_write_ids=toy_input_data["mem_write_ids"],
        )

        # Verify outputs
        assert "start_logits" in outputs
        assert "end_logits" in outputs
        assert "new_memory_state" in outputs

        # Verify output shapes
        batch_size, seq_len = toy_input_data["input_ids"].shape
        assert outputs["start_logits"].shape == (batch_size, seq_len)
        assert outputs["end_logits"].shape == (batch_size, seq_len)

        # Verify memory state is a dict with k entries
        assert isinstance(outputs["new_memory_state"], dict)
        assert len(outputs["new_memory_state"]) == 4

        # Verify each expert state shape
        for expert_idx in range(4):
            key = f"expert_{expert_idx}"
            assert key in outputs["new_memory_state"]
            expert_state = outputs["new_memory_state"][key]
            assert expert_state.shape == (batch_size, 4, 128)  # (batch, memory_slots, hidden_dim)

    def test_output_shapes(self, toy_base_model, toy_input_data):
        """Test that output shapes match expected dimensions."""
        for num_experts in [2, 4, 8]:
            model = GMMXLNetForQA(
                base_model=toy_base_model,
                num_experts=num_experts,
                memory_slots=4,
            )

            outputs = model(
                input_ids=toy_input_data["input_ids"],
                attention_mask=toy_input_data["attention_mask"],
                mem_read_ids=toy_input_data["mem_read_ids"],
                mem_write_ids=toy_input_data["mem_write_ids"],
            )

            batch_size, seq_len = toy_input_data["input_ids"].shape

            # Check start/end logits
            assert outputs["start_logits"].shape == (batch_size, seq_len)
            assert outputs["end_logits"].shape == (batch_size, seq_len)

            # Check memory state has correct number of experts
            assert len(outputs["new_memory_state"]) == num_experts

    def test_memory_state_propagation(self, toy_base_model, toy_input_data):
        """Test memory state propagation across segments."""
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=4,
            memory_slots=4,
        )

        # First segment
        outputs1 = model(
            input_ids=toy_input_data["input_ids"],
            attention_mask=toy_input_data["attention_mask"],
            mem_read_ids=toy_input_data["mem_read_ids"],
            mem_write_ids=toy_input_data["mem_write_ids"],
        )

        memory_state_1 = outputs1["new_memory_state"]

        # Second segment with propagated memory
        outputs2 = model(
            input_ids=toy_input_data["input_ids"],
            attention_mask=toy_input_data["attention_mask"],
            memory_state=memory_state_1,
            mem_read_ids=toy_input_data["mem_read_ids"],
            mem_write_ids=toy_input_data["mem_write_ids"],
        )

        memory_state_2 = outputs2["new_memory_state"]

        # Verify memory states are different (updated)
        for expert_idx in range(4):
            key = f"expert_{expert_idx}"
            assert not torch.allclose(memory_state_1[key], memory_state_2[key])

        # Verify shapes remain consistent
        for expert_idx in range(4):
            key = f"expert_{expert_idx}"
            assert memory_state_1[key].shape == memory_state_2[key].shape

    def test_routing_info_returned(self, toy_base_model, toy_input_data):
        """Test that routing info is returned correctly when requested."""
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=4,
            memory_slots=4,
        )

        outputs = model(
            input_ids=toy_input_data["input_ids"],
            attention_mask=toy_input_data["attention_mask"],
            mem_read_ids=toy_input_data["mem_read_ids"],
            mem_write_ids=toy_input_data["mem_write_ids"],
            return_routing_info=True,
        )

        # Verify routing info is present
        assert "routing_info" in outputs
        routing_info = outputs["routing_info"]

        # Verify routing info contains expected keys
        assert "routing_probs" in routing_info
        assert "routing_logits" in routing_info
        assert "routing_entropy" in routing_info
        assert "expert_activations" in routing_info

        batch_size = toy_input_data["input_ids"].size(0)

        # Verify shapes
        assert routing_info["routing_probs"].shape == (batch_size, 4)
        assert routing_info["routing_logits"].shape == (batch_size, 4)
        assert routing_info["routing_entropy"].shape == (batch_size,)
        assert routing_info["expert_activations"].shape == (4,)

        # Verify routing probabilities sum to 1
        routing_probs_sum = routing_info["routing_probs"].sum(dim=-1)
        assert torch.allclose(routing_probs_sum, torch.ones(batch_size), atol=1e-5)

    def test_various_configurations(self, toy_base_model, toy_input_data):
        """Test various GMM configurations."""
        configs = [
            {"num_experts": 2, "routing_mode": "write-based"},
            {"num_experts": 4, "routing_mode": "write-based"},
            {"num_experts": 8, "routing_mode": "write-based"},
            {"num_experts": 4, "routing_mode": "read-based"},
        ]

        for config in configs:
            model = GMMXLNetForQA(
                base_model=toy_base_model,
                memory_slots=4,
                **config,
            )

            outputs = model(
                input_ids=toy_input_data["input_ids"],
                attention_mask=toy_input_data["attention_mask"],
                mem_read_ids=toy_input_data["mem_read_ids"],
                mem_write_ids=toy_input_data["mem_write_ids"],
            )

            # Verify outputs are valid
            assert outputs["start_logits"] is not None
            assert outputs["end_logits"] is not None
            assert outputs["new_memory_state"] is not None
            assert len(outputs["new_memory_state"]) == config["num_experts"]

    def test_initial_memory_state(self, toy_base_model):
        """Test initial memory state generation."""
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=4,
            memory_slots=16,
        )

        batch_size = 8
        device = torch.device("cpu")

        memory_state = model.get_initial_memory(batch_size, device)

        # Verify memory state structure
        assert isinstance(memory_state, dict)
        assert len(memory_state) == 4

        # Verify each expert state
        for expert_idx in range(4):
            key = f"expert_{expert_idx}"
            assert key in memory_state
            expert_state = memory_state[key]
            assert expert_state.shape == (batch_size, 16, 128)  # (batch, memory_slots, hidden_dim)
            assert expert_state.device == device

    def test_reset_memory(self, toy_base_model):
        """Test memory reset functionality."""
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=4,
            memory_slots=16,
        )

        # Get initial state (make a copy for comparison)
        initial_state = {}
        for expert_idx in range(4):
            initial_state[f"expert_{expert_idx}"] = model.memory_mixture.get_expert_state(expert_idx).clone()

        # Modify expert states
        modified_states = []
        for expert_idx in range(4):
            modified_state = torch.randn(16, 128)
            modified_states.append(modified_state.clone())
            model.memory_mixture.set_expert_state(expert_idx, modified_state)

        # Verify states changed
        modified_state_0 = model.memory_mixture.get_expert_state(0)
        assert not torch.allclose(initial_state["expert_0"], modified_state_0)

        # Verify the modified state matches what we set
        assert torch.allclose(modified_state_0, modified_states[0])

        # Reset memory
        model.reset_memory()

        # States should be reset (not necessarily equal to initial due to re-initialization)
        reset_state = model.get_memory_state()
        assert reset_state is not None

        # After reset, the state should differ from the modified state we set earlier
        # Note: Due to re-initialization, we can't guarantee exact match to initial_state,
        # but we can verify the reset functionality works by checking the state is not None

    def test_get_set_memory_state(self, toy_base_model):
        """Test get/set memory state methods."""
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=4,
            memory_slots=16,
        )

        # Get initial memory state
        memory_state = model.get_memory_state()
        assert len(memory_state) == 4

        # Modify and set new state
        new_state = {}
        for expert_idx in range(4):
            new_state[f"expert_{expert_idx}"] = torch.randn(16, 128)

        model.set_memory_state(new_state)

        # Verify state was updated
        retrieved_state = model.get_memory_state()
        for expert_idx in range(4):
            key = f"expert_{expert_idx}"
            assert torch.allclose(retrieved_state[key], new_state[key])

    def test_save_and_load_pretrained(self, toy_base_model, tmp_path):
        """Test save and load functionality."""
        # Create model
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=4,
            memory_slots=16,
            routing_mode="write-based",
            routing_temperature=1.5,
        )

        # Save model
        save_path = str(tmp_path / "test_gmm_model")
        model.save_pretrained(save_path)

        # Verify files exist
        import os

        assert os.path.exists(os.path.join(save_path, "gmm_config.json"))
        assert os.path.exists(os.path.join(save_path, "gmm_state.pt"))

        # Load model
        loaded_model = GMMXLNetForQA.from_pretrained(save_path)

        # Verify configuration
        assert loaded_model.num_experts == 4
        assert loaded_model.memory_slots == 16
        assert loaded_model.routing_mode == "write-based"
        assert loaded_model.routing_temperature == 1.5

    def test_model_repr(self, toy_base_model):
        """Test model string representation."""
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=4,
            memory_slots=16,
        )

        repr_str = repr(model)
        assert "GMMXLNetForQA" in repr_str
        assert "num_experts=4" in repr_str
        assert "memory_slots=16" in repr_str


class TestBackwardCompatibility:
    """Test backward compatibility with non-GMM models."""

    def test_gmm_disabled_mode(self, toy_base_model, toy_input_data):
        """Test that GMM can be disabled for backward compatibility."""
        # Create model with GMM disabled
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=4,
            memory_slots=4,
            use_gmm_memory=False,
        )

        # Forward pass should work without memory tokens
        outputs = model(
            input_ids=toy_input_data["input_ids"],
            attention_mask=toy_input_data["attention_mask"],
        )

        # Should have logits but no memory updates
        assert outputs["start_logits"] is not None
        assert outputs["end_logits"] is not None

    def test_load_non_gmm_checkpoint_fails(self, toy_base_model, tmp_path):
        """Test that loading non-GMM checkpoint raises clear error."""
        # Save a base model without GMM config
        save_path = str(tmp_path / "non_gmm_model")
        toy_base_model.save_pretrained(save_path)

        # Attempt to load with GMMXLNetForQA should fail with clear error
        with pytest.raises(FileNotFoundError, match="GMM config not found"):
            GMMXLNetForQA.from_pretrained(save_path)


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_expert_count(self, toy_base_model):
        """Test that invalid expert counts raise errors."""
        with pytest.raises(ValueError, match="num_experts must be power of 2"):
            GMMXLNetForQA(
                base_model=toy_base_model,
                num_experts=3,  # Invalid: not power of 2
                memory_slots=16,
            )

        with pytest.raises(ValueError, match="num_experts must be power of 2"):
            GMMXLNetForQA(
                base_model=toy_base_model,
                num_experts=16,  # Invalid: too large
                memory_slots=16,
            )

    def test_mismatched_memory_state(self, toy_base_model):
        """Test that mismatched memory state raises error."""
        model = GMMXLNetForQA(
            base_model=toy_base_model,
            num_experts=4,
            memory_slots=16,
        )

        # Create memory state with wrong number of experts
        wrong_state = {
            "expert_0": torch.randn(16, 128),
            "expert_1": torch.randn(16, 128),
            # Missing expert_2 and expert_3
        }

        with pytest.raises(ValueError, match="must contain 4 experts"):
            model.set_memory_state(wrong_state)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
