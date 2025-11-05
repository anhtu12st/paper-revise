"""
Integration tests for GMMTrainingConfig.

Tests configuration creation, validation, serialization, and integration
with GMMXLNetForQA training workflow.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from transformers import XLNetConfig, XLNetForQuestionAnsweringSimple

from gmmxlnet.models.gmm_xlnet_qa import GMMXLNetForQA
from gmmxlnet.training.config import (
    GMMTrainingConfig,
    gmm_balanced_config,
    gmm_large_config,
    gmm_small_config,
)
from memxlnet.training import TrainingConfig


class TestGMMTrainingConfigPresets:
    """Test preset configuration factory functions."""

    def test_gmm_small_config(self):
        """Test small configuration preset (k=2)."""
        config = gmm_small_config()

        assert config.use_gmm_memory is True
        assert config.num_memory_experts == 2
        assert config.routing_temperature == 1.0
        assert config.routing_mode == "write-based"
        assert config.load_balance_weight == 0.01

    def test_gmm_balanced_config(self):
        """Test balanced configuration preset (k=4)."""
        config = gmm_balanced_config()

        assert config.use_gmm_memory is True
        assert config.num_memory_experts == 4
        assert config.routing_temperature == 1.0

    def test_gmm_large_config(self):
        """Test large configuration preset (k=8)."""
        config = gmm_large_config()

        assert config.use_gmm_memory is True
        assert config.num_memory_experts == 8
        assert config.routing_temperature == 1.0

    def test_preset_with_overrides(self):
        """Test preset configurations can be overridden."""
        config = gmm_small_config(
            routing_temperature=2.0,
            entropy_regularization_weight=0.1,
            num_epochs=5,
        )

        assert config.num_memory_experts == 2  # Preset value
        assert config.routing_temperature == 2.0  # Overridden
        assert config.entropy_regularization_weight == 0.1  # Overridden
        assert config.num_epochs == 5  # Overridden


class TestGMMTrainingConfigValidation:
    """Test configuration parameter validation."""

    def test_valid_num_memory_experts(self):
        """Test valid expert counts are accepted."""
        for k in [2, 3, 4, 5, 6, 7, 8]:
            config = GMMTrainingConfig(use_gmm_memory=True, num_memory_experts=k)
            assert config.num_memory_experts == k

    def test_invalid_num_memory_experts(self):
        """Test invalid expert counts are rejected."""
        for invalid_k in [0, 1, 9, 16, -1]:
            with pytest.raises(ValueError, match="num_memory_experts must be"):
                GMMTrainingConfig(use_gmm_memory=True, num_memory_experts=invalid_k)

    def test_invalid_routing_temperature(self):
        """Test invalid temperatures are rejected."""
        for invalid_temp in [0, -1, -0.5]:
            with pytest.raises(ValueError, match="routing_temperature must be > 0"):
                GMMTrainingConfig(use_gmm_memory=True, routing_temperature=invalid_temp)

    def test_invalid_routing_mode(self):
        """Test invalid routing modes are rejected."""
        with pytest.raises(ValueError, match="routing_mode must be one of"):
            GMMTrainingConfig(use_gmm_memory=True, routing_mode="invalid-mode")

    def test_invalid_weights(self):
        """Test negative weights are rejected."""
        with pytest.raises(ValueError, match="load_balance_weight must be >= 0"):
            GMMTrainingConfig(use_gmm_memory=True, load_balance_weight=-0.1)

        with pytest.raises(ValueError, match="entropy_regularization_weight must be >= 0"):
            GMMTrainingConfig(use_gmm_memory=True, entropy_regularization_weight=-0.1)

    def test_expert_init_strategies_length_mismatch(self):
        """Test expert_init_strategies length must match num_memory_experts."""
        with pytest.raises(ValueError, match="expert_init_strategies length must match"):
            GMMTrainingConfig(
                use_gmm_memory=True,
                num_memory_experts=4,
                expert_init_strategies=["learned", "learned"],  # Only 2, needs 4
            )

    def test_invalid_expert_init_strategy(self):
        """Test invalid initialization strategies are rejected."""
        with pytest.raises(ValueError, match="must be one of"):
            GMMTrainingConfig(
                use_gmm_memory=True,
                num_memory_experts=2,
                expert_init_strategies=["learned", "invalid"],
            )

    def test_validation_skipped_when_gmm_disabled(self):
        """Test validation is skipped when use_gmm_memory=False."""
        # This should not raise error even with invalid GMM params
        config = GMMTrainingConfig(
            use_gmm_memory=False,
            num_memory_experts=100,  # Would be invalid if GMM enabled
        )
        assert config.use_gmm_memory is False

    def test_memory_num_tokens_zero_rejected(self):
        """Test memory_num_tokens=0 is rejected when GMM enabled."""
        with pytest.raises(ValueError, match="GMM memory requires memory_num_tokens > 0"):
            GMMTrainingConfig(
                use_gmm_memory=True,
                memory_num_tokens=0,
            )

    def test_memory_num_tokens_negative_rejected(self):
        """Test negative memory_num_tokens is rejected when GMM enabled."""
        with pytest.raises(ValueError, match="GMM memory requires memory_num_tokens > 0"):
            GMMTrainingConfig(
                use_gmm_memory=True,
                memory_num_tokens=-1,
            )

    def test_memory_num_tokens_too_large_rejected(self):
        """Test memory_num_tokens > 32 is rejected when GMM enabled."""
        with pytest.raises(ValueError, match="memory_num_tokens should be <= 32"):
            GMMTrainingConfig(
                use_gmm_memory=True,
                memory_num_tokens=33,
            )

    def test_memory_num_tokens_valid_accepted(self):
        """Test valid memory_num_tokens values are accepted when GMM enabled."""
        for valid_tokens in [1, 8, 16, 32]:
            config = GMMTrainingConfig(
                use_gmm_memory=True,
                memory_num_tokens=valid_tokens,
            )
            assert config.memory_num_tokens == valid_tokens

    def test_memory_num_tokens_bypass_when_gmm_disabled(self):
        """Test memory_num_tokens validation bypassed when GMM disabled."""
        # These should not raise errors when GMM is disabled
        config1 = GMMTrainingConfig(
            use_gmm_memory=False,
            memory_num_tokens=0,
        )
        assert config1.memory_num_tokens == 0

        config2 = GMMTrainingConfig(
            use_gmm_memory=False,
            memory_num_tokens=100,  # Would be invalid if GMM enabled
        )
        assert config2.memory_num_tokens == 100

    

class TestGMMTrainingConfigSerialization:
    """Test configuration serialization and deserialization."""

    def test_to_dict(self):
        """Test configuration can be serialized to dict."""
        config = gmm_balanced_config(num_epochs=5)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["use_gmm_memory"] is True
        assert config_dict["num_memory_experts"] == 4
        assert config_dict["num_epochs"] == 5
        assert config_dict["memory_type"] == "gmm"  # Metadata added

    def test_from_dict(self):
        """Test configuration can be deserialized from dict."""
        config_dict = {
            "use_gmm_memory": True,
            "num_memory_experts": 4,
            "routing_temperature": 1.5,
            "routing_mode": "read-based",
            "num_epochs": 3,
            "memory_type": "gmm",  # Should be filtered out
        }
        config = GMMTrainingConfig.from_dict(config_dict)

        assert config.use_gmm_memory is True
        assert config.num_memory_experts == 4
        assert config.routing_temperature == 1.5
        assert config.routing_mode == "read-based"

    def test_to_dict_from_dict_round_trip(self):
        """Test serialization round-trip preserves configuration."""
        original = gmm_balanced_config(routing_temperature=2.0, entropy_regularization_weight=0.1, num_epochs=5)
        config_dict = original.to_dict()
        restored = GMMTrainingConfig.from_dict(config_dict)

        assert restored.use_gmm_memory == original.use_gmm_memory
        assert restored.num_memory_experts == original.num_memory_experts
        assert restored.routing_temperature == original.routing_temperature
        assert restored.entropy_regularization_weight == original.entropy_regularization_weight
        assert restored.num_epochs == original.num_epochs

    def test_to_json_from_json(self):
        """Test JSON serialization and deserialization."""
        config = gmm_balanced_config(num_epochs=5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save to JSON
            config.to_json(temp_path)

            # Load from JSON
            loaded = GMMTrainingConfig.from_json(temp_path)

            assert loaded.use_gmm_memory == config.use_gmm_memory
            assert loaded.num_memory_experts == config.num_memory_experts
            assert loaded.num_epochs == config.num_epochs

            # Verify JSON format
            with open(temp_path) as f:
                json_data = json.load(f)
                assert json_data["memory_type"] == "gmm"
        finally:
            os.unlink(temp_path)


class TestGMMTrainingConfigBackwardCompatibility:
    """Test backward compatibility with base TrainingConfig."""

    def test_inherits_from_training_config(self):
        """Test GMMTrainingConfig inherits from TrainingConfig."""
        assert issubclass(GMMTrainingConfig, TrainingConfig)

    def test_has_all_base_config_fields(self):
        """Test GMM config has all base TrainingConfig fields."""
        TrainingConfig()
        gmm_config = GMMTrainingConfig()

        # Check a sample of base fields
        assert hasattr(gmm_config, "model_name")
        assert hasattr(gmm_config, "max_seq_length")
        assert hasattr(gmm_config, "num_epochs")
        assert hasattr(gmm_config, "learning_rate")
        assert hasattr(gmm_config, "warmup_freeze_base_epochs")

    def test_base_config_still_works(self):
        """Test existing TrainingConfig still works (backward compatibility)."""
        config = TrainingConfig(
            num_epochs=5,
            learning_rate=1e-4,
        )
        assert config.num_epochs == 5
        assert config.learning_rate == 1e-4


class TestGMMTrainingConfigWarmupCompatibility:
    """Test compatibility with warmup strategies."""

    def test_warmup_freeze_base_epochs(self):
        """Test GMM config works with warmup_freeze_base_epochs."""
        config = gmm_balanced_config(warmup_freeze_base_epochs=2)

        assert config.use_gmm_memory is True
        assert config.warmup_freeze_base_epochs == 2

    def test_warmup_disable_global_softmax_epochs(self):
        """Test GMM config works with warmup_disable_global_softmax_epochs."""
        config = gmm_balanced_config(warmup_disable_global_softmax_epochs=2)

        assert config.use_gmm_memory is True
        assert config.warmup_disable_global_softmax_epochs == 2

    def test_all_warmup_parameters_combined(self):
        """Test GMM config works with all warmup parameters."""
        config = gmm_balanced_config(
            warmup_freeze_base_epochs=1,
            warmup_disable_global_softmax_epochs=1,
            warmup_disable_any_positive_epochs=1,
        )

        assert config.use_gmm_memory is True
        assert config.warmup_freeze_base_epochs == 1
        assert config.warmup_disable_global_softmax_epochs == 1
        assert config.warmup_disable_any_positive_epochs == 1


class TestGMMTrainingIntegration:
    """Test integration with GMMXLNetForQA training workflow."""

    @pytest.fixture
    def toy_model(self):
        """Create a minimal GMMXLNetForQA model for testing."""
        # Create tiny base model
        xlnet_config = XLNetConfig(
            vocab_size=1000,
            d_model=64,  # Very small for fast testing
            n_layer=1,
            n_head=2,
            d_inner=128,
        )
        base_model = XLNetForQuestionAnsweringSimple(xlnet_config)

        # Create GMM model with k=2 experts (using correct parameter names)
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=2,  # Correct parameter name
            memory_slots=4,  # Correct parameter name
            routing_temperature=1.0,
            routing_mode="write-based",
        )
        return model

    @pytest.fixture
    def toy_data(self):
        """Create minimal toy dataset."""
        batch_size = 2
        seq_len = 32

        # Create random input data
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Random answer positions
        start_positions = torch.tensor([15, 16])
        end_positions = torch.tensor([17, 18])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }

    def test_gmm_config_with_model_initialization(self, toy_model):
        """Test GMMTrainingConfig can be used with GMMXLNetForQA."""
        config = gmm_small_config()

        # Verify config parameters match what model expects
        assert config.use_gmm_memory is True
        assert config.num_memory_experts == 2
        assert toy_model.num_experts == 2

    def test_minimal_training_step(self, toy_model, toy_data):
        """Test minimal training step completes without errors."""
        config = gmm_small_config(learning_rate=1e-3)

        # Setup optimizer
        optimizer = AdamW(toy_model.parameters(), lr=config.learning_rate)

        # Forward pass
        outputs = toy_model(**toy_data)

        # Handle both dict and object returns
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        assert loss is not None
        initial_loss = loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Second forward pass
        outputs2 = toy_model(**toy_data)
        loss2 = outputs2["loss"] if isinstance(outputs2, dict) else outputs2.loss
        second_loss = loss2.item()

        # Loss should change after optimization step
        assert initial_loss != second_loss

    def test_training_multiple_steps_decreases_loss(self, toy_model, toy_data):
        """Test that training for multiple steps decreases loss."""
        config = gmm_small_config(learning_rate=1e-2)  # Higher LR for faster convergence
        optimizer = AdamW(toy_model.parameters(), lr=config.learning_rate)

        # Record initial loss
        with torch.no_grad():
            initial_outputs = toy_model(**toy_data)
            initial_loss_tensor = initial_outputs["loss"] if isinstance(initial_outputs, dict) else initial_outputs.loss
            initial_loss = initial_loss_tensor.item()

        # Train for 10 steps
        for _ in range(10):
            outputs = toy_model(**toy_data)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Check final loss
        with torch.no_grad():
            final_outputs = toy_model(**toy_data)
            final_loss_tensor = final_outputs["loss"] if isinstance(final_outputs, dict) else final_outputs.loss
            final_loss = final_loss_tensor.item()

        # Loss should decrease (allowing some tolerance for randomness)
        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"

    def test_routing_statistics_valid(self, toy_model, toy_data):
        """Test that routing statistics are valid during training."""
        gmm_small_config()

        # Forward pass
        outputs = toy_model(**toy_data)

        # Check routing probabilities
        if hasattr(outputs, "routing_probs") and outputs.routing_probs is not None:
            routing_probs = outputs.routing_probs
            # Should sum to 1 across experts
            assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(routing_probs.shape[:-1]), atol=1e-5)
            # Should be non-negative
            assert (routing_probs >= 0).all()

    def test_config_saved_with_checkpoint(self, toy_model):
        """Test configuration can be saved alongside model checkpoint."""
        config = gmm_balanced_config(num_epochs=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save config
            config_path = Path(tmpdir) / "config.json"
            config.to_json(str(config_path))

            # Save model
            model_path = Path(tmpdir) / "model.pt"
            torch.save(toy_model.state_dict(), model_path)

            # Verify config can be loaded
            loaded_config = GMMTrainingConfig.from_json(str(config_path))
            assert loaded_config.use_gmm_memory == config.use_gmm_memory
            assert loaded_config.num_memory_experts == config.num_memory_experts

            # Verify config.json contains GMM metadata
            with open(config_path) as f:
                config_json = json.load(f)
                assert config_json["memory_type"] == "gmm"
