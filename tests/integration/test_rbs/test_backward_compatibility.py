"""
Integration tests for RBS-XLNet backward compatibility.

Tests ensure that RBS-XLNet can load and work with existing GMM and base MemXLNet models.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from rbsqa.models.rbs_xlnet import RBSXLNetForQA


class TestGMMModelCompatibility:
    """Test compatibility with existing GMM models."""

    @pytest.fixture
    def mock_gmm_checkpoint(self):
        """Create a mock GMM checkpoint structure."""
        checkpoint = {
            "gmm_config.json": {
                "model_class": "GMMXLNetForQA",
                "memory_type": "gmm",
                "version": "1.0",
                "num_experts": 4,
                "memory_slots": 16,
                "routing_mode": "write-based",
                "routing_temperature": 1.0,
                "use_gmm_memory": True
            },
            "gmm_state.pt": {
                "memory_mixture": {},
                "gating_network": {},
                "expert_updater": {},
                "memory_reader": {}
            }
        }
        return checkpoint

    def test_load_gmm_model_as_rbs(self, mock_gmm_checkpoint):
        """Test loading GMM model with RBS interface."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock GMM checkpoint
            config_path = os.path.join(temp_dir, "gmm_config.json")
            state_path = os.path.join(temp_dir, "gmm_state.pt")

            with open(config_path, "w") as f:
                json.dump(mock_gmm_checkpoint["gmm_config.json"], f)

            torch.save(mock_gmm_checkpoint["gmm_state.pt"], state_path)

            # Mock GMMXLNetForQA.from_pretrained
            with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
                mock_backbone = MagicMock()
                mock_backbone.hidden_dim = 768
                mock_gmm_class.from_pretrained.return_value = mock_backbone

                # Load as RBS model
                model = RBSXLNetForQA.from_pretrained(temp_dir)

                # Should have extracted GMM configuration correctly
                assert model.config.num_memory_experts == 4
                assert model.config.memory_num_tokens == 16

                # Should have RBS components (default enabled)
                assert model.belief_tracker is not None
                assert model.halting_policy is not None

    def test_gmm_legacy_mode_forward(self, mock_gmm_checkpoint):
        """Test forward pass compatibility with GMM legacy mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock checkpoint
            config_path = os.path.join(temp_dir, "gmm_config.json")
            state_path = os.path.join(temp_dir, "gmm_state.pt")

            with open(config_path, "w") as f:
                json.dump(mock_gmm_checkpoint["gmm_config.json"], f)

            torch.save(mock_gmm_checkpoint["gmm_state.pt"], state_path)

            # Mock GMM backbone with realistic output
            mock_backbone = MagicMock()

            def mock_forward(*args, **kwargs):
                batch_size, seq_len = 2, 50
                return {
                    "start_logits": torch.randn(batch_size, seq_len),
                    "end_logits": torch.randn(batch_size, seq_len),
                    "new_memory_state": {
                        f"expert_{i}": torch.randn(batch_size, 16, 768)
                        for i in range(4)
                    },
                    "routing_info": {
                        "routing_probs": torch.softmax(torch.randn(batch_size, 4), dim=-1)
                    }
                }

            mock_backbone.forward = mock_forward
            mock_backbone.get_initial_memory = lambda batch_size, device: {
                f"expert_{i}": torch.randn(batch_size, 16, 768, device=device)
                for i in range(4)
            }
            mock_backbone.hidden_dim = 768

            with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
                mock_gmm_class.from_pretrained.return_value = mock_backbone

                model = RBSXLNetForQA.from_pretrained(temp_dir)

                # Test forward pass without segment info (legacy mode)
                batch_size, seq_len = 2, 50
                input_ids = torch.randint(1, 1000, (batch_size, seq_len))
                attention_mask = torch.ones(batch_size, seq_len)

                output = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

                # Should work without segment info
                assert output.start_logits.shape == (batch_size, seq_len)
                assert output.end_logits.shape == (batch_size, seq_len)
                assert output.segment_info is None
                assert output.belief_state is None  # No segment info, no belief tracking

    def test_gmm_to_rbs_mode_upgrade(self, mock_gmm_checkpoint):
        """Test upgrading GMM model to RBS mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock checkpoint
            config_path = os.path.join(temp_dir, "gmm_config.json")
            state_path = os.path.join(temp_dir, "gmm_state.pt")

            with open(config_path, "w") as f:
                json.dump(mock_gmm_checkpoint["gmm_config.json"], f)

            torch.save(mock_gmm_checkpoint["gmm_state.pt"], state_path)

            mock_backbone = MagicMock()
            mock_backbone.hidden_dim = 768
            mock_backbone.forward = MagicMock(return_value={
                "start_logits": torch.randn(2, 50),
                "end_logits": torch.randn(2, 50),
                "new_memory_state": {f"expert_{i}": torch.randn(2, 16, 768) for i in range(4)},
                "routing_info": {"routing_probs": torch.softmax(torch.randn(2, 4), dim=-1)}
            })
            mock_backbone.get_initial_memory = lambda batch_size, device: {
                f"expert_{i}": torch.randn(batch_size, 16, 768, device=device) for i in range(4)
            }

            with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
                mock_gmm_class.from_pretrained.return_value = mock_backbone

                # Load model
                model = RBSXLNetForQA.from_pretrained(temp_dir)

                # Initially loaded with RBS enabled
                assert model.config.use_rbs_mode is True

                # Test forward pass with RBS features
                batch_size, seq_len = 2, 50
                input_ids = torch.randint(1, 1000, (batch_size, seq_len))
                attention_mask = torch.ones(batch_size, seq_len)

                segment_info = {'segment_id': 0, 'global_offset': 0, 'total_segments': 3}

                output = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_info=segment_info,
                    return_dict=True
                )

                # Should have RBS-specific outputs
                assert output.belief_state is not None
                assert output.segment_info == segment_info


class TestBaseModelCompatibility:
    """Test compatibility with base MemXLNet models (no GMM)."""

    @pytest.fixture
    def mock_base_checkpoint(self):
        """Create a mock base model checkpoint structure."""
        checkpoint = {
            "config.json": {
                "model_type": "xlnet",
                "hidden_size": 768,
                "num_hidden_layers": 12
            }
        }
        return checkpoint

    def test_load_base_model_fallback(self, mock_base_checkpoint):
        """Test loading base model with graceful fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock base checkpoint
            config_path = os.path.join(temp_dir, "config.json")

            with open(config_path, "w") as f:
                json.dump(mock_base_checkpoint["config.json"], f)

            # Mock the imports and model loading
            with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
                mock_backbone = MagicMock()
                mock_backbone.hidden_dim = 768
                mock_gmm_class.from_pretrained.return_value = mock_backbone

                # Should load with default RBS settings
                model = RBSXLNetForQA.from_pretrained(temp_dir)

                # Should use default configuration
                assert model.config.num_memory_experts == 4
                assert model.config.memory_num_tokens == 16

    def test_no_checkpoint_error(self):
        """Test error handling when no valid checkpoint found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory

            with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
                mock_backbone = MagicMock()
                mock_backbone.hidden_dim = 768
                mock_gmm_class.from_pretrained.return_value = mock_backbone

                # Should still work with default config
                model = RBSXLNetForQA.from_pretrained(temp_dir)

                assert model.config.use_rbs_mode is True  # Default enabled
                assert model.belief_tracker is not None
                assert model.halting_policy is not None


class TestConfigurationCompatibility:
    """Test configuration compatibility across model versions."""

    def test_mixed_config_values(self):
        """Test handling of mixed/missing configuration values."""
        with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
            mock_backbone = MagicMock()
            mock_backbone.hidden_dim = 768
            mock_gmm_class.from_pretrained.return_value = mock_backbone

            # Create model with partial config
            model = RBSXLNetForQA(
                base_model_name="test-model",
                memory_num_tokens=8,  # Different from default
                # num_memory_experts should use default
                use_rbs_mode=True
            )

            assert model.config.memory_num_tokens == 8
            assert model.config.num_memory_experts == 4  # Default

    def test_config_validation_on_load(self):
        """Test configuration validation during model loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid RBS config
            invalid_config = {
                "base_model_name": "test-model",
                "memory_num_tokens": -1,  # Invalid
                "num_memory_experts": 4,
                "use_rbs_mode": True,
                "belief_confidence_threshold": 1.5  # Invalid
            }

            config_path = os.path.join(temp_dir, "rbs_config.json")
            with open(config_path, "w") as f:
                json.dump(invalid_config, f)

            with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
                mock_backbone = MagicMock()
                mock_backbone.hidden_dim = 768
                mock_gmm_class.from_pretrained.return_value = mock_backbone

                # Should raise validation error
                with pytest.raises(ValueError):
                    RBSXLNetForQA.from_pretrained(temp_dir)


class TestModelUpgrading:
    """Test upgrading models from different versions."""

    def test_gmm_to_rbs_conversion(self):
        """Test converting GMM model to RBS with new capabilities."""
        # Simulate loading a GMM model and adding RBS capabilities
        with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
            mock_backbone = MagicMock()
            mock_backbone.hidden_dim = 768
            mock_backbone.forward = MagicMock(return_value={
                "start_logits": torch.randn(2, 50),
                "end_logits": torch.randn(2, 50),
                "new_memory_state": {f"expert_{i}": torch.randn(2, 16, 768) for i in range(4)},
                "routing_info": {"routing_probs": torch.softmax(torch.randn(2, 4), dim=-1)}
            })
            mock_backbone.get_initial_memory = lambda batch_size, device: {
                f"expert_{i}": torch.randn(batch_size, 16, 768, device=device) for i in range(4)
            }

            mock_gmm_class.from_pretrained.return_value = mock_backbone

            # Create model in legacy mode initially
            model = RBSXLNetForQA(
                base_model_name="test-model",
                use_rbs_mode=False
            )

            # Verify legacy mode
            assert model.config.use_rbs_mode is False
            assert model.belief_tracker is None
            assert model.halting_policy is None

            # Test legacy forward pass
            batch_size, seq_len = 2, 50
            input_ids = torch.randint(1, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)

            output = model.forward(input_ids, attention_mask, return_dict=True)
            assert output.belief_state is None

            # The model can be used as-is or a new instance can be created with RBS enabled

    def test_memory_state_compatibility(self):
        """Test memory state compatibility across model versions."""
        with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
            mock_backbone = MagicMock()
            mock_backbone.hidden_dim = 768

            def mock_get_memory():
                return {
                    f"expert_{i}": torch.randn(16, 768)
                    for i in range(4)
                }

            def mock_set_memory(memory_state):
                # Should not raise errors with valid memory state
                assert len(memory_state) == 4
                for key, value in memory_state.items():
                    assert key.startswith("expert_")
                    assert isinstance(value, torch.Tensor)

            mock_backbone.get_memory_state = mock_get_memory
            mock_backbone.set_memory_state = mock_set_memory
            mock_gmm_class.from_pretrained.return_value = mock_backbone

            # Test with RBS model
            model = RBSXLNetForQA(
                base_model_name="test-model",
                use_rbs_mode=True
            )

            # Test getting memory state
            memory_state = model.get_memory_state()
            assert len(memory_state) == 4

            # Test setting memory state
            new_memory_state = {
                f"expert_{i}": torch.randn(16, 768)
                for i in range(4)
            }
            model.set_memory_state(new_memory_state)  # Should not raise

    def test_model_export_compatibility(self):
        """Test that model export format is compatible."""
        with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
            mock_backbone = MagicMock()
            mock_backbone.hidden_dim = 768
            mock_backbone.save_pretrained = MagicMock()
            mock_gmm_class.from_pretrained.return_value = mock_backbone

            model = RBSXLNetForQA(
                base_model_name="test-model",
                memory_num_tokens=8,
                num_memory_experts=2
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                model.save_pretrained(temp_dir)

                # Verify that all expected files exist
                assert os.path.exists(os.path.join(temp_dir, "rbs_config.json"))
                assert os.path.exists(os.path.join(temp_dir, "gmm_backbone"))
                assert os.path.exists(os.path.join(temp_dir, "belief_tracker.pt"))
                assert os.path.exists(os.path.join(temp_dir, "halting_policy.pt"))

                # Verify config contains all expected fields
                with open(os.path.join(temp_dir, "rbs_config.json"), "r") as f:
                    config = json.load(f)

                assert config['memory_num_tokens'] == 8
                assert config['num_memory_experts'] == 2
                assert config['base_model_name'] == "test-model"
                assert config['use_rbs_mode'] is True


class TestInferenceCompatibility:
    """Test inference compatibility across model modes."""

    def test_adaptive_inference_fallback(self):
        """Test adaptive inference fallback to full mode."""
        with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
            mock_backbone = MagicMock()
            mock_backbone.hidden_dim = 768

            def mock_forward(*args, **kwargs):
                batch_size = args[0].size(0) if args else 2
                seq_len = args[0].size(-1) if args else 50
                return {
                    "start_logits": torch.randn(batch_size, seq_len),
                    "end_logits": torch.randn(batch_size, seq_len),
                    "memory_state": {f"expert_{i}": torch.randn(batch_size, 16, 768) for i in range(4)}
                }

            mock_backbone.forward = mock_forward
            mock_gmm_class.from_pretrained.return_value = mock_backbone

            # Test with RBS disabled (should fallback to full processing)
            model = RBSXLNetForQA(
                base_model_name="test-model",
                use_rbs_mode=False
            )

            question_ids = torch.randint(1, 1000, (1, 10))
            context_segments = [torch.randint(1, 1000, (1, 20)) for _ in range(3)]

            result = model.adaptive_inference(
                question_input_ids=question_ids,
                context_segments=context_segments
            )

            # Should process all segments (no early stopping)
            assert result.segments_processed == 3
            assert result.total_segments == 3
            assert result.efficiency_score == 1.0
            assert len(result.belief_history) == 0  # No belief tracking
            assert len(result.halting_history) == 0  # No halting decisions

    def test_batch_size_compatibility(self):
        """Test compatibility with different batch sizes."""
        with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
            mock_backbone = MagicMock()
            mock_backbone.hidden_dim = 768
            mock_backbone.forward = MagicMock(return_value={
                "start_logits": torch.randn(4, 50),
                "end_logits": torch.randn(4, 50),
                "new_memory_state": {f"expert_{i}": torch.randn(4, 16, 768) for i in range(4)},
                "routing_info": {"routing_probs": torch.softmax(torch.randn(4, 4), dim=-1)}
            })
            mock_backbone.get_initial_memory = lambda batch_size, device: {
                f"expert_{i}": torch.randn(batch_size, 16, 768, device=device) for i in range(4)
            }
            mock_gmm_class.from_pretrained.return_value = mock_backbone

            model = RBSXLNetForQA(base_model_name="test-model")

            # Test different batch sizes
            for batch_size in [1, 2, 4, 8]:
                input_ids = torch.randint(1, 1000, (batch_size, 50))
                attention_mask = torch.ones(batch_size, 50)

                output = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

                assert output.start_logits.shape[0] == batch_size
                assert output.end_logits.shape[0] == batch_size