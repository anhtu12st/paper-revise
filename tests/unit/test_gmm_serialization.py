"""
Unit tests for GMMXLNetForQA serialization and loading.

Tests cover:
- Save/load round-trip with various expert counts
- Backward compatibility with non-GMM checkpoints
- Version detection and error handling
- HuggingFace Hub integration
- Deterministic outputs after loading
- Validation of loaded state
"""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from gmmxlnet.models import GMMXLNetForQA


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoint tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_base_model():
    """Create mock base XLNet model for testing."""
    mock_model = MagicMock()
    mock_model.config.d_model = 768
    mock_model.parameters.return_value = [torch.zeros(1)]

    # Mock save_pretrained
    mock_model.save_pretrained = MagicMock()

    return mock_model


@pytest.fixture
def gmm_model(mock_base_model):
    """Create GMMXLNetForQA model for testing."""
    model = GMMXLNetForQA(
        base_model=mock_base_model,
        num_experts=4,
        memory_slots=16,
        routing_mode="write-based",
        routing_temperature=1.0,
        use_gmm_memory=True,
    )
    return model


@pytest.mark.unit
class TestSavePretrainedBasic:
    """Test suite for basic save_pretrained functionality."""

    def test_save_creates_directory(self, gmm_model, temp_checkpoint_dir):
        """Test that save_pretrained creates the save directory."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        gmm_model.save_pretrained(save_path)

        assert os.path.exists(save_path)
        assert os.path.isdir(save_path)

    def test_save_creates_gmm_config(self, gmm_model, temp_checkpoint_dir):
        """Test that save_pretrained creates gmm_config.json."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        gmm_model.save_pretrained(save_path)

        config_path = os.path.join(save_path, "gmm_config.json")
        assert os.path.exists(config_path)

        # Load and verify config
        with open(config_path) as f:
            config = json.load(f)

        assert config["model_class"] == "GMMXLNetForQA"
        assert config["memory_type"] == "gmm"
        assert config["version"] == "1.0"
        assert config["num_experts"] == 4
        assert config["memory_slots"] == 16
        assert config["routing_mode"] == "write-based"
        assert config["routing_temperature"] == 1.0
        assert config["use_gmm_memory"] is True

    def test_save_creates_gmm_state(self, gmm_model, temp_checkpoint_dir):
        """Test that save_pretrained creates gmm_state.pt."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        gmm_model.save_pretrained(save_path)

        state_path = os.path.join(save_path, "gmm_state.pt")
        assert os.path.exists(state_path)

        # Load and verify state dict structure
        state_dict = torch.load(state_path, weights_only=True)
        assert "memory_mixture" in state_dict
        assert "gating_network" in state_dict
        assert "expert_updater" in state_dict
        assert "memory_reader" in state_dict

    def test_save_calls_base_model_save(self, gmm_model, temp_checkpoint_dir):
        """Test that save_pretrained calls base model's save_pretrained."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        gmm_model.save_pretrained(save_path)

        gmm_model.base.save_pretrained.assert_called_once_with(save_path)


@pytest.mark.unit
class TestSavePretrainedWithDifferentConfigs:
    """Test save_pretrained with various model configurations."""

    @pytest.mark.parametrize("num_experts", [2, 4, 8])
    def test_save_with_different_expert_counts(self, mock_base_model, temp_checkpoint_dir, num_experts):
        """Test save/load with different expert counts."""
        model = GMMXLNetForQA(
            base_model=mock_base_model,
            num_experts=num_experts,
            memory_slots=16,
            routing_mode="write-based",
        )

        save_path = os.path.join(temp_checkpoint_dir, f"model_k{num_experts}")
        model.save_pretrained(save_path)

        # Verify config
        config_path = os.path.join(save_path, "gmm_config.json")
        with open(config_path) as f:
            config = json.load(f)

        assert config["num_experts"] == num_experts

    @pytest.mark.parametrize("routing_mode", ["write-based", "read-based"])
    def test_save_with_different_routing_modes(self, mock_base_model, temp_checkpoint_dir, routing_mode):
        """Test save with different routing modes."""
        model = GMMXLNetForQA(
            base_model=mock_base_model,
            num_experts=4,
            memory_slots=16,
            routing_mode=routing_mode,
        )

        save_path = os.path.join(temp_checkpoint_dir, f"model_{routing_mode}")
        model.save_pretrained(save_path)

        # Verify config
        config_path = os.path.join(save_path, "gmm_config.json")
        with open(config_path) as f:
            config = json.load(f)

        assert config["routing_mode"] == routing_mode

    def test_save_with_gmm_disabled(self, mock_base_model, temp_checkpoint_dir):
        """Test save when GMM memory is disabled."""
        model = GMMXLNetForQA(
            base_model=mock_base_model,
            num_experts=4,
            memory_slots=16,
            use_gmm_memory=False,
        )

        save_path = os.path.join(temp_checkpoint_dir, "model_no_gmm")
        model.save_pretrained(save_path)

        # Config should exist
        config_path = os.path.join(save_path, "gmm_config.json")
        assert os.path.exists(config_path)

        # State should not exist since GMM is disabled
        state_path = os.path.join(save_path, "gmm_state.pt")
        assert not os.path.exists(state_path)


@pytest.mark.unit
class TestSavePretrainedHubIntegration:
    """Test HuggingFace Hub integration in save_pretrained."""

    @patch("huggingface_hub.HfApi")
    def test_push_to_hub_basic(self, mock_hf_api, gmm_model, temp_checkpoint_dir):
        """Test push_to_hub=True uploads to Hub."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")

        # Configure mock
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance

        # Save with push_to_hub
        gmm_model.save_pretrained(save_path, push_to_hub=True, repo_id="test-user/test-model")

        # Verify upload_folder was called
        mock_api_instance.upload_folder.assert_called_once()
        call_kwargs = mock_api_instance.upload_folder.call_args[1]
        assert call_kwargs["folder_path"] == save_path
        assert call_kwargs["repo_id"] == "test-user/test-model"
        assert call_kwargs["repo_type"] == "model"

    def test_push_to_hub_without_repo_id_raises_error(self, gmm_model, temp_checkpoint_dir):
        """Test that push_to_hub=True without repo_id raises ValueError."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")

        with pytest.raises(ValueError, match="repo_id must be provided"):
            gmm_model.save_pretrained(save_path, push_to_hub=True)

    @patch("huggingface_hub.HfApi")
    def test_push_to_hub_with_custom_commit_message(self, mock_hf_api, gmm_model, temp_checkpoint_dir):
        """Test push_to_hub with custom commit message."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")

        # Configure mock
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance

        # Save with custom commit message
        gmm_model.save_pretrained(
            save_path,
            push_to_hub=True,
            repo_id="test-user/test-model",
            commit_message="Custom commit message",
        )

        # Verify commit message was passed
        call_kwargs = mock_api_instance.upload_folder.call_args[1]
        assert call_kwargs["commit_message"] == "Custom commit message"


@pytest.mark.unit
class TestFromPretrainedBasic:
    """Test suite for basic from_pretrained functionality."""

    def test_from_pretrained_loads_config(self, gmm_model, temp_checkpoint_dir):
        """Test that from_pretrained loads GMM configuration."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        gmm_model.save_pretrained(save_path)

        # Mock XLNet base model loading
        with patch("transformers.XLNetForQuestionAnsweringSimple") as mock_xlnet:
            mock_base = MagicMock()
            mock_base.config.d_model = 768
            mock_base.parameters.return_value = iter([torch.zeros(1)])
            mock_xlnet.from_pretrained.return_value = mock_base

            loaded_model = GMMXLNetForQA.from_pretrained(save_path)

            assert loaded_model.num_experts == 4
            assert loaded_model.memory_slots == 16
            assert loaded_model.routing_mode == "write-based"
            assert loaded_model.use_gmm_memory is True

    def test_from_pretrained_loads_state(self, gmm_model, temp_checkpoint_dir):
        """Test that from_pretrained loads GMM state dict."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        gmm_model.save_pretrained(save_path)

        # Mock XLNet base model loading
        with patch("transformers.XLNetForQuestionAnsweringSimple") as mock_xlnet:
            mock_base = MagicMock()
            mock_base.config.d_model = 768
            mock_base.parameters.return_value = iter([torch.zeros(1)])
            mock_xlnet.from_pretrained.return_value = mock_base

            # Load model
            loaded_model = GMMXLNetForQA.from_pretrained(save_path)

            # Verify components exist
            assert loaded_model.memory_mixture is not None
            assert loaded_model.gating_network is not None
            assert loaded_model.expert_updater is not None
            assert loaded_model.memory_reader is not None

    def test_from_pretrained_missing_config_raises_error(self, temp_checkpoint_dir):
        """Test that loading without gmm_config.json raises FileNotFoundError."""
        save_path = os.path.join(temp_checkpoint_dir, "empty_model")
        os.makedirs(save_path)

        # Mock XLNet base model loading to avoid loading errors
        with patch("transformers.XLNetForQuestionAnsweringSimple") as mock_xlnet:
            mock_base = MagicMock()
            mock_base.config.d_model = 768
            mock_xlnet.from_pretrained.return_value = mock_base

            with pytest.raises(FileNotFoundError, match="GMM config not found"):
                GMMXLNetForQA.from_pretrained(save_path)

    def test_from_pretrained_wrong_memory_type_raises_error(self, temp_checkpoint_dir):
        """Test that loading checkpoint with wrong memory_type raises ValueError."""
        save_path = os.path.join(temp_checkpoint_dir, "wrong_type_model")
        os.makedirs(save_path)

        # Create config with wrong memory_type
        config = {
            "model_class": "GMMXLNetForQA",
            "memory_type": "standard",  # Wrong type
            "num_experts": 4,
        }
        config_path = os.path.join(save_path, "gmm_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Mock XLNet base model loading
        with patch("transformers.XLNetForQuestionAnsweringSimple") as mock_xlnet:
            mock_base = MagicMock()
            mock_xlnet.from_pretrained.return_value = mock_base

            with pytest.raises(ValueError, match="memory_type is 'standard', expected 'gmm'"):
                GMMXLNetForQA.from_pretrained(save_path)


@pytest.mark.unit
class TestSaveLoadRoundTrip:
    """Test save/load round-trip with state preservation."""

    @pytest.mark.parametrize("num_experts", [2, 4, 8])
    def test_round_trip_preserves_expert_states(self, mock_base_model, temp_checkpoint_dir, num_experts):
        """Test that save/load round-trip preserves expert states."""
        # Create model
        model = GMMXLNetForQA(
            base_model=mock_base_model,
            num_experts=num_experts,
            memory_slots=16,
            routing_mode="write-based",
        )

        # Get initial expert states
        initial_states = {}
        for i in range(num_experts):
            initial_states[i] = model.memory_mixture.get_expert_state(i).clone()

        # Save and load
        save_path = os.path.join(temp_checkpoint_dir, f"model_k{num_experts}")
        model.save_pretrained(save_path)

        with patch("transformers.XLNetForQuestionAnsweringSimple") as mock_xlnet:
            mock_base = MagicMock()
            mock_base.config.d_model = 768
            mock_base.parameters.return_value = iter([torch.zeros(1)])
            mock_xlnet.from_pretrained.return_value = mock_base

            loaded_model = GMMXLNetForQA.from_pretrained(save_path)

            # Verify expert states are preserved
            for i in range(num_experts):
                loaded_state = loaded_model.memory_mixture.get_expert_state(i)
                assert torch.allclose(initial_states[i], loaded_state, atol=1e-6)

    def test_round_trip_preserves_routing_parameters(self, gmm_model, temp_checkpoint_dir):
        """Test that save/load round-trip preserves routing network parameters."""
        # Get initial routing parameters
        initial_routing_params = {
            k: v.clone() for k, v in gmm_model.gating_network.state_dict().items()
        }

        # Save and load
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        gmm_model.save_pretrained(save_path)

        with patch("transformers.XLNetForQuestionAnsweringSimple") as mock_xlnet:
            mock_base = MagicMock()
            mock_base.config.d_model = 768
            mock_base.parameters.return_value = iter([torch.zeros(1)])
            mock_xlnet.from_pretrained.return_value = mock_base

            loaded_model = GMMXLNetForQA.from_pretrained(save_path)

            # Verify routing parameters are preserved
            loaded_routing_params = loaded_model.gating_network.state_dict()
            for key in initial_routing_params:
                assert torch.allclose(
                    initial_routing_params[key], loaded_routing_params[key], atol=1e-6
                )


@pytest.mark.unit
class TestValidationAfterLoading:
    """Test validation checks after loading model."""

    def test_validation_detects_expert_count_mismatch(self, gmm_model, temp_checkpoint_dir):
        """Test that validation detects expert count mismatch."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        gmm_model.save_pretrained(save_path)

        # Manually corrupt the config to have wrong expert count
        config_path = os.path.join(save_path, "gmm_config.json")
        with open(config_path) as f:
            config = json.load(f)
        config["num_experts"] = 2  # Change from 4 to 2
        with open(config_path, "w") as f:
            json.dump(config, f)

        with patch("transformers.XLNetForQuestionAnsweringSimple") as mock_xlnet:
            mock_base = MagicMock()
            mock_base.config.d_model = 768
            mock_base.parameters.return_value = iter([torch.zeros(1)])
            mock_xlnet.from_pretrained.return_value = mock_base

            # Loading should fail - PyTorch catches state dict mismatch
            with pytest.raises((ValueError, RuntimeError)):
                GMMXLNetForQA.from_pretrained(save_path)

    def test_validation_checks_routing_network_shape(self, gmm_model, temp_checkpoint_dir):
        """Test that validation checks routing network shape."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        gmm_model.save_pretrained(save_path)

        # Load the state and corrupt routing network shape
        state_path = os.path.join(save_path, "gmm_state.pt")
        state_dict = torch.load(state_path, weights_only=True)

        # Replace routing network with wrong shape
        state_dict["gating_network"]["routing_projection.weight"] = torch.randn(2, 768)  # Wrong: 2 instead of 4
        torch.save(state_dict, state_path)

        with patch("transformers.XLNetForQuestionAnsweringSimple") as mock_xlnet:
            mock_base = MagicMock()
            mock_base.config.d_model = 768
            mock_base.parameters.return_value = iter([torch.zeros(1)])
            mock_xlnet.from_pretrained.return_value = mock_base

            # Loading should fail - PyTorch catches shape mismatch
            with pytest.raises((ValueError, RuntimeError)):
                GMMXLNetForQA.from_pretrained(save_path)


@pytest.mark.unit
class TestDeterministicOutputs:
    """Test that loaded models produce deterministic outputs."""

    def test_outputs_match_after_loading(self, mock_base_model, temp_checkpoint_dir):
        """Test that model outputs are identical after save/load."""
        # Create model
        model = GMMXLNetForQA(
            base_model=mock_base_model,
            num_experts=4,
            memory_slots=16,
            routing_mode="write-based",
        )

        # Create dummy input
        batch_size = 2
        seq_len = 10
        device = torch.device("cpu")

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        memory_state = model.get_initial_memory(batch_size, device)

        # Mock the base model forward to return consistent outputs
        mock_outputs = MagicMock()
        mock_outputs.loss = None
        mock_outputs.start_logits = torch.randn(batch_size, seq_len)
        mock_outputs.end_logits = torch.randn(batch_size, seq_len)
        mock_outputs.hidden_states = (torch.randn(batch_size, seq_len, 768),)
        model.base.return_value = mock_outputs

        # Get outputs before save
        with torch.no_grad():
            outputs_before = model(
                input_ids=input_ids,
                memory_state=memory_state,
                mem_read_ids=list(range(16)),
                mem_write_ids=list(range(16)),
            )

        # Save and load
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        model.save_pretrained(save_path)

        with patch("transformers.XLNetForQuestionAnsweringSimple") as mock_xlnet:
            mock_base_loaded = MagicMock()
            mock_base_loaded.config.d_model = 768
            mock_base_loaded.parameters.return_value = iter([torch.zeros(1)])
            mock_base_loaded.return_value = mock_outputs  # Same outputs
            mock_xlnet.from_pretrained.return_value = mock_base_loaded

            loaded_model = GMMXLNetForQA.from_pretrained(save_path)

            # Get outputs after load
            with torch.no_grad():
                outputs_after = loaded_model(
                    input_ids=input_ids,
                    memory_state=memory_state,
                    mem_read_ids=list(range(16)),
                    mem_write_ids=list(range(16)),
                )

        # Verify memory states are identical
        for key in outputs_before["new_memory_state"]:
            assert torch.allclose(
                outputs_before["new_memory_state"][key],
                outputs_after["new_memory_state"][key],
                atol=1e-6,
            ), f"Memory state mismatch for {key}"


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test backward compatibility with non-GMM checkpoints."""

    def test_loading_without_gmm_state_file(self, gmm_model, temp_checkpoint_dir):
        """Test that loading works when gmm_state.pt is missing (backward compatibility)."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        gmm_model.save_pretrained(save_path)

        # Remove gmm_state.pt to simulate old checkpoint
        state_path = os.path.join(save_path, "gmm_state.pt")
        os.remove(state_path)

        with patch("transformers.XLNetForQuestionAnsweringSimple") as mock_xlnet:
            mock_base = MagicMock()
            mock_base.config.d_model = 768
            mock_base.parameters.return_value = iter([torch.zeros(1)])
            mock_xlnet.from_pretrained.return_value = mock_base

            # Should load without error (using default initialization)
            loaded_model = GMMXLNetForQA.from_pretrained(save_path)
            assert loaded_model is not None
            assert loaded_model.use_gmm_memory is True


@pytest.mark.unit
class TestKwargsOverride:
    """Test that kwargs can override config values."""

    def test_kwargs_override_config_values(self, gmm_model, temp_checkpoint_dir):
        """Test that kwargs passed to from_pretrained override config values."""
        save_path = os.path.join(temp_checkpoint_dir, "test_model")
        gmm_model.save_pretrained(save_path)

        with patch("transformers.XLNetForQuestionAnsweringSimple") as mock_xlnet:
            mock_base = MagicMock()
            mock_base.config.d_model = 768
            mock_base.parameters.return_value = iter([torch.zeros(1)])
            mock_xlnet.from_pretrained.return_value = mock_base

            # Override routing_temperature via kwargs
            loaded_model = GMMXLNetForQA.from_pretrained(save_path, routing_temperature=2.0)

            assert loaded_model.routing_temperature == 2.0  # Overridden value
            assert loaded_model.num_experts == 4  # Original value preserved
