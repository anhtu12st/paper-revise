"""
Unit tests for RBS-XLNet model functionality.

Tests cover initialization, forward pass, adaptive inference, and serialization.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from rbsqa.models.rbs_xlnet import RBSXLNetForQA
from rbsqa.models import RBSModelConfig, RBSModelOutput, RBSInferenceResult


class TestRBSModelConfig:
    """Test RBS model configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RBSModelConfig()

        assert config.base_model_name == "xlnet-base-cased"
        assert config.memory_num_tokens == 16
        assert config.num_memory_experts == 4
        assert config.use_rbs_mode is True
        assert config.belief_confidence_threshold == 0.7
        assert config.adaptive_confidence_threshold == 0.8

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should pass
        config = RBSModelConfig()
        config.validate()  # Should not raise

        # Invalid memory tokens
        config = RBSModelConfig(memory_num_tokens=0)
        with pytest.raises(ValueError, match="memory_num_tokens must be positive"):
            config.validate()

        # Invalid experts
        config = RBSModelConfig(num_memory_experts=-1)
        with pytest.raises(ValueError, match="num_memory_experts must be positive"):
            config.validate()

        # Invalid confidence threshold
        config = RBSModelConfig(belief_confidence_threshold=1.5)
        with pytest.raises(ValueError, match="belief_confidence_threshold must be in \\[0.0, 1.0\\]"):
            config.validate()

        # Invalid scoring method
        config = RBSModelConfig(belief_re_scoring_method="invalid_method")
        with pytest.raises(ValueError, match="Unknown belief_re_scoring_method"):
            config.validate()

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = RBSModelConfig(
            memory_num_tokens=8,
            num_memory_experts=2,
            use_rbs_mode=False
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['memory_num_tokens'] == 8
        assert config_dict['num_memory_experts'] == 2
        assert config_dict['use_rbs_mode'] is False


@pytest.fixture
def mock_gmm_backbone():
    """Mock GMM backbone for testing."""
    mock_backbone = MagicMock()

    # Mock forward pass output
    def mock_forward(*args, **kwargs):
        # Get input_ids from kwargs or args
        input_ids = kwargs.get('input_ids', args[0] if args else torch.randn(2, 50))
        if isinstance(input_ids, MagicMock):
            input_ids = torch.randn(2, 50)  # Default size for testing

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(-1)

        return {
            "start_logits": torch.randn(batch_size, seq_len),
            "end_logits": torch.randn(batch_size, seq_len),
            "new_memory_state": {
                f"expert_{i}": torch.randn(batch_size, 16, 768)
                for i in range(4)
            },
            "routing_info": {
                "routing_probs": torch.softmax(torch.randn(batch_size, 4), dim=-1),
                "routing_entropy": torch.randn(batch_size)
            },
            "hidden_states": None,
            "attentions": None
        }

    mock_backbone.forward = mock_forward
    mock_backbone.get_initial_memory = lambda batch_size, device: {
        f"expert_{i}": torch.randn(batch_size, 16, 768, device=device)
        for i in range(4)
    }
    mock_backbone.get_memory_state = lambda: {
        f"expert_{i}": torch.randn(16, 768)
        for i in range(4)
    }
    mock_backbone.set_memory_state = lambda state: None
    mock_backbone.hidden_dim = 768
    mock_backbone.save_pretrained = lambda path: None
    mock_backbone.from_pretrained = lambda path: mock_backbone

    return mock_backbone


@pytest.fixture
def rbs_model(mock_gmm_backbone):
    """Create RBS model for testing."""
    with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
        mock_gmm_class.from_pretrained.return_value = mock_gmm_backbone
        mock_gmm_class.return_value = mock_gmm_backbone

        with patch('transformers.XLNetForQuestionAnsweringSimple.from_pretrained') as mock_xlnet:
            mock_xlnet.return_value = MagicMock()

            model = RBSXLNetForQA(
                base_model_name="test-model",
                memory_num_tokens=16,
                num_memory_experts=4,
                use_rbs_mode=True
            )
            model.gmm_backbone = mock_gmm_backbone

        return model


class TestRBSModelInitialization:
    """Test RBS model initialization."""

    def test_rbs_mode_initialization(self, mock_gmm_backbone):
        """Test initialization with RBS mode enabled."""
        with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
            mock_gmm_class.from_pretrained.return_value = mock_gmm_backbone
            mock_gmm_class.return_value = mock_gmm_backbone

            with patch('transformers.XLNetForQuestionAnsweringSimple.from_pretrained') as mock_xlnet:
                mock_xlnet.return_value = MagicMock()

                model = RBSXLNetForQA(
                    base_model_name="test-model",
                    use_rbs_mode=True
                )

            assert model.config.use_rbs_mode is True
            assert model.belief_tracker is not None
            assert model.halting_policy is not None
            assert model.training_mode == "supervised"
            assert model.inference_mode == "adaptive"

    def test_legacy_mode_initialization(self, mock_gmm_backbone):
        """Test initialization with legacy mode (RBS disabled)."""
        with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
            mock_gmm_class.from_pretrained.return_value = mock_gmm_backbone
            mock_gmm_class.return_value = mock_gmm_backbone

            with patch('transformers.XLNetForQuestionAnsweringSimple.from_pretrained') as mock_xlnet:
                mock_xlnet.return_value = MagicMock()

                model = RBSXLNetForQA(
                    base_model_name="test-model",
                    use_rbs_mode=False
                )

            assert model.config.use_rbs_mode is False
            assert model.belief_tracker is None
            assert model.halting_policy is None

    def test_custom_belief_halting_config(self, mock_gmm_backbone):
        """Test initialization with custom belief and halting configurations."""
        with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
            mock_gmm_class.from_pretrained.return_value = mock_gmm_backbone
            mock_gmm_class.return_value = mock_gmm_backbone

            with patch('transformers.XLNetForQuestionAnsweringSimple.from_pretrained') as mock_xlnet:
                mock_xlnet.return_value = MagicMock()

                belief_config = {
                    'max_segments': 20,
                    'confidence_threshold': 0.8
                }
                halting_config = {
                    'hidden_dim': 128,
                    'temperature': 0.5
                }

                model = RBSXLNetForQA(
                    base_model_name="test-model",
                    belief_state_config=belief_config,
                    halting_config=halting_config
                )

            assert model.belief_tracker.max_segments == 20
            assert model.belief_tracker.confidence_threshold == 0.8
            assert model.halting_policy.temperature == 0.5


class TestRBSForwardPass:
    """Test RBS model forward pass."""

    def test_rbs_mode_forward(self, rbs_model):
        """Test forward pass in RBS mode with segment info."""
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(1, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        segment_info = {
            'segment_id': 0,
            'global_offset': 0,
            'total_segments': 5
        }

        output = rbs_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_info=segment_info,
            return_dict=True
        )

        assert isinstance(output, RBSModelOutput)
        assert output.start_logits.shape == (batch_size, seq_len)
        assert output.end_logits.shape == (batch_size, seq_len)
        assert output.belief_state is not None
        assert output.segment_info == segment_info

    def test_legacy_mode_forward(self, rbs_model):
        """Test forward pass in legacy mode without segment info."""
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(1, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Disable RBS mode
        rbs_model.config.use_rbs_mode = False

        output = rbs_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        assert isinstance(output, RBSModelOutput)
        assert output.start_logits.shape == (batch_size, seq_len)
        assert output.end_logits.shape == (batch_size, seq_len)
        assert output.belief_state is None
        assert output.segment_info is None
        assert output.halting_decision is None

    def test_forward_with_memory_state(self, rbs_model):
        """Test forward pass with provided memory state."""
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(1, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        memory_state = {
            f"expert_{i}": torch.randn(batch_size, 16, 768)
            for i in range(4)
        }

        output = rbs_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_state=memory_state,
            return_dict=True
        )

        assert isinstance(output, RBSModelOutput)
        assert output.memory_state is not None

    def test_forward_return_tuple(self, rbs_model):
        """Test forward pass with return_dict=False."""
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(1, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        output = rbs_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        assert isinstance(output, tuple)
        assert len(output) == 10  # Should match RBSModelOutput.to_tuple()


class TestAdaptiveInference:
    """Test adaptive inference functionality."""

    def test_adaptive_inference_rbs_mode(self, rbs_model):
        """Test adaptive inference in RBS mode."""
        batch_size, question_len = 1, 10
        question_ids = torch.randint(1, 1000, (batch_size, question_len))

        context_segments = [
            torch.randint(1, 1000, (1, 20)) for _ in range(5)
        ]

        rbs_model.set_inference_mode("adaptive")

        result = rbs_model.adaptive_inference(
            question_input_ids=question_ids,
            context_segments=context_segments,
            max_segments=3
        )

        assert isinstance(result, RBSInferenceResult)
        assert result.segments_processed <= 3
        assert result.total_segments == 5
        assert result.efficiency_score >= 1.0
        assert len(result.belief_history) <= 3

    def test_adaptive_inference_full_mode(self, rbs_model):
        """Test adaptive inference in full mode (fallback)."""
        batch_size, question_len = 1, 10
        question_ids = torch.randint(1, 1000, (batch_size, question_len))

        context_segments = [
            torch.randint(1, 1000, (1, 20)) for _ in range(3)
        ]

        rbs_model.set_inference_mode("full")

        result = rbs_model.adaptive_inference(
            question_input_ids=question_ids,
            context_segments=context_segments
        )

        assert isinstance(result, RBSInferenceResult)
        assert result.segments_processed == 3
        assert result.total_segments == 3
        assert result.efficiency_score == 1.0

    def test_adaptive_inference_legacy_mode(self, rbs_model):
        """Test adaptive inference with RBS mode disabled."""
        batch_size, question_len = 1, 10
        question_ids = torch.randint(1, 1000, (batch_size, question_len))

        context_segments = [
            torch.randint(1, 1000, (1, 20)) for _ in range(3)
        ]

        # Disable RBS mode
        rbs_model.config.use_rbs_mode = False

        result = rbs_model.adaptive_inference(
            question_input_ids=question_ids,
            context_segments=context_segments
        )

        assert isinstance(result, RBSInferenceResult)
        assert result.segments_processed == 3
        assert result.efficiency_score == 1.0

    def test_adaptive_inference_early_stopping(self, rbs_model):
        """Test early stopping based on confidence threshold."""
        batch_size, question_len = 1, 10
        question_ids = torch.randint(1, 1000, (batch_size, question_len))

        context_segments = [
            torch.randint(1, 1000, (1, 20)) for _ in range(10)
        ]

        # Set low confidence threshold for early stopping
        rbs_model.config.adaptive_confidence_threshold = 0.5

        # Mock high confidence belief state
        mock_belief = MagicMock()
        mock_belief.confidence = 0.9
        mock_belief.best_span = (5, 10)

        # Patch belief tracker to return high confidence
        original_update = rbs_model.belief_tracker.update_belief
        def mock_update(*args, **kwargs):
            original_result = original_update(*args, **kwargs)
            original_result.confidence = 0.9
            original_result.best_span = (5, 10)
            return original_result

        rbs_model.belief_tracker.update_belief = mock_update

        result = rbs_model.adaptive_inference(
            question_input_ids=question_ids,
            context_segments=context_segments
        )

        # Should stop early due to high confidence
        assert result.segments_processed < 10
        assert result.efficiency_score > 1.0


class TestHaltingFeatures:
    """Test halting feature extraction."""

    def test_extract_halting_features(self, rbs_model):
        """Test extraction of halting features."""
        # Create mock belief state
        mock_belief = MagicMock()
        mock_belief.confidence = 0.7
        mock_belief.confidence_history = [0.3, 0.5, 0.7]
        mock_belief.revision_count = 1

        # Create mock GMM outputs
        gmm_outputs = {
            "routing_info": {
                "routing_probs": torch.tensor([[0.4, 0.3, 0.2, 0.1]]),
                "routing_entropy": torch.tensor([1.2])
            },
            "aggregated_memory": torch.randn(1, 16, 768)
        }

        segment_info = {
            'segment_id': 2,
            'total_segments': 10
        }

        features = rbs_model._extract_halting_features(
            mock_belief, gmm_outputs, segment_info
        )

        assert features.current_confidence == 0.7
        assert features.segments_processed == 3
        assert features.segments_remaining == 8
        assert features.revision_count == 1
        assert len(features.confidence_trend) == 3
        assert abs(features.routing_entropy - 1.2) < 0.1

    def test_compute_context_quality(self, rbs_model):
        """Test context quality computation."""
        # Low variance memory should have high quality
        high_quality_memory = torch.ones(1, 16, 768) * 0.5
        quality_score = rbs_model._compute_context_quality(high_quality_memory)
        assert quality_score > 0.5

        # High variance memory should have low quality
        low_quality_memory = torch.randn(1, 16, 768) * 10
        quality_score = rbs_model._compute_context_quality(low_quality_memory)
        assert quality_score < 0.5

    def test_compute_segment_relevance(self, rbs_model):
        """Test segment relevance computation."""
        # High routing probability indicates high relevance
        gmm_outputs = {
            "routing_info": {
                "routing_probs": torch.tensor([[0.8, 0.1, 0.05, 0.05]])
            }
        }
        relevance = rbs_model._compute_segment_relevance(gmm_outputs)
        assert abs(relevance - 0.8) < 0.01

        # Missing routing info should return default based on tensor content
        gmm_outputs = {"routing_info": {}}
        relevance = rbs_model._compute_segment_relevance(gmm_outputs)
        assert relevance == 0.0  # Default when routing_probs is empty tensor


class TestModeManagement:
    """Test training and inference mode management."""

    def test_set_training_mode(self, rbs_model):
        """Test setting training mode."""
        rbs_model.set_training_mode("supervised")
        assert rbs_model.training_mode == "supervised"

        rbs_model.set_training_mode("rl")
        assert rbs_model.training_mode == "rl"

        with pytest.raises(ValueError):
            rbs_model.set_training_mode("invalid")

    def test_set_inference_mode(self, rbs_model):
        """Test setting inference mode."""
        rbs_model.set_inference_mode("adaptive")
        assert rbs_model.inference_mode == "adaptive"

        rbs_model.set_inference_mode("full")
        assert rbs_model.inference_mode == "full"

        with pytest.raises(ValueError):
            rbs_model.set_inference_mode("invalid")


class TestSerialization:
    """Test model serialization and deserialization."""

    def test_save_pretrained(self, rbs_model):
        """Test saving model with RBS state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rbs_model.save_pretrained(temp_dir)

            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, "rbs_config.json"))
            assert os.path.exists(os.path.join(temp_dir, "belief_tracker.pt"))
            assert os.path.exists(os.path.join(temp_dir, "halting_policy.pt"))
            # Note: gmm_backbone directory might not be created with mocked backbone

            # Check config content
            with open(os.path.join(temp_dir, "rbs_config.json"), "r") as f:
                saved_config = json.load(f)

            assert saved_config['use_rbs_mode'] is True
            assert saved_config['num_memory_experts'] == 4

    def test_from_pretrained(self, rbs_model):
        """Test loading model from checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the model
            rbs_model.save_pretrained(temp_dir)

            # Load the model
            with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
                mock_backbone = MagicMock()
                mock_backbone.hidden_dim = 768
                mock_gmm_class.from_pretrained.return_value = mock_backbone

                with patch('transformers.XLNetForQuestionAnsweringSimple.from_pretrained') as mock_xlnet:
                    mock_xlnet.return_value = MagicMock()

                    loaded_model = RBSXLNetForQA.from_pretrained(temp_dir)

                assert loaded_model.config.use_rbs_mode is True
                assert loaded_model.config.num_memory_experts == 4
                assert loaded_model.belief_tracker is not None
                assert loaded_model.halting_policy is not None

    def test_from_pretrained_legacy_fallback(self, rbs_model):
        """Test loading legacy model without RBS config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only GMM config (no RBS config)
            gmm_config = {
                "model_class": "GMMXLNetForQA",
                "memory_type": "gmm",
                "num_experts": 2,
                "memory_slots": 8
            }

            with open(os.path.join(temp_dir, "gmm_config.json"), "w") as f:
                json.dump(gmm_config, f)

            with patch('gmmxlnet.models.GMMXLNetForQA') as mock_gmm_class:
                mock_backbone = MagicMock()
                mock_gmm_class.from_pretrained.return_value = mock_backbone

                loaded_model = RBSXLNetForQA.from_pretrained(temp_dir)

                # Should use default RBS config but extract GMM settings
                assert loaded_model.config.num_memory_experts == 2
                assert loaded_model.config.memory_num_tokens == 8


class TestMemoryState:
    """Test memory state management."""

    def test_get_memory_state(self, rbs_model):
        """Test getting current memory state."""
        memory_state = rbs_model.get_memory_state()

        assert isinstance(memory_state, dict)
        assert len(memory_state) == 4  # 4 experts
        for expert_key, expert_state in memory_state.items():
            assert expert_key.startswith("expert_")
            assert isinstance(expert_state, torch.Tensor)

    def test_set_memory_state(self, rbs_model):
        """Test setting memory state."""
        new_memory_state = {
            f"expert_{i}": torch.randn(16, 768)
            for i in range(4)
        }

        # Should not raise any errors
        rbs_model.set_memory_state(new_memory_state)


class TestModelRepresentation:
    """Test model string representation."""

    def test_model_repr(self, rbs_model):
        """Test model string representation."""
        repr_str = repr(rbs_model)

        assert "RBSXLNetForQA" in repr_str
        assert "num_experts=4" in repr_str
        assert "memory_tokens=16" in repr_str
        assert "rbs_mode=True" in repr_str
        assert "training_mode=supervised" in repr_str
        assert "inference_mode=adaptive" in repr_str


class TestInferenceResultUtilities:
    """Test inference result utility methods."""

    def test_inference_result_savings(self):
        """Test savings computation in inference result."""
        # Early stopping case
        result = RBSInferenceResult(
            answer_span=(0, 5),
            confidence=0.8,
            segments_processed=3,
            total_segments=10,
            belief_history=[],
            halting_history=[],
            memory_state={},
            efficiency_score=10/3
        )

        savings = result.compute_savings()
        assert savings == 70.0  # (10-3)/10 * 100

        # No savings case
        result = RBSInferenceResult(
            answer_span=(0, 5),
            confidence=0.8,
            segments_processed=10,
            total_segments=10,
            belief_history=[],
            halting_history=[],
            memory_state={},
            efficiency_score=1.0
        )

        savings = result.compute_savings()
        assert savings == 0.0

    def test_inference_result_to_dict(self):
        """Test inference result serialization."""
        # Mock belief state with revision count
        mock_belief = MagicMock()
        mock_belief.revision_count = 1
        mock_belief.get_trend_analysis.return_value = {"trend": "increasing"}

        result = RBSInferenceResult(
            answer_span=(5, 10),
            confidence=0.85,
            segments_processed=4,
            total_segments=8,
            belief_history=[mock_belief],
            halting_history=[],
            memory_state={},
            efficiency_score=2.0
        )

        result_dict = result.to_dict()

        assert result_dict['answer_span'] == (5, 10)
        assert result_dict['confidence'] == 0.85
        assert result_dict['savings_percent'] == 50.0
        assert result_dict['efficiency_score'] == 2.0
        assert result_dict['num_revisions'] == 1  # Count of belief states with revision_count > 0
        assert result_dict['final_confidence_trend'] == {"trend": "increasing"}