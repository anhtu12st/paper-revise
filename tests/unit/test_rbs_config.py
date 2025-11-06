"""
Unit tests for RBS-QA Configuration components.

Tests cover RBSTrainingConfig class and its preset factory functions
with comprehensive validation of configuration parameters and constraints.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

from rbsqa.config import (
    RBSTrainingConfig,
    rbs_lightweight_config,
    rbs_balanced_config,
    rbs_advanced_config,
    rbs_research_config
)


class TestRBSTrainingConfig:
    """Test cases for RBSTrainingConfig class."""

    def test_rbs_config_initialization_default(self):
        """Test RBSTrainingConfig initialization with defaults."""
        config = RBSTrainingConfig()

        # Test default values
        assert config.use_belief_state is True
        assert config.belief_state_threshold == 0.7
        assert config.enable_re_scoring is True
        assert config.confidence_calibration is True
        assert config.re_scoring_method == "context_weighted"
        assert config.enable_trend_analysis is True
        assert config.max_segments == 32
        assert config.belief_state_memory_limit == 100
        assert config.halting_patience == 3
        assert config.enable_learnable_re_scoring is False
        assert config.belief_update_frequency == "every_segment"
        assert config.non_monotonic_revision_threshold == 0.05
        assert config.enable_belief_state_visualization is False

    def test_rbs_config_initialization_custom(self):
        """Test RBSTrainingConfig initialization with custom values."""
        config = RBSTrainingConfig(
            belief_state_threshold=0.8,
            re_scoring_method="learned",
            max_segments=64,
            enable_learnable_re_scoring=True,
            belief_update_frequency="every_other"
        )

        assert config.belief_state_threshold == 0.8
        assert config.re_scoring_method == "learned"
        assert config.max_segments == 64
        assert config.enable_learnable_re_scoring is True
        assert config.belief_update_frequency == "every_other"

    def test_rbs_config_inherits_gmm_defaults(self):
        """Test that RBSTrainingConfig inherits GMM defaults correctly."""
        config = RBSTrainingConfig()

        # Check GMM defaults are present
        assert hasattr(config, 'use_gmm_memory')
        assert hasattr(config, 'num_memory_experts')
        assert hasattr(config, 'routing_temperature')
        assert hasattr(config, 'memory_num_tokens')

    def test_rbs_config_validation_belief_state_threshold(self):
        """Test validation of belief_state_threshold parameter."""
        # Valid values
        RBSTrainingConfig(belief_state_threshold=0.0)
        RBSTrainingConfig(belief_state_threshold=0.5)
        RBSTrainingConfig(belief_state_threshold=1.0)

        # Invalid values
        with pytest.raises(ValueError, match="belief_state_threshold must be in \\[0.0, 1.0\\]"):
            RBSTrainingConfig(belief_state_threshold=-0.1)

        with pytest.raises(ValueError, match="belief_state_threshold must be in \\[0.0, 1.0\\]"):
            RBSTrainingConfig(belief_state_threshold=1.1)

    def test_rbs_config_validation_re_scoring_method(self):
        """Test validation of re_scoring_method parameter."""
        valid_methods = ["context_weighted", "learned", "exponential_decay"]
        for method in valid_methods:
            if method == "learned":
                # learned method requires enable_learnable_re_scoring=True
                config = RBSTrainingConfig(re_scoring_method=method, enable_learnable_re_scoring=True)
            else:
                config = RBSTrainingConfig(re_scoring_method=method)
            assert config.re_scoring_method == method

        # Invalid method
        with pytest.raises(ValueError, match="re_scoring_method must be one of"):
            RBSTrainingConfig(re_scoring_method="invalid_method")

    def test_rbs_config_validation_max_segments(self):
        """Test validation of max_segments parameter."""
        # Valid values
        RBSTrainingConfig(max_segments=1)
        RBSTrainingConfig(max_segments=100)

        # Invalid values
        with pytest.raises(ValueError, match="max_segments must be a positive integer"):
            RBSTrainingConfig(max_segments=0)

        with pytest.raises(ValueError, match="max_segments must be a positive integer"):
            RBSTrainingConfig(max_segments=-5)

        with pytest.raises(ValueError, match="max_segments must be a positive integer"):
            RBSTrainingConfig(max_segments=2.5)

    def test_rbs_config_validation_belief_state_memory_limit(self):
        """Test validation of belief_state_memory_limit parameter."""
        # Valid values
        RBSTrainingConfig(belief_state_memory_limit=10)
        RBSTrainingConfig(belief_state_memory_limit=1000)

        # Invalid values
        with pytest.raises(ValueError, match="belief_state_memory_limit must be a positive integer"):
            RBSTrainingConfig(belief_state_memory_limit=0)

        with pytest.raises(ValueError, match="belief_state_memory_limit must be a positive integer"):
            RBSTrainingConfig(belief_state_memory_limit=-1)

    def test_rbs_config_validation_halting_patience(self):
        """Test validation of halting_patience parameter."""
        # Valid values
        RBSTrainingConfig(halting_patience=0)
        RBSTrainingConfig(halting_patience=10)

        # Invalid values
        with pytest.raises(ValueError, match="halting_patience must be a non-negative integer"):
            RBSTrainingConfig(halting_patience=-1)

        with pytest.raises(ValueError, match="halting_patience must be a non-negative integer"):
            RBSTrainingConfig(halting_patience=2.5)

    def test_rbs_config_validation_belief_update_frequency(self):
        """Test validation of belief_update_frequency parameter."""
        valid_frequencies = ["every_segment", "every_other", "every_third", "every_fifth"]
        for freq in valid_frequencies:
            config = RBSTrainingConfig(belief_update_frequency=freq)
            assert config.belief_update_frequency == freq

        # Invalid frequency
        with pytest.raises(ValueError, match="belief_update_frequency must be one of"):
            RBSTrainingConfig(belief_update_frequency="invalid_frequency")

    def test_rbs_config_validation_non_monotonic_revision_threshold(self):
        """Test validation of non_monotonic_revision_threshold parameter."""
        # Valid values
        RBSTrainingConfig(non_monotonic_revision_threshold=0.0)
        RBSTrainingConfig(non_monotonic_revision_threshold=0.1)
        RBSTrainingConfig(non_monotonic_revision_threshold=1.0)

        # Invalid values
        with pytest.raises(ValueError, match="non_monotonic_revision_threshold must be >= 0"):
            RBSTrainingConfig(non_monotonic_revision_threshold=-0.1)

    def test_rbs_config_gmm_dependency_validation(self):
        """Test validation that RBS requires GMM to be enabled."""
        with pytest.raises(ValueError, match="RBS-QA requires GMM memory to be enabled"):
            RBSTrainingConfig(use_gmm_memory=False)

    def test_rbs_config_learnable_re_scoring_consistency(self):
        """Test validation of learnable re-scoring consistency."""
        # Valid: enable_learnable_re_scoring=True with learned method
        RBSTrainingConfig(
            enable_learnable_re_scoring=True,
            re_scoring_method="learned"
        )

        # Invalid: enable_learnable_re_scoring=True but not learned method
        with pytest.raises(ValueError, match="enable_learnable_re_scoring=True requires re_scoring_method='learned'"):
            RBSTrainingConfig(
                enable_learnable_re_scoring=True,
                re_scoring_method="context_weighted"
            )

        # Invalid: learned method but enable_learnable_re_scoring=False
        with pytest.raises(ValueError, match="re_scoring_method='learned' requires enable_learnable_re_scoring=True"):
            RBSTrainingConfig(
                enable_learnable_re_scoring=False,
                re_scoring_method="learned"
            )

    def test_get_belief_update_interval(self):
        """Test belief update interval calculation."""
        test_cases = [
            ("every_segment", 1),
            ("every_other", 2),
            ("every_third", 3),
            ("every_fifth", 5)
        ]

        for frequency, expected_interval in test_cases:
            config = RBSTrainingConfig(belief_update_frequency=frequency)
            assert config.get_belief_update_interval() == expected_interval

    def test_should_enable_belief_features(self):
        """Test belief feature enabling logic."""
        # All features enabled
        config = RBSTrainingConfig(
            use_belief_state=True,
            enable_re_scoring=True,
            confidence_calibration=True,
            enable_trend_analysis=True,
            enable_learnable_re_scoring=True,
            re_scoring_method="learned",  # Consistent with learnable_re_scoring
            enable_belief_state_visualization=True
        )

        features = config.should_enable_belief_features()
        expected_features = {
            "tracking": True,
            "re_scoring": True,
            "calibration": True,
            "trend_analysis": True,
            "learnable_re_scoring": True,
            "visualization": True
        }
        assert features == expected_features

        # Belief state disabled - all dependent features should be False
        config = RBSTrainingConfig(use_belief_state=False)
        features = config.should_enable_belief_features()
        expected_features = {
            "tracking": False,
            "re_scoring": False,
            "calibration": False,
            "trend_analysis": False,
            "learnable_re_scoring": False,
            "visualization": False
        }
        assert features == expected_features

        # Mixed configuration
        config = RBSTrainingConfig(
            use_belief_state=True,
            enable_re_scoring=False,
            enable_trend_analysis=False
        )
        features = config.should_enable_belief_features()
        assert features["tracking"] is True
        assert features["re_scoring"] is False
        assert features["trend_analysis"] is False

    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = RBSTrainingConfig(
            belief_state_threshold=0.8,
            re_scoring_method="learned",
            enable_learnable_re_scoring=True,  # Consistent with learned method
            max_segments=64,
            use_gmm_memory=True,
            num_memory_experts=4
        )

        config_dict = config.to_dict()

        # Check RBS-specific parameters
        assert config_dict["belief_state_threshold"] == 0.8
        assert config_dict["re_scoring_method"] == "learned"
        assert config_dict["max_segments"] == 64

        # Check GMM parameters are preserved
        assert config_dict["use_gmm_memory"] is True
        assert config_dict["num_memory_experts"] == 4

        # Check metadata
        assert config_dict["reasoning_type"] == "rbs"
        assert config_dict["rbs_version"] == "1.0"

    def test_from_dict(self):
        """Test configuration deserialization from dictionary."""
        original_dict = {
            # GMM parameters
            "use_gmm_memory": True,
            "num_memory_experts": 6,
            "memory_num_tokens": 20,
            "routing_temperature": 0.9,
            "load_balance_weight": 0.015,

            # RBS parameters
            "use_belief_state": True,
            "belief_state_threshold": 0.75,
            "re_scoring_method": "exponential_decay",
            "max_segments": 48,
            "enable_learnable_re_scoring": False,

            # Metadata (should be ignored)
            "reasoning_type": "rbs",
            "rbs_version": "1.0"
        }

        config = RBSTrainingConfig.from_dict(original_dict)

        assert config.use_gmm_memory is True
        assert config.num_memory_experts == 6
        assert config.memory_num_tokens == 20
        assert config.belief_state_threshold == 0.75
        assert config.re_scoring_method == "exponential_decay"
        assert config.max_segments == 48
        assert config.enable_learnable_re_scoring is False

    def test_json_serialization_roundtrip(self):
        """Test JSON save/load functionality."""
        original_config = RBSTrainingConfig(
            belief_state_threshold=0.85,
            re_scoring_method="learned",
            max_segments=128,
            use_gmm_memory=True,
            num_memory_experts=8,
            memory_num_tokens=32,
            enable_learnable_re_scoring=True,
            enable_belief_state_visualization=True
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save to JSON
            original_config.to_json(temp_path)

            # Load from JSON
            loaded_config = RBSTrainingConfig.from_json(temp_path)

            # Verify all parameters match
            assert loaded_config.belief_state_threshold == original_config.belief_state_threshold
            assert loaded_config.re_scoring_method == original_config.re_scoring_method
            assert loaded_config.max_segments == original_config.max_segments
            assert loaded_config.use_gmm_memory == original_config.use_gmm_memory
            assert loaded_config.num_memory_experts == original_config.num_memory_experts
            assert loaded_config.memory_num_tokens == original_config.memory_num_tokens
            assert loaded_config.enable_learnable_re_scoring == original_config.enable_learnable_re_scoring
            assert loaded_config.enable_belief_state_visualization == original_config.enable_belief_state_visualization

        finally:
            os.unlink(temp_path)

    def test_json_file_handling(self):
        """Test JSON file error handling."""
        config = RBSTrainingConfig()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Test saving
            config.to_json(temp_path)

            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                data = json.load(f)
                assert "use_belief_state" in data

            # Test loading
            loaded_config = RBSTrainingConfig.from_json(temp_path)
            assert loaded_config.use_belief_state == config.use_belief_state

        finally:
            os.unlink(temp_path)

    def test_from_json_file_not_found(self):
        """Test loading from non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            RBSTrainingConfig.from_json("non_existent_file.json")

    def test_from_json_invalid_json(self):
        """Test loading from invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                RBSTrainingConfig.from_json(temp_path)
        finally:
            os.unlink(temp_path)


class TestRBSConfigPresets:
    """Test cases for RBS configuration factory functions."""

    def test_rbs_lightweight_config_defaults(self):
        """Test lightweight config factory defaults."""
        config = rbs_lightweight_config()

        # Check lightweight-specific defaults
        assert config.use_gmm_memory is True
        assert config.num_memory_experts == 2  # Small number of experts
        assert config.memory_num_tokens == 8   # Small memory
        assert config.routing_temperature == 1.2

        # RBS-specific lightweight defaults
        assert config.belief_state_threshold == 0.8  # Higher threshold
        assert config.re_scoring_method == "context_weighted"  # Simple method
        assert config.confidence_calibration is False  # Disabled for speed
        assert config.enable_trend_analysis is False  # Disabled for speed
        assert config.max_segments == 16  # Fewer segments
        assert config.belief_state_memory_limit == 50  # Smaller memory
        assert config.belief_update_frequency == "every_other"  # Less frequent updates
        assert config.non_monotonic_revision_threshold == 0.1  # Higher threshold

    def test_rbs_lightweight_config_overrides(self):
        """Test lightweight config factory with parameter overrides."""
        config = rbs_lightweight_config(
            belief_state_threshold=0.6,
            max_segments=32,
            num_memory_experts=4
        )

        assert config.belief_state_threshold == 0.6  # Overridden
        assert config.max_segments == 32  # Overridden
        assert config.num_memory_experts == 4  # Overridden
        assert config.memory_num_tokens == 8  # Default preserved

    def test_rbs_balanced_config_defaults(self):
        """Test balanced config factory defaults."""
        config = rbs_balanced_config()

        # Check balanced-specific defaults
        assert config.use_gmm_memory is True
        assert config.num_memory_experts == 4
        assert config.memory_num_tokens == 16
        assert config.routing_temperature == 1.0

        # RBS-specific balanced defaults
        assert config.belief_state_threshold == 0.7
        assert config.re_scoring_method == "context_weighted"
        assert config.confidence_calibration is True
        assert config.enable_trend_analysis is True
        assert config.max_segments == 32
        assert config.belief_state_memory_limit == 100
        assert config.belief_update_frequency == "every_segment"
        assert config.non_monotonic_revision_threshold == 0.05

    def test_rbs_advanced_config_defaults(self):
        """Test advanced config factory defaults."""
        config = rbs_advanced_config()

        # Check advanced-specific defaults
        assert config.use_gmm_memory is True
        assert config.num_memory_experts == 6
        assert config.memory_num_tokens == 24
        assert config.routing_temperature == 0.8  # Lower for sharper routing
        assert config.entropy_regularization_weight == 0.001  # Encourage diversity
        assert config.load_balance_weight == 0.02  # Stronger load balancing

        # RBS-specific advanced defaults
        assert config.belief_state_threshold == 0.6  # Lower for thorough processing
        assert config.re_scoring_method == "learned"
        assert config.enable_learnable_re_scoring is True
        assert config.max_segments == 64
        assert config.belief_state_memory_limit == 200
        assert config.halting_patience == 5
        assert config.non_monotonic_revision_threshold == 0.02  # More sensitive

    def test_rbs_research_config_defaults(self):
        """Test research config factory defaults."""
        config = rbs_research_config()

        # Check research-specific defaults
        assert config.use_gmm_memory is True
        assert config.num_memory_experts == 8  # Maximum for study
        assert config.memory_num_tokens == 32
        assert config.routing_temperature == 1.0
        assert config.entropy_regularization_weight == 0.005
        assert config.load_balance_weight == 0.015

        # RBS-specific research defaults (comprehensive)
        assert config.belief_state_threshold == 0.5  # Very low for full dynamics
        assert config.re_scoring_method == "learned"
        assert config.enable_learnable_re_scoring is True
        assert config.max_segments == 128  # Large for long-term study
        assert config.belief_state_memory_limit == 500  # Large history
        assert config.halting_patience == 10  # High patience
        assert config.non_monotonic_revision_threshold == 0.01  # Very sensitive
        assert config.enable_belief_state_visualization is True

    def test_config_preset_overrides(self):
        """Test that all config presets respect parameter overrides."""
        base_params = {
            "belief_state_threshold": 0.9,
            "max_segments": 256,
            "num_memory_experts": 8,
            "memory_num_tokens": 32  # Use valid value within GMM constraints
        }

        # Test all presets
        presets = [rbs_lightweight_config, rbs_balanced_config, rbs_advanced_config, rbs_research_config]

        for preset_func in presets:
            config = preset_func(**base_params)

            # Check that overrides were applied
            assert config.belief_state_threshold == 0.9
            assert config.max_segments == 256
            assert config.num_memory_experts == 8
            assert config.memory_num_tokens == 32

    def test_config_preset_validation(self):
        """Test that config presets validate properly."""
        # All presets should create valid configurations
        presets = [
            rbs_lightweight_config(),
            rbs_balanced_config(),
            rbs_advanced_config(),
            rbs_research_config()
        ]

        for config in presets:
            # Should not raise any validation errors
            assert isinstance(config, RBSTrainingConfig)
            assert config.use_gmm_memory is True
            assert config.use_belief_state is True

    @pytest.mark.parametrize("preset_func", [
        rbs_lightweight_config,
        rbs_balanced_config,
        rbs_advanced_config,
        rbs_research_config
    ])
    def test_config_preset_serialization(self, preset_func):
        """Test that all config presets can be serialized."""
        config = preset_func()

        # Test dictionary serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "reasoning_type" in config_dict
        assert config_dict["reasoning_type"] == "rbs"

        # Test JSON serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            config.to_json(temp_path)
            loaded_config = RBSTrainingConfig.from_json(temp_path)

            # Verify the loaded config matches the original
            assert loaded_config.use_gmm_memory == config.use_gmm_memory
            assert loaded_config.use_belief_state == config.use_belief_state
            assert loaded_config.belief_state_threshold == config.belief_state_threshold
            assert loaded_config.max_segments == config.max_segments

        finally:
            os.unlink(temp_path)

    def test_config_preset_feature_compatibility(self):
        """Test that config presets have compatible feature combinations."""
        presets = [
            ("lightweight", rbs_lightweight_config()),
            ("balanced", rbs_balanced_config()),
            ("advanced", rbs_advanced_config()),
            ("research", rbs_research_config())
        ]

        for name, config in presets:
            features = config.should_enable_belief_features()

            # All presets should have basic tracking
            assert features["tracking"] is True, f"{name} preset should have tracking enabled"

            # Check consistency of learnable re-scoring
            if config.re_scoring_method == "learned":
                assert features["learnable_re_scoring"] is True, f"{name} preset with learned method should have learnable re-scoring"
                assert config.enable_learnable_re_scoring is True, f"{name} preset should be consistent"

            # Check GMM compatibility
            assert config.use_gmm_memory is True, f"{name} preset should have GMM enabled"

    def test_config_resource_scaling(self):
        """Test that config presets scale resource usage appropriately."""
        lightweight = rbs_lightweight_config()
        balanced = rbs_balanced_config()
        advanced = rbs_advanced_config()
        research = rbs_research_config()

        # Memory usage should increase with preset complexity
        memory_limits = [
            lightweight.belief_state_memory_limit,
            balanced.belief_state_memory_limit,
            advanced.belief_state_memory_limit,
            research.belief_state_memory_limit
        ]

        assert memory_limits[0] <= memory_limits[1] <= memory_limits[2] <= memory_limits[3]

        # GMM experts should scale appropriately
        expert_counts = [
            lightweight.num_memory_experts,
            balanced.num_memory_experts,
            advanced.num_memory_experts,
            research.num_memory_experts
        ]

        assert expert_counts[0] <= expert_counts[1] <= expert_counts[2] <= expert_counts[3]

        # Memory tokens should scale
        memory_tokens = [
            lightweight.memory_num_tokens,
            balanced.memory_num_tokens,
            advanced.memory_num_tokens,
            research.memory_num_tokens
        ]

        assert memory_tokens[0] <= memory_tokens[1] <= memory_tokens[2] <= memory_tokens[3]

        # Max segments should scale
        max_segments = [
            lightweight.max_segments,
            balanced.max_segments,
            advanced.max_segments,
            research.max_segments
        ]

        assert max_segments[0] <= max_segments[1] <= max_segments[2] <= max_segments[3]


if __name__ == "__main__":
    pytest.main([__file__])