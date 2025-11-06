"""
Unit tests for RBS-QA evaluation configuration.
"""

import pytest
from dataclasses import fields

from rbsqa.evaluation.config import RBSEvaluationConfig


class TestRBSEvaluationConfig:
    """Test RBS evaluation configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RBSEvaluationConfig()

        assert config.output_dir == "./evaluation_results"
        assert config.max_segments_per_example == 32
        assert config.generate_visualizations == True
        assert config.save_detailed_results == True
        assert config.baseline_comparisons == ["gmm", "base_xlnet"]
        assert config.statistical_tests == True
        assert config.error_analysis_depth == 50
        assert config.generate_html_report == True
        assert config.generate_markdown_report == True
        assert config.confidence_thresholds == [0.1, 0.3, 0.5, 0.7, 0.9]
        assert config.efficiency_bins == 10
        assert config.include_qualitative_analysis == True
        assert config.save_belief_states == False
        assert config.verbose_logging == True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RBSEvaluationConfig(
            output_dir="/custom/output",
            max_segments_per_example=64,
            generate_visualizations=False,
            baseline_comparisons=["gmm"],
            error_analysis_depth=100,
            confidence_thresholds=[0.2, 0.4, 0.6, 0.8],
            efficiency_bins=20,
            verbose_logging=False
        )

        assert config.output_dir == "/custom/output"
        assert config.max_segments_per_example == 64
        assert config.generate_visualizations == False
        assert config.baseline_comparisons == ["gmm"]
        assert config.error_analysis_depth == 100
        assert config.confidence_thresholds == [0.2, 0.4, 0.6, 0.8]
        assert config.efficiency_bins == 20
        assert config.verbose_logging == False

    def test_config_fields(self):
        """Test that all expected fields are present."""
        config = RBSEvaluationConfig()
        field_names = {f.name for f in fields(config)}

        expected_fields = {
            'output_dir',
            'max_segments_per_example',
            'generate_visualizations',
            'save_detailed_results',
            'baseline_comparisons',
            'statistical_tests',
            'error_analysis_depth',
            'generate_html_report',
            'generate_markdown_report',
            'confidence_thresholds',
            'efficiency_bins',
            'include_qualitative_analysis',
            'save_belief_states',
            'verbose_logging'
        }

        assert field_names == expected_fields

    def test_config_field_types(self):
        """Test that fields have correct types."""
        config = RBSEvaluationConfig()

        assert isinstance(config.output_dir, str)
        assert isinstance(config.max_segments_per_example, int)
        assert isinstance(config.generate_visualizations, bool)
        assert isinstance(config.save_detailed_results, bool)
        assert isinstance(config.baseline_comparisons, list)
        assert isinstance(config.statistical_tests, bool)
        assert isinstance(config.error_analysis_depth, int)
        assert isinstance(config.generate_html_report, bool)
        assert isinstance(config.generate_markdown_report, bool)
        assert isinstance(config.confidence_thresholds, list)
        assert isinstance(config.efficiency_bins, int)
        assert isinstance(config.include_qualitative_analysis, bool)
        assert isinstance(config.save_belief_states, bool)
        assert isinstance(config.verbose_logging, bool)

    def test_baseline_comparisons_content(self):
        """Test baseline comparisons list content."""
        config = RBSEvaluationConfig()

        assert len(config.baseline_comparisons) > 0
        assert all(isinstance(comp, str) for comp in config.baseline_comparisons)
        assert "gmm" in config.baseline_comparisons
        assert "base_xlnet" in config.baseline_comparisons

    def test_confidence_thresholds_content(self):
        """Test confidence thresholds list content."""
        config = RBSEvaluationConfig()

        assert len(config.confidence_thresholds) > 0
        assert all(isinstance(thresh, (int, float)) for thresh in config.confidence_thresholds)
        assert all(0.0 <= thresh <= 1.0 for thresh in config.confidence_thresholds)
        assert config.confidence_thresholds == sorted(config.confidence_thresholds)

    def test_integer_constraints(self):
        """Test integer field constraints."""
        config = RBSEvaluationConfig()

        assert config.max_segments_per_example > 0
        assert config.error_analysis_depth > 0
        assert config.efficiency_bins > 0

    def test_config_immutability(self):
        """Test that config can be modified when needed."""
        config = RBSEvaluationConfig()

        # Should be able to modify values
        config.output_dir = "/new/path"
        config.max_segments_per_example = 128

        assert config.output_dir == "/new/path"
        assert config.max_segments_per_example == 128

    def test_equality(self):
        """Test config equality comparison."""
        config1 = RBSEvaluationConfig()
        config2 = RBSEvaluationConfig()

        assert config1 == config2

        # Modify one config
        config3 = RBSEvaluationConfig(output_dir="/different")
        assert config1 != config3

    def test_repr(self):
        """Test config string representation."""
        config = RBSEvaluationConfig()
        repr_str = repr(config)

        assert "RBSEvaluationConfig" in repr_str
        assert "output_dir" in repr_str

    def test_config_with_lists(self):
        """Test configuration with list fields."""
        custom_baselines = ["custom1", "custom2", "custom3"]
        custom_thresholds = [0.1, 0.5, 0.9]

        config = RBSEvaluationConfig(
            baseline_comparisons=custom_baselines,
            confidence_thresholds=custom_thresholds
        )

        assert config.baseline_comparisons == custom_baselines
        assert config.confidence_thresholds == custom_thresholds

    def test_extreme_values(self):
        """Test configuration with extreme values."""
        config = RBSEvaluationConfig(
            max_segments_per_example=1,
            error_analysis_depth=1,
            efficiency_bins=1,
            confidence_thresholds=[0.0, 1.0]
        )

        assert config.max_segments_per_example == 1
        assert config.error_analysis_depth == 1
        assert config.efficiency_bins == 1
        assert config.confidence_thresholds == [0.0, 1.0]