"""
Unit tests for memory visualization utilities.

Tests the MemoryVisualizer class for creating attention and usage visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from memxlnet.utils.memory_visualization import MemoryVisualizer


class TestMemoryVisualizer:
    """Test MemoryVisualizer class."""

    @pytest.fixture(autouse=True)
    def close_figures(self):
        """Automatically close all matplotlib figures after each test."""
        yield
        plt.close("all")

    @pytest.fixture
    def visualizer(self, tmp_path):
        """Create a memory visualizer instance with temporary output directory."""
        return MemoryVisualizer(output_dir=str(tmp_path))

    @pytest.fixture
    def sample_attention(self):
        """Create sample attention weights."""
        # Shape: (num_heads=2, num_slots=8)
        return np.random.rand(2, 8)

    @pytest.fixture
    def sample_usage(self):
        """Create sample usage history."""
        # Shape: (num_steps=10, num_slots=8)
        return np.random.rand(10, 8)

    def test_initialization(self, tmp_path):
        """Test visualizer initialization."""
        visualizer = MemoryVisualizer(output_dir=str(tmp_path), figsize=(10, 6), dpi=150, colormap="plasma")

        assert visualizer.output_dir == tmp_path
        assert visualizer.figsize == (10, 6)
        assert visualizer.dpi == 150
        assert visualizer.colormap == "plasma"

    def test_output_dir_creation(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "visualizations" / "nested"
        MemoryVisualizer(output_dir=str(output_dir))

        assert output_dir.exists()

    def test_plot_attention_heatmap_2d(self, visualizer, sample_attention):
        """Test plotting 2D attention heatmap."""
        fig = visualizer.plot_attention_heatmap(
            weights=sample_attention, title="Test Attention", save_path="test_attention.png"
        )

        # Check figure created
        assert fig is not None

        # Check file saved
        output_file = visualizer.output_dir / "test_attention.png"
        assert output_file.exists()

    def test_plot_attention_heatmap_1d(self, visualizer):
        """Test plotting 1D attention (single head)."""
        weights = np.random.rand(8)  # Single head
        fig = visualizer.plot_attention_heatmap(weights=weights, title="Single Head Attention")

        assert fig is not None

    def test_plot_attention_heatmap_custom_labels(self, visualizer, sample_attention):
        """Test heatmap with custom labels."""
        fig = visualizer.plot_attention_heatmap(
            weights=sample_attention, title="Custom Labels", xlabel="Slots", ylabel="Attention Heads"
        )

        assert fig is not None

    def test_plot_attention_heatmap_with_values(self, visualizer):
        """Test heatmap with values shown in cells."""
        # Small array to show values
        weights = np.array([[0.1, 0.2], [0.3, 0.4]])

        fig = visualizer.plot_attention_heatmap(weights=weights, show_values=True)

        assert fig is not None

    def test_plot_usage_timeline(self, visualizer, sample_usage):
        """Test plotting usage timeline."""
        fig = visualizer.plot_usage_timeline(usage_history=sample_usage, save_path="usage_timeline.png")

        assert fig is not None

        output_file = visualizer.output_dir / "usage_timeline.png"
        assert output_file.exists()

    def test_plot_usage_timeline_from_list(self, visualizer):
        """Test usage timeline from list of arrays."""
        usage_list = [np.random.rand(8) for _ in range(10)]

        fig = visualizer.plot_usage_timeline(usage_history=usage_list)

        assert fig is not None

    def test_plot_temporal_links(self, visualizer):
        """Test plotting temporal link matrix."""
        # Square matrix (num_slots x num_slots)
        temporal_links = np.random.rand(8, 8)

        fig = visualizer.plot_temporal_links(temporal_links=temporal_links, save_path="temporal_links.png")

        assert fig is not None

        output_file = visualizer.output_dir / "temporal_links.png"
        assert output_file.exists()

    def test_plot_multi_head_comparison(self, visualizer, sample_attention):
        """Test multi-head attention comparison plot."""
        fig = visualizer.plot_multi_head_comparison(
            heads_weights=sample_attention, save_path="multi_head.png", operation="Read"
        )

        assert fig is not None

        output_file = visualizer.output_dir / "multi_head.png"
        assert output_file.exists()

    def test_plot_multi_head_comparison_single_head(self, visualizer):
        """Test multi-head comparison with single head."""
        weights = np.random.rand(1, 8)  # Single head

        fig = visualizer.plot_multi_head_comparison(heads_weights=weights, operation="Write")

        assert fig is not None

    def test_plot_attention_distribution(self, visualizer, sample_attention):
        """Test attention distribution comparison plot."""
        read_weights = sample_attention
        write_weights = np.random.rand(2, 8)

        fig = visualizer.plot_attention_distribution(
            read_weights=read_weights, write_weights=write_weights, save_path="distribution.png"
        )

        assert fig is not None

        output_file = visualizer.output_dir / "distribution.png"
        assert output_file.exists()

    def test_plot_attention_distribution_1d(self, visualizer):
        """Test distribution with 1D weights."""
        read_weights = np.random.rand(8)
        write_weights = np.random.rand(8)

        fig = visualizer.plot_attention_distribution(read_weights=read_weights, write_weights=write_weights)

        assert fig is not None

    def test_create_summary_report(self, visualizer):
        """Test creating comprehensive summary report."""
        memory_data = {
            "read_weights": [np.random.rand(2, 8) for _ in range(5)],
            "write_weights": [np.random.rand(2, 8) for _ in range(5)],
            "usage": np.random.rand(5, 8),
            "temporal_links": [np.random.rand(8, 8) for _ in range(5)],
        }

        visualizer.create_summary_report(memory_data=memory_data, save_path="summary")

        # Check that multiple files were created
        expected_files = [
            "summary_read_attention.png",
            "summary_write_attention.png",
            "summary_usage_timeline.png",
            "summary_temporal_links.png",
            "summary_distribution.png",
        ]

        for filename in expected_files:
            output_file = visualizer.output_dir / filename
            assert output_file.exists(), f"Missing file: {filename}"

    def test_create_summary_report_partial_data(self, visualizer):
        """Test summary report with partial data."""
        # Only some data available
        memory_data = {
            "read_weights": [np.random.rand(2, 8)],
            "write_weights": [np.random.rand(2, 8)],
        }

        visualizer.create_summary_report(memory_data=memory_data, save_path="partial_summary")

        # Should create at least some files
        read_file = visualizer.output_dir / "partial_summary_read_attention.png"
        write_file = visualizer.output_dir / "partial_summary_write_attention.png"

        assert read_file.exists()
        assert write_file.exists()

    def test_nested_save_path(self, visualizer):
        """Test saving with nested directory structure."""
        visualizer.plot_attention_heatmap(weights=np.random.rand(2, 8), save_path="nested/dir/attention.png")

        output_file = visualizer.output_dir / "nested" / "dir" / "attention.png"
        assert output_file.exists()

    def test_custom_figsize(self, visualizer):
        """Test using custom figure size."""
        fig = visualizer.plot_attention_heatmap(weights=np.random.rand(2, 8), figsize=(8, 6))

        # Figure should have custom size
        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 6

    def test_large_attention_matrix(self, visualizer):
        """Test handling large attention matrices."""
        # Large matrix (many heads and slots)
        large_weights = np.random.rand(8, 64)

        fig = visualizer.plot_attention_heatmap(weights=large_weights)

        assert fig is not None

    def test_zero_attention_weights(self, visualizer):
        """Test handling zero attention weights."""
        zero_weights = np.zeros((2, 8))

        fig = visualizer.plot_attention_heatmap(weights=zero_weights)

        assert fig is not None

    def test_plot_without_save(self, visualizer, sample_attention):
        """Test creating plots without saving to file."""
        fig = visualizer.plot_attention_heatmap(
            weights=sample_attention,
            save_path=None,  # Don't save
        )

        assert fig is not None

    def test_all_visualizations_different_shapes(self, visualizer):
        """Test that visualizations work with various array shapes."""
        # Different shapes for testing
        shapes = [
            (1, 8),  # Single head
            (4, 16),  # Multiple heads
            (8, 64),  # Many slots
        ]

        for num_heads, num_slots in shapes:
            weights = np.random.rand(num_heads, num_slots)

            # All plot types should work
            fig1 = visualizer.plot_attention_heatmap(weights)
            assert fig1 is not None

            fig2 = visualizer.plot_multi_head_comparison(weights)
            assert fig2 is not None

            # Distribution plot
            fig3 = visualizer.plot_attention_distribution(weights, weights)
            assert fig3 is not None


class TestVisualizationIntegration:
    """Integration tests for complete visualization workflows."""

    @pytest.fixture(autouse=True)
    def close_figures(self):
        """Automatically close all matplotlib figures after each test."""
        yield
        plt.close("all")

    def test_complete_analysis_workflow(self, tmp_path):
        """Test complete analysis workflow from data to visualizations."""
        visualizer = MemoryVisualizer(output_dir=str(tmp_path))

        # Simulate multi-segment memory evolution
        num_segments = 5
        num_heads = 2
        num_slots = 16

        segment_data = []
        for i in range(num_segments):
            segment_data.append(
                {
                    "read_weights": np.random.rand(num_heads, num_slots),
                    "write_weights": np.random.rand(num_heads, num_slots),
                    "usage": np.random.rand(num_slots),
                }
            )

        # Create visualizations for each segment
        for i, data in enumerate(segment_data):
            visualizer.plot_attention_heatmap(
                data["read_weights"], title=f"Read Attention - Segment {i}", save_path=f"segment_{i}_read.png"
            )

            visualizer.plot_attention_heatmap(
                data["write_weights"], title=f"Write Attention - Segment {i}", save_path=f"segment_{i}_write.png"
            )

        # Check all files created
        for i in range(num_segments):
            assert (tmp_path / f"segment_{i}_read.png").exists()
            assert (tmp_path / f"segment_{i}_write.png").exists()

    def test_comparison_analysis(self, tmp_path):
        """Test comparing different memory configurations."""
        visualizer = MemoryVisualizer(output_dir=str(tmp_path))

        # Simulate two models
        model_a_read = np.random.rand(2, 8)
        model_b_read = np.random.rand(4, 16)

        # Create comparison visualizations
        visualizer.plot_multi_head_comparison(model_a_read, save_path="model_a_read_heads.png")

        visualizer.plot_multi_head_comparison(model_b_read, save_path="model_b_read_heads.png")

        # Both should exist
        assert (tmp_path / "model_a_read_heads.png").exists()
        assert (tmp_path / "model_b_read_heads.png").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
