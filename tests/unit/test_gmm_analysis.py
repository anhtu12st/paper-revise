"""
Unit tests for GMM analysis and visualization tools.

Tests cover:
- Routing tracking functionality
- Metric calculations
- Visualization function outputs
- JSON export format
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")  # Non-interactive backend for testing

from gmmxlnet.utils import GMMAnalyzer
from gmmxlnet.utils.routing_visualization import (
    generate_all_visualizations,
    plot_expert_activation_timeline,
    plot_expert_utilization_bar,
    plot_routing_entropy_distribution,
    plot_routing_heatmap,
    plot_specialization_dendrogram,
)


@pytest.fixture
def mock_model():
    """Create a mock GMM model for testing."""
    model = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)

    # Mock memory mixture
    memory_mixture = Mock()
    memory_mixture.num_experts = 4

    # Mock expert memories
    expert_memories = [
        torch.randn(8, 768)
        for _ in range(4)  # (memory_slots, hidden_dim)
    ]
    memory_mixture.expert_memories = expert_memories

    model.memory_mixture = memory_mixture

    return model


@pytest.fixture
def mock_routing_data():
    """Create mock routing data for testing."""
    num_segments = 10
    num_experts = 4

    routing_data = []
    for i in range(num_segments):
        # Create routing probabilities (softmax normalized)
        probs = torch.randn(num_experts).softmax(dim=0).numpy()
        routing_data.append(
            {
                "batch_idx": i // 2,
                "item_idx": i % 2,
                "segment_idx": i,
                "routing_probs": probs.tolist(),
                "document_id": f"doc_{i // 3}",
            }
        )

    return routing_data


@pytest.fixture
def analyzer_with_data(mock_model, mock_routing_data):
    """Create an analyzer with tracked routing data."""
    analyzer = GMMAnalyzer(model=mock_model, device="cpu")
    analyzer.routing_data = mock_routing_data

    # Set activation counts
    analyzer._activation_counts = torch.zeros(4)
    analyzer._total_activations = len(mock_routing_data)

    for record in mock_routing_data:
        probs = torch.tensor(record["routing_probs"])
        analyzer._activation_counts += probs

    return analyzer


class TestGMMAnalyzer:
    """Test suite for GMMAnalyzer class."""

    def test_initialization(self, mock_model):
        """Test GMMAnalyzer initialization."""
        analyzer = GMMAnalyzer(model=mock_model, device="cpu")

        assert analyzer.model == mock_model
        assert analyzer.device == "cpu"
        assert analyzer.num_experts == 4
        assert len(analyzer.routing_data) == 0
        assert analyzer._activation_counts is None
        assert analyzer._total_activations == 0

    def test_initialization_no_memory_mixture(self):
        """Test initialization fails without memory_mixture."""
        model = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        delattr(model, "memory_mixture")

        with pytest.raises(ValueError, match="memory_mixture"):
            GMMAnalyzer(model=model)

    def test_reset_tracking(self, analyzer_with_data):
        """Test reset_tracking clears all data."""
        # Verify data exists
        assert len(analyzer_with_data.routing_data) > 0
        assert analyzer_with_data._activation_counts is not None

        # Reset
        analyzer_with_data.reset_tracking()

        # Verify cleared
        assert len(analyzer_with_data.routing_data) == 0
        assert analyzer_with_data._activation_counts is None
        assert analyzer_with_data._total_activations == 0

    def test_export_routing_to_json(self, analyzer_with_data):
        """Test routing data export to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "routing.json"

            analyzer_with_data.export_routing_to_json(str(output_path))

            # Verify file exists and contains valid JSON
            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            # Check structure
            assert "num_experts" in data
            assert "segments_processed" in data
            assert "routing_data" in data
            assert data["num_experts"] == 4
            assert data["segments_processed"] == 10
            assert len(data["routing_data"]) == 10

    def test_compute_expert_activations(self, analyzer_with_data):
        """Test expert activation frequency computation."""
        activations = analyzer_with_data.compute_expert_activations()

        # Check structure
        assert len(activations) == 4
        assert all(f"expert_{i}" in activations for i in range(4))

        # Check values are valid frequencies
        for freq in activations.values():
            assert 0.0 <= freq <= 1.0
            assert isinstance(freq, float)

        # Check frequencies sum to approximately 1.0 per segment
        total_freq = sum(activations.values())
        assert abs(total_freq - 1.0) < 0.01

    def test_compute_expert_activations_no_data(self, mock_model):
        """Test activation computation fails without data."""
        analyzer = GMMAnalyzer(model=mock_model, device="cpu")

        with pytest.raises(ValueError, match="No routing data"):
            analyzer.compute_expert_activations()

    def test_compute_routing_entropy(self, analyzer_with_data):
        """Test routing entropy computation."""
        entropy = analyzer_with_data.compute_routing_entropy()

        # Entropy should be positive and less than log(num_experts)
        assert 0.0 <= entropy <= np.log(4)
        assert isinstance(entropy, float)

    def test_compute_routing_entropy_uniform(self):
        """Test entropy is maximal for uniform distribution."""
        # Create analyzer with uniform routing
        model = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        memory_mixture = Mock()
        memory_mixture.num_experts = 4
        memory_mixture.expert_memories = [torch.randn(8, 768) for _ in range(4)]
        model.memory_mixture = memory_mixture

        analyzer = GMMAnalyzer(model=model, device="cpu")

        # Add uniform routing data
        uniform_probs = [0.25, 0.25, 0.25, 0.25]
        analyzer.routing_data = [{"routing_probs": uniform_probs} for _ in range(10)]

        entropy = analyzer.compute_routing_entropy()

        # Should be close to log(4)
        expected_entropy = np.log(4)
        assert abs(entropy - expected_entropy) < 0.01

    def test_compute_routing_entropy_no_data(self, mock_model):
        """Test entropy computation fails without data."""
        analyzer = GMMAnalyzer(model=mock_model, device="cpu")

        with pytest.raises(ValueError, match="No routing data"):
            analyzer.compute_routing_entropy()

    def test_extract_expert_embeddings(self, analyzer_with_data):
        """Test expert embedding extraction."""
        embeddings = analyzer_with_data.extract_expert_embeddings()

        # Check shape
        assert embeddings.shape == (4, 768)  # (num_experts, hidden_dim)
        assert embeddings.dtype == torch.float32

    def test_compute_expert_diversity(self, analyzer_with_data):
        """Test expert diversity computation."""
        diversity = analyzer_with_data.compute_expert_diversity()

        # Diversity should be between 0 and 1
        assert 0.0 <= diversity <= 1.0
        assert isinstance(diversity, float)

    def test_compute_expert_diversity_orthogonal(self):
        """Test diversity is maximal for orthogonal experts."""
        model = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        memory_mixture = Mock()
        memory_mixture.num_experts = 4

        # Create orthogonal expert embeddings
        dim = 768
        expert_memories = []
        for i in range(4):
            memory = torch.zeros(8, dim)
            # Each expert focuses on different dimensions
            memory[:, i * (dim // 4) : (i + 1) * (dim // 4)] = 1.0
            expert_memories.append(memory)

        memory_mixture.expert_memories = expert_memories
        model.memory_mixture = memory_mixture

        analyzer = GMMAnalyzer(model=model, device="cpu")
        analyzer.routing_data = [{"routing_probs": [0.25] * 4}]
        analyzer._activation_counts = torch.ones(4)
        analyzer._total_activations = 1

        diversity = analyzer.compute_expert_diversity()

        # Should be close to 1.0 for orthogonal experts
        assert diversity > 0.9

    def test_compute_utilization_balance(self, analyzer_with_data):
        """Test utilization balance computation."""
        balance = analyzer_with_data.compute_utilization_balance()

        # Balance should be between 0 and 1
        assert 0.0 <= balance <= 1.0
        assert isinstance(balance, float)

    def test_compute_utilization_balance_perfect(self):
        """Test balance is 1.0 for equal utilization."""
        model = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        memory_mixture = Mock()
        memory_mixture.num_experts = 4
        memory_mixture.expert_memories = [torch.randn(8, 768) for _ in range(4)]
        model.memory_mixture = memory_mixture

        analyzer = GMMAnalyzer(model=model, device="cpu")
        analyzer.routing_data = [{"routing_probs": [0.25] * 4}]

        # Equal activation counts
        analyzer._activation_counts = torch.ones(4)
        analyzer._total_activations = 4

        balance = analyzer.compute_utilization_balance()

        # Should be very close to 1.0
        assert balance > 0.99

    def test_compute_specialization_score(self, analyzer_with_data):
        """Test composite specialization score computation."""
        scores = analyzer_with_data.compute_specialization_score()

        # Check structure
        assert "routing_entropy" in scores
        assert "expert_diversity" in scores
        assert "utilization_balance" in scores

        # Check all values are valid
        for score in scores.values():
            assert isinstance(score, float)
            assert not np.isnan(score)

    def test_compute_expert_similarity(self, analyzer_with_data):
        """Test expert similarity computation."""
        similarity = analyzer_with_data.compute_expert_similarity(0, 1)

        # Similarity should be between -1 and 1
        assert -1.0 <= similarity <= 1.0
        assert isinstance(similarity, float)

        # Self-similarity should be 1.0
        self_sim = analyzer_with_data.compute_expert_similarity(0, 0)
        assert abs(self_sim - 1.0) < 0.01

    def test_cluster_experts(self, analyzer_with_data):
        """Test expert clustering."""
        linkage_matrix = analyzer_with_data.cluster_experts(method="ward")

        # Linkage matrix should have shape (n-1, 4)
        assert linkage_matrix.shape == (3, 4)
        assert isinstance(linkage_matrix, np.ndarray)

    def test_generate_analysis_report(self, analyzer_with_data):
        """Test comprehensive analysis report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            report = analyzer_with_data.generate_analysis_report(
                output_path=str(output_path),
                include_model_id="test-model",
                dataset_name="test-dataset",
            )

            # Check report structure
            assert "model_id" in report
            assert "num_experts" in report
            assert "evaluation_dataset" in report
            assert "metrics" in report
            assert "expert_activations" in report
            assert "routing_data" in report

            # Verify file was created
            assert output_path.exists()

            # Verify JSON is valid
            with open(output_path) as f:
                loaded_report = json.load(f)
            assert loaded_report == report


class TestVisualizationFunctions:
    """Test suite for visualization functions."""

    def test_plot_routing_heatmap(self, mock_routing_data):
        """Test routing heatmap generation."""
        fig = plot_routing_heatmap(mock_routing_data)

        assert fig is not None
        assert len(fig.axes) == 2  # Main plot + colorbar

        # Close figure
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_routing_heatmap_save(self, mock_routing_data):
        """Test routing heatmap saves to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "heatmap.png"

            fig = plot_routing_heatmap(mock_routing_data, str(output_path))

            assert output_path.exists()

            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_expert_activation_timeline(self, mock_routing_data):
        """Test activation timeline generation."""
        fig = plot_expert_activation_timeline(mock_routing_data)

        assert fig is not None
        assert len(fig.axes) == 1

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_specialization_dendrogram(self):
        """Test dendrogram generation."""
        # Create mock linkage matrix
        linkage_matrix = np.array(
            [
                [0, 1, 0.5, 2],
                [2, 3, 1.0, 2],
                [4, 5, 1.5, 4],
            ]
        )

        fig = plot_specialization_dendrogram(linkage_matrix)

        assert fig is not None
        assert len(fig.axes) == 1

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_expert_utilization_bar(self):
        """Test utilization bar chart generation."""
        activation_freq = {
            "expert_0": 0.28,
            "expert_1": 0.24,
            "expert_2": 0.26,
            "expert_3": 0.22,
        }

        fig = plot_expert_utilization_bar(activation_freq)

        assert fig is not None
        assert len(fig.axes) == 1

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_routing_entropy_distribution(self, mock_routing_data):
        """Test entropy distribution histogram generation."""
        fig = plot_routing_entropy_distribution(mock_routing_data)

        assert fig is not None
        assert len(fig.axes) == 1

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_generate_all_visualizations(self, analyzer_with_data):
        """Test generating all visualizations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_files = generate_all_visualizations(
                analyzer=analyzer_with_data,
                output_dir=tmpdir,
                formats=["png"],
                dpi=100,
            )

            # Check that all expected files were created
            assert "heatmap_png" in saved_files
            assert "timeline_png" in saved_files
            assert "dendrogram_png" in saved_files
            assert "utilization_png" in saved_files
            assert "entropy_png" in saved_files

            # Verify files exist
            for file_path in saved_files.values():
                assert Path(file_path).exists()


@pytest.mark.integration
class TestIntegrationWithModel:
    """Integration tests with actual model components."""

    def test_analyzer_with_mock_forward_pass(self, mock_model):
        """Test analyzer tracking with mock model forward pass."""
        # Create analyzer
        analyzer = GMMAnalyzer(model=mock_model, device="cpu")

        # Mock model outputs with routing_info dict (matching real model interface)
        mock_outputs = {
            "routing_info": {
                "routing_probs": torch.tensor(
                    [
                        [0.4, 0.3, 0.2, 0.1],
                        [0.1, 0.2, 0.3, 0.4],
                    ]
                )
            }
        }

        mock_model.return_value = mock_outputs

        # Create mock dataloader
        mock_batch = {
            "input_ids": torch.zeros(2, 10, dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "doc_id": ["doc_0", "doc_1"],
        }

        mock_dataloader = [mock_batch]

        # Track routing
        summary = analyzer.track_routing(mock_dataloader)

        # Verify tracking worked
        assert summary["segments_processed"] == 2
        assert len(analyzer.routing_data) == 2
        assert summary["num_experts"] == 4

    def test_analyzer_with_real_gmmxlnet_model(self):
        """Test analyzer with actual GMMXLNetForQA instance (not mock)."""
        from transformers import XLNetConfig, XLNetForQuestionAnsweringSimple

        from gmmxlnet.models import GMMXLNetForQA

        # Create toy base XLNet model
        xlnet_config = XLNetConfig(
            vocab_size=1000,
            d_model=128,  # Small for fast testing
            n_layer=2,
            n_head=4,
            d_inner=512,
        )
        base_model = XLNetForQuestionAnsweringSimple(xlnet_config)

        # Initialize GMMXLNetForQA with toy base model
        model = GMMXLNetForQA(
            base_model=base_model,
            num_experts=4,
            memory_slots=8,
            routing_mode="write-based",
        )
        model.eval()

        # Create analyzer
        analyzer = GMMAnalyzer(model=model, device="cpu")

        # Create simple dataloader with dummy input
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        # Mock memory token IDs (required for GMM routing)
        mem_read_ids = [10, 11, 12, 13, 14, 15, 16, 17]  # 8 read tokens
        mem_write_ids = [18, 19, 20, 21, 22, 23, 24, 25]  # 8 write tokens

        # Inject memory tokens into input_ids
        for i in range(batch_size):
            # Place write tokens
            for j, mem_id in enumerate(mem_write_ids):
                input_ids[i, 5 + j] = mem_id
            # Place read tokens
            for j, mem_id in enumerate(mem_read_ids):
                input_ids[i, 20 + j] = mem_id

        # Initialize memory state
        memory_state = model.get_initial_memory(batch_size=batch_size, device="cpu")

        mock_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "memory_state": memory_state,
            "mem_read_ids": mem_read_ids,
            "mem_write_ids": mem_write_ids,
            "doc_id": ["doc_0", "doc_1"],
        }
        mock_dataloader = [mock_batch]

        # Track routing - this is the critical test that would fail with API mismatch
        with torch.no_grad():
            summary = analyzer.track_routing(mock_dataloader)

        # Verify tracking worked correctly
        assert summary["segments_processed"] == batch_size
        assert len(analyzer.routing_data) == batch_size
        assert summary["num_experts"] == 4

        # Verify routing data structure
        for routing_entry in analyzer.routing_data:
            assert "routing_probs" in routing_entry
            assert len(routing_entry["routing_probs"]) == 4
            assert "document_id" in routing_entry
            assert "segment_idx" in routing_entry

        # Verify routing probabilities sum to 1 (valid probability distribution)
        for routing_entry in analyzer.routing_data:
            probs_sum = sum(routing_entry["routing_probs"])
            assert abs(probs_sum - 1.0) < 1e-5, f"Routing probs should sum to 1, got {probs_sum}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
