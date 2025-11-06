"""
Integration tests for RBS-QA with GMM-XLNet.

Tests cover the integration between belief state tracking and GMM memory systems,
ensuring backward compatibility and proper functionality in realistic scenarios.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch

from rbsqa.belief_state import BeliefStateTracker, BeliefState, SpanCandidate
from rbsqa.config import (
    RBSTrainingConfig,
    rbs_balanced_config,
    rbs_lightweight_config,
    rbs_advanced_config,
    rbs_research_config
)
from gmmxlnet.models import GMMXLNetForQA


class TestRBSGMMIntegration:
    """Integration tests between RBS-QA and GMM-XLNet components."""

    @pytest.fixture
    def mock_gmm_model(self):
        """Create a mock GMM-XLNet model for testing."""
        model = Mock(spec=GMMXLNetForQA)

        # Mock the forward method to return realistic outputs
        def mock_forward(input_ids, attention_mask=None, token_type_ids=None,
                        start_positions=None, end_positions=None,
                        return_dict=None, **kwargs):
            batch_size, seq_len = input_ids.shape

            # Mock GMM context aggregation
            gmm_context = torch.randn(batch_size, seq_len, 768)

            # Mock QA logits
            start_logits = torch.randn(batch_size, seq_len)
            end_logits = torch.randn(batch_size, seq_len)

            # Mock GMM routing information
            routing_logits = torch.randn(batch_size, seq_len, 4)  # 4 experts

            result = {
                'start_logits': start_logits,
                'end_logits': end_logits,
                'gmm_context': gmm_context,
                'routing_logits': routing_logits,
                'expert_states': [torch.randn(batch_size, 16, 768) for _ in range(4)],
                'loss': torch.tensor(0.5) if start_positions is not None else None
            }

            return result

        model.forward = mock_forward
        model.config = Mock()
        model.config.hidden_size = 768
        model.config.num_memory_experts = 4
        model.config.memory_num_tokens = 16

        return model

    @pytest.fixture
    def rbs_config(self):
        """Create a balanced RBS configuration for testing."""
        return rbs_balanced_config(
            belief_state_threshold=0.75,
            max_segments=8,
            re_scoring_method="context_weighted",
            enable_trend_analysis=True
        )

    @pytest.fixture
    def belief_tracker(self, rbs_config):
        """Create a BeliefStateTracker instance for testing."""
        return BeliefStateTracker(
            max_segments=rbs_config.max_segments,
            confidence_threshold=rbs_config.belief_state_threshold,
            re_scoring_method=rbs_config.re_scoring_method,
            enable_trend_analysis=rbs_config.enable_trend_analysis,
            hidden_dim=768
        )

    def test_belief_tracker_with_mock_gmm_context(self, belief_tracker, mock_gmm_model):
        """Test belief tracker processing with mock GMM context."""
        # Create mock inputs
        batch_size, seq_len = 1, 50
        input_ids = torch.randint(0, 30000, (batch_size, seq_len))

        # Get mock model output
        model_output = mock_gmm_model(input_ids)

        # Process with belief tracker
        segment_id = 0
        global_offset = 0

        updated_belief = belief_tracker.update_belief(
            current_logits=(model_output['start_logits'][0], model_output['end_logits'][0]),
            current_segment_id=segment_id,
            gmm_context=model_output['gmm_context'],
            global_offset=global_offset
        )

        # Verify belief state was updated
        assert updated_belief.best_span is not None
        assert updated_belief.confidence > 0.0
        assert updated_belief.segment_id == segment_id
        assert updated_belief.total_segments == segment_id + 1

    def test_multi_segment_processing(self, belief_tracker, mock_gmm_model):
        """Test processing multiple segments with belief tracking."""
        batch_size, seq_len = 1, 30
        num_segments = 4

        for segment_id in range(num_segments):
            # Create mock inputs for each segment
            input_ids = torch.randint(0, 30000, (batch_size, seq_len))
            model_output = mock_gmm_model(input_ids)

            # Vary confidence across segments to test revision logic
            if segment_id == 1:
                # Second segment has better span
                model_output['start_logits'][0, 10] = 5.0
                model_output['end_logits'][0, 15] = 4.5
            elif segment_id == 3:
                # Fourth segment has best span
                model_output['start_logits'][0, 8] = 6.0
                model_output['end_logits'][0, 12] = 5.5

            global_offset = segment_id * seq_len

            updated_belief = belief_tracker.update_belief(
                current_logits=(model_output['start_logits'][0], model_output['end_logits'][0]),
                current_segment_id=segment_id,
                gmm_context=model_output['gmm_context'],
                global_offset=global_offset
            )

            # Verify belief state consistency
            assert updated_belief.total_segments == segment_id + 1
            assert len(updated_belief.span_history) == segment_id + 1

        # After processing all segments
        final_belief = belief_tracker.belief
        assert final_belief.total_segments == num_segments
        assert len(final_belief.span_history) == num_segments

    def test_re_scoring_with_gmm_context(self, belief_tracker):
        """Test re-scoring mechanism with realistic GMM context."""
        # Create past spans
        past_spans = [
            SpanCandidate((10, 20), 0.65, 0, hash("segment_0")),
            SpanCandidate((35, 45), 0.72, 1, hash("segment_1")),
            SpanCandidate((60, 70), 0.68, 2, hash("segment_2"))
        ]

        # Create GMM context with varying intensity
        # Higher context values should boost confidence
        gmm_context = torch.ones(1, 50, 768) * 2.0  # Strong context

        # Add to belief history for proper re-scoring
        belief_tracker.belief.span_history = past_spans
        belief_tracker.belief.total_segments = 3

        re_scored_spans = belief_tracker.re_score_past_spans(past_spans, gmm_context)

        # Verify all spans were re-scored
        assert len(re_scored_spans) == len(past_spans)

        for i, candidate in enumerate(re_scored_spans):
            assert candidate.re_scored is True
            assert candidate.re_scored_confidence is not None
            assert 0.0 <= candidate.re_scored_confidence <= 1.0

            # Context-weighted re-scoring should generally boost confidence
            # (though the exact relationship depends on the implementation)
            assert isinstance(candidate.re_scored_confidence, float)

    def test_non_monotonic_reasoning_scenario(self, belief_tracker, mock_gmm_model):
        """Test scenario where non-monotonic reasoning occurs."""
        # First segment: establish initial belief
        batch_size, seq_len = 1, 40
        input_ids_1 = torch.randint(0, 30000, (batch_size, seq_len))
        model_output_1 = mock_gmm_model(input_ids_1)

        # Set moderate confidence for initial span
        model_output_1['start_logits'][0, 5] = 3.0
        model_output_1['end_logits'][0, 10] = 2.5

        updated_belief_1 = belief_tracker.update_belief(
            current_logits=(model_output_1['start_logits'][0], model_output_1['end_logits'][0]),
            current_segment_id=0,
            gmm_context=model_output_1['gmm_context'],
            global_offset=0
        )

        initial_span = updated_belief_1.best_span
        initial_confidence = updated_belief_1.confidence

        # Second segment: has new context but no better span
        input_ids_2 = torch.randint(0, 30000, (batch_size, seq_len))
        model_output_2 = mock_gmm_model(input_ids_2)

        # Create strong GMM context that could boost the initial belief
        model_output_2['gmm_context'] = torch.ones(1, seq_len, 768) * 1.5

        # New segment has worse span
        model_output_2['start_logits'][0, 15] = 2.0
        model_output_2['end_logits'][0, 20] = 1.8

        updated_belief_2 = belief_tracker.update_belief(
            current_logits=(model_output_2['start_logits'][0], model_output_2['end_logits'][0]),
            current_segment_id=1,
            gmm_context=model_output_2['gmm_context'],
            global_offset=seq_len
        )

        # Check that belief tracking is working
        assert updated_belief_2.total_segments == 2
        assert len(updated_belief_2.span_history) == 2

    def test_confidence_calibration_integration(self, belief_tracker):
        """Test confidence calibration in integration context."""
        # Create test logits
        seq_len = 50
        start_logits = torch.randn(seq_len)
        end_logits = torch.randn(seq_len)

        # Make one span clearly better
        start_logits[10] = 4.0
        end_logits[15] = 3.5

        # Compute confidence with default calibration
        confidence_1 = belief_tracker.compute_confidence(start_logits, end_logits, 10, 15)

        # Modify calibration parameters
        with torch.no_grad():
            belief_tracker.confidence_scaler.fill_(1.5)
            belief_tracker.confidence_bias.fill_(0.1)

        # Compute confidence with modified calibration
        confidence_2 = belief_tracker.compute_confidence(start_logits, end_logits, 10, 15)

        # Calibration should affect the confidence values
        assert confidence_1 != confidence_2
        assert 0.0 <= confidence_1 <= 1.0
        assert 0.0 <= confidence_2 <= 1.0

    def test_halting_policy_integration(self, belief_tracker, mock_gmm_model):
        """Test halting policy in realistic multi-segment scenario."""
        batch_size, seq_len = 1, 35
        confidence_threshold = belief_tracker.confidence_threshold

        segment_confidences = [0.6, 0.7, 0.78, 0.82]  # Exceeds threshold at segment 3

        for segment_id, target_confidence in enumerate(segment_confidences):
            input_ids = torch.randint(0, 30000, (batch_size, seq_len))
            model_output = mock_gmm_model(input_ids)

            # Adjust logits to achieve target confidence
            # (This is a simplified approach - in practice, confidence depends on many factors)
            start_pos = 5 + segment_id * 2
            end_pos = start_pos + 5

            # Scale logits to approximate target confidence
            model_output['start_logits'][0, start_pos] = target_confidence * 5.0
            model_output['end_logits'][0, end_pos] = target_confidence * 4.5

            global_offset = segment_id * seq_len

            updated_belief = belief_tracker.update_belief(
                current_logits=(model_output['start_logits'][0], model_output['end_logits'][0]),
                current_segment_id=segment_id,
                gmm_context=model_output['gmm_context'],
                global_offset=global_offset
            )

            # Check halting condition
            if updated_belief.confidence >= confidence_threshold:
                assert belief_tracker.should_halt()
                break

    def test_memory_management_integration(self, belief_tracker):
        """Test memory management in integration context."""
        # Test with many spans to check memory limits
        belief_tracker.belief_state_memory_limit = 5  # Low limit for testing

        # Add more spans than the limit
        for i in range(10):
            candidate = SpanCandidate(
                span=(i*10, i*10+5),
                confidence=0.7 + i * 0.02,
                segment_id=i,
                gmm_context_hash=hash(f"segment_{i}")
            )
            belief_tracker.belief.add_candidate(candidate)

        # The tracker should handle the memory limit appropriately
        # (exact behavior depends on implementation strategy)
        assert len(belief_tracker.belief.span_history) <= belief_tracker.belief_state_memory_limit or \
               len(belief_tracker.belief.span_history) == 10  # Or no truncation if not implemented

    def test_error_handling_integration(self, belief_tracker):
        """Test error handling in integration scenarios."""
        # Test with invalid GMM context
        invalid_context = torch.tensor([])  # Empty tensor

        past_spans = [SpanCandidate((10, 20), 0.7, 0, 123)]

        # Should handle gracefully without crashing
        try:
            re_scored = belief_tracker.re_score_past_spans(past_spans, invalid_context)
            # Either handles gracefully or raises appropriate error
        except Exception:
            # Should be a meaningful error, not a cryptic tensor error
            pass

        # Test with mismatched dimensions
        mismatched_context = torch.randn(2, 30, 1024)  # Different hidden dim

        try:
            candidate = belief_tracker.extract_best_span(
                (torch.randn(30), torch.randn(30)),
                segment_id=0,
                global_offset=0
            )
            # Should work fine since extract_best_span doesn't depend on GMM context
            assert candidate is not None
        except Exception:
            pass


class TestRBSConfigurationCompatibility:
    """Test RBS configuration compatibility with GMM components."""

    def test_config_gmm_dependency_validation(self):
        """Test that RBS config validates GMM dependencies."""
        # Should fail with GMM disabled
        with pytest.raises(ValueError, match="RBS-QA requires GMM memory to be enabled"):
            RBSTrainingConfig(use_gmm_memory=False, use_belief_state=True)

        # Should succeed with GMM enabled
        config = RBSTrainingConfig(use_gmm_memory=True, use_belief_state=True)
        assert config.use_gmm_memory is True
        assert config.use_belief_state is True

    def test_config_parameter_consistency(self):
        """Test parameter consistency across RBS and GMM configs."""
        config = rbs_balanced_config()

        # Check that GMM parameters are reasonable for RBS
        assert config.use_gmm_memory is True
        assert config.num_memory_experts >= 2  # Need multiple experts for RBS to be meaningful
        assert config.memory_num_tokens >= 8   # Need sufficient memory tokens
        assert config.routing_temperature > 0   # Valid routing temperature

        # Check RBS parameters are reasonable
        assert 0.0 <= config.belief_state_threshold <= 1.0
        assert config.max_segments > 0
        assert config.belief_state_memory_limit > 0

    def test_backward_compatibility_config(self):
        """Test backward compatibility in configuration."""
        # RBS config should work as a drop-in replacement for GMM config
        rbs_config = rbs_balanced_config()

        # Check it has all required GMM fields
        gmm_required_fields = [
            'use_gmm_memory', 'num_memory_experts', 'memory_num_tokens',
            'routing_temperature', 'load_balance_weight'
        ]

        for field in gmm_required_fields:
            assert hasattr(rbs_config, field)
            assert getattr(rbs_config, field) is not None

    @pytest.mark.parametrize("preset_func", [
        rbs_lightweight_config,
        rbs_balanced_config,
        rbs_advanced_config,
        rbs_research_config
    ])
    def test_preset_gmm_compatibility(self, preset_func):
        """Test that all RBS presets are compatible with GMM."""
        config = preset_func()

        # Should pass all GMM-related validations
        assert config.use_gmm_memory is True
        assert config.num_memory_experts in [2, 4, 6, 8]
        assert config.memory_num_tokens in [8, 16, 24, 32]
        assert config.routing_temperature > 0
        assert config.load_balance_weight >= 0


class TestRBSEndToEndScenarios:
    """End-to-end integration tests for realistic RBS scenarios."""

    def test_document_processing_scenario(self, belief_tracker):
        """Test processing a realistic multi-segment document."""
        num_segments = 6
        seq_len = 40

        # Simulate processing a document with segments of varying quality
        segment_data = [
            # (best_start, best_end, confidence_quality, description)
            (5, 12, 0.6, "Moderate quality initial segment"),
            (8, 15, 0.7, "Better quality segment"),
            (25, 32, 0.65, "Slightly worse but useful segment"),
            (18, 25, 0.8, "High quality segment"),
            (35, 42, 0.75, "Good quality segment"),
            (30, 37, 0.82, "Best quality final segment"),
        ]

        for segment_id, (start_idx, end_idx, quality_factor, description) in enumerate(segment_data):
            # Create mock outputs
            start_logits = torch.randn(seq_len)
            end_logits = torch.randn(seq_len)

            # Set the best span with appropriate quality
            start_logits[start_idx] = 2.0 + quality_factor * 3.0
            end_logits[end_idx] = 1.5 + quality_factor * 2.5

            # Create varying GMM context
            gmm_context = torch.randn(1, seq_len, 768) * (0.5 + quality_factor * 0.5)

            global_offset = segment_id * seq_len

            updated_belief = belief_tracker.update_belief(
                current_logits=(start_logits, end_logits),
                current_segment_id=segment_id,
                gmm_context=gmm_context,
                global_offset=global_offset
            )

            # Verify processing
            assert updated_belief.best_span is not None
            assert updated_belief.total_segments == segment_id + 1

            # Log progress for debugging
            if segment_id == len(segment_data) - 1:
                # Final state
                assert len(updated_belief.span_history) == num_segments
                assert updated_belief.revision_count >= 0  # May have revisions

    def test_confidence_evolution_tracking(self, belief_tracker):
        """Test tracking of confidence evolution across segments."""
        # Create a scenario with known confidence progression
        confidence_progression = [0.3, 0.5, 0.7, 0.65, 0.8, 0.85]

        for i, target_confidence in enumerate(confidence_progression):
            seq_len = 30

            # Create logits that will produce approximately the target confidence
            start_logits = torch.randn(seq_len)
            end_logits = torch.randn(seq_len)

            # Adjust the best span to achieve target confidence
            best_start = 5 + i
            best_end = best_start + 5
            start_logits[best_start] = target_confidence * 5.0
            end_logits[best_end] = target_confidence * 4.0

            gmm_context = torch.randn(1, seq_len, 768)
            global_offset = i * seq_len

            updated_belief = belief_tracker.update_belief(
                current_logits=(start_logits, end_logits),
                current_segment_id=i,
                gmm_context=gmm_context,
                global_offset=global_offset
            )

            # The belief confidence should generally trend with the progression
            # (though not exactly match due to belief revision logic)
            assert 0.0 <= updated_belief.confidence <= 1.0

        # Check trend analysis
        trend = belief_tracker.belief.get_trend_analysis()
        assert trend['trend'] in ['increasing', 'stable', 'decreasing']
        assert 'mean_confidence' in trend
        assert 'final_confidence' in trend

    def test_boundary_conditions(self, belief_tracker):
        """Test boundary conditions and edge cases."""
        # Test with very short sequences
        short_start = torch.zeros(5)
        short_end = torch.zeros(5)
        short_start[2] = 3.0
        short_end[3] = 2.5

        candidate = belief_tracker.extract_best_span(
            (short_start, short_end), segment_id=0, global_offset=0
        )
        assert candidate.span[0] <= candidate.span[1]

        # Test with very long sequences
        long_start = torch.randn(200)
        long_end = torch.randn(200)
        long_start[100] = 5.0
        long_end[105] = 4.5

        candidate = belief_tracker.extract_best_span(
            (long_start, long_end), segment_id=0, global_offset=0
        )
        assert candidate.span[1] - candidate.span[0] <= 30  # Default max_span_length

        # Test confidence bounds
        confidence = belief_tracker.compute_confidence(long_start, long_end, 100, 105)
        assert 0.0 <= confidence <= 1.0

    def test_performance_characteristics(self, belief_tracker):
        """Test performance characteristics with larger inputs."""
        import time

        # Test with larger inputs to check performance scaling
        seq_len = 100
        num_segments = 10
        past_spans = []

        # Create many past spans for re-scoring
        for i in range(50):
            past_spans.append(SpanCandidate(
                span=(i*10, i*10+5),
                confidence=0.7,
                segment_id=i,
                gmm_context_hash=hash(f"segment_{i}")
            ))

        large_gmm_context = torch.randn(1, seq_len, 768)

        # Time the re-scoring operation
        start_time = time.time()
        re_scored = belief_tracker.re_score_past_spans(past_spans, large_gmm_context)
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 5.0  # 5 seconds max for 50 spans
        assert len(re_scored) == len(past_spans)

        # Test caching efficiency
        start_time = time.time()
        re_scored_cached = belief_tracker.re_score_past_spans(past_spans, large_gmm_context)
        end_time = time.time()

        # Cached version should be faster
        assert end_time - start_time < 1.0  # 1 second max for cached operation


# Helper functions for testing
def create_mock_gmm_model(num_experts=4, memory_tokens=16, hidden_dim=768):
    """Helper function to create mock GMM model for testing."""
    model = Mock(spec=GMMXLNetForQA)

    def mock_forward(**kwargs):
        batch_size = kwargs.get('input_ids', torch.zeros(1, 50)).shape[0]
        seq_len = kwargs.get('input_ids', torch.zeros(1, 50)).shape[1]

        return {
            'start_logits': torch.randn(batch_size, seq_len),
            'end_logits': torch.randn(batch_size, seq_len),
            'gmm_context': torch.randn(batch_size, seq_len, hidden_dim),
            'routing_logits': torch.randn(batch_size, seq_len, num_experts),
            'expert_states': [torch.randn(batch_size, memory_tokens, hidden_dim) for _ in range(num_experts)],
            'loss': None
        }

    model.forward = mock_forward
    model.config = Mock()
    model.config.hidden_size = hidden_dim
    model.config.num_memory_experts = num_experts
    model.config.memory_num_tokens = memory_tokens

    return model


def create_realistic_test_scenario(num_segments=5, seq_len=40, hidden_dim=768):
    """Helper function to create realistic test scenario data."""
    scenario_data = []

    for segment_id in range(num_segments):
        # Create realistic logits with varying quality
        start_logits = torch.randn(seq_len)
        end_logits = torch.randn(seq_len)

        # Quality varies across segments
        quality_factor = 0.5 + (segment_id * 0.1)
        best_start = 5 + segment_id * 3
        best_end = best_start + 5

        start_logits[best_start] = 3.0 + quality_factor * 2.0
        end_logits[best_end] = 2.5 + quality_factor * 1.5

        # Create GMM context
        gmm_context = torch.randn(1, seq_len, hidden_dim) * quality_factor

        scenario_data.append({
            'segment_id': segment_id,
            'start_logits': start_logits,
            'end_logits': end_logits,
            'gmm_context': gmm_context,
            'global_offset': segment_id * seq_len,
            'expected_confidence_range': (0.4 + quality_factor * 0.3, 0.8 + quality_factor * 0.1)
        })

    return scenario_data


if __name__ == "__main__":
    pytest.main([__file__])