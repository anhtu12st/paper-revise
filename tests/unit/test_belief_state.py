"""
Unit tests for RBS-QA Belief State Tracker components.

Tests cover BeliefState, SpanCandidate, and BeliefStateTracker classes
with comprehensive validation of functionality and edge cases.
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Tuple

from rbsqa.belief_state import (
    BeliefState,
    SpanCandidate,
    BeliefStateTracker
)
from rbsqa.config import RBSTrainingConfig


class TestSpanCandidate:
    """Test cases for SpanCandidate data structure."""

    def test_span_candidate_initialization_valid(self):
        """Test valid SpanCandidate initialization."""
        candidate = SpanCandidate(
            span=(10, 20),
            confidence=0.85,
            segment_id=2,
            gmm_context_hash=12345
        )

        assert candidate.span == (10, 20)
        assert candidate.confidence == 0.85
        assert candidate.segment_id == 2
        assert candidate.gmm_context_hash == 12345
        assert not candidate.re_scored
        assert candidate.re_scored_confidence is None

    def test_span_candidate_invalid_confidence(self):
        """Test SpanCandidate with invalid confidence values."""
        # Test confidence > 1.0
        with pytest.raises(ValueError, match="Confidence must be in \\[0.0, 1.0\\]"):
            SpanCandidate(span=(10, 20), confidence=1.5, segment_id=0, gmm_context_hash=123)

        # Test confidence < 0.0
        with pytest.raises(ValueError, match="Confidence must be in \\[0.0, 1.0\\]"):
            SpanCandidate(span=(10, 20), confidence=-0.1, segment_id=0, gmm_context_hash=123)

    def test_span_candidate_invalid_span(self):
        """Test SpanCandidate with invalid span."""
        # Test start > end
        with pytest.raises(ValueError, match="Invalid span: start .* > end .*"):
            SpanCandidate(span=(20, 10), confidence=0.5, segment_id=0, gmm_context_hash=123)

    def test_span_candidate_invalid_segment_id(self):
        """Test SpanCandidate with invalid segment ID."""
        with pytest.raises(ValueError, match="Segment ID must be non-negative"):
            SpanCandidate(span=(10, 20), confidence=0.5, segment_id=-1, gmm_context_hash=123)

    def test_span_candidate_to_dict(self):
        """Test SpanCandidate serialization."""
        candidate = SpanCandidate(
            span=(15, 25),
            confidence=0.75,
            segment_id=3,
            gmm_context_hash=54321,
            start_logits=2.5,
            end_logits=3.2,
            re_scored=True,
            re_scored_confidence=0.80
        )

        result = candidate.to_dict()
        expected = {
            'span': (15, 25),
            'confidence': 0.75,
            'segment_id': 3,
            'gmm_context_hash': 54321,
            'start_logits': 2.5,
            'end_logits': 3.2,
            're_scored': True,
            're_scored_confidence': 0.80
        }

        assert result == expected


class TestBeliefState:
    """Test cases for BeliefState data structure."""

    def test_belief_state_initialization_default(self):
        """Test BeliefState initialization with defaults."""
        belief = BeliefState()

        assert belief.best_span is None
        assert belief.confidence == 0.0
        assert belief.segment_id == -1
        assert belief.span_history == []
        assert belief.confidence_history == []
        assert belief.revision_count == 0
        assert belief.total_segments == 0
        assert belief.gmm_context_hashes == []

    def test_belief_state_initialization_custom(self):
        """Test BeliefState initialization with custom values."""
        belief = BeliefState(
            best_span=(50, 60),
            confidence=0.9,
            segment_id=5,
            revision_count=2,
            total_segments=6
        )

        assert belief.best_span == (50, 60)
        assert belief.confidence == 0.9
        assert belief.segment_id == 5
        assert belief.revision_count == 2
        assert belief.total_segments == 6

    def test_belief_state_invalid_confidence(self):
        """Test BeliefState with invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be in \\[0.0, 1.0\\]"):
            BeliefState(confidence=1.5)

        with pytest.raises(ValueError, match="Confidence must be in \\[0.0, 1.0\\]"):
            BeliefState(confidence=-0.1)

    def test_belief_state_invalid_segment_id(self):
        """Test BeliefState with invalid segment ID."""
        with pytest.raises(ValueError, match="Segment ID must be >= -1"):
            BeliefState(segment_id=-2)

    def test_add_candidate(self):
        """Test adding candidates to belief history."""
        belief = BeliefState()
        candidate1 = SpanCandidate((10, 20), 0.8, 1, 123)
        candidate2 = SpanCandidate((30, 40), 0.9, 2, 456)

        belief.add_candidate(candidate1)
        belief.add_candidate(candidate2)

        assert len(belief.span_history) == 2
        assert len(belief.confidence_history) == 2
        assert belief.span_history[0] == candidate1
        assert belief.span_history[1] == candidate2
        assert belief.confidence_history == [0.8, 0.9]

    def test_update_best_span_new(self):
        """Test updating best span for the first time."""
        belief = BeliefState()
        belief.update_best_span((15, 25), 0.85, 3)

        assert belief.best_span == (15, 25)
        assert belief.confidence == 0.85
        assert belief.segment_id == 3
        assert belief.revision_count == 0  # First update is not a revision

    def test_update_best_span_revision(self):
        """Test updating best span with revision."""
        belief = BeliefState()
        belief.update_best_span((10, 20), 0.7, 1)  # Initial
        belief.update_best_span((30, 40), 0.8, 3)  # Revision (different segment)

        assert belief.best_span == (30, 40)
        assert belief.confidence == 0.8
        assert belief.segment_id == 3
        assert belief.revision_count == 1

    def test_update_best_span_no_revision_small_change(self):
        """Test updating best span without revision (small change)."""
        belief = BeliefState()
        belief.update_best_span((10, 20), 0.7, 1)  # Initial
        belief.update_best_span((12, 22), 0.75, 1)  # Same segment, small change

        assert belief.best_span == (12, 22)
        assert belief.confidence == 0.75
        assert belief.segment_id == 1
        assert belief.revision_count == 0  # Small change, no revision

    def test_get_trend_analysis_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        belief = BeliefState()

        # No data
        trend = belief.get_trend_analysis()
        assert trend['trend'] == 'stable'
        assert trend['slope'] == 0.0
        assert trend['variance'] == 0.0

        # Single data point
        belief.add_candidate(SpanCandidate((10, 20), 0.5, 0, 123))
        trend = belief.get_trend_analysis()
        assert trend['trend'] == 'stable'
        assert trend['slope'] == 0.0
        assert trend['variance'] == 0.0

    def test_get_trend_analysis_multiple_data(self):
        """Test trend analysis with multiple data points."""
        belief = BeliefState()

        # Add increasing confidence values
        confidences = [0.3, 0.5, 0.7, 0.9]
        for i, conf in enumerate(confidences):
            belief.add_candidate(SpanCandidate((10+i*10, 15+i*10), conf, i, 123+i))

        trend = belief.get_trend_analysis()
        assert trend['trend'] == 'increasing'
        assert trend['slope'] > 0
        assert trend['mean_confidence'] == sum(confidences) / len(confidences)
        assert trend['final_confidence'] == confidences[-1]

    def test_get_trend_analysis_decreasing(self):
        """Test trend analysis with decreasing confidence."""
        belief = BeliefState()

        # Add decreasing confidence values
        confidences = [0.9, 0.7, 0.5, 0.3]
        for i, conf in enumerate(confidences):
            belief.add_candidate(SpanCandidate((10+i*10, 15+i*10), conf, i, 123+i))

        trend = belief.get_trend_analysis()
        assert trend['trend'] == 'decreasing'
        assert trend['slope'] < 0

    def test_to_dict(self):
        """Test BeliefState serialization."""
        belief = BeliefState(
            best_span=(100, 110),
            confidence=0.95,
            segment_id=7,
            revision_count=3,
            total_segments=8
        )

        candidate = SpanCandidate((50, 60), 0.8, 2, 456)
        belief.add_candidate(candidate)
        belief.gmm_context_hashes = [123, 456, 789]

        result = belief.to_dict()

        assert result['best_span'] == (100, 110)
        assert result['confidence'] == 0.95
        assert result['segment_id'] == 7
        assert result['revision_count'] == 3
        assert result['total_segments'] == 8
        assert len(result['span_history']) == 1
        assert result['confidence_history'] == [0.8]
        assert result['gmm_context_hashes'] == [123, 456, 789]
        assert 'trend_analysis' in result


class TestBeliefStateTracker:
    """Test cases for BeliefStateTracker class."""

    def test_tracker_initialization_valid(self):
        """Test valid BeliefStateTracker initialization."""
        tracker = BeliefStateTracker(
            max_segments=16,
            confidence_threshold=0.8,
            re_scoring_method="context_weighted",
            enable_trend_analysis=True
        )

        assert tracker.max_segments == 16
        assert tracker.confidence_threshold == 0.8
        assert tracker.re_scoring_method == "context_weighted"
        assert tracker.enable_trend_analysis == True
        assert isinstance(tracker.confidence_scaler, nn.Parameter)
        assert isinstance(tracker.confidence_bias, nn.Parameter)
        assert tracker.belief.best_span is None
        assert tracker.belief.confidence == 0.0

    def test_tracker_initialization_invalid_parameters(self):
        """Test BeliefStateTracker initialization with invalid parameters."""
        # Invalid max_segments
        with pytest.raises(ValueError, match="max_segments must be positive"):
            BeliefStateTracker(max_segments=0)

        with pytest.raises(ValueError, match="max_segments must be positive"):
            BeliefStateTracker(max_segments=-5)

        # Invalid confidence_threshold
        with pytest.raises(ValueError, match="confidence_threshold must be in \\[0.0, 1.0\\]"):
            BeliefStateTracker(confidence_threshold=1.5)

        with pytest.raises(ValueError, match="confidence_threshold must be in \\[0.0, 1.0\\]"):
            BeliefStateTracker(confidence_threshold=-0.1)

        # Invalid re_scoring_method
        with pytest.raises(ValueError, match="Unknown re_scoring_method"):
            BeliefStateTracker(re_scoring_method="invalid_method")

    def test_tracker_with_learnable_re_scoring(self):
        """Test BeliefStateTracker with learnable re-scoring."""
        tracker = BeliefStateTracker(
            re_scoring_method="learned",
            enable_learnable_re_scoring=True,
            hidden_dim=256
        )

        assert tracker.re_score_network is not None
        assert isinstance(tracker.re_score_network, nn.Sequential)

    def test_reset_belief(self):
        """Test belief state reset functionality."""
        tracker = BeliefStateTracker()

        # Set some initial state
        tracker.belief.best_span = (10, 20)
        tracker.belief.confidence = 0.8
        tracker.belief.revision_count = 2
        tracker.re_scoring_cache['test'] = 1.0

        # Reset
        tracker.reset_belief()

        assert tracker.belief.best_span is None
        assert tracker.belief.confidence == 0.0
        assert tracker.belief.segment_id == -1
        assert tracker.belief.revision_count == 0
        assert tracker.belief.total_segments == 0
        assert len(tracker.re_scoring_cache) == 0

    def test_extract_best_span(self):
        """Test span extraction from logits."""
        tracker = BeliefStateTracker()

        # Create mock logits with clear best indices
        seq_len = 50
        start_logits = torch.zeros(seq_len)
        end_logits = torch.zeros(seq_len)
        start_logits[10] = 5.0  # Best start
        end_logits[15] = 4.0    # Best end

        candidate = tracker.extract_best_span(
            (start_logits, end_logits),
            segment_id=2,
            global_offset=100
        )

        assert candidate.span == (110, 115)  # Global indices
        assert candidate.segment_id == 2
        assert 0.0 <= candidate.confidence <= 1.0
        assert candidate.start_logits == 5.0
        assert candidate.end_logits == 4.0

    def test_extract_best_span_invalid_span_correction(self):
        """Test span extraction with automatic correction of invalid spans."""
        tracker = BeliefStateTracker()

        # Create logits where end < start
        seq_len = 50
        start_logits = torch.zeros(seq_len)
        end_logits = torch.zeros(seq_len)
        start_logits[20] = 5.0  # Best start is after end
        end_logits[10] = 4.0    # Best end is before start

        candidate = tracker.extract_best_span(
            (start_logits, end_logits),
            segment_id=0,
            global_offset=0
        )

        # Should be corrected to start <= end
        assert candidate.span[0] <= candidate.span[1]

    def test_extract_best_span_max_length_limit(self):
        """Test span extraction with maximum length limit."""
        tracker = BeliefStateTracker()

        seq_len = 100
        start_logits = torch.zeros(seq_len)
        end_logits = torch.zeros(seq_len)
        start_logits[10] = 5.0
        end_logits[80] = 4.0  # Would create span of length 70

        candidate = tracker.extract_best_span(
            (start_logits, end_logits),
            segment_id=0,
            global_offset=0,
            max_span_length=20
        )

        # Span should be limited to max_length
        assert candidate.span[1] - candidate.span[0] <= 20

    def test_compute_confidence(self):
        """Test confidence computation from logits."""
        tracker = BeliefStateTracker()

        seq_len = 20
        start_logits = torch.randn(seq_len)
        end_logits = torch.randn(seq_len)
        start_idx, end_idx = 5, 10

        confidence = tracker.compute_confidence(start_logits, end_logits, start_idx, end_idx)

        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)

    def test_compute_confidence_different_inputs(self):
        """Test confidence computation with different input types."""
        tracker = BeliefStateTracker()

        # Test with list inputs
        seq_len = 10
        start_logits = torch.randn(seq_len).tolist()
        end_logits = torch.randn(seq_len).tolist()
        start_idx, end_idx = 2, 5

        confidence = tracker.compute_confidence(start_logits, end_logits, start_idx, end_idx)
        assert 0.0 <= confidence <= 1.0

    def test_re_score_past_spans_context_weighted(self):
        """Test re-scoring with context_weighted method."""
        tracker = BeliefStateTracker(re_scoring_method="context_weighted")

        # Create past candidates
        past_spans = [
            SpanCandidate((10, 20), 0.7, 0, 123),
            SpanCandidate((30, 40), 0.6, 1, 456)
        ]

        # Create mock GMM context
        gmm_context = torch.randn(1, 50, 768)  # batch=1, seq_len=50, hidden_dim=768

        re_scored = tracker.re_score_past_spans(past_spans, gmm_context)

        assert len(re_scored) == 2
        for candidate in re_scored:
            assert candidate.re_scored
            assert candidate.re_scored_confidence is not None
            assert 0.0 <= candidate.re_scored_confidence <= 1.0

    def test_re_score_past_spans_exponential_decay(self):
        """Test re-scoring with exponential_decay method."""
        tracker = BeliefStateTracker(re_scoring_method="exponential_decay")

        past_spans = [
            SpanCandidate((10, 20), 0.8, 0, 123),  # Older span
            SpanCandidate((30, 40), 0.7, 3, 456)   # More recent span
        ]

        gmm_context = torch.randn(1, 50, 768)

        # Add some span history to set the current segment
        tracker.belief.total_segments = 4

        re_scored = tracker.re_score_past_spans(past_spans, gmm_context)

        # Older span should be decayed more
        assert re_scored[0].re_scored_confidence < re_scored[1].re_scored_confidence

    def test_re_score_caching(self):
        """Test that re-scoring uses caching efficiently."""
        tracker = BeliefStateTracker(re_scoring_method="context_weighted")

        past_spans = [SpanCandidate((10, 20), 0.7, 0, 123)]
        gmm_context = torch.randn(1, 50, 768)

        # First call
        re_scored_1 = tracker.re_score_past_spans(past_spans, gmm_context)
        cache_size_1 = len(tracker.re_scoring_cache)

        # Second call with same inputs
        re_scored_2 = tracker.re_score_past_spans(past_spans, gmm_context)
        cache_size_2 = len(tracker.re_scoring_cache)

        # Cache should not grow (should reuse)
        assert cache_size_1 == cache_size_2
        assert re_scored_1[0].re_scored_confidence == re_scored_2[0].re_scored_confidence

    def test_should_halt_confidence_threshold(self):
        """Test halting decision based on confidence threshold."""
        tracker = BeliefStateTracker(confidence_threshold=0.8)

        # Below threshold
        tracker.belief.confidence = 0.7
        assert not tracker.should_halt()

        # Above threshold
        tracker.belief.confidence = 0.85
        assert tracker.should_halt()

    def test_should_halt_max_segments_no_revision(self):
        """Test halting when max segments reached without revisions."""
        tracker = BeliefStateTracker(max_segments=10, confidence_threshold=0.9)

        tracker.belief.total_segments = 10
        tracker.belief.confidence = 0.6  # Below threshold
        tracker.belief.revision_count = 0  # No revisions

        assert tracker.should_halt()

    def test_should_halt_stable_trend(self):
        """Test halting with stable confidence trend."""
        tracker = BeliefStateTracker(
            confidence_threshold=0.8,
            enable_trend_analysis=True
        )

        # Set confidence just below threshold but stable
        tracker.belief.confidence = 0.75
        tracker.belief.total_segments = 5

        # Create stable history
        stable_confidences = [0.74, 0.75, 0.76, 0.75, 0.74]
        for i, conf in enumerate(stable_confidences):
            candidate = SpanCandidate((10, 15+i), conf, i, 123+i)
            tracker.belief.add_candidate(candidate)

        assert tracker.should_halt()

    def test_update_belief_first_segment(self):
        """Test belief update for first segment."""
        tracker = BeliefStateTracker()

        # Create mock logits
        start_logits = torch.zeros(50)
        end_logits = torch.zeros(50)
        start_logits[10] = 5.0
        end_logits[15] = 4.0

        gmm_context = torch.randn(1, 50, 768)

        updated_belief = tracker.update_belief(
            (start_logits, end_logits),
            current_segment_id=0,
            gmm_context=gmm_context,
            global_offset=0
        )

        assert updated_belief.best_span is not None
        assert updated_belief.confidence > 0.0
        assert updated_belief.segment_id == 0
        assert updated_belief.total_segments == 1
        assert updated_belief.revision_count == 0
        assert len(updated_belief.span_history) == 1

    def test_update_belief_with_revision(self):
        """Test belief update that causes a revision."""
        tracker = BeliefStateTracker()

        # First segment
        start_logits_1 = torch.zeros(50)
        end_logits_1 = torch.zeros(50)
        start_logits_1[5] = 4.0
        end_logits_1[10] = 3.0

        gmm_context_1 = torch.randn(1, 50, 768)

        tracker.update_belief(
            (start_logits_1, end_logits_1),
            current_segment_id=0,
            gmm_context=gmm_context_1,
            global_offset=0
        )

        original_span = tracker.belief.best_span
        original_confidence = tracker.belief.confidence

        # Second segment with better span
        start_logits_2 = torch.zeros(50)
        end_logits_2 = torch.zeros(50)
        start_logits_2[20] = 6.0  # Higher confidence
        end_logits_2[25] = 5.0    # Higher confidence

        gmm_context_2 = torch.randn(1, 50, 768)

        updated_belief = tracker.update_belief(
            (start_logits_2, end_logits_2),
            current_segment_id=1,
            gmm_context=gmm_context_2,
            global_offset=50
        )

        # Should have updated to new, better span
        assert updated_belief.best_span != original_span
        assert updated_belief.segment_id == 1
        assert len(updated_belief.span_history) == 2

    def test_update_belief_frequency_filtering(self):
        """Test belief update frequency filtering."""
        # This would need integration with the full RBS system
        # For now, test the basic functionality
        pass


class TestBeliefStateTrackerIntegration:
    """Integration tests for BeliefStateTracker with realistic scenarios."""

    def test_full_tracking_scenario(self):
        """Test a complete belief tracking scenario."""
        tracker = BeliefStateTracker(
            confidence_threshold=0.4,  # Lower threshold for reliable test
            max_segments=5,
            enable_trend_analysis=False  # Disable trend analysis for predictable behavior
        )

        # Simulate processing multiple segments with clear progression
        segment_data = [
            # (start_idx, end_idx, start_logit, end_logit, segment_id, global_offset)
            (5, 12, 3.0, 2.5, 0, 0),    # First segment
            (8, 15, 3.5, 3.0, 1, 50),   # Better span in second segment
            (25, 32, 4.0, 3.8, 2, 100), # Even better span
            (30, 35, 2.5, 2.0, 3, 150), # Worse span (should not update)
            (40, 48, 6.0, 5.5, 4, 200), # Much better span, should win
        ]

        for start_idx, end_idx, start_logit, end_logit, seg_id, offset in segment_data:
            start_logits = torch.zeros(50)
            end_logits = torch.zeros(50)
            start_logits[start_idx] = start_logit
            end_logits[end_idx] = end_logit

            # Use consistent context to avoid re-scoring surprises
            gmm_context = torch.ones(1, 50, 768) * 0.1

            updated_belief = tracker.update_belief(
                (start_logits, end_logits),
                current_segment_id=seg_id,
                gmm_context=gmm_context,
                global_offset=offset
            )

            # Verify belief is reasonable
            assert updated_belief.best_span is not None
            assert updated_belief.confidence > 0.0

        # After final segment, should be ready to halt
        assert tracker.should_halt()
        # With much higher logits, last segment should win despite re-scoring
        assert tracker.belief.segment_id == 4  # Last segment
        assert len(tracker.belief.span_history) == 5

    def test_non_monotonic_revision_scenario(self):
        """Test scenario where belief revision occurs."""
        tracker = BeliefStateTracker(
            re_scoring_method="context_weighted",
            enable_trend_analysis=True
        )

        # First segment establishes initial belief
        start_logits_1 = torch.zeros(30)
        end_logits_1 = torch.zeros(30)
        start_logits_1[5] = 3.0
        end_logits_1[10] = 2.5

        gmm_context_1 = torch.ones(1, 30, 768) * 0.5  # Low context values

        tracker.update_belief(
            (start_logits_1, end_logits_1),
            current_segment_id=0,
            gmm_context=gmm_context_1,
            global_offset=0
        )

        initial_span = tracker.belief.best_span
        initial_confidence = tracker.belief.confidence

        # Second segment with strong context that boosts initial belief
        # Start with same span from first segment for re-scoring
        past_candidate = SpanCandidate(
            span=initial_span,
            confidence=initial_confidence,
            segment_id=0,
            gmm_context_hash=hash("segment_0")
        )
        tracker.belief.span_history = [past_candidate]

        # New segment with worse span but better context
        start_logits_2 = torch.zeros(30)
        end_logits_2 = torch.zeros(30)
        start_logits_2[15] = 2.0  # Lower than initial
        end_logits_2[20] = 1.5    # Lower than initial

        gmm_context_2 = torch.ones(1, 30, 768) * 2.0  # High context values

        # Test re-scoring directly
        re_scored_spans = tracker.re_score_past_spans([past_candidate], gmm_context_2)

        # The re-scored span should have higher confidence
        assert re_scored_spans[0].re_scored_confidence > initial_confidence

    def test_confidence_calibration_effect(self):
        """Test that confidence calibration parameters affect output."""
        # Create two trackers with different calibration parameters
        tracker_1 = BeliefStateTracker()
        tracker_2 = BeliefStateTracker()

        # Manually set different calibration parameters
        with torch.no_grad():
            tracker_2.confidence_scaler.fill_(1.5)  # Higher scaling
            tracker_2.confidence_bias.fill_(0.2)    # Positive bias

        # Same input logits
        start_logits = torch.zeros(20)
        end_logits = torch.zeros(20)
        start_logits[5] = 2.0
        end_logits[10] = 1.5

        confidence_1 = tracker_1.compute_confidence(start_logits, end_logits, 5, 10)
        confidence_2 = tracker_2.compute_confidence(start_logits, end_logits, 5, 10)

        # Calibration should affect the confidence values
        # (exact relationship depends on the sigmoid function)
        assert confidence_1 != confidence_2

    @pytest.mark.parametrize("re_scoring_method", ["context_weighted", "exponential_decay"])
    def test_re_scoring_methods(self, re_scoring_method):
        """Test different re-scoring methods."""
        tracker = BeliefStateTracker(re_scoring_method=re_scoring_method)

        past_spans = [
            SpanCandidate((10, 20), 0.6, 0, 123),
            SpanCandidate((30, 40), 0.7, 1, 456)
        ]

        gmm_context = torch.randn(1, 50, 768)

        re_scored = tracker.re_score_past_spans(past_spans, gmm_context)

        assert len(re_scored) == 2
        for candidate in re_scored:
            assert candidate.re_scored
            assert candidate.re_scored_confidence is not None
            assert 0.0 <= candidate.re_scored_confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])