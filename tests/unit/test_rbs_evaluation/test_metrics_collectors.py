"""
Unit tests for RBS-QA metrics collectors.
"""

import pytest
import numpy as np
from typing import List, Dict

from rbsqa.evaluation.metrics_collectors import (
    AccuracyMetricsCollector,
    EfficiencyMetricsCollector,
    ReasoningMetricsCollector,
    HaltingMetricsCollector,
)


class TestAccuracyMetricsCollector:
    """Test accuracy metrics collector."""

    def setup_method(self):
        """Setup test data."""
        self.collector = AccuracyMetricsCollector()

        # Sample predictions and ground truths
        self.predictions = [
            {'answer_span': (10, 15), 'example_id': 0},
            {'answer_span': (20, 25), 'example_id': 1},
            {'answer_span': (30, 35), 'example_id': 2},
        ]

        self.ground_truths = [
            {'answer_span': (10, 15), 'example_id': 0},
            {'answer_span': (21, 25), 'example_id': 1},  # Partial overlap
            {'answer_span': (40, 45), 'example_id': 2},  # No overlap
        ]

    def test_compute_basic(self):
        """Test basic metrics computation."""
        results = self.collector.compute(self.predictions, self.ground_truths)

        assert 'f1' in results
        assert 'exact_match' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'total_examples' in results

        assert results['total_examples'] == 3
        assert 0.0 <= results['f1'] <= 1.0
        assert 0.0 <= results['exact_match'] <= 1.0

    def test_compute_perfect_match(self):
        """Test computation with perfect matches."""
        perfect_predictions = [
            {'answer_span': (10, 15), 'example_id': 0},
            {'answer_span': (20, 25), 'example_id': 1},
        ]

        perfect_ground_truths = [
            {'answer_span': (10, 15), 'example_id': 0},
            {'answer_span': (20, 25), 'example_id': 1},
        ]

        results = self.collector.compute(perfect_predictions, perfect_ground_truths)

        assert results['f1'] == 1.0
        assert results['exact_match'] == 1.0
        assert results['precision'] == 1.0
        assert results['recall'] == 1.0

    def test_compute_empty_spans(self):
        """Test computation with empty spans."""
        predictions = [{'answer_span': (0, -1), 'example_id': 0}]
        ground_truths = [{'answer_span': (0, -1), 'example_id': 0}]

        results = self.collector.compute(predictions, ground_truths)
        assert results['f1'] == 1.0

    def test_compute_length_mismatch(self):
        """Test error handling for mismatched lengths."""
        with pytest.raises(ValueError, match="Predictions and ground truths must have same length"):
            self.collector.compute(self.predictions[:2], self.ground_truths)

    def test_compute_span_metrics(self):
        """Test span metrics computation."""
        # Perfect match
        f1, precision, recall = self.collector._compute_span_metrics((10, 15), (10, 15))
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

        # Partial overlap
        f1, precision, recall = self.collector._compute_span_metrics((10, 15), (12, 18))
        assert 0.0 < f1 < 1.0
        assert 0.0 < precision <= 1.0
        assert 0.0 < recall <= 1.0

        # No overlap
        f1, precision, recall = self.collector._compute_span_metrics((10, 15), (20, 25))
        assert f1 == 0.0
        assert precision == 0.0
        assert recall == 0.0


class TestEfficiencyMetricsCollector:
    """Test efficiency metrics collector."""

    def setup_method(self):
        """Setup test data."""
        self.collector = EfficiencyMetricsCollector()

        self.predictions = [
            {
                'answer_span': (10, 15),
                'segments_processed': 2,
                'total_segments': 5,
                'efficiency_score': 2.5,
                'example_id': 0
            },
            {
                'answer_span': (20, 25),
                'segments_processed': 3,
                'total_segments': 6,
                'efficiency_score': 2.0,
                'example_id': 1
            },
        ]

        self.ground_truths = [
            {'answer_span': (10, 15), 'example_id': 0},
            {'answer_span': (20, 25), 'example_id': 1},
        ]

    def test_compute_basic(self):
        """Test basic efficiency metrics computation."""
        results = self.collector.compute(self.predictions, self.ground_truths)

        assert 'avg_efficiency_score' in results
        assert 'avg_segments_processed' in results
        assert 'avg_total_segments' in results
        assert 'avg_time_saved' in results
        assert 'efficiency_std' in results
        assert 'efficiency_median' in results
        assert 'efficiency_min' in results
        assert 'efficiency_max' in results
        assert 'total_examples' in results

        assert results['total_examples'] == 2
        assert results['avg_efficiency_score'] == 2.25  # (2.5 + 2.0) / 2
        assert results['avg_segments_processed'] == 2.5  # (2 + 3) / 2

    def test_compute_time_saved(self):
        """Test time saved calculation."""
        results = self.collector.compute(self.predictions, self.ground_truths)

        # Time saved = (total - processed) / total
        # Example 1: (5 - 2) / 5 = 0.6
        # Example 2: (6 - 3) / 6 = 0.5
        # Average: (0.6 + 0.5) / 2 = 0.55
        expected_time_saved = 0.55
        assert abs(results['avg_time_saved'] - expected_time_saved) < 0.01

    def test_compute_statistics(self):
        """Test statistical measures."""
        results = self.collector.compute(self.predictions, self.ground_truths)

        efficiency_scores = [2.5, 2.0]
        expected_std = np.std(efficiency_scores)
        expected_median = np.median(efficiency_scores)

        assert abs(results['efficiency_std'] - expected_std) < 0.01
        assert results['efficiency_median'] == expected_median
        assert results['efficiency_min'] == min(efficiency_scores)
        assert results['efficiency_max'] == max(efficiency_scores)


class TestReasoningMetricsCollector:
    """Test reasoning metrics collector."""

    def setup_method(self):
        """Setup test data."""
        self.collector = ReasoningMetricsCollector()

        # Mock belief history for testing
        class MockBelief:
            def __init__(self, best_span, confidence):
                self.best_span = best_span
                self.confidence = confidence

        self.predictions = [
            {
                'answer_span': (10, 15),
                'belief_history': [
                    MockBelief((8, 12), 0.6),
                    MockBelief((10, 15), 0.8),  # Revision: correct
                ],
                'example_id': 0
            },
            {
                'answer_span': (20, 25),
                'belief_history': [
                    MockBelief((20, 25), 0.7),  # No revision
                ],
                'example_id': 1
            },
        ]

        self.ground_truths = [
            {'answer_span': (10, 15), 'example_id': 0},
            {'answer_span': (20, 25), 'example_id': 1},
        ]

    def test_compute_basic(self):
        """Test basic reasoning metrics computation."""
        results = self.collector.compute(self.predictions, self.ground_truths)

        assert 'revision_frequency' in results
        assert 'avg_revisions_per_example' in results
        assert 'beneficial_revision_rate' in results
        assert 'avg_confidence_improvement' in results
        assert 'total_examples' in results

        assert results['total_examples'] == 2
        assert results['revision_frequency'] == 0.5  # 1 out of 2 examples has revisions
        assert results['avg_revisions_per_example'] == 0.5  # 1 revision total / 2 examples

    def test_compute_no_revisions(self):
        """Test computation with no belief revisions."""
        # Mock belief for testing
        class MockBelief:
            def __init__(self, best_span, confidence):
                self.best_span = best_span
                self.confidence = confidence

        no_revision_predictions = [
            {
                'answer_span': (10, 15),
                'belief_history': [MockBelief((10, 15), 0.8)],
                'example_id': 0
            }
        ]

        results = self.collector.compute(no_revision_predictions, self.ground_truths[:1])

        assert results['revision_frequency'] == 0.0
        assert results['avg_revisions_per_example'] == 0.0

    def test_compute_confidence_improvement(self):
        """Test confidence improvement calculation."""
        results = self.collector.compute(self.predictions, self.ground_truths)

        # One example with confidence improvement: 0.8 - 0.6 = 0.2
        # Average across examples with revisions: 0.2 / 1 = 0.2
        expected_improvement = 0.2
        assert abs(results['avg_confidence_improvement'] - expected_improvement) < 0.01

    def test_is_span_correct(self):
        """Test span correctness checking."""
        assert self.collector._is_span_correct((10, 15), (10, 15)) == True
        assert self.collector._is_span_correct((10, 15), (11, 15)) == False
        assert self.collector._is_span_correct((10, 15), (10, 16)) == False


class TestHaltingMetricsCollector:
    """Test halting metrics collector."""

    def setup_method(self):
        """Setup test data."""
        self.collector = HaltingMetricsCollector()

        # Mock halting decision
        class MockHaltingDecision:
            def __init__(self, action):
                self.action = action

        self.predictions = [
            {
                'answer_span': (10, 15),
                'segments_processed': 2,
                'total_segments': 5,
                'halting_history': [MockHaltingDecision("HALT")],
                'example_id': 0
            },
            {
                'answer_span': (20, 25),
                'segments_processed': 5,
                'total_segments': 6,
                'halting_history': [MockHaltingDecision("CONTINUE"), MockHaltingDecision("HALT")],
                'example_id': 1
            },
            {
                'answer_span': (30, 35),
                'segments_processed': 6,
                'total_segments': 6,
                'halting_history': [],  # No halting decision
                'example_id': 2
            },
        ]

        self.ground_truths = [
            {'answer_span': (10, 15), 'example_id': 0},
            {'answer_span': (20, 25), 'example_id': 1},
            {'answer_span': (30, 35), 'example_id': 2},
        ]

    def test_compute_basic(self):
        """Test basic halting metrics computation."""
        results = self.collector.compute(self.predictions, self.ground_truths)

        assert 'halting_decision_rate' in results
        assert 'early_halt_rate' in results
        assert 'late_halt_rate' in results
        assert 'halt_accuracy' in results
        assert 'avg_halt_position' in results
        assert 'total_examples' in results

        assert results['total_examples'] == 3
        assert results['halting_decision_rate'] == 2/3  # 2 out of 3 examples have halting decisions

    def test_halt_timing_categorization(self):
        """Test halting timing categorization."""
        results = self.collector.compute(self.predictions, self.ground_truths)

        # Example 1: 2/5 = 0.4 (early halt)
        # Example 2: 5/6 ≈ 0.83 (late halt)
        assert results['early_halt_rate'] == 0.5  # 1 out of 2 halting decisions
        assert results['late_halt_rate'] == 0.5   # 1 out of 2 halting decisions

    def test_halt_position_calculation(self):
        """Test average halt position calculation."""
        results = self.collector.compute(self.predictions, self.ground_truths)

        # Average relative position: (0.4 + 0.83) / 2 ≈ 0.615
        expected_avg_position = (0.4 + 5/6) / 2
        assert abs(results['avg_halt_position'] - expected_avg_position) < 0.01

    def test_no_halting_decisions(self):
        """Test computation with no halting decisions."""
        no_halt_predictions = [
            {
                'answer_span': (10, 15),
                'segments_processed': 3,
                'total_segments': 5,
                'halting_history': [],
                'example_id': 0
            }
        ]

        results = self.collector.compute(no_halt_predictions, self.ground_truths[:1])

        assert results['halting_decision_rate'] == 0.0
        assert results['early_halt_rate'] == 0.0
        assert results['late_halt_rate'] == 0.0

    def test_dict_halting_decision(self):
        """Test halting decision as dictionary."""
        dict_predictions = [
            {
                'answer_span': (10, 15),
                'segments_processed': 2,
                'total_segments': 5,
                'halting_history': [{'action': 'HALT'}],
                'example_id': 0
            }
        ]

        results = self.collector.compute(dict_predictions, self.ground_truths[:1])

        assert results['halting_decision_rate'] == 1.0