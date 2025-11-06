"""
Unit tests for RBS-QA analysis tools.
"""

import pytest
import numpy as np
from typing import List, Dict

from rbsqa.evaluation.analyzers import (
    BeliefStateAnalyzer,
    HaltingPolicyAnalyzer,
    ComparativeAnalyzer,
)


class TestBeliefStateAnalyzer:
    """Test belief state analyzer."""

    def setup_method(self):
        """Setup test data."""
        self.analyzer = BeliefStateAnalyzer()

        self.main_results = {
            'reasoning': {
                'revision_frequency': 0.3,
                'avg_revisions_per_example': 1.2
            }
        }

        self.test_dataset = [
            {
                'id': 'example_1',
                'question_input_ids': [101, 1029, 102] + [103] * 20 + [102],
                'context_segments': [[101] + [103] * 50 + [102] for _ in range(5)],
                'answer_span': (10, 15),
                'question': "What is the answer?"
            }
        ] * 10  # 10 examples

    def test_analyze_belief_patterns(self):
        """Test belief pattern analysis."""
        results = self.analyzer.analyze_belief_patterns(self.main_results, self.test_dataset)

        assert 'confidence_calibration' in results
        assert 'belief_convergence_patterns' in results
        assert 'segment_trigger_analysis' in results
        assert 'summary' in results

        # Check summary fields
        summary = results['summary']
        assert 'total_examples_analyzed' in summary
        assert 'avg_belief_revisions' in summary
        assert 'belief_stability_score' in summary

        assert summary['total_examples_analyzed'] == len(self.test_dataset)

    def test_compute_belief_stability(self):
        """Test belief stability computation."""
        # Test with no revisions (high stability)
        no_revisions = [[] for _ in range(10)]
        stability = self.analyzer._compute_belief_stability(no_revisions)
        assert stability == 1.0

        # Test with many revisions (low stability)
        many_revisions = [[1, 2, 3, 4, 5] for _ in range(10)]
        stability = self.analyzer._compute_belief_stability(many_revisions)
        assert 0.0 <= stability <= 1.0
        assert stability < 1.0

    def test_analyze_confidence_calibration(self):
        """Test confidence calibration analysis."""
        belief_histories = [[] for _ in range(10)]
        confidence_progressions = [[] for _ in range(10)]

        results = self.analyzer._analyze_confidence_calibration(
            belief_histories, confidence_progressions
        )

        assert 'calibration_error' in results
        assert 'confidence_buckets' in results
        assert 'accuracy_by_confidence' in results

    def test_analyze_convergence_patterns(self):
        """Test belief convergence analysis."""
        belief_histories = [[] for _ in range(10)]
        revision_patterns = [[] for _ in range(10)]

        results = self.analyzer._analyze_convergence_patterns(
            belief_histories, revision_patterns
        )

        assert 'avg_convergence_steps' in results
        assert 'early_convergence_rate' in results
        assert 'late_convergence_rate' in results
        assert 'no_convergence_rate' in results

    def test_analyze_segment_triggers(self):
        """Test segment trigger analysis."""
        revision_patterns = [[] for _ in range(10)]

        results = self.analyzer._analyze_segment_triggers(revision_patterns)

        assert 'trigger_segments_by_position' in results
        assert 'trigger_segments_by_content' in results
        assert 'avg_segments_before_revision' in results


class TestHaltingPolicyAnalyzer:
    """Test halting policy analyzer."""

    def setup_method(self):
        """Setup test data."""
        self.analyzer = HaltingPolicyAnalyzer()

        self.main_results = {
            'halting': {
                'halting_decision_rate': 0.8,
                'avg_halt_position': 0.6
            },
            'summary': {
                'f1': 0.75,
                'avg_efficiency_score': 1.5
            }
        }

        self.test_dataset = [
            {
                'id': 'example_1',
                'question_input_ids': [101, 1029, 102] + [103] * 20 + [102],
                'context_segments': [[101] + [103] * 50 + [102] for _ in range(5)],
                'answer_span': (10, 15),
                'question': "What is the answer?"
            }
        ] * 10  # 10 examples

    def test_analyze_halting_patterns(self):
        """Test halting pattern analysis."""
        results = self.analyzer.analyze_halting_patterns(self.main_results, self.test_dataset)

        assert 'confidence_threshold_analysis' in results
        assert 'efficiency_vs_accuracy_tradeoff' in results
        assert 'halting_decision_patterns' in results
        assert 'summary' in results

        # Check summary fields
        summary = results['summary']
        assert 'avg_halt_confidence' in summary
        assert 'optimal_threshold_range' in summary
        assert 'policy_consistency_score' in summary

    def test_analyze_confidence_thresholds(self):
        """Test confidence threshold analysis."""
        halting_decisions = [None] * 10
        halt_confidences = [0.5, 0.6, 0.7, 0.8, 0.9] * 2

        results = self.analyzer._analyze_confidence_thresholds(
            halting_decisions, halt_confidences
        )

        assert 'optimal_threshold' in results
        assert 'threshold_performance' in results

        threshold_perf = results['threshold_performance']
        for threshold in ['0.5', '0.6', '0.7', '0.8', '0.9']:
            assert threshold in threshold_perf
            assert 'accuracy' in threshold_perf[threshold]
            assert 'efficiency' in threshold_perf[threshold]

    def test_analyze_efficiency_accuracy_tradeoff(self):
        """Test efficiency-accuracy tradeoff analysis."""
        halt_positions = [0.3, 0.5, 0.7, 0.9] * 3  # 12 positions

        results = self.analyzer._analyze_efficiency_accuracy_tradeoff(
            halt_positions, self.main_results
        )

        assert 'pareto_optimal_points' in results
        assert 'efficiency_gain_per_accuracy_loss' in results
        assert 'optimal_balance_point' in results

    def test_analyze_halting_decision_patterns(self):
        """Test halting decision pattern analysis."""
        halting_decisions = [None] * 10
        halt_positions = [0.2, 0.4, 0.6, 0.8] * 3  # 12 positions

        results = self.analyzer._analyze_halting_decision_patterns(
            halting_decisions, halt_positions
        )

        assert 'decision_consistency' in results
        assert 'position_variance' in results
        assert 'early_halt_triggers' in results
        assert 'late_halt_triggers' in results

    def test_compute_policy_consistency(self):
        """Test policy consistency computation."""
        # Test with consistent positions
        consistent_positions = [0.5] * 10
        consistency = self.analyzer._compute_policy_consistency(consistent_positions)
        assert consistency == 1.0

        # Test with varied positions
        varied_positions = [0.2, 0.4, 0.6, 0.8] * 3
        consistency = self.analyzer._compute_policy_consistency(varied_positions)
        assert 0.0 <= consistency <= 1.0
        assert consistency < 1.0

        # Test with empty list
        consistency = self.analyzer._compute_policy_consistency([])
        assert consistency == 1.0

        # Test with single position
        consistency = self.analyzer._compute_policy_consistency([0.5])
        assert consistency == 1.0


class TestComparativeAnalyzer:
    """Test comparative analyzer."""

    def setup_method(self):
        """Setup test data."""
        self.analyzer = ComparativeAnalyzer()

        self.rbs_results = {
            'main': {
                'summary': {
                    'f1': 0.82,
                    'exact_match': 0.75,
                    'avg_efficiency_score': 1.6,
                    'combined_score': 0.79
                }
            }
        }

        self.baseline_results = {
            'gmm': {
                'summary': {
                    'f1': 0.78,
                    'exact_match': 0.70,
                    'avg_efficiency_score': 1.0,
                    'combined_score': 0.71
                }
            },
            'base_xlnet': {
                'summary': {
                    'f1': 0.75,
                    'exact_match': 0.68,
                    'avg_efficiency_score': 0.8,
                    'combined_score': 0.67
                }
            }
        }

    def test_analyze(self):
        """Test comparative analysis."""
        results = self.analyzer.analyze(self.rbs_results, self.baseline_results)

        assert 'relative_improvements' in results
        assert 'statistical_significance' in results
        assert 'comparative_summary' in results
        assert 'key_insights' in results

        # Check relative improvements
        improvements = results['relative_improvements']
        assert 'gmm' in improvements
        assert 'base_xlnet' in improvements

        for baseline_name, improvement in improvements.items():
            assert 'f1_improvement_pct' in improvement
            assert 'efficiency_improvement_pct' in improvement
            assert 'combined_improvement_pct' in improvement
            assert 'wins' in improvement

    def test_compare_with_baseline(self):
        """Test baseline comparison."""
        rbs_summary = self.rbs_results['main']['summary']
        baseline_summary = self.baseline_results['gmm']['summary']

        comparison = self.analyzer._compare_with_baseline(
            rbs_summary, baseline_summary, 'gmm'
        )

        assert 'f1_improvement_pct' in comparison
        assert 'efficiency_improvement_pct' in comparison
        assert 'combined_improvement_pct' in comparison
        assert 'absolute_f1_gain' in comparison
        assert 'absolute_efficiency_gain' in comparison
        assert 'wins' in comparison

        # Check that improvements are calculated correctly
        expected_f1_improvement = ((0.82 - 0.78) / 0.78) * 100
        assert abs(comparison['f1_improvement_pct'] - expected_f1_improvement) < 0.01

        expected_efficiency_improvement = ((1.6 - 1.0) / 1.0) * 100
        assert abs(comparison['efficiency_improvement_pct'] - expected_efficiency_improvement) < 0.01

    def test_perform_statistical_tests(self):
        """Test statistical significance testing."""
        comparisons = {
            'gmm': {'f1_improvement_pct': 5.1, 'efficiency_improvement_pct': 60.0},
            'base_xlnet': {'f1_improvement_pct': 9.3, 'efficiency_improvement_pct': 100.0}
        }

        results = self.analyzer._perform_statistical_tests(comparisons)

        assert 'significance_level' in results
        assert 'significant_improvements' in results

        sig_improvements = results['significant_improvements']
        assert 'gmm' in sig_improvements
        assert 'base_xlnet' in sig_improvements

        for baseline in sig_improvements:
            assert 'f1' in sig_improvements[baseline]
            assert 'efficiency' in sig_improvements[baseline]
            assert 'p_value' in sig_improvements[baseline]['f1']
            assert 'significant' in sig_improvements[baseline]['f1']

    def test_generate_comparative_summary(self):
        """Test comparative summary generation."""
        comparisons = {
            'gmm': {
                'f1_improvement_pct': 5.1,
                'efficiency_improvement_pct': 60.0,
                'wins': {'accuracy': True, 'efficiency': True, 'combined': True}
            },
            'base_xlnet': {
                'f1_improvement_pct': 9.3,
                'efficiency_improvement_pct': 100.0,
                'wins': {'accuracy': True, 'efficiency': True, 'combined': True}
            }
        }

        summary = self.analyzer._generate_comparative_summary(comparisons)

        assert 'total_baselines_tested' in summary
        assert 'accuracy_win_rate' in summary
        assert 'efficiency_win_rate' in summary
        assert 'combined_win_rate' in summary
        assert 'avg_f1_improvement_pct' in summary
        assert 'avg_efficiency_improvement_pct' in summary
        assert 'overall_assessment' in summary

        # Check calculated values
        assert summary['total_baselines_tested'] == 2
        assert summary['accuracy_win_rate'] == 1.0  # Wins against both baselines
        assert summary['efficiency_win_rate'] == 1.0  # Wins against both baselines

        expected_f1_improvement = (5.1 + 9.3) / 2
        assert abs(summary['avg_f1_improvement_pct'] - expected_f1_improvement) < 0.01

    def test_assess_overall_performance(self):
        """Test overall performance assessment."""
        # Test significant improvements
        assessment = self.analyzer._assess_overall_performance(10.0, 50.0)
        assert "Significant improvement" in assessment

        # Test accuracy-focused improvement
        assessment = self.analyzer._assess_overall_performance(10.0, 10.0)
        assert "accuracy improvement" in assessment

        # Test efficiency-focused improvement
        assessment = self.analyzer._assess_overall_performance(2.0, 50.0)
        assert "efficiency" in assessment

        # Test modest improvements
        assessment = self.analyzer._assess_overall_performance(2.0, 10.0)
        assert "Modest improvements" in assessment

        # Test no improvement
        assessment = self.analyzer._assess_overall_performance(-1.0, -5.0)
        assert "Limited or no improvement" in assessment

    def test_extract_key_insights(self):
        """Test key insights extraction."""
        comparisons = {
            'gmm': {
                'f1_improvement_pct': 15.0,
                'efficiency_improvement_pct': 60.0,
                'wins': {'accuracy': True, 'efficiency': True}
            },
            'base_xlnet': {
                'f1_improvement_pct': -2.0,
                'efficiency_improvement_pct': 80.0,
                'wins': {'accuracy': False, 'efficiency': True}
            }
        }

        insights = self.analyzer._extract_key_insights(comparisons)

        assert isinstance(insights, list)
        assert len(insights) > 0

        # Check for expected insights
        insights_text = " ".join(insights)
        assert "Strong accuracy improvement" in insights_text or "Major efficiency gain" in insights_text

    def test_extract_key_insights_no_improvements(self):
        """Test insights extraction with no improvements."""
        comparisons = {
            'baseline1': {
                'f1_improvement_pct': 1.0,
                'efficiency_improvement_pct': 5.0,
                'wins': {'accuracy': True, 'efficiency': True}
            }
        }

        insights = self.analyzer._extract_key_insights(comparisons)

        assert isinstance(insights, list)
        assert len(insights) == 1
        assert "competitive performance" in insights[0]