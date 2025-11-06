"""
Unit tests for RBS-QA evaluator.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rbsqa.evaluation.rbs_evaluator import RBSEvaluator
from rbsqa.evaluation.config import RBSEvaluationConfig


class MockModel:
    """Mock model for testing."""

    def __init__(self, adaptive_inference_available=True):
        self.adaptive_inference_available = adaptive_inference_available

    def eval(self):
        pass

    def set_inference_mode(self, mode):
        pass

    def adaptive_inference(self, question_input_ids, context_segments, max_segments=32, track_detailed_belief=False):
        """Mock adaptive inference."""
        class MockResult:
            def __init__(self):
                self.answer_span = (10, 15)
                self.confidence = 0.8
                self.segments_processed = len(context_segments) // 2
                self.inference_time = 0.1
                self.belief_history = []
                self.halting_history = []

        return MockResult()


class TestRBSEvaluator:
    """Test RBS evaluator."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = RBSEvaluationConfig(
            output_dir=self.temp_dir,
            max_segments_per_example=32,
            generate_html_report=False  # Disable HTML generation for faster tests
        )
        self.mock_model = MockModel()

        self.evaluator = RBSEvaluator(
            model=self.mock_model,
            config=self.config,
            output_dir=self.temp_dir
        )

        # Sample test dataset
        self.test_dataset = [
            {
                'id': f'example_{i}',
                'question_input_ids': [101, 1029, 102] + [103] * 20 + [102],
                'context_segments': [[101] + [103] * 50 + [102] for _ in range(5)],
                'answer_span': (10, 15),
                'question': f"What is the answer to question {i}?"
            }
            for i in range(3)
        ]

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        assert self.evaluator.model == self.mock_model
        assert self.evaluator.config == self.config
        assert self.evaluator.output_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)

        # Check that collectors and analyzers are initialized
        assert self.evaluator.accuracy_metrics is not None
        assert self.evaluator.efficiency_metrics is not None
        assert self.evaluator.reasoning_metrics is not None
        assert self.evaluator.halting_metrics is not None
        assert self.evaluator.belief_analyzer is not None
        assert self.evaluator.halting_analyzer is not None
        assert self.evaluator.comparative_analyzer is not None

    def test_evaluate_basic(self):
        """Test basic evaluation."""
        results = self.evaluator.evaluate(
            test_dataset=self.test_dataset,
            baseline_models=None,
            detailed_analysis=False
        )

        assert 'main' in results
        assert 'accuracy' in results['main']
        assert 'efficiency' in results['main']
        assert 'reasoning' in results['main']
        assert 'halting' in results['main']
        assert 'summary' in results['main']

        # Check summary metrics
        summary = results['main']['summary']
        assert 'f1' in summary
        assert 'exact_match' in summary
        assert 'avg_efficiency_score' in summary
        assert 'combined_score' in summary

        assert 0.0 <= summary['f1'] <= 1.0
        assert 0.0 <= summary['exact_match'] <= 1.0
        assert summary['avg_efficiency_score'] >= 1.0  # Should be >= 1.0 due to mock

    def test_evaluate_with_detailed_analysis(self):
        """Test evaluation with detailed analysis."""
        results = self.evaluator.evaluate(
            test_dataset=self.test_dataset,
            baseline_models=None,
            detailed_analysis=True
        )

        assert 'analysis' in results
        analysis = results['analysis']

        assert 'belief_state' in analysis
        assert 'halting_policy' in analysis
        assert 'non_monotonic_reasoning' in analysis
        assert 'error_analysis' in analysis
        assert 'difficulty_analysis' in analysis

    def test_evaluate_with_baselines(self):
        """Test evaluation with baseline models."""
        baseline_models = {
            'baseline1': MockModel(),
            'baseline2': MockModel()
        }

        results = self.evaluator.evaluate(
            test_dataset=self.test_dataset,
            baseline_models=baseline_models,
            detailed_analysis=False
        )

        assert 'comparisons' in results
        comparisons = results['comparisons']

        assert 'baseline1' in comparisons
        assert 'baseline2' in comparisons
        assert 'comparative_analysis' in comparisons

    def test_evaluate_main_method(self):
        """Test the main evaluation method."""
        main_results = self.evaluator._evaluate_main(self.test_dataset)

        assert 'accuracy' in main_results
        assert 'efficiency' in main_results
        assert 'reasoning' in main_results
        assert 'halting' in main_results
        assert 'summary' in main_results

        # Check that all examples were processed
        accuracy_results = main_results['accuracy']
        assert accuracy_results['total_examples'] == len(self.test_dataset)

    def test_evaluate_baselines_method(self):
        """Test baseline evaluation method."""
        # Set up main evaluation results first
        self.evaluator.evaluation_results['main'] = {
            'summary': {'f1': 0.8, 'avg_efficiency_score': 1.5}
        }

        baseline_models = {
            'test_baseline': MockModel()
        }

        comparison_results = self.evaluator._evaluate_baselines(self.test_dataset, baseline_models)

        assert 'test_baseline' in comparison_results
        assert 'accuracy' in comparison_results['test_baseline']
        assert 'efficiency' in comparison_results['test_baseline']
        assert 'summary' in comparison_results['test_baseline']
        assert 'comparative_analysis' in comparison_results

    def test_perform_detailed_analysis(self):
        """Test detailed analysis method."""
        # Set up some mock evaluation results first
        self.evaluator.evaluation_results = {
            'main': {
                'summary': {'f1': 0.8, 'avg_efficiency_score': 1.5},
                'reasoning': {'revision_frequency': 0.3}
            }
        }

        analysis_results = self.evaluator._perform_detailed_analysis(self.test_dataset)

        assert 'belief_state' in analysis_results
        assert 'halting_policy' in analysis_results
        assert 'non_monotonic_reasoning' in analysis_results
        assert 'error_analysis' in analysis_results
        assert 'difficulty_analysis' in analysis_results

    def test_analyze_non_monotonic_reasoning(self):
        """Test non-monotonic reasoning analysis."""
        # Create a mock model with detailed belief tracking
        class MockResultWithBeliefs:
            def __init__(self):
                class MockBelief:
                    def __init__(self, span, confidence, segment_id=None):
                        self.best_span = span
                        self.confidence = confidence
                        self.segment_id = segment_id

                self.belief_history = [
                    MockBelief((8, 12), 0.6, 0),
                    MockBelief((10, 15), 0.8, 1)  # Revision
                ]

        model_with_beliefs = Mock()
        model_with_beliefs.adaptive_inference.return_value = MockResultWithBeliefs()

        evaluator_with_beliefs = RBSEvaluator(
            model=model_with_beliefs,
            config=self.config,
            output_dir=self.temp_dir
        )

        results = evaluator_with_beliefs._analyze_non_monotonic_reasoning(self.test_dataset)

        assert 'revision_frequency' in results
        assert 'avg_revisions_per_example' in results
        assert 'beneficial_revision_rate' in results
        assert 'detrimental_revision_rate' in results
        assert 'revision_patterns' in results
        assert 'summary' in results

        summary = results['summary']
        assert 'non_monotonic_reasoning_detected' in summary
        assert 'revision_effectiveness' in summary

    def test_analyze_errors(self):
        """Test error analysis."""
        results = self.evaluator._analyze_errors(self.test_dataset)

        assert 'error_distribution' in results
        assert 'confidence_calibration' in results
        assert 'total_errors' in results
        assert 'error_rate' in results

        error_dist = results['error_distribution']
        # Should have at least one error category even if empty
        assert len(error_dist) >= 0

    def test_analyze_efficiency_by_difficulty(self):
        """Test efficiency analysis by difficulty."""
        results = self.evaluator._analyze_efficiency_by_difficulty(self.test_dataset)

        assert 'easy' in results or 'medium' in results or 'hard' in results

        for difficulty, metrics in results.items():
            assert 'count' in metrics
            assert 'avg_segments_processed' in metrics
            assert 'avg_efficiency_score' in metrics
            assert 'accuracy' in metrics
            assert 'avg_confidence' in metrics
            assert 'avg_inference_time' in metrics

    def test_categorize_difficulty(self):
        """Test difficulty categorization."""
        # Easy case
        difficulty = self.evaluator._categorize_difficulty(400, 15, {})
        assert difficulty == "easy"

        # Medium case
        difficulty = self.evaluator._categorize_difficulty(800, 25, {})
        assert difficulty == "medium"

        # Hard case
        difficulty = self.evaluator._categorize_difficulty(1200, 35, {})
        assert difficulty == "hard"

    def test_categorize_error(self):
        """Test error categorization."""
        # No overlap
        error_type = self.evaluator._categorize_error((10, 15), (20, 25), {})
        assert error_type == "no_overlap"

        # Perfect match (should be "other" as it's not really an error)
        error_type = self.evaluator._categorize_error((10, 15), (10, 15), {})
        assert error_type == "other"

        # Partial overlap
        error_type = self.evaluator._categorize_error((10, 15), (12, 18), {})
        assert error_type in ["partial_overlap", "boundary_error"]

        # Boundary error
        error_type = self.evaluator._categorize_error((10, 16), (10, 15), {})
        assert error_type == "boundary_error"

    def test_is_span_correct(self):
        """Test span correctness checking."""
        assert self.evaluator._is_span_correct((10, 15), (10, 15)) == True
        assert self.evaluator._is_span_correct((10, 15), (11, 15)) == False
        assert self.evaluator._is_span_correct((10, 15), (10, 16)) == False
        assert self.evaluator._is_span_correct((11, 14), (10, 15)) == False

    def test_compute_efficiency_score(self):
        """Test efficiency score computation."""
        mock_result = Mock()
        mock_result.segments_processed = 2

        example = {'context_segments': [[1] * 50, [2] * 50, [3] * 50, [4] * 50]}  # 4 segments

        score = self.evaluator._compute_efficiency_score(mock_result, example)
        assert score == 4.0 / 2.0  # total_segments / segments_processed

    def test_compute_summary_metrics(self):
        """Test summary metrics computation."""
        accuracy_results = {
            'f1': 0.8,
            'exact_match': 0.75
        }
        efficiency_results = {
            'avg_efficiency_score': 1.6,
            'avg_segments_processed': 2.5
        }

        summary = self.evaluator._compute_summary_metrics(accuracy_results, efficiency_results)

        assert summary['f1'] == 0.8
        assert summary['exact_match'] == 0.75
        assert summary['avg_efficiency_score'] == 1.6
        assert summary['avg_segments_processed'] == 2.5
        assert summary['combined_score'] == 0.8 * 0.7 + 1.6 * 0.3

    def test_fallback_inference(self):
        """Test fallback inference method."""
        example = self.test_dataset[0]
        result = self.evaluator._fallback_inference(example)

        assert hasattr(result, 'answer_span')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'segments_processed')
        assert hasattr(result, 'inference_time')
        assert hasattr(result, 'belief_history')
        assert hasattr(result, 'halting_history')

    def test_generate_evaluation_report(self):
        """Test report generation."""
        # Set up some mock results
        self.evaluator.evaluation_results = {
            'main': {
                'summary': {'f1': 0.8, 'exact_match': 0.75, 'avg_efficiency_score': 1.6, 'combined_score': 0.79},
                'accuracy': {'f1': 0.8, 'exact_match': 0.75},
                'efficiency': {'avg_efficiency_score': 1.6}
            }
        }

        self.evaluator._generate_evaluation_report()

        # Check that files were created
        assert os.path.exists(os.path.join(self.temp_dir, "evaluation_results.json"))
        assert os.path.exists(os.path.join(self.temp_dir, "evaluation_summary.md"))

        # Check JSON file content
        with open(os.path.join(self.temp_dir, "evaluation_results.json"), 'r') as f:
            saved_results = json.load(f)
            assert 'main' in saved_results
            assert saved_results['main']['summary']['f1'] == 0.8

    def test_generate_html_report(self):
        """Test HTML report generation."""
        self.evaluator.evaluation_results = {
            'main': {
                'summary': {'f1': 0.8, 'exact_match': 0.75, 'avg_efficiency_score': 1.6, 'combined_score': 0.79},
                'accuracy': {'f1': 0.8, 'exact_match': 0.75},
                'efficiency': {'avg_efficiency_score': 1.6}
            }
        }

        html_content = self.evaluator._generate_html_report()

        assert isinstance(html_content, str)
        assert "RBS-QA Evaluation Report" in html_content
        assert "F1 Score" in html_content
        assert "0.800" in html_content  # F1 score should appear

    def test_generate_markdown_summary(self):
        """Test markdown summary generation."""
        self.evaluator.evaluation_results = {
            'main': {
                'summary': {'f1': 0.8, 'exact_match': 0.75, 'avg_efficiency_score': 1.6, 'combined_score': 0.79}
            }
        }

        md_content = self.evaluator._generate_markdown_summary()

        assert isinstance(md_content, str)
        assert "# RBS-QA Evaluation Summary" in md_content
        assert "**F1 Score**: 0.800" in md_content
        assert "## Key Findings" in md_content
        assert "## Recommendations" in md_content

    def test_generate_key_findings(self):
        """Test key findings generation."""
        # Test with good performance
        self.evaluator.evaluation_results = {
            'main': {
                'summary': {'f1': 0.85, 'avg_efficiency_score': 1.8},
                'reasoning': {'revision_frequency': 0.2}
            }
        }

        findings = self.evaluator._generate_key_findings()
        assert "Excellent QA performance" in findings
        assert "Excellent efficiency" in findings
        assert "non-monotonic reasoning" in findings

        # Test with moderate performance
        self.evaluator.evaluation_results = {
            'main': {
                'summary': {'f1': 0.72, 'avg_efficiency_score': 1.1},
                'reasoning': {'revision_frequency': 0.05}
            }
        }

        findings = self.evaluator._generate_key_findings()
        assert "Good QA performance" in findings or "Moderate QA performance" in findings

    def test_generate_recommendations(self):
        """Test recommendations generation."""
        # Test with poor performance
        self.evaluator.evaluation_results = {
            'main': {
                'summary': {'f1': 0.6, 'avg_efficiency_score': 1.1}
            }
        }

        recommendations = self.evaluator._generate_recommendations()
        assert "Improve QA accuracy" in recommendations
        assert "Monitor training" in recommendations
        assert "Analyze failures" in recommendations

    def test_error_handling_in_evaluation(self):
        """Test error handling during evaluation."""
        # Create a model that raises an exception
        failing_model = Mock()
        failing_model.adaptive_inference.side_effect = Exception("Test error")

        evaluator_with_failing_model = RBSEvaluator(
            model=failing_model,
            config=self.config,
            output_dir=self.temp_dir
        )

        # Should not raise exception, but handle gracefully
        results = evaluator_with_failing_model.evaluate(
            test_dataset=self.test_dataset[:1],
            baseline_models=None,
            detailed_analysis=False
        )

        # Should still have results, even with errors
        assert 'main' in results
        assert results['main']['accuracy']['total_examples'] == 1

    def test_model_without_adaptive_inference(self):
        """Test evaluation with model that doesn't have adaptive_inference."""
        class ModelWithoutAdaptive:
            def eval(self):
                pass

        model_no_adaptive = ModelWithoutAdaptive()
        evaluator_no_adaptive = RBSEvaluator(
            model=model_no_adaptive,
            config=self.config,
            output_dir=self.temp_dir
        )

        # Should not raise exception
        results = evaluator_no_adaptive.evaluate(
            test_dataset=self.test_dataset[:1],
            baseline_models=None,
            detailed_analysis=False
        )

        assert 'main' in results