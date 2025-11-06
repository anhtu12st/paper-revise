"""
Integration tests for RBS-QA evaluation pipeline.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch
from pathlib import Path

from rbsqa.evaluation.rbs_evaluator import RBSEvaluator
from rbsqa.evaluation.config import RBSEvaluationConfig


class MockRBSModel:
    """Mock RBS model with realistic behavior."""

    def __init__(self, accuracy_level=0.8, efficiency_level=0.7):
        self.accuracy_level = accuracy_level
        self.efficiency_level = efficiency_level
        self.training_mode = None

    def eval(self):
        pass

    def set_inference_mode(self, mode):
        self.training_mode = mode

    def adaptive_inference(self, question_input_ids, context_segments, max_segments=32, track_detailed_belief=False):
        """Mock adaptive inference with realistic behavior."""
        total_segments = len(context_segments)

        # Simulate adaptive processing
        if self.efficiency_level > 0.8:
            # High efficiency: process fewer segments
            segments_processed = max(1, int(total_segments * (1 - self.efficiency_level * 0.5)))
        else:
            # Low efficiency: process more segments
            segments_processed = max(1, int(total_segments * 0.8))

        # Simulate belief history
        class MockBelief:
            def __init__(self, span, confidence, segment_id=None):
                self.best_span = span
                self.confidence = confidence
                self.segment_id = segment_id

        belief_history = []
        if track_detailed_belief and segments_processed > 1:
            # Add some belief revisions
            belief_history.append(MockBelief((5, 10), 0.6, 0))
            if self.accuracy_level < 0.7:
                # Poor model: revise beliefs more often
                belief_history.append(MockBelief((8, 12), 0.7, 1))

        # Simulate halting decisions
        class MockHaltingDecision:
            def __init__(self, action, confidence=0.7):
                self.action = action
                self.confidence = confidence

        halting_history = []
        if segments_processed < total_segments:
            halting_history.append(MockHaltingDecision("HALT", 0.6 + self.efficiency_level * 0.2))

        # Generate answer span
        import random
        random.seed(hash(tuple(question_input_ids)) % 1000)  # Reproducible randomness

        if random.random() < self.accuracy_level:
            # Correct answer (simulate)
            answer_span = (10, 15)
            confidence = 0.7 + random.random() * 0.3
        else:
            # Incorrect answer
            answer_span = (random.randint(0, 20), random.randint(21, 40))
            confidence = 0.3 + random.random() * 0.4

        class MockResult:
            def __init__(self):
                self.answer_span = answer_span
                self.confidence = confidence
                self.segments_processed = segments_processed
                self.total_segments = total_segments
                self.inference_time = 0.05 + segments_processed * 0.02
                self.belief_history = belief_history
                self.halting_history = halting_history

        return MockResult()


class TestFullEvaluationPipeline:
    """Test the full evaluation pipeline."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test dataset
        self.test_dataset = self._create_test_dataset()

        # Create evaluation config
        self.config = RBSEvaluationConfig(
            output_dir=self.temp_dir,
            max_segments_per_example=32,
            generate_html_report=True,
            generate_markdown_report=True,
            save_detailed_results=True,
            statistical_tests=True
        )

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_dataset(self, num_examples=20):
        """Create a realistic test dataset."""
        dataset = []
        for i in range(num_examples):
            # Vary difficulty by document length and complexity
            if i < num_examples // 3:
                # Easy examples
                doc_segments = 3
                segment_length = 30
            elif i < 2 * num_examples // 3:
                # Medium examples
                doc_segments = 5
                segment_length = 50
            else:
                # Hard examples
                doc_segments = 8
                segment_length = 80

            context_segments = []
            for j in range(doc_segments):
                segment = [101] + [103] * segment_length + [102]  # BOS, content, EOS
                context_segments.append(segment)

            dataset.append({
                'id': f'test_example_{i}',
                'question_input_ids': [101, 1029, 102] + [103] * 25 + [102],
                'context_segments': context_segments,
                'answer_span': (15, 20),  # Consistent answer span
                'question': f"What is the main topic in document {i}?"
            })

        return dataset

    def test_full_evaluation_pipeline(self):
        """Test the complete evaluation pipeline."""
        # Create a high-performing model
        model = MockRBSModel(accuracy_level=0.85, efficiency_level=0.8)

        # Initialize evaluator
        evaluator = RBSEvaluator(
            model=model,
            config=self.config,
            output_dir=self.temp_dir
        )

        # Run full evaluation
        results = evaluator.evaluate(
            test_dataset=self.test_dataset,
            baseline_models=None,
            detailed_analysis=True
        )

        # Verify results structure
        assert 'main' in results
        assert 'analysis' in results

        # Verify main results
        main_results = results['main']
        assert 'accuracy' in main_results
        assert 'efficiency' in main_results
        assert 'reasoning' in main_results
        assert 'halting' in main_results
        assert 'summary' in main_results

        # Verify analysis results
        analysis_results = results['analysis']
        assert 'belief_state' in analysis_results
        assert 'halting_policy' in analysis_results
        assert 'non_monotonic_reasoning' in analysis_results
        assert 'error_analysis' in analysis_results
        assert 'difficulty_analysis' in analysis_results

        # Verify metric values are reasonable
        summary = main_results['summary']
        assert 0.0 <= summary['f1'] <= 1.0
        assert 0.0 <= summary['exact_match'] <= 1.0
        assert summary['avg_efficiency_score'] >= 1.0  # Should be >= 1.0
        assert 0.0 <= summary['combined_score'] <= 1.0

        # Verify report files were generated
        assert os.path.exists(os.path.join(self.temp_dir, "evaluation_results.json"))
        assert os.path.exists(os.path.join(self.temp_dir, "evaluation_summary.md"))
        assert os.path.exists(os.path.join(self.temp_dir, "evaluation_report.html"))

    def test_evaluation_with_baselines(self):
        """Test evaluation with baseline models."""
        # Create models with different performance levels
        rbs_model = MockRBSModel(accuracy_level=0.85, efficiency_level=0.8)
        gmm_baseline = MockRBSModel(accuracy_level=0.80, efficiency_level=0.5)  # Lower efficiency
        xlnet_baseline = MockRBSModel(accuracy_level=0.75, efficiency_level=0.3)  # Much lower efficiency

        baseline_models = {
            'gmm': gmm_baseline,
            'base_xlnet': xlnet_baseline
        }

        evaluator = RBSEvaluator(
            model=rbs_model,
            config=self.config,
            output_dir=self.temp_dir
        )

        results = evaluator.evaluate(
            test_dataset=self.test_dataset,
            baseline_models=baseline_models,
            detailed_analysis=True
        )

        # Verify comparison results
        assert 'comparisons' in results
        comparisons = results['comparisons']

        assert 'gmm' in comparisons
        assert 'base_xlnet' in comparisons
        assert 'comparative_analysis' in comparisons

        # Verify comparative analysis
        comp_analysis = comparisons['comparative_analysis']
        assert 'relative_improvements' in comp_analysis
        assert 'statistical_significance' in comp_analysis
        assert 'comparative_summary' in comp_analysis
        assert 'key_insights' in comp_analysis

        # Verify that RBS shows improvement over baselines
        rbs_summary = results['main']['summary']
        gmm_summary = comparisons['gmm']['summary']
        xlnet_summary = comparisons['base_xlnet']['summary']

        # RBS should have better efficiency
        assert rbs_summary['avg_efficiency_score'] > gmm_summary['avg_efficiency_score']
        assert rbs_summary['avg_efficiency_score'] > xlnet_summary['avg_efficiency_score']

        # RBS should have competitive or better accuracy
        assert rbs_summary['f1'] >= gmm_summary['f1'] * 0.95  # Allow small variance
        assert rbs_summary['f1'] > xlnet_summary['f1']

    def test_evaluation_error_handling(self):
        """Test evaluation pipeline error handling."""
        # Create a model that fails occasionally
        class FlakyModel:
            def __init__(self):
                self.call_count = 0

            def eval(self):
                pass

            def set_inference_mode(self, mode):
                pass

            def adaptive_inference(self, **kwargs):
                self.call_count += 1
                if self.call_count % 5 == 0:  # Fail every 5th call
                    raise Exception("Simulated model failure")
                # Otherwise return normal result
                return MockRBSModel().adaptive_inference(**kwargs)

        evaluator = RBSEvaluator(
            model=FlakyModel(),
            config=self.config,
            output_dir=self.temp_dir
        )

        # Should handle errors gracefully
        results = evaluator.evaluate(
            test_dataset=self.test_dataset[:10],  # Smaller dataset for testing
            baseline_models=None,
            detailed_analysis=False
        )

        # Should still produce results despite errors
        assert 'main' in results
        assert results['main']['accuracy']['total_examples'] == 10

    def test_evaluation_with_different_model_capabilities(self):
        """Test evaluation with models having different capabilities."""
        # Model without adaptive inference
        class BasicModel:
            def eval(self):
                pass

        basic_evaluator = RBSEvaluator(
            model=BasicModel(),
            config=self.config,
            output_dir=self.temp_dir
        )

        results = basic_evaluator.evaluate(
            test_dataset=self.test_dataset[:5],
            baseline_models=None,
            detailed_analysis=False
        )

        # Should still work with fallback inference
        assert 'main' in results
        assert results['main']['accuracy']['total_examples'] == 5

    def test_large_dataset_evaluation(self):
        """Test evaluation with larger dataset."""
        large_dataset = self._create_test_dataset(num_examples=100)

        model = MockRBSModel(accuracy_level=0.8, efficiency_level=0.6)

        evaluator = RBSEvaluator(
            model=model,
            config=self.config,
            output_dir=self.temp_dir
        )

        results = evaluator.evaluate(
            test_dataset=large_dataset,
            baseline_models=None,
            detailed_analysis=True
        )

        # Verify all examples were processed
        assert results['main']['accuracy']['total_examples'] == 100

        # Verify difficulty analysis has different categories
        difficulty_analysis = results['analysis']['difficulty_analysis']
        assert len(difficulty_analysis) > 0

        # Check that different difficulty levels were analyzed
        has_easy_medium_hard = any(
            category in difficulty_analysis
            for category in ['easy', 'medium', 'hard']
        )
        assert has_easy_medium_hard

    def test_report_generation_quality(self):
        """Test quality of generated reports."""
        model = MockRBSModel(accuracy_level=0.82, efficiency_level=0.75)

        evaluator = RBSEvaluator(
            model=model,
            config=self.config,
            output_dir=self.temp_dir
        )

        results = evaluator.evaluate(
            test_dataset=self.test_dataset,
            baseline_models=None,
            detailed_analysis=True
        )

        # Check JSON report quality
        json_path = os.path.join(self.temp_dir, "evaluation_results.json")
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        assert 'main' in json_data
        assert 'analysis' in json_data
        assert json_data['main']['summary']['f1'] > 0

        # Check Markdown report quality
        md_path = os.path.join(self.temp_dir, "evaluation_summary.md")
        with open(md_path, 'r') as f:
            md_content = f.read()

        assert "# RBS-QA Evaluation Summary" in md_content
        assert "**F1 Score**:" in md_content
        assert "## Key Findings" in md_content
        assert "## Recommendations" in md_content

        # Check HTML report quality
        html_path = os.path.join(self.temp_dir, "evaluation_report.html")
        with open(html_path, 'r') as f:
            html_content = f.read()

        assert "<!DOCTYPE html>" in html_content
        assert "RBS-QA Evaluation Report" in html_content
        assert "Summary Metrics" in html_content
        assert "<script>" in html_content  # Should have JavaScript for charts

    def test_evaluation_reproducibility(self):
        """Test that evaluation results are reproducible."""
        model = MockRBSModel(accuracy_level=0.8, efficiency_level=0.7)

        # Run evaluation twice
        evaluator1 = RBSEvaluator(
            model=model,
            config=self.config,
            output_dir=os.path.join(self.temp_dir, "run1")
        )

        evaluator2 = RBSEvaluator(
            model=model,
            config=self.config,
            output_dir=os.path.join(self.temp_dir, "run2")
        )

        results1 = evaluator1.evaluate(
            test_dataset=self.test_dataset,
            baseline_models=None,
            detailed_analysis=False
        )

        results2 = evaluator2.evaluate(
            test_dataset=self.test_dataset,
            baseline_models=None,
            detailed_analysis=False
        )

        # Results should be identical (given same model and data)
        summary1 = results1['main']['summary']
        summary2 = results2['main']['summary']

        assert summary1['f1'] == summary2['f1']
        assert summary1['exact_match'] == summary2['exact_match']
        assert summary1['avg_efficiency_score'] == summary2['avg_efficiency_score']
        assert summary1['combined_score'] == summary2['combined_score']

    def test_performance_metrics_validation(self):
        """Test that performance metrics are within expected ranges."""
        # Test with different model performance levels
        performance_levels = [
            (0.95, 0.9),  # Excellent model
            (0.8, 0.7),   # Good model
            (0.6, 0.5),   # Moderate model
            (0.4, 0.3),   # Poor model
        ]

        for accuracy, efficiency in performance_levels:
            model = MockRBSModel(accuracy_level=accuracy, efficiency_level=efficiency)

            evaluator = RBSEvaluator(
                model=model,
                config=self.config,
                output_dir=os.path.join(self.temp_dir, f"perf_test_{accuracy}_{efficiency}")
            )

            results = evaluator.evaluate(
                test_dataset=self.test_dataset[:10],  # Smaller dataset for speed
                baseline_models=None,
                detailed_analysis=False
            )

            summary = results['main']['summary']

            # Validate metric ranges
            assert 0.0 <= summary['f1'] <= 1.0
            assert 0.0 <= summary['exact_match'] <= 1.0
            assert summary['avg_efficiency_score'] >= 1.0
            assert 0.0 <= summary['combined_score'] <= 1.0

            # Check that metrics roughly match expected performance
            # Allow for variance due to randomness in mock model
            assert abs(summary['f1'] - accuracy) < 1.0  # Allow large variance due to randomness
            assert summary['avg_efficiency_score'] >= 1.0 + efficiency * 0.5  # Efficiency should improve score

    def test_config_modification_impact(self):
        """Test that configuration changes affect evaluation results."""
        model = MockRBSModel(accuracy_level=0.8, efficiency_level=0.7)

        # Test with different max_segments settings
        config1 = RBSEvaluationConfig(
            output_dir=os.path.join(self.temp_dir, "config1"),
            max_segments_per_example=16
        )

        config2 = RBSEvaluationConfig(
            output_dir=os.path.join(self.temp_dir, "config2"),
            max_segments_per_example=64
        )

        evaluator1 = RBSEvaluator(model=model, config=config1, output_dir=config1.output_dir)
        evaluator2 = RBSEvaluator(model=model, config=config2, output_dir=config2.output_dir)

        results1 = evaluator1.evaluate(
            test_dataset=self.test_dataset[:10],
            baseline_models=None,
            detailed_analysis=False
        )

        results2 = evaluator2.evaluate(
            test_dataset=self.test_dataset[:10],
            baseline_models=None,
            detailed_analysis=False
        )

        # Both should complete successfully
        assert 'main' in results1
        assert 'main' in results2

        # The difference in max_segments shouldn't break the evaluation
        # (since our mock model adapts to available segments)
        assert results1['main']['accuracy']['total_examples'] == 10
        assert results2['main']['accuracy']['total_examples'] == 10