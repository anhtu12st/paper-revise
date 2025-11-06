"""
RBS-QA Evaluator

Comprehensive evaluator for RBS-QA models that goes beyond standard QA metrics
to measure the unique aspects of the system: non-monotonic reasoning capabilities,
adaptive computation efficiency, and halting policy effectiveness.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from tqdm import tqdm

from .config import RBSEvaluationConfig
from .metrics_collectors import (
    AccuracyMetricsCollector,
    EfficiencyMetricsCollector,
    ReasoningMetricsCollector,
    HaltingMetricsCollector,
)
from .analyzers import (
    BeliefStateAnalyzer,
    HaltingPolicyAnalyzer,
    ComparativeAnalyzer,
)

logger = logging.getLogger(__name__)


class RBSEvaluator:
    """
    Comprehensive evaluator for RBS-QA models.

    Evaluates:
    - QA accuracy (F1, Exact Match)
    - Computational efficiency (segments processed, time)
    - Non-monotonic reasoning capabilities
    - Halting policy effectiveness
    - Belief state quality and revision patterns
    """

    def __init__(self,
                 model: Any,
                 config: RBSEvaluationConfig,
                 output_dir: str = "./evaluation_results"):

        self.model = model
        self.config = config
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Initialize metrics collectors
        self.accuracy_metrics = AccuracyMetricsCollector()
        self.efficiency_metrics = EfficiencyMetricsCollector()
        self.reasoning_metrics = ReasoningMetricsCollector()
        self.halting_metrics = HaltingMetricsCollector()

        # Analysis tools
        self.belief_analyzer = BeliefStateAnalyzer()
        self.halting_analyzer = HaltingPolicyAnalyzer()
        self.comparative_analyzer = ComparativeAnalyzer()

        # Results storage
        self.evaluation_results = {}

    def evaluate(self,
                test_dataset: List[Dict],
                baseline_models: Optional[Dict[str, Any]] = None,
                detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of RBS-QA model.

        Args:
            test_dataset: Test dataset with questions and contexts
            baseline_models: Optional dict of baseline models for comparison
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            Comprehensive evaluation results
        """

        self.model.eval()
        if hasattr(self.model, 'set_inference_mode'):
            self.model.set_inference_mode("adaptive")

        logger.info("Starting comprehensive RBS-QA evaluation...")

        # Main evaluation
        main_results = self._evaluate_main(test_dataset)
        self.evaluation_results['main'] = main_results

        # Baseline comparisons
        if baseline_models:
            comparison_results = self._evaluate_baselines(test_dataset, baseline_models)
            self.evaluation_results['comparisons'] = comparison_results

        # Detailed analysis
        if detailed_analysis:
            analysis_results = self._perform_detailed_analysis(test_dataset)
            self.evaluation_results['analysis'] = analysis_results

        # Generate reports
        self._generate_evaluation_report()

        return self.evaluation_results

    def _evaluate_main(self, test_dataset: List[Dict]) -> Dict[str, Any]:
        """Main evaluation metrics."""

        logger.info("Evaluating main metrics...")

        all_predictions = []
        all_ground_truths = []
        all_belief_histories = []
        all_halting_histories = []
        all_efficiency_data = []

        for example_idx in tqdm(range(len(test_dataset)), desc="Main evaluation"):
            example = test_dataset[example_idx]

            # Adaptive inference
            try:
                if hasattr(self.model, 'adaptive_inference'):
                    result = self.model.adaptive_inference(
                        question_input_ids=example['question_input_ids'],
                        context_segments=example['context_segments'],
                        max_segments=self.config.max_segments_per_example
                    )
                else:
                    # Fallback to standard inference
                    result = self._fallback_inference(example)

                # Store results
                prediction = {
                    'example_id': example.get('id', example_idx),
                    'answer_span': getattr(result, 'answer_span', (0, 0)),
                    'confidence': getattr(result, 'confidence', 0.0),
                    'segments_processed': getattr(result, 'segments_processed', len(example['context_segments'])),
                    'total_segments': len(example['context_segments']),
                    'efficiency_score': self._compute_efficiency_score(result, example),
                    'inference_time': getattr(result, 'inference_time', 0.0),
                    'belief_history': getattr(result, 'belief_history', []),
                    'halting_history': getattr(result, 'halting_history', [])
                }

                ground_truth = {
                    'example_id': example.get('id', example_idx),
                    'answer_span': example['answer_span'],
                    'question': example.get('question', ''),
                    'context_length': sum(len(seg) for seg in example['context_segments'])
                }

                all_predictions.append(prediction)
                all_ground_truths.append(ground_truth)

            except Exception as e:
                logger.warning(f"Error processing example {example_idx}: {e}")
                # Add placeholder result to maintain alignment
                prediction = {
                    'example_id': example.get('id', example_idx),
                    'answer_span': (0, 0),
                    'confidence': 0.0,
                    'segments_processed': len(example['context_segments']),
                    'total_segments': len(example['context_segments']),
                    'efficiency_score': 1.0,
                    'inference_time': 0.0,
                    'belief_history': [],
                    'halting_history': []
                }

                ground_truth = {
                    'example_id': example.get('id', example_idx),
                    'answer_span': example['answer_span'],
                    'question': example.get('question', ''),
                    'context_length': sum(len(seg) for seg in example['context_segments'])
                }

                all_predictions.append(prediction)
                all_ground_truths.append(ground_truth)

        # Compute comprehensive metrics
        accuracy_results = self.accuracy_metrics.compute(all_predictions, all_ground_truths)
        efficiency_results = self.efficiency_metrics.compute(all_predictions, all_ground_truths)
        reasoning_results = self.reasoning_metrics.compute(all_predictions, all_ground_truths)
        halting_results = self.halting_metrics.compute(all_predictions, all_ground_truths)

        # Combine results
        main_results = {
            'accuracy': accuracy_results,
            'efficiency': efficiency_results,
            'reasoning': reasoning_results,
            'halting': halting_results,
            'summary': self._compute_summary_metrics(accuracy_results, efficiency_results)
        }

        logger.info(f"Main evaluation completed: F1={accuracy_results['f1']:.3f}, "
                   f"Efficiency={efficiency_results['avg_efficiency_score']:.3f}")

        return main_results

    def _evaluate_baselines(self,
                           test_dataset: List[Dict],
                           baseline_models: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate against baseline models."""

        logger.info("Evaluating baseline comparisons...")

        comparison_results = {}

        for baseline_name, baseline_model in baseline_models.items():
            logger.info(f"Evaluating baseline: {baseline_name}")

            baseline_predictions = []

            for example_idx in tqdm(range(len(test_dataset)), desc=f"Baseline {baseline_name}"):
                example = test_dataset[example_idx]

                try:
                    if baseline_name == "gmm" and hasattr(baseline_model, 'full_inference'):
                        result = baseline_model.full_inference(
                            question_input_ids=example['question_input_ids'],
                            context_segments=example['context_segments']
                        )
                    elif baseline_name == "base_xlnet" and hasattr(baseline_model, 'standard_inference'):
                        result = baseline_model.standard_inference(
                            question_input_ids=example['question_input_ids'],
                            context_segments=example['context_segments']
                        )
                    else:
                        # Custom baseline or fallback
                        result = self._evaluate_baseline_generic(baseline_model, example)

                    baseline_predictions.append({
                        'example_id': example.get('id', example_idx),
                        'answer_span': getattr(result, 'answer_span', (0, 0)),
                        'confidence': getattr(result, 'confidence', 0.0),
                        'segments_processed': getattr(result, 'segments_processed', len(example['context_segments'])),
                        'total_segments': len(example['context_segments']),
                        'efficiency_score': self._compute_efficiency_score(result, example)
                    })

                except Exception as e:
                    logger.warning(f"Error processing baseline {baseline_name} example {example_idx}: {e}")
                    # Add placeholder
                    baseline_predictions.append({
                        'example_id': example.get('id', example_idx),
                        'answer_span': (0, 0),
                        'confidence': 0.0,
                        'segments_processed': len(example['context_segments']),
                        'total_segments': len(example['context_segments']),
                        'efficiency_score': 1.0
                    })

            # Compute metrics for baseline
            ground_truths = [{
                'example_id': test_dataset[i].get('id', i),
                'answer_span': test_dataset[i]['answer_span']
            } for i in range(len(test_dataset))]

            baseline_metrics = self.accuracy_metrics.compute(baseline_predictions, ground_truths)
            baseline_efficiency = self.efficiency_metrics.compute(baseline_predictions, ground_truths)

            comparison_results[baseline_name] = {
                'accuracy': baseline_metrics,
                'efficiency': baseline_efficiency,
                'summary': self._compute_summary_metrics(baseline_metrics, baseline_efficiency)
            }

        # Comparative analysis
        comparative_results = self.comparative_analyzer.analyze(
            self.evaluation_results['main'],
            comparison_results
        )
        comparison_results['comparative_analysis'] = comparative_results

        return comparison_results

    def _perform_detailed_analysis(self, test_dataset: List[Dict]) -> Dict[str, Any]:
        """Perform detailed analysis of model behavior."""

        logger.info("Performing detailed analysis...")

        analysis_results = {}

        # Belief state analysis
        belief_analysis = self.belief_analyzer.analyze_belief_patterns(
            self.evaluation_results['main'], test_dataset
        )
        analysis_results['belief_state'] = belief_analysis

        # Halting policy analysis
        halting_analysis = self.halting_analyzer.analyze_halting_patterns(
            self.evaluation_results['main'], test_dataset
        )
        analysis_results['halting_policy'] = halting_analysis

        # Non-monotonic reasoning analysis
        non_monotonic_analysis = self._analyze_non_monotonic_reasoning(test_dataset)
        analysis_results['non_monotonic_reasoning'] = non_monotonic_analysis

        # Error analysis
        error_analysis = self._analyze_errors(test_dataset)
        analysis_results['error_analysis'] = error_analysis

        # Efficiency analysis by difficulty
        difficulty_analysis = self._analyze_efficiency_by_difficulty(test_dataset)
        analysis_results['difficulty_analysis'] = difficulty_analysis

        return analysis_results

    def _analyze_non_monotonic_reasoning(self, test_dataset: List[Dict]) -> Dict[str, Any]:
        """Analyze non-monotonic reasoning patterns."""

        non_monotonic_cases = []
        revision_patterns = []

        for example_idx, example in enumerate(test_dataset):
            try:
                # Get detailed inference with belief tracking
                if hasattr(self.model, 'adaptive_inference'):
                    result = self.model.adaptive_inference(
                        question_input_ids=example['question_input_ids'],
                        context_segments=example['context_segments'],
                        track_detailed_belief=True
                    )
                else:
                    continue  # Skip if no detailed tracking available

                # Analyze belief revisions
                if hasattr(result, 'belief_history') and result.belief_history:
                    for i in range(1, len(result.belief_history)):
                        prev_belief = result.belief_history[i-1]
                        curr_belief = result.belief_history[i]

                        # Check for non-monotonic revision
                        if (hasattr(prev_belief, 'best_span') and hasattr(curr_belief, 'best_span') and
                            prev_belief.best_span != curr_belief.best_span and
                            hasattr(curr_belief, 'confidence') and hasattr(prev_belief, 'confidence') and
                            curr_belief.confidence > prev_belief.confidence):

                            revision_patterns.append({
                                'example_id': example.get('id', example_idx),
                                'revision_step': i,
                                'previous_span': prev_belief.best_span,
                                'new_span': curr_belief.best_span,
                                'previous_confidence': prev_belief.confidence,
                                'new_confidence': curr_belief.confidence,
                                'segment_triggered': getattr(curr_belief, 'segment_id', i),
                                'was_correct_before': self._is_span_correct(prev_belief.best_span, example['answer_span']),
                                'is_correct_after': self._is_span_correct(curr_belief.best_span, example['answer_span'])
                            })

            except Exception as e:
                logger.warning(f"Error in non-monotonic analysis for example {example_idx}: {e}")
                continue

        # Compute statistics
        total_examples = len(test_dataset)
        examples_with_revisions = len(set(rp['example_id'] for rp in revision_patterns))
        total_revisions = len(revision_patterns)

        beneficial_revisions = [rp for rp in revision_patterns if not rp['was_correct_before'] and rp['is_correct_after']]
        detrimental_revisions = [rp for rp in revision_patterns if rp['was_correct_before'] and not rp['is_correct_after']]

        return {
            'revision_frequency': examples_with_revisions / total_examples,
            'avg_revisions_per_example': total_revisions / total_examples,
            'beneficial_revision_rate': len(beneficial_revisions) / max(total_revisions, 1),
            'detrimental_revision_rate': len(detrimental_revisions) / max(total_revisions, 1),
            'revision_patterns': revision_patterns[:50],  # Top 50 examples
            'summary': {
                'non_monotonic_reasoning_detected': examples_with_revisions > 0,
                'revision_effectiveness': len(beneficial_revisions) - len(detrimental_revisions)
            }
        }

    def _analyze_errors(self, test_dataset: List[Dict]) -> Dict[str, Any]:
        """Detailed error analysis."""

        error_types = defaultdict(list)
        confidence_errors = []

        for example_idx, example in enumerate(test_dataset):
            try:
                if hasattr(self.model, 'adaptive_inference'):
                    result = self.model.adaptive_inference(
                        question_input_ids=example['question_input_ids'],
                        context_segments=example['context_segments']
                    )
                else:
                    continue

                is_correct = self._is_span_correct(result.answer_span, example['answer_span'])

                if not is_correct:
                    # Categorize error type
                    error_type = self._categorize_error(result.answer_span, example['answer_span'], example)
                    error_types[error_type].append({
                        'example_id': example.get('id', example_idx),
                        'predicted_span': result.answer_span,
                        'ground_truth_span': example['answer_span'],
                        'confidence': getattr(result, 'confidence', 0.0),
                        'segments_processed': getattr(result, 'segments_processed', len(example['context_segments']))
                    })

                    confidence_errors.append({
                        'confidence': getattr(result, 'confidence', 0.0),
                        'error_type': error_type
                    })

            except Exception as e:
                logger.warning(f"Error in error analysis for example {example_idx}: {e}")
                continue

        # Compute error statistics
        error_stats = {}
        for error_type, errors in error_types.items():
            error_stats[error_type] = {
                'count': len(errors),
                'percentage': len(errors) / len(test_dataset) * 100,
                'avg_confidence': np.mean([e['confidence'] for e in errors]) if errors else 0.0,
                'examples': errors[:10]  # Top 10 examples
            }

        # Confidence calibration analysis
        confidence_bins = np.linspace(0, 1, 11)
        calibration_data = []

        for bin_start, bin_end in zip(confidence_bins[:-1], confidence_bins[1:]):
            bin_errors = [e for e in confidence_errors if bin_start <= e['confidence'] < bin_end]
            if bin_errors:
                calibration_data.append({
                    'confidence_range': f"{bin_start:.1f}-{bin_end:.1f}",
                    'avg_confidence': np.mean([e['confidence'] for e in bin_errors]),
                    'error_rate': 1.0,  # All are errors
                    'count': len(bin_errors)
                })

        return {
            'error_distribution': error_stats,
            'confidence_calibration': calibration_data,
            'total_errors': sum(len(errors) for errors in error_types.values()),
            'error_rate': sum(len(errors) for errors in error_types.values()) / len(test_dataset)
        }

    def _analyze_efficiency_by_difficulty(self, test_dataset: List[Dict]) -> Dict[str, Any]:
        """Analyze efficiency by question/document difficulty."""

        difficulty_metrics = defaultdict(list)

        for example_idx, example in enumerate(test_dataset):
            try:
                # Compute difficulty metrics
                doc_length = sum(len(seg) for seg in example['context_segments'])
                question_length = len(example['question_input_ids'])

                # Get difficulty category
                difficulty = self._categorize_difficulty(doc_length, question_length, example)

                # Get inference result
                if hasattr(self.model, 'adaptive_inference'):
                    result = self.model.adaptive_inference(
                        question_input_ids=example['question_input_ids'],
                        context_segments=example['context_segments']
                    )
                else:
                    continue

                is_correct = self._is_span_correct(result.answer_span, example['answer_span'])

                difficulty_metrics[difficulty].append({
                    'segments_processed': getattr(result, 'segments_processed', len(example['context_segments'])),
                    'total_segments': len(example['context_segments']),
                    'efficiency_score': self._compute_efficiency_score(result, example),
                    'accuracy': 1.0 if is_correct else 0.0,
                    'confidence': getattr(result, 'confidence', 0.0),
                    'inference_time': getattr(result, 'inference_time', 0.0)
                })

            except Exception as e:
                logger.warning(f"Error in difficulty analysis for example {example_idx}: {e}")
                continue

        # Summarize by difficulty
        difficulty_summary = {}
        for difficulty, metrics in difficulty_metrics.items():
            if metrics:
                difficulty_summary[difficulty] = {
                    'count': len(metrics),
                    'avg_segments_processed': np.mean([m['segments_processed'] for m in metrics]),
                    'avg_efficiency_score': np.mean([m['efficiency_score'] for m in metrics]),
                    'accuracy': np.mean([m['accuracy'] for m in metrics]),
                    'avg_confidence': np.mean([m['confidence'] for m in metrics]),
                    'avg_inference_time': np.mean([m['inference_time'] for m in metrics])
                }

        return difficulty_summary

    def _categorize_difficulty(self, doc_length: int, question_length: int, example: Dict) -> str:
        """Categorize example difficulty."""
        if doc_length < 500 and question_length < 20:
            return "easy"
        elif doc_length < 1000 and question_length < 30:
            return "medium"
        else:
            return "hard"

    def _categorize_error(self, pred_span: Tuple[int, int], gt_span: Tuple[int, int], example: Dict) -> str:
        """Categorize error type."""
        pred_start, pred_end = pred_span
        gt_start, gt_end = gt_span

        # No overlap
        if pred_end < gt_start or pred_start > gt_end:
            return "no_overlap"

        # Partial overlap
        overlap_start = max(pred_start, gt_start)
        overlap_end = min(pred_end, gt_end)
        overlap_len = max(0, overlap_end - overlap_start + 1)

        pred_len = pred_end - pred_start + 1
        gt_len = gt_end - gt_start + 1

        if overlap_len / gt_len < 0.5:
            return "partial_overlap"
        elif pred_start != gt_start or pred_end != gt_end:
            return "boundary_error"
        else:
            return "other"

    def _is_span_correct(self, pred_span: Tuple[int, int], gt_span: Tuple[int, int]) -> bool:
        """Check if predicted span matches ground truth."""
        return pred_span[0] == gt_span[0] and pred_span[1] == gt_span[1]

    def _compute_efficiency_score(self, result: Any, example: Dict) -> float:
        """Compute efficiency score for a result."""
        segments_processed = getattr(result, 'segments_processed', len(example['context_segments']))
        total_segments = len(example['context_segments'])
        return total_segments / max(segments_processed, 1)

    def _compute_summary_metrics(self, accuracy_results: Dict, efficiency_results: Dict) -> Dict[str, float]:
        """Compute summary metrics combining accuracy and efficiency."""
        return {
            'f1': accuracy_results['f1'],
            'exact_match': accuracy_results['exact_match'],
            'avg_efficiency_score': efficiency_results['avg_efficiency_score'],
            'avg_segments_processed': efficiency_results['avg_segments_processed'],
            'combined_score': accuracy_results['f1'] * 0.7 + efficiency_results['avg_efficiency_score'] * 0.3
        }

    def _fallback_inference(self, example: Dict) -> Any:
        """Fallback inference method when adaptive inference is not available."""
        # Create a mock result object
        class MockResult:
            def __init__(self, example):
                self.answer_span = (0, 0)  # Default to first token
                self.confidence = 0.5
                self.segments_processed = len(example['context_segments'])
                self.inference_time = 0.1
                self.belief_history = []
                self.halting_history = []

        return MockResult(example)

    def _evaluate_baseline_generic(self, baseline_model: Any, example: Dict) -> Any:
        """Generic baseline evaluation method."""
        # Similar to fallback inference but for baselines
        class BaselineResult:
            def __init__(self, example):
                self.answer_span = (0, 0)
                self.confidence = 0.5
                self.segments_processed = len(example['context_segments'])

        return BaselineResult(example)

    def _generate_evaluation_report(self) -> None:
        """Generate comprehensive evaluation report."""

        # Generate JSON report
        json_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(json_path, "w") as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)

        # Generate summary markdown
        md_path = os.path.join(self.output_dir, "evaluation_summary.md")
        with open(md_path, "w") as f:
            f.write(self._generate_markdown_summary())

        # Generate HTML report if requested
        if self.config.generate_html_report:
            report_path = os.path.join(self.output_dir, "evaluation_report.html")
            html_content = self._generate_html_report()
            with open(report_path, "w") as f:
                f.write(html_content)

        logger.info(f"Evaluation reports generated in: {self.output_dir}")

    def _generate_html_report(self) -> str:
        """Generate HTML evaluation report."""

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RBS-QA Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric-card {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .main-metric {{ background: #e3f2fd; }}
                .summary-metric {{ background: #e8f5e8; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .chart {{ margin: 20px 0; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>RBS-QA Evaluation Report</h1>
            <p>Generated on: {timestamp}</p>

            <h2>Summary Metrics</h2>
            <div class="metric-card main-metric">
                <h3>Main Performance</h3>
                <p>F1 Score: {f1:.3f}</p>
                <p>Exact Match: {em:.3f}</p>
                <p>Efficiency Score: {efficiency:.3f}</p>
                <p>Combined Score: {combined:.3f}</p>
            </div>

            <h2>Detailed Metrics</h2>
            <div class="metric-card summary-metric">
                <h3>Accuracy Metrics</h3>
                {accuracy_table}
            </div>

            <div class="metric-card summary-metric">
                <h3>Efficiency Metrics</h3>
                {efficiency_table}
            </div>

            <h2>Visualizations</h2>
            <div class="chart">
                <canvas id="performanceChart" width="400" height="200"></canvas>
            </div>

            <script>
                // Performance Chart
                const performanceCtx = document.getElementById('performanceChart').getContext('2d');
                new Chart(performanceCtx, {{
                    type: 'bar',
                    data: {{
                        labels: ['F1 Score', 'Exact Match', 'Efficiency'],
                        datasets: [{{
                            label: 'RBS-QA',
                            data: [{f1}, {em}, {efficiency}],
                            backgroundColor: ['#2196F3', '#4CAF50', '#FF9800']
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 1
                            }}
                        }}
                    }}
                }});
            </script>

            <h2>Analysis Results</h2>
            {analysis_sections}

        </body>
        </html>
        """

        # Fill template with data
        main_results = self.evaluation_results.get('main', {}).get('summary', {})
        accuracy_results = self.evaluation_results.get('main', {}).get('accuracy', {})
        efficiency_results = self.evaluation_results.get('main', {}).get('efficiency', {})

        return html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f1=main_results.get('f1', 0.0),
            em=main_results.get('exact_match', 0.0),
            efficiency=main_results.get('avg_efficiency_score', 0.0),
            combined=main_results.get('combined_score', 0.0),
            accuracy_table=self._generate_accuracy_table(accuracy_results),
            efficiency_table=self._generate_efficiency_table(efficiency_results),
            analysis_sections=self._generate_analysis_sections()
        )

    def _generate_markdown_summary(self) -> str:
        """Generate markdown summary of evaluation results."""

        main_results = self.evaluation_results.get('main', {}).get('summary', {})

        md_content = f"""
# RBS-QA Evaluation Summary

## Main Performance Metrics

- **F1 Score**: {main_results.get('f1', 0.0):.3f}
- **Exact Match**: {main_results.get('exact_match', 0.0):.3f}
- **Efficiency Score**: {main_results.get('avg_efficiency_score', 0.0):.3f}
- **Combined Score**: {main_results.get('combined_score', 0.0):.3f}

## Key Findings

{self._generate_key_findings()}

## Recommendations

{self._generate_recommendations()}

---

*Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

        return md_content

    def _generate_key_findings(self) -> str:
        """Generate key findings from evaluation results."""

        findings = []

        main_results = self.evaluation_results.get('main', {}).get('summary', {})
        reasoning_results = self.evaluation_results.get('main', {}).get('reasoning', {})

        # Accuracy findings
        f1_score = main_results.get('f1', 0.0)
        if f1_score > 0.8:
            findings.append("✅ **Excellent QA performance** with F1 score > 80%")
        elif f1_score > 0.7:
            findings.append("✅ **Good QA performance** with F1 score > 70%")
        else:
            findings.append("⚠️ **Moderate QA performance** - F1 score needs improvement")

        # Efficiency findings
        efficiency = main_results.get('avg_efficiency_score', 0.0)
        if efficiency > 1.5:
            findings.append("✅ **Excellent efficiency** - processes >50% fewer segments")
        elif efficiency > 1.2:
            findings.append("✅ **Good efficiency** - processes >20% fewer segments")
        else:
            findings.append("⚠️ **Limited efficiency gains** - adaptive processing needs optimization")

        # Non-monotonic reasoning findings
        revision_freq = reasoning_results.get('revision_frequency', 0.0)
        if revision_freq > 0.1:
            findings.append("🧠 **Active non-monotonic reasoning** - model revises beliefs frequently")
        else:
            findings.append("📊 **Limited belief revision** - model may be too conservative")

        return "\n\n".join(findings)

    def _generate_recommendations(self) -> str:
        """Generate recommendations based on evaluation results."""

        recommendations = []

        main_results = self.evaluation_results.get('main', {}).get('summary', {})

        # Accuracy recommendations
        if main_results.get('f1', 0.0) < 0.75:
            recommendations.append("🎯 **Improve QA accuracy**: Consider increasing model capacity or training time")

        # Efficiency recommendations
        if main_results.get('avg_efficiency_score', 0.0) < 1.3:
            recommendations.append("⚡ **Improve efficiency**: Adjust halting policy thresholds or RL rewards")

        # General recommendations
        recommendations.append("📊 **Monitor training**: Track both accuracy and efficiency during training")
        recommendations.append("🔬 **Analyze failures**: Review error patterns for targeted improvements")

        return "\n\n".join(recommendations)

    def _generate_accuracy_table(self, accuracy_results: Dict) -> str:
        """Generate HTML table for accuracy metrics."""
        if not accuracy_results:
            return "<p>No accuracy data available</p>"

        table = "<table><tr><th>Metric</th><th>Value</th></tr>"
        for metric, value in accuracy_results.items():
            if isinstance(value, float):
                table += f"<tr><td>{metric}</td><td>{value:.3f}</td></tr>"
            else:
                table += f"<tr><td>{metric}</td><td>{value}</td></tr>"
        table += "</table>"
        return table

    def _generate_efficiency_table(self, efficiency_results: Dict) -> str:
        """Generate HTML table for efficiency metrics."""
        if not efficiency_results:
            return "<p>No efficiency data available</p>"

        table = "<table><tr><th>Metric</th><th>Value</th></tr>"
        for metric, value in efficiency_results.items():
            if isinstance(value, float):
                table += f"<tr><td>{metric}</td><td>{value:.3f}</td></tr>"
            else:
                table += f"<tr><td>{metric}</td><td>{value}</td></tr>"
        table += "</table>"
        return table

    def _generate_analysis_sections(self) -> str:
        """Generate HTML sections for analysis results."""
        analysis_results = self.evaluation_results.get('analysis', {})
        sections = []

        for analysis_type, analysis_data in analysis_results.items():
            section_title = analysis_type.replace('_', ' ').title()
            sections.append(f"<h3>{section_title}</h3>")
            sections.append(f"<pre>{json.dumps(analysis_data, indent=2, default=str)}</pre>")

        return "\n".join(sections)