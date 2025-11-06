"""
RBS-QA Metrics Collectors

Classes for collecting and computing various evaluation metrics for RBS-QA models.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class AccuracyMetricsCollector:
    """Collects and computes accuracy metrics."""

    def compute(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
        """Compute standard QA accuracy metrics."""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")

        total_f1 = 0.0
        total_em = 0.0
        total_precision = 0.0
        total_recall = 0.0

        for pred, gt in zip(predictions, ground_truths):
            f1, precision, recall = self._compute_span_metrics(
                pred['answer_span'], gt['answer_span']
            )
            em = float(pred['answer_span'][0] == gt['answer_span'][0] and
                       pred['answer_span'][1] == gt['answer_span'][1])

            total_f1 += f1
            total_em += em
            total_precision += precision
            total_recall += recall

        n = len(predictions)
        return {
            'f1': total_f1 / n,
            'exact_match': total_em / n,
            'precision': total_precision / n,
            'recall': total_recall / n,
            'total_examples': n
        }

    def _compute_span_metrics(self, pred_span: Tuple[int, int], gt_span: Tuple[int, int]) -> Tuple[float, float, float]:
        """Compute precision, recall, F1 for spans."""
        pred_start, pred_end = pred_span
        gt_start, gt_end = gt_span

        pred_tokens = set(range(pred_start, pred_end + 1))
        gt_tokens = set(range(gt_start, gt_end + 1))

        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0, 1.0, 1.0
        elif len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0, 0.0, 0.0

        intersection = len(pred_tokens & gt_tokens)
        precision = intersection / len(pred_tokens)
        recall = intersection / len(gt_tokens)

        if precision + recall == 0:
            return 0.0, 0.0, 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall


class EfficiencyMetricsCollector:
    """Collects and computes efficiency metrics."""

    def compute(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
        """Compute efficiency metrics."""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")

        total_efficiency = 0.0
        total_segments_processed = 0
        total_segments_available = 0
        total_time_saved = 0.0
        efficiency_scores = []

        for pred, gt in zip(predictions, ground_truths):
            efficiency = pred['efficiency_score']
            segments_processed = pred['segments_processed']
            total_segments = pred['total_segments']

            total_efficiency += efficiency
            total_segments_processed += segments_processed
            total_segments_available += total_segments
            efficiency_scores.append(efficiency)

            # Time saved assuming linear scaling
            time_saved = (total_segments - segments_processed) / max(total_segments, 1)
            total_time_saved += time_saved

        n = len(predictions)

        # Compute additional statistics
        efficiency_scores = np.array(efficiency_scores)

        return {
            'avg_efficiency_score': total_efficiency / n,
            'avg_segments_processed': total_segments_processed / n,
            'avg_total_segments': total_segments_available / n,
            'avg_time_saved': total_time_saved / n,
            'efficiency_std': np.std(efficiency_scores),
            'efficiency_median': np.median(efficiency_scores),
            'efficiency_min': np.min(efficiency_scores),
            'efficiency_max': np.max(efficiency_scores),
            'total_examples': n
        }


class ReasoningMetricsCollector:
    """Collects and computes reasoning metrics."""

    def compute(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
        """Compute non-monotonic reasoning metrics."""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")

        examples_with_revisions = 0
        total_revisions = 0
        beneficial_revisions = 0
        total_confidence_improvement = 0.0
        confidence_changes = []

        for pred, gt in zip(predictions, ground_truths):
            belief_history = pred.get('belief_history', [])

            if len(belief_history) > 1:
                examples_with_revisions += 1

                # Count revisions
                revisions = 0
                confidence_improvements = []

                for i in range(1, len(belief_history)):
                    if belief_history[i].best_span != belief_history[i-1].best_span:
                        revisions += 1

                        # Check if revision was beneficial
                        prev_correct = self._is_span_correct(belief_history[i-1].best_span, gt['answer_span'])
                        curr_correct = self._is_span_correct(belief_history[i].best_span, gt['answer_span'])

                        if not prev_correct and curr_correct:
                            beneficial_revisions += 1

                        # Confidence change
                        conf_change = belief_history[i].confidence - belief_history[i-1].confidence
                        confidence_improvements.append(conf_change)
                        confidence_changes.append(conf_change)

                total_revisions += revisions
                if confidence_improvements:
                    total_confidence_improvement += np.mean(confidence_improvements)

        n = len(predictions)
        confidence_changes = np.array(confidence_changes)

        return {
            'revision_frequency': examples_with_revisions / n,
            'avg_revisions_per_example': total_revisions / n,
            'beneficial_revision_rate': beneficial_revisions / max(total_revisions, 1),
            'avg_confidence_improvement': total_confidence_improvement / max(examples_with_revisions, 1),
            'confidence_change_std': np.std(confidence_changes) if len(confidence_changes) > 0 else 0.0,
            'confidence_change_mean': np.mean(confidence_changes) if len(confidence_changes) > 0 else 0.0,
            'total_examples': n
        }

    def _is_span_correct(self, pred_span: Tuple[int, int], gt_span: Tuple[int, int]) -> bool:
        """Check if span is correct."""
        return pred_span[0] == gt_span[0] and pred_span[1] == gt_span[1]


class HaltingMetricsCollector:
    """Collects and computes halting policy metrics."""

    def compute(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
        """Compute halting policy metrics."""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")

        total_halting_decisions = 0
        early_halts = 0
        late_halts = 0
        correct_halts = 0
        halt_positions = []

        for pred, gt in zip(predictions, ground_truths):
            halting_history = pred.get('halting_history', [])

            if halting_history:
                total_halting_decisions += 1

                # Find halting decision
                halt_decision = None
                for decision in halting_history:
                    if hasattr(decision, 'action') and decision.action == "HALT":
                        halt_decision = decision
                        break
                    elif isinstance(decision, dict) and decision.get('action') == "HALT":
                        halt_decision = decision
                        break

                if halt_decision:
                    # Categorize halting timing
                    segments_processed = pred['segments_processed']
                    total_segments = pred['total_segments']

                    # Store relative halt position
                    relative_position = segments_processed / max(total_segments, 1)
                    halt_positions.append(relative_position)

                    if relative_position < 0.5:
                        early_halts += 1
                    elif relative_position > 0.8:
                        late_halts += 1

                    # Check if halting led to correct answer
                    is_correct = self._is_span_correct(pred['answer_span'], gt['answer_span'])
                    if is_correct:
                        correct_halts += 1

        n = len(predictions)
        halt_positions = np.array(halt_positions)

        return {
            'halting_decision_rate': total_halting_decisions / n,
            'early_halt_rate': early_halts / max(total_halting_decisions, 1),
            'late_halt_rate': late_halts / max(total_halting_decisions, 1),
            'halt_accuracy': correct_halts / max(total_halting_decisions, 1),
            'avg_halt_position': np.mean(halt_positions) if len(halt_positions) > 0 else 0.0,
            'halt_position_std': np.std(halt_positions) if len(halt_positions) > 0 else 0.0,
            'total_examples': n
        }

    def _is_span_correct(self, pred_span: Tuple[int, int], gt_span: Tuple[int, int]) -> bool:
        """Check if span is correct."""
        return pred_span[0] == gt_span[0] and pred_span[1] == gt_span[1]