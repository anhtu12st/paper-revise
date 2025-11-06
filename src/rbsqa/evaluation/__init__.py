"""
RBS-QA Evaluation Module

Comprehensive evaluation framework for RBS-QA models including:
- Accuracy metrics (F1, Exact Match)
- Efficiency metrics (segments processed, time)
- Non-monotonic reasoning analysis
- Halting policy performance analysis
- Comparative analysis tools
"""

from .rbs_evaluator import RBSEvaluator
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

__all__ = [
    "RBSEvaluator",
    "RBSEvaluationConfig",
    "AccuracyMetricsCollector",
    "EfficiencyMetricsCollector",
    "ReasoningMetricsCollector",
    "HaltingMetricsCollector",
    "BeliefStateAnalyzer",
    "HaltingPolicyAnalyzer",
    "ComparativeAnalyzer",
]