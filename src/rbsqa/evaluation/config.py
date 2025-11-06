"""
RBS-QA Evaluation Configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RBSEvaluationConfig:
    """Configuration for RBS evaluation."""

    output_dir: str = "./evaluation_results"
    max_segments_per_example: int = 32
    generate_visualizations: bool = True
    save_detailed_results: bool = True
    baseline_comparisons: List[str] = field(default_factory=lambda: ["gmm", "base_xlnet"])
    statistical_tests: bool = True
    error_analysis_depth: int = 50  # Number of examples to analyze in detail
    generate_html_report: bool = True
    generate_markdown_report: bool = True
    confidence_thresholds: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    efficiency_bins: int = 10
    include_qualitative_analysis: bool = True
    save_belief_states: bool = False
    verbose_logging: bool = True