#!/usr/bin/env python3
"""
Analyze and Visualize Paper Experiment Results
===============================================

This script creates publication-ready figures and analysis
from all paper experiments.

Generates:
- Memory token count vs F1 plot
- Ablation study comparison
- Segment length analysis
- Training curves comparison

Usage:
    python scripts/paper_experiments/analyze_results.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all experiment results."""
    results_file = results_dir / "all_results.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}\nPlease run evaluate_all.py first")

    with open(results_file) as f:
        return json.load(f)


def plot_memory_token_comparison(results: list[dict], output_dir: Path):
    """Plot F1 score vs memory token count."""
    logger.info("Creating memory token comparison plot...")

    # Extract memory token experiments
    token_experiments = [
        ("01_baseline_no_memory", 0),
        ("06_ablation_4tokens", 4),
        ("02_main_memory_8tokens", 8),
        ("03_main_memory_16tokens", 16),
        ("07_ablation_32tokens", 32),
    ]

    tokens = []
    f1_scores = []
    em_scores = []

    for exp_id, token_count in token_experiments:
        result = next((r for r in results if r.get("experiment_id") == exp_id), None)
        if result and result.get("status") == "success":
            tokens.append(token_count)
            f1_scores.append(result["metrics"]["f1"])
            em_scores.append(result["metrics"]["exact_match"])

    if not tokens:
        logger.warning("No memory token results found")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(tokens, f1_scores, "o-", linewidth=2, markersize=8, label="F1 Score")
    ax.plot(tokens, em_scores, "s--", linewidth=2, markersize=8, label="Exact Match")

    ax.set_xlabel("Number of Memory Tokens", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Performance vs Memory Token Count", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "memory_tokens_comparison.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_ablation_comparison(results: list[dict], output_dir: Path):
    """Plot ablation study comparison."""
    logger.info("Creating ablation comparison plot...")

    # Ablation experiments
    ablation_experiments = [
        ("02_main_memory_8tokens", "Full Model\n(8 tokens)"),
        ("01_baseline_no_memory", "No Memory"),
        ("04_ablation_no_progressive", "No Progressive"),
        ("05_ablation_no_gating", "No Gating"),
    ]

    names = []
    f1_scores = []
    has_ans_f1 = []
    no_ans_f1 = []

    for exp_id, label in ablation_experiments:
        result = next((r for r in results if r.get("experiment_id") == exp_id), None)
        if result and result.get("status") == "success":
            names.append(label)
            metrics = result["metrics"]
            f1_scores.append(metrics["f1"])
            has_ans_f1.append(metrics["has_answer_f1"])
            no_ans_f1.append(metrics["no_answer_f1"])

    if not names:
        logger.warning("No ablation results found")
        return

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(names))
    width = 0.25

    ax.bar([i - width for i in x], f1_scores, width, label="Overall F1", alpha=0.8)
    ax.bar(x, has_ans_f1, width, label="Has-Answer F1", alpha=0.8)
    ax.bar([i + width for i in x], no_ans_f1, width, label="No-Answer F1", alpha=0.8)

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title("Ablation Study Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / "ablation_comparison.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_segment_analysis(results: list[dict], output_dir: Path):
    """Plot performance vs maximum segment count."""
    logger.info("Creating segment analysis plot...")

    # Segment experiments
    segment_experiments = [
        ("08_segments_2seg", 2),
        ("09_segments_4seg", 4),
        ("10_segments_6seg", 6),
    ]

    segments = []
    f1_scores = []
    em_scores = []

    for exp_id, seg_count in segment_experiments:
        result = next((r for r in results if r.get("experiment_id") == exp_id), None)
        if result and result.get("status") == "success":
            segments.append(seg_count)
            f1_scores.append(result["metrics"]["f1"])
            em_scores.append(result["metrics"]["exact_match"])

    if not segments:
        logger.warning("No segment results found")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(segments, f1_scores, "o-", linewidth=2, markersize=8, label="F1 Score")
    ax.plot(segments, em_scores, "s--", linewidth=2, markersize=8, label="Exact Match")

    ax.set_xlabel("Maximum Segments per Document", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Performance vs Document Length", fontsize=14, fontweight="bold")
    ax.set_xticks(segments)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "segment_analysis.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def create_summary_statistics(results: list[dict], output_dir: Path):
    """Create summary statistics table."""
    logger.info("Creating summary statistics...")

    # Group experiments
    groups = {
        "Baseline": ["01_baseline_no_memory"],
        "Main": ["02_main_memory_8tokens", "03_main_memory_16tokens"],
        "Ablations": [
            "04_ablation_no_progressive",
            "05_ablation_no_gating",
            "06_ablation_4tokens",
            "07_ablation_32tokens",
        ],
        "Segments": ["08_segments_2seg", "09_segments_4seg", "10_segments_6seg"],
    }

    summary = {}
    for group_name, exp_ids in groups.items():
        group_results = [r for r in results if r.get("experiment_id") in exp_ids and r.get("status") == "success"]

        if group_results:
            f1_scores = [r["metrics"]["f1"] for r in group_results]
            em_scores = [r["metrics"]["exact_match"] for r in group_results]

            summary[group_name] = {
                "count": len(group_results),
                "avg_f1": sum(f1_scores) / len(f1_scores),
                "max_f1": max(f1_scores),
                "min_f1": min(f1_scores),
                "avg_em": sum(em_scores) / len(em_scores),
            }

    # Save as JSON
    summary_file = output_dir / "summary_statistics.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80 + "\n")

    for group_name, stats in summary.items():
        print(f"{group_name}:")
        print(f"  • Experiments: {stats['count']}")
        print(f"  • Avg F1: {stats['avg_f1']:.2f}%")
        print(f"  • Max F1: {stats['max_f1']:.2f}%")
        print(f"  • Min F1: {stats['min_f1']:.2f}%")
        print(f"  • Avg EM: {stats['avg_em']:.2f}%")
        print()

    logger.info(f"✅ Saved: {summary_file}")


def main():
    """Analyze all experiment results."""
    print("\n" + "=" * 80)
    print("📊 PAPER EXPERIMENTS: ANALYZE RESULTS")
    print("=" * 80 + "\n")

    results_dir = Path("./results/paper_experiments")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    logger.info("Loading experiment results...")
    results = load_results(results_dir)

    successful = [r for r in results if r.get("status") == "success"]
    logger.info(f"Found {len(successful)} successful experiments")

    if not successful:
        logger.error("No successful experiments found!")
        logger.error("Please run training and evaluation first.")
        return

    # Create plots
    plot_memory_token_comparison(results, figures_dir)
    plot_ablation_comparison(results, figures_dir)
    plot_segment_analysis(results, figures_dir)

    # Create summary
    create_summary_statistics(results, results_dir)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📁 Figures saved to: {figures_dir}")
    print(f"📁 Statistics saved to: {results_dir}")
    print("\nGenerated files:")
    print("  • memory_tokens_comparison.pdf/png")
    print("  • ablation_comparison.pdf/png")
    print("  • segment_analysis.pdf/png")
    print("  • summary_statistics.json")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
