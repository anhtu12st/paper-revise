#!/usr/bin/env python3
"""
Analyze and Visualize Hybrid Experiment Results
================================================

Creates publication-quality figures for the paper from evaluation results.

Generates:
1. Standard SQuAD v2 comparison (baseline vs memory variants)
2. Long SQuAD v2 comparison (showing memory benefits on long docs)
3. Memory scaling analysis (4, 8, 16, 32 tokens)
4. Segment length analysis (2, 4, 6 segments)
5. Ablation study results
6. Combined document length vs improvement plot

Usage:
    python scripts/paper_experiments_v2/analyze_results.py

Requires:
    - outputs/paper_v2_evaluation_results/all_results.json (from evaluate_all.py)
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["figure.dpi"] = 300


def load_results(results_path: Path):
    """Load evaluation results."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    # Filter out failed/not trained experiments
    completed = [r for r in results if r.get("status") not in ["failed", "not_trained"]]

    logger.info(f"Loaded {len(completed)}/{len(results)} completed experiments")
    return completed


def plot_squad_comparison(results, output_dir):
    """Plot Standard SQuAD v2 comparison."""
    squad_results = [r for r in results if r["dataset"] == "Standard SQuAD v2"]

    if not squad_results:
        logger.warning("No Standard SQuAD v2 results found")
        return

    # Extract data
    experiments = [r["experiment_name"] for r in squad_results]
    f1_scores = [r.get("f1", 0) for r in squad_results]
    em_scores = [r.get("exact_match", 0) for r in squad_results]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x - width / 2, f1_scores, width, label="F1", alpha=0.8, color="#2ecc71")
    bars2 = ax.bar(x + width / 2, em_scores, width, label="EM", alpha=0.8, color="#3498db")

    # Formatting
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score (%)")
    ax.set_title("Standard SQuAD v2 Results (Short Documents, 1-2 Segments)")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    # Save
    for ext in ["pdf", "png"]:
        output_path = output_dir / f"squad_v2_comparison.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Saved: {output_path}")

    plt.close()


def plot_long_squad_comparison(results, output_dir):
    """Plot Long SQuAD v2 comparison."""
    long_results = [r for r in results if r["dataset"] == "Long SQuAD v2"]

    if not long_results:
        logger.warning("No Long SQuAD v2 results found")
        return

    # Extract data
    experiments = [r["experiment_name"] for r in long_results]
    f1_scores = [r.get("f1", 0) for r in long_results]
    em_scores = [r.get("exact_match", 0) for r in long_results]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x - width / 2, f1_scores, width, label="F1", alpha=0.8, color="#e74c3c")
    bars2 = ax.bar(x + width / 2, em_scores, width, label="EM", alpha=0.8, color="#e67e22")

    # Formatting
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score (%)")
    ax.set_title("Long SQuAD v2 Results (Long Documents, 6-12 Segments)")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    # Save
    for ext in ["pdf", "png"]:
        output_path = output_dir / f"long_squad_v2_comparison.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Saved: {output_path}")

    plt.close()


def plot_memory_scaling(results, output_dir):
    """Plot memory token scaling analysis."""
    # Get memory scaling experiments from Standard SQuAD v2
    memory_experiments = {
        0: next((r for r in results if r["experiment_id"] == "01"), None),  # Baseline
        4: next((r for r in results if r["experiment_id"] == "04"), None),  # 4 tokens
        8: next((r for r in results if r["experiment_id"] == "02"), None),  # 8 tokens
        16: next((r for r in results if r["experiment_id"] == "05"), None),  # 16 tokens
        32: next((r for r in results if r["experiment_id"] == "06"), None),  # 32 tokens
    }

    # Filter out None values
    memory_experiments = {k: v for k, v in memory_experiments.items() if v is not None}

    if len(memory_experiments) < 2:
        logger.warning("Not enough memory scaling experiments found")
        return

    # Extract data
    memory_sizes = sorted(memory_experiments.keys())
    f1_scores = [memory_experiments[m].get("f1", 0) for m in memory_sizes]
    em_scores = [memory_experiments[m].get("exact_match", 0) for m in memory_sizes]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(memory_sizes, f1_scores, marker="o", linewidth=2, markersize=8, label="F1", color="#2ecc71")
    ax.plot(memory_sizes, em_scores, marker="s", linewidth=2, markersize=8, label="EM", color="#3498db")

    # Formatting
    ax.set_xlabel("Number of Memory Tokens")
    ax.set_ylabel("Score (%)")
    ax.set_title("Memory Token Scaling Analysis (Standard SQuAD v2)")
    ax.set_xticks(memory_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, (mem, f1, em) in enumerate(zip(memory_sizes, f1_scores, em_scores)):
        ax.annotate(f"{f1:.1f}", (mem, f1), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
        ax.annotate(f"{em:.1f}", (mem, em), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=10)

    plt.tight_layout()

    # Save
    for ext in ["pdf", "png"]:
        output_path = output_dir / f"memory_scaling.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Saved: {output_path}")

    plt.close()


def plot_segment_analysis(results, output_dir):
    """Plot segment length analysis."""
    # Get segment experiments from Long SQuAD v2
    segment_experiments = {
        2: next((r for r in results if r["experiment_id"] == "02"), None),  # Standard SQuAD v2 main
        4: next((r for r in results if r["experiment_id"] == "11"), None),  # 4 segments
        6: next((r for r in results if r["experiment_id"] == "12"), None),  # 6 segments
    }

    # Filter out None values
    segment_experiments = {k: v for k, v in segment_experiments.items() if v is not None}

    if len(segment_experiments) < 2:
        logger.warning("Not enough segment experiments found")
        return

    # Extract data
    segments = sorted(segment_experiments.keys())
    f1_scores = [segment_experiments[s].get("f1", 0) for s in segments]
    em_scores = [segment_experiments[s].get("exact_match", 0) for s in segments]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(segments, f1_scores, marker="o", linewidth=2, markersize=8, label="F1", color="#e74c3c")
    ax.plot(segments, em_scores, marker="s", linewidth=2, markersize=8, label="EM", color="#e67e22")

    # Formatting
    ax.set_xlabel("Maximum Segments per Document")
    ax.set_ylabel("Score (%)")
    ax.set_title("Performance vs Document Length (8 Memory Tokens)")
    ax.set_xticks(segments)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, (seg, f1, em) in enumerate(zip(segments, f1_scores, em_scores)):
        ax.annotate(f"{f1:.1f}", (seg, f1), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
        ax.annotate(f"{em:.1f}", (seg, em), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=10)

    plt.tight_layout()

    # Save
    for ext in ["pdf", "png"]:
        output_path = output_dir / f"segment_analysis.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Saved: {output_path}")

    plt.close()


def plot_ablation_study(results, output_dir):
    """Plot ablation study results."""
    # Get ablation experiments
    ablations = {
        "Main (8 tokens)": next((r for r in results if r["experiment_id"] == "08"), None),
        "No Progressive": next((r for r in results if r["experiment_id"] == "09"), None),
        "No Gating": next((r for r in results if r["experiment_id"] == "10"), None),
    }

    # Filter out None values
    ablations = {k: v for k, v in ablations.items() if v is not None}

    if len(ablations) < 2:
        logger.warning("Not enough ablation experiments found")
        return

    # Extract data
    experiments = list(ablations.keys())
    f1_scores = [ablations[e].get("f1", 0) for e in experiments]
    em_scores = [ablations[e].get("exact_match", 0) for e in experiments]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x - width / 2, f1_scores, width, label="F1", alpha=0.8, color="#9b59b6")
    bars2 = ax.bar(x + width / 2, em_scores, width, label="EM", alpha=0.8, color="#8e44ad")

    # Formatting
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score (%)")
    ax.set_title("Ablation Study (Long SQuAD v2)")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    # Save
    for ext in ["pdf", "png"]:
        output_path = output_dir / f"ablation_study.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Saved: {output_path}")

    plt.close()


def plot_combined_improvement(results, output_dir):
    """Plot combined document length vs improvement analysis."""
    # Get baseline and main experiments for both datasets
    baseline_squad = next((r for r in results if r["experiment_id"] == "01"), None)
    main_squad = next((r for r in results if r["experiment_id"] == "02"), None)
    baseline_long = next((r for r in results if r["experiment_id"] == "07"), None)
    main_long = next((r for r in results if r["experiment_id"] == "08"), None)

    if not all([baseline_squad, main_squad, baseline_long, main_long]):
        logger.warning("Not all baseline/main experiments found for combined plot")
        return

    # Calculate improvements
    squad_improvement = main_squad.get("f1", 0) - baseline_squad.get("f1", 0)
    long_improvement = main_long.get("f1", 0) - baseline_long.get("f1", 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = ["Standard SQuAD v2\n(1-2 segments)", "Long SQuAD v2\n(6-12 segments)"]
    improvements = [squad_improvement, long_improvement]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.bar(datasets, improvements, color=colors, alpha=0.8, width=0.5)

    # Formatting
    ax.set_ylabel("F1 Improvement over Baseline (%)")
    ax.set_title("Memory Benefits Scale with Document Length")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:+.2f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=14,
            fontweight="bold",
        )

    # Add baseline scores as annotations
    ax.text(
        0,
        improvements[0] - 1,
        f"Baseline: {baseline_squad.get('f1', 0):.1f}%\nMemory: {main_squad.get('f1', 0):.1f}%",
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.text(
        1,
        improvements[1] - 1,
        f"Baseline: {baseline_long.get('f1', 0):.1f}%\nMemory: {main_long.get('f1', 0):.1f}%",
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save
    for ext in ["pdf", "png"]:
        output_path = output_dir / f"combined_improvement.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Saved: {output_path}")

    plt.close()


def main():
    print("\n" + "=" * 80)
    print("📊 ANALYZING HYBRID EXPERIMENT RESULTS")
    print("=" * 80 + "\n")

    # Load results
    results_path = Path("./outputs/paper_v2_evaluation_results/all_results.json")
    try:
        results = load_results(results_path)
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.error("Please run evaluate_all.py first to generate results")
        return

    # Create output directory
    output_dir = Path("./outputs/paper_v2_evaluation_results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    print("\n" + "=" * 80)
    print("📈 GENERATING FIGURES")
    print("=" * 80 + "\n")

    logger.info("Generating Standard SQuAD v2 comparison...")
    plot_squad_comparison(results, output_dir)

    logger.info("Generating Long SQuAD v2 comparison...")
    plot_long_squad_comparison(results, output_dir)

    logger.info("Generating memory scaling analysis...")
    plot_memory_scaling(results, output_dir)

    logger.info("Generating segment analysis...")
    plot_segment_analysis(results, output_dir)

    logger.info("Generating ablation study...")
    plot_ablation_study(results, output_dir)

    logger.info("Generating combined improvement plot...")
    plot_combined_improvement(results, output_dir)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80 + "\n")

    print(f"📁 All figures saved to: {output_dir}")
    print("\nGenerated figures:")
    print("  1. squad_v2_comparison.{pdf,png} - Standard SQuAD v2 results")
    print("  2. long_squad_v2_comparison.{pdf,png} - Long SQuAD v2 results")
    print("  3. memory_scaling.{pdf,png} - Memory token scaling")
    print("  4. segment_analysis.{pdf,png} - Document length analysis")
    print("  5. ablation_study.{pdf,png} - Ablation results")
    print("  6. combined_improvement.{pdf,png} - Key findings summary")


if __name__ == "__main__":
    main()
