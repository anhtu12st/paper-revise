#!/usr/bin/env python3
"""
Evaluate All Paper Experiments
===============================

This script evaluates all trained models from paper experiments
and saves results for comparison.

Evaluates:
- All 10 experiments (01-10)
- Finds best model for each
- Runs full SQuAD v2 evaluation
- Saves results to results/paper_experiments/

Usage:
    python scripts/paper_experiments/evaluate_all.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from memxlnet.evaluation.evaluator import evaluate_model_from_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Experiment metadata
EXPERIMENTS = [
    ("01_baseline_no_memory", "Baseline (No Memory)"),
    ("02_main_memory_8tokens", "Main (8 Tokens)"),
    ("03_main_memory_16tokens", "Main (16 Tokens)"),
    ("04_ablation_no_progressive", "Ablation: No Progressive"),
    ("05_ablation_no_gating", "Ablation: No Gating"),
    ("06_ablation_4tokens", "Ablation: 4 Tokens"),
    ("07_ablation_32tokens", "Ablation: 32 Tokens"),
    ("08_segments_2seg", "Segments: 2 Max"),
    ("09_segments_4seg", "Segments: 4 Max"),
    ("10_segments_6seg", "Segments: 6 Max"),
]


def find_best_model(exp_dir: Path) -> Path:
    """Find best model checkpoint in experiment directory."""
    # Try stage directories (progressive training)
    stage_dirs = sorted(exp_dir.glob("stage_*"))

    if stage_dirs:
        # Look in last stage
        last_stage = stage_dirs[-1]
        best_model = last_stage / "best_model"
        if best_model.exists():
            return best_model

        # Fallback to final model
        final_model = last_stage / "final_model"
        if final_model.exists():
            return final_model

    # Try direct best_model
    best_model = exp_dir / "best_model"
    if best_model.exists():
        return best_model

    # Try final_model
    final_model = exp_dir / "final_model"
    if final_model.exists():
        return final_model

    raise FileNotFoundError(f"No model found in {exp_dir}")


def find_training_config(exp_dir: Path) -> Path:
    """Find training config in experiment directory."""
    # Try stage directories first
    stage_dirs = sorted(exp_dir.glob("stage_*"))

    if stage_dirs:
        # Look in last stage
        last_stage = stage_dirs[-1]
        config = last_stage / "best_model" / "training_config.json"
        if config.exists():
            return config

        config = last_stage / "final_model" / "training_config.json"
        if config.exists():
            return config

    # Try direct paths
    for config_path in [
        exp_dir / "training_config.json",
        exp_dir / "best_model" / "training_config.json",
        exp_dir / "final_model" / "training_config.json",
    ]:
        if config_path.exists():
            return config_path

    raise FileNotFoundError(f"No config found in {exp_dir}")


def evaluate_experiment(exp_id: str, exp_name: str, output_dir: Path) -> dict[str, Any]:
    """Evaluate a single experiment."""
    print(f"\n{'=' * 80}")
    print(f"📊 EVALUATING: {exp_name} ({exp_id})")
    print(f"{'=' * 80}\n")

    exp_dir = Path(f"./outputs/paper_exp_{exp_id}")

    if not exp_dir.exists():
        logger.warning(f"⚠️  Experiment directory not found: {exp_dir}")
        return {
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "status": "not_found",
            "error": f"Directory not found: {exp_dir}",
        }

    try:
        # Find model and config
        model_path = find_best_model(exp_dir)
        config_path = find_training_config(exp_dir)

        logger.info(f"📁 Model: {model_path}")
        logger.info(f"📁 Config: {config_path}")

        # Run evaluation
        results = evaluate_model_from_config(
            str(config_path),
            str(model_path),
            max_eval_samples=None,  # Full eval
            save_results=True,
        )

        # Add metadata
        results["experiment_id"] = exp_id
        results["experiment_name"] = exp_name
        results["model_path"] = str(model_path)
        results["status"] = "success"

        # Save individual result
        result_file = output_dir / f"{exp_id}_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"✅ Evaluation complete: F1={results['metrics']['f1']:.2f}%")

        return results

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return {"experiment_id": exp_id, "experiment_name": exp_name, "status": "error", "error": str(e)}


def create_comparison_table(all_results: list, output_dir: Path):
    """Create comparison table of all experiments."""
    print(f"\n{'=' * 80}")
    print("📊 CREATING COMPARISON TABLE")
    print(f"{'=' * 80}\n")

    # Extract key metrics
    comparison = []
    for result in all_results:
        if result["status"] != "success":
            continue

        metrics = result["metrics"]
        comparison.append(
            {
                "Experiment": result["experiment_name"],
                "F1": f"{metrics['f1']:.2f}",
                "EM": f"{metrics['exact_match']:.2f}",
                "HasAns F1": f"{metrics['has_answer_f1']:.2f}",
                "NoAns F1": f"{metrics['no_answer_f1']:.2f}",
            }
        )

    # Save as JSON
    json_file = output_dir / "comparison_table.json"
    with open(json_file, "w") as f:
        json.dump(comparison, f, indent=2)

    # Save as CSV
    csv_file = output_dir / "comparison_table.csv"
    with open(csv_file, "w") as f:
        if comparison:
            headers = list(comparison[0].keys())
            f.write(",".join(headers) + "\n")
            for row in comparison:
                f.write(",".join(str(row[h]) for h in headers) + "\n")

    # Save as LaTeX
    latex_file = output_dir / "comparison_table.tex"
    with open(latex_file, "w") as f:
        if comparison:
            headers = list(comparison[0].keys())
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}\n")
            f.write("\\hline\n")
            f.write(" & ".join(headers) + " \\\\\n")
            f.write("\\hline\n")
            for row in comparison:
                f.write(" & ".join(str(row[h]) for h in headers) + " \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Comparison of all experiments}\n")
            f.write("\\label{tab:experiments}\n")
            f.write("\\end{table}\n")

    # Print table
    print("\nComparison Table:")
    print("-" * 80)
    if comparison:
        headers = list(comparison[0].keys())
        col_widths = [max(len(h), max(len(str(row[h])) for row in comparison)) for h in headers]

        # Header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))

        # Rows
        for row in comparison:
            print(" | ".join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths)))

    print("-" * 80)
    print("\n✅ Comparison saved to:")
    print(f"   • JSON: {json_file}")
    print(f"   • CSV: {csv_file}")
    print(f"   • LaTeX: {latex_file}")


def main():
    """Evaluate all experiments."""
    print("\n" + "=" * 80)
    print("📊 PAPER EXPERIMENTS: EVALUATE ALL")
    print("=" * 80)
    print(f"\nTotal experiments: {len(EXPERIMENTS)}\n")

    # Create output directory
    output_dir = Path("./results/paper_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate all experiments
    all_results = []
    for exp_id, exp_name in EXPERIMENTS:
        result = evaluate_experiment(exp_id, exp_name, output_dir)
        all_results.append(result)

    # Save all results
    summary_file = output_dir / "all_results.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Create comparison table
    create_comparison_table(all_results, output_dir)

    # Summary
    successful = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "error")
    not_found = sum(1 for r in all_results if r["status"] == "not_found")

    print(f"\n{'=' * 80}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n✅ Successful: {successful}/{len(EXPERIMENTS)}")
    print(f"❌ Failed: {failed}/{len(EXPERIMENTS)}")
    print(f"⚠️  Not Found: {not_found}/{len(EXPERIMENTS)}")
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
