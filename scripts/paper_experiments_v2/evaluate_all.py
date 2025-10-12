#!/usr/bin/env python3
"""
Evaluate All Hybrid Experiments
================================

Evaluates all 12 experiments (6 Standard SQuAD v2 + 6 Long SQuAD v2)
and generates comprehensive comparison tables and analysis.

Usage:
    python scripts/paper_experiments_v2/evaluate_all.py

Output:
    - Individual JSON results for each experiment
    - CSV comparison tables (Standard SQuAD v2, Long SQuAD v2, Combined)
    - LaTeX tables for paper
    - Summary JSON with all results
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from memxlnet.evaluation import XLNetRecurrentEvaluator
from memxlnet.training import TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Experiment definitions
SQUAD_EXPERIMENTS = [
    {
        "id": "01",
        "name": "Baseline (No Memory)",
        "output_dir": "./outputs/paper_v2_squad_01_baseline_no_memory",
        "dataset": "Standard SQuAD v2",
        "memory_tokens": 0,
        "max_segs": 2,
    },
    {
        "id": "02",
        "name": "Main (8 tokens)",
        "output_dir": "./outputs/paper_v2_squad_02_main_8tokens",
        "dataset": "Standard SQuAD v2",
        "memory_tokens": 8,
        "max_segs": 2,
    },
    {
        "id": "03",
        "name": "Ablation (No Gating)",
        "output_dir": "./outputs/paper_v2_squad_03_ablation_no_gating",
        "dataset": "Standard SQuAD v2",
        "memory_tokens": 8,
        "max_segs": 2,
    },
    {
        "id": "04",
        "name": "Ablation (4 tokens)",
        "output_dir": "./outputs/paper_v2_squad_04_ablation_4tokens",
        "dataset": "Standard SQuAD v2",
        "memory_tokens": 4,
        "max_segs": 2,
    },
    {
        "id": "05",
        "name": "Ablation (16 tokens)",
        "output_dir": "./outputs/paper_v2_squad_05_ablation_16tokens",
        "dataset": "Standard SQuAD v2",
        "memory_tokens": 16,
        "max_segs": 2,
    },
    {
        "id": "06",
        "name": "Ablation (32 tokens)",
        "output_dir": "./outputs/paper_v2_squad_06_ablation_32tokens",
        "dataset": "Standard SQuAD v2",
        "memory_tokens": 32,
        "max_segs": 2,
    },
]

LONG_SQUAD_EXPERIMENTS = [
    {
        "id": "07",
        "name": "Baseline (No Memory)",
        "output_dir": "./outputs/paper_v2_long_07_baseline_no_memory",
        "dataset": "Long SQuAD v2",
        "memory_tokens": 0,
        "max_segs": 6,
    },
    {
        "id": "08",
        "name": "Main (8 tokens)",
        "output_dir": "./outputs/paper_v2_long_08_main_8tokens",
        "dataset": "Long SQuAD v2",
        "memory_tokens": 8,
        "max_segs": 6,
    },
    {
        "id": "09",
        "name": "Ablation (No Progressive)",
        "output_dir": "./outputs/paper_v2_long_09_ablation_no_progressive",
        "dataset": "Long SQuAD v2",
        "memory_tokens": 8,
        "max_segs": 6,
    },
    {
        "id": "10",
        "name": "Ablation (No Gating)",
        "output_dir": "./outputs/paper_v2_long_10_ablation_no_gating",
        "dataset": "Long SQuAD v2",
        "memory_tokens": 8,
        "max_segs": 6,
    },
    {
        "id": "11",
        "name": "4 Segments",
        "output_dir": "./outputs/paper_v2_long_11_segments_4seg",
        "dataset": "Long SQuAD v2",
        "memory_tokens": 8,
        "max_segs": 4,
    },
    {
        "id": "12",
        "name": "6 Segments",
        "output_dir": "./outputs/paper_v2_long_12_segments_6seg",
        "dataset": "Long SQuAD v2",
        "memory_tokens": 8,
        "max_segs": 6,
    },
]

ALL_EXPERIMENTS = SQUAD_EXPERIMENTS + LONG_SQUAD_EXPERIMENTS


def evaluate_experiment(exp: dict) -> dict:
    """Evaluate a single experiment."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Evaluating Experiment {exp['id']}: {exp['name']}")
    logger.info(f"Dataset: {exp['dataset']}")
    logger.info(f"Output: {exp['output_dir']}")
    logger.info(f"{'=' * 80}")

    output_dir = Path(exp["output_dir"])

    # Check if training completed
    if not (output_dir / "training_config.json").exists():
        logger.warning(f"⚠️  Training config not found: {output_dir}")
        return {
            "experiment_id": exp["id"],
            "experiment_name": exp["name"],
            "dataset": exp["dataset"],
            "status": "not_trained",
            "error": "Training config not found",
        }

    # Check if already evaluated
    eval_results_path = output_dir / "evaluation_results.json"
    if eval_results_path.exists():
        logger.info(f"✅ Evaluation results already exist: {eval_results_path}")
        with open(eval_results_path) as f:
            results = json.load(f)
            results["experiment_id"] = exp["id"]
            results["experiment_name"] = exp["name"]
            results["dataset"] = exp["dataset"]
            results["memory_tokens"] = exp["memory_tokens"]
            results["max_segs"] = exp["max_segs"]
            return results

    # Run evaluation
    try:
        config = TrainingConfig.from_json(output_dir / "training_config.json")
        evaluator = XLNetRecurrentEvaluator(config)

        logger.info("🔄 Running evaluation...")
        results = evaluator.evaluate()

        # Save results
        with open(eval_results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"✅ Results saved: {eval_results_path}")
        logger.info(f"   F1: {results.get('f1', 0):.2f}%")
        logger.info(f"   EM: {results.get('exact_match', 0):.2f}%")

        # Add metadata
        results["experiment_id"] = exp["id"]
        results["experiment_name"] = exp["name"]
        results["dataset"] = exp["dataset"]
        results["memory_tokens"] = exp["memory_tokens"]
        results["max_segs"] = exp["max_segs"]

        return results

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return {
            "experiment_id": exp["id"],
            "experiment_name": exp["name"],
            "dataset": exp["dataset"],
            "status": "failed",
            "error": str(e),
        }


def create_comparison_table(results: list[dict], dataset_name: str) -> pd.DataFrame:
    """Create comparison table for a specific dataset."""
    data = []
    for r in results:
        if r.get("status") in ["not_trained", "failed"]:
            data.append(
                {
                    "Experiment": r["experiment_name"],
                    "F1": "-",
                    "EM": "-",
                    "HasAns F1": "-",
                    "HasAns EM": "-",
                    "NoAns F1": "-",
                    "NoAns EM": "-",
                    "Total": "-",
                    "Status": r.get("status", "unknown"),
                }
            )
        else:
            data.append(
                {
                    "Experiment": r["experiment_name"],
                    "F1": f"{r.get('f1', 0):.2f}",
                    "EM": f"{r.get('exact_match', 0):.2f}",
                    "HasAns F1": f"{r.get('HasAns_f1', 0):.2f}",
                    "HasAns EM": f"{r.get('HasAns_exact', 0):.2f}",
                    "NoAns F1": f"{r.get('NoAns_f1', 0):.2f}",
                    "NoAns EM": f"{r.get('NoAns_exact', 0):.2f}",
                    "Total": r.get("total", 0),
                    "Status": "completed",
                }
            )

    df = pd.DataFrame(data)
    return df


def create_latex_table(df: pd.DataFrame, dataset_name: str) -> str:
    """Create LaTeX table for paper."""
    latex = f"% {dataset_name} Results\n"
    latex += "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{dataset_name} Results}}\n"
    latex += "\\begin{tabular}{l|cc|cc|cc|c}\n"
    latex += "\\hline\n"
    latex += "Experiment & F1 & EM & HasAns F1 & HasAns EM & NoAns F1 & NoAns EM & Total \\\\\n"
    latex += "\\hline\n"

    for _, row in df.iterrows():
        if row["Status"] == "completed":
            latex += f"{row['Experiment']} & {row['F1']} & {row['EM']} & "
            latex += f"{row['HasAns F1']} & {row['HasAns EM']} & "
            latex += f"{row['NoAns F1']} & {row['NoAns EM']} & {row['Total']} \\\\\n"
        else:
            latex += f"{row['Experiment']} & \\multicolumn{{7}}{{c}}{{Not Available}} \\\\\n"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def main():
    print("\n" + "=" * 80)
    print("📊 EVALUATING ALL HYBRID EXPERIMENTS")
    print("=" * 80 + "\n")

    print(f"Total experiments: {len(ALL_EXPERIMENTS)}")
    print(f"  - Standard SQuAD v2: {len(SQUAD_EXPERIMENTS)}")
    print(f"  - Long SQuAD v2: {len(LONG_SQUAD_EXPERIMENTS)}")
    print()

    # Evaluate all experiments
    all_results = []
    for exp in ALL_EXPERIMENTS:
        results = evaluate_experiment(exp)
        all_results.append(results)

    # Split results by dataset
    squad_results = [r for r in all_results if r["dataset"] == "Standard SQuAD v2"]
    long_squad_results = [r for r in all_results if r["dataset"] == "Long SQuAD v2"]

    # Create output directory
    output_dir = Path("./outputs/paper_v2_evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual results
    print("\n" + "=" * 80)
    print("💾 SAVING RESULTS")
    print("=" * 80 + "\n")

    # Save all results JSON
    all_results_path = output_dir / "all_results.json"
    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Saved all results: {all_results_path}")

    # Create comparison tables
    print("\n" + "=" * 80)
    print("📊 CREATING COMPARISON TABLES")
    print("=" * 80 + "\n")

    # Standard SQuAD v2 table
    squad_df = create_comparison_table(squad_results, "Standard SQuAD v2")
    squad_csv_path = output_dir / "squad_v2_comparison.csv"
    squad_df.to_csv(squad_csv_path, index=False)
    print(f"✅ Saved Standard SQuAD v2 table: {squad_csv_path}")
    print("\nStandard SQuAD v2 Results:")
    print(squad_df.to_string(index=False))

    # Long SQuAD v2 table
    long_squad_df = create_comparison_table(long_squad_results, "Long SQuAD v2")
    long_squad_csv_path = output_dir / "long_squad_v2_comparison.csv"
    long_squad_df.to_csv(long_squad_csv_path, index=False)
    print(f"\n✅ Saved Long SQuAD v2 table: {long_squad_csv_path}")
    print("\nLong SQuAD v2 Results:")
    print(long_squad_df.to_string(index=False))

    # Combined table
    combined_df = create_comparison_table(all_results, "All Experiments")
    combined_csv_path = output_dir / "combined_comparison.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"\n✅ Saved combined table: {combined_csv_path}")

    # Create LaTeX tables
    print("\n" + "=" * 80)
    print("📝 CREATING LATEX TABLES")
    print("=" * 80 + "\n")

    squad_latex = create_latex_table(squad_df, "Standard SQuAD v2")
    squad_latex_path = output_dir / "squad_v2_table.tex"
    with open(squad_latex_path, "w") as f:
        f.write(squad_latex)
    print(f"✅ Saved Standard SQuAD v2 LaTeX: {squad_latex_path}")

    long_squad_latex = create_latex_table(long_squad_df, "Long SQuAD v2")
    long_squad_latex_path = output_dir / "long_squad_v2_table.tex"
    with open(long_squad_latex_path, "w") as f:
        f.write(long_squad_latex)
    print(f"✅ Saved Long SQuAD v2 LaTeX: {long_squad_latex_path}")

    # Summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80 + "\n")

    completed = sum(1 for r in all_results if r.get("status") != "failed" and r.get("status") != "not_trained")
    failed = sum(1 for r in all_results if r.get("status") == "failed")
    not_trained = sum(1 for r in all_results if r.get("status") == "not_trained")

    print(f"Total experiments: {len(all_results)}")
    print(f"  ✅ Completed: {completed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏳ Not trained: {not_trained}")

    if completed > 0:
        print("\n🎉 Key Findings:")

        # Standard SQuAD v2 insights
        squad_completed = [r for r in squad_results if r.get("status") not in ["failed", "not_trained"]]
        if squad_completed:
            baseline_squad = next((r for r in squad_completed if r["experiment_id"] == "01"), None)
            main_squad = next((r for r in squad_completed if r["experiment_id"] == "02"), None)

            if baseline_squad and main_squad:
                improvement = main_squad.get("f1", 0) - baseline_squad.get("f1", 0)
                print("\n📊 Standard SQuAD v2:")
                print(f"   Baseline F1: {baseline_squad.get('f1', 0):.2f}%")
                print(f"   Main (8 tokens) F1: {main_squad.get('f1', 0):.2f}%")
                print(f"   Improvement: {improvement:+.2f}%")

        # Long SQuAD v2 insights
        long_completed = [r for r in long_squad_results if r.get("status") not in ["failed", "not_trained"]]
        if long_completed:
            baseline_long = next((r for r in long_completed if r["experiment_id"] == "07"), None)
            main_long = next((r for r in long_completed if r["experiment_id"] == "08"), None)

            if baseline_long and main_long:
                improvement = main_long.get("f1", 0) - baseline_long.get("f1", 0)
                print("\n📊 Long SQuAD v2:")
                print(f"   Baseline F1: {baseline_long.get('f1', 0):.2f}%")
                print(f"   Main (8 tokens) F1: {main_long.get('f1', 0):.2f}%")
                print(f"   Improvement: {improvement:+.2f}%")

    print(f"\n📁 All results saved to: {output_dir}")
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
