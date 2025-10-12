#!/usr/bin/env python3
"""
Comprehensive Memory Implementation Comparison
==============================================

Run side-by-side training with token-based vs differentiable memory
and generate detailed comparison report.

This is more thorough than quick_test_token_vs_diff.py:
- Larger training set (1000 samples)
- More epochs (2)
- Detailed metrics tracking
- Performance analysis
- Resource usage comparison

Runtime: ~30-60 minutes

Usage:
    python scripts/compare_memory_implementations.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_base_config():
    """Create base configuration shared by both implementations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    return {
        "model_name": "xlnet-base-cased",
        "max_seq_length": 384,
        "doc_stride": 64,
        "dataset_name": "squad_v2",
        "train_split": "train",
        "eval_split": "validation",
        "cache_dir": "./.cache_comparison",
        # Moderate dataset size for thorough testing
        "max_train_samples": 1000,
        "max_eval_samples": 200,
        "use_lazy_loading": False,
        "progressive_segments": [2],
        "max_n_segs": 2,
        "memory_num_tokens": 8,
        "memory_update": "gated",
        "memory_init": "learned",
        "use_global_softmax": True,
        # Training settings
        "num_epochs": 2,
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "eval_steps": 50,  # Eval every 50 steps
        "save_steps": 500,
        "logging_steps": 25,
        "save_total_limit": 2,
        "no_answer_threshold": 1.5,
        "use_any_positive_logic": True,
        "device": device,
        "fp16": has_cuda,
        "warmup_freeze_base_epochs": 0,
        "warmup_disable_any_positive_epochs": 0,
        "push_to_hub_on_save": False,
        "use_wandb": False,
    }


def train_with_config(config, implementation_name):
    """Train model with given configuration and track metrics."""
    logger.info("=" * 80)
    logger.info(f"🚀 TRAINING: {implementation_name}")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        trainer = XLNetRecurrentTrainer(config)
        trainer.train()

        training_time = time.time() - start_time

        # Get final metrics
        metrics = trainer.best_metrics if hasattr(trainer, "best_metrics") else {}

        result = {
            "status": "success",
            "implementation": implementation_name,
            "training_time_seconds": training_time,
            "training_time_formatted": f"{training_time / 60:.1f} minutes",
            "metrics": metrics,
            "f1": metrics.get("f1", 0.0),
            "exact_match": metrics.get("exact_match", 0.0),
            "has_answer_f1": metrics.get("has_answer_f1", 0.0),
            "no_answer_f1": metrics.get("no_answer_f1", 0.0),
            "has_answer_exact": metrics.get("has_answer_exact", 0.0),
            "no_answer_exact": metrics.get("no_answer_exact", 0.0),
            "configuration": {
                "memory_impl": config.memory_impl,
                "memory_num_tokens": config.memory_num_tokens,
                "num_epochs": config.num_epochs,
                "train_batch_size": config.train_batch_size,
                "learning_rate": config.learning_rate,
            },
        }

        logger.info(f"✅ {implementation_name} completed")
        logger.info(f"   F1: {result['f1']:.2f}%")
        logger.info(f"   EM: {result['exact_match']:.2f}%")
        logger.info(f"   Training time: {result['training_time_formatted']}")

        return result

    except Exception as e:
        training_time = time.time() - start_time
        logger.error(f"❌ {implementation_name} failed: {e}")

        return {
            "status": "error",
            "implementation": implementation_name,
            "error": str(e),
            "training_time_seconds": training_time,
            "f1": 0.0,
            "exact_match": 0.0,
        }


def run_token_based_experiment():
    """Run experiment with token-based memory."""
    base_config = create_base_config()
    config = TrainingConfig(
        **base_config,
        memory_impl="token",
        warmup_disable_global_softmax_epochs=1,
        output_dir="./outputs/comparison_token",
        run_name="comparison-token",
    )

    return train_with_config(config, "Token-Based Memory")


def run_differentiable_experiment():
    """Run experiment with differentiable memory."""
    base_config = create_base_config()
    config = TrainingConfig(
        **base_config,
        memory_impl="differentiable",
        use_differentiable_memory=True,
        num_memory_heads=2,
        memory_slots=16,
        memory_sharpness=1.5,
        enable_usage_tracking=True,
        enable_temporal_links=True,
        warmup_disable_global_softmax_epochs=0,
        output_dir="./outputs/comparison_differentiable",
        run_name="comparison-differentiable",
    )

    return train_with_config(config, "Differentiable Memory")


def print_detailed_comparison(token_result, diff_result):
    """Print comprehensive comparison report."""
    print("\n" + "=" * 80)
    print("📊 DETAILED COMPARISON REPORT")
    print("=" * 80)

    # Overall metrics
    print("\n📈 PERFORMANCE METRICS:")
    print(f"{'Metric':<30} {'Token-Based':<20} {'Differentiable':<20} {'Improvement':<15}")
    print("-" * 85)

    metrics = [
        ("F1 Score", "f1"),
        ("Exact Match", "exact_match"),
        ("Has-Answer F1", "has_answer_f1"),
        ("Has-Answer EM", "has_answer_exact"),
        ("No-Answer F1", "no_answer_f1"),
        ("No-Answer EM", "no_answer_exact"),
    ]

    for label, key in metrics:
        token_val = token_result.get(key, 0.0)
        diff_val = diff_result.get(key, 0.0)
        improvement = diff_val - token_val

        improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        if abs(improvement) < 0.01:
            improvement_str = "~0%"

        print(f"{label:<30} {token_val:>19.2f} {diff_val:>19.2f} {improvement_str:>14}")

    # Training time
    print("\n⏱️  TRAINING TIME:")
    print(f"   Token-Based:     {token_result.get('training_time_formatted', 'N/A')}")
    print(f"   Differentiable:  {diff_result.get('training_time_formatted', 'N/A')}")

    if "training_time_seconds" in token_result and "training_time_seconds" in diff_result:
        time_diff = diff_result["training_time_seconds"] - token_result["training_time_seconds"]
        time_diff_pct = (time_diff / token_result["training_time_seconds"]) * 100
        print(f"   Difference:      {time_diff / 60:+.1f} minutes ({time_diff_pct:+.1f}%)")

    # Analysis
    print("\n🔍 ANALYSIS:")

    token_f1 = token_result.get("f1", 0.0)
    diff_f1 = diff_result.get("f1", 0.0)

    if diff_f1 > 0 and token_f1 == 0:
        print("   ✅ CONFIRMED: Differentiable memory works, token-based fails")
        print("   → Root cause: Memory implementation mismatch")
        print("   → Solution: Use differentiable memory for all training")
    elif diff_f1 > token_f1 + 5:
        print(f"   ✅ Differentiable memory significantly better (+{diff_f1 - token_f1:.2f}% F1)")
        print("   → Recommend using differentiable memory")
    elif diff_f1 > token_f1:
        print(f"   ✅ Differentiable memory slightly better (+{diff_f1 - token_f1:.2f}% F1)")
    elif token_f1 > diff_f1 + 5:
        print(f"   ⚠️  Unexpected: Token-based significantly better (+{token_f1 - diff_f1:.2f}% F1)")
        print("   → This contradicts hypothesis - investigate further")
    else:
        print("   ⚠️  Performance similar - issue may be elsewhere")

    # Has/No answer breakdown
    if diff_result.get("has_answer_f1", 0.0) > 0:
        print("\n📊 BREAKDOWN (Differentiable Memory):")
        print(
            f"   • Has-Answer Performance: F1={diff_result['has_answer_f1']:.2f}%, EM={diff_result.get('has_answer_exact', 0.0):.2f}%"
        )
        print(
            f"   • No-Answer Performance:  F1={diff_result['no_answer_f1']:.2f}%, EM={diff_result.get('no_answer_exact', 0.0):.2f}%"
        )

        # Identify weaknesses
        if diff_result["has_answer_f1"] < 40:
            print("   ⚠️  Weak has-answer performance - may need more training")
        if diff_result["no_answer_f1"] < 40:
            print("   ⚠️  Weak no-answer performance - may need threshold tuning")

    print()


def save_comparison_report(token_result, diff_result, output_file):
    """Save comprehensive comparison to JSON."""
    report = {
        "comparison_type": "token_vs_differentiable",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {"token_based": token_result, "differentiable": diff_result},
        "summary": {
            "winner": "differentiable" if diff_result.get("f1", 0) > token_result.get("f1", 0) else "token",
            "f1_improvement": diff_result.get("f1", 0.0) - token_result.get("f1", 0.0),
            "em_improvement": diff_result.get("exact_match", 0.0) - token_result.get("exact_match", 0.0),
            "hypothesis_confirmed": diff_result.get("f1", 0) > 0 and token_result.get("f1", 0) == 0,
        },
        "recommendations": [],
    }

    # Generate recommendations
    if report["summary"]["hypothesis_confirmed"]:
        report["recommendations"].append("Use differentiable memory for all training")
        report["recommendations"].append("Update 02_main_squad_8tokens.py to use differentiable memory")
    elif diff_result.get("f1", 0) > token_result.get("f1", 0) + 5:
        report["recommendations"].append("Prefer differentiable memory for better performance")
    else:
        report["recommendations"].append("Further investigation needed")

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"📄 Report saved to: {output_file}")


def main():
    """Run comprehensive comparison."""
    print("\n" + "=" * 80)
    print("🔬 COMPREHENSIVE MEMORY IMPLEMENTATION COMPARISON")
    print("=" * 80)
    print()
    print("This will train two models side-by-side:")
    print("1. Token-Based Memory (original config)")
    print("2. Differentiable Memory (working config)")
    print()
    print("Training: 1000 samples, 2 epochs, ~30-60 minutes")
    print()

    results = {}

    # Run experiments
    print("🏃 Starting experiments...\n")

    token_result = run_token_based_experiment()
    results["token"] = token_result

    diff_result = run_differentiable_experiment()
    results["differentiable"] = diff_result

    # Print comparison
    print_detailed_comparison(token_result, diff_result)

    # Save results
    output_dir = Path("./outputs/comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "memory_comparison_report.json"
    save_comparison_report(token_result, diff_result, report_file)

    # Final recommendation
    print("=" * 80)
    print("📋 FINAL RECOMMENDATION")
    print("=" * 80)

    if diff_result.get("f1", 0) > token_result.get("f1", 0):
        print("\n✅ Use differentiable memory for training")
        print()
        print("Next steps:")
        print("1. Run: python scripts/paper_experiments_v2/squad/02b_main_squad_8tokens_differentiable.py")
        print("2. Monitor F1 score during training")
        print("3. Expected result: F1 ~60-75%")
    else:
        print("\n⚠️  Results inconclusive - further investigation needed")
        print()
        print("Debugging steps:")
        print("1. Check training logs for errors")
        print("2. Verify data preprocessing")
        print("3. Test with smaller learning rate")

    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
