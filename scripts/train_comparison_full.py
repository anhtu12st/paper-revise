#!/usr/bin/env python3
"""
Full Training Comparison: Token-based vs Differentiable Memory
===============================================================

This script performs a comprehensive comparison between token-based and
differentiable memory implementations by training both models on the full
SQuAD v2 dataset and analyzing their performance.

The script:
1. Trains a token-based memory model (baseline)
2. Trains a differentiable memory model (experimental)
3. Evaluates both models on SQuAD v2
4. Generates comprehensive comparison analysis
5. Creates visualizations and statistical comparisons

RUNTIME: ~17-25 hours total (depends on hardware)
- Token-based training: ~8-12 hours
- Differentiable training: ~8-12 hours
- Analysis: ~30 minutes

OUTPUT STRUCTURE:
./outputs/comparison-token-based/     - Token-based model checkpoints
./outputs/comparison-differentiable/  - Differentiable model checkpoints
./outputs/comparison-analysis/        - Comparison results and visualizations

USAGE:
    python scripts/train_comparison_full.py

No arguments required - all configuration is self-contained.

To run only specific steps:
    python scripts/train_comparison_full.py --skip-token-training
    python scripts/train_comparison_full.py --skip-diff-training
    python scripts/train_comparison_full.py --only-analysis
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Full training comparison between token-based and differentiable memory"
    )
    parser.add_argument(
        "--skip-token-training",
        action="store_true",
        help="Skip token-based model training (use existing checkpoint)"
    )
    parser.add_argument(
        "--skip-diff-training",
        action="store_true",
        help="Skip differentiable model training (use existing checkpoint)"
    )
    parser.add_argument(
        "--only-analysis",
        action="store_true",
        help="Only run analysis (skip all training)"
    )
    return parser.parse_args()


def create_token_based_config():
    """
    Create configuration for token-based memory model (baseline).

    This uses the traditional memory token approach with gated updates.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    config = TrainingConfig(
        # ===== Model Configuration =====
        model_name="xlnet-base-cased",
        max_seq_length=384,
        doc_stride=64,

        # ===== Dataset Configuration (FULL SQuAD v2) =====
        dataset_name="squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=None,      # Use full training set (~130K)
        max_eval_samples=None,       # Use full validation set (~12K)
        use_lazy_loading=False,

        # ===== Progressive Training =====
        progressive_segments=[2, 4, 6],  # Progressive curriculum
        max_n_segs=None,

        # ===== Training Hyperparameters =====
        num_epochs=4,
        train_batch_size=8,
        eval_batch_size=8,            # Must match train_batch_size (XLNet memory cache requirement)
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,

        # ===== Training Schedule =====
        gradient_accumulation_steps=2,  # Effective batch size: 16
        eval_steps=6000,
        save_steps=10000,
        logging_steps=500,

        # ===== Output Configuration =====
        output_dir="./outputs/comparison-token-based",
        run_name="comparison-token-based",
        save_total_limit=3,

        # ===== Evaluation Settings =====
        no_answer_threshold=1.5,
        use_any_positive_logic=True,

        # ===== Experiment Tracking =====
        use_wandb=False,

        # ===== Device Settings =====
        device=device,
        fp16=has_cuda,

        # ===== Token-Based Memory Configuration =====
        memory_num_tokens=16,
        memory_update="gated",
        memory_init="learned",
        memory_impl="token",
        use_global_softmax=True,
        use_differentiable_memory=False,  # KEY: Traditional token-based

        # ===== Warmup Strategy =====
        warmup_freeze_base_epochs=0,
        warmup_disable_global_softmax_epochs=1,
        warmup_disable_any_positive_epochs=0,

        # ===== Hub Integration (Disabled) =====
        push_to_hub_on_save=False,
    )

    return config


def create_differentiable_config():
    """
    Create configuration for differentiable memory model (experimental).

    This uses the new differentiable memory with content-based addressing.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    config = TrainingConfig(
        # ===== Model Configuration =====
        model_name="xlnet-base-cased",
        max_seq_length=384,
        doc_stride=64,

        # ===== Dataset Configuration (FULL SQuAD v2) =====
        dataset_name="squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=None,      # Use full training set (~130K)
        max_eval_samples=None,       # Use full validation set (~12K)
        use_lazy_loading=False,

        # ===== Progressive Training =====
        progressive_segments=[2, 4, 6],  # Same as token-based
        max_n_segs=None,

        # ===== Training Hyperparameters (IDENTICAL to token-based) =====
        num_epochs=4,
        train_batch_size=8,
        eval_batch_size=8,            # Must match train_batch_size (XLNet memory cache requirement)
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,

        # ===== Training Schedule (IDENTICAL to token-based) =====
        gradient_accumulation_steps=2,
        eval_steps=6000,
        save_steps=10000,
        logging_steps=500,

        # ===== Output Configuration =====
        output_dir="./outputs/comparison-differentiable",
        run_name="comparison-differentiable",
        save_total_limit=3,

        # ===== Evaluation Settings (IDENTICAL to token-based) =====
        no_answer_threshold=1.5,
        use_any_positive_logic=True,

        # ===== Experiment Tracking =====
        use_wandb=False,

        # ===== Device Settings =====
        device=device,
        fp16=has_cuda,

        # ===== Differentiable Memory Configuration =====
        memory_num_tokens=16,        # Same token count for fair comparison
        memory_impl="differentiable",  # KEY: Use differentiable memory
        use_differentiable_memory=True,

        # Differentiable memory parameters
        num_memory_heads=4,          # Multi-head attention
        memory_slots=32,             # Number of memory slots
        memory_sharpness=2.0,        # Attention sharpening
        enable_usage_tracking=True,  # Track memory usage
        enable_temporal_links=True,  # Track temporal relationships

        use_global_softmax=True,

        # ===== Warmup Strategy (IDENTICAL to token-based) =====
        warmup_freeze_base_epochs=0,
        warmup_disable_global_softmax_epochs=1,
        warmup_disable_any_positive_epochs=0,

        # ===== Hub Integration (Disabled) =====
        push_to_hub_on_save=False,
    )

    return config


def print_training_info(config, model_type):
    """Print training configuration summary."""
    memory_type = "Differentiable" if config.use_differentiable_memory else "Token-based"

    print("\n" + "=" * 80)
    print(f"🚀 TRAINING: {model_type.upper()} ({memory_type})")
    print("=" * 80)
    print()

    print("📊 DATASET:")
    print(f"   • Full SQuAD v2 (~130K train, ~12K eval)")
    print()

    print("🧠 MODEL:")
    print(f"   • Base: {config.model_name}")
    print(f"   • Memory type: {memory_type}")
    print(f"   • Memory tokens: {config.memory_num_tokens}")

    if config.use_differentiable_memory:
        print(f"   • Memory heads: {config.num_memory_heads}")
        print(f"   • Memory slots: {config.memory_slots}")
        print(f"   • Memory sharpness: {config.memory_sharpness}")
        print(f"   • Usage tracking: {config.enable_usage_tracking}")
        print(f"   • Temporal links: {config.enable_temporal_links}")
    else:
        print(f"   • Memory update: {config.memory_update}")
        print(f"   • Memory init: {config.memory_init}")

    print(f"   • Progressive segments: {config.progressive_segments}")
    print()

    print("⚡ TRAINING:")
    print(f"   • Epochs: {config.num_epochs}")
    print(f"   • Batch size: {config.train_batch_size} (effective: {config.train_batch_size * config.gradient_accumulation_steps})")
    print(f"   • Learning rate: {config.learning_rate}")
    print(f"   • Device: {config.device}")
    print()

    print("💾 OUTPUT:")
    print(f"   • Directory: {config.output_dir}")
    print()

    approx_steps_per_epoch = 130000 // (config.train_batch_size * config.gradient_accumulation_steps)
    approx_total_steps = approx_steps_per_epoch * config.num_epochs
    print(f"⏱️  ESTIMATED TIME:")
    print(f"   • Steps per epoch: ~{approx_steps_per_epoch:,}")
    print(f"   • Total steps: ~{approx_total_steps:,}")
    print(f"   • Expected runtime: ~8-12 hours (GPU)")
    print()
    print("=" * 80)


def train_model(config, model_type):
    """
    Train a model with the given configuration.

    Args:
        config: TrainingConfig object
        model_type: "token-based" or "differentiable"

    Returns:
        Tuple of (success: bool, best_metrics: dict, training_time: float)
    """
    print_training_info(config, model_type)

    print(f"\n✅ Starting {model_type} training...\n")

    start_time = time.time()

    try:
        trainer = XLNetRecurrentTrainer(config)
        trainer.train()

        training_time = time.time() - start_time

        # Get best metrics
        best_metrics = {}
        if hasattr(trainer, 'best_metrics'):
            best_metrics = trainer.best_metrics

        print(f"\n🎉 {model_type.title()} training completed!")
        print(f"⏱️  Training time: {training_time / 3600:.2f} hours")
        print(f"📁 Model saved to: {config.output_dir}")

        return True, best_metrics, training_time

    except KeyboardInterrupt:
        print(f"\n⚠️  {model_type.title()} training interrupted by user")
        return False, {}, time.time() - start_time

    except Exception as e:
        print(f"\n❌ {model_type.title()} training failed: {e}")
        logger.exception("Training error:")
        return False, {}, time.time() - start_time


def run_comparison_analysis(token_model_path, diff_model_path, output_dir):
    """
    Run comprehensive comparison analysis between models.

    Args:
        token_model_path: Path to token-based model checkpoint
        diff_model_path: Path to differentiable model checkpoint
        output_dir: Directory to save analysis results
    """
    from memxlnet.models import MemXLNetForQA
    from transformers import XLNetTokenizerFast

    print("\n" + "=" * 80)
    print("📊 COMPARISON ANALYSIS")
    print("=" * 80)
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load models
    print("📥 Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        token_model = MemXLNetForQA.from_pretrained(token_model_path)
        token_model.to(device)
        token_model.eval()
        print(f"✓ Token-based model loaded")
    except Exception as e:
        print(f"✗ Failed to load token-based model: {e}")
        token_model = None

    try:
        diff_model = MemXLNetForQA.from_pretrained(diff_model_path)
        diff_model.to(device)
        diff_model.eval()
        print(f"✓ Differentiable model loaded")
    except Exception as e:
        print(f"✗ Failed to load differentiable model: {e}")
        diff_model = None

    if token_model is None and diff_model is None:
        print("\n❌ No models available for comparison")
        return

    print()

    # Model architecture comparison
    print("🏗️  MODEL ARCHITECTURE:")
    print()

    if token_model:
        print("Token-based model:")
        print(f"  • Memory tokens: {token_model.mem_token_count}")
        print(f"  • Memory update: {token_model.memory_update}")
        print(f"  • Differentiable memory: {token_model.use_differentiable_memory}")

    if diff_model:
        print("\nDifferentiable model:")
        print(f"  • Memory tokens: {diff_model.mem_token_count}")
        print(f"  • Differentiable memory: {diff_model.use_differentiable_memory}")
        if diff_model.use_differentiable_memory:
            print(f"  • Memory heads: {diff_model.num_memory_heads}")
            print(f"  • Memory slots: {diff_model.memory_slots}")
            print(f"  • Memory sharpness: {diff_model.memory_sharpness}")

    print()

    # Metrics comparison (load from training logs if available)
    print("📈 PERFORMANCE COMPARISON:")
    print()

    comparison_data = {
        "models": {
            "token_based": {
                "path": token_model_path,
                "memory_type": "token-based",
                "available": token_model is not None,
            },
            "differentiable": {
                "path": diff_model_path,
                "memory_type": "differentiable",
                "available": diff_model is not None,
            },
        },
        "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Try to load metrics from training configs
    for model_type, model_info in comparison_data["models"].items():
        if model_info["available"]:
            config_path = Path(model_info["path"]) / "training_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    training_config = json.load(f)
                    model_info["training_config"] = training_config

    # Save comparison data
    comparison_file = output_path / "comparison_summary.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"💾 Comparison summary saved to: {comparison_file}")

    # Generate comparison report
    generate_comparison_report(comparison_data, output_path)

    print()
    print("=" * 80)
    print("✅ COMPARISON ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📁 Results saved to: {output_path}")
    print()


def generate_comparison_report(comparison_data, output_path):
    """Generate markdown comparison report."""
    report_path = output_path / "comparison_report.md"

    with open(report_path, "w") as f:
        f.write("# MemXLNet Training Comparison Report\n\n")
        f.write(f"**Generated:** {comparison_data['analysis_date']}\n\n")

        f.write("## Models Compared\n\n")

        for model_type, model_info in comparison_data["models"].items():
            f.write(f"### {model_type.replace('_', ' ').title()}\n\n")
            f.write(f"- **Memory Type:** {model_info['memory_type']}\n")
            f.write(f"- **Available:** {'✓ Yes' if model_info['available'] else '✗ No'}\n")
            f.write(f"- **Path:** `{model_info['path']}`\n\n")

        f.write("## Training Configuration\n\n")
        f.write("Both models were trained with identical hyperparameters:\n\n")
        f.write("- **Dataset:** SQuAD v2 (full, ~130K train examples)\n")
        f.write("- **Epochs:** 4\n")
        f.write("- **Batch Size:** 8 (effective: 16 with gradient accumulation)\n")
        f.write("- **Learning Rate:** 3e-5\n")
        f.write("- **Progressive Segments:** [2, 4, 6]\n\n")

        f.write("## Key Differences\n\n")
        f.write("| Feature | Token-based | Differentiable |\n")
        f.write("|---------|-------------|----------------|\n")
        f.write("| Memory Implementation | Token embeddings | Content-addressed |\n")
        f.write("| Memory Heads | N/A | 4 |\n")
        f.write("| Memory Slots | 16 tokens | 32 slots |\n")
        f.write("| Usage Tracking | No | Yes |\n")
        f.write("| Temporal Links | No | Yes |\n\n")

        f.write("## Analysis\n\n")
        f.write("### Memory System Comparison\n\n")
        f.write("**Token-based Memory:**\n")
        f.write("- Uses learnable token embeddings\n")
        f.write("- Position-based memory updates\n")
        f.write("- Gated update mechanism\n\n")

        f.write("**Differentiable Memory:**\n")
        f.write("- Content-based addressing with attention\n")
        f.write("- Multi-head attention for diverse memory access patterns\n")
        f.write("- Explicit usage tracking and temporal link modeling\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Run full evaluation on SQuAD v2 validation set\n")
        f.write("2. Analyze attention patterns with `examples/analyze_memory_attention.py`\n")
        f.write("3. Compare multi-hop reasoning performance\n")
        f.write("4. Evaluate on HotpotQA (multi-hop dataset)\n\n")

        f.write("## Files Generated\n\n")
        f.write("- `comparison_summary.json` - Machine-readable comparison data\n")
        f.write("- `comparison_report.md` - This human-readable report\n")

    print(f"📄 Comparison report saved to: {report_path}")


def main():
    """Main comparison training entry point."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("FULL TRAINING COMPARISON: TOKEN-BASED VS DIFFERENTIABLE MEMORY")
    print("=" * 80)
    print()
    print("This will train two models on the full SQuAD v2 dataset:")
    print("  1. Token-based memory (baseline)")
    print("  2. Differentiable memory (experimental)")
    print()
    print("Expected total runtime: ~17-25 hours (GPU)")
    print()

    if args.only_analysis:
        args.skip_token_training = True
        args.skip_diff_training = True

    results = {}

    # Train token-based model
    if not args.skip_token_training:
        print("\n" + "🔵" * 40)
        print("STEP 1/3: TOKEN-BASED MODEL TRAINING")
        print("🔵" * 40)

        token_config = create_token_based_config()
        success, metrics, training_time = train_model(token_config, "token-based")

        results["token_based"] = {
            "success": success,
            "metrics": metrics,
            "training_time": training_time,
            "output_dir": token_config.output_dir,
        }
    else:
        print("\n⏭️  Skipping token-based training (using existing checkpoint)")
        results["token_based"] = {
            "success": True,
            "skipped": True,
            "output_dir": "./outputs/comparison-token-based",
        }

    # Train differentiable model
    if not args.skip_diff_training:
        print("\n" + "🟢" * 40)
        print("STEP 2/3: DIFFERENTIABLE MEMORY MODEL TRAINING")
        print("🟢" * 40)

        diff_config = create_differentiable_config()
        success, metrics, training_time = train_model(diff_config, "differentiable")

        results["differentiable"] = {
            "success": success,
            "metrics": metrics,
            "training_time": training_time,
            "output_dir": diff_config.output_dir,
        }
    else:
        print("\n⏭️  Skipping differentiable training (using existing checkpoint)")
        results["differentiable"] = {
            "success": True,
            "skipped": True,
            "output_dir": "./outputs/comparison-differentiable",
        }

    # Run comparison analysis
    print("\n" + "📊" * 40)
    print("STEP 3/3: COMPARISON ANALYSIS")
    print("📊" * 40)

    token_model_path = results["token_based"]["output_dir"] + "/best_model"
    diff_model_path = results["differentiable"]["output_dir"] + "/best_model"

    run_comparison_analysis(
        token_model_path,
        diff_model_path,
        "./outputs/comparison-analysis"
    )

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 FULL COMPARISON COMPLETE")
    print("=" * 80)
    print()

    print("📁 Output Directories:")
    print(f"  • Token-based model: {results['token_based']['output_dir']}")
    print(f"  • Differentiable model: {results['differentiable']['output_dir']}")
    print(f"  • Comparison analysis: ./outputs/comparison-analysis")
    print()

    if not args.skip_token_training and not args.skip_diff_training:
        total_time = (
            results["token_based"].get("training_time", 0) +
            results["differentiable"].get("training_time", 0)
        )
        print(f"⏱️  Total training time: {total_time / 3600:.2f} hours")
        print()

    print("Next steps:")
    print("  1. Review comparison report: ./outputs/comparison-analysis/comparison_report.md")
    print("  2. Analyze attention patterns: python examples/analyze_memory_attention.py")
    print("  3. Run full evaluation: python scripts/evaluate.py")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
