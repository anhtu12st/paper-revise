#!/usr/bin/env python3
"""
Investigate Training Issue
==========================

This script helps diagnose why the comparison script produced empty metrics.
It simulates what should have happened during evaluation.

Usage:
    python scripts/investigate_training_issue.py
"""

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def investigate_evaluation_issue():
    """Investigate why evaluation didn't produce metrics."""
    print("\n" + "=" * 80)
    print("🔍 INVESTIGATING TRAINING/EVALUATION ISSUE")
    print("=" * 80)
    print()

    # Recreate the exact config used in comparison
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    print(f"💻 Device: {device}")
    print(f"🎮 CUDA available: {has_cuda}")
    print()

    # Create config matching comparison script
    config = TrainingConfig(
        model_name="xlnet-base-cased",
        max_seq_length=384,
        doc_stride=64,
        dataset_name="squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache_comparison",
        max_train_samples=100,  # Small for quick test
        max_eval_samples=50,  # Small for quick test
        use_lazy_loading=False,
        progressive_segments=[2],
        max_n_segs=2,
        memory_num_tokens=8,
        memory_update="gated",
        memory_init="learned",
        use_global_softmax=True,
        num_epochs=1,  # Just 1 epoch for testing
        train_batch_size=4,
        eval_batch_size=4,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        eval_steps=25,  # Eval every 25 steps for quick testing
        save_steps=500,
        logging_steps=10,
        save_total_limit=2,
        no_answer_threshold=0.0,
        use_any_positive_logic=True,
        device=device,
        fp16=has_cuda,
        warmup_freeze_base_epochs=0,
        warmup_disable_any_positive_epochs=0,
        push_to_hub_on_save=False,
        use_wandb=False,
        memory_impl="differentiable",
        use_differentiable_memory=True,
        num_memory_heads=2,
        memory_slots=16,
        memory_sharpness=1.5,
        enable_usage_tracking=True,
        enable_temporal_links=True,
        output_dir="./outputs/investigation_test",
        run_name="investigation-test",
    )

    print("📝 Configuration:")
    print(f"   Training samples: {config.max_train_samples}")
    print(f"   Eval samples: {config.max_eval_samples}")
    print(f"   Eval steps: {config.eval_steps}")
    print(f"   Epochs: {config.num_epochs}")
    print()

    try:
        print("🚀 Creating trainer...")
        trainer = XLNetRecurrentTrainer(config)

        print("✅ Trainer created successfully")
        print()

        # Check if trainer has best_metrics attribute
        print("🔍 Checking trainer attributes...")
        has_best_metrics = hasattr(trainer, "best_metrics")
        print(f"   - has 'best_metrics': {has_best_metrics}")

        # Check what attributes the trainer does have
        relevant_attrs = [
            attr for attr in dir(trainer) if not attr.startswith("_") and not callable(getattr(trainer, attr))
        ][:20]
        print(f"   - First 20 non-private attributes: {relevant_attrs}")
        print()

        print("🧪 Testing evaluation directly (without training)...")
        print("   This tests if evaluation code works at all...")
        print()

        # Prepare data first to get eval_dataloader
        print("   Preparing evaluation data...")
        train_dataloader, eval_dataloader, eval_dataset = trainer.prepare_data()
        print(f"   ✅ Prepared eval dataloader with {len(eval_dataloader)} batches")
        print()

        # Try to run evaluation directly
        print("   Running evaluation...")
        metrics = trainer.evaluate(eval_dataloader, eval_dataset)

        print("\n✅ Evaluation completed!")
        print(f"\n📊 Metrics returned: {metrics}")
        print()

        if not metrics:
            print("⚠️  WARNING: Empty metrics dict returned!")
            print("   This is the same issue seen in the comparison script.")
            print()
            print("   Possible causes:")
            print("   1. Evaluation code has a bug")
            print("   2. No predictions were extracted")
            print("   3. Metrics calculation failed silently")
            print()
        else:
            print("✅ Metrics look good! Issue might be specific to long training runs.")
            print()

        # Check if trainer now has best_metrics after evaluation
        if hasattr(trainer, "best_metrics"):
            print(f"✅ trainer.best_metrics exists: {trainer.best_metrics}")
        else:
            print("⚠️  trainer.best_metrics still doesn't exist after evaluation")
        print()

        print("=" * 80)
        print("🎯 DIAGNOSIS")
        print("=" * 80)
        print()

        if not metrics:
            print("❌ ROOT CAUSE: Evaluation returns empty metrics")
            print()
            print("   The field name fix we applied should resolve this.")
            print("   The issue was:")
            print("   - Evaluation looked for 'offset_mappings' and 'contexts' (plural)")
            print("   - But collate function stored as 'offset_mapping' and 'context' (singular)")
            print("   - This caused all arrays to be empty, leading to 0 predictions")
            print()
            print("   SOLUTION: The fix has been applied to trainer.py lines 1589 and 1595")
            print("   Run the comparison script again to verify the fix works.")
        else:
            print("✅ Evaluation works correctly in this test")
            print()
            print("   Possible explanations for comparison script failure:")
            print("   1. eval_steps=5000 was too large, so no evaluation ran")
            print("   2. Evaluation ran but metrics weren't saved to trainer.best_metrics")
            print("   3. Training was interrupted before evaluation")
            print()
            print("   SOLUTION: Compare script needs updating to:")
            print("   - Call trainer.evaluate() explicitly after training")
            print("   - Or reduce eval_steps to ensure evaluation runs during training")

        print()
        return 0 if metrics else 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main():
    return investigate_evaluation_issue()


if __name__ == "__main__":
    sys.exit(main())
