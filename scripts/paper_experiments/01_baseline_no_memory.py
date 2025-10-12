#!/usr/bin/env python3
"""
Baseline Experiment: XLNet without Memory Tokens
=================================================

This script trains a standard XLNet model WITHOUT memory tokens
to establish baseline performance.

Output: outputs/paper_exp_01_baseline_no_memory/
"""

import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_config():
    """Create baseline configuration without memory tokens."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    config = TrainingConfig(
        # Model
        model_name="xlnet-base-cased",
        # Sequence settings
        max_seq_length=384,
        doc_stride=64,
        # Dataset
        dataset_name="squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=None,  # Full dataset
        max_eval_samples=None,
        use_lazy_loading=False,
        # Progressive training
        progressive_segments=[2, 4, 6],
        max_n_segs=6,
        # NO MEMORY TOKENS (baseline)
        memory_num_tokens=0,
        memory_update="none",
        memory_init="zeros",
        memory_impl="token",
        use_global_softmax=False,  # No global softmax without memory
        # Training hyperparameters
        num_epochs=3,
        train_batch_size=16,
        eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        # Performance
        gradient_accumulation_steps=1,
        eval_steps=6000,
        save_steps=10000,
        logging_steps=500,
        # Output
        output_dir="./outputs/paper_exp_01_baseline_no_memory",
        run_name="paper-baseline-no-memory",
        save_total_limit=3,
        # Evaluation
        no_answer_threshold=1.5,
        use_any_positive_logic=True,
        # Device
        device=device,
        fp16=has_cuda,
        # Warmup
        warmup_freeze_base_epochs=0,
        warmup_disable_global_softmax_epochs=0,
        warmup_disable_any_positive_epochs=0,
        # Hub (optional)
        push_to_hub_on_save=False,
    )

    return config


def print_experiment_info(config):
    """Print experiment information."""
    print("\n" + "=" * 80)
    print("📊 PAPER EXPERIMENT 01: BASELINE (NO MEMORY)")
    print("=" * 80)
    print()
    print("🎯 OBJECTIVE:")
    print("   Establish baseline XLNet performance without memory augmentation")
    print()
    print("📋 CONFIGURATION:")
    print(f"   • Model: {config.model_name}")
    print(f"   • Memory tokens: {config.memory_num_tokens} (DISABLED)")
    print(f"   • Progressive segments: {config.progressive_segments}")
    print(f"   • Epochs: {config.num_epochs}")
    print(f"   • Batch size: {config.train_batch_size}")
    print(f"   • Learning rate: {config.learning_rate}")
    print()
    print("💾 OUTPUT:")
    print(f"   • Directory: {config.output_dir}")
    print()
    print("=" * 80 + "\n")


def main():
    """Run baseline experiment."""
    print("🚀 Starting Baseline Experiment (No Memory)...\n")

    config = create_config()
    print_experiment_info(config)

    trainer = XLNetRecurrentTrainer(config)

    try:
        trainer.train()
        print("\n✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {config.output_dir}")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print(f"📁 Partial results: {config.output_dir}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
