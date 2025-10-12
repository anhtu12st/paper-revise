#!/usr/bin/env python3
"""
Standard SQuAD v2 - Baseline (No Memory)
=========================================

Dataset: squad_v2 (standard, short documents)
Max segments: 2 (realistic for SQuAD v2)
Memory: None (baseline)

Purpose: Establish baseline performance on standard benchmark.

Output: outputs/paper_v2_squad_01_baseline/
"""

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_config():
    """Create baseline configuration for standard SQuAD v2."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    config = TrainingConfig(
        model_name="xlnet-base-cased",
        max_seq_length=384,
        doc_stride=64,
        # Standard SQuAD v2 dataset
        dataset_name="squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=None,
        max_eval_samples=None,
        use_lazy_loading=False,
        # Realistic segment count for SQuAD v2 (most docs are 1-2 segments)
        progressive_segments=[2],
        max_n_segs=2,
        # NO MEMORY (baseline)
        memory_num_tokens=0,
        memory_update="none",
        memory_init="zeros",
        memory_impl="token",
        use_global_softmax=False,
        # Training hyperparameters
        num_epochs=3,
        train_batch_size=16,
        eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        eval_steps=6000,
        save_steps=10000,
        logging_steps=500,
        # Output
        output_dir="./outputs/paper_v2_squad_01_baseline",
        run_name="paper-v2-squad-baseline",
        save_total_limit=3,
        # Evaluation
        no_answer_threshold=1.5,
        use_any_positive_logic=True,
        # Device
        device=device,
        fp16=has_cuda,
        warmup_freeze_base_epochs=0,
        warmup_disable_global_softmax_epochs=0,
        warmup_disable_any_positive_epochs=0,
        push_to_hub_on_save=False,
    )
    return config


def print_experiment_info(config):
    """Print experiment information."""
    print("\n" + "=" * 80)
    print("📊 HYBRID EXPERIMENT 01: SQUAD V2 - BASELINE (NO MEMORY)")
    print("=" * 80)
    print()
    print("🎯 PURPOSE:")
    print("   Establish baseline performance on standard SQuAD v2")
    print()
    print("📋 CONFIGURATION:")
    print(f"   • Dataset: {config.dataset_name} (standard, short documents)")
    print(f"   • Max segments: {config.max_n_segs} (realistic for SQuAD v2)")
    print(f"   • Memory tokens: {config.memory_num_tokens} (DISABLED)")
    print(f"   • Epochs: {config.num_epochs}")
    print(f"   • Batch size: {config.train_batch_size}")
    print()
    print("💾 OUTPUT:")
    print(f"   • Directory: {config.output_dir}")
    print()
    print("=" * 80 + "\n")


def main():
    """Run experiment."""
    print("🚀 Starting Squad V2 Baseline Experiment...\n")
    config = create_config()
    print_experiment_info(config)
    trainer = XLNetRecurrentTrainer(config)

    try:
        trainer.train()
        print("\n✅ Experiment completed!")
        print(f"📁 Results: {config.output_dir}")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        raise


if __name__ == "__main__":
    main()
