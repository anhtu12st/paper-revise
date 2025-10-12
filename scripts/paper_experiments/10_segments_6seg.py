#!/usr/bin/env python3
"""
Segment Analysis: 6 Segments Maximum
=====================================

This script trains with maximum 6 segments to analyze
performance on long documents.

Output: outputs/paper_exp_10_segments_6seg/
"""

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_config():
    """Create configuration for 6 segments."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    config = TrainingConfig(
        model_name="xlnet-base-cased",
        max_seq_length=384,
        doc_stride=64,
        dataset_name="squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=None,
        max_eval_samples=None,
        use_lazy_loading=False,
        # UP TO 6 SEGMENTS (full progression)
        progressive_segments=[2, 4, 6],
        max_n_segs=6,
        # Standard memory config
        memory_num_tokens=8,
        memory_update="gated",
        memory_init="learned",
        memory_impl="token",
        use_global_softmax=True,
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
        output_dir="./outputs/paper_exp_10_segments_6seg",
        run_name="paper-segments-6seg",
        save_total_limit=3,
        no_answer_threshold=1.5,
        use_any_positive_logic=True,
        device=device,
        fp16=has_cuda,
        warmup_freeze_base_epochs=0,
        warmup_disable_global_softmax_epochs=1,
        warmup_disable_any_positive_epochs=0,
        push_to_hub_on_save=False,
    )
    return config


def print_experiment_info(config):
    """Print experiment information."""
    print("\n" + "=" * 80)
    print("📊 PAPER EXPERIMENT 10: SEGMENT ANALYSIS - 6 SEGMENTS")
    print("=" * 80)
    print()
    print("🎯 OBJECTIVE:")
    print("   Analyze performance on long documents (6 segments max)")
    print()
    print("📋 KEY DIFFERENCE:")
    print(f"   • Max segments: {config.max_n_segs}")
    print(f"   • Progressive: {config.progressive_segments} (full curriculum)")
    print()
    print("💾 OUTPUT:")
    print(f"   • Directory: {config.output_dir}")
    print()
    print("=" * 80 + "\n")


def main():
    """Run segment analysis experiment."""
    print("🚀 Starting Segment Analysis 10 (6 Segments)...\n")
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
