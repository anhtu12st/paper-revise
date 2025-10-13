#!/usr/bin/env python3
"""
Standard SQuAD v2 - Main (8 Memory Tokens, Differentiable Memory)
===================================================================

Dataset: squad_v2 (standard)
Max segments: 2
Memory: 8 tokens (DIFFERENTIABLE implementation)

Purpose: Main result on standard benchmark with WORKING configuration.

This is the FIXED version of 02_main_squad_8tokens.py that uses
differentiable memory instead of token-based memory.

Key differences from 02_main_squad_8tokens.py:
- memory_impl="differentiable" instead of "token"
- Added differentiable memory parameters (heads, slots, sharpness)
- warmup_disable_global_softmax_epochs=0 (enable immediately)

Output: outputs/paper_v2_squad_02b_main_8tokens_diff/
"""

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO)


def create_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    return TrainingConfig(
        model_name="xlnet-base-cased",
        max_seq_length=384,
        doc_stride=64,
        dataset_name="squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=1000,
        max_eval_samples=200,
        use_lazy_loading=True,
        # Chunked dataset settings (FAST: 2-5 min vs 30-60 min preprocessing)
        use_chunked_dataset=True,
        chunked_dataset_dir="./preprocessed_data/squad_v2",
        chunked_load_mode="streaming",  # Memory-efficient streaming
        progressive_segments=[2],
        max_n_segs=2,
        memory_num_tokens=0,  # ✅ FIXED: Differentiable memory doesn't need special tokens
        memory_update="gated",
        memory_init="learned",
        # ✅ FIXED: Use differentiable memory instead of token-based
        memory_impl="differentiable",
        use_differentiable_memory=True,
        # Differentiable memory parameters
        num_memory_heads=2,
        memory_slots=16,
        memory_sharpness=1.5,
        enable_usage_tracking=True,
        enable_temporal_links=True,
        # Global softmax and training settings
        use_global_softmax=False,
        num_epochs=3,
        train_batch_size=8,
        eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        eval_steps=1000,
        save_steps=10000,
        logging_steps=500,
        output_dir="./outputs/paper_v2_squad_02b_main_8tokens_diff",
        run_name="paper-v2-squad-main-8tokens-diff",
        save_total_limit=3,
        no_answer_threshold=1.5,
        use_any_positive_logic=True,
        device=device,
        fp16=has_cuda,
        # ✅ FIXED: Enable global softmax immediately (was 1 in original)
        warmup_freeze_base_epochs=0,
        warmup_disable_global_softmax_epochs=0,
        warmup_disable_any_positive_epochs=0,
        push_to_hub_on_save=False,
    )


def main():
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT 02B: SQUAD V2 - MAIN (8 TOKENS, DIFFERENTIABLE)")
    print("=" * 80 + "\n")
    print("🔧 Configuration changes from 02_main_squad_8tokens.py:")
    print("   ✅ memory_impl: 'token' → 'differentiable'")
    print("   ✅ use_differentiable_memory: False → True")
    print("   ✅ Added: num_memory_heads=2, memory_slots=16, sharpness=1.5")
    print("   ✅ warmup_disable_global_softmax_epochs: 1 → 0")
    print()
    print("Expected outcome: F1 > 0% (should achieve ~60-75% F1)")
    print()

    config = create_config()
    trainer = XLNetRecurrentTrainer(config)

    try:
        trainer.train()

        # Print final metrics if available
        if hasattr(trainer, "best_metrics"):
            metrics = trainer.best_metrics
            print("\n" + "=" * 80)
            print("📊 FINAL RESULTS")
            print("=" * 80)
            print(f"   F1 Score: {metrics.get('f1', 0.0):.2f}%")
            print(f"   Exact Match: {metrics.get('exact_match', 0.0):.2f}%")
            if "has_answer_f1" in metrics:
                print(f"   Has-Answer F1: {metrics['has_answer_f1']:.2f}%")
                print(f"   No-Answer F1: {metrics['no_answer_f1']:.2f}%")
            print("=" * 80)

            # Success criteria
            if metrics.get("f1", 0.0) > 0:
                print("\n✅ SUCCESS: Model is learning (F1 > 0%)")
                if metrics.get("f1", 0.0) >= 60:
                    print("🎉 EXCELLENT: Achieved strong performance (F1 >= 60%)")
            else:
                print("\n⚠️  WARNING: F1 still 0% - further investigation needed")

        print(f"\n✅ Completed: {config.output_dir}")

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        raise


if __name__ == "__main__":
    main()
