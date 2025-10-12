#!/usr/bin/env python3
"""
Long SQuAD v2 - Main (8 Tokens)
===============================

Dataset: huutuan/long_squad_v2 (long documents, 6-12 segments)
Progressive segments: [2, 4, 6]
Memory: 8 tokens with gating

Purpose: Main result on long documents

Output: outputs/paper_v2_long_08_main_8tokens/
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
        dataset_name="huutuan/long_squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=None,
        max_eval_samples=None,
        use_lazy_loading=False,
        progressive_segments=[2, 4, 6],
        max_n_segs=6,
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
        output_dir="./outputs/paper_v2_long_08_main_8tokens",
        run_name="paper-v2-long-main_8tokens",
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


def main():
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT 08: LONG SQUAD V2 - MAIN (8 TOKENS)")
    print("=" * 80 + "\n")
    config = create_config()
    trainer = XLNetRecurrentTrainer(config)
    try:
        trainer.train()
        print(f"\n✅ Completed: {config.output_dir}")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        raise


if __name__ == "__main__":
    main()
