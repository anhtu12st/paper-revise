#!/usr/bin/env python3
"""
Standalone MemXLNet Training Script for SQuAD v2
================================================

This script trains a Memory-Augmented XLNet model on SQuAD v2 and automatically
uploads checkpoints to HuggingFace Hub.

SETUP:
------
1. Set HF_TOKEN environment variable:
   export HF_TOKEN='your_huggingface_token'

2. Configure hub_model_id below (line 35)

3. Run:
   python scripts/train_memxlnet_squad.py

The script will:
- Train MemXLNet with 8 memory tokens on full SQuAD v2
- Save checkpoints to ./outputs/memxlnet-squad
- Automatically push best models to HuggingFace Hub
- Use GPU if available, otherwise CPU
"""

import logging
import os

import torch

from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# ============================================================================

# Your HuggingFace repository (e.g., "username/memxlnet-squad")
HUB_MODEL_ID = "anhtu12st/memxlnet-squad"

# Optional: Set HF token here if not using environment variable
# HF_TOKEN = "hf_..."  # Uncomment and set if needed

# ============================================================================
# TRAINING CONFIGURATION - DEFAULT VALUES
# ============================================================================

def get_training_config():
    """Create production-ready training configuration."""

    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    # Get HuggingFace token from environment or config above
    hf_token = os.environ.get("HF_TOKEN", None)

    config = TrainingConfig(
        # ===== Model Configuration =====
        model_name="xlnet-base-cased",
        max_seq_length=384,
        doc_stride=64,

        # ===== Dataset Configuration =====
        dataset_name="squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=None,      # Use full training set
        max_eval_samples=None,       # Use full evaluation set
        use_lazy_loading=False,

        # ===== Progressive Training =====
        progressive_segments=[2],    # Start with 2 segments
        max_n_segs=None,

        # ===== Data Processing =====
        streaming_chunk_size=3000,
        max_memory_gb=64.0,
        use_streaming=False,

        # ===== Training Hyperparameters =====
        num_epochs=4,
        train_batch_size=16,
        eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,

        # ===== Training Schedule =====
        gradient_accumulation_steps=1,
        eval_steps=6000,
        save_steps=10000,
        logging_steps=500,

        # ===== Output Configuration =====
        output_dir="./outputs/memxlnet-squad",
        run_name="memxlnet-squad-production",
        save_total_limit=5,

        # ===== Evaluation Settings =====
        no_answer_threshold=1.5,
        use_any_positive_logic=True,

        # ===== Experiment Tracking =====
        use_wandb=False,
        wandb_project="memxlnet-squad",

        # ===== Device Settings =====
        device=device,
        fp16=has_cuda,

        # ===== Memory-Augmented XLNet Settings =====
        memory_num_tokens=8,
        memory_update="gated",
        memory_init="learned",
        memory_impl="token",
        bptt_horizon=6,
        use_global_softmax=True,

        # ===== Warmup Strategy =====
        warmup_freeze_base_epochs=0,
        warmup_disable_global_softmax_epochs=1,
        warmup_disable_any_positive_epochs=0,

        # ===== HuggingFace Hub Integration =====
        hub_model_id=HUB_MODEL_ID,
        push_to_hub_on_save=True,
        hub_private=False,
        hub_token=hf_token,
        hub_strategy="best_only",
    )

    return config


def print_training_summary(config):
    """Print training configuration summary."""
    print("\n" + "=" * 80)
    print("🚀 MemXLNet Training on SQuAD v2")
    print("=" * 80)
    print()

    print("📊 DATASET:")
    print(f"   • Dataset: {config.dataset_name}")
    print(f"   • Train split: {config.train_split} (full dataset)")
    print(f"   • Eval split: {config.eval_split} (full dataset)")
    print()

    print("🧠 MODEL:")
    print(f"   • Base: {config.model_name}")
    print(f"   • Memory tokens: {config.memory_num_tokens}")
    print(f"   • Memory update: {config.memory_update}")
    print(f"   • Global softmax: {config.use_global_softmax}")
    print(f"   • Progressive segments: {config.progressive_segments}")
    print()

    print("⚡ TRAINING:")
    print(f"   • Epochs: {config.num_epochs}")
    print(f"   • Train batch size: {config.train_batch_size}")
    print(f"   • Eval batch size: {config.eval_batch_size}")
    print(f"   • Learning rate: {config.learning_rate}")
    print(f"   • Device: {config.device}")
    print(f"   • FP16: {config.fp16}")
    print()

    print("💾 OUTPUT:")
    print(f"   • Local: {config.output_dir}")
    print(f"   • Eval every: {config.eval_steps:,} steps")
    print(f"   • Save every: {config.save_steps:,} steps")
    print()

    print("🤗 HUGGINGFACE HUB:")
    if config.push_to_hub_on_save and config.hub_model_id:
        print(f"   • Repository: {config.hub_model_id}")
        print(f"   • Strategy: {config.hub_strategy} (only best models)")
        print(f"   • Private: {config.hub_private}")
        token_status = "✓ Set" if config.hub_token else "✗ Not set (using HF_TOKEN env)"
        print(f"   • Token: {token_status}")
    else:
        print("   • ✗ Disabled")
    print()
    print("=" * 80)
    print()


def main():
    """Main training entry point."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create configuration
    print("🔧 Initializing training configuration...")
    config = get_training_config()

    # Print summary
    print_training_summary(config)

    # Validate HuggingFace configuration
    if config.push_to_hub_on_save and not config.hub_model_id:
        print("❌ Error: hub_model_id not set!")
        print("   Please edit this script and set HUB_MODEL_ID (line 35)")
        return 1

    if config.push_to_hub_on_save and not config.hub_token and not os.environ.get("HF_TOKEN"):
        print("⚠️  Warning: HF_TOKEN not found in environment!")
        print("   Hub uploads may fail without authentication.")
        print()
        response = input("Continue anyway? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            return 1

    # Initialize trainer
    print("🏗️  Initializing trainer...")
    trainer = XLNetRecurrentTrainer(config)

    # Start training
    print("\n✅ Starting training...\n")

    try:
        trainer.train()
        print("\n" + "=" * 80)
        print("🎉 Training completed successfully!")
        print("=" * 80)
        print(f"📁 Local checkpoints: {config.output_dir}")
        if config.push_to_hub_on_save and config.hub_model_id:
            print(f"🤗 Hub repository: https://huggingface.co/{config.hub_model_id}")
        print("=" * 80)
        return 0

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠️  Training interrupted by user")
        print("=" * 80)
        print(f"📁 Partial results: {config.output_dir}")
        return 1

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ Training failed: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    exit(main())
