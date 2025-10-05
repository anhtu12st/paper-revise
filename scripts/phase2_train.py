"""
Phase 2: Short-Range Recurrence Training for XLNet Recurrent Memory (MemXLNet-QA)

This script continues from Phase 1 (single-segment warmup) and enables
explicit memory with gated updates, training over a small number of segments
per document to establish short-range recurrence (e.g., 3 → 6 segments).
"""

import logging
import torch
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

# Set up logging
logger = logging.getLogger(__name__)


def create_phase2_config():
    """
    Create training configuration for Phase 2 (short-range recurrence).

    Assumes Phase 1 produced a best checkpoint (single-segment warmup).
    Set model_name to that checkpoint to continue training with memory enabled.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    config = TrainingConfig(
        # Start from Phase 1 best model (adjust path if different in your runs)
        model_name="xlnet-base-cased",

        # Sequence settings
        max_seq_length=384,
        doc_stride=64,

        # Dataset (full SQuAD v2 long)
        dataset_name="squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=None,   # Full train
        max_eval_samples=None,    # Full eval
        use_lazy_loading=False,

        # Train with longer context to actually learn memory usage
        progressive_segments=[2],
        max_n_segs=None,

        # Streaming/data
        streaming_chunk_size=3000,
        max_memory_gb=64.0,
        use_streaming=False,

        # Training hyperparameters
        num_epochs=4,
        train_batch_size=16,       # reduce to allow longer context / overlap
        eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,

        # Performance & cadence
        gradient_accumulation_steps=1,
        eval_steps=6_000,
        save_steps=10_000,
        logging_steps=500,

        # Output
        output_dir="./outputs/xlnet-squad-phase2-1",
        run_name="xlnet-squad-phase2-1-gated",
        save_total_limit=5,

        # Evaluation
        no_answer_threshold=1.5,
        use_any_positive_logic=True,

        # Experiment tracking
        use_wandb=False,
        wandb_project="xlnet-long-squad-phase2",

        # Device
        device=device,
        fp16=has_cuda,

        # MemXLNet-QA flags (Phase 2: enable gated memory + global span)
        memory_num_tokens=8,       # start smaller to reduce early noise
        memory_update="gated",
        memory_init="learned",
        memory_impl="token",
        use_global_softmax=True,

        # Warmup behavior: train base together; defer global softmax for 1 epoch
        warmup_freeze_base_epochs=0,
        warmup_disable_global_softmax_epochs=1,
        warmup_disable_any_positive_epochs=0,

        # HuggingFace Hub integration (optional)
        hub_model_id="anhtu12st/memxlnet-squad",  # Set to your Hub repository ID
        push_to_hub_on_save=True,  # Enable automatic push to Hub
        hub_private=False,  # Make repository public
        hub_token=None,  # Uses HF_TOKEN environment variable
        hub_strategy="best_only",  # Only push best models to Hub
    )

    return config


def print_training_info(config):
    """Print a concise summary of Phase 2 training setup."""
    approx_train_examples = 130000
    approx_eval_examples = 12000

    eff_bsz = config.train_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = approx_train_examples // eff_bsz if eff_bsz else 0
    total_steps = steps_per_epoch * config.num_epochs

    print("🚀" + "=" * 80)
    print("📊 PHASE 2: SHORT-RANGE RECURRENCE (MEMORY-ENABLED)")
    print("🚀" + "=" * 80)
    print()

    print("📚 DATASET:")
    print(f"   • Train ~{approx_train_examples:,} | Eval ~{approx_eval_examples:,}")
    print(f"   • {config.dataset_name} [{config.train_split}/{config.eval_split}]")
    print()

    print("🎯 CONFIG:")
    print(f"   • Init model: {config.model_name}")
    print(f"   • Seq len: {config.max_seq_length} | Doc stride: {config.doc_stride}")
    print(f"   • Progressive segments: {config.progressive_segments}")
    print(f"   • Memory: tokens={config.memory_num_tokens}, update={config.memory_update}, global_softmax={config.use_global_softmax}")
    print()

    print("⚡ PERFORMANCE:")
    print(f"   • Device: {config.device} | FP16: {config.fp16}")
    print(f"   • Train/Eval batch: {config.train_batch_size}/{config.eval_batch_size}")
    print(f"   • Grad accum: {config.gradient_accumulation_steps} -> Effective BS: {eff_bsz}")
    print(f"   • Max memory: {config.max_memory_gb}GB")
    print()

    print("📈 SCHEDULE:")
    print(f"   • Steps/epoch: ~{steps_per_epoch:,}")
    print(f"   • Total steps: ~{total_steps:,}")
    print(f"   • Eval every: {config.eval_steps:,} steps | Save every: {config.save_steps:,} steps | Log every: {config.logging_steps:,} steps")
    print()

    print("💾 OUTPUT:")
    print(f"   • Dir: {config.output_dir}")
    print(f"   • Keep last checkpoints: {config.save_total_limit}")
    print()

    print("🤗 HUGGINGFACE HUB:")
    if config.push_to_hub_on_save and config.hub_model_id:
        print(f"   • Repository: {config.hub_model_id}")
        print(f"   • Push strategy: {config.hub_strategy}")
        print(f"   • Private: {config.hub_private}")
    else:
        print("   • Disabled (models saved locally only)")
    print("\n" + "=" * 84)


def main():
    print("🔧 Setting up Phase 2 (Short-Range Recurrence) Training...\n")
    config = create_phase2_config()
    print_training_info(config)

    # Auto-continue without interactive prompt; adjust if you want an input gate
    print("✅ Starting Phase 2 training...\n")

    trainer = XLNetRecurrentTrainer(config)

    try:
        trainer.train()
        print("\n🎉 Phase 2 training completed!")
        print(f"📁 Models saved to: {config.output_dir}")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print(f"📁 Partial results available in: {config.output_dir}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()