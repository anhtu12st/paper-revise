#!/usr/bin/env python3
"""Main training script for MemXLNet-QA.

This script provides a command-line interface for training MA-XLNet models
with memory-augmented question answering capabilities.
"""

from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer


def main():
    """Run basic training with default configuration."""
    print("🚀 MemXLNet-QA Training")
    print("=" * 50)

    # Create default training configuration
    config = TrainingConfig(
        model_name="xlnet-base-cased",
        dataset_name="squad_v2",
        output_dir="outputs/xlnet-squad",
        num_epochs=3,
        memory_num_tokens=32,
        memory_update="gated",
        memory_init="learned",
        use_global_softmax=True,
    )

    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Output: {config.output_dir}")
    print(f"Memory tokens: {config.memory_num_tokens}")
    print("=" * 50)

    # Create and run trainer
    trainer = XLNetRecurrentTrainer(config)
    trainer.train()

    print("\n✅ Training complete!")
    print(f"Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
