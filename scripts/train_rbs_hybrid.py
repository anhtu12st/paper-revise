#!/usr/bin/env python3
"""
RBS-QA Hybrid Training Script

Usage:
    python train_rbs_hybrid.py --config configs/rbs_balanced.yaml
    python train_rbs_hybrid.py --output_dir ./experiment1 --num_epochs 15
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import set_seed

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rbsqa.training.hybrid_trainer import RBSHybridTrainer
from rbsqa.configs.hybrid_training_config import RBSTrainingConfig, create_balanced_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RBS-QA Hybrid Training")

    # Config file
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")

    # Override common parameters
    parser.add_argument("--output_dir", type=str, default="./outputs/rbs_experiment",
                       help="Output directory")
    parser.add_argument("--run_name", type=str, default="rbs-experiment",
                       help="Run name for logging")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--memory_num_tokens", type=int, default=None,
                       help="Number of memory tokens")
    parser.add_argument("--num_memory_experts", type=int, default=None,
                       help="Number of memory experts")

    # Training mode
    parser.add_argument("--use_rl_training", action="store_true", default=None,
                       help="Enable RL training")
    parser.add_argument("--no_rl_training", dest="use_rl_training", action="store_false",
                       help="Disable RL training")

    # Data
    parser.add_argument("--train_file", type=str, required=True,
                       help="Training data file")
    parser.add_argument("--eval_file", type=str, default=None,
                       help="Evaluation data file")

    # Model
    parser.add_argument("--model_name_or_path", type=str, default="xlnet-base-cased",
                       help="Base model name or path")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume from checkpoint")

    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--no_mixed_precision", dest="mixed_precision", action="store_false",
                       help="Disable mixed precision")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="rbs-qa",
                       help="WandB project name")

    # Quick config presets
    parser.add_argument("--quick_debug", action="store_true",
                       help="Use quick debug configuration")
    parser.add_argument("--balanced", action="store_true",
                       help="Use balanced configuration preset")

    return parser.parse_args()


def load_dataset_from_file(file_path: str):
    """Load dataset from file. This is a placeholder - actual implementation depends on data format."""
    # For now, create a simple dummy dataset for testing
    # In practice, this should load the actual RBS dataset

    class DummyDataset:
        def __init__(self, file_path: str, split: str = "train"):
            self.file_path = file_path
            self.split = split
            self.data = self._load_dummy_data()

        def _load_dummy_data(self):
            # Create dummy data for testing
            # Each item should have: input_ids, attention_mask, start_positions, end_positions
            # and optionally segment_ids, question_input_ids, context_segments

            dummy_data = []
            for i in range(100):  # 100 examples
                seq_len = 512
                # Create dummy token IDs
                input_ids = torch.randint(1, 1000, (seq_len,))
                attention_mask = torch.ones(seq_len)

                # Dummy start and end positions
                start_pos = torch.randint(0, seq_len // 2, (1,)).item()
                end_pos = torch.randint(start_pos + 1, seq_len, (1,)).item()

                # Segment information (multi-segment document)
                num_segments = 4
                segment_length = seq_len // num_segments
                segment_ids = torch.arange(num_segments).repeat_interleave(segment_length)[:seq_len]

                item = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'start_positions': torch.tensor(start_pos),
                    'end_positions': torch.tensor(end_pos),
                    'segment_ids': segment_ids,
                    'segment_offsets': torch.tensor([i * segment_length for i in range(num_segments)]),
                    'num_segments': torch.tensor(num_segments),
                    'global_start_positions': torch.tensor(start_pos),
                    'global_end_positions': torch.tensor(end_pos),
                    'question_length': 64,  # First 64 tokens are question
                }

                # Split into question and context
                item['question_input_ids'] = input_ids[:64]
                context_segments = []
                for seg_idx in range(num_segments):
                    start = seg_idx * segment_length
                    end = min((seg_idx + 1) * segment_length, seq_len)
                    context_segments.append(input_ids[start:end])
                item['context_segments'] = context_segments

                dummy_data.append(item)

            return dummy_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return DummyDataset(file_path)


def main():
    args = parse_args()

    # Load or create config
    if args.config:
        config = RBSTrainingConfig.load(args.config)
    elif args.quick_debug:
        config = create_balanced_config(
            num_epochs=2,
            batch_size=2,
            memory_num_tokens=4,
            num_memory_experts=2,
            use_rl_training=True,
            output_dir=args.output_dir,
            run_name=args.run_name
        )
    elif args.balanced:
        config = create_balanced_config(
            num_epochs=args.num_epochs or 10,
            batch_size=args.batch_size or 8,
            memory_num_tokens=args.memory_num_tokens or 16,
            num_memory_experts=args.num_memory_experts or 4,
            use_rl_training=args.use_rl_training if args.use_rl_training is not None else True,
            output_dir=args.output_dir,
            run_name=args.run_name
        )
    else:
        config = RBSTrainingConfig()
        config.output_dir = args.output_dir
        config.run_name = args.run_name

    # Override config with command line arguments
    override_keys = [
        'num_epochs', 'batch_size', 'learning_rate', 'memory_num_tokens',
        'num_memory_experts', 'use_rl_training', 'device', 'mixed_precision',
        'seed', 'use_wandb', 'wandb_project'
    ]

    for key in override_keys:
        value = getattr(args, key, None)
        if value is not None:
            setattr(config, key, value)

    # Setup output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=getattr(logging, config.log_level.upper()),
        handlers=[
            logging.FileHandler(os.path.join(config.output_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Set seed
    set_seed(config.seed)
    torch.manual_seed(config.seed)

    # Setup device
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info(f"Loading training data from: {args.train_file}")
    train_dataset = load_dataset_from_file(args.train_file)

    eval_dataset = None
    if args.eval_file:
        logger.info(f"Loading evaluation data from: {args.eval_file}")
        eval_dataset = load_dataset_from_file(args.eval_file)

    # Initialize model
    logger.info(f"Initializing RBS-XLNet model with base: {args.model_name_or_path}")

    # Import here to avoid circular imports
    from rbsqa.models.rbs_xlnet import RBSXLNetForQA

    model = RBSXLNetForQA(
        base_model_name=args.model_name_or_path,
        memory_num_tokens=config.memory_num_tokens,
        num_memory_experts=config.num_memory_experts,
        use_rbs_mode=config.use_rbs_mode
    )

    model = model.to(device)

    # Initialize trainer
    logger.info("Initializing hybrid trainer...")
    trainer = RBSHybridTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    # Save config
    config_path = os.path.join(config.output_dir, "training_config.json")
    config.save(config_path)
    logger.info(f"Training config saved to: {config_path}")

    # Start training
    logger.info("Starting hybrid training...")
    training_results = trainer.train()

    logger.info("Training completed successfully!")
    logger.info(f"Final results: {training_results}")


if __name__ == "__main__":
    main()