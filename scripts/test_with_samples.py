#!/usr/bin/env python3
"""
Test with Sample Data
=====================

Quickly test experiments with a small subset of chunked data.
Useful for debugging and development without waiting for full preprocessing.

Usage:
    # Test with first 100 documents
    python scripts/test_with_samples.py --num-examples 100

    # Test specific chunks
    python scripts/test_with_samples.py --chunks 0 1 2

    # Test with custom chunked dataset
    python scripts/test_with_samples.py --dataset-dir ./preprocessed_data/squad_v2 --num-examples 50
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test training with sample data from chunked datasets")

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./preprocessed_data/squad_v2",
        help="Chunked dataset directory (default: ./preprocessed_data/squad_v2)",
    )
    parser.add_argument("--num-examples", type=int, default=100, help="Number of examples to load (default: 100)")
    parser.add_argument("--chunks", nargs="+", type=int, help="Specific chunk indices to load (optional)")
    parser.add_argument("--memory-tokens", type=int, default=8, help="Number of memory tokens (default: 8)")
    parser.add_argument("--max-segments", type=int, default=2, help="Maximum segments per document (default: 2)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size (default: 4)")
    parser.add_argument("--eval-steps", type=int, default=50, help="Evaluation frequency (default: 50 steps)")
    parser.add_argument("--output-dir", type=str, default="./outputs/test_sample", help="Output directory for test run")

    return parser.parse_args()


def main():
    """Main testing function."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("🧪 QUICK TEST WITH SAMPLE DATA")
    print("=" * 80 + "\n")

    print("📋 Configuration:")
    print(f"   • Dataset directory: {args.dataset_dir}")
    print(f"   • Sample size: {args.num_examples if args.chunks is None else f'chunks {args.chunks}'}")
    print(f"   • Memory tokens: {args.memory_tokens}")
    print(f"   • Max segments: {args.max_segments}")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Output: {args.output_dir}")
    print()

    # Detect dataset type from directory name
    dataset_dir_path = Path(args.dataset_dir)
    if "long_squad" in str(dataset_dir_path).lower():
        dataset_name = "huutuan/long_squad_v2"
        print("📊 Detected: Long SQuAD v2 dataset")
    else:
        dataset_name = "squad_v2"
        print("📊 Detected: Standard SQuAD v2 dataset")

    # Determine load mode
    load_mode = "first_n" if args.chunks is None else "chunks"

    print("\n🚀 Loading sample data...")

    # Create configuration with chunked dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainingConfig(
        model_name="xlnet-base-cased",
        max_seq_length=384,
        doc_stride=64,
        dataset_name=dataset_name,
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        # Chunked dataset settings
        use_chunked_dataset=True,
        chunked_dataset_dir=str(args.dataset_dir),
        chunked_load_mode=load_mode,
        chunked_num_examples=args.num_examples if args.chunks is None else None,
        chunked_chunk_indices=args.chunks if args.chunks else None,
        # Training settings
        max_n_segs=args.max_segments,
        memory_num_tokens=args.memory_tokens,
        memory_update="gated",
        memory_init="learned",
        memory_impl="token",
        use_global_softmax=True,
        num_epochs=args.epochs,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size * 2,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps * 2,
        logging_steps=10,
        output_dir=args.output_dir,
        run_name="test-sample",
        save_total_limit=1,
        no_answer_threshold=1.5,
        use_any_positive_logic=True,
        device=device,
        fp16=False,  # Disable FP16 for stability in testing
        warmup_freeze_base_epochs=0,
        warmup_disable_global_softmax_epochs=0,
        warmup_disable_any_positive_epochs=0,
        push_to_hub_on_save=False,
    )

    print("✅ Configuration created")
    print("\n⚙️  Initializing trainer...")

    try:
        trainer = XLNetRecurrentTrainer(config)
        print("✅ Trainer initialized\n")

        print("🎯 Starting training...\n")
        trainer.train()

        print("\n" + "=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n📁 Results: {args.output_dir}")
        print("💡 Tip: Check logs to verify chunked dataset loading worked correctly")
        print()

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.exception("Full error traceback:")
        raise


if __name__ == "__main__":
    main()
