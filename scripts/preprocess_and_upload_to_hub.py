#!/usr/bin/env python3
"""
Preprocess SQuAD v2 Dataset and Upload to HuggingFace Hub
=========================================================

This script preprocesses SQuAD v2 dataset with various memory token configurations
and uploads the preprocessed datasets to HuggingFace Hub for fast training startup.

IMPORTANT: Run this script ONCE on a high-RAM machine (20GB+ RAM recommended).
           Then all future training runs can download the preprocessed data
           from Hub, avoiding expensive preprocessing.

SETUP:
------
1. Set HF_TOKEN environment variable:
   export HF_TOKEN='your_huggingface_token'

2. Configure hub_username below (line 42)

3. Run:
   python scripts/preprocess_and_upload_to_hub.py

WHAT THIS DOES:
---------------
- Processes SQuAD v2 train and validation splits
- Creates variants with different memory token counts (0, 8, 16, 32)
- Uploads each to a separate Hub repository
- Generates README with usage instructions

OUTPUT:
-------
Creates PRIVATE Hub repositories like:
- username/memxlnet-squad-mem0  (no memory tokens) 🔒 PRIVATE
- username/memxlnet-squad-mem8  (8 memory tokens) 🔒 PRIVATE
- username/memxlnet-squad-mem16 (16 memory tokens) 🔒 PRIVATE
- username/memxlnet-squad-mem32 (32 memory tokens) 🔒 PRIVATE

Each repository contains both train and validation splits.

Note: To make repositories public, edit CONFIGS on line 58 and change True to False.
"""

import logging
import os

from transformers import XLNetTokenizerFast

from memxlnet.data import configure_memory_tokens, upload_processed_dataset_to_hub

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# ============================================================================

# Your HuggingFace username (required!)
HUB_USERNAME = "anhtu12st"  # Change this to your username!

# Dataset configurations to process
# Format: (memory_tokens, max_seq_length, doc_stride, max_n_segs, private)
# 🔒 All datasets are PRIVATE by default. Change True to False for public repositories.
CONFIGS = [
    (0, 384, 64, None, True),  # No memory tokens (baseline) - PRIVATE
    (8, 384, 64, None, True),  # 8 memory tokens (recommended) - PRIVATE
    (16, 384, 64, None, True),  # 16 memory tokens - PRIVATE
    (32, 384, 64, None, True),  # 32 memory tokens - PRIVATE
]

# Dataset settings
DATASET_NAME = "squad_v2"
SPLITS = ["train", "validation"]

# Optional: Limit samples for testing (set to None for full dataset)
MAX_TRAIN_SAMPLES = None  # Set to e.g., 1000 for testing
MAX_EVAL_SAMPLES = None  # Set to e.g., 500 for testing

# ============================================================================
# MAIN PREPROCESSING LOGIC - DO NOT MODIFY BELOW THIS LINE
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_config():
    """Validate configuration before starting."""
    if not HUB_USERNAME:
        logger.error("❌ Error: HUB_USERNAME not set!")
        logger.error("   Please edit this script and set HUB_USERNAME (line 42)")
        return False

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("⚠️  Warning: HF_TOKEN not found in environment!")
        logger.warning("   Set it with: export HF_TOKEN='your_token'")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            return False

    return True


def create_hub_repo_name(memory_tokens: int) -> str:
    """Create Hub repository name based on memory token count."""
    return f"{HUB_USERNAME}/memxlnet-squad-mem{memory_tokens}"


def process_and_upload_config(
    memory_tokens: int,
    max_seq_length: int,
    doc_stride: int,
    max_n_segs: int | None,
    private: bool,
):
    """Process and upload dataset for a specific configuration."""
    hub_repo = create_hub_repo_name(memory_tokens)

    logger.info("=" * 80)
    logger.info(f"🚀 Processing configuration: {memory_tokens} memory tokens")
    logger.info(f"   Repository: {hub_repo}")
    logger.info(f"   Max seq length: {max_seq_length}")
    logger.info(f"   Doc stride: {doc_stride}")
    logger.info("=" * 80)

    # Setup tokenizer with memory tokens
    logger.info("🔧 Setting up tokenizer...")
    tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
    if memory_tokens > 0:
        configure_memory_tokens(tokenizer, memory_num_tokens=memory_tokens)
        logger.info(f"   Added {memory_tokens} memory tokens to tokenizer")
        logger.info(f"   Tokenizer vocab size: {len(tokenizer)}")

    hf_token = os.environ.get("HF_TOKEN")

    # Process and upload each split
    for split in SPLITS:
        max_examples = MAX_TRAIN_SAMPLES if split == "train" else MAX_EVAL_SAMPLES

        logger.info(f"\n📊 Processing {split} split...")
        if max_examples:
            logger.info(f"   (Limited to {max_examples} examples for testing)")

        try:
            url = upload_processed_dataset_to_hub(
                dataset_name=DATASET_NAME,
                split=split,
                hub_dataset_id=hub_repo,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_examples=max_examples,
                max_n_segs=max_n_segs,
                hub_token=hf_token,
                private=private,
            )
            logger.info(f"✅ Successfully uploaded {split} to: {url}")

        except Exception as e:
            logger.error(f"❌ Failed to process {split}: {e}")
            raise

    logger.info(f"\n🎉 Completed configuration: {memory_tokens} memory tokens")
    logger.info(f"   View at: https://huggingface.co/datasets/{hub_repo}")


def main():
    """Main preprocessing entry point."""
    logger.info("\n" + "=" * 80)
    logger.info("🔄 SQuAD v2 Preprocessing and Upload to HuggingFace Hub")
    logger.info("=" * 80 + "\n")

    # Validate configuration
    if not validate_config():
        logger.error("Aborted.")
        return 1

    # Print summary
    logger.info("📋 Processing Summary:")
    logger.info(f"   Dataset: {DATASET_NAME}")
    logger.info(f"   Splits: {', '.join(SPLITS)}")
    logger.info(f"   Configurations: {len(CONFIGS)}")
    logger.info(f"   Target username: {HUB_USERNAME}")

    # Check privacy settings
    all_private = all(private for _, _, _, _, private in CONFIGS)
    if all_private:
        logger.info("   🔒 Privacy: All repositories will be PRIVATE")
    else:
        logger.info("   ⚠️  Privacy: Some repositories will be PUBLIC")

    if MAX_TRAIN_SAMPLES or MAX_EVAL_SAMPLES:
        logger.info(f"   ⚠️  LIMITED MODE: train={MAX_TRAIN_SAMPLES}, eval={MAX_EVAL_SAMPLES}")
    logger.info("")

    # Confirm before starting
    response = input("Start preprocessing? [y/N]: ")
    if response.lower() not in ["y", "yes"]:
        logger.info("Aborted.")
        return 1

    logger.info("\n🚀 Starting preprocessing...\n")

    # Process each configuration
    for idx, (mem_tokens, max_len, stride, max_segs, private) in enumerate(CONFIGS, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Configuration {idx}/{len(CONFIGS)}")
        logger.info(f"{'=' * 80}\n")

        try:
            process_and_upload_config(
                memory_tokens=mem_tokens,
                max_seq_length=max_len,
                doc_stride=stride,
                max_n_segs=max_segs,
                private=private,
            )
        except Exception as e:
            logger.error(f"❌ Configuration {idx} failed: {e}")
            logger.error("Continuing with next configuration...")
            continue

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("🎉 All preprocessing completed!")
    logger.info("=" * 80)
    logger.info("\n📚 Preprocessed datasets created:")
    for mem_tokens, _, _, _, _ in CONFIGS:
        repo = create_hub_repo_name(mem_tokens)
        logger.info(f"   • {repo}")
        logger.info(f"     https://huggingface.co/datasets/{repo}")

    logger.info("\n💡 Usage in training:")
    logger.info("   config = TrainingConfig(")
    logger.info(f'       hub_dataset_id="{create_hub_repo_name(8)}",  # Example with 8 memory tokens')
    logger.info("       use_hub_dataset=True,")
    logger.info("   )")
    logger.info("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
