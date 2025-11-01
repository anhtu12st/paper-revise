#!/usr/bin/env python3
"""
Inspect Preprocessed Data Script
=================================

This script inspects the preprocessed SQuAD v2 data to understand:
- What files exist
- How many features are cached
- Sample feature structure
- Memory token information
- Token distributions

Usage:
    python scripts/inspect_preprocessed_data.py
    python scripts/inspect_preprocessed_data.py --data-dir preprocessed_data/squad_v2
    python scripts/inspect_preprocessed_data.py --show-samples 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def inspect_directory_structure(data_dir: Path):
    """Inspect the directory structure."""
    logger.info("=" * 80)
    logger.info("📁 DIRECTORY STRUCTURE")
    logger.info("=" * 80)

    if not data_dir.exists():
        logger.warning(f"❌ Directory does not exist: {data_dir}")
        logger.info("")
        logger.info("To create preprocessed data, run:")
        logger.info("  uv run python scripts/preprocess_squad_v2.py --memory-tokens 0")
        return False

    logger.info(f"✅ Directory exists: {data_dir}")
    logger.info("")

    # List all files and directories
    all_items = list(data_dir.rglob("*"))
    dirs = [d for d in all_items if d.is_dir()]
    files = [f for f in all_items if f.is_file()]

    logger.info(f"Total directories: {len(dirs)}")
    logger.info(f"Total files: {len(files)}")
    logger.info("")

    # Show directory tree
    logger.info("Directory tree:")
    for item in sorted(data_dir.rglob("*"))[:50]:  # Limit to 50 items
        if item.is_dir():
            rel_path = item.relative_to(data_dir)
            logger.info(f"  📁 {rel_path}/")
        elif item.is_file():
            rel_path = item.relative_to(data_dir)
            size = item.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"  📄 {rel_path} ({size:.2f} MB)")

    if len(list(data_dir.rglob("*"))) > 50:
        logger.info(f"  ... and {len(list(data_dir.rglob('*'))) - 50} more items")

    logger.info("")
    return True


def inspect_manifest(data_dir: Path):
    """Inspect manifest files if they exist."""
    logger.info("=" * 80)
    logger.info("📋 MANIFEST FILES")
    logger.info("=" * 80)

    manifest_files = list(data_dir.glob("*manifest.json"))

    if not manifest_files:
        logger.info("No manifest files found")
        logger.info("")
        return

    for manifest_path in manifest_files:
        logger.info(f"\n📄 {manifest_path.name}:")
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            for key, value in manifest.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for k, v in value.items():
                        logger.info(f"    {k}: {v}")
                else:
                    logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not read manifest: {e}")

    logger.info("")


def inspect_cached_chunks(data_dir: Path, split: str, max_chunks: int = 3):
    """Inspect cached chunk files."""
    logger.info("=" * 80)
    logger.info(f"📦 CACHED CHUNKS ({split.upper()} split)")
    logger.info("=" * 80)

    # Look for cache files
    cache_pattern = f"squad_v2*_{split}_chunk_*.cache"
    cache_files = list(data_dir.glob(cache_pattern))

    # Also check in subdirectories
    if not cache_files:
        cache_files = list(data_dir.glob(f"**/*_{split}_chunk_*.cache"))

    if not cache_files:
        logger.info(f"No cache files found for split: {split}")
        logger.info(f"  Pattern searched: {cache_pattern}")
        logger.info("")
        return None

    # Sort by chunk number
    cache_files = sorted(cache_files)

    logger.info(f"Found {len(cache_files)} chunk files")
    logger.info("")

    # Load and inspect first few chunks
    total_features = 0
    sample_features = []

    for i, cache_file in enumerate(cache_files[:max_chunks]):
        logger.info(f"📦 Chunk {i}: {cache_file.name}")

        try:
            chunk_data = torch.load(cache_file, map_location="cpu")

            if isinstance(chunk_data, list):
                num_features = len(chunk_data)
                total_features += num_features
                logger.info(f"  Features in this chunk: {num_features}")

                # Collect sample features from first chunk
                if i == 0 and chunk_data:
                    sample_features = chunk_data[:3]  # Get first 3 features

            else:
                logger.warning(f"  ⚠️ Unexpected format: {type(chunk_data)}")

        except Exception as e:
            logger.warning(f"  ⚠️ Could not load chunk: {e}")

    if len(cache_files) > max_chunks:
        logger.info(f"  ... and {len(cache_files) - max_chunks} more chunks")

    # Estimate total features
    if cache_files and total_features > 0:
        avg_per_chunk = total_features / min(len(cache_files), max_chunks)
        estimated_total = int(avg_per_chunk * len(cache_files))
        logger.info("")
        logger.info(f"📊 Estimated total features: ~{estimated_total}")

    logger.info("")
    return sample_features


def inspect_feature_structure(features: list):
    """Inspect the structure of cached features."""
    if not features:
        logger.info("No features to inspect")
        return

    logger.info("=" * 80)
    logger.info("🔍 FEATURE STRUCTURE")
    logger.info("=" * 80)

    # Show keys in first feature
    first_feature = features[0]
    logger.info(f"Number of features inspected: {len(features)}")
    logger.info("")
    logger.info("Keys in feature dictionary:")

    for key in sorted(first_feature.keys()):
        value = first_feature[key]

        if isinstance(value, list):
            if value and isinstance(value[0], tuple):
                logger.info(f"  {key:25s}: list of {len(value)} tuples")
            else:
                logger.info(f"  {key:25s}: list of {len(value)} items")
        elif isinstance(value, str):
            logger.info(f"  {key:25s}: str (length={len(value)})")
        elif isinstance(value, (int, float, bool)):
            logger.info(f"  {key:25s}: {type(value).__name__} = {value}")
        elif isinstance(value, dict):
            logger.info(f"  {key:25s}: dict with {len(value)} keys")
        else:
            logger.info(f"  {key:25s}: {type(value).__name__}")

    logger.info("")


def inspect_sample_features(features: list, num_samples: int = 2):
    """Show detailed information about sample features."""
    if not features:
        return

    logger.info("=" * 80)
    logger.info(f"📝 SAMPLE FEATURES (showing {min(num_samples, len(features))} samples)")
    logger.info("=" * 80)

    for i, feature in enumerate(features[:num_samples]):
        logger.info(f"\n📄 Sample {i + 1}:")
        logger.info(f"  Example ID: {feature.get('example_id', 'N/A')}")
        logger.info(f"  Segment index: {feature.get('segment_index', 'N/A')}")
        logger.info(f"  Total segments: {feature.get('total_segments', 'N/A')}")

        # Input details
        if "input_ids" in feature:
            input_ids = feature["input_ids"]
            logger.info(f"  Input IDs: list of {len(input_ids)} tokens")

            # Check for memory tokens (IDs > 32000)
            if any(id > 32000 for id in input_ids):
                mem_token_count = sum(1 for id in input_ids if id > 32000)
                logger.info(f"    ⚠️ Contains {mem_token_count} memory tokens (ID > 32000)")

        # Token type IDs
        if "token_type_ids" in feature:
            token_type_ids = feature["token_type_ids"]
            context_tokens = sum(1 for t in token_type_ids if t == 1)
            question_tokens = sum(1 for t in token_type_ids if t == 0)
            logger.info(f"  Token types: {context_tokens} context, {question_tokens} question/special")

        # Question and context
        if "question" in feature:
            question = feature["question"]
            logger.info(f"  Question: '{question[:80]}{'...' if len(question) > 80 else ''}'")

        if "context" in feature:
            context = feature["context"]
            logger.info(f"  Context: '{context[:80]}{'...' if len(context) > 80 else ''}'")

        # Answer info
        if "has_answer" in feature:
            has_answer = feature["has_answer"]
            logger.info(f"  Has answer: {has_answer}")

        if "chosen_answer_text" in feature:
            answer = feature["chosen_answer_text"]
            logger.info(f"  Answer: '{answer}'")

        # Position info
        if "cls_index" in feature:
            logger.info(f"  CLS index: {feature['cls_index']}")

        if "start_positions" in feature and "end_positions" in feature:
            logger.info(f"  Answer positions: [{feature['start_positions']}, {feature['end_positions']}]")

        # Offset mapping
        if "offset_mapping" in feature:
            offset_mapping = feature["offset_mapping"]
            logger.info(f"  Offset mapping: {len(offset_mapping)} entries")
            # Check for (0,0) offsets
            zero_offsets = sum(1 for o in offset_mapping if o == (0, 0))
            logger.info(f"    (0,0) offsets: {zero_offsets} (question/special tokens)")

    logger.info("")


def main():
    parser = argparse.ArgumentParser(description="Inspect preprocessed SQuAD v2 data")
    parser.add_argument(
        "--data-dir", type=str, default="./preprocessed_data/squad_v2", help="Preprocessed data directory"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "validation"], help="Which split to inspect"
    )
    parser.add_argument("--show-samples", type=int, default=2, help="Number of sample features to show in detail")
    parser.add_argument("--max-chunks", type=int, default=3, help="Maximum number of chunks to inspect")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    logger.info("\n" + "=" * 80)
    logger.info("🔍 PREPROCESSED DATA INSPECTOR")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Split to inspect: {args.split}")
    logger.info("")

    # Step 1: Check directory structure
    if not inspect_directory_structure(data_dir):
        return 1

    # Step 2: Check manifests
    inspect_manifest(data_dir)

    # Step 3: Inspect cached chunks
    sample_features = inspect_cached_chunks(data_dir, args.split, args.max_chunks)

    # Step 4: Inspect feature structure
    if sample_features:
        inspect_feature_structure(sample_features)
        inspect_sample_features(sample_features, args.show_samples)

    # Final summary
    logger.info("=" * 80)
    logger.info("✅ INSPECTION COMPLETE")
    logger.info("=" * 80)

    if not sample_features:
        logger.info("")
        logger.info("⚠️ No cached data found!")
        logger.info("")
        logger.info("To create preprocessed data, run:")
        logger.info("  uv run python scripts/preprocess_squad_v2.py --memory-tokens 0 \\")
        logger.info("      --max-train-samples 1000 --max-eval-samples 200")
        logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
