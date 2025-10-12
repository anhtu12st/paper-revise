#!/usr/bin/env python3
"""
Preprocess Datasets to Chunked Arrow Format
============================================

Preprocesses SQuAD v2 or Long SQuAD v2 datasets into small, GitHub-friendly chunks:
- Chunks of ~1000 examples each (~50MB Arrow files)
- Manifest file for fast metadata lookup
- Streaming-compatible format
- Reusable across all experiments

Usage:
    # Preprocess Standard SQuAD v2
    python scripts/preprocess_datasets_chunked.py --dataset squad_v2

    # Preprocess Long SQuAD v2
    python scripts/preprocess_datasets_chunked.py --dataset huutuan/long_squad_v2

    # Custom settings
    python scripts/preprocess_datasets_chunked.py \\
        --dataset squad_v2 \\
        --output-dir ./preprocessed_data \\
        --chunk-size 1000 \\
        --max-seq-length 384 \\
        --doc-stride 64 \\
        --memory-tokens 8

Output:
    preprocessed_data/
    └── squad_v2/  (or long_squad_v2)
        ├── manifest.json
        ├── train/
        │   ├── chunk_0000.arrow
        │   ├── chunk_0001.arrow
        │   └── ...
        └── validation/
            ├── chunk_0000.arrow
            └── ...
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import XLNetTokenizerFast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memxlnet.data.dataset import configure_memory_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess datasets to chunked Arrow format")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'squad_v2' or 'huutuan/long_squad_v2')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./preprocessed_data",
        help="Output directory for chunked data (default: ./preprocessed_data)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of examples per chunk (default: 1000, ~50MB per chunk)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=384,
        help="Maximum sequence length (default: 384)",
    )
    parser.add_argument(
        "--doc-stride",
        type=int,
        default=64,
        help="Document stride for sliding window (default: 64)",
    )
    parser.add_argument(
        "--memory-tokens",
        type=int,
        default=8,
        help="Number of memory tokens (default: 8)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Dataset splits to process (default: train validation)",
    )
    parser.add_argument(
        "--temp-cache-dir",
        type=str,
        default="./.cache_chunked_preprocessing",
        help="Temporary cache directory for processing (default: ./.cache_chunked_preprocessing)",
    )

    return parser.parse_args()


def create_manifest(output_dir: Path, dataset_name: str, split: str, config: dict, chunks_info: list) -> dict:
    """Create manifest file with dataset metadata.

    Args:
        output_dir: Output directory
        dataset_name: Dataset name
        split: Dataset split (train/validation)
        config: Preprocessing configuration
        chunks_info: List of chunk metadata

    Returns:
        Manifest dictionary
    """
    manifest = {
        "dataset_name": dataset_name,
        "split": split,
        "total_examples": sum(chunk["num_documents"] for chunk in chunks_info),
        "total_chunks": len(chunks_info),
        "examples_per_chunk": config["chunk_size"],
        "config": {
            "max_seq_length": config["max_seq_length"],
            "doc_stride": config["doc_stride"],
            "memory_num_tokens": config["memory_tokens"],
        },
        "chunks": chunks_info,
    }

    manifest_path = output_dir / f"{split}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"✅ Created manifest: {manifest_path}")
    return manifest


def preprocess_split(
    dataset_name: str,
    split: str,
    output_dir: Path,
    chunk_size: int,
    max_seq_length: int,
    doc_stride: int,
    memory_tokens: int,
    tokenizer: XLNetTokenizerFast,
    temp_cache_dir: str,
):
    """Preprocess a single dataset split into chunks.

    Args:
        dataset_name: Dataset name
        split: Split name (train/validation)
        output_dir: Output directory for chunks
        chunk_size: Examples per chunk
        max_seq_length: Maximum sequence length
        doc_stride: Document stride
        memory_tokens: Number of memory tokens
        tokenizer: XLNet tokenizer
        temp_cache_dir: Temporary cache directory
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Processing {dataset_name} - {split}")
    logger.info(f"{'=' * 80}\n")

    # Create split directory
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Process dataset directly with SquadLikeQADataset (single-pass processing)
    logger.info("📊 Processing dataset directly...")
    from memxlnet.data.dataset import SquadLikeQADataset

    # SquadLikeQADataset loads and processes the raw dataset internally
    logger.info("🔄 Converting to segmented format...")
    full_dataset = SquadLikeQADataset(
        split=split,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_examples=None,  # Process all examples
        dataset_name=dataset_name,
        max_n_segs=None,  # Load all segments
    )

    logger.info(f"✅ Processed {len(full_dataset.features)} features from {len(full_dataset.document_map)} documents")

    # Split into chunks and save as Arrow files
    # SquadLikeQADataset has features (segments) and document_map
    documents = list(full_dataset.document_map.keys())
    logger.info(f"📦 Splitting {len(documents)} documents into chunks of {chunk_size}...")

    chunks_info = []
    chunk_idx = 0
    document_idx = 0

    with tqdm(total=len(documents), desc=f"Creating chunks for {split}") as pbar:
        while document_idx < len(documents):
            # Get chunk of documents
            end_idx = min(document_idx + chunk_size, len(documents))
            chunk_doc_ids = documents[document_idx:end_idx]

            # Save chunk as Arrow file
            chunk_filename = f"chunk_{chunk_idx:04d}.arrow"
            chunk_path = split_dir / chunk_filename

            # Convert to HuggingFace dataset and save
            from datasets import Dataset as HFDataset

            # Flatten document segments for Arrow format
            chunk_data = {
                "document_id": [],
                "segment_id": [],
                "example_id": [],
                "input_ids": [],
                "attention_mask": [],
                "token_type_ids": [],
                "offset_mapping": [],
                "context": [],
                "start_positions": [],
                "end_positions": [],
                # Metadata fields for evaluation (CRITICAL: includes cls_index for XLNet)
                "question": [],
                "cls_index": [],
                "has_answer": [],
                "chosen_answer_text": [],
                "chosen_answer_char_span": [],
            }

            # Process each document in this chunk
            for local_doc_idx, doc_id in enumerate(chunk_doc_ids):
                segment_indices = full_dataset.document_map[doc_id]

                for seg_idx, feature_idx in enumerate(segment_indices):
                    feature = full_dataset.features[feature_idx]

                    chunk_data["document_id"].append(document_idx + local_doc_idx)
                    chunk_data["segment_id"].append(seg_idx)
                    chunk_data["example_id"].append(feature.get("example_id", doc_id))
                    chunk_data["input_ids"].append(feature["input_ids"])
                    chunk_data["attention_mask"].append(feature["attention_mask"])
                    chunk_data["token_type_ids"].append(feature["token_type_ids"])
                    chunk_data["offset_mapping"].append(feature["offset_mapping"])
                    chunk_data["context"].append(feature["context"])
                    chunk_data["start_positions"].append(feature["start_positions"])
                    chunk_data["end_positions"].append(feature["end_positions"])
                    # Add metadata fields (with defaults if missing)
                    chunk_data["question"].append(feature.get("question", ""))
                    chunk_data["cls_index"].append(feature.get("cls_index", 0))
                    chunk_data["has_answer"].append(feature.get("has_answer", False))
                    chunk_data["chosen_answer_text"].append(feature.get("chosen_answer_text", ""))
                    chunk_data["chosen_answer_char_span"].append(feature.get("chosen_answer_char_span", [-1, -1]))

            # Create HuggingFace dataset and save
            hf_dataset = HFDataset.from_dict(chunk_data)

            # Save as Arrow (memory-mapped, fast loading)
            hf_dataset.save_to_disk(str(chunk_path))

            # Record chunk info
            chunks_info.append(
                {
                    "chunk_id": chunk_idx,
                    "path": f"{split}/{chunk_filename}",
                    "num_documents": end_idx - document_idx,
                    "num_segments": len(chunk_data["segment_id"]),
                    "document_range": [document_idx, end_idx],
                }
            )

            logger.info(
                f"✅ Saved chunk {chunk_idx}: {end_idx - document_idx} documents, "
                f"{len(chunk_data['segment_id'])} segments → {chunk_path}"
            )

            # Update indices
            chunk_idx += 1
            document_idx = end_idx
            pbar.update(end_idx - document_idx)

    # Create manifest
    config = {
        "chunk_size": chunk_size,
        "max_seq_length": max_seq_length,
        "doc_stride": doc_stride,
        "memory_tokens": memory_tokens,
    }
    manifest = create_manifest(output_dir, dataset_name, split, config, chunks_info)

    logger.info(f"\n✅ {split} preprocessing complete!")
    logger.info(f"   Total documents: {manifest['total_examples']}")
    logger.info(f"   Total chunks: {manifest['total_chunks']}")
    logger.info(f"   Chunks directory: {split_dir}")


def main():
    """Main preprocessing function."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("📦 CHUNKED DATASET PREPROCESSING")
    print("=" * 80 + "\n")

    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Chunk size: {args.chunk_size} documents per chunk")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Document stride: {args.doc_stride}")
    print(f"Memory tokens: {args.memory_tokens}")
    print(f"Splits: {', '.join(args.splits)}")
    print()

    # Determine output subdirectory name
    dataset_key = args.dataset.replace("/", "_").replace("-", "_")
    output_dir = Path(args.output_dir) / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Output directory: {output_dir}")

    # Initialize tokenizer
    logger.info("🔧 Initializing tokenizer...")
    tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")

    # Configure memory tokens if needed
    if args.memory_tokens > 0:
        logger.info(f"🧠 Configuring {args.memory_tokens} memory tokens...")
        mem_token_info = configure_memory_tokens(tokenizer, args.memory_tokens)
        if mem_token_info:
            logger.info(
                f"✅ Added memory tokens: R={len(mem_token_info['mem_read_ids'])}, "
                f"W={len(mem_token_info['mem_write_ids'])}"
            )

    # Process each split
    for split in args.splits:
        preprocess_split(
            dataset_name=args.dataset,
            split=split,
            output_dir=output_dir,
            chunk_size=args.chunk_size,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            memory_tokens=args.memory_tokens,
            tokenizer=tokenizer,
            temp_cache_dir=args.temp_cache_dir,
        )

    # Create main manifest
    main_manifest = {
        "dataset_name": args.dataset,
        "preprocessing_config": {
            "chunk_size": args.chunk_size,
            "max_seq_length": args.max_seq_length,
            "doc_stride": args.doc_stride,
            "memory_num_tokens": args.memory_tokens,
        },
        "splits": args.splits,
        "output_dir": str(output_dir),
    }

    main_manifest_path = output_dir / "manifest.json"
    with open(main_manifest_path, "w") as f:
        json.dump(main_manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 80 + "\n")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Main manifest: {main_manifest_path}")
    print(f"📦 Splits processed: {', '.join(args.splits)}")
    print("\nNext steps:")
    print(f"  1. Check chunks: ls -lh {output_dir}/train/")
    print(f"  2. View manifest: cat {main_manifest_path}")
    print("  3. Use in experiments: set use_chunked_dataset=True in TrainingConfig")
    print()


if __name__ == "__main__":
    main()
