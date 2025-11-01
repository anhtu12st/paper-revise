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
        default=0,
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
    """Preprocess a single dataset split into chunks with streaming to avoid OOM.

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

    # Use streaming processing to avoid loading entire dataset into memory
    logger.info("🌊 Using streaming processing to avoid OOM...")
    from datasets import load_dataset

    from memxlnet.data.dataset import SquadLikeQADataset

    # Load raw dataset info to get total count
    logger.info("📊 Loading dataset metadata...")
    raw_dataset = load_dataset(dataset_name, split=split)
    total_examples = len(raw_dataset)
    logger.info(f"📊 Total examples in {split}: {total_examples}")

    # Process in batches to avoid OOM
    processing_batch_size = 100  # Process 100 raw documents at a time
    logger.info(f"🔄 Processing in batches of {processing_batch_size} raw documents...")
    logger.info(f"📦 Will save chunks of {chunk_size} processed documents each...")

    chunks_info = []
    chunk_idx = 0
    global_doc_idx = 0  # Global document counter across all chunks

    # Accumulator for current chunk being built
    current_chunk_data = {
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
        # NEW: Segment selection metadata for smart progressive training
        "segment_index_in_doc": [],  # Position of segment within document
        "has_answer_in_segment": [],  # Whether this specific segment contains the answer
    }
    current_chunk_doc_count = 0
    chunk_start_doc_idx = 0

    # Process raw dataset in batches
    raw_idx = 0
    with tqdm(total=total_examples, desc=f"Processing {split}") as pbar:
        while raw_idx < total_examples:
            # Get batch of raw examples
            batch_end = min(raw_idx + processing_batch_size, total_examples)
            raw_batch = raw_dataset[raw_idx:batch_end]

            # Process batch manually to avoid loading entire dataset
            logger.info(f"🔄 Processing raw examples {raw_idx}-{batch_end}...")

            # Process each example in the raw batch
            batch_size_actual = len(raw_batch["id"])
            for batch_local_idx in range(batch_size_actual):
                # Extract single example from batch
                # Handle nested structure properly
                answers_data = raw_batch["answers"][batch_local_idx]

                example = {
                    "id": raw_batch["id"][batch_local_idx],
                    "question": raw_batch["question"][batch_local_idx],
                    "context": raw_batch["context"][batch_local_idx],
                    "answers": {
                        "text": answers_data["text"]
                        if isinstance(answers_data, dict)
                        else answers_data.get("text", []),
                        "answer_start": answers_data["answer_start"]
                        if isinstance(answers_data, dict)
                        else answers_data.get("answer_start", []),
                    },
                    "title": raw_batch.get("title", [""] * batch_size_actual)[batch_local_idx]
                    if "title" in raw_batch
                    else "",
                }

                # Process this example into segments using the _process_example logic
                from memxlnet.data.dataset import SquadLikeQADataset

                # Create a temporary minimal dataset object just to call _process_example
                temp_ds = SquadLikeQADataset.__new__(SquadLikeQADataset)
                processed_segments = temp_ds._process_example(
                    example=example,
                    example_idx=global_doc_idx,
                    tokenizer=tokenizer,
                    max_seq_length=max_seq_length,
                    doc_stride=doc_stride,
                    max_n_segs=None,  # Keep all segments
                )

                # Add all segments from this example to current chunk
                for seg_idx, feature in enumerate(processed_segments):
                    current_chunk_data["document_id"].append(global_doc_idx)
                    current_chunk_data["segment_id"].append(seg_idx)
                    current_chunk_data["example_id"].append(f"doc_{global_doc_idx}")
                    current_chunk_data["input_ids"].append(feature["input_ids"])
                    current_chunk_data["attention_mask"].append(feature["attention_mask"])
                    current_chunk_data["token_type_ids"].append(feature["token_type_ids"])
                    current_chunk_data["offset_mapping"].append(feature["offset_mapping"])
                    current_chunk_data["context"].append(feature["context"])
                    current_chunk_data["start_positions"].append(feature["start_positions"])
                    current_chunk_data["end_positions"].append(feature["end_positions"])
                    # Add metadata fields (with defaults if missing)
                    current_chunk_data["question"].append(feature.get("question", ""))
                    current_chunk_data["cls_index"].append(feature.get("cls_index", 0))
                    current_chunk_data["has_answer"].append(feature.get("has_answer", False))
                    current_chunk_data["chosen_answer_text"].append(feature.get("chosen_answer_text", ""))
                    current_chunk_data["chosen_answer_char_span"].append(
                        feature.get("chosen_answer_char_span", [-1, -1])
                    )
                    # Add segment selection metadata
                    current_chunk_data["segment_index_in_doc"].append(feature.get("segment_index_in_doc", seg_idx))
                    current_chunk_data["has_answer_in_segment"].append(feature.get("has_answer_in_segment", False))

                current_chunk_doc_count += 1
                global_doc_idx += 1

                # Save chunk if it reaches the chunk size
                if current_chunk_doc_count >= chunk_size:
                    _save_chunk(
                        current_chunk_data,
                        split_dir,
                        chunk_idx,
                        chunk_start_doc_idx,
                        global_doc_idx,
                        chunks_info,
                        split,
                    )

                    # Reset for next chunk
                    current_chunk_data = {k: [] for k in current_chunk_data.keys()}
                    current_chunk_doc_count = 0
                    chunk_start_doc_idx = global_doc_idx
                    chunk_idx += 1

            # Update progress
            raw_idx = batch_end
            pbar.update(batch_end - raw_idx + processing_batch_size)

    # Save remaining data as final chunk
    if current_chunk_doc_count > 0:
        _save_chunk(
            current_chunk_data,
            split_dir,
            chunk_idx,
            chunk_start_doc_idx,
            global_doc_idx,
            chunks_info,
            split,
        )

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


def _save_chunk(
    chunk_data: dict,
    split_dir: Path,
    chunk_idx: int,
    start_doc_idx: int,
    end_doc_idx: int,
    chunks_info: list,
    split: str,
) -> None:
    """Save a chunk to disk.

    Args:
        chunk_data: Dictionary containing all chunk data
        split_dir: Directory for this split
        chunk_idx: Index of this chunk
        start_doc_idx: Starting document index
        end_doc_idx: Ending document index
        chunks_info: List to append chunk metadata to
        split: Split name
    """
    from datasets import Dataset as HFDataset

    chunk_filename = f"chunk_{chunk_idx:04d}.arrow"
    chunk_path = split_dir / chunk_filename

    # Create HuggingFace dataset and save
    hf_dataset = HFDataset.from_dict(chunk_data)
    hf_dataset.save_to_disk(str(chunk_path))

    # Record chunk info
    chunks_info.append(
        {
            "chunk_id": chunk_idx,
            "path": f"{split}/{chunk_filename}",
            "num_documents": end_doc_idx - start_doc_idx,
            "num_segments": len(chunk_data["segment_id"]),
            "document_range": [start_doc_idx, end_doc_idx],
        }
    )

    logger.info(
        f"✅ Saved chunk {chunk_idx}: {end_doc_idx - start_doc_idx} documents, "
        f"{len(chunk_data['segment_id'])} segments → {chunk_path}"
    )


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
