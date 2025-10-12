#!/usr/bin/env python3
"""
Inspect Chunked Datasets
=========================

Inspect metadata and contents of preprocessed chunked datasets.
Useful for verifying preprocessing and understanding dataset structure.

Usage:
    # Inspect main manifest
    python scripts/inspect_chunks.py ./preprocessed_data/squad_v2

    # Inspect specific split
    python scripts/inspect_chunks.py ./preprocessed_data/squad_v2 --split train

    # Show detailed statistics
    python scripts/inspect_chunks.py ./preprocessed_data/squad_v2 --verbose

    # Load and inspect first chunk
    python scripts/inspect_chunks.py ./preprocessed_data/squad_v2 --load-chunk 0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def format_size(bytes_size):
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"


def inspect_manifest(manifest_path, verbose=False):
    """Inspect and display manifest information."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    print("\n" + "=" * 80)
    print("📋 MANIFEST INFORMATION")
    print("=" * 80 + "\n")

    # Check if main manifest or split manifest
    if "splits" in manifest:
        # Main manifest
        print("📊 Main Manifest")
        print(f"   Dataset: {manifest['dataset_name']}")
        print(f"   Splits: {', '.join(manifest['splits'])}")
        print(f"   Output directory: {manifest['output_dir']}")
        print("\n⚙️  Preprocessing Configuration:")
        config = manifest["preprocessing_config"]
        print(f"   • Max sequence length: {config['max_seq_length']}")
        print(f"   • Doc stride: {config['doc_stride']}")
        print(f"   • Memory tokens: {config['memory_num_tokens']}")
        print(f"   • Examples per chunk: {config['chunk_size']}")
    else:
        # Split manifest
        print(f"📊 Split: {manifest['split']}")
        print(f"   Dataset: {manifest['dataset_name']}")
        print(f"   Total examples: {manifest['total_examples']:,}")
        print(f"   Total chunks: {manifest['total_chunks']}")
        print(f"   Examples per chunk: {manifest['examples_per_chunk']}")

        print("\n⚙️  Configuration:")
        config = manifest["config"]
        print(f"   • Max sequence length: {config['max_seq_length']}")
        print(f"   • Doc stride: {config['doc_stride']}")
        print(f"   • Memory tokens: {config['memory_num_tokens']}")

        print("\n📦 Chunks:")
        if verbose:
            # Detailed chunk information
            for chunk in manifest["chunks"]:
                print(f"\n   Chunk {chunk['chunk_id']}:")
                print(f"      Path: {chunk['path']}")
                print(f"      Documents: {chunk['num_documents']:,}")
                print(f"      Segments: {chunk['num_segments']:,}")
                print(f"      Doc range: {chunk['document_range'][0]:,} - {chunk['document_range'][1]:,}")

                # Try to get file size
                manifest_dir = Path(manifest_path).parent
                chunk_path = manifest_dir / chunk["path"]
                if chunk_path.exists():
                    # Get directory size
                    total_size = sum(f.stat().st_size for f in chunk_path.rglob("*") if f.is_file())
                    print(f"      Size: {format_size(total_size)}")
        else:
            # Summary
            print(f"   Total: {manifest['total_chunks']} chunks")
            print(f"   Documents: {manifest['total_examples']:,} total")

            # Calculate average segments per document
            total_segments = sum(chunk["num_segments"] for chunk in manifest["chunks"])
            avg_segments = total_segments / manifest["total_examples"]
            print(f"   Segments: {total_segments:,} total ({avg_segments:.2f} avg per document)")

            # Estimate total size
            manifest_dir = Path(manifest_path).parent
            first_chunk_path = manifest_dir / manifest["chunks"][0]["path"]
            if first_chunk_path.exists():
                first_chunk_size = sum(f.stat().st_size for f in first_chunk_path.rglob("*") if f.is_file())
                estimated_total = first_chunk_size * manifest["total_chunks"]
                print(f"   Estimated size: ~{format_size(estimated_total)}")
                print(f"   (Based on first chunk: {format_size(first_chunk_size)})")


def load_and_inspect_chunk(manifest_path, chunk_idx):
    """Load and inspect a specific chunk."""
    from memxlnet.data import load_chunked_dataset

    manifest_dir = Path(manifest_path).parent
    split_name = Path(manifest_path).stem.replace("_manifest", "")

    print("\n" + "=" * 80)
    print(f"📦 LOADING CHUNK {chunk_idx}")
    print("=" * 80 + "\n")

    # Load the specific chunk
    dataset = load_chunked_dataset(
        dataset_dir=str(manifest_dir), split=split_name, mode="chunks", chunk_indices=[chunk_idx]
    )

    print(f"✅ Loaded {len(dataset)} documents from chunk {chunk_idx}")

    if len(dataset) > 0:
        # Inspect first document
        print("\n📄 First Document Sample:")
        first_doc = dataset[0]
        print(f"   Number of segments: {len(first_doc)}")

        if len(first_doc) > 0:
            first_segment = first_doc[0]
            print("\n   First Segment:")
            print(f"      Example ID: {first_segment.get('example_id', 'N/A')}")
            print(f"      Input length: {len(first_segment['input_ids'])} tokens")
            print(f"      Context preview: {first_segment['context'][:100]}...")
            print(f"      Has answer: {first_segment['start_positions'] != first_segment['end_positions']}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect chunked dataset metadata")

    parser.add_argument("dataset_dir", type=str, help="Path to chunked dataset directory")
    parser.add_argument("--split", type=str, help="Specific split to inspect (train/validation)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    parser.add_argument("--load-chunk", type=int, metavar="N", help="Load and inspect a specific chunk by index")

    return parser.parse_args()


def main():
    """Main inspection function."""
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        print(f"❌ Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    print("\n🔍 Inspecting chunked dataset...")
    print(f"📁 Directory: {dataset_dir}")

    # Determine which manifest to inspect
    if args.split:
        # Specific split
        manifest_path = dataset_dir / f"{args.split}_manifest.json"
        if not manifest_path.exists():
            print(f"❌ Error: Split manifest not found: {manifest_path}")
            sys.exit(1)
    else:
        # Main manifest
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.exists():
            # Try to find any split manifest
            split_manifests = list(dataset_dir.glob("*_manifest.json"))
            if split_manifests:
                print(f"⚠️  Main manifest not found, using: {split_manifests[0].name}")
                manifest_path = split_manifests[0]
            else:
                print(f"❌ Error: No manifest files found in {dataset_dir}")
                sys.exit(1)

    # Inspect manifest
    inspect_manifest(manifest_path, verbose=args.verbose)

    # Load and inspect chunk if requested
    if args.load_chunk is not None:
        load_and_inspect_chunk(manifest_path, args.load_chunk)

    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    main()
