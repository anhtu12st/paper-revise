#!/usr/bin/env python3
"""
Memory Attention Analysis Example
==================================

This script demonstrates how to use HopTracker and MemoryVisualizer to analyze
differentiable memory attention patterns and multi-hop reasoning.

Features demonstrated:
- Loading a trained differentiable memory model
- Running inference on a multi-segment document
- Tracking reasoning hops with HopTracker
- Visualizing attention patterns with MemoryVisualizer
- Generating comprehensive analysis reports

USAGE:
    python examples/analyze_memory_attention.py

Prerequisites:
    - A trained MemXLNet model with differentiable memory
    - Default model path: ./outputs/validation-diff-memory/best_model
    - Override with --model-path argument
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import XLNetTokenizerFast

from memxlnet.models import MemXLNetForQA
from memxlnet.utils import HopTracker, MemoryVisualizer, extract_simple_entities


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze memory attention patterns in MemXLNet")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./outputs/validation-diff-memory/best_model",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./analysis_output", help="Directory to save analysis results"
    )
    return parser.parse_args()


def create_multi_segment_example():
    """
    Create a synthetic multi-segment document for analysis.

    Returns:
        Tuple of (question, segments, answer, answer_segment)
    """
    question = "Who is the president of France?"

    # Multiple segments that require multi-hop reasoning
    segments = [
        "Emmanuel Macron was born in Amiens, France on December 21, 1977.",
        "In 2017, Emmanuel Macron won the French presidential election.",
        "As president, Macron implemented several economic reforms.",
        "France is a country in Western Europe with Paris as its capital.",
    ]

    answer = "Emmanuel Macron"
    answer_segment = 1  # Answer found in segment 1

    return question, segments, answer, answer_segment


def analyze_model_outputs(
    model,
    tokenizer,
    question,
    segments,
    answer,
    answer_segment,
    device,
    output_dir,
):
    """
    Analyze model outputs with HopTracker and MemoryVisualizer.

    Args:
        model: MemXLNet model
        tokenizer: XLNet tokenizer
        question: Question text
        segments: List of document segments
        answer: Answer text
        answer_segment: Segment containing answer
        device: Device (cuda/cpu)
        output_dir: Output directory for visualizations
    """
    print("\n" + "=" * 80)
    print("MEMORY ATTENTION ANALYSIS")
    print("=" * 80)
    print()

    # Initialize trackers
    hop_tracker = HopTracker(min_attention_threshold=0.1)
    visualizer = MemoryVisualizer(output_dir=output_dir)

    # Mark answer for hop tracking
    hop_tracker.mark_answer(answer, answer_segment)

    # Store memory evolution for animation
    segment_data = []

    # Process each segment
    print(f"📄 Processing {len(segments)} segments...")
    memory_state = None

    for seg_idx, segment_text in enumerate(segments):
        print(f"  Segment {seg_idx}: {segment_text[:60]}...")

        # Tokenize
        inputs = tokenizer(question, segment_text, return_tensors="pt", max_length=256, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Add memory state if available
        if memory_state is not None:
            inputs["memory_state"] = memory_state

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract memory information
        if "memory_info" in outputs:
            memory_info = outputs["memory_info"]

            # Get attention weights (move to CPU and convert to numpy)
            read_weights = memory_info["read_weights"].cpu().numpy()
            write_weights = memory_info["write_weights"].cpu().numpy()

            # Handle batch dimension
            if read_weights.ndim == 3:
                read_weights = read_weights[0]  # (heads, slots)
                write_weights = write_weights[0]

            # Get usage if available
            usage = None
            if "usage" in memory_info:
                usage = memory_info["usage"].cpu().numpy()
                if usage.ndim == 2:
                    usage = usage[0]

            # Average attention across heads for tracking
            avg_attention = read_weights.mean(axis=0)

            # Extract entities from segment
            entities = extract_simple_entities(segment_text)

            # Track this segment
            hop_tracker.track_segment(
                segment_idx=seg_idx, attention_weights=avg_attention, entities=entities, segment_text=segment_text
            )

            # Store for visualization
            segment_data.append(
                {
                    "read_weights": read_weights,
                    "write_weights": write_weights,
                    "usage": usage if usage is not None else np.zeros(read_weights.shape[1]),
                }
            )

            # Update memory state for next segment
            memory_state = outputs["new_memory_state"]

        else:
            print("    ⚠️  No memory info available (token-based memory?)")

    print(f"✓ Processed {len(segments)} segments\n")

    # Analyze reasoning hops
    print("=" * 80)
    print("HOP ANALYSIS")
    print("=" * 80)
    print()

    # Detect bridge entities
    bridge_entities = hop_tracker.detect_bridge_entities(min_segments=2)
    print(f"🔗 Detected {len(bridge_entities)} bridge entities:")
    for entity in bridge_entities[:5]:  # Show top 5
        segments_str = ", ".join(str(s) for s in entity.segments)
        answer_mark = " ⭐ (ANSWER)" if entity.is_answer else ""
        print(f"   • {entity.text}: segments [{segments_str}], avg attention: {entity.avg_attention:.3f}{answer_mark}")
    print()

    # Detect reasoning hops
    hops = hop_tracker.detect_hops()
    print(f"🎯 Detected {len(hops)} reasoning hops:")
    for i, hop in enumerate(hops[:10]):  # Show top 10
        print(
            f"   {i + 1}. Segment {hop.from_segment} → {hop.to_segment} "
            f"via '{hop.bridging_entity}' (confidence: {hop.confidence:.3f})"
        )
    print()

    # Get hop sequence to answer
    hop_sequence = hop_tracker.get_hop_sequence(to_answer=True)
    print(f"📍 Hop sequence to answer (segment {answer_segment}):")
    if hop_sequence:
        for i, hop in enumerate(hop_sequence):
            print(f"   {i + 1}. Segment {hop.from_segment} → {hop.to_segment} via '{hop.bridging_entity}'")
    else:
        print("   No clear hop path detected")
    print()

    # Statistics
    stats = hop_tracker.get_statistics()
    print("📊 Statistics:")
    print(f"   • Total segments: {stats['num_segments']}")
    print(f"   • Unique entities: {stats['num_entities']}")
    print(f"   • Bridge entities: {stats['num_bridge_entities']}")
    print(f"   • Total hops: {stats['num_hops']}")
    print(f"   • Avg hop length: {stats['avg_hop_length']:.2f}")
    print()

    # Save hop analysis
    analysis_file = Path(output_dir) / "hop_analysis.json"
    hop_tracker.export_analysis(str(analysis_file))
    print(f"💾 Hop analysis saved to: {analysis_file}\n")

    # Create visualizations
    print("=" * 80)
    print("VISUALIZATIONS")
    print("=" * 80)
    print()

    print("Creating attention visualizations...")

    # Per-segment visualizations
    for seg_idx, data in enumerate(segment_data):
        # Read attention
        visualizer.plot_attention_heatmap(
            data["read_weights"],
            title=f"Read Attention - Segment {seg_idx}",
            save_path=f"segment_{seg_idx}_read_attention.png",
        )

        # Write attention
        visualizer.plot_attention_heatmap(
            data["write_weights"],
            title=f"Write Attention - Segment {seg_idx}",
            save_path=f"segment_{seg_idx}_write_attention.png",
        )

    print(f"✓ Created {len(segment_data)} segment visualizations")

    # Multi-head comparison for last segment
    last_segment = segment_data[-1]
    visualizer.plot_multi_head_comparison(
        last_segment["read_weights"], save_path="final_multi_head_read.png", operation="Read"
    )
    visualizer.plot_multi_head_comparison(
        last_segment["write_weights"], save_path="final_multi_head_write.png", operation="Write"
    )
    print("✓ Created multi-head comparison plots")

    # Attention distribution
    visualizer.plot_attention_distribution(
        last_segment["read_weights"], last_segment["write_weights"], save_path="attention_distribution.png"
    )
    print("✓ Created attention distribution plot")

    # Usage timeline
    usage_history = np.array([data["usage"] for data in segment_data])
    visualizer.plot_usage_timeline(usage_history, save_path="usage_timeline.png")
    print("✓ Created usage timeline")

    # Comprehensive summary
    memory_data = {
        "read_weights": [data["read_weights"] for data in segment_data],
        "write_weights": [data["write_weights"] for data in segment_data],
        "usage": usage_history,
    }
    visualizer.create_summary_report(memory_data, save_path="summary")
    print("✓ Created summary report")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📁 All results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  • hop_analysis.json - Reasoning hop data")
    print("  • segment_*_*.png - Per-segment attention heatmaps")
    print("  • final_multi_head_*.png - Multi-head comparisons")
    print("  • attention_distribution.png - Distribution analysis")
    print("  • usage_timeline.png - Memory usage over time")
    print("  • summary_*.png - Comprehensive summary plots")
    print()


def main():
    """Main analysis entry point."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("MEMORY ATTENTION ANALYSIS EXAMPLE")
    print("=" * 80)
    print()

    # Check model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("\nTo use this script:")
        print("1. Train a model with differentiable memory:")
        print("   python scripts/validate_differentiable_memory.py")
        print("2. Run this analysis script:")
        print("   python examples/analyze_memory_attention.py")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"📥 Loading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    model = MemXLNetForQA.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    tokenizer = XLNetTokenizerFast.from_pretrained(str(model_path))

    # Verify differentiable memory
    if not model.use_differentiable_memory:
        print("\n⚠️  Warning: Model does not have differentiable memory enabled!")
        print("This script is designed for differentiable memory models.")
        print("Results may be limited.\n")

    print(f"✓ Model loaded: {model.mem_token_count} memory tokens")
    if model.use_differentiable_memory:
        print("  • Memory type: Differentiable")
        print(f"  • Memory heads: {model.num_memory_heads}")
        print(f"  • Memory slots: {model.memory_slots}")
    else:
        print("  • Memory type: Token-based")

    # Create example data
    print("\n📝 Creating multi-segment example...")
    question, segments, answer, answer_segment = create_multi_segment_example()

    print(f"  Question: {question}")
    print(f"  Answer: {answer} (in segment {answer_segment})")
    print(f"  Segments: {len(segments)}")

    # Run analysis
    analyze_model_outputs(
        model=model,
        tokenizer=tokenizer,
        question=question,
        segments=segments,
        answer=answer,
        answer_segment=answer_segment,
        device=device,
        output_dir=str(output_dir),
    )

    print("\n✨ Analysis complete!")
    print(f"View results in: {output_dir}")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
