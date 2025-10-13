#!/usr/bin/env python3
"""
Answer Coverage Analysis Script
================================

This script analyzes how well the preprocessed dataset covers all valid answers
in SQuAD v2, particularly for validation examples with multiple valid answers.

It checks:
- How many examples have multiple valid answers
- How many of those answers are successfully mapped to segments
- Which answers are missed due to segmentation

Usage:
    python scripts/analyze_answer_coverage.py
    python scripts/analyze_answer_coverage.py --max-examples 100
    python scripts/analyze_answer_coverage.py --split validation
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from memxlnet.data.dataset import SquadLikeQADataset
from transformers import XLNetTokenizerFast


def analyze_answer_coverage(split: str = "validation", max_examples: int | None = None):
    """Analyze answer coverage in preprocessed dataset."""
    print("=" * 80)
    print("📊 ANSWER COVERAGE ANALYSIS")
    print("=" * 80)
    print()

    # Load raw SQuAD v2 dataset
    print(f"Loading SQuAD v2 {split} split...")
    raw_dataset = load_dataset("squad_v2", split=split)
    if max_examples:
        raw_dataset = raw_dataset.select(range(min(max_examples, len(raw_dataset))))
    print(f"✓ Loaded {len(raw_dataset)} examples")
    print()

    # Create preprocessed dataset
    print("Creating preprocessed dataset...")
    tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
    processed_dataset = SquadLikeQADataset(
        split=split,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=64,
        max_examples=max_examples,
        dataset_name="squad_v2",
    )
    print(f"✓ Created {len(processed_dataset)} features from {len(raw_dataset)} examples")
    print()

    # Analysis statistics
    stats = {
        "total_examples": len(raw_dataset),
        "examples_with_multiple_answers": 0,
        "total_answer_variants": 0,
        "examples_with_no_answers": 0,
        "features_with_answer_mapped": 0,
        "features_with_no_answer_mapped": 0,
        "all_answers_covered": 0,
        "partial_answers_covered": 0,
        "no_answers_covered": 0,
    }

    # Track which answers are covered per example
    coverage_by_example = {}

    # Analyze each example
    print("Analyzing answer coverage...")
    for example_idx, example in enumerate(raw_dataset):
        example_id = f"doc_{example_idx}"
        answers = example["answers"]["text"]

        # Get all segments for this example
        segment_indices = processed_dataset.get_document_segments(example_id)

        if not answers:
            # No-answer example
            stats["examples_with_no_answers"] += 1
            coverage_by_example[example_id] = {
                "num_answers": 0,
                "answers_covered": 0,
                "coverage_rate": 1.0,  # No answer is "covered" trivially
            }
            continue

        # Track which answers are covered
        answers_covered = set()
        stats["total_answer_variants"] += len(answers)

        if len(answers) > 1:
            stats["examples_with_multiple_answers"] += 1

        # Check each segment
        for seg_idx in segment_indices:
            feature = processed_dataset[seg_idx]
            chosen_answer = feature.get("chosen_answer_text", "")

            # Check if chosen answer matches any valid answer
            if chosen_answer:
                for idx, valid_answer in enumerate(answers):
                    if chosen_answer.strip().lower() == valid_answer.strip().lower():
                        answers_covered.add(idx)
                        stats["features_with_answer_mapped"] += 1
                        break
            else:
                stats["features_with_no_answer_mapped"] += 1

        # Calculate coverage for this example
        coverage_rate = len(answers_covered) / len(answers) if answers else 0.0
        coverage_by_example[example_id] = {
            "num_answers": len(answers),
            "answers_covered": len(answers_covered),
            "coverage_rate": coverage_rate,
            "all_answers": answers,
            "covered_indices": list(answers_covered),
        }

        # Categorize coverage
        if coverage_rate == 1.0:
            stats["all_answers_covered"] += 1
        elif coverage_rate > 0.0:
            stats["partial_answers_covered"] += 1
        else:
            stats["no_answers_covered"] += 1

    # Print summary statistics
    print()
    print("=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total examples analyzed: {stats['total_examples']}")
    print(f"Examples with no answer: {stats['examples_with_no_answers']}")
    print(
        f"Examples with multiple answers: {stats['examples_with_multiple_answers']} "
        f"({stats['examples_with_multiple_answers'] / stats['total_examples'] * 100:.1f}%)"
    )
    print(f"Total answer variants: {stats['total_answer_variants']}")
    print()
    print("Coverage Statistics:")
    answerable_examples = stats["total_examples"] - stats["examples_with_no_answers"]
    if answerable_examples > 0:
        print(
            f"  All answers covered: {stats['all_answers_covered']} "
            f"({stats['all_answers_covered'] / answerable_examples * 100:.1f}%)"
        )
        print(
            f"  Partial coverage: {stats['partial_answers_covered']} "
            f"({stats['partial_answers_covered'] / answerable_examples * 100:.1f}%)"
        )
        print(
            f"  No coverage: {stats['no_answers_covered']} "
            f"({stats['no_answers_covered'] / answerable_examples * 100:.1f}%)"
        )
    print()
    print("Feature Statistics:")
    total_features = stats["features_with_answer_mapped"] + stats["features_with_no_answer_mapped"]
    if total_features > 0:
        print(
            f"  Features with answer mapped: {stats['features_with_answer_mapped']} "
            f"({stats['features_with_answer_mapped'] / total_features * 100:.1f}%)"
        )
        print(
            f"  Features with no answer: {stats['features_with_no_answer_mapped']} "
            f"({stats['features_with_no_answer_mapped'] / total_features * 100:.1f}%)"
        )
    print()

    # Show examples with partial or no coverage
    print("=" * 80)
    print("⚠️  EXAMPLES WITH INCOMPLETE COVERAGE")
    print("=" * 80)

    incomplete_examples = [
        (ex_id, info)
        for ex_id, info in coverage_by_example.items()
        if info["num_answers"] > 0 and info["coverage_rate"] < 1.0
    ]

    if incomplete_examples:
        # Sort by number of uncovered answers
        incomplete_examples.sort(key=lambda x: x[1]["num_answers"] - x[1]["answers_covered"], reverse=True)

        print(f"Found {len(incomplete_examples)} examples with incomplete coverage")
        print()
        print("Top 10 examples with most uncovered answers:")
        for i, (ex_id, info) in enumerate(incomplete_examples[:10]):
            example_idx = int(ex_id.split("_")[1])
            print(f"\n{i+1}. Example {example_idx} ({ex_id}):")
            print(f"   Total answers: {info['num_answers']}")
            print(f"   Covered: {info['answers_covered']} ({info['coverage_rate']*100:.0f}%)")
            print(f"   All answers: {info['all_answers']}")
            print(f"   Covered indices: {info['covered_indices']}")

            # Show which answers are missing
            uncovered = [
                f"#{idx}: '{info['all_answers'][idx]}'"
                for idx in range(info["num_answers"])
                if idx not in info["covered_indices"]
            ]
            if uncovered:
                print(f"   ❌ Uncovered: {', '.join(uncovered)}")
    else:
        print("✅ All examples have complete coverage!")

    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)

    return stats, coverage_by_example


def main():
    parser = argparse.ArgumentParser(description="Analyze answer coverage in preprocessed SQuAD v2 data")
    parser.add_argument(
        "--split", type=str, default="validation", choices=["train", "validation"], help="Dataset split to analyze"
    )
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum examples to analyze (for testing)")

    args = parser.parse_args()

    stats, coverage_by_example = analyze_answer_coverage(split=args.split, max_examples=args.max_examples)

    return 0


if __name__ == "__main__":
    sys.exit(main())
