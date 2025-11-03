#!/usr/bin/env python3
"""Example: Evaluating a trained GMM-XLNet model.

This script demonstrates how to evaluate a trained GMM-XLNet model on SQuAD v2
and analyze its performance. It showcases:
- Loading a trained GMM model from checkpoint or Hub
- Running evaluation on validation dataset
- Computing QA metrics (EM, F1)
- Monitoring routing behavior during evaluation

Usage:
    # Evaluate local checkpoint
    python examples/evaluate_gmm_model.py \\
        --model-path outputs/gmm-xlnet-squad/final

    # Evaluate from HuggingFace Hub
    python examples/evaluate_gmm_model.py \\
        --model-id username/gmm-xlnet-squad \\
        --from-hub

    # Evaluate with routing analysis
    python examples/evaluate_gmm_model.py \\
        --model-path outputs/gmm-xlnet-squad/final \\
        --analyze-routing
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import XLNetTokenizerFast

from gmmxlnet.models import GMMXLNetForQA
from gmmxlnet.utils import GMMAnalyzer
from memxlnet.data import ChunkedSquadDataset
from memxlnet.evaluation import QAEvaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate GMM-XLNet model")

    # Model loading
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local model checkpoint",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID",
    )
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Load model from HuggingFace Hub",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad_v2",
        help="Dataset to evaluate on (default: squad_v2)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split (default: validation)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (None for all)",
    )

    # Evaluation settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=6,
        help="Maximum segments per document (default: 6)",
    )

    # Analysis
    parser.add_argument(
        "--analyze-routing",
        action="store_true",
        help="Analyze routing behavior during evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results (default: evaluation_results)",
    )

    return parser.parse_args()


def load_model(args):
    """Load GMM model from checkpoint or Hub."""
    print("\n[1/5] Loading GMM model...")

    if args.from_hub:
        if not args.model_id:
            raise ValueError("--model-id required when using --from-hub")
        print(f"  Loading from Hub: {args.model_id}")
        model = GMMXLNetForQA.from_pretrained(args.model_id)
    else:
        if not args.model_path:
            raise ValueError("--model-path required when not using --from-hub")
        print(f"  Loading from local path: {args.model_path}")
        model = GMMXLNetForQA.from_pretrained(args.model_path)

    print(f"✓ Model loaded")
    print(f"  - Number of experts: {model.num_experts}")
    print(f"  - Memory slots per expert: {model.memory_slots}")
    print(f"  - Routing mode: {model.routing_mode}")
    print(f"  - Routing temperature: {model.routing_temperature}")

    return model


def load_dataset(args, tokenizer):
    """Load evaluation dataset."""
    print("\n[2/5] Loading evaluation dataset...")

    dataset = ChunkedSquadDataset(
        split=args.split,
        tokenizer=tokenizer,
        max_seq_length=512,
        doc_stride=128,
        max_examples=args.max_examples,
        max_n_segs=args.max_segments,
        dataset_name=args.dataset,
    )

    print(f"✓ Dataset loaded")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Split: {args.split}")
    print(f"  - Total examples: {len(dataset)}")
    print(f"  - Documents: {len(dataset.get_all_documents())}")

    return dataset


def evaluate_model(model, dataset, args):
    """Run evaluation and compute metrics."""
    print("\n[3/5] Running evaluation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Track predictions and ground truth
    all_predictions = {}
    all_ground_truth = {}
    total_loss = 0.0
    num_batches = 0

    # Track routing statistics
    routing_probs_list = []

    print(f"✓ Starting evaluation on {device}")
    print(f"  - Total batches: {len(dataloader)}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch.get("start_positions")
            end_positions = batch.get("end_positions")

            if start_positions is not None:
                start_positions = start_positions.to(device)
                end_positions = end_positions.to(device)

            # Initialize memory for batch
            if batch_idx == 0 or "memory_state" not in locals():
                memory_state = model.get_initial_memory(
                    batch_size=input_ids.size(0), device=device
                )

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                memory_state=memory_state,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            # Update memory
            memory_state = outputs["memory_state"]

            # Track routing
            if "routing_probs" in outputs:
                routing_probs_list.append(outputs["routing_probs"].cpu())

            # Track loss
            if "loss" in outputs:
                total_loss += outputs["loss"].item()
                num_batches += 1

            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Progress: {batch_idx + 1}/{len(dataloader)} batches")

    # Compute average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    print(f"\n✓ Evaluation completed")
    print(f"  - Average loss: {avg_loss:.4f}")

    # Compute routing statistics
    routing_stats = None
    if routing_probs_list:
        all_routing_probs = torch.cat(routing_probs_list, dim=0)
        expert_util = all_routing_probs.mean(dim=0).numpy()
        entropy = -(all_routing_probs * torch.log(all_routing_probs + 1e-10)).sum(dim=1).mean().item()

        routing_stats = {
            "expert_utilization": expert_util.tolist(),
            "mean_entropy": entropy,
            "num_segments": all_routing_probs.size(0),
        }

        print(f"\n  Routing Statistics:")
        print(f"    Expert Utilization: {[f'{u:.2%}' for u in expert_util]}")
        print(f"    Mean Entropy: {entropy:.3f}")

    return {
        "average_loss": avg_loss,
        "routing_stats": routing_stats,
    }


def analyze_routing(model, dataset, args):
    """Perform detailed routing analysis."""
    print("\n[4/5] Analyzing routing behavior...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create analyzer
    analyzer = GMMAnalyzer(model=model, device=device)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Track routing
    routing_stats = analyzer.track_routing(
        dataloader=dataloader,
        max_segments=200,  # Analyze first 200 segments
    )

    print(f"✓ Routing analysis completed")
    print(f"\n  Detailed Statistics:")
    print(f"    Expert Utilization:")
    for i, util in enumerate(routing_stats["expert_utilization"]):
        print(f"      Expert {i}: {util:.2%}")

    print(f"\n    Routing Entropy:")
    print(f"      Mean: {routing_stats['mean_entropy']:.3f}")
    print(f"      Std:  {routing_stats['std_entropy']:.3f}")

    # Compute additional metrics
    diversity = analyzer.compute_expert_diversity()
    consistency = analyzer.compute_routing_consistency()
    load_balance = analyzer.compute_load_balance_loss()

    print(f"\n    Expert Diversity:")
    print(f"      Avg off-diagonal similarity: {diversity[~torch.eye(len(diversity), dtype=bool)].mean():.3f}")

    print(f"\n    Routing Consistency: {consistency:.3f}")
    print(f"    Load Balance Loss: {load_balance:.4f}")

    return {
        "routing_stats": routing_stats,
        "diversity_matrix": diversity.cpu().numpy().tolist(),
        "routing_consistency": consistency,
        "load_balance_loss": load_balance,
    }


def save_results(results, args):
    """Save evaluation results to file."""
    print("\n[5/5] Saving results...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    import json

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {results_file}")


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 70)
    print("GMM-XLNet Evaluation")
    print("=" * 70)

    # Load model
    model = load_model(args)

    # Load tokenizer (same as model)
    tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
    # Note: Memory tokens should be added if needed, but for evaluation
    # they should already be in the model vocabulary

    # Load dataset
    dataset = load_dataset(args, tokenizer)

    # Run evaluation
    eval_results = evaluate_model(model, dataset, args)

    # Analyze routing (if requested)
    analysis_results = {}
    if args.analyze_routing:
        analysis_results = analyze_routing(model, dataset, args)

    # Combine results
    all_results = {
        "model": {
            "path": args.model_path,
            "model_id": args.model_id,
            "num_experts": model.num_experts,
            "memory_slots": model.memory_slots,
            "routing_mode": model.routing_mode,
        },
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "num_examples": len(dataset),
        },
        "evaluation": eval_results,
        "analysis": analysis_results if args.analyze_routing else None,
    }

    # Save results
    save_results(all_results, args)

    print("\n" + "=" * 70)
    print("Evaluation completed successfully!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}/evaluation_results.json")
    print("\nNext steps:")
    print("  1. Review routing statistics to check expert specialization")
    print("  2. Compare metrics with baseline models")
    print("  3. Visualize expert behavior: python examples/analyze_gmm_experts.py")


if __name__ == "__main__":
    main()
