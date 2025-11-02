#!/usr/bin/env python3
"""
Example script for analyzing GMM expert specialization and routing behavior.

This script demonstrates the complete workflow for interpretability analysis:
1. Load a trained GMM model
2. Run evaluation and track routing probabilities
3. Compute specialization metrics
4. Generate visualizations
5. Save analysis report

Usage:
    python examples/analyze_gmm_experts.py \\
        --model-path outputs/gmmxlnet-k4 \\
        --data-path data/squad_dev.json \\
        --output-dir analysis_results \\
        --max-segments 100
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gmmxlnet.models import GMMXLNetForQA
from gmmxlnet.utils import GMMAnalyzer, generate_all_visualizations
from memxlnet.data import ChunkedSquadDataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze GMM expert specialization and routing behavior"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained GMM model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to evaluation dataset (JSON file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Directory to save analysis results (default: analysis_results)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Maximum number of segments to process (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model identifier for report (default: model-path)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="squad_dev",
        help="Dataset name for report (default: squad_dev)",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["png", "pdf"],
        help="Output formats for visualizations (default: png pdf)",
    )

    return parser.parse_args()


def load_model(model_path: str, device: str) -> GMMXLNetForQA:
    """Load trained GMM model from checkpoint."""
    print(f"Loading model from: {model_path}")
    model = GMMXLNetForQA.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"Model loaded with {model.memory_mixture.num_experts} experts")
    return model


def prepare_dataloader(
    data_path: str,
    batch_size: int,
    model: GMMXLNetForQA,
) -> DataLoader:
    """Prepare evaluation dataloader."""
    print(f"Loading dataset from: {data_path}")

    # Load dataset
    dataset = ChunkedSquadDataset(
        data_path=data_path,
        tokenizer=model.tokenizer,
        max_length=512,
        stride=256,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    print(f"Dataset loaded: {len(dataset)} examples")
    return dataloader


def main():
    """Main analysis workflow."""
    args = parse_args()

    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    model = load_model(args.model_path, device)

    # Prepare dataloader
    dataloader = prepare_dataloader(args.data_path, args.batch_size, model)

    # Initialize analyzer
    print("\nInitializing GMMAnalyzer...")
    analyzer = GMMAnalyzer(model=model, device=device)

    # Track routing probabilities
    print(f"\nTracking routing behavior (max_segments={args.max_segments or 'all'})...")
    summary = analyzer.track_routing(
        dataloader=dataloader,
        max_segments=args.max_segments,
    )

    print(f"Tracking complete: {summary['segments_processed']} segments processed")
    print(f"Mean routing entropy: {summary['mean_routing_entropy']:.4f}")
    print(f"Expert activation frequencies: {summary['expert_activation_freq']}")

    # Export routing data to JSON
    routing_json_path = output_dir / "routing_data.json"
    print(f"\nExporting routing data to: {routing_json_path}")
    analyzer.export_routing_to_json(str(routing_json_path))

    # Compute specialization metrics
    print("\nComputing specialization metrics...")
    expert_embeddings = analyzer.extract_expert_embeddings()
    metrics = analyzer.compute_specialization_score(expert_embeddings)

    print("Specialization Metrics:")
    print(f"  - Routing Entropy: {metrics['routing_entropy']:.4f}")
    print(f"  - Expert Diversity: {metrics['expert_diversity']:.4f}")
    print(f"  - Utilization Balance: {metrics['utilization_balance']:.4f}")

    # Generate visualizations
    print(f"\nGenerating visualizations (formats: {args.formats})...")
    viz_dir = output_dir / "visualizations"
    saved_files = generate_all_visualizations(
        analyzer=analyzer,
        output_dir=str(viz_dir),
        formats=args.formats,
        dpi=300,
    )

    print(f"Visualizations saved to: {viz_dir}")
    for viz_type, path in saved_files.items():
        print(f"  - {viz_type}: {path}")

    # Generate comprehensive analysis report
    report_path = output_dir / "analysis_report.json"
    print(f"\nGenerating comprehensive analysis report...")
    report = analyzer.generate_analysis_report(
        output_path=str(report_path),
        include_model_id=args.model_id or args.model_path,
        dataset_name=args.dataset_name,
    )

    print(f"Analysis report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Model: {report['model_id']}")
    print(f"Dataset: {report['evaluation_dataset']}")
    print(f"Experts: {report['num_experts']}")
    print(f"\nMetrics:")
    for metric_name, metric_value in report["metrics"].items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print(f"\nExpert Activations:")
    for expert_id, activation in report["expert_activations"].items():
        print(f"  {expert_id}: {activation:.4f}")
    print(f"\nAll results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
