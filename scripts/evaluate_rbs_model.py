#!/usr/bin/env python3
"""
RBS-QA Model Evaluation Script

Usage:
    python evaluate_rbs_model.py --model_path ./outputs/rbs_experiment --test_file test.json
    python evaluate_rbs_model.py --model_path ./outputs/rbs_experiment --test_file test.json --baseline_gmm ./outputs/gmm_experiment
"""

import argparse
import logging
import sys
from pathlib import Path
import json

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rbsqa.evaluation.rbs_evaluator import RBSEvaluator, RBSEvaluationConfig
try:
    from rbsqa.models.rbs_xlnet import RBSXLNetForQA
    from rbsqa.data.rbs_dataset import RBSQADataset
except ImportError:
    # Fallback imports if RBS modules not available
    RBSXLNetForQA = None
    RBSQADataset = None

# Optional baseline imports
try:
    from gmmxlnet.models.gmm_xlnet_qa import GMMXLNetForQA
except ImportError:
    GMMXLNetForQA = None

try:
    from memxlnet.models.memxlnet_qa import MemXLNetForQA
except ImportError:
    MemXLNetForQA = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RBS-QA Model Evaluation")

    # Model and data
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained RBS model")
    parser.add_argument("--test_file", type=str, required=True,
                       help="Test dataset file")
    parser.add_argument("--baseline_gmm", type=str, default=None,
                       help="Path to GMM baseline model")
    parser.add_argument("--baseline_xlnet", type=str, default=None,
                       help="Path to base XLNet model")

    # Configuration
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--max_segments", type=int, default=32,
                       help="Maximum segments per example")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")

    # Analysis options
    parser.add_argument("--detailed_analysis", action="store_true", default=True,
                       help="Perform detailed analysis")
    parser.add_argument("--no_detailed_analysis", dest="detailed_analysis", action="store_false",
                       help="Skip detailed analysis")
    parser.add_argument("--generate_visualizations", action="store_true", default=True,
                       help="Generate visualizations")
    parser.add_argument("--statistical_tests", action="store_true", default=True,
                       help="Perform statistical significance tests")

    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")

    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    return parser.parse_args()


def load_test_dataset(test_file: str) -> list:
    """Load test dataset from file."""
    try:
        if RBSQADataset:
            return RBSQADataset.from_file(test_file)
        else:
            # Fallback to loading JSON directly
            with open(test_file, 'r') as f:
                data = json.load(f)

            if isinstance(data, dict) and 'data' in data:
                return data['data']
            elif isinstance(data, list):
                return data
            else:
                raise ValueError(f"Unexpected data format in {test_file}")
    except Exception as e:
        logger.error(f"Error loading test dataset: {e}")
        # Return dummy dataset for testing
        return create_dummy_dataset()


def create_dummy_dataset() -> list:
    """Create a dummy dataset for testing purposes."""
    logger.warning("Creating dummy dataset for testing")
    dummy_data = []
    for i in range(10):
        dummy_data.append({
            'id': f'dummy_{i}',
            'question_input_ids': [101, 1029, 102] + [103] * 20 + [102],  # Example token IDs
            'context_segments': [
                [101] + [103] * 50 + [102] for _ in range(5)  # 5 segments
            ],
            'answer_span': (10, 15),
            'question': f"What is the answer to question {i}?"
        })
    return dummy_data


def load_model(model_path: str, model_class, model_name: str, device: str):
    """Load a model from path."""
    if model_class is None:
        logger.warning(f"{model_name} class not available, skipping")
        return None

    try:
        logger.info(f"Loading {model_name} from: {model_path}")
        model = model_class.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")
        return None


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=getattr(logging, args.log_level.upper())
    )
    global logger
    logger = logging.getLogger(__name__)

    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    # Load test dataset
    logger.info(f"Loading test dataset from: {args.test_file}")
    test_dataset = load_test_dataset(args.test_file)
    logger.info(f"Loaded {len(test_dataset)} test examples")

    if len(test_dataset) == 0:
        logger.error("No test examples loaded. Exiting.")
        return

    # Load RBS model
    rbs_model = load_model(args.model_path, RBSXLNetForQA, "RBS model", device)
    if rbs_model is None:
        logger.error("Failed to load RBS model. Exiting.")
        return

    # Load baseline models
    baseline_models = {}

    if args.baseline_gmm:
        gmm_model = load_model(args.baseline_gmm, GMMXLNetForQA, "GMM baseline", device)
        if gmm_model:
            baseline_models['gmm'] = gmm_model

    if args.baseline_xlnet:
        xlnet_model = load_model(args.baseline_xlnet, MemXLNetForQA, "base XLNet", device)
        if xlnet_model:
            baseline_models['base_xlnet'] = xlnet_model

    # Setup evaluation config
    eval_config = RBSEvaluationConfig(
        output_dir=args.output_dir,
        max_segments_per_example=args.max_segments,
        generate_visualizations=args.generate_visualizations,
        save_detailed_results=args.detailed_analysis,
        baseline_comparisons=list(baseline_models.keys()),
        statistical_tests=args.statistical_tests
    )

    # Initialize evaluator
    logger.info("Initializing RBS evaluator...")
    evaluator = RBSEvaluator(
        model=rbs_model,
        config=eval_config,
        output_dir=args.output_dir
    )

    # Run evaluation
    logger.info("Starting comprehensive evaluation...")
    try:
        evaluation_results = evaluator.evaluate(
            test_dataset=test_dataset,
            baseline_models=baseline_models if baseline_models else None,
            detailed_analysis=args.detailed_analysis
        )

        # Print summary
        main_summary = evaluation_results.get('main', {}).get('summary', {})
        logger.info("=== Evaluation Summary ===")
        logger.info(f"F1 Score: {main_summary.get('f1', 0.0):.3f}")
        logger.info(f"Exact Match: {main_summary.get('exact_match', 0.0):.3f}")
        logger.info(f"Efficiency Score: {main_summary.get('avg_efficiency_score', 0.0):.3f}")
        logger.info(f"Combined Score: {main_summary.get('combined_score', 0.0):.3f}")

        if 'comparisons' in evaluation_results:
            logger.info("\n=== Baseline Comparisons ===")
            for baseline_name, baseline_results in evaluation_results['comparisons'].items():
                if 'summary' in baseline_results:
                    baseline_summary = baseline_results['summary']
                    logger.info(f"{baseline_name}:")
                    logger.info(f"  F1: {baseline_summary.get('f1', 0.0):.3f}")
                    logger.info(f"  Efficiency: {baseline_summary.get('avg_efficiency_score', 0.0):.3f}")

        logger.info(f"\nDetailed results saved to: {args.output_dir}")

        # Print key findings
        if 'analysis' in evaluation_results:
            logger.info("\n=== Key Findings ===")
            analysis = evaluation_results['analysis']

            if 'non_monotonic_reasoning' in analysis:
                nm_reasoning = analysis['non_monotonic_reasoning']
                logger.info(f"Non-monotonic reasoning detected: {nm_reasoning.get('summary', {}).get('non_monotonic_reasoning_detected', False)}")
                logger.info(f"Revision frequency: {nm_reasoning.get('revision_frequency', 0.0):.3f}")

            if 'halting_policy' in analysis:
                halting = analysis['halting_policy']
                logger.info(f"Avg halt confidence: {halting.get('summary', {}).get('avg_halt_confidence', 0.0):.3f}")
                logger.info(f"Policy consistency: {halting.get('summary', {}).get('policy_consistency_score', 0.0):.3f}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error("Evaluation failed, but partial results may be saved.")
        return

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()