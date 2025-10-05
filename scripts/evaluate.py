#!/usr/bin/env python3
"""Evaluation script for MemXLNet-QA models.

This script loads a trained model and evaluates it on the validation set.

Usage:
    python scripts/evaluate.py <config_path> [model_path]

Examples:
    python scripts/evaluate.py outputs/xlnet-squad/training_config.json
    python scripts/evaluate.py config.json outputs/xlnet-squad/best_model
"""

import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memxlnet.evaluation.evaluator import main as evaluate_main


def main():
    """Run evaluation from command line."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate.py <config_path> [model_path]")
        print("\nExamples:")
        print("  python scripts/evaluate.py outputs/xlnet-squad/training_config.json")
        print("  python scripts/evaluate.py config.json outputs/xlnet-squad/best_model")
        sys.exit(1)

    config_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    evaluate_main(config_path, model_path)


if __name__ == "__main__":
    main()
