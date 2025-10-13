#!/usr/bin/env python3
"""
Debug Predictions Script
========================

Quick script to inspect what a trained model is actually predicting.
This helps diagnose why F1 scores are 0%.

Usage:
    python scripts/debug_predictions.py --model-path outputs/comparison_differentiable
"""

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def debug_model_predictions(model_path: str, num_samples: int = 10):
    """Load a model and inspect its predictions on a few samples."""
    print("\n" + "=" * 80)
    print("🔍 DEBUGGING MODEL PREDICTIONS")
    print("=" * 80)
    print()

    # Load model
    print(f"📁 Loading model from: {model_path}")

    try:
        # Create a minimal config to load the model
        config = TrainingConfig(
            model_name="xlnet-base-cased",
            max_seq_length=384,
            doc_stride=64,
            dataset_name="squad_v2",
            train_split="train",
            eval_split="validation",
            max_train_samples=10,
            max_eval_samples=num_samples,
            memory_num_tokens=8,
            memory_impl="differentiable",
            use_differentiable_memory=True,
            num_epochs=1,
            output_dir=model_path,
            no_answer_threshold=0.0,
        )

        trainer = XLNetRecurrentTrainer(config)

        # Try to load checkpoint
        checkpoint_path = Path(model_path) / "checkpoint-best"
        if not checkpoint_path.exists():
            checkpoint_path = Path(model_path) / "checkpoint-final"

        if checkpoint_path.exists():
            print(f"✅ Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path / "pytorch_model.bin", map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint)
        else:
            print("⚠️  No checkpoint found, using freshly initialized model")
            print("   (Predictions will be random)")

        # Prepare evaluation data - FIXED: Use create_dataset_from_cache
        print(f"\n📚 Loading {num_samples} evaluation samples...")
        from memxlnet.data.dataset import create_dataset_from_cache

        eval_dataset = create_dataset_from_cache(
            dataset_name=config.dataset_name,
            split=config.eval_split,
            cache_dir=config.cache_dir,
            max_examples=num_samples,
            max_seq_length=config.max_seq_length,
            doc_stride=config.doc_stride,
            max_n_segs=None,
            tokenizer=trainer.tokenizer,
        )

        print(f"✅ Loaded dataset with {len(eval_dataset)} features")

        # Make predictions on a few samples
        print("\n🎯 Making predictions...")
        trainer.model.eval()

        samples_to_check = min(num_samples, len(eval_dataset))

        with torch.no_grad():
            for i in range(samples_to_check):
                # Get feature from dataset
                feature = eval_dataset[i]

                # Convert feature to tensors (handle both dict and tensor inputs)
                if isinstance(feature["input_ids"], torch.Tensor):
                    input_ids = feature["input_ids"].unsqueeze(0).to(trainer.device)
                    attention_mask = feature["attention_mask"].unsqueeze(0).to(trainer.device)
                    token_type_ids = (
                        feature["token_type_ids"].unsqueeze(0).to(trainer.device)
                        if "token_type_ids" in feature
                        else torch.zeros_like(input_ids)
                    )
                else:
                    input_ids = torch.tensor([feature["input_ids"]]).to(trainer.device)
                    attention_mask = torch.tensor([feature["attention_mask"]]).to(trainer.device)
                    token_type_ids = torch.tensor(
                        [feature.get("token_type_ids", [0] * len(feature["input_ids"]))]
                    ).to(trainer.device)

                # Get model outputs
                if hasattr(trainer.model, "get_initial_memory"):
                    memory_state = trainer.model.get_initial_memory(1, device=trainer.device)
                    outputs = trainer.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        memory_state=memory_state,
                    )
                else:
                    outputs = trainer.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )

                start_logits = outputs.start_logits[0].cpu().numpy()
                end_logits = outputs.end_logits[0].cpu().numpy()

                # Find best answer span
                best_start_idx = start_logits.argmax()
                best_end_idx = end_logits.argmax()
                best_score = start_logits[best_start_idx] + end_logits[best_end_idx]

                # Find CLS token (no-answer) score
                cls_token_id = trainer.tokenizer.cls_token_id
                input_ids_list = input_ids[0].tolist()
                cls_positions = [j for j, tok in enumerate(input_ids_list) if tok == cls_token_id]
                cls_idx = cls_positions[-1] if cls_positions else len(start_logits) - 1
                no_answer_score = start_logits[cls_idx] + end_logits[cls_idx]

                # Extract answer text
                if best_start_idx <= best_end_idx and best_start_idx < len(input_ids_list):
                    answer_tokens = input_ids_list[best_start_idx : best_end_idx + 1]
                    answer_text = trainer.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                else:
                    answer_text = "[INVALID SPAN]"

                print(f"\n📝 Sample {i + 1}:")
                print(f"   Example ID: {feature.get('example_id', 'unknown')}")
                print(f"   Best answer span: [{best_start_idx}, {best_end_idx}]")
                print(f"   Best answer score: {best_score:.4f}")
                print(f"   No-answer score (CLS@{cls_idx}): {no_answer_score:.4f}")
                print(f"   Score difference: {best_score - no_answer_score:.4f}")
                print(f"   Predicted text: '{answer_text}'")

                # Show ground truth if available
                if "chosen_answer_text" in feature:
                    gt_answer = feature["chosen_answer_text"]
                    print(f"   Ground truth: '{gt_answer}'")
                    match = "✅ MATCH!" if answer_text.strip().lower() == gt_answer.strip().lower() else "❌ MISMATCH"
                    print(f"   {match}")

                # Show some logit statistics
                print(
                    f"   Start logits - min: {start_logits.min():.2f}, max: {start_logits.max():.2f}, mean: {start_logits.mean():.2f}"
                )
                print(
                    f"   End logits   - min: {end_logits.min():.2f}, max: {end_logits.max():.2f}, mean: {end_logits.mean():.2f}"
                )

        print("\n" + "=" * 80)
        print("✅ Debugging complete")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during debugging: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Debug model predictions")
    parser.add_argument(
        "--model-path", type=str, default="./outputs/comparison_differentiable", help="Path to model directory"
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to inspect")

    args = parser.parse_args()

    return debug_model_predictions(args.model_path, args.num_samples)


if __name__ == "__main__":
    sys.exit(main())
