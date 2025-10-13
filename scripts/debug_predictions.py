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
import re
import string
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from memxlnet.data.dataset import create_dataset_from_cache
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def normalize_answer(s):
    """Normalize answer text for comparison (same as SQuAD evaluation)."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


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
            model_name=model_path,
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

        # Prepare evaluation data - FIXED: Use create_dataset_from_cache
        print(f"\n📚 Loading {num_samples} evaluation samples...")

        # Disable Hub dataset to use local processing
        eval_dataset = create_dataset_from_cache(
            dataset_name=config.dataset_name,
            split=config.eval_split,
            cache_dir=config.cache_dir,
            max_examples=num_samples,
            max_seq_length=config.max_seq_length,
            doc_stride=config.doc_stride,
            max_n_segs=None,
            tokenizer=trainer.tokenizer,
            use_hub_dataset=False,  # Don't try to load from Hub
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

                # Model returns dict, not object with attributes
                start_logits = outputs["start_logits"][0].cpu().numpy()
                end_logits = outputs["end_logits"][0].cpu().numpy()

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

                # Extract answer text using offset mapping (better than token decoding)
                offset_mapping = feature.get("offset_mapping", [])
                context = feature.get("context", "")
                token_type_ids_list = token_type_ids[0].tolist() if hasattr(token_type_ids, 'tolist') else token_type_ids

                # Check if span is in context tokens
                is_context_span = False
                if best_start_idx < len(token_type_ids_list) and best_end_idx < len(token_type_ids_list):
                    is_context_span = token_type_ids_list[best_start_idx] == 1 and token_type_ids_list[best_end_idx] == 1

                # Extract using offsets if available
                if best_start_idx == cls_idx or best_end_idx == cls_idx:
                    answer_text = ""  # No-answer prediction
                elif offset_mapping and best_start_idx < len(offset_mapping) and best_end_idx < len(offset_mapping) and context:
                    start_char = offset_mapping[best_start_idx][0]
                    end_char = offset_mapping[best_end_idx][1]
                    if start_char < end_char and end_char <= len(context):
                        answer_text = context[start_char:end_char].strip()
                    else:
                        answer_text = "[INVALID OFFSET]"
                elif best_start_idx <= best_end_idx and best_start_idx < len(input_ids_list):
                    answer_tokens = input_ids_list[best_start_idx : best_end_idx + 1]
                    answer_text = trainer.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                else:
                    answer_text = "[INVALID SPAN]"

                # Count context vs question tokens
                context_token_count = sum(1 for t in token_type_ids_list if t == 1)
                question_token_count = sum(1 for t in token_type_ids_list if t == 0)

                print(f"\n📝 Sample {i + 1}:")
                print(f"   Example ID: {feature.get('example_id', 'unknown')}")
                print(f"   Sequence info: {len(input_ids_list)} tokens ({context_token_count} context, {question_token_count} question/special)")
                print(f"   Best answer span: [{best_start_idx}, {best_end_idx}] (is_context={is_context_span}, span_length={best_end_idx - best_start_idx + 1})")
                print(f"   Best answer score: {best_score:.4f}")
                print(f"   No-answer score (CLS@{cls_idx}): {no_answer_score:.4f}")
                print(f"   Score difference: {best_score - no_answer_score:.4f}")
                print(f"   Predicted text: '{answer_text[:200]}{'...' if len(answer_text) > 200 else ''}'")

                # Show ground truth if available - compare against ALL valid answers (SQuAD v2)
                if "all_valid_answers" in feature and feature["all_valid_answers"]:
                    # SQuAD v2 validation has multiple valid answers
                    all_gt_answers = feature["all_valid_answers"]
                    print(f"   Ground truth answers ({len(all_gt_answers)}): {all_gt_answers}")

                    # Check if prediction matches ANY valid answer (with proper normalization)
                    normalized_pred = normalize_answer(answer_text)
                    matches = []
                    for idx, gt_answer in enumerate(all_gt_answers):
                        if normalized_pred == normalize_answer(gt_answer):
                            matches.append(idx)

                    if matches:
                        matched_answers = [all_gt_answers[i] for i in matches]
                        print(f"   ✅ MATCH! (matches answer(s): {matched_answers})")
                    else:
                        print(f"   ❌ MISMATCH (doesn't match any of {len(all_gt_answers)} valid answers)")
                        print(f"      Normalized prediction: '{normalized_pred}'")
                        print(f"      Normalized GT answers: {[normalize_answer(a) for a in all_gt_answers]}")
                elif "chosen_answer_text" in feature:
                    # Fallback to single answer (training data or old cache)
                    gt_answer = feature["chosen_answer_text"]
                    print(f"   Ground truth: '{gt_answer}'")
                    normalized_pred = normalize_answer(answer_text)
                    normalized_gt = normalize_answer(gt_answer)
                    match = "✅ MATCH!" if normalized_pred == normalized_gt else "❌ MISMATCH"
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
