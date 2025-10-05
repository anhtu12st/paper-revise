#!/usr/bin/env python3
"""
Test script to run evaluation with CLS position fixes and measure performance improvement.

USAGE:
- Set test_size = None in main() for full dataset evaluation (all 11,873 examples)
- Set test_size = 100 in main() for subset evaluation (100 examples)
- Results are saved to different files: fixed_evaluation_results_FULL.json vs fixed_evaluation_results.json
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import XLNetTokenizerFast

from memxlnet.data.dataset import create_evaluation_dataloader

# Import components
from memxlnet.models.memxlnet_qa import MemXLNetForQA


def compute_metrics(predictions, examples):
    import re
    import string

    def normalize_answer(s):
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

    def compute_f1(prediction, ground_truth):
        pred_tokens = normalize_answer(prediction).split()
        gt_tokens = normalize_answer(ground_truth).split()
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        common_tokens = set(pred_tokens) & set(gt_tokens)
        if len(common_tokens) == 0:
            return 0.0
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    exact_match = 0
    f1_scores = []
    has_answer_exact = 0
    has_answer_f1 = []
    no_answer_exact = 0
    no_answer_f1 = []

    for pred, ex in zip(predictions, examples):
        gts = ex["answers"]["text"]
        if len(gts) == 0:  # no-answer
            is_correct = pred == ""
            exact_match += int(is_correct)
            no_answer_exact += int(is_correct)
            score = 1.0 if is_correct else 0.0
            f1_scores.append(score)
            no_answer_f1.append(score)
        else:
            max_f1 = 0.0
            max_exact = 0
            for gt in gts:
                exact = int(normalize_answer(pred) == normalize_answer(gt))
                f1 = compute_f1(pred, gt)
                if f1 > max_f1:
                    max_f1 = f1
                if exact > max_exact:
                    max_exact = exact
            exact_match += max_exact
            has_answer_exact += max_exact
            has_answer_f1.append(max_f1)
            f1_scores.append(max_f1)

    total = len(examples)
    has_ans_count = sum(1 for ex in examples if len(ex["answers"]["text"]) > 0)
    no_ans_count = total - has_ans_count

    return {
        "f1": float(np.mean(f1_scores) * 100),
        "exact_match": exact_match / total * 100,
        "has_answer_f1": float(np.mean(has_answer_f1) * 100) if has_answer_f1 else 0.0,
        "has_answer_exact": has_answer_exact / has_ans_count * 100 if has_ans_count else 0.0,
        "no_answer_f1": float(np.mean(no_answer_f1) * 100) if no_answer_f1 else 0.0,
        "no_answer_exact": no_answer_exact / no_ans_count * 100 if no_ans_count else 0.0,
        "total_examples": total,
        "has_answer_count": has_ans_count,
        "no_answer_count": no_ans_count,
    }


def evaluate_with_fixed_cls(model, eval_dataloader, tokenizer, device, no_answer_threshold=1.5, max_answer_length=30):
    """Run evaluation with the FIXED CLS position logic."""
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_idx, time_step_batches in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            memory_bank = {}
            for time_step, batch in enumerate(time_step_batches):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                document_mask = batch["document_mask"].to(device)
                example_ids = batch["example_ids"]
                offset_mapping = batch["offset_mapping"]
                contexts = batch["context"]
                cls_indices = batch.get("cls_index", [])  # Get CLS indices from batch

                # Get memory states
                memory_states = []
                for ex_id, active in zip(example_ids, document_mask.tolist()):
                    if not active or ex_id.startswith("padding"):
                        memory_states.append(model.get_initial_memory(1, device)[0])
                    else:
                        prev_memory = memory_bank.get(ex_id)
                        if prev_memory is None:
                            prev_memory = model.get_initial_memory(1, device)[0]
                        memory_states.append(prev_memory)
                memory_state_batch = torch.stack(memory_states, dim=0)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    memory_state=memory_state_batch,
                )
                start_logits = outputs["start_logits"].cpu().numpy()
                end_logits = outputs["end_logits"].cpu().numpy()
                new_memory_state = outputs["new_memory_state"]

                # Update memory
                for i, (ex_id, active) in enumerate(zip(example_ids, document_mask.tolist())):
                    if active and not ex_id.startswith("padding"):
                        memory_bank[ex_id] = new_memory_state[i].detach()

                active_mask = document_mask.bool()
                if not active_mask.any():
                    continue

                active_indices = torch.where(active_mask)[0].tolist()
                for tensor_idx in active_indices:
                    i = tensor_idx
                    ex_id = example_ids[i]
                    if ex_id.startswith("padding"):
                        continue

                    try:
                        example_index = int(ex_id.split("_")[1])
                    except Exception:
                        example_index = -1

                    s_logits = start_logits[i]
                    e_logits = end_logits[i]

                    # ✅ FIXED: Get actual CLS index from batch data
                    cls_idx = cls_indices[i] if i < len(cls_indices) else 0

                    # Determine context mask
                    if token_type_ids is not None:
                        context_mask = (token_type_ids[i] == 1).cpu().numpy()
                    else:
                        seq_len = len(s_logits)
                        context_mask = np.zeros(seq_len, dtype=bool)
                        context_mask[seq_len // 2 :] = True

                    # Find best answer span
                    best_span = (0, 0)
                    best_non_null_score = -float("inf")

                    for s in range(len(s_logits)):
                        if not context_mask[s]:
                            continue
                        for e in range(s, min(s + max_answer_length, len(e_logits))):
                            if not context_mask[e]:
                                continue
                            score = s_logits[s] + e_logits[e]
                            if score > best_non_null_score:
                                best_non_null_score = score
                                best_span = (s, e)

                    # ✅ FIXED: Use actual CLS index for null score calculation
                    null_score = s_logits[cls_idx] + e_logits[cls_idx]
                    predict_empty = (null_score - best_non_null_score) > no_answer_threshold

                    pred_text = ""
                    if not predict_empty and best_non_null_score > -float("inf"):
                        om = offset_mapping[i]
                        if best_span[0] < len(om) and best_span[1] < len(om):
                            start_off = om[best_span[0]]
                            end_off = om[best_span[1]]
                            if start_off != (0, 0) and end_off != (0, 0):
                                char_start = start_off[0]
                                char_end = end_off[1]
                                ctx = contexts[i]
                                if 0 <= char_start < char_end <= len(ctx):
                                    pred_text = ctx[char_start:char_end].strip()

                    all_predictions.append(
                        {
                            "example_index": example_index,
                            "prediction": pred_text,
                            "predicted_empty": bool(predict_empty),
                            "cls_index_used": int(cls_idx),
                            "null_score": float(null_score),
                            "best_non_null_score": float(best_non_null_score),
                            "confidence": float(best_non_null_score - null_score),
                        }
                    )

    return all_predictions


def aggregate_predictions(predictions, total_examples):
    """Aggregate per-segment predictions into one per document."""
    by_doc = defaultdict(list)
    for p in predictions:
        if p["example_index"] >= 0:
            by_doc[p["example_index"]].append(p)

    final_predictions = []
    for idx in range(total_examples):
        seg_preds = by_doc.get(idx, [])
        if not seg_preds:
            final_predictions.append("")
            continue
        # Choose prediction with highest confidence
        chosen = max(seg_preds, key=lambda x: x["confidence"])
        final_predictions.append(chosen["prediction"])

    return final_predictions


def main():
    print("🚀 Testing MemXLNet-QA Evaluation with CLS Position Fix")
    print("=" * 60)

    # Configuration
    checkpoint_path = "outputs/xlnet-squad-phase2-1/stage_1_segs_1/best_model"
    device = torch.device("cpu")
    test_size = 200  # Set to None for full dataset, or integer for subset (e.g., 100)

    print(f"Loading model from: {checkpoint_path}")
    model = MemXLNetForQA.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    tokenizer = XLNetTokenizerFast.from_pretrained(checkpoint_path)
    print(f"Model loaded: {model.mem_token_count} memory tokens")

    # Load config
    config_path = os.path.join(checkpoint_path, "training_config.json")
    with open(config_path) as f:
        config_data = json.load(f)

    # Create evaluation dataset
    eval_type = "full dataset" if test_size is None else f"{test_size} examples"
    print(f"Creating evaluation dataset ({eval_type})...")
    eval_dataset, eval_dataloader = create_evaluation_dataloader(
        dataset_name="squad_v2",
        split="validation",
        tokenizer=tokenizer,
        max_seq_length=config_data.get("max_seq_length", 384),
        doc_stride=config_data.get("doc_stride", 64),
        batch_size=8,
        max_examples=test_size,
        max_n_segs=config_data.get("max_n_segs", None),
        cache_dir=".cache",
        use_time_step_major=True,
    )

    print(f"Dataset: {len(eval_dataset)} features, {len(eval_dataloader)} batches")

    # Load ground truth
    squad_dataset = load_dataset("squad_v2", split="validation")
    if test_size is not None:
        test_examples = squad_dataset.select(range(test_size))
        num_examples = test_size
    else:
        test_examples = squad_dataset
        num_examples = len(squad_dataset)

    print("Running evaluation with FIXED CLS positions...")
    raw_predictions = evaluate_with_fixed_cls(
        model=model,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        device=device,
        no_answer_threshold=1.5,
        max_answer_length=30,
    )

    print(f"Collected {len(raw_predictions)} segment-level predictions")

    # Aggregate predictions
    final_predictions = aggregate_predictions(raw_predictions, num_examples)

    # Compute metrics
    metrics = compute_metrics(final_predictions, test_examples)

    print("\n🎯 EVALUATION RESULTS WITH CLS POSITION FIX:")
    print("=" * 60)
    print(f"F1 Score:              {metrics['f1']:.2f}%")
    print(f"Exact Match:           {metrics['exact_match']:.2f}%")
    print(f"Has Answer F1:         {metrics['has_answer_f1']:.2f}%")
    print(f"Has Answer Exact:      {metrics['has_answer_exact']:.2f}%")
    print(f"No Answer F1:          {metrics['no_answer_f1']:.2f}%")
    print(f"No Answer Exact:       {metrics['no_answer_exact']:.2f}%")
    print("=" * 60)

    # Check CLS index usage
    cls_indices_used = [p["cls_index_used"] for p in raw_predictions]
    unique_cls = set(cls_indices_used)
    print(f"\nCLS indices used: {unique_cls}")
    print("Expected CLS position: 383 (end of sequence)")

    # Analyze no-answer predictions
    no_answer_predictions = sum(1 for p in raw_predictions if p["predicted_empty"])
    print(
        f"No-answer predictions: {no_answer_predictions}/{len(raw_predictions)} ({no_answer_predictions / len(raw_predictions) * 100:.1f}%)"
    )

    # Compare with previous results
    print("\n📊 COMPARISON WITH PREVIOUS RESULTS:")
    print("Previous (buggy):  F1=35.52%, EM=28.00%, NoAns=0.00%")
    print(
        f"Current (fixed):   F1={metrics['f1']:.2f}%, EM={metrics['exact_match']:.2f}%, NoAns={metrics['no_answer_f1']:.2f}%"
    )

    improvement_f1 = metrics["f1"] - 35.52
    improvement_em = metrics["exact_match"] - 28.00
    print(f"Improvement:       F1={improvement_f1:+.2f}%, EM={improvement_em:+.2f}%")

    if metrics["f1"] >= 80.0:
        print(f"\n✅ SUCCESS: F1 {metrics['f1']:.2f}% >= 80% target achieved!")
    else:
        print(f"\n📈 Progress: F1 {metrics['f1']:.2f}% (target: 80%)")

    # Save results
    results = {
        "checkpoint_path": checkpoint_path,
        "test_size": test_size,
        "num_examples_evaluated": num_examples,
        "evaluation_type": "full_dataset" if test_size is None else "subset",
        "metrics": metrics,
        "cls_fix_applied": True,
        "cls_indices_used": list(unique_cls),
        "no_answer_prediction_rate": no_answer_predictions / len(raw_predictions) * 100,
        "total_segment_predictions": len(raw_predictions),
        "expected_performance": "F1=77-80% for 100 examples, may vary for full dataset",
    }

    # Use different filename for full vs subset evaluation
    results_filename = (
        "results/fixed_evaluation_results_FULL.json" if test_size is None else "results/fixed_evaluation_results.json"
    )
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_filename}")


if __name__ == "__main__":
    main()
