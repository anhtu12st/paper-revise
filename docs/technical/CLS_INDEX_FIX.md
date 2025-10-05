# CLS Index Fix - Training Evaluation Alignment

## Issue
The training script's evaluation logic did **not** use the same CLS position handling as the test evaluation script, leading to potential discrepancies in no-answer predictions.

### Previous Behavior (Buggy)
- **Test script** (`test_evaluation.py`): Retrieved actual CLS index from batch data: `cls_indices = batch.get("cls_index", [])`
- **Trainer** (`trainer.py`): Searched for CLS token in input_ids at runtime, with fallback heuristics

This mismatch could cause:
- Different null scores during training vs evaluation
- Incorrect no-answer predictions
- Performance degradation (especially for SQuAD v2 no-answer questions)

## Solution

Updated the trainer to match the test evaluation script's approach:

### Changes Made

#### 1. **Collect CLS indices from batch** (`_evaluate_one_document`)
   - Added `doc_cls_indices = []` collection list
   - Extract cls_index from batch: `chunk.get("cls_index", [])`
   - Include in predictions_data dictionary

#### 2. **Pass CLS indices through pipeline** (`evaluate`)
   - Added `all_cls_indices = []` collection list
   - Collect from doc_predictions: `all_cls_indices.extend(doc_predictions["cls_indices"])`
   - Pass to `extract_predictions_from_logits()`

#### 3. **Accept CLS indices parameter** (`extract_predictions_from_logits`)
   - New parameter: `all_cls_indices: list[int] | None = None`
   - Include cls_index in entry dict: `entry["cls_index"] = all_cls_indices[i]`
   - Handle in array length validation

#### 4. **Use batch CLS index** (`_extract_prediction_any_positive`)
   ```python
   # ✅ FIXED: Use actual CLS index from batch (like test_evaluation.py)
   cls_index = chunk_data.get("cls_index", None)

   # Fallback: If cls_index not in batch, search for it (legacy compatibility)
   if cls_index is None:
       if ids is not None and isinstance(ids, list):
           cls_token_id = self.tokenizer.cls_token_id
           cls_positions = [i for i, tok in enumerate(ids) if tok == cls_token_id]
           if cls_positions:
               cls_index = cls_positions[-1]
   ```

#### 5. **Use batch CLS index** (`_extract_prediction_best_score`)
   - Same fix as above - use `chunk_data.get("cls_index", None)` first
   - Fallback to search only if not present

### Key Improvements

✅ **Consistency**: Training evaluation now matches test evaluation exactly
✅ **Accuracy**: Uses precomputed CLS positions from dataset preprocessing
✅ **Performance**: No runtime search for CLS token needed
✅ **Backward Compatibility**: Falls back to search if cls_index not in batch

## Files Modified

- `src/memxlnet/training/trainer.py`:
  - `_evaluate_one_document()` - collect cls_indices from batches
  - `evaluate()` - pass cls_indices through pipeline
  - `extract_predictions_from_logits()` - accept and propagate cls_indices
  - `_extract_prediction_any_positive()` - use batch cls_index
  - `_extract_prediction_best_score()` - use batch cls_index

## Expected Impact

- **No-answer predictions**: Should now correctly use the actual CLS position (typically position 383 for 384-length sequences in XLNet)
- **F1/EM scores**: Should match between training evaluation and standalone test scripts
- **SQuAD v2 performance**: Better no-answer detection accuracy

## Testing

To verify the fix:
1. Run training with evaluation: `python scripts/train_memxlnet_squad.py`
2. Run standalone test: `python tests/integration/test_evaluation.py`
3. Compare metrics - they should now be identical

## Related Files

- Dataset preprocessing: `src/memxlnet/data/dataset.py` (computes cls_index at line 214)
- Test evaluation: `tests/integration/test_evaluation.py` (reference implementation)
- Training script: `scripts/train_memxlnet_squad.py` (uses fixed trainer)

## Date
January 2025
