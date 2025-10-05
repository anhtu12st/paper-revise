# Copilot Project Instructions

Purpose: Rapid ramp-up for AI coding agents contributing to MemXLNet-QA (memory‑augmented XLNet for long‑context QA). Focus on what is unique here: progressive segment curriculum, explicit memory tokens, time‑step‑major batching, and evaluation/no‑answer logic.

## 1. Core Architecture (Know These Files First)
- `src/train.py`: Orchestrates training via `XLNetRecurrentTrainer` + `TrainingConfig`. Implements progressive segment stages and document-level (global) softmax logic.
- `src/data.py`: Builds segmented SQuAD-like dataset (`SquadLikeQADataset`) and time‑step‑major batches (list of per-time-step mini-batches) plus memory token injection (`configure_memory_tokens`).
- `src/memxlnet_qa.py`: Wrapper `MemXLNetForQA` adding explicit learned/zero memory, gated updates, extraction of memory write token hidden states.
- `src/evaluate.py`: Loads a saved `training_config.json` and re-runs evaluation using trainer path.
- `phase2_train.py`: Opinionated Phase 2 config (short-range recurrence) showing recommended memory settings.

## 2. Memory & Segmentation Mechanics
- Memory enabled when `memory_num_tokens > 0`; tokenizer extended with `[MEM_READ_i]` / `[MEM_WRITE_i]` pairs (count = `memory_num_tokens`).
- Gated update path (most runs) = concatenate(old,new) -> sigmoid gate + tanh update -> blend.
- Trainer builds a per-document `memory_bank` keyed by `example_id` (synthetic ids like `doc_42`). Memory cleared after a document finishes to cap growth.
- Progressive segments (`progressive_segments` list) drive curriculum (e.g. `[1,2]` or `[2]`). Each stage writes into nested output folders (`stage_1_segs_1/`, `stage_2_segs_2/`, etc.).
- Global span selection: if `use_global_softmax` epoch flag active, concatenates segment logits per document before CE loss; else averages per-segment CE.
- "Any positive" logic (`use_any_positive_logic`) during prediction: only predict no‑answer if ALL segments vote no answer.

## 3. Dataloader Pattern (Critical Difference)
Standard batch -> NOT used for recurrence. Instead: `create_dataloader` returns batches of documents restructured into a list `time_step_batches` (time‑step major). For each time step: feed batch with aligned segment indices; memory propagated across list iteration.

## 4. Output & Run Metadata Conventions
- Root output: `outputs/<run_name or derived>/` plus stage subdirs. Each stage may contain `best_model/`, `final_model/`, and `run_metadata.json` summarizing memory + segmentation parameters.
- Memory-enabled checkpoints may include `memxlnet_config.json` (saved by wrapper) and extended tokenizer vocab (extra special tokens).
- Test / quick runs use smaller dirs: `test_output/`, `test_resume_output/` referencing earlier checkpoints for resume validation.

## 5. Config System Nuances
`TrainingConfig` stores everything; minimal aliasing internally (earlier docs mention aliases but current code defines single canonical names). Post-init ensures dirs exist. Key toggles: memory_* fields, `progressive_segments`, `use_global_softmax`, warmup disables (`warmup_disable_global_softmax_epochs`, etc.). Warmup logic flips internal epoch-scoped flags (`_use_global_softmax_epoch`).

## 6. Evaluation Flow
Evaluation reuses trainer path: builds datasets the same way, gathers per-segment logits, groups by `example_id`, then applies either global or per-segment aggregation and any-positive logic before SQuAD-style metrics (Exact, F1, HasAns/NoAns splits). No-answer threshold (`no_answer_threshold`) currently passed but main gating logic is structural (any-positive) rather than calibrated probabilities.

## 7. Typical Workflows (Agent Should Reuse)
- Phase 2 full memory train: `python phase2_train.py` (preferred entry). 
- Baseline/simple: `python -m src.train` with adjusted `TrainingConfig` inline.
- Evaluation of saved config: `python -m src.evaluate outputs/<run>/training_config.json [optional_model_dir]`.
- Checkpoint validation script: `python test_checkpoint_validation.py` (loads several checkpoints, runs smoke inference + optional tiny eval with memory off for speed).

## 8. Safe Extension Guidelines
- When adding new memory strategies: extend `MemXLNetForQA` (e.g. new `memory_update` mode) and gate selection in `_update_memory`; keep backward compatibility with existing saved `memxlnet_config.json` keys.
- For new curriculum stages: ensure folder naming pattern `stage_<n>_segs_<k>` preserved (downstream tooling may parse this).
- If altering batching: preserve interface where `train_one_document_batch` receives `List[Dict[str,Tensor]]` ordered by time step.
- Add new config fields with defaults to avoid breaking old serialized `training_config.json`.

## 9. Common Pitfalls & Gotchas
- Forgetting to resize embeddings after adding memory tokens: trainer already calls `resize_token_embeddings`; replicate if you manually wrap models.
- Memory growth: MUST remove completed docs' entries from `memory_bank` (trainer already does post-loss). Preserve this cleanup if refactoring.
- Global softmax off-by-one: offset construction relies on segment lengths in order appended; keep ordering stable if introducing padding logic.
- Tiny eval / CPU runs: set `memory_num_tokens=0` to bypass memory wrapper for speed in smoke tests.

## 10. What NOT to Change Lightly
- Field names in `TrainingConfig` (persisted to disk in `training_config.json`).
- Stage directory naming scheme & `run_metadata.json` keys (`memory.memory_num_tokens`, etc.).
- Contract of model forward returning `new_memory_state` when memory enabled.

## 11. Quick Reference Snippets
Inject memory tokens:
```python
mem_info = configure_memory_tokens(tokenizer, memory_num_tokens=8)
model = MemXLNetForQA(base_model, mem_token_count=8, memory_update='gated')
```
Time-step-major iteration (conceptual):
```python
for time_step_batches in train_dataloader:
    loss = trainer.train_one_document_batch(time_step_batches)
```

## 12. Future-Safe Hooks (Placeholders You Can Extend)
- `_freeze_base_transformer` already abstracts freezing logic.
- `_update_memory` in wrapper is single dispatch point for new update rules.
- `extract_predictions_from_logits` houses answer assembly logic; add alternative aggregation strategies here.

Keep instructions concise—focus contributions on memory system correctness, batching integrity, and reproducible config evolution.
