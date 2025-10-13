# MemXLNet-QA: AI Coding Agent Instructions

## Project Overview

MemXLNet-QA is a **memory-augmented XLNet for long-context question answering**. It extends base XLNet with explicit memory tokens and time-step-major batching to handle documents that exceed single-segment context windows.

**Core Innovation**: Memory tokens (`[MEM_READ_i]`, `[MEM_WRITE_i]`) enable recurrent state propagation across document segments, combined with progressive curriculum training.

## Architecture Fundamentals

### 1. XLNet-Specific Constraints (CRITICAL)

**XLNet places CLS token at END of sequence** (position 383 for 384-length sequences), NOT at position 0 like BERT.
- Hardcoding `cls_index = 0` causes 40+ point F1 drops
- Always use batch-provided `cls_index` from preprocessing: `chunk.get("cls_index", None)`
- See `docs/technical/CLS_INDEX_FIX.md` for details and `src/memxlnet/data/dataset.py:214` for computation

**Left-padding**: XLNet uses left-padding, affecting memory token placement and attention mask calculations.

### 2. Time-Step-Major Batching

Unlike standard document-major batching, segments are batched by position to enable memory flow:

```python
# CORRECT: Time-step-major (all doc's segment 0, then all doc's segment 1)
batch_0 = [doc1_seg0, doc2_seg0, doc3_seg0]  # memory flows through
batch_1 = [doc1_seg1, doc2_seg1, doc3_seg1]  # using states from batch_0
```

Implementation: `src/memxlnet/data/dataset.py:MemoryCollateConfig` and `create_dataloader()`

### 3. Memory Token System

Two implementations available:
- **Token-based** (default): Explicit `[MEM_READ_i]` / `[MEM_WRITE_i]` tokens in vocabulary
- **Differentiable**: Content-based addressing with multi-head attention (experimental)

Memory tokens are inserted into sequences:
```python
# Context + WRITE tokens + SEP + READ tokens + Question + SEP + CLS
[PAD]... context [MEM_WRITE_0]...[MEM_WRITE_N] [SEP] [MEM_READ_0]...[MEM_READ_N] question [SEP] [CLS]
```

See `docs/guides/MEMORY_TOKENS_GUIDE.md` for full details.

## Development Workflows

### Setup & Environment

```bash
uv sync                           # Install dependencies
make test                         # Run test suite
make lint                         # Format and lint code
```

**Package manager**: Uses `uv` (not pip). All commands run with `uv run python ...`

### Training Pipeline

**Phase-based training** (recommended approach):
1. **Phase 1**: Single-segment warmup (baseline model)
2. **Phase 2**: Multi-segment with memory enabled

```bash
# Phase 2 training (main configuration)
uv run python scripts/phase2_train.py

# Configuration in scripts/phase2_train.py:
# - progressive_segments=[2, 4, 6]  # Curriculum learning
# - memory_num_tokens=8             # Memory slots
# - memory_update="gated"           # Gated updates for stability
# - warmup_disable_global_softmax_epochs=1  # Local predictions first
```

Key training parameters in `src/memxlnet/training/trainer.py:TrainingConfig`:
- `progressive_segments`: Curriculum learning schedule `[2, 4, 6]` trains on increasing segment counts
- `warmup_freeze_base_epochs`: Freeze base XLNet initially (phase control)
- `warmup_disable_global_softmax_epochs`: Delay global span prediction
- `use_any_positive_logic`: SQuAD v2 no-answer handling

### Evaluation Workflow

```bash
# Evaluate from local checkpoint
uv run python scripts/evaluate.py outputs/xlnet-squad-phase2-1/training_config.json

# Evaluate from HuggingFace Hub
uv run python scripts/evaluate_cls_fix.py --model-id anhtu12st/memxlnet-squad-phase2-mem16
```

### HuggingFace Hub Integration

**Automatic push during training**:
```python
config = TrainingConfig(
    hub_model_id="username/memxlnet-squad-phase2-mem16",
    push_to_hub_on_save=True,
    hub_strategy="best_only",  # Only push best models
    hub_token=None,  # Uses HF_TOKEN env var
)
```

**Manual upload**:
```bash
export HF_TOKEN='your_token'
python scripts/upload_checkpoint_to_hub.py \
    --checkpoint-path outputs/my-model/best_model \
    --hub-id username/memxlnet-squad-phase2-mem16
```

**Hub naming convention**: `{username}/memxlnet-squad-{variant}`
- `phase2-mem16`: Phase 2 trained, 16 memory tokens
- `phase2-mem8`: Phase 2 trained, 8 memory tokens
- `baseline`: No memory tokens

### Paper Experiments

Structured experiment framework in `scripts/paper_experiments/`:

```bash
# Run all 10 experiments + evaluation + analysis
./scripts/run_all_experiments.sh

# Individual experiments
uv run python scripts/paper_experiments/02_main_memory_8tokens.py
```

**Experiment types**:
- 01: Baseline (no memory)
- 02-03: Main contributions (8/16 tokens)
- 04-07: Ablations (gating, progressive training, token counts)
- 08-10: Segment analysis (2/4/6 segments)

Results saved to `results/paper_experiments/` with automatic table/figure generation.

## Code Patterns & Conventions

### Data Processing

**Unicode normalization** is critical for robust char-to-token alignment:
```python
# All text normalized with NFD before tokenization
import unicodedata
text = unicodedata.normalize('NFD', text)
```

**Answer span mapping** across segments with validation:
```python
# Dataset preprocessing validates answer positions
# See src/memxlnet/data/dataset.py:_process_example()
```

**Caching**: Preprocessed datasets cached to `.cache/` or `preprocessed_data/` for reuse:
```python
# Preprocessed chunked format for large datasets
python scripts/preprocess_datasets_chunked.py --dataset squad_v2
```

### Model Loading

```python
# Always use MemXLNetForQA.from_pretrained() - handles both Hub and local paths
from memxlnet.models import MemXLNetForQA
model = MemXLNetForQA.from_pretrained("path/to/checkpoint")  # or "username/model-id"
```

### Testing

Pytest markers for selective test runs:
```bash
pytest -m unit           # Fast unit tests only
pytest -m integration    # Integration tests
pytest -m "not slow"     # Skip slow tests
```

Test fixtures in `tests/fixtures/` provide sample data.

## Common Pitfalls

1. **CLS Position**: Never hardcode CLS position - always get from batch or search in sequence
2. **Memory State Reset**: Must reset memory between documents, not between segments
3. **Progressive Segments**: Training must handle variable segment counts per batch
4. **Answer Mapping**: With doc_stride, answers can appear in multiple segments (handled in postprocessing)
5. **Global Softmax**: Disable initially via `warmup_disable_global_softmax_epochs` for stability

## Key Files Reference

- **Main model**: `src/memxlnet/models/memxlnet_qa.py` - MemXLNetForQA implementation
- **Training loop**: `src/memxlnet/training/trainer.py` - XLNetRecurrentTrainer
- **Data pipeline**: `src/memxlnet/data/dataset.py` - Time-step-major batching logic
- **Memory modules**: `src/memxlnet/models/memory_modules.py` - Differentiable memory
- **Evaluation**: `src/memxlnet/evaluation/evaluator.py` - SQuAD v2 metrics

## Documentation Structure

Essential docs (always check before implementing):
- `docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md` - Comprehensive training/eval examples  
- `docs/api/API_REFERENCE.md` - Complete API documentation
- `docs/technical/DATA_PROCESSING.md` - Data pipeline deep dive
- `docs/guides/MEMORY_TOKENS_GUIDE.md` - Memory system internals
- `docs/guides/TESTING_VALIDATION_GUIDE.md` - Debugging strategies

Quick reference: `CLAUDE.md` (this file provides high-level overview only)
