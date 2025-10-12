# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

MemXLNet-QA is a memory-augmented XLNet for long-context question answering. Extends base XLNet with memory tokens and time-step-major batching.

## 📚 Documentation

> **⚠️ IMPORTANT:** Always check documentation for details. CLAUDE.md provides quick reference only.

**Core Docs:**
- **[API Reference](docs/api/API_REFERENCE.md)** - Complete API documentation
- **[Usage Guide](docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md)** - Training and evaluation examples
- **[Implementation Details](docs/technical/MA_XLNET_IMPLEMENTATION.md)** - Architecture details

**Specialized:**
- **[Memory Tokens Guide](docs/guides/MEMORY_TOKENS_GUIDE.md)** - Memory system details
- **[Testing Guide](docs/guides/TESTING_VALIDATION_GUIDE.md)** - Testing and troubleshooting
- **[Planned Features](docs/PLANNED_FEATURES.md)** - Feature status and roadmap

## Directory Structure

```
paper-revise/
├── src/memxlnet/          # Main package
├── scripts/               # Training, evaluation, Hub upload scripts
├── tests/                 # Test suite
├── docs/                  # Documentation
└── examples/              # Usage examples
```

## Essential Commands

```bash
# Setup
uv sync

# Training (see Usage Guide for details)
uv run python scripts/phase2_train.py

# Evaluation from Hub (recommended)
uv run python scripts/evaluate_cls_fix.py --model-id username/memxlnet-squad-phase2-mem16

# Upload checkpoint to Hub
export HF_TOKEN='your_token'
python scripts/upload_checkpoint_to_hub.py \
    --checkpoint-path outputs/my-model \
    --hub-id username/memxlnet-squad-phase2-mem16

# Tests
uv run pytest tests/
```

**For detailed commands:** See [Usage Guide](docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md)

## Critical Implementation Notes

### 1. CLS Token Position (CRITICAL)
- XLNet places CLS at **END** of sequence, not position 0
- Hardcoded position causes 40+ point F1 drops
- **Fixed in:** `src/memxlnet/data/dataset.py`
- **Details:** See [Testing Guide](docs/guides/TESTING_VALIDATION_GUIDE.md)

### 2. Memory Systems
- **Token-based:** Explicit `[MEM_READ_i]` and `[MEM_WRITE_i]` tokens
- **Differentiable:** Content-based addressing with multi-head attention (new)
- **Details:** See [Memory Tokens Guide](docs/guides/MEMORY_TOKENS_GUIDE.md)

### 3. Time-Step-Major Batching
- Required for memory propagation across segments
- Batches segments by position, not document
- **Details:** See [Data Processing](docs/technical/DATA_PROCESSING.md)

### 4. Progressive Training
- Curriculum learning with increasing segment counts
- Example: `progressive_segments=[2, 4, 6]`
- **Details:** See [Implementation Details](docs/technical/MA_XLNET_IMPLEMENTATION.md)

### 5. Phase-2 Warmup Controls
- `warmup_freeze_base_epochs=1` - Freeze base model initially
- `warmup_disable_global_softmax_epochs=1` - Local predictions first
- `warmup_disable_any_positive_epochs=1` - Simple extraction first
- **Details:** See [API Reference](docs/api/API_REFERENCE.md)

## Configuration

```python
from memxlnet.training import TrainingConfig

config = TrainingConfig(
    memory_num_tokens=16,
    progressive_segments=[2, 4],
    num_epochs=3,
    hub_model_id="username/model-name",  # Optional: auto-push to Hub
    hub_dataset_id="username/memxlnet-squad-mem16",  # Optional: use Hub data
)
```

**All options:** See [API Reference](docs/api/API_REFERENCE.md)

## HuggingFace Hub Integration

### Setup
```bash
export HF_TOKEN='your_token_here'
```

### Naming Conventions
- **Datasets:** `{username}/memxlnet-squad-mem{N}` (e.g., `mem16`)
- **Models:** `{username}/memxlnet-squad-{variant}` (e.g., `phase2-mem16`)
- **Revisions:** `stage-1-segs-1`, `best-model`, `v1.0`

### Upload Checkpoint
```bash
python scripts/upload_checkpoint_to_hub.py \
    --checkpoint-path outputs/my-model \
    --hub-id username/memxlnet-squad-phase2-mem16 \
    --revision stage-1-segs-1
```

### Evaluate from Hub
```bash
python scripts/evaluate_cls_fix.py \
    --model-id username/memxlnet-squad-phase2-mem16 \
    --revision stage-1-segs-1
```

### Load from Hub
```python
from memxlnet.models import MemXLNetForQA
model = MemXLNetForQA.from_pretrained("username/memxlnet-squad")
```

**Details:** See [Usage Guide](docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md) and [API Reference](docs/api/API_REFERENCE.md)

## Common Tasks

| Task | Documentation |
|------|---------------|
| Training | [Usage Guide](docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md) |
| Evaluation | [Usage Guide](docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md) |
| Memory configuration | [Memory Tokens Guide](docs/guides/MEMORY_TOKENS_GUIDE.md) |
| Architecture | [Implementation Details](docs/technical/MA_XLNET_IMPLEMENTATION.md) |
| Troubleshooting | [Testing Guide](docs/guides/TESTING_VALIDATION_GUIDE.md) |
| API reference | [API Reference](docs/api/API_REFERENCE.md) |
| Feature status | [Planned Features](docs/PLANNED_FEATURES.md) |

## Development

### Tests
```bash
pytest tests/              # All tests
pytest tests/unit/         # Unit tests only
```

### Common Issues
- **Import errors:** `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`
- **OOM errors:** Reduce batch size or memory tokens
- **Cache issues:** `rm -rf cache_* .cache_*`
- **Model loading:** Always use `MemXLNetForQA.from_pretrained()`

**Troubleshooting:** See [Testing Guide](docs/guides/TESTING_VALIDATION_GUIDE.md)

## Dependencies

- `torch>=2.8.0`, `transformers>=4.56.2`, `huggingface_hub>=0.20.0`
- See `pyproject.toml` for full list

## Important Notes

1. Use `scripts/phase2_train.py` for training
2. GPU recommended (16GB+ VRAM)
3. Memory tokens must match when loading models
4. Time-step-major batching required for memory-enabled evaluation

## Resources

- **[docs/](docs/)** - Complete documentation
- **[examples/](examples/)** - Usage examples
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
