# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemXLNet-QA is a memory-augmented XLNet implementation for long-context question answering. It extends base XLNet with explicit memory tokens and time-step-major batching for improved performance on multi-segment documents.

## 📚 Documentation Map

> **⚠️ Note:** Some documentation describes planned features. See **[docs/PLANNED_FEATURES.md](docs/PLANNED_FEATURES.md)** for status.

**Essential Reading:**
- **[API Reference](docs/api/API_REFERENCE.md)** - Complete API documentation ✅ Verified
- **[Usage Guide](docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md)** - Usage examples (includes planned features)
- **[Implementation Details](docs/technical/MA_XLNET_IMPLEMENTATION.md)** - Technical architecture

**Specialized Topics:**
- **[Memory Tokens Guide](docs/guides/MEMORY_TOKENS_GUIDE.md)** - Memory system deep dive ✅ Verified
- **[Streaming Guide](docs/guides/STREAMING_GUIDE.md)** - Memory-efficient data processing ✅ New
- **[Data Processing](docs/technical/DATA_PROCESSING.md)** - Data pipeline details
- **[Testing Guide](docs/guides/TESTING_VALIDATION_GUIDE.md)** - Testing and validation
- **[Quick Reference](docs/guides/ENHANCED_MA_XLNET_QUICK_REFERENCE.md)** - Developer cheat sheet
- **[Planned Features](docs/PLANNED_FEATURES.md)** - Roadmap and feature status 🆕

## Directory Structure

```
paper-revise/
├── src/memxlnet/          # Main package (models, data, training, evaluation, utils)
├── scripts/               # Executable scripts
│   ├── train.py                       # Basic training
│   ├── evaluate.py                    # Model evaluation
│   ├── evaluate_cls_fix.py            # Hub-first evaluation with CLS fix
│   ├── phase2_train.py                # Phase 2 training (recommended)
│   ├── train_memxlnet_squad.py        # Hub-integrated training
│   ├── preprocess_and_upload_to_hub.py  # Dataset preprocessing for Hub
│   └── upload_checkpoint_to_hub.py    # Upload checkpoints to Hub
├── tests/                 # Test suite (unit/, integration/)
├── docs/                  # Documentation (api/, guides/, technical/)
├── notebooks/             # Interactive analysis notebooks
├── examples/              # Usage examples
├── pyproject.toml         # Package configuration
└── CLAUDE.md             # This file
```

## Essential Commands

### Environment Setup
```bash
# Create virtual environment and install
uv sync
```

### Basic Training
```bash
# Phase 2 training (recommended)
uv run python scripts/phase2_train.py

# Quick smoke test
uv run python -c "
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer
config = TrainingConfig(max_train_samples=10, num_epochs=1)
trainer = XLNetRecurrentTrainer(config)
print('✅ Trainer initialized')
"
```

### Basic Evaluation
```bash
# Evaluate from HuggingFace Hub (recommended for cross-server reproducibility)
export HF_TOKEN='your_token'  # if model is private
uv run python scripts/evaluate_cls_fix.py --model-id username/memxlnet-squad-phase2-mem16

# Quick test with subset (100 examples)
uv run python scripts/evaluate_cls_fix.py --model-id username/memxlnet-squad-phase2-mem16 --test-size 100

# Evaluate local checkpoint (legacy)
uv run python scripts/evaluate.py outputs/xlnet-squad-phase2-1/training_config.json

# Run test suite
uv run pytest tests/
```

### HuggingFace Hub Integration
```bash
# ONE-TIME: Preprocess and upload dataset to Hub (requires 20GB+ RAM)
# Edit scripts/preprocess_and_upload_to_hub.py to set your HUB_USERNAME first
export HF_TOKEN='your_huggingface_token'
uv run python scripts/preprocess_and_upload_to_hub.py

# FAST TRAINING: Use Hub-preprocessed datasets (requires only 4-6GB RAM)
# Edit scripts/train_memxlnet_squad.py to set your HUB_USERNAME first
uv run python scripts/train_memxlnet_squad.py
# This downloads preprocessed data from Hub and starts training in minutes!
```

### Verify Installation
```bash
# Test all imports
uv run python -c "
from memxlnet import __version__
from memxlnet.models import MemXLNetForQA
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer
from memxlnet.data import create_dataloader, configure_memory_tokens
print('✅ All imports successful')
"
```

## Critical Implementation Notes

### 1. CLS Token Position Bug Fix (January 2025)
**CRITICAL:** XLNet places CLS token at the END of sequences, not position 0.
- **Bug**: Hardcoded CLS position caused 40+ point F1 drops
- **Fix**: Use actual CLS index from batch data
- **Status**: ✅ FIXED in `src/memxlnet/data/dataset.py`
- If you see 0% no-answer predictions or low F1, verify CLS handling

### 2. Differentiable Memory System (✅ NEW - January 2025)
**Alternative to token-based memory** with content-based addressing and multi-head attention:
- **Status**: ✅ Fully implemented and tested (Phase 1 complete)
- **Features**: Multi-head attention (1-8 heads), usage tracking, temporal links
- **Test Coverage**: 39 tests (26 unit + 13 integration)
- **Example**: `examples/train_with_differentiable_memory.py`

**Quick Start:**
```python
from memxlnet.models import MemXLNetForQA

model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=16,
    use_differentiable_memory=True,  # Enable differentiable memory
    num_memory_heads=4,               # Multi-head attention
    memory_sharpness=2.0,             # Attention sharpening
    enable_usage_tracking=True,       # Track slot usage
    enable_temporal_links=True,       # Track relationships
    memory_slots=32,                  # Number of memory slots
)

# Access memory information in outputs
outputs = model(**inputs)
if "memory_info" in outputs:
    memory_info = outputs["memory_info"]
    # Contains: read_weights, write_weights, memory_state, usage, temporal_links
```

**See**: [Planned Features](docs/PLANNED_FEATURES.md) for details and Phase 2 roadmap

### 3. Memory Token System (Traditional)
Memory uses explicit special tokens:
- `[MEM_READ_i]` - Read from memory (injected at segment start)
- `[MEM_WRITE_i]` - Write to memory (positioned in sequence)
- Tokens automatically added to tokenizer vocabulary
- See [Memory Tokens Guide](docs/guides/MEMORY_TOKENS_GUIDE.md) for details

### 4. Time-Step-Major Batching
Required for memory propagation across document segments:
- Standard: `[doc1_seg1, doc1_seg2] → [doc2_seg1, doc2_seg2]`
- Time-step-major: `[doc1_seg1, doc2_seg1] → [doc1_seg2, doc2_seg2]`
- Implemented in `TimeStepMajorDataLoader`
- See [Data Processing](docs/technical/DATA_PROCESSING.md) for details

### 5. Progressive Training
Curriculum learning with increasing segment counts:
- Example: `progressive_segments=[2, 4, 6]` trains 2→4→6 segments
- Creates subdirectories: `stage_1_segs_2/`, `stage_2_segs_4/`, etc.
- See [Implementation Details](docs/technical/MA_XLNET_IMPLEMENTATION.md)

### 6. Phase-2 Warmup Controls
Fine-grained control over model warmup for stable memory-augmented training:

#### `warmup_freeze_base_epochs` (default: 1)
Freeze the base XLNet transformer for the first N epochs while training only memory components.
- **Purpose**: Allows memory modules to learn without disrupting pretrained weights
- **Recommended**: 1-2 epochs for stable initialization
- **Example**: `warmup_freeze_base_epochs=1` freezes base for first epoch

#### `warmup_disable_global_softmax_epochs` (default: 1)
Disable document-level answer span aggregation for the first N epochs.
- **Purpose**: Forces model to learn local (per-segment) predictions first
- **Recommended**: 1 epoch before enabling global reasoning
- **Example**: `warmup_disable_global_softmax_epochs=1` uses segment-level softmax initially

#### `warmup_disable_any_positive_epochs` (default: 1)
Disable "any-positive" extraction logic for the first N epochs.
- **Purpose**: Simplifies training by using single-span extraction initially
- **Recommended**: 1 epoch for gradual complexity increase
- **Example**: `warmup_disable_any_positive_epochs=1` starts with simple extraction

**Typical Usage:**
```python
config = TrainingConfig(
    memory_num_tokens=16,
    num_epochs=5,

    # Warmup controls for stable training
    warmup_freeze_base_epochs=1,           # Freeze base for epoch 1
    warmup_disable_global_softmax_epochs=1,  # Local softmax for epoch 1
    warmup_disable_any_positive_epochs=1,    # Simple extraction for epoch 1
)
```

**Training Timeline:**
- **Epoch 1**: Base frozen, segment-level predictions, simple extraction
- **Epoch 2+**: All features enabled, full model training

## Key Configuration Parameters

```python
from memxlnet.training import TrainingConfig

config = TrainingConfig(
    # Memory system
    memory_num_tokens=16,        # Number of memory tokens (0=disabled)
    memory_update="gated",       # Update mechanism: "gated", "none"
    memory_init="learned",       # Initialization: "learned", "zeros"

    # Progressive training
    progressive_segments=[2, 4], # Curriculum learning stages
    max_n_segs=6,               # Max segments per document

    # Training
    num_epochs=3,
    train_batch_size=16,
    learning_rate=3e-5,

    # Evaluation
    use_global_softmax=True,    # Global span selection
    no_answer_threshold=1.5,    # SQuAD v2 threshold

    # HuggingFace Hub integration - Models (optional)
    hub_model_id="username/model-name",  # Hub repository ID for models
    push_to_hub_on_save=True,            # Auto-push models to Hub
    hub_strategy="best_only",            # "best_only", "every_save", "end"

    # HuggingFace Hub integration - Datasets (optional)
    hub_dataset_id="username/memxlnet-squad-mem16",  # Preprocessed dataset repo
    use_hub_dataset=True,                # Download preprocessed data from Hub
    force_reprocess=False,               # Skip reprocessing if Hub data exists
)
```

**Full configuration options:** See [API Reference](docs/api/API_REFERENCE.md)

## HuggingFace Hub Integration

MemXLNet supports automatic model versioning **and** preprocessed dataset sharing via HuggingFace Hub.

### Setup
```bash
# Set your HuggingFace token
export HF_TOKEN='your_token_here'

# Or add to .env file
echo "HF_TOKEN=your_token_here" >> .env
```

### Training with Hub Integration
```python
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

config = TrainingConfig(
    # Your training settings...
    memory_num_tokens=16,
    num_epochs=3,

    # Hub configuration
    hub_model_id="username/memxlnet-squad",  # Your Hub repository
    push_to_hub_on_save=True,                # Enable auto-push
    hub_strategy="best_only",                # Only push best models
    hub_private=False,                       # Public repository
)

trainer = XLNetRecurrentTrainer(config)
trainer.train()  # Automatically pushes best models to Hub
```

### Training with Hub Datasets

Download preprocessed datasets for faster training startup:

```python
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

config = TrainingConfig(
    # Your training settings...
    memory_num_tokens=16,
    num_epochs=3,

    # Hub dataset configuration (FAST: 2-5 min startup vs 30-60 min preprocessing)
    hub_dataset_id="username/memxlnet-squad-mem16",  # Preprocessed data repo
    use_hub_dataset=True,                            # Download from Hub
    force_reprocess=False,                           # Use Hub data if available

    # Hub model configuration (optional)
    hub_model_id="username/memxlnet-squad",          # Model output repo
    push_to_hub_on_save=True,                        # Auto-push models
    hub_strategy="best_only",
)

trainer = XLNetRecurrentTrainer(config)
trainer.train()  # Fast startup with preprocessed Hub data!
```

**Benefits:**
- **Faster startup**: 2-5 minutes vs 30-60 minutes for preprocessing
- **Lower RAM**: 4-6GB vs 20-30GB for preprocessing
- **Reproducibility**: Same preprocessed data across all runs

**See:** `scripts/preprocess_and_upload_to_hub.py` for uploading datasets
**See:** `scripts/train_memxlnet_squad.py` for complete example

### Hub Push Strategies
- **`best_only`**: Only push when a new best model is found (recommended)
- **`every_save`**: Push every checkpoint (includes periodic saves)
- **`end`**: Only push final model at end of training

### Loading from Hub
```python
from memxlnet.models import MemXLNetForQA
from transformers import XLNetTokenizerFast

# Load directly from Hub
model = MemXLNetForQA.from_pretrained("username/memxlnet-squad")
tokenizer = XLNetTokenizerFast.from_pretrained("username/memxlnet-squad")
```

### HuggingFace Naming Conventions

MemXLNet follows standardized naming patterns for Hub repositories:

**Dataset repositories:**
```
{username}/memxlnet-squad-mem{N}

Examples:
- anhtu12st/memxlnet-squad-mem0   (no memory tokens)
- anhtu12st/memxlnet-squad-mem8   (8 memory tokens)
- anhtu12st/memxlnet-squad-mem16  (16 memory tokens)
```

**Model repositories:**
```
{username}/memxlnet-squad-{variant}

Common variants:
- memxlnet-squad-phase2-mem16  (Phase 2 trained, 16 memory tokens)
- memxlnet-squad-phase2-mem8   (Phase 2 trained, 8 memory tokens)
- memxlnet-squad-baseline      (no memory tokens)
- memxlnet-squad-cls-fixed     (with CLS position bug fix)
```

**Revisions/Tags within repositories:**
```
- best-model     (default, latest best checkpoint)
- stage-1-segs-1 (training stage checkpoints)
- stage-1-segs-2
- v1.0, v2.0     (version tags)
```

### Upload Checkpoints to Hub

Upload local checkpoints for cross-server evaluation:

```bash
# Upload checkpoint to Hub
export HF_TOKEN='your_token'
python scripts/upload_checkpoint_to_hub.py \
    --checkpoint-path outputs/xlnet-squad-phase2-1/stage_1_segs_1/best_model \
    --hub-id anhtu12st/memxlnet-squad-phase2-mem16 \
    --revision stage-1-segs-1

# Dry run (validate without uploading)
python scripts/upload_checkpoint_to_hub.py \
    --checkpoint-path outputs/my-model/best_model \
    --hub-id username/memxlnet-squad-phase2-mem16 \
    --dry-run

# Create private repository
python scripts/upload_checkpoint_to_hub.py \
    --checkpoint-path outputs/my-model/best_model \
    --hub-id username/memxlnet-squad-phase2-mem16 \
    --private
```

### Hub-First Evaluation

Evaluate models directly from HuggingFace Hub (recommended for reproducibility):

```bash
# Full dataset evaluation
python scripts/evaluate_cls_fix.py \
    --model-id anhtu12st/memxlnet-squad-phase2-mem16

# Specific revision
python scripts/evaluate_cls_fix.py \
    --model-id anhtu12st/memxlnet-squad-phase2-mem16 \
    --revision stage-1-segs-1

# Quick test (100 examples)
python scripts/evaluate_cls_fix.py \
    --model-id anhtu12st/memxlnet-squad-phase2-mem16 \
    --test-size 100

# Custom output directory
python scripts/evaluate_cls_fix.py \
    --model-id anhtu12st/memxlnet-squad-phase2-mem16 \
    --output-dir ./my_results
```

**Benefits of Hub-first evaluation:**
- ✅ Works on any server without local checkpoints
- ✅ Reproducible across environments
- ✅ Versioned via Hub revisions/tags
- ✅ No large files in git

### Manual Upload (Alternative)
Use the provided notebook for interactive uploads:
```bash
jupyter notebook notebooks/upload_to_huggingface.ipynb
```

## Common Tasks → Documentation

| Task | Documentation |
|------|---------------|
| Train a model | [Usage Guide](docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md) |
| Evaluate from Hub | `scripts/evaluate_cls_fix.py --model-id username/model` (see above) |
| Upload checkpoint to Hub | `scripts/upload_checkpoint_to_hub.py` (see Hub Integration section) |
| Configure memory | [Memory Tokens Guide](docs/guides/MEMORY_TOKENS_GUIDE.md) |
| Stream large datasets | [Streaming Guide](docs/guides/STREAMING_GUIDE.md) |
| Hub integration | [API Reference](docs/api/API_REFERENCE.md#hub-integration) |
| Understand architecture | [Implementation Details](docs/technical/MA_XLNET_IMPLEMENTATION.md) |
| Debug evaluation | [Testing Guide](docs/guides/TESTING_VALIDATION_GUIDE.md) |
| Process data | [Data Processing](docs/technical/DATA_PROCESSING.md) |
| API reference | [API Reference](docs/api/API_REFERENCE.md) |
| Quick examples | [Quick Reference](docs/guides/ENHANCED_MA_XLNET_QUICK_REFERENCE.md) |
| Check feature status | [Planned Features](docs/PLANNED_FEATURES.md) |

## Development Workflow

### Running Tests
```bash
pytest tests/                    # All tests
pytest tests/unit/               # Unit tests
pytest tests/integration/        # Integration tests
pytest tests/regression/         # Regression tests
```

### Common Issues

**Import errors:** Add project to PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Memory/OOM errors:** Reduce batch size or use fewer memory tokens
```python
config = TrainingConfig(train_batch_size=8, memory_num_tokens=8)
```

**Cache issues:** Clear cache directories
```bash
rm -rf cache_* .cache_*
```

**Model loading:** Always use `MemXLNetForQA.from_pretrained()`
```python
from memxlnet.models import MemXLNetForQA
model = MemXLNetForQA.from_pretrained("path/to/checkpoint")
```

**Detailed troubleshooting:** See [Testing Guide](docs/guides/TESTING_VALIDATION_GUIDE.md)

## Package Structure

The package follows standard Python conventions:
- **Main package**: `src/memxlnet/` (installed as `memxlnet`)
- **Import pattern**: `from memxlnet.models import MemXLNetForQA`
- **Scripts**: Executable entry points in `scripts/`
- **Tests**: Organized by type (unit/integration/regression)

## Dependencies

Core dependencies (from `pyproject.toml`):
- `torch>=2.8.0` - PyTorch framework
- `transformers>=4.56.2` - HuggingFace transformers
- `huggingface_hub>=0.20.0` - HuggingFace Hub integration
- `datasets>=4.1.1` - Dataset loading
- `matplotlib>=3.10.6` - Visualization

Optional:
- `pytest>=7.4.0` - Testing (install with `pip install -e ".[dev]"`)

## Important Notes

1. **Phase 2 training is recommended** - Use `scripts/phase2_train.py` for best results
2. **Memory tokens must match** - Ensure tokenizer has memory tokens when loading models
3. **Time-step-major batching required** - For memory-enabled evaluation
4. **Configuration is saved** - `training_config.json` saved for evaluation reproducibility
5. **GPU recommended** - Training is memory-intensive, 16GB+ VRAM recommended

## Quick Examples

### Minimal Training
```python
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

config = TrainingConfig(
    memory_num_tokens=16,
    max_train_samples=1000,
    num_epochs=2
)
trainer = XLNetRecurrentTrainer(config)
trainer.train()
```

### Load and Evaluate
```python
from memxlnet.models import MemXLNetForQA
from transformers import XLNetTokenizerFast

# Load from local checkpoint
model = MemXLNetForQA.from_pretrained("outputs/my-model")
tokenizer = XLNetTokenizerFast.from_pretrained("outputs/my-model")

# Or load from HuggingFace Hub
model = MemXLNetForQA.from_pretrained("username/memxlnet-squad")
tokenizer = XLNetTokenizerFast.from_pretrained("username/memxlnet-squad")

# See docs for full evaluation pipeline
```

## Additional Resources

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[docs/README.md](docs/README.md)** - Documentation index
- **[examples/](examples/)** - Working code examples
- **[notebooks/](notebooks/)** - Interactive analysis notebooks
