# CLAUDE.md - Developer Quick Reference

This file provides quick guidance for Claude Code when working with this repository.

## Project Overview

MemXLNet-QA is a memory-augmented XLNet implementation for long-context question answering. It extends base XLNet with explicit memory tokens and time-step-major batching for improved performance on multi-segment documents.

## 📚 Documentation Map

**Essential Reading:**
- **[API Reference](docs/api/API_REFERENCE.md)** - Complete API documentation
- **[Usage Guide](docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md)** - Comprehensive usage examples
- **[Implementation Details](docs/technical/MA_XLNET_IMPLEMENTATION.md)** - Technical architecture

**Specialized Topics:**
- **[Memory Tokens Guide](docs/guides/MEMORY_TOKENS_GUIDE.md)** - Memory system deep dive
- **[Data Processing](docs/technical/DATA_PROCESSING.md)** - Data pipeline details
- **[Testing Guide](docs/guides/TESTING_VALIDATION_GUIDE.md)** - Testing and validation
- **[Quick Reference](docs/guides/ENHANCED_MA_XLNET_QUICK_REFERENCE.md)** - Developer cheat sheet

## Directory Structure

```
paper-revise/
├── src/memxlnet/          # Main package (models, data, training, evaluation, utils)
├── scripts/               # Executable scripts (train.py, evaluate.py, phase2_train.py)
├── tests/                 # Test suite (unit/, integration/, regression/)
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
uv venv
source .venv/bin/activate  # macOS/Linux
uv pip install -e .
```

### Basic Training
```bash
# Phase 2 training (recommended)
python scripts/phase2_train.py

# Quick smoke test
python -c "
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer
config = TrainingConfig(max_train_samples=10, num_epochs=1)
trainer = XLNetRecurrentTrainer(config)
print('✅ Trainer initialized')
"
```

### Basic Evaluation
```bash
# Evaluate trained model
python scripts/evaluate.py outputs/xlnet-squad-phase2-1/training_config.json

# Run test suite
pytest tests/
```

### Verify Installation
```bash
# Test all imports
python -c "
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

### 2. Memory Token System
Memory uses explicit special tokens:
- `[MEM_READ_i]` - Read from memory (injected at segment start)
- `[MEM_WRITE_i]` - Write to memory (positioned in sequence)
- Tokens automatically added to tokenizer vocabulary
- See [Memory Tokens Guide](docs/guides/MEMORY_TOKENS_GUIDE.md) for details

### 3. Time-Step-Major Batching
Required for memory propagation across document segments:
- Standard: `[doc1_seg1, doc1_seg2] → [doc2_seg1, doc2_seg2]`
- Time-step-major: `[doc1_seg1, doc2_seg1] → [doc1_seg2, doc2_seg2]`
- Implemented in `TimeStepMajorDataLoader`
- See [Data Processing](docs/technical/DATA_PROCESSING.md) for details

### 4. Progressive Training
Curriculum learning with increasing segment counts:
- Example: `progressive_segments=[2, 4, 6]` trains 2→4→6 segments
- Creates subdirectories: `stage_1_segs_2/`, `stage_2_segs_4/`, etc.
- See [Implementation Details](docs/technical/MA_XLNET_IMPLEMENTATION.md)

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

    # HuggingFace Hub integration (optional)
    hub_model_id="username/model-name",  # Hub repository ID
    push_to_hub_on_save=True,            # Auto-push to Hub
    hub_strategy="best_only",            # "best_only", "every_save", "end"
)
```

**Full configuration options:** See [API Reference](docs/api/API_REFERENCE.md)

## HuggingFace Hub Integration

MemXLNet supports automatic model versioning and sharing via HuggingFace Hub.

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

### Manual Upload
Use the provided notebook for one-time uploads:
```bash
jupyter notebook notebooks/upload_to_huggingface.ipynb
```

## Common Tasks → Documentation

| Task | Documentation |
|------|---------------|
| Train a model | [Usage Guide](docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md) |
| Configure memory | [Memory Tokens Guide](docs/guides/MEMORY_TOKENS_GUIDE.md) |
| Understand architecture | [Implementation Details](docs/technical/MA_XLNET_IMPLEMENTATION.md) |
| Debug evaluation | [Testing Guide](docs/guides/TESTING_VALIDATION_GUIDE.md) |
| Process data | [Data Processing](docs/technical/DATA_PROCESSING.md) |
| API reference | [API Reference](docs/api/API_REFERENCE.md) |
| Quick examples | [Quick Reference](docs/guides/ENHANCED_MA_XLNET_QUICK_REFERENCE.md) |

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
