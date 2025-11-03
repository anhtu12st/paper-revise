# MemXLNet-QA: Memory-Augmented XLNet for Long-Context QA

A memory-augmented XLNet implementation for long-form document question answering, featuring explicit memory tokens, time-step-major batching, and progressive training for improved performance on SQuAD v2 and similar datasets. Includes advanced multi-expert memory (GMM) for increased capacity and interpretability.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8+](https://img.shields.io/badge/PyTorch-2.8+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Key Features

- **Explicit Memory Tokens** - `[MEM_READ]` and `[MEM_WRITE]` tokens for controllable memory operations
- **Time-Step-Major Batching** - Novel batching strategy ensuring proper memory state propagation
- **Gated Memory Updates** - Learnable gates for stable memory updates across segments
- **Progressive Training** - Curriculum learning with increasing document complexity
- **Multi-Expert Memory (GMM)** - Advanced architecture with k independent memory experts and learned routing (NEW)
- **SQuAD v2 Ready** - Robust no-answer handling with calibrated thresholds
- **Production Ready** - Complete package with tests, documentation, and examples

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/memxlnet-qa.git
cd memxlnet-qa

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
uv pip install -e .
```

### Basic Training

```bash
# Phase 2 training (recommended for best results)
python scripts/phase2_train.py

# Quick test with minimal data
python -c "
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer
config = TrainingConfig(max_train_samples=100, num_epochs=1)
trainer = XLNetRecurrentTrainer(config)
trainer.train()
"
```

### Basic Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py outputs/xlnet-squad-phase2-1/training_config.json

# Run test suite
pytest tests/
```

## 💡 Core Concepts

### Memory Token System

MemXLNet uses explicit special tokens to interface with external memory:

```python
# Memory tokens are added to tokenizer
mem_read_tokens = ["[MEM_READ_0]", "[MEM_READ_1]", ...]   # Read from memory
mem_write_tokens = ["[MEM_WRITE_0]", "[MEM_WRITE_1]", ...] # Write to memory

# Memory state flows across document segments
memory_state = model.get_initial_memory(batch_size, device)
for segment in document_segments:
    outputs = model(segment, memory_state=memory_state)
    memory_state = outputs["new_memory_state"]
```

**Learn more:** [Memory Tokens Guide](docs/guides/MEMORY_TOKENS_GUIDE.md)

### Time-Step-Major Batching

Critical innovation for proper memory propagation:

```python
# Standard batching (memory states can't flow properly)
# [doc1_seg1, doc1_seg2] → [doc2_seg1, doc2_seg2]

# Time-step-major batching (enables memory flow)
# [doc1_seg1, doc2_seg1] → [doc1_seg2, doc2_seg2]
```

**Learn more:** [Data Processing](docs/technical/DATA_PROCESSING.md)

### Multi-Expert Memory (GMM)

Advanced feature for increased memory capacity and specialization:

```python
from gmmxlnet.models import GMMXLNetForQA
from gmmxlnet.training import gmm_balanced_config

# Train with 4 expert memories
config = gmm_balanced_config(
    num_memory_experts=4,      # k=4 expert memories
    memory_num_tokens=16,      # 16 slots per expert
    routing_temperature=1.0,   # Balanced routing
)

# Analyze expert specialization
from gmmxlnet.utils import GMMAnalyzer
analyzer = GMMAnalyzer(model=model)
routing_stats = analyzer.track_routing(eval_dataloader)
print(f"Expert Utilization: {routing_stats['expert_utilization']}")
```

**Learn more:** [GMM XLNet Guide](docs/guides/GMM_XLNET_GUIDE.md)

## 📖 Usage Example

```python
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

# Configure training
config = TrainingConfig(
    model_name="xlnet-base-cased",
    dataset_name="squad_v2",

    # Memory configuration
    memory_num_tokens=16,
    memory_update="gated",
    memory_init="learned",

    # Progressive training
    progressive_segments=[2, 4, 6],
    max_n_segs=6,

    # Training parameters
    num_epochs=3,
    train_batch_size=16,
    learning_rate=3e-5,

    output_dir="./outputs/my-model"
)

# Train model
trainer = XLNetRecurrentTrainer(config)
trainer.train()

# Evaluate model
from memxlnet.models import MemXLNetForQA
model = MemXLNetForQA.from_pretrained("./outputs/my-model")
# See evaluation docs for complete pipeline
```

## 📚 Documentation

### Essential Guides
- 📖 **[API Reference](docs/api/API_REFERENCE.md)** - Complete API documentation
- 📘 **[Usage Guide](docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md)** - Comprehensive examples and patterns
- 🎯 **[Quick Reference](docs/guides/ENHANCED_MA_XLNET_QUICK_REFERENCE.md)** - Developer cheat sheet

### Technical Deep Dives
- 🔧 **[Implementation Details](docs/technical/MA_XLNET_IMPLEMENTATION.md)** - Architecture and design
- 🗂️ **[Data Processing](docs/technical/DATA_PROCESSING.md)** - Data pipeline and batching
- 🧠 **[Memory Tokens](docs/guides/MEMORY_TOKENS_GUIDE.md)** - Memory system deep dive

### Development
- ✅ **[Testing Guide](docs/guides/TESTING_VALIDATION_GUIDE.md)** - Testing strategies and validation
- 🤝 **[Contributing](CONTRIBUTING.md)** - Contribution guidelines
- 📋 **[Developer Guide](CLAUDE.md)** - Quick reference for developers

## 🏗️ Architecture

### Core Components

```
src/
├── memxlnet/          # Base MemXLNet package
│   ├── models/        # MemXLNetForQA and memory modules
│   ├── data/          # Dataset processing and time-step-major batching
│   ├── training/      # Training orchestration and progressive training
│   ├── evaluation/    # Evaluation pipeline and metrics
│   └── utils/         # Utilities and helpers
└── gmmxlnet/          # GMM multi-expert extension
    ├── models/        # GMMXLNetForQA and expert components
    ├── training/      # GMM training configuration
    └── utils/         # GMM analysis and visualization
```

### Key Design Patterns

- **Memory Wrapper**: `MemXLNetForQA` extends base XLNet with memory capabilities
- **Time-Step-Major DataLoader**: Specialized batching for memory propagation
- **Progressive Segments**: Curriculum learning with increasing complexity
- **Configuration-Driven**: Comprehensive `TrainingConfig` with 50+ parameters

**Learn more:** [Implementation Details](docs/technical/MA_XLNET_IMPLEMENTATION.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/regression/        # Regression tests

# Verify installation
python -c "
from memxlnet import __version__
from memxlnet.models import MemXLNetForQA
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer
print('✅ All imports successful')
"
```

## 📊 Project Structure

```
paper-revise/
├── src/
│   ├── memxlnet/          # Base MemXLNet package
│   └── gmmxlnet/          # GMM multi-expert extension
├── scripts/               # Training and evaluation scripts
├── tests/                 # Comprehensive test suite
├── docs/                  # Documentation
│   ├── api/              # API documentation
│   ├── guides/           # User guides (inc. GMM guide)
│   └── technical/        # Technical docs
├── examples/              # Usage examples (inc. GMM training)
├── pyproject.toml         # Package configuration
└── README.md             # This file
```

## ⚙️ Dependencies

**Core:**
- Python 3.12+
- PyTorch 2.8+
- Transformers 4.56+
- Datasets 4.1+

**Development:**
- pytest 7.4+ (testing)
- ruff 0.1+ (linting)
- mypy 1.7+ (type checking)

Install with optional dependencies:
```bash
pip install -e ".[dev]"  # Development tools
```

## 🔬 Research Background

### Problem
Long-context question answering faces challenges:
- Quadratic attention cost with sequence length
- Limited positional encoding capacity
- Suboptimal memory utilization in segment-level processing

### Solution
MemXLNet addresses these through:
1. **Explicit Memory Tokens** - Replace implicit recurrence with controllable read/write
2. **Time-Step-Major Processing** - Ensure proper memory state flow
3. **Progressive Training** - Curriculum learning with increasing complexity
4. **Multi-Expert Memory (GMM)** - k independent expert memories for increased capacity
5. **Robust Calibration** - Conservative no-answer handling for SQuAD v2

### Results
- Strong answer quality on long documents
- Improved no-answer calibration vs. segment-independent processing
- Efficient memory usage through gated updates
- GMM enables expert specialization and increased memory capacity

## 🐛 Troubleshooting

**Import errors?**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Out of memory?**
```python
config = TrainingConfig(train_batch_size=8, memory_num_tokens=8)
```

**Model loading issues?**
```python
# Always use MemXLNetForQA.from_pretrained()
from memxlnet.models import MemXLNetForQA
model = MemXLNetForQA.from_pretrained("path/to/checkpoint")
```

**See full troubleshooting:** [Testing Guide](docs/guides/TESTING_VALIDATION_GUIDE.md)

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{memxlnet-qa,
  title={MemXLNet-QA: Memory-Augmented XLNet for Long-Context Question Answering},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📞 Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: See [docs/](docs/) for comprehensive guides
- **Examples**: Check [examples/](examples/) for working code
- **Developer Guide**: See [CLAUDE.md](CLAUDE.md) for development reference

## 🔗 Related Work

- [Transformer-XL](https://arxiv.org/abs/1901.02860) - Segment-level recurrence
- [XLNet](https://arxiv.org/abs/1906.08237) - Permutation language modeling
- [SQuAD 2.0](https://arxiv.org/abs/1806.03822) - Reading comprehension with unanswerable questions
- [Longformer](https://arxiv.org/abs/2004.05150) - Efficient attention for long documents

---

**Built with ❤️ for long-context question answering**
