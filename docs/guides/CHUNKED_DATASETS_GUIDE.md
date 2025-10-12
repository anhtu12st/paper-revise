# Chunked Datasets Guide

**GitHub-Friendly Preprocessing for Fast Experiment Startup**

## 📋 Overview

The Chunked Datasets system solves a critical problem in machine learning workflows: **preprocessing time**. Instead of waiting 30-60 minutes for data preprocessing at the start of every experiment, preprocess once and load in 2-5 minutes!

### The Problem

Running experiments in different environments (cloud instances, local machine, CI/CD) means:
- **Each run preprocesses from scratch**: 30-60 minutes per experiment
- **12 experiments = 6-12 hours wasted** on the same preprocessing
- **Large cache files** that are hard to version control or share

### The Solution

**Chunked Datasets** provide:
- ✅ **One-time preprocessing**: Preprocess once, reuse everywhere
- ✅ **Fast loading**: 2-5 minutes vs 30-60 minutes
- ✅ **GitHub-friendly**: ~50MB chunks, easy to version control
- ✅ **Streaming support**: Handle TB-scale datasets efficiently
- ✅ **Partial loading**: Test with 100 examples, train with full dataset
- ✅ **No code changes**: Drop-in replacement for existing pipeline

### Time Savings

| Scenario | Before | After | Time Saved |
|----------|--------|-------|------------|
| Single experiment | 30-60 min | 2-5 min | ~50 min |
| 12 experiments | 6-12 hours | 1-2 hours | **5-11 hours!** |
| Quick test (100 examples) | 30-60 min | Instant | ~60 min |

---

## 🚀 Quick Start

### 1. Preprocess Your Dataset (One-Time)

```bash
# Preprocess SQuAD v2 (takes 30-60 minutes, but only once!)
python scripts/preprocess_datasets_chunked.py --dataset squad_v2

# Preprocess Long SQuAD v2
python scripts/preprocess_datasets_chunked.py --dataset huutuan/long_squad_v2

# Custom settings
python scripts/preprocess_datasets_chunked.py \
    --dataset squad_v2 \
    --output-dir ./preprocessed_data \
    --chunk-size 1000 \
    --max-seq-length 384 \
    --doc-stride 64 \
    --memory-tokens 8
```

**Output Structure:**
```
preprocessed_data/
└── squad_v2/
    ├── manifest.json                 # Main manifest
    ├── train_manifest.json           # Train split manifest
    ├── validation_manifest.json      # Validation split manifest
    ├── train/
    │   ├── chunk_0000.arrow         # ~50MB each
    │   ├── chunk_0001.arrow
    │   └── ...
    └── validation/
        ├── chunk_0000.arrow
        └── ...
```

### 2. Run Experiments (Fast!)

Your existing experiment scripts already support chunked datasets! Just run them:

```bash
# Any experiment now loads in 2-5 minutes!
python scripts/paper_experiments_v2/squad/01_baseline_squad_no_memory.py
python scripts/paper_experiments_v2/long_squad/08_main_8tokens.py
```

The scripts automatically use chunked datasets when available.

### 3. Test with Sample Data

```bash
# Test with just 100 examples (instant startup!)
python scripts/test_with_samples.py --num-examples 100 --epochs 1

# Test with specific chunks
python scripts/test_with_samples.py --chunks 0 1 2

# Inspect what was preprocessed
python scripts/inspect_chunks.py ./preprocessed_data/squad_v2 --verbose
```

---

## 📦 How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PREPROCESSING (ONE-TIME)                                 │
│    scripts/preprocess_datasets_chunked.py                   │
│                                                              │
│    Raw Dataset → Process → Split into Chunks (~50MB each)   │
│                                                              │
│    Output: Arrow files + manifest.json                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. LOADING (2-5 MINUTES)                                    │
│    src/memxlnet/data/chunked_dataset.py                     │
│                                                              │
│    • Read manifest.json (instant)                           │
│    • Load Arrow chunks (fast, memory-mapped)                │
│    • 4 loading modes: streaming, first_n, chunks, full      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. TRAINING                                                  │
│    Existing pipeline unchanged!                             │
│    TimeStepMajorDataLoader → Trainer → Model               │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **`scripts/preprocess_datasets_chunked.py`**
   - Processes datasets into Arrow chunks
   - Generates manifest files with metadata
   - GitHub-friendly chunk sizes (~50MB)

2. **`src/memxlnet/data/chunked_dataset.py`**
   - `ChunkedDataset` class with 4 loading modes
   - Compatible with existing dataloaders
   - Memory-efficient streaming support

3. **Manifest Files**
   - JSON metadata for fast lookup
   - Contains chunk information, dataset stats
   - No need to load data to inspect dataset

---

## 🎯 Loading Modes

ChunkedDataset supports 4 flexible loading modes:

### 1. Streaming (Recommended for Training)

**Memory-efficient, lazy-loading on-demand:**

```python
from memxlnet.data import load_chunked_dataset

dataset = load_chunked_dataset(
    dataset_dir="./preprocessed_data/squad_v2",
    split="train",
    mode="streaming",  # Load chunks on-demand
    max_n_segs=2
)
```

**Use when:**
- Training on full dataset
- Limited memory available
- Dataset is very large

### 2. First N (Quick Testing)

**Load first N documents instantly:**

```python
dataset = load_chunked_dataset(
    dataset_dir="./preprocessed_data/squad_v2",
    split="train",
    mode="first_n",
    num_examples=1000,  # Load first 1000 docs
    max_n_segs=2
)
```

**Use when:**
- Quick debugging/testing
- Smoke tests
- Development iteration

### 3. Chunks (Specific Ranges)

**Load specific chunk indices:**

```python
dataset = load_chunked_dataset(
    dataset_dir="./preprocessed_data/squad_v2",
    split="train",
    mode="chunks",
    chunk_indices=[0, 1, 2],  # Load chunks 0-2
    max_n_segs=2
)
```

**Use when:**
- Debugging specific data ranges
- Distributed training (each worker loads different chunks)
- Testing on specific data subsets

### 4. Full (Load Everything)

**Load all chunks into memory:**

```python
dataset = load_chunked_dataset(
    dataset_dir="./preprocessed_data/squad_v2",
    split="train",
    mode="full",  # Load everything
    max_n_segs=2
)
```

**Use when:**
- Small datasets that fit in memory
- Maximum performance (no lazy loading overhead)
- Frequent random access needed

---

## 🔧 Configuration Options

### TrainingConfig Integration

```python
from memxlnet.training import TrainingConfig

config = TrainingConfig(
    # ... your existing settings ...

    # Chunked dataset settings
    use_chunked_dataset=True,  # Enable chunked dataset loading
    chunked_dataset_dir="./preprocessed_data/squad_v2",
    chunked_load_mode="streaming",  # "streaming", "first_n", "chunks", "full"
    chunked_num_examples=None,  # For "first_n" mode
    chunked_chunk_indices=None,  # For "chunks" mode

    # Standard settings work as usual
    max_n_segs=2,
    memory_num_tokens=8,
    # ...
)
```

### Preprocessing Options

```bash
python scripts/preprocess_datasets_chunked.py \
    --dataset squad_v2 \
    --output-dir ./preprocessed_data \
    --chunk-size 1000 \           # Documents per chunk
    --max-seq-length 384 \         # XLNet sequence length
    --doc-stride 64 \              # Sliding window stride
    --memory-tokens 8 \            # Memory tokens to configure
    --splits train validation      # Which splits to process
```

---

## 💡 Best Practices

### 1. Preprocessing

✅ **DO:**
- Preprocess once and commit to version control (if chunks are small)
- Use default chunk size (1000) for ~50MB chunks
- Preprocess all splits you'll need (train, validation)

❌ **DON'T:**
- Preprocess every time you run experiments
- Make chunks too large (>100MB is hard for GitHub)
- Process with different settings for different experiments

### 2. Loading

✅ **DO:**
- Use `streaming` mode for training (memory-efficient)
- Use `first_n` mode for quick tests (instant)
- Check manifest with `inspect_chunks.py` before loading

❌ **DON'T:**
- Use `full` mode for large datasets (memory issues)
- Load more data than needed for testing
- Forget to set `max_n_segs` appropriately

### 3. Development Workflow

```bash
# 1. ONE-TIME: Preprocess datasets
python scripts/preprocess_datasets_chunked.py --dataset squad_v2
python scripts/preprocess_datasets_chunked.py --dataset huutuan/long_squad_v2

# 2. QUICK TEST: Verify everything works (instant)
python scripts/test_with_samples.py --num-examples 100 --epochs 1

# 3. FULL EXPERIMENTS: Run all experiments (2-5 min each)
python scripts/paper_experiments_v2/squad/01_baseline_squad_no_memory.py
python scripts/paper_experiments_v2/squad/02_main_squad_8tokens.py
# ... all other experiments
```

---

## 🔍 Helper Scripts

### test_with_samples.py

**Quick testing with sample data:**

```bash
# Test with first 100 documents
python scripts/test_with_samples.py --num-examples 100

# Test with specific chunks
python scripts/test_with_samples.py --chunks 0 1 2

# Full options
python scripts/test_with_samples.py \
    --dataset-dir ./preprocessed_data/squad_v2 \
    --num-examples 100 \
    --memory-tokens 8 \
    --max-segments 2 \
    --epochs 1 \
    --batch-size 4 \
    --eval-steps 50 \
    --output-dir ./outputs/test_sample
```

### inspect_chunks.py

**Inspect chunked dataset metadata:**

```bash
# Inspect main manifest
python scripts/inspect_chunks.py ./preprocessed_data/squad_v2

# Inspect specific split
python scripts/inspect_chunks.py ./preprocessed_data/squad_v2 --split train

# Show detailed statistics
python scripts/inspect_chunks.py ./preprocessed_data/squad_v2 --verbose

# Load and inspect first chunk
python scripts/inspect_chunks.py ./preprocessed_data/squad_v2 --load-chunk 0
```

**Example Output:**
```
📋 MANIFEST INFORMATION
================================================================================

📊 Split: train
   Dataset: squad_v2
   Total examples: 130,319
   Total chunks: 131
   Examples per chunk: 1000

⚙️  Configuration:
   • Max sequence length: 384
   • Doc stride: 64
   • Memory tokens: 8

📦 Chunks:
   Total: 131 chunks
   Documents: 130,319 total
   Segments: 261,938 total (2.01 avg per document)
   Estimated size: ~6.55 GB
   (Based on first chunk: 51.23 MB)
```

---

## 🎨 Advanced Usage

### Custom Preprocessing Pipeline

```python
from memxlnet.data.dataset import process_and_cache_dataset, create_dataset_from_cache
from datasets import Dataset as HFDataset

# 1. Process with custom settings
num_features = process_and_cache_dataset(
    dataset_name="squad_v2",
    split="train",
    cache_dir="./.cache_custom",
    max_examples=None,
    max_seq_length=512,  # Longer sequences
    doc_stride=128,      # Different stride
    streaming_chunk_size=5000,
    tokenizer=tokenizer,
)

# 2. Load processed data
dataset = create_dataset_from_cache(
    dataset_name="squad_v2",
    split="train",
    cache_dir="./.cache_custom",
    max_examples=None,
    max_seq_length=512,
    doc_stride=128,
    max_n_segs=4,
    tokenizer=tokenizer,
)

# 3. Save as chunks (manual chunking)
# See scripts/preprocess_datasets_chunked.py for reference
```

### Distributed Training

```python
# Worker 0 loads chunks 0-43
dataset_worker0 = load_chunked_dataset(
    dataset_dir="./preprocessed_data/squad_v2",
    split="train",
    mode="chunks",
    chunk_indices=list(range(0, 44)),
)

# Worker 1 loads chunks 44-87
dataset_worker1 = load_chunked_dataset(
    dataset_dir="./preprocessed_data/squad_v2",
    split="train",
    mode="chunks",
    chunk_indices=list(range(44, 88)),
)

# Worker 2 loads chunks 88-131
dataset_worker2 = load_chunked_dataset(
    dataset_dir="./preprocessed_data/squad_v2",
    split="train",
    mode="chunks",
    chunk_indices=list(range(88, 132)),
)
```

---

## 🐛 Troubleshooting

### "Manifest not found"

**Problem:** Cannot find manifest file.

**Solution:**
```bash
# Check if preprocessing completed
ls -la ./preprocessed_data/squad_v2/

# Re-run preprocessing if needed
python scripts/preprocess_datasets_chunked.py --dataset squad_v2
```

### "Chunk index out of range"

**Problem:** Requested chunk index doesn't exist.

**Solution:**
```bash
# Inspect available chunks
python scripts/inspect_chunks.py ./preprocessed_data/squad_v2 --verbose

# Use valid chunk indices
python scripts/test_with_samples.py --chunks 0 1 2  # Not 999
```

### Memory Issues

**Problem:** Out of memory when loading.

**Solution:**
```python
# Use streaming mode (memory-efficient)
config = TrainingConfig(
    use_chunked_dataset=True,
    chunked_load_mode="streaming",  # Not "full"
    # ...
)
```

### Slow Loading

**Problem:** Loading still takes long.

**Possible causes:**
1. **Disk I/O bottleneck**: Use SSD instead of HDD
2. **Wrong mode**: Use `streaming` instead of `full` for large datasets
3. **Too many workers**: Reduce `num_workers` in dataloader

---

## 📊 Performance Comparison

| Operation | Traditional | Chunked | Speedup |
|-----------|------------|---------|---------|
| **First experiment** | 30-60 min | 30-60 min (preprocessing) + 2-5 min (loading) | None |
| **Second experiment** | 30-60 min | 2-5 min | **6-12x faster!** |
| **12 experiments** | 6-12 hours | 1-2 hours | **6x faster!** |
| **Quick test (100 docs)** | 30-60 min | Instant | **∞ faster!** |

### Disk Space

| Dataset | Chunk Size | Total Size | Chunks |
|---------|------------|------------|--------|
| SQuAD v2 Train | ~50MB | ~6.5GB | 131 |
| SQuAD v2 Val | ~50MB | ~500MB | 11 |
| Long SQuAD v2 Train | ~50MB | ~8GB | 160 |

---

## 🔗 Related Documentation

- **[API Reference](../api/API_REFERENCE.md)** - Complete API documentation
- **[Data Processing](../technical/DATA_PROCESSING.md)** - Data pipeline details
- **[Streaming Guide](STREAMING_GUIDE.md)** - Memory-efficient processing
- **[Quick Reference](ENHANCED_MA_XLNET_QUICK_REFERENCE.md)** - Developer cheat sheet

---

## ✅ Summary

**Chunked Datasets solve the preprocessing time problem:**

1. **Preprocess once** (30-60 min) with `preprocess_datasets_chunked.py`
2. **Load fast** (2-5 min) with `ChunkedDataset`
3. **Test instantly** with `test_with_samples.py`
4. **Save 5-11 hours** on 12 experiments!

**The system is:**
- ✅ GitHub-friendly (~50MB chunks)
- ✅ Memory-efficient (streaming support)
- ✅ Flexible (4 loading modes)
- ✅ Compatible (drop-in replacement)
- ✅ Production-ready

**Get started:**
```bash
# 1. Preprocess (once)
python scripts/preprocess_datasets_chunked.py --dataset squad_v2

# 2. Test (instant)
python scripts/test_with_samples.py --num-examples 100

# 3. Run experiments (fast!)
python scripts/paper_experiments_v2/squad/02_main_squad_8tokens.py
```

🎉 **Happy experimenting!**
