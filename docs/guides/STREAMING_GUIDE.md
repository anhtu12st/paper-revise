# Streaming Data Processing Guide

This guide covers memory-efficient data processing using the `StreamingSquadProcessor` for handling large datasets that don't fit in RAM.

## Overview

The streaming processor enables processing of arbitrarily large datasets by:
- Loading data in streaming mode from HuggingFace
- Processing examples in configurable chunks
- Saving processed chunks incrementally
- Automatic memory management with garbage collection
- Optional memory monitoring with psutil

**Key Benefits:**
- ✅ Handle datasets larger than available RAM
- ✅ Predictable memory usage
- ✅ Full compatibility with memory tokens
- ✅ Robust answer span mapping
- ✅ Automatic cleanup and monitoring

## Table of Contents

1. [Quick Start](#quick-start)
2. [StreamingSquadProcessor API](#streamingsquadprocessor-api)
3. [Configuration Options](#configuration-options)
4. [Memory Management](#memory-management)
5. [Integration with Training](#integration-with-training)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Usage

```python
from memxlnet.data.streaming import StreamingSquadProcessor
from transformers import XLNetTokenizerFast

# Initialize tokenizer
tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')

# Create streaming processor
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    streaming_chunk_size=1000,  # Process 1000 examples at a time
    max_memory_gb=8.0           # Trigger cleanup at 8GB
)

# Process dataset in streaming mode
total_features = processor.process_dataset_streaming(
    dataset_name="squad_v2",
    split="train",
    max_examples=None,          # Process all examples
    max_n_segs=None            # No segment limit
)

print(f"Processed {total_features} features")
```

### With Memory Tokens

```python
from memxlnet.data import configure_memory_tokens
from memxlnet.data.streaming import StreamingSquadProcessor

# Configure tokenizer with memory tokens
tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')
mem_config = configure_memory_tokens(tokenizer, memory_num_tokens=16)

# The processor automatically detects memory tokens
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,  # Now has 32 additional memory tokens
    max_seq_length=384,
    doc_stride=128
)

# Process dataset - memory tokens are handled automatically
total_features = processor.process_dataset_streaming(
    dataset_name="squad_v2",
    split="train"
)
# Output: ✓ Memory tokens detected: 32 memory token pairs
```

### With Caching

```python
from memxlnet.data.dataset import ChunkedCacheManager
from memxlnet.data.streaming import StreamingSquadProcessor

# Create cache manager
cache_manager = ChunkedCacheManager(
    cache_dir="./.cache",
    chunk_size=500
)

# Create processor
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    streaming_chunk_size=1000
)

# Process with incremental caching
total_features = processor.process_dataset_streaming(
    dataset_name="squad_v2",
    split="train",
    cache_manager=cache_manager,  # Saves chunks as they're processed
    max_n_segs=6
)

# Cached chunks can be loaded later without reprocessing
```

## StreamingSquadProcessor API

### Constructor

```python
StreamingSquadProcessor(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    streaming_chunk_size: int = 1000,
    max_memory_gb: float = 8.0,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer` | PreTrainedTokenizerBase | required | Tokenizer (should include memory tokens if using MemXLNet) |
| `max_seq_length` | int | 384 | Maximum sequence length for segments |
| `doc_stride` | int | 128 | Overlap between consecutive segments |
| `streaming_chunk_size` | int | 1000 | Number of examples to process at once |
| `max_memory_gb` | float | 8.0 | Maximum memory usage before triggering cleanup |

### Methods

#### `process_dataset_streaming()`

Process a dataset using streaming mode for memory efficiency.

```python
def process_dataset_streaming(
    dataset_name: str,
    split: str = "train",
    max_examples: int | None = None,
    cache_manager: Any | None = None,
    max_n_segs: int | None = None,
) -> int
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | required | HuggingFace dataset name (e.g., 'squad_v2') |
| `split` | str | "train" | Dataset split ('train', 'validation') |
| `max_examples` | int \| None | None | Maximum examples to process (None for all) |
| `cache_manager` | ChunkedCacheManager \| None | None | Cache manager for incremental saving |
| `max_n_segs` | int \| None | None | Maximum segments per document |

**Returns:**
- `int`: Number of processed features

**Example:**
```python
total_features = processor.process_dataset_streaming(
    dataset_name="squad_v2",
    split="train",
    max_examples=10000,      # Process first 10K examples
    cache_manager=cache_mgr,  # Save incrementally
    max_n_segs=6             # Limit to 6 segments per doc
)
```

## Configuration Options

### Streaming Chunk Size

Controls how many examples are processed before saving and cleanup:

```python
# Small chunks - More frequent cleanup, slower but safer
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    streaming_chunk_size=500   # Process 500 at a time
)

# Large chunks - Less overhead, faster but uses more memory
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    streaming_chunk_size=2000  # Process 2000 at a time
)
```

**Recommendations:**
- **Limited RAM (<8GB):** Use 500-1000
- **Moderate RAM (8-16GB):** Use 1000-2000
- **High RAM (>16GB):** Use 2000-5000

### Memory Limit

Set threshold for automatic cleanup:

```python
# Conservative - Cleanup at 4GB
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    max_memory_gb=4.0
)

# Aggressive - Use up to 16GB before cleanup
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    max_memory_gb=16.0
)
```

### Document Segmentation

Control how documents are split:

```python
# More segments - Better coverage but more processing
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    max_seq_length=512,   # Longer segments
    doc_stride=64         # More overlap
)

# Fewer segments - Faster processing but less coverage
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    max_seq_length=256,   # Shorter segments
    doc_stride=192        # Less overlap
)
```

## Memory Management

### Automatic Cleanup

The processor automatically manages memory through:

1. **Chunk-based Processing**: Processes data in batches
2. **Immediate Saving**: Saves processed chunks immediately
3. **Buffer Clearing**: Clears intermediate data structures
4. **Garbage Collection**: Forces GC after each chunk
5. **Memory Monitoring**: Tracks usage and triggers cleanup

### Memory Monitoring (with psutil)

Install psutil for detailed monitoring:

```bash
pip install psutil
```

With psutil installed, you get:
- Real-time memory usage tracking
- Automatic cleanup when exceeding limits
- Memory usage logging

**Example output:**
```
🚀 Initialized StreamingSquadProcessor
   Max sequence length: 384
   Doc stride: 128
   Streaming chunk size: 1000
   Max memory limit: 8.0 GB
   Memory monitoring: enabled

📚 Loading dataset in streaming mode: squad_v2 (train)
🔄 Processed 1000 examples, generated 3245 features (total: 3245 features)
⚠️ Memory usage (8.2 GB) exceeds limit (8.0 GB)
🧹 Running garbage collection...
📉 Memory usage after cleanup: 6.1 GB
```

### Manual Memory Control

You can also control memory manually:

```python
# Process with explicit memory management
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    max_memory_gb=6.0
)

# Override chunk size for specific datasets
processor.streaming_chunk_size = 500  # Reduce for large documents

# Process dataset
total_features = processor.process_dataset_streaming(
    dataset_name="squad_v2",
    split="train"
)
```

## Integration with Training

### Hub-Based Workflow

The streaming processor is ideal for preprocessing datasets once and uploading to HuggingFace Hub:

```python
from memxlnet.data.streaming import StreamingSquadProcessor
from memxlnet.data.dataset import upload_processed_dataset_to_hub
from memxlnet.data import configure_memory_tokens
from transformers import XLNetTokenizerFast
import os

# Set HuggingFace token
os.environ["HF_TOKEN"] = "your_token_here"

# Configure tokenizer with memory tokens
tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
mem_config = configure_memory_tokens(tokenizer, 16)

# Upload preprocessed dataset to Hub (uses streaming internally)
upload_processed_dataset_to_hub(
    dataset_name="squad_v2",
    splits=["train", "validation"],
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=64,
    max_n_segs=None,
    hub_dataset_id="username/memxlnet-squad-mem16",
    hub_private=True,
    cache_dir="./.cache"
)

# Future training runs can download from Hub instead of reprocessing!
```

### Training Configuration

Use streaming-preprocessed datasets in training:

```python
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

config = TrainingConfig(
    # Use Hub dataset (preprocessed with streaming)
    hub_dataset_id="username/memxlnet-squad-mem16",
    use_hub_dataset=True,  # Download preprocessed data

    # Training settings
    memory_num_tokens=16,
    num_epochs=3,
    train_batch_size=8,
)

trainer = XLNetRecurrentTrainer(config)
trainer.train()  # Fast startup - no local preprocessing needed!
```

## Best Practices

### 1. **One-Time Preprocessing**

Use streaming to preprocess large datasets once, then upload to Hub:

```python
# On high-RAM machine (once)
processor = StreamingSquadProcessor(tokenizer=tokenizer)
processor.process_dataset_streaming("squad_v2", "train", cache_manager=cache_mgr)
upload_to_hub(...)  # Upload cached data

# On low-RAM machines (repeated)
dataset = load_dataset_from_hub("username/preprocessed-squad")
# No preprocessing needed!
```

### 2. **Adjust Chunk Size Based on RAM**

```python
import psutil

# Dynamically set chunk size based on available RAM
available_ram_gb = psutil.virtual_memory().available / (1024**3)

if available_ram_gb < 4:
    chunk_size = 500
elif available_ram_gb < 8:
    chunk_size = 1000
elif available_ram_gb < 16:
    chunk_size = 2000
else:
    chunk_size = 5000

processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    streaming_chunk_size=chunk_size
)
```

### 3. **Use Segment Limits for Testing**

```python
# Quick test with limited segments
processor.process_dataset_streaming(
    dataset_name="squad_v2",
    split="validation",
    max_examples=100,      # Test on 100 examples
    max_n_segs=3          # Limit to 3 segments per doc
)

# Full processing after testing
processor.process_dataset_streaming(
    dataset_name="squad_v2",
    split="train",
    max_examples=None,     # Process all
    max_n_segs=None       # No limit
)
```

### 4. **Monitor Progress**

The processor logs progress automatically:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now you'll see detailed progress:
# 🔄 Processed 1000 examples, generated 3245 features (total: 3245 features)
# 🔄 Processed 2000 examples, generated 6512 features (total: 6512 features)
# ...
```

## Troubleshooting

### Issue: Out of Memory Errors

**Symptoms:** Process crashes with `MemoryError` or system becomes unresponsive

**Solutions:**
```python
# 1. Reduce chunk size
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    streaming_chunk_size=500  # Reduce from default 1000
)

# 2. Lower memory limit for more frequent cleanup
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    max_memory_gb=4.0  # More aggressive cleanup
)

# 3. Reduce segment parameters
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    max_seq_length=256,  # Shorter segments
    doc_stride=192       # Less overlap
)
```

### Issue: Slow Processing

**Symptoms:** Processing takes very long time

**Solutions:**
```python
# 1. Increase chunk size (if you have RAM)
processor = StreamingSquadProcessor(
    tokenizer=tokenizer,
    streaming_chunk_size=2000  # Process more at once
)

# 2. Limit segments per document
processor.process_dataset_streaming(
    dataset_name="squad_v2",
    split="train",
    max_n_segs=6  # Cap at 6 segments per document
)

# 3. Use fewer examples for testing
processor.process_dataset_streaming(
    dataset_name="squad_v2",
    split="train",
    max_examples=10000  # Limit total examples
)
```

### Issue: Memory Monitoring Not Working

**Symptoms:** No memory usage logs appear

**Cause:** psutil not installed

**Solution:**
```bash
pip install psutil
```

Without psutil, the processor still works but without detailed monitoring.

### Issue: Inconsistent Feature Counts

**Symptoms:** Different feature counts on repeated runs

**Cause:** Caching or partial processing

**Solution:**
```python
# Clear cache before reprocessing
import shutil
shutil.rmtree("./.cache", ignore_errors=True)

# Then reprocess
processor.process_dataset_streaming(
    dataset_name="squad_v2",
    split="train"
)
```

## Advanced Features

### Custom Processing Logic

Extend the processor for custom needs:

```python
from memxlnet.data.streaming import StreamingSquadProcessor

class CustomProcessor(StreamingSquadProcessor):
    def _process_single_example(self, example, max_n_segs):
        """Override to add custom processing."""
        features = super()._process_single_example(example, max_n_segs)

        # Add custom fields to each feature
        for feature in features:
            feature['custom_field'] = self._compute_custom_value(feature)

        return features

    def _compute_custom_value(self, feature):
        # Custom computation
        return len(feature['context'])

# Use custom processor
processor = CustomProcessor(tokenizer=tokenizer)
```

### Monitoring Callback

Track progress programmatically:

```python
processed_counts = []

def monitor_progress(processed, total_features):
    processed_counts.append((processed, total_features))
    print(f"Progress: {processed} examples -> {total_features} features")

# Process with monitoring (requires custom implementation)
# This is a conceptual example
```

## Summary

The `StreamingSquadProcessor` enables memory-efficient processing of large datasets through:

✅ **Streaming mode** - Load data incrementally
✅ **Chunk-based processing** - Process in manageable batches
✅ **Automatic cleanup** - Manage memory automatically
✅ **Incremental caching** - Save results as you go
✅ **Memory monitoring** - Track and control usage
✅ **Full compatibility** - Works with memory tokens and all features

**Use streaming when:**
- Dataset size > Available RAM
- Preprocessing large training sets
- Creating reusable preprocessed datasets for Hub
- Working on memory-constrained environments

---

**See Also:**
- [API Reference](../api/API_REFERENCE.md) - Complete API documentation
- [Data Processing](../technical/DATA_PROCESSING.md) - Technical details
- [Hub Integration](../guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md#hub-integration) - Upload/download workflows
