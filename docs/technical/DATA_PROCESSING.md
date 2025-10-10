# Data Processing Pipeline for MemXLNet-QA

This document provides a comprehensive overview of how data is transformed, processed, and fed to the MemXLNet-QA model for training and evaluation. Understanding this pipeline is crucial for working with the system effectively.

## Table of Contents

1. [Overview](#overview)
2. [Raw Data to Model Input Flow](#raw-data-to-model-input-flow)
3. [Document Segmentation Strategy](#document-segmentation-strategy)
4. [Memory Token Integration](#memory-token-integration)
5. [Time-Step-Major Batching](#time-step-major-batching)
6. [Answer Span Mapping](#answer-span-mapping)
7. [Caching and Efficiency](#caching-and-efficiency)
8. [HuggingFace Hub Dataset Integration](#huggingface-hub-dataset-integration)
9. [Training vs Evaluation Differences](#training-vs-evaluation-differences)
10. [Performance Optimizations](#performance-optimizations)
11. [Common Issues and Solutions](#common-issues-and-solutions)

## Overview

The MemXLNet-QA data processing pipeline transforms raw SQuAD v2 question-answer pairs into model-ready batches that support:

- **Document Segmentation**: Long contexts split into overlapping segments
- **Unicode Normalization**: Robust handling of multilingual content and accented characters
- **Enhanced Position Mapping**: Fixed character-to-token alignment with automatic error correction
- **Memory Token Integration**: Special tokens for reading/writing memory states
- **Time-Step-Major Batching**: All first segments processed together, then all second segments
- **Answer Span Mapping**: Correct answer positions across multiple segments with validation
- **Memory State Propagation**: Persistent memory across document segments

### High-Level Data Flow

```
Raw SQuAD Data → Document Segmentation → Memory Token Addition →
Time-Step-Major Batching → Memory-Aware Model Processing →
Answer Extraction & Aggregation
```

## Raw Data to Model Input Flow

### 1. Input Data Structure (SQuAD v2)

```python
# Raw SQuAD v2 example
{
    "id": "56be4db0acb8001400a502ec",
    "title": "Super_Bowl_50",
    "context": "Super Bowl 50 was an American football game...",
    "question": "Which NFL team represented the AFC at Super Bowl 50?",
    "answers": {
        "text": ["Denver Broncos", "Denver Broncos", "Denver Broncos"],
        "answer_start": [177, 177, 177]
    }
}
```

### 2. Preprocessing Steps

The `SquadLikeQADataset` class handles the complete preprocessing pipeline:

```python
# Key preprocessing steps in SquadLikeQADataset.__init__()
for example_idx, example in enumerate(raw_dataset):
    # 1. Process into segments with answer mapping
    features = self._process_example(example, example_idx, tokenizer,
                                   max_seq_length, doc_stride, max_n_segs)

    # 2. Add document tracking metadata
    example_id = f"doc_{example_idx}"
    for feature in features:
        feature['example_id'] = example_id
        feature['segment_index'] = segment_idx
        feature['total_segments'] = len(features)

    # 3. Build document map for memory tracking
    self.document_map[example_id] = feature_indices
```

### 3. Feature Structure After Processing

Each processed feature contains:

```python
{
    # Core model inputs
    'input_ids': [101, 2054, 2029, ...],           # Tokenized sequence
    'attention_mask': [1, 1, 1, ...],              # Attention mask
    'token_type_ids': [0, 0, 0, 1, 1, ...],        # 0=question, 1=context

    # Answer positions
    'start_positions': 15,                          # Answer start token index
    'end_positions': 17,                            # Answer end token index

    # Metadata for evaluation
    'example_id': 'doc_42',                         # Document identifier
    'segment_index': 1,                             # Segment within document
    'total_segments': 3,                            # Total segments for document
    'offset_mapping': [(0,0), (0,5), (6,11), ...], # Char-to-token mapping
    'context': "Super Bowl 50 was an American...",  # Original context text
}
```

## Document Segmentation Strategy

### Why Segmentation is Needed

Long documents exceed model context windows (typically 384-512 tokens), requiring intelligent segmentation that:

1. **Preserves Context**: Overlapping segments maintain context continuity
2. **Maps Answers**: Correct answer positions across segments
3. **Enables Memory**: Segments processed sequentially with memory

### Segmentation Algorithm

The `_process_example()` method implements sliding window segmentation:

```python
def _process_example(self, example, example_idx, tokenizer, max_seq_length, doc_stride, max_n_segs):
    question = example["question"].lstrip()
    context = example["context"]

    # Tokenize with stride to create overlapping segments
    tokenized = tokenizer(
        question, context,                    # Question + context pair
        truncation="only_second",             # Truncate context only
        max_length=max_seq_length,           # 384 tokens typically
        stride=doc_stride,                   # 64-128 token overlap
        return_overflowing_tokens=True,      # Create multiple segments
        return_offsets_mapping=True,         # For answer mapping
        padding="max_length"                 # Pad to fixed length
    )
```

### Segmentation Parameters

- **`max_seq_length`**: Maximum tokens per segment (384-512)
- **`doc_stride`**: Overlap between consecutive segments (64-128 tokens)
- **`max_n_segs`**: Limit segments per document (None for unlimited)

### Example Segmentation

```
Original Context: "The Super Bowl is the annual championship game of the National Football League..."
                  (1500 characters, ~300 tokens)

Segment 1: [PAD]... context[0:250] [SEP] question [SEP] [CLS]
Segment 2: [PAD]... context[186:436] [SEP] question [SEP] [CLS] (64 token overlap)
Segment 3: [PAD]... context[372:622] [SEP] question [SEP] [CLS]
```

**Note**: XLNet uses left-padding and places the CLS token at the **end** of sequences, unlike BERT which uses right-padding with CLS at the start.

## Memory Token Integration

### Memory Token Architecture

Memory tokens enable the model to maintain state across document segments:

```python
# Memory token types
READ_TOKENS = ["[MEM_READ_0]", "[MEM_READ_1]", ..., "[MEM_READ_N]"]
WRITE_TOKENS = ["[MEM_WRITE_0]", "[MEM_WRITE_1]", ..., "[MEM_WRITE_N]"]
```

### Token Addition Process

```python
def configure_memory_tokens(tokenizer, memory_num_tokens):
    # Create memory token strings
    mem_read_tokens = [f"[MEM_READ_{i}]" for i in range(memory_num_tokens)]
    mem_write_tokens = [f"[MEM_WRITE_{i}]" for i in range(memory_num_tokens)]

    # Add to tokenizer vocabulary
    tokenizer.add_special_tokens({
        "additional_special_tokens": mem_read_tokens + mem_write_tokens
    })

    # Get token IDs for model use
    mem_read_ids = tokenizer.convert_tokens_to_ids(mem_read_tokens)
    mem_write_ids = tokenizer.convert_tokens_to_ids(mem_write_tokens)

    return {"mem_read_ids": mem_read_ids, "mem_write_ids": mem_write_ids}
```

### Memory Token Usage in Sequences

```
Without Memory: [PAD]... context [SEP] question [SEP] [CLS]

With Memory:    [PAD]... context [MEM_WRITE_0] [MEM_WRITE_1] ... [SEP]
                [MEM_READ_0] [MEM_READ_1] ... question [SEP] [CLS]
```

**XLNet-specific ordering**: Context comes first, question second, CLS at end. Memory READ tokens appear with the question (at the end), WRITE tokens appear with the context (earlier in sequence).

### Memory Token Processing

1. **READ tokens**: Embeddings replaced with current memory state
2. **WRITE tokens**: Hidden states extracted to update memory
3. **Memory state**: Propagated across segments within same document

## Time-Step-Major Batching

### Problem with Regular Batching

Regular batching processes documents sequentially, breaking memory flow:

```python
# Regular batching - BREAKS MEMORY FLOW
Batch 1: [doc1_seg1, doc2_seg1, doc3_seg1]  # First segments
Batch 2: [doc1_seg2, doc2_seg2, doc3_seg2]  # Second segments (no memory link)
```

### Time-Step-Major Solution

Time-step-major batching processes all first segments, then all second segments:

```python
# Time-step-major batching - PRESERVES MEMORY
Document Batch: [doc1, doc2, doc3]

Time Step 1: [doc1_seg1, doc2_seg1, doc3_seg1]  # All first segments
Memory Bank: {doc1: mem1, doc2: mem2, doc3: mem3}

Time Step 2: [doc1_seg2, doc2_seg2, doc3_seg2]  # All second segments + memory
```

### TimeStepMajorDataLoader Implementation

```python
class TimeStepMajorDataLoader:
    def __iter__(self):
        for batch_docs in document_batches:
            time_step_batches = []

            # Create batch for each time step
            for time_step in range(max_segments):
                step_batch = []
                document_mask = []

                for doc_id in batch_docs:
                    if time_step < len(doc_segments[doc_id]):
                        # Real segment
                        step_batch.append(dataset[doc_segments[doc_id][time_step]])
                        document_mask.append(True)
                    else:
                        # Padding for shorter documents
                        step_batch.append(create_padding_feature())
                        document_mask.append(False)

                # Collate batch with document tracking
                collated_batch = collate_fn(step_batch)
                collated_batch['document_mask'] = torch.tensor(document_mask)
                time_step_batches.append(collated_batch)

            yield time_step_batches  # List of batches for this document set
```

### Memory Bank Management

Memory states are tracked per document across time steps:

```python
# Training/evaluation loop with memory
memory_bank = {}  # doc_id -> memory_state

for time_step_batches in dataloader:
    for time_step, batch in enumerate(time_step_batches):
        # Get memory states for this batch
        batch_memory = []
        for doc_id in batch['example_ids']:
            if doc_id in memory_bank:
                batch_memory.append(memory_bank[doc_id])
            else:
                batch_memory.append(model.get_initial_memory(1, device))

        memory_state = torch.cat(batch_memory, dim=0)

        # Forward pass with memory
        outputs = model(batch, memory_state=memory_state)

        # Update memory bank
        new_memory = outputs["new_memory_state"]
        for i, doc_id in enumerate(batch['example_ids']):
            if batch['document_mask'][i]:  # Only active documents
                memory_bank[doc_id] = new_memory[i:i+1]
```

## Answer Span Mapping

### Challenge: Answer Positions Across Segments

When documents are segmented, answer spans may:
1. **Fall entirely within one segment** ✅ Easy case
2. **Span across segment boundaries** ❌ Complex case
3. **Appear in multiple segments** ✅ Multiple valid answers

### Answer Position Algorithm

```python
def map_answer_positions(self, example, tokenized, segment_idx):
    answers = example["answers"]

    if len(answers["answer_start"]) == 0:
        # No answer case (SQuAD v2)
        return cls_index, cls_index

    # Get character positions
    start_char = answers["answer_start"][0]
    end_char = start_char + len(answers["text"][0])

    # Get token offset mapping for this segment
    offsets = tokenized["offset_mapping"][segment_idx]
    sequence_ids = tokenized.sequence_ids(segment_idx)

    # Find context boundaries (token_type_id == 1)
    context_start = first_context_token_index
    context_end = last_context_token_index

    # Check if answer is in this segment
    if not (offsets[context_start][0] <= start_char and
            offsets[context_end][1] >= end_char):
        # Answer not in this segment
        return cls_index, cls_index

    # Find token positions for character positions
    token_start = find_token_for_char_position(start_char, offsets)
    token_end = find_token_for_char_position(end_char, offsets)

    return token_start, token_end
```

### Multiple Answer Handling

For documents with answers in multiple segments:

```python
# During evaluation, aggregate predictions across segments
def aggregate_predictions(self, doc_predictions):
    # Select prediction with highest confidence
    best_prediction = max(doc_predictions, key=lambda x: x['confidence'])
    return best_prediction['text']
```

## Caching and Efficiency

### Chunked Cache Manager

The `ChunkedCacheManager` enables memory-efficient processing:

```python
class ChunkedCacheManager:
    def __init__(self, cache_dir, chunk_size=1000):
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size

    def save_chunk(self, data, dataset_name, split, chunk_id):
        cache_path = f"{self.cache_dir}/{dataset_name}_{split}_chunk_{chunk_id}.cache"
        torch.save(data, cache_path)

    def load_chunk(self, dataset_name, split, chunk_id):
        cache_path = f"{self.cache_dir}/{dataset_name}_{split}_chunk_{chunk_id}.cache"
        return torch.load(cache_path) if os.path.exists(cache_path) else []
```

### Cache Key Generation

Cache keys include memory token information to avoid conflicts:

```python
def generate_cache_key(dataset_name, tokenizer):
    cache_suffix = ""
    if tokenizer and len(tokenizer) > 32000:  # Has memory tokens
        cache_suffix = f"_mem{len(tokenizer) - 32000}"
    return f"{dataset_name}{cache_suffix}"
```

### Processing Pipeline with Caching

```python
def process_and_cache_dataset(dataset_name, split, cache_dir, tokenizer, ...):
    # Generate cache key with memory token awareness
    cache_key = generate_cache_key(dataset_name, tokenizer)

    # Check if cached version exists
    if cache_manager.cache_exists(cache_key, split):
        return load_cached_features_count(cache_key, split)

    # Process dataset if not cached
    dataset = SquadLikeQADataset(split, tokenizer, ...)
    features = [dataset[i] for i in range(len(dataset))]

    # Save to cache
    cache_manager.save_chunk(features, cache_key, split, 0)
    return len(features)
```

## HuggingFace Hub Dataset Integration

MemXLNet-QA supports uploading and downloading preprocessed datasets to/from HuggingFace Hub, dramatically reducing preprocessing time and memory usage for subsequent training runs.

### Benefits of Hub Datasets

| Approach | RAM Usage | Processing Time | Workflow |
|----------|-----------|-----------------|----------|
| **Local Processing** | 20-30GB | 30-60 minutes | Process every time |
| **Hub Datasets** | 4-6GB | 2-5 minutes | Download preprocessed data |

### Uploading Preprocessed Datasets to Hub

Use `upload_processed_dataset_to_hub()` to preprocess once and share across runs:

```python
from memxlnet.data import upload_processed_dataset_to_hub
from transformers import XLNetTokenizerFast

# Configure tokenizer with memory tokens
tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
from memxlnet.data import configure_memory_tokens
mem_config = configure_memory_tokens(tokenizer, memory_num_tokens=16)

# Upload preprocessed dataset to Hub
upload_processed_dataset_to_hub(
    dataset_name="squad_v2",              # Source dataset
    splits=["train", "validation"],       # Splits to process
    tokenizer=tokenizer,                  # Configured tokenizer
    max_seq_length=384,
    doc_stride=64,
    max_n_segs=None,                      # Process all segments
    hub_dataset_id="username/memxlnet-squad-mem16",  # Hub repo
    hub_private=True,                     # Private repository
    max_train_samples=None,               # Process full dataset
    max_eval_samples=None,
    cache_dir="./.cache"
)
```

### Downloading Preprocessed Datasets from Hub

Use `load_dataset_from_hub()` for fast training startup:

```python
from memxlnet.data import load_dataset_from_hub

# Download preprocessed dataset from Hub
dataset = load_dataset_from_hub(
    hub_dataset_id="username/memxlnet-squad-mem16",
    split="train",
    cache_dir="./.cache"
)

print(f"Loaded {len(dataset)} preprocessed examples from Hub")
```

### Training Configuration with Hub Datasets

Configure `TrainingConfig` to automatically use Hub datasets:

```python
from memxlnet.training import TrainingConfig

config = TrainingConfig(
    # Hub dataset configuration
    hub_dataset_id="username/memxlnet-squad-mem16",  # Hub repository
    use_hub_dataset=True,          # Try loading from Hub first
    force_reprocess=False,          # Skip reprocessing if Hub data exists

    # Model and training settings
    memory_num_tokens=16,           # Must match preprocessed dataset!
    max_seq_length=384,
    doc_stride=64,
    num_epochs=3,
)

# Trainer will automatically download from Hub if available
trainer = XLNetRecurrentTrainer(config)
trainer.train()  # Fast startup - no preprocessing needed!
```

### Hub Dataset Workflow

**One-time preprocessing (on high-RAM machine):**
```bash
# Preprocess and upload to Hub
python scripts/preprocess_and_upload_to_hub.py
```

**Fast training startup (on any machine):**
```python
config = TrainingConfig(
    hub_dataset_id="username/memxlnet-squad-mem16",
    use_hub_dataset=True,  # Download preprocessed data
)
trainer = XLNetRecurrentTrainer(config)
trainer.train()  # Starts training in minutes!
```

### Cache Priority Order

The data loading system follows this priority:

1. **Hub Dataset** (if `use_hub_dataset=True` and `hub_dataset_id` specified)
2. **Local Cache** (if valid cache exists in `cache_dir`)
3. **Fresh Processing** (if above fail or `force_reprocess=True`)

```python
# Explicit control over data source
config = TrainingConfig(
    use_hub_dataset=True,       # Try Hub first
    force_reprocess=False,      # Don't reprocess if Hub/cache available
    hub_dataset_id="username/memxlnet-squad-mem16",
)
```

### Important Considerations

- **Memory tokens must match**: Hub dataset memory token count must match training config
- **Private repositories**: Set `hub_private=True` when uploading to keep datasets private
- **HF_TOKEN required**: Set `HF_TOKEN` environment variable for Hub operations
- **Dataset versioning**: Use different Hub repos for different configurations (mem8, mem16, mem32)

Example naming convention:
- `username/memxlnet-squad-mem8` - 8 memory tokens
- `username/memxlnet-squad-mem16` - 16 memory tokens
- `username/memxlnet-squad-mem32` - 32 memory tokens

## Training vs Evaluation Differences

### Training Data Flow

```python
# Training: Progressive segment training
config = TrainingConfig(
    progressive_segments=[1, 2, 4],  # Curriculum learning
    max_n_segs=4,                    # Limit segments during training
    train_batch_size=8,              # Documents per batch
)

# Training loop uses time-step-major batching
for time_step_batches in train_dataloader:
    for batch in time_step_batches:
        loss = model(batch, memory_state=memory_bank[batch['example_ids']])
        loss.backward()
```

### Evaluation Data Flow

```python
# Evaluation: All segments, careful aggregation
eval_dataset, eval_dataloader = create_evaluation_dataloader(
    dataset_name="squad_v2",
    split="validation",
    tokenizer=tokenizer,
    max_n_segs=None,                 # Process all segments
    use_time_step_major=True,        # Enable memory
)

# Evaluation aggregates predictions across segments
for time_step_batches in eval_dataloader:
    raw_predictions = evaluate_with_memory(model, time_step_batches)
    final_predictions = aggregate_document_predictions(raw_predictions)
```

### Key Differences

| Aspect | Training | Evaluation |
|--------|----------|------------|
| Segments | Limited (`max_n_segs`) | All segments |
| Progressive | Yes (`progressive_segments`) | No |
| Loss | Per-segment + document-level | N/A |
| Prediction | Training targets | Aggregated across segments |
| Memory | Backpropagation through time | Forward-only propagation |

## Performance Optimizations

### 1. Memory-Efficient Segmentation

```python
# Limit segments during development
config.max_n_segs = 2  # Instead of processing all segments

# Use streaming for large datasets
config.use_streaming = True
config.streaming_chunk_size = 1000
```

### 2. Batch Size Optimization

```python
# Balance memory usage and speed
config.train_batch_size = 4      # Documents (not segments)
config.gradient_accumulation_steps = 4  # Effective batch size = 16
```

### 3. Caching Strategy

```python
# Cache processed features
process_and_cache_dataset(...)  # Run once

# Reuse cached features
dataset = create_dataset_from_cache(...)  # Fast loading
```

### 4. Mixed Precision Training

```python
config.fp16 = True  # Reduce memory usage
```

## Common Issues and Solutions

### Issue 1: Memory Tokens Not Found

**Problem**: Model can't find memory tokens in sequences

**Solution**: Ensure tokenizer includes memory tokens before processing
```python
# Configure memory tokens BEFORE dataset creation
mem_config = configure_memory_tokens(tokenizer, memory_num_tokens)
dataset = SquadLikeQADataset(split, tokenizer, ...)  # Now includes memory tokens
```

### Issue 2: Answer Spans Incorrect

**Problem**: Answer positions don't match text spans

**Solution**: Check offset mapping and token alignment
```python
# Debug answer mapping
def debug_answer_mapping(feature, example):
    input_ids = feature['input_ids']
    start_pos = feature['start_positions']
    end_pos = feature['end_positions']
    offsets = feature['offset_mapping']

    # Extract predicted text
    predicted_tokens = input_ids[start_pos:end_pos+1]
    predicted_text = tokenizer.decode(predicted_tokens)
    expected_text = example['answers']['text'][0]

    print(f"Expected: {expected_text}")
    print(f"Predicted: {predicted_text}")
    print(f"Offsets: {offsets[start_pos:end_pos+1]}")
```

### Issue 3: Time-Step-Major Batching Errors

**Problem**: Document mask or memory states misaligned

**Solution**: Verify document tracking and padding
```python
# Debug batch structure
def debug_time_step_batch(batch):
    print(f"Batch size: {len(batch['example_ids'])}")
    print(f"Example IDs: {batch['example_ids']}")
    print(f"Document mask: {batch['document_mask']}")
    print(f"Active documents: {batch['document_mask'].sum()}")
```

### Issue 4: Memory Usage Too High

**Problem**: Out of memory during processing

**Solutions**:
```python
# Reduce batch size
config.train_batch_size = 2

# Limit segments
config.max_n_segs = 2

# Use gradient accumulation
config.gradient_accumulation_steps = 8

# Enable streaming
config.use_streaming = True
```

### Issue 5: Cache Conflicts

**Problem**: Wrong cached data loaded for memory vs non-memory models

**Solution**: Cache keys automatically include memory token info
```python
# Cache keys are automatically generated with memory awareness
cache_key = f"squad_v2_mem{num_memory_tokens}"  # For memory models
cache_key = "squad_v2"                          # For standard models
```

## Next Steps

- See [MEMORY_TOKENS_GUIDE.md](MEMORY_TOKENS_GUIDE.md) for detailed memory system documentation
- See [DATA_FLOW_DIAGRAMS.md](DATA_FLOW_DIAGRAMS.md) for visual representations
- See [API_REFERENCE.md](API_REFERENCE.md) for complete function documentation
- See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for practical implementation examples

## Summary

The MemXLNet-QA data processing pipeline is designed to handle long documents efficiently while maintaining answer accuracy and enabling memory-augmented processing. Key innovations include:

1. **Overlapping segmentation** with answer span mapping
2. **Memory token integration** for persistent state
3. **Time-step-major batching** for proper memory flow
4. **Efficient caching** with memory-aware keys
5. **Document-level prediction aggregation** for evaluation

Understanding this pipeline is essential for effectively training, evaluating, and extending the MemXLNet-QA system.