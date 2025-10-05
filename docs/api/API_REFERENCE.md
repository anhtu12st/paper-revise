# API Reference for MemXLNet-QA Data Processing

This document provides comprehensive API documentation for all data processing functions, classes, and utilities in the MemXLNet-QA system.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Data Processing Functions](#data-processing-functions)
3. [Text Utilities (Unicode & Position Mapping)](#text-utilities-unicode--position-mapping)
4. [Memory Token Integration](#memory-token-integration)
5. [Caching System](#caching-system)
6. [Time-Step-Major Batching](#time-step-major-batching)
7. [Configuration Classes](#configuration-classes)
8. [Utility Functions](#utility-functions)
9. [Training Components](#training-components)
10. [Evaluation Components](#evaluation-components)

## Core Classes

### SquadLikeQADataset

**Location**: `src/data.py`

Enhanced SQuAD v2 dataset preprocessor with document tracking and time-step-major support.

```python
class SquadLikeQADataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        max_examples: Optional[int] = None,
        dataset_name: str = "squad_v2",
        max_n_segs: Optional[int] = None,
    ) -> None
```

**Parameters**:
- `split` (str): Dataset split ('train', 'validation', etc.)
- `tokenizer` (PreTrainedTokenizerBase): Tokenizer for text processing. Should include memory tokens if using memory.
- `max_seq_length` (int): Maximum tokens per segment (including special tokens). Default: 384
- `doc_stride` (int): Overlap between consecutive segments in tokens. Default: 128
- `max_examples` (Optional[int]): Maximum examples to process (None for all). Default: None
- `dataset_name` (str): HuggingFace dataset name. Default: "squad_v2"
- `max_n_segs` (Optional[int]): Maximum segments per document (None for unlimited). Default: None

**Attributes**:
- `features` (List[Dict]): List of processed feature dictionaries
- `document_map` (Dict[str, List[int]]): Maps example_id -> list of feature indices for document tracking

**Methods**:

#### `get_document_segments(example_id: str) -> List[int]`

Get all segment indices for a document.

**Parameters**:
- `example_id` (str): Document identifier

**Returns**:
- `List[int]`: List of feature indices belonging to this document

#### `get_all_documents() -> List[str]`

Get all document IDs.

**Returns**:
- `List[str]`: List of all document identifiers in the dataset

#### `__len__() -> int`

Get total number of features.

**Returns**:
- `int`: Number of processed features

#### `__getitem__(idx: int) -> Dict[str, torch.Tensor]`

Get a feature by index.

**Parameters**:
- `idx` (int): Feature index

**Returns**:
- `Dict[str, torch.Tensor]`: Feature dictionary containing:
  - `input_ids`: Token IDs tensor
  - `attention_mask`: Attention mask tensor
  - `token_type_ids`: Token type IDs tensor
  - `start_positions`: Answer start position
  - `end_positions`: Answer end position
  - Plus metadata fields: `example_id`, `segment_index`, `total_segments`, `offset_mapping`, `context`

**Example**:
```python
from transformers import XLNetTokenizerFast
from memxlnet.data import SquadLikeQADataset

tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
dataset = SquadLikeQADataset(
    split="validation",
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=64,
    max_examples=100
)

print(f"Dataset size: {len(dataset)}")
print(f"Documents: {len(dataset.get_all_documents())}")

# Get first feature
feature = dataset[0]
print(f"Feature keys: {list(feature.keys())}")
```

### TimeStepMajorDataLoader

**Location**: `src/data.py`

DataLoader that reorganizes regular batches into time-step-major format for proper memory state propagation.

```python
class TimeStepMajorDataLoader:
    def __init__(
        self,
        dataset: SquadLikeQADataset,
        batch_size: int,
        shuffle: bool = False,
        max_segments: Optional[int] = None,
    )
```

**Parameters**:
- `dataset` (SquadLikeQADataset): Dataset with document tracking
- `batch_size` (int): Number of documents to process simultaneously
- `shuffle` (bool): Whether to shuffle document order. Default: False
- `max_segments` (Optional[int]): Maximum segments per document (None for all). Default: None

**Attributes**:
- `documents` (List[str]): List of document IDs
- `document_segments` (Dict[str, List[int]]): Maps document ID to segment indices
- `max_doc_segments` (int): Maximum number of segments across all documents

**Methods**:

#### `__iter__() -> Iterator[List[Dict]]`

Iterate over time-step-major batches.

**Returns**:
- `Iterator[List[Dict]]`: Iterator yielding lists of batches for each document group

**Yields**:
- `List[Dict]`: List of time-step batches, where each batch contains:
  - Tensor fields: `input_ids`, `attention_mask`, `token_type_ids`, etc.
  - Metadata fields: `example_ids`, `document_mask`, etc.

#### `__len__() -> int`

Get number of document batches.

**Returns**:
- `int`: Number of document batches

**Example**:
```python
from memxlnet.data import TimeStepMajorDataLoader

# Create time-step-major dataloader
dataloader = TimeStepMajorDataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=True,
    max_segments=3
)

# Iterate over document batches
for doc_batch_idx, time_step_batches in enumerate(dataloader):
    print(f"Document batch {doc_batch_idx}: {len(time_step_batches)} time steps")

    for time_step, batch in enumerate(time_step_batches):
        print(f"  Time step {time_step}: {len(batch['example_ids'])} examples")
        print(f"  Active documents: {batch['document_mask'].sum()}")
```

### ChunkedCacheManager

**Location**: `src/data.py`

Manages chunked caching of large datasets for memory-efficient processing.

```python
class ChunkedCacheManager:
    def __init__(self, cache_dir: str, chunk_size: int = 1000)
```

**Parameters**:
- `cache_dir` (str): Directory for storing cache files
- `chunk_size` (int): Number of features per chunk file. Default: 1000

**Methods**:

#### `get_cache_path(dataset_name: str, split: str, chunk_id: int) -> str`

Get the cache file path for a specific chunk.

**Parameters**:
- `dataset_name` (str): Name of the dataset
- `split` (str): Dataset split name
- `chunk_id` (int): Chunk identifier

**Returns**:
- `str`: Full path to the cache file

#### `cache_exists(dataset_name: str, split: str) -> bool`

Check if cached chunks exist for the dataset.

**Parameters**:
- `dataset_name` (str): Name of the dataset
- `split` (str): Dataset split name

**Returns**:
- `bool`: True if cache exists, False otherwise

#### `save_chunk(data: List[Dict], dataset_name: str, split: str, chunk_id: int)`

Save a chunk of processed data to cache.

**Parameters**:
- `data` (List[Dict]): List of processed feature dictionaries
- `dataset_name` (str): Name of the dataset
- `split` (str): Dataset split name
- `chunk_id` (int): Chunk identifier

#### `load_chunk(dataset_name: str, split: str, chunk_id: int) -> List[Dict]`

Load a chunk of processed data from cache.

**Parameters**:
- `dataset_name` (str): Name of the dataset
- `split` (str): Dataset split name
- `chunk_id` (int): Chunk identifier

**Returns**:
- `List[Dict]`: List of cached feature dictionaries, empty list if not found

#### `get_total_chunks(dataset_name: str, split: str) -> int`

Get the total number of cached chunks for a dataset.

**Parameters**:
- `dataset_name` (str): Name of the dataset
- `split` (str): Dataset split name

**Returns**:
- `int`: Total number of cached chunks

**Example**:
```python
from memxlnet.data import ChunkedCacheManager

cache_manager = ChunkedCacheManager("./.cache", chunk_size=500)

# Check if cache exists
if cache_manager.cache_exists("squad_v2", "validation"):
    print("Cache found, loading...")
    chunk_data = cache_manager.load_chunk("squad_v2", "validation", 0)
    print(f"Loaded {len(chunk_data)} features from cache")
else:
    print("No cache found, processing...")
    # Process data...
    cache_manager.save_chunk(features, "squad_v2", "validation", 0)
```

## Data Processing Functions

### process_and_cache_dataset

**Location**: `src/data.py`

Process and cache dataset features for memory-efficient loading.

```python
def process_and_cache_dataset(
    dataset_name: str,
    split: str,
    cache_dir: str,
    max_examples: Optional[int],
    max_seq_length: int,
    doc_stride: int,
    streaming_chunk_size: int,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_n_segs: Optional[int] = None,
) -> int
```

**Parameters**:
- `dataset_name` (str): HuggingFace dataset name (e.g., 'squad_v2')
- `split` (str): Dataset split ('train', 'validation', etc.)
- `cache_dir` (str): Directory for storing cached features
- `max_examples` (Optional[int]): Maximum examples to process (None for all)
- `max_seq_length` (int): Maximum tokens per segment
- `doc_stride` (int): Overlap between segments in tokens
- `streaming_chunk_size` (int): Size of chunks for streaming processing
- `tokenizer` (Optional[PreTrainedTokenizerBase]): Tokenizer to use. If None, loads default xlnet-base-cased. Pass checkpoint tokenizer to use memory tokens properly.
- `max_n_segs` (Optional[int]): Maximum segments per document

**Returns**:
- `int`: Number of processed features

**Description**:
This function handles the complete preprocessing pipeline:
1. Load raw dataset from HuggingFace
2. Process into segments with answer span mapping
3. Cache processed features for fast reloading
4. Handle memory tokens if present in tokenizer

The function automatically detects memory-enabled tokenizers (those with >32000 tokens) and includes this in the cache key to avoid conflicts between standard and memory-enabled processing.

**Example**:
```python
from transformers import XLNetTokenizerFast
from memxlnet.data import process_and_cache_dataset, configure_memory_tokens

# Setup tokenizer with memory tokens
tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
configure_memory_tokens(tokenizer, memory_num_tokens=8)

# Process and cache dataset
total_features = process_and_cache_dataset(
    dataset_name="squad_v2",
    split="validation",
    cache_dir="./.cache",
    max_examples=1000,
    max_seq_length=384,
    doc_stride=64,
    streaming_chunk_size=500,
    tokenizer=tokenizer,
    max_n_segs=4
)

print(f"Processed and cached {total_features} features")
```

### create_dataset_from_cache

**Location**: `src/data.py`

Create dataset from cache or fresh processing.

```python
def create_dataset_from_cache(
    dataset_name: str,
    split: str,
    cache_dir: str,
    max_examples: Optional[int],
    max_seq_length: int,
    doc_stride: int,
    max_n_segs: Optional[int],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
)
```

**Parameters**:
- `dataset_name` (str): HuggingFace dataset name (e.g., 'squad_v2')
- `split` (str): Dataset split ('train', 'validation', etc.)
- `cache_dir` (str): Directory containing cached features
- `max_examples` (Optional[int]): Maximum examples to process (None for all)
- `max_seq_length` (int): Maximum tokens per segment
- `doc_stride` (int): Overlap between segments in tokens
- `max_n_segs` (Optional[int]): Maximum segments per document
- `tokenizer` (Optional[PreTrainedTokenizerBase]): Tokenizer to use. If None, loads default xlnet-base-cased. Pass checkpoint tokenizer to use memory tokens properly.

**Returns**:
- `SquadLikeQADataset`: Dataset instance with document tracking and segmentation

**Description**:
This function creates a SquadLikeQADataset instance, using cached data if available or processing fresh data if needed. It's the primary interface for creating evaluation datasets.

Note: This function always creates the dataset fresh rather than loading from cache. For cache-based loading, use ChunkedCacheManager directly.

**Example**:
```python
from memxlnet.data import create_dataset_from_cache

dataset = create_dataset_from_cache(
    dataset_name="squad_v2",
    split="validation",
    cache_dir="./.cache",
    max_examples=500,
    max_seq_length=384,
    doc_stride=64,
    max_n_segs=3,
    tokenizer=tokenizer
)

print(f"Created dataset with {len(dataset)} features")
```

### create_evaluation_dataloader

**Location**: `src/data.py`

Convenience function to create evaluation dataset and dataloader using proper pipeline.

```python
def create_evaluation_dataloader(
    dataset_name: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 384,
    doc_stride: int = 64,
    batch_size: int = 8,
    max_examples: Optional[int] = None,
    max_n_segs: Optional[int] = None,
    cache_dir: str = "./.cache",
    use_time_step_major: bool = True,
) -> Tuple[SquadLikeQADataset, "DataLoader"]
```

**Parameters**:
- `dataset_name` (str): Name of dataset (e.g., "squad_v2")
- `split` (str): Dataset split (e.g., "validation")
- `tokenizer` (PreTrainedTokenizerBase): Tokenizer to use (should include memory tokens if applicable)
- `max_seq_length` (int): Maximum sequence length including special tokens. Default: 384
- `doc_stride` (int): Document stride for overlapping segments. Default: 64
- `batch_size` (int): Batch size for dataloader. Default: 8
- `max_examples` (Optional[int]): Maximum examples to process (None for all). Default: None
- `max_n_segs` (Optional[int]): Maximum segments per document (None for unlimited). Default: None
- `cache_dir` (str): Directory for caching processed data. Default: "./.cache"
- `use_time_step_major` (bool): Whether to use time-step-major batching. Default: True

**Returns**:
- `Tuple[SquadLikeQADataset, DataLoader]`: Tuple of (dataset, dataloader) ready for evaluation

**Description**:
This is the recommended function for setting up evaluation pipelines. It handles the complete workflow from raw data to ready-to-use dataloader, with proper caching and memory token support.

Pipeline Steps:
1. Process and cache dataset if needed (with memory token awareness)
2. Create dataset from processed features
3. Create appropriate dataloader (time-step-major for memory models)

The function automatically detects memory tokens in the tokenizer and uses appropriate caching to avoid conflicts between standard and memory-enabled processing.

**Example**:
```python
from memxlnet.data import create_evaluation_dataloader
from transformers import XLNetTokenizerFast

# Load tokenizer with memory tokens
tokenizer = XLNetTokenizerFast.from_pretrained("path/to/checkpoint")

# Create evaluation pipeline
eval_dataset, eval_dataloader = create_evaluation_dataloader(
    dataset_name="squad_v2",
    split="validation",
    tokenizer=tokenizer,
    batch_size=8,
    max_examples=100,
    use_time_step_major=True
)

print(f"Created evaluation pipeline:")
print(f"  Dataset: {len(eval_dataset)} features")
print(f"  DataLoader: {len(eval_dataloader)} batches")

# Use in evaluation
for time_step_batches in eval_dataloader:
    for batch in time_step_batches:
        # Process batch with model
        pass
```

## Text Utilities (Unicode & Position Mapping)

The `src/text_utils.py` module provides comprehensive Unicode normalization and position mapping utilities for robust multilingual QA processing.

### Core Functions

#### `normalize_unicode(text: str) -> str`

Normalize Unicode text using NFC (Canonical Decomposition followed by Canonical Composition) for consistent character representation.

**Parameters:**
- `text` (str): Input text that may contain Unicode characters

**Returns:**
- `str`: Normalized text with consistent Unicode encoding

**Example:**
```python
from memxlnet.data.text_utils import normalize_unicode

text = "café"  # May be encoded as 'c+a+f+é' or 'c+a+f+e+◌́'
normalized = normalize_unicode(text)
print(normalized)  # Always 'café' with consistent encoding
```

#### `normalize_answer_for_comparison(s: str) -> str`

Enhanced answer normalization with Unicode support for SQuAD-style evaluation. Combines Unicode normalization with standard SQuAD preprocessing.

**Parameters:**
- `s` (str): Answer string to normalize

**Returns:**
- `str`: Normalized string suitable for comparison

**Example:**
```python
from memxlnet.data.text_utils import normalize_answer_for_comparison

answer1 = "François Mitterrand"
answer2 = "francois mitterrand"
norm1 = normalize_answer_for_comparison(answer1)
norm2 = normalize_answer_for_comparison(answer2)
print(norm1 == norm2)  # True - normalized to same form
```

#### `find_answer_span_with_normalization(context: str, answer_text: str) -> Tuple[Optional[int], Optional[int]]`

Find answer span in context using Unicode normalization and multiple matching strategies.

**Parameters:**
- `context` (str): The context text to search in
- `answer_text` (str): The answer text to find

**Returns:**
- `Tuple[Optional[int], Optional[int]]`: Start and end character positions, or (None, None) if not found

**Example:**
```python
from memxlnet.data.text_utils import find_answer_span_with_normalization

context = "Le café de la rue Saint-Honoré est très bon."
answer = "café"
start, end = find_answer_span_with_normalization(context, answer)
print(f"Found at positions: {start}-{end}")  # 3-7
print(context[start:end])  # "café"
```

#### `validate_answer_positions(context: str, answer_text: str, start_char: int, end_char: int) -> bool`

Validate that given character positions actually contain the expected answer text.

**Parameters:**
- `context` (str): The full context text
- `answer_text` (str): The expected answer text
- `start_char` (int): Start character position
- `end_char` (int): End character position

**Returns:**
- `bool`: True if positions are valid, False otherwise

**Example:**
```python
from memxlnet.data.text_utils import validate_answer_positions

context = "François était président."
answer = "François"
is_valid = validate_answer_positions(context, answer, 0, 8)
print(is_valid)  # True
```

#### `fix_answer_positions(context: str, answer_text: str, start_char: int) -> Tuple[int, int]`

Attempt to fix answer positions that may have off-by-one errors or other alignment issues.

**Parameters:**
- `context` (str): The context text
- `answer_text` (str): The answer text
- `start_char` (int): The claimed start position

**Returns:**
- `Tuple[int, int]`: Corrected (start_char, end_char) positions

**Example:**
```python
from memxlnet.data.text_utils import fix_answer_positions

context = "The answer is here."
answer = "answer"
# Suppose we have an off-by-one error
wrong_start = 3  # Should be 4
corrected_start, corrected_end = fix_answer_positions(context, answer, wrong_start)
print(f"Corrected: {corrected_start}-{corrected_end}")  # 4-10
```

#### `compare_answers_fuzzy(answer1: str, answer2: str, threshold: float = 0.8) -> bool`

Compare two answers with fuzzy matching to handle minor tokenization differences.

**Parameters:**
- `answer1` (str): First answer string
- `answer2` (str): Second answer string
- `threshold` (float): Similarity threshold (0.0 to 1.0)

**Returns:**
- `bool`: True if answers are similar enough, False otherwise

**Example:**
```python
from memxlnet.data.text_utils import compare_answers_fuzzy

answer1 = "New York City"
answer2 = "new york"
is_similar = compare_answers_fuzzy(answer1, answer2)
print(is_similar)  # True - normalized forms match
```

### Test Utilities

#### `run_unicode_tests() -> List[dict]`

Run comprehensive Unicode test suite with predefined test cases covering multiple languages and edge cases.

**Returns:**
- `List[dict]`: List of test results with validation status

**Example:**
```python
from memxlnet.data.text_utils import run_unicode_tests

results = run_unicode_tests()
successful_tests = sum(1 for r in results if r['position_valid'] and r['normalized_match'])
print(f"Passed: {successful_tests}/{len(results)} tests")

# Detailed results
for result in results:
    print(f"Test '{result['name']}': {result['position_valid']}")
```

#### `UNICODE_TEST_CASES`

Predefined test cases covering various Unicode scenarios:

```python
from memxlnet.data.text_utils import UNICODE_TEST_CASES

for test_case in UNICODE_TEST_CASES:
    print(f"Testing: {test_case['name']}")
    print(f"  Context: {test_case['context']}")
    print(f"  Answer: {test_case['answer']}")
    print(f"  Expected span: {test_case['start']}-{test_case['end']}")
```

### Integration with Data Processing

The text utilities are automatically integrated into the data processing pipeline:

```python
# In src/data.py - automatic integration
from memxlnet.data.text_utils import (
    normalize_unicode,
    validate_answer_positions,
    fix_answer_positions,
    find_answer_span_with_normalization
)

# Applied during dataset processing
question = normalize_unicode(example["question"].lstrip())
context = normalize_unicode(example["context"])

# Automatic position validation and correction
if not validate_answer_positions(context, answer_text, start_char, end_char):
    corrected_start, corrected_end = fix_answer_positions(context, answer_text, start_char)
    if validate_answer_positions(context, answer_text, corrected_start, corrected_end):
        start_char, end_char = corrected_start, corrected_end
```

## Memory Token Integration

### configure_memory_tokens

**Location**: `src/data.py`

Add memory read/write special tokens to the tokenizer and return their ids.

```python
def configure_memory_tokens(tokenizer: PreTrainedTokenizerBase, memory_num_tokens: int) -> Dict[str, Any]
```

**Parameters**:
- `tokenizer` (PreTrainedTokenizerBase): The tokenizer to modify. Will be updated in-place.
- `memory_num_tokens` (int): Number of memory token pairs to create.

**Returns**:
- `Dict[str, Any]`: Dictionary containing:
  - `mem_read_ids` (List[int]): List of token IDs for memory read tokens [MEM_READ_i]
  - `mem_write_ids` (List[int]): List of token IDs for memory write tokens [MEM_WRITE_i]

**Description**:
This function adds memory-specific special tokens to the tokenizer vocabulary, enabling the model to read from and write to persistent memory states across document segments.

For a minimal implementation, we create two groups of special tokens:
- `[MEM_READ_i]`: Tokens whose embeddings are replaced with memory state
- `[MEM_WRITE_i]`: Tokens whose hidden states update the memory state

**Example**:
```python
from transformers import XLNetTokenizerFast
from memxlnet.data import configure_memory_tokens

tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')
print(f"Original vocab size: {len(tokenizer)}")

mem_config = configure_memory_tokens(tokenizer, 4)

print(f"New vocab size: {len(tokenizer)}")
print(f"Read token IDs: {mem_config['mem_read_ids']}")   # [32000, 32001, 32002, 32003]
print(f"Write token IDs: {mem_config['mem_write_ids']}") # [32004, 32005, 32006, 32007]

# Tokenizer vocabulary now includes:
# [MEM_READ_0], [MEM_READ_1], [MEM_READ_2], [MEM_READ_3]
# [MEM_WRITE_0], [MEM_WRITE_1], [MEM_WRITE_2], [MEM_WRITE_3]
```

### MemoryCollateConfig

**Location**: `src/data.py`

Configuration dataclass for memory-aware collation.

```python
@dataclass
class MemoryCollateConfig:
    enable: bool
    mem_read_ids: Optional[List[int]]
    mem_write_ids: Optional[List[int]]
    max_seq_length: int
    cls_token_id: int
    pad_token_id: int
```

**Attributes**:
- `enable` (bool): Whether memory collation is enabled
- `mem_read_ids` (Optional[List[int]]): List of memory read token IDs
- `mem_write_ids` (Optional[List[int]]): List of memory write token IDs
- `max_seq_length` (int): Maximum sequence length for padding
- `cls_token_id` (int): CLS token ID for special handling
- `pad_token_id` (int): Padding token ID

**Example**:
```python
from memxlnet.data import MemoryCollateConfig, configure_memory_tokens

# Setup memory tokens
mem_config = configure_memory_tokens(tokenizer, 8)

# Create collate configuration
collate_config = MemoryCollateConfig(
    enable=True,
    mem_read_ids=mem_config['mem_read_ids'],
    mem_write_ids=mem_config['mem_write_ids'],
    max_seq_length=384,
    cls_token_id=tokenizer.cls_token_id,
    pad_token_id=tokenizer.pad_token_id
)
```

## Caching System

### _memory_aware_collate_fn

**Location**: `src/data.py`

Enhanced collate function for time-step-major batching with memory support.

```python
def _memory_aware_collate_fn(batch, memory_collate_config: Optional[MemoryCollateConfig] = None)
```

**Parameters**:
- `batch` (List[Dict]): List of feature dictionaries from dataset
- `memory_collate_config` (Optional[MemoryCollateConfig]): Configuration for memory token handling (optional)

**Returns**:
- `Dict`: Dictionary containing:
  - Tensor fields: Stacked tensors (input_ids, attention_mask, etc.)
  - Metadata fields: Lists of metadata (example_id, context, etc.)
  - `document_mask`: Boolean tensor indicating active documents
  - `example_ids`: List of example identifiers

**Description**:
This function collates batches for both regular and time-step-major processing, handling tensor stacking, metadata preservation, and document tracking.

The function automatically adds document_mask and example_ids for time-step-major processing compatibility.

**Example**:
```python
from memxlnet.data import _memory_aware_collate_fn

# Example batch from dataset
batch = [dataset[i] for i in range(4)]

# Collate batch
collated = _memory_aware_collate_fn(batch)

print(f"Collated keys: {list(collated.keys())}")
print(f"Input IDs shape: {collated['input_ids'].shape}")
print(f"Document mask: {collated['document_mask']}")
```

### create_dataloader

**Location**: `src/data.py`

Create dataloader with optional time-step-major batching.

```python
def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    memory_collate_config: Optional[MemoryCollateConfig] = None,
    use_time_step_major: bool = True,
) -> DataLoader
```

**Parameters**:
- `dataset` (Dataset): Dataset to create dataloader for
- `batch_size` (int): Batch size for loading
- `shuffle` (bool): Whether to shuffle data
- `num_workers` (int): Number of workers for data loading
- `memory_collate_config` (Optional[MemoryCollateConfig]): Configuration for memory token handling
- `use_time_step_major` (bool): Whether to use time-step-major batching. Default: True

**Returns**:
- `DataLoader`: DataLoader instance (regular or time-step-major)

**Description**:
This function creates either a regular PyTorch DataLoader or a TimeStepMajorDataLoader depending on the dataset type and configuration.

Time-step-major batching is only used for SquadLikeQADataset instances when use_time_step_major=True. Otherwise, returns regular DataLoader.

**Example**:
```python
from memxlnet.data import create_dataloader, MemoryCollateConfig

# Create memory-aware dataloader
dataloader = create_dataloader(
    dataset=dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    use_time_step_major=True
)

# Iterate over batches
for batch_data in dataloader:
    if isinstance(batch_data, list):  # Time-step-major
        print(f"Time-step-major batch: {len(batch_data)} time steps")
        for time_step, batch in enumerate(batch_data):
            print(f"  Time step {time_step}: {batch['input_ids'].shape}")
    else:  # Regular batch
        print(f"Regular batch: {batch_data['input_ids'].shape}")
```

## Configuration Classes

### TrainingConfig

**Location**: `src/train.py`

Training configuration with all hyperparameters and settings.

```python
@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "xlnet-base-cased"
    max_seq_length: int = 384
    doc_stride: int = 128

    # Dataset settings
    dataset_name: str = "squad_v2"
    train_split: str = "train"
    eval_split: str = "validation"
    cache_dir: str = "./.cache"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    # Memory settings
    memory_num_tokens: int = 32
    memory_update: str = "gated"
    memory_init: str = "learned"
    memory_impl: str = "token"
    bptt_horizon: int = 6

    # Training parameters
    num_epochs: int = 3
    train_batch_size: int = 4
    eval_batch_size: int = 4
    learning_rate: float = 3e-5
    # ... many more parameters
```

**Key Attributes**:

#### Model Configuration
- `model_name` (str): Base model name or path. Default: "xlnet-base-cased"
- `max_seq_length` (int): Maximum tokens per segment. Default: 384
- `doc_stride` (int): Overlap between segments. Default: 128

#### Dataset Configuration
- `dataset_name` (str): HuggingFace dataset name. Default: "squad_v2"
- `train_split` (str): Training split name. Default: "train"
- `eval_split` (str): Evaluation split name. Default: "validation"
- `cache_dir` (str): Cache directory. Default: "./.cache"
- `max_train_samples` (Optional[int]): Limit training samples
- `max_eval_samples` (Optional[int]): Limit evaluation samples

#### Memory Configuration
- `memory_num_tokens` (int): Number of memory token pairs. Default: 32
- `memory_update` (str): Update mechanism ("gated", "simple", "none"). Default: "gated"
- `memory_init` (str): Initialization ("learned", "zeros"). Default: "learned"
- `memory_impl` (str): Implementation type. Default: "token"
- `bptt_horizon` (int): Backpropagation through time horizon. Default: 6

#### Progressive Training
- `progressive_segments` (Optional[List[int]]): List of segment counts for curriculum learning
- `max_n_segs` (Optional[int]): Maximum segments per document

#### Training Parameters
- `num_epochs` (int): Number of training epochs. Default: 3
- `train_batch_size` (int): Training batch size (documents). Default: 4
- `eval_batch_size` (int): Evaluation batch size. Default: 4
- `learning_rate` (float): Learning rate. Default: 3e-5
- `weight_decay` (float): Weight decay. Default: 0.01
- `warmup_ratio` (float): Warmup ratio. Default: 0.1

**Field Aliases**:
The configuration supports multiple field names for backward compatibility:
- `model_name` / `model_name_or_path`
- `max_train_samples` / `max_train_examples`
- `num_epochs` / `num_train_epochs`

**Example**:
```python
from memxlnet.training import TrainingConfig

# Basic configuration
config = TrainingConfig(
    model_name="xlnet-base-cased",
    memory_num_tokens=16,
    max_n_segs=4,
    num_epochs=2,
    train_batch_size=8
)

# Memory-enabled configuration
memory_config = TrainingConfig(
    memory_num_tokens=16,
    memory_update="gated",
    memory_init="learned",
    progressive_segments=[1, 2, 4],
    use_global_softmax=True
)

# Configuration with field aliases
alias_config = TrainingConfig(
    model_name_or_path="path/to/model",  # Same as model_name
    max_train_examples=1000,             # Same as max_train_samples
    num_train_epochs=5                   # Same as num_epochs
)
```

## Utility Functions

### f1_score

**Location**: `src/train.py`

Calculate F1 score between prediction and ground truth.

```python
def f1_score(prediction, ground_truth):
```

**Parameters**:
- `prediction` (str): Predicted answer text
- `ground_truth` (str): Ground truth answer text

**Returns**:
- `float`: F1 score between 0 and 1

**Description**:
Computes token-level F1 score between normalized prediction and ground truth texts.

**Example**:
```python
from memxlnet.training import f1_score

prediction = "Denver Broncos"
ground_truth = "The Denver Broncos"
score = f1_score(prediction, ground_truth)
print(f"F1 Score: {score:.3f}")  # 1.0 (after normalization)
```

### exact_match_score

**Location**: `src/train.py`

Calculate exact match score between prediction and ground truth.

```python
def exact_match_score(prediction, ground_truth):
```

**Parameters**:
- `prediction` (str): Predicted answer text
- `ground_truth` (str): Ground truth answer text

**Returns**:
- `float`: 1.0 if exact match, 0.0 otherwise

**Description**:
Computes exact match score between normalized prediction and ground truth texts.

**Example**:
```python
from memxlnet.training import exact_match_score

prediction = "Denver Broncos"
ground_truth = "The Denver Broncos"
score = exact_match_score(prediction, ground_truth)
print(f"EM Score: {score}")  # 1.0 (after normalization)
```

### metric_max_over_ground_truths

**Location**: `src/train.py`

Calculate maximum metric over all ground truths.

```python
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
```

**Parameters**:
- `metric_fn` (Callable): Metric function (f1_score or exact_match_score)
- `prediction` (str): Predicted answer text
- `ground_truths` (List[str]): List of ground truth answer texts

**Returns**:
- `float`: Maximum metric score across all ground truths

**Description**:
For questions with multiple valid answers, computes the metric against each ground truth and returns the maximum score.

**Example**:
```python
from memxlnet.training import metric_max_over_ground_truths, f1_score

prediction = "Broncos"
ground_truths = ["Denver Broncos", "The Broncos", "Broncos"]
max_f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
print(f"Max F1: {max_f1:.3f}")
```

## Training Components

### XLNetRecurrentTrainer

**Location**: `src/train.py`

Main trainer class for MemXLNet-QA with memory-aware training.

```python
class XLNetRecurrentTrainer:
    def __init__(self, config: TrainingConfig)
```

**Parameters**:
- `config` (TrainingConfig): Training configuration

**Key Methods**:

#### `prepare_data() -> Tuple[DataLoader, DataLoader, Dataset]`

Prepare training and evaluation data loaders.

**Returns**:
- `Tuple[DataLoader, DataLoader, Dataset]`: (train_loader, eval_loader, eval_dataset)

#### `train() -> Dict[str, Any]`

Run the complete training loop.

**Returns**:
- `Dict[str, Any]`: Training results and metrics

#### `evaluate(eval_loader, eval_dataset) -> Dict[str, float]`

Evaluate model on evaluation dataset.

**Parameters**:
- `eval_loader`: Evaluation data loader
- `eval_dataset`: Evaluation dataset

**Returns**:
- `Dict[str, float]`: Evaluation metrics

**Example**:
```python
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

# Create configuration
config = TrainingConfig(
    memory_num_tokens=8,
    num_epochs=2,
    train_batch_size=4,
    max_train_samples=100,
    max_eval_samples=50
)

# Initialize trainer
trainer = XLNetRecurrentTrainer(config)

# Prepare data
train_loader, eval_loader, eval_dataset = trainer.prepare_data()

# Run training
results = trainer.train()

# Evaluate
metrics = trainer.evaluate(eval_loader, eval_dataset)
print(f"F1: {metrics['f1']:.3f}")
```

## Evaluation Components

### evaluate_squad_v2

**Location**: `src/train.py`

Evaluate SQuAD v2 predictions with official metrics.

```python
def evaluate_squad_v2(predictions, references, no_answer_threshold=0.0):
```

**Parameters**:
- `predictions` (Dict[str, str]): Dict with question_ids as keys and predictions as values
- `references` (Dict[str, Dict]): Dict with question_ids as keys and reference data as values
- `no_answer_threshold` (float): Threshold for no-answer predictions. Default: 0.0

**Returns**:
- `Dict[str, float]`: Dictionary with evaluation metrics:
  - `exact`: Overall exact match percentage
  - `f1`: Overall F1 score percentage
  - `HasAns_exact`: Exact match for questions with answers
  - `HasAns_f1`: F1 score for questions with answers
  - `NoAns_exact`: Exact match for questions without answers
  - `NoAns_f1`: F1 score for questions without answers
  - `HasAns_total`: Number of questions with answers
  - `NoAns_total`: Number of questions without answers

**Description**:
Computes official SQuAD v2 metrics including separate scores for questions with and without answers.

**Example**:
```python
from memxlnet.training import evaluate_squad_v2

# Example predictions and references
predictions = {
    "question_1": "Denver Broncos",
    "question_2": "",  # No answer
    "question_3": "Super Bowl 50"
}

references = {
    "question_1": {"answers": ["Denver Broncos", "The Broncos"]},
    "question_2": {"answers": []},  # No answer question
    "question_3": {"answers": ["Super Bowl 50"]}
}

metrics = evaluate_squad_v2(predictions, references)
print(f"Overall F1: {metrics['f1']:.2f}%")
print(f"HasAns F1: {metrics['HasAns_f1']:.2f}%")
print(f"NoAns F1: {metrics['NoAns_f1']:.2f}%")
```

### main (evaluation entry point)

**Location**: `src/evaluate.py`

Evaluate a trained model using the saved training configuration.

```python
def main(config_path: str, model_path: Optional[str] = None):
```

**Parameters**:
- `config_path` (str): Path to training_config.json
- `model_path` (Optional[str]): Optional path to model directory (if different from config)

**Returns**:
- `Dict[str, Any]`: Evaluation metrics

**Description**:
Entry point for evaluating trained models. Loads configuration, prepares data, and runs evaluation.

**Example**:
```python
from memxlnet.evaluation import main

# Evaluate with saved config
metrics = main("outputs/model/training_config.json")

# Evaluate with custom model path
metrics = main("outputs/model/training_config.json", "path/to/model")

print(f"Evaluation Results:")
for key, value in metrics.items():
    print(f"  {key}: {value}")
```

## Error Handling and Validation

### Common Error Patterns

#### Missing Dependencies
```python
# In data.py - graceful handling of missing datasets package
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

# Later usage with validation
if load_dataset is None:
    raise RuntimeError("The 'datasets' package is required for data loading.")
```

#### Memory Token Validation
```python
def validate_memory_config(tokenizer, memory_num_tokens):
    """Validate memory token configuration."""
    if memory_num_tokens <= 0:
        raise ValueError("memory_num_tokens must be positive")

    expected_vocab_size = 32000 + 2 * memory_num_tokens
    if len(tokenizer) != expected_vocab_size:
        raise ValueError(f"Expected vocab size {expected_vocab_size}, got {len(tokenizer)}")

    return True
```

#### Shape Validation
```python
def validate_memory_shapes(memory_state, expected_shape):
    """Validate memory state shapes."""
    if memory_state.shape != expected_shape:
        raise ValueError(f"Memory shape {memory_state.shape} != expected {expected_shape}")

    if torch.isnan(memory_state).any():
        raise ValueError("Memory state contains NaN values")

    return True
```

## Best Practices

### Performance Optimization

```python
# Use appropriate batch sizes for memory constraints
def get_optimal_batch_size(available_memory_gb, memory_num_tokens):
    """Calculate optimal batch size based on available memory."""
    base_memory_per_sample = 0.1  # GB per sample
    memory_overhead = memory_num_tokens * 0.01  # Additional overhead
    total_memory_per_sample = base_memory_per_sample + memory_overhead
    return max(1, int(available_memory_gb * 0.8 / total_memory_per_sample))

# Cache frequently used computations
from functools import lru_cache

@lru_cache(maxsize=128)
def get_memory_token_positions(input_ids_tuple, mem_token_id):
    """Cache memory token position lookups."""
    input_ids = torch.tensor(input_ids_tuple)
    return (input_ids == mem_token_id).nonzero(as_tuple=True)
```

### Debugging Utilities

```python
def debug_dataset_processing(dataset, num_samples=5):
    """Debug dataset processing with sample outputs."""
    print(f"Dataset size: {len(dataset)}")
    print(f"Document count: {len(dataset.get_all_documents())}")

    for i in range(min(num_samples, len(dataset))):
        feature = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Example ID: {feature.get('example_id', 'N/A')}")
        print(f"  Input shape: {feature['input_ids'].shape}")
        print(f"  Answer span: [{feature['start_positions']}, {feature['end_positions']}]")

def debug_memory_flow(memory_bank, time_step):
    """Debug memory state flow across time steps."""
    print(f"Time step {time_step}:")
    print(f"  Memory bank size: {len(memory_bank)}")

    for doc_id, memory_state in memory_bank.items():
        norm = memory_state.norm().item()
        print(f"  {doc_id}: norm={norm:.4f}, shape={memory_state.shape}")
```

This comprehensive API reference provides detailed documentation for all components of the MemXLNet-QA data processing system, enabling developers to understand and effectively use the system for their own applications.