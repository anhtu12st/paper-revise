import os
from dataclasses import dataclass
from typing import Any, cast

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from memxlnet.data.text_utils import (
    characterize_boundary_delta,
    choose_best_occurrence,
    find_all_occurrences,
    find_answer_span_with_normalization,
    fix_answer_positions,
    normalize_unicode,
    validate_answer_positions,
)


@dataclass
class MemoryCollateConfig:
    enable: bool
    mem_read_ids: list[int] | None
    mem_write_ids: list[int] | None
    max_seq_length: int
    cls_token_id: int
    pad_token_id: int


def configure_memory_tokens(tokenizer: PreTrainedTokenizerBase, memory_num_tokens: int) -> dict[str, Any]:
    """Add memory read/write special tokens to the tokenizer and return their ids.

    This function adds memory-specific special tokens to the tokenizer vocabulary,
    enabling the model to read from and write to persistent memory states across
    document segments.

    Args:
        tokenizer: The tokenizer to modify. Will be updated in-place.
        memory_num_tokens: Number of memory token pairs to create.

    Returns:
        Dictionary containing:
            - mem_read_ids: List of token IDs for memory read tokens [MEM_READ_i]
            - mem_write_ids: List of token IDs for memory write tokens [MEM_WRITE_i]

    Example:
        >>> from transformers import XLNetTokenizerFast
        >>> tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')
        >>> mem_config = configure_memory_tokens(tokenizer, 4)
        >>> print(mem_config['mem_read_ids'])  # [32000, 32001, 32002, 32003]
        >>> print(mem_config['mem_write_ids']) # [32004, 32005, 32006, 32007]

    Note:
        For a minimal implementation, we create two groups of special tokens:
        - [MEM_READ_i]: Tokens whose embeddings are replaced with memory state
        - [MEM_WRITE_i]: Tokens whose hidden states update the memory state
    """
    mem_read_tokens = [f"[MEM_READ_{i}]" for i in range(memory_num_tokens)]
    mem_write_tokens = [f"[MEM_WRITE_{i}]" for i in range(memory_num_tokens)]
    add_tokens = mem_read_tokens + mem_write_tokens
    tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
    # Compute ids after adding tokens
    mem_read_ids = tokenizer.convert_tokens_to_ids(mem_read_tokens)
    mem_write_ids = tokenizer.convert_tokens_to_ids(mem_write_tokens)
    return {"mem_read_ids": mem_read_ids, "mem_write_ids": mem_write_ids}


class SquadLikeQADataset(Dataset):
    """Enhanced SQuAD v2 dataset preprocessor with document tracking and time-step-major support.

    This dataset class processes SQuAD-like QA data into segments with overlapping windows,
    while maintaining document-level tracking for memory state propagation. Each document
    is split into multiple segments that can be processed sequentially with persistent
    memory state.

    Key Features:
        - Document-aware segmentation with configurable stride
        - Memory token integration for recurrent processing
        - Time-step-major batch organization support
        - Answer span mapping across segments
        - Metadata preservation for evaluation

    Attributes:
        features: List of processed feature dictionaries
        document_map: Maps example_id -> list of feature indices for document tracking
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        max_examples: int | None = None,
        dataset_name: str = "squad_v2",
        max_n_segs: int | None = None,
    ) -> None:
        """Initialize SQuAD-like QA dataset with document segmentation.

        Args:
            split: Dataset split ('train', 'validation', etc.)
            tokenizer: Tokenizer for text processing. Should include memory tokens if using memory.
            max_seq_length: Maximum tokens per segment (including special tokens)
            doc_stride: Overlap between consecutive segments in tokens
            max_examples: Maximum examples to process (None for all)
            dataset_name: HuggingFace dataset name (e.g., 'squad_v2')
            max_n_segs: Maximum segments per document (None for unlimited)

        Raises:
            RuntimeError: If 'datasets' package is not installed
        """
        super().__init__()
        if load_dataset is None:
            raise RuntimeError("The 'datasets' package is required for data loading.")
        raw = load_dataset(dataset_name, split=split)
        if max_examples is not None:
            raw = raw.select(range(min(max_examples, len(raw))))

        self.features: list[dict[str, Any]] = []
        self.document_map: dict[str, list[int]] = {}  # Maps example_id -> list of feature indices

        # Process examples with document tracking
        for example_idx, example in enumerate(raw):
            features = self._process_example(example, example_idx, tokenizer, max_seq_length, doc_stride, max_n_segs)

            # Track document segments
            example_id = f"doc_{example_idx}"
            self.document_map[example_id] = []

            for feature in features:
                feature_idx = len(self.features)
                feature["example_id"] = example_id
                feature["segment_index"] = len(self.document_map[example_id])
                feature["total_segments"] = len(features)
                self.features.append(feature)
                self.document_map[example_id].append(feature_idx)

    def _process_example(
        self,
        example: dict[str, Any],
        example_idx: int,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        doc_stride: int,
        max_n_segs: int | None,
    ) -> list[dict[str, Any]]:
        """Process a single example into multiple segments with answer span mapping.

        This method takes a single SQuAD example and creates multiple overlapping
        segments, properly mapping answer spans to token positions in each segment.

        Args:
            example: Raw SQuAD example with 'question', 'context', 'answers'
            example_idx: Index of the example in the dataset
            tokenizer: Tokenizer for processing text
            max_seq_length: Maximum sequence length per segment
            doc_stride: Overlap between segments
            max_n_segs: Maximum segments to create

        Returns:
            List of feature dictionaries, each containing:
                - input_ids: Token IDs for the segment
                - attention_mask: Attention mask
                - token_type_ids: Token type IDs (0=question, 1=context)
                - start_positions: Start position of answer (or CLS if no answer)
                - end_positions: End position of answer (or CLS if no answer)
                - offset_mapping: Character-to-token offset mapping
                - context: Original context text
                - question: Original (normalized) question text
                - cls_index: Index of CLS token within this segment
                - has_answer: Whether original example had any gold answers
                - chosen_answer_text: The normalized answer text actually mapped in this segment ('' if none)
                - chosen_answer_char_span: [start_char, end_char] of chosen answer in full context ([-1,-1] if none)
                - boundary_info: Optional diagnostics about boundary adjustments
        """
        pad_on_right = tokenizer.padding_side == "right"

        # Normalize texts (strip leading spaces from question per SQuAD convention)
        question = normalize_unicode(example["question"].lstrip())
        context = normalize_unicode(example["context"])

        tokenized = tokenizer(
            question if pad_on_right else context,
            context if pad_on_right else question,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")  # noqa: F841 (reserved for future)
        offset_mapping = tokenized.pop("offset_mapping")

        features: list[dict[str, Any]] = []

        # Respect segment cap
        num_segments = len(tokenized["input_ids"])
        if max_n_segs is not None:
            num_segments = min(num_segments, max_n_segs)

        answers = example["answers"]
        gold_candidates: list[tuple[str, int]] = list(zip(answers.get("text", []), answers.get("answer_start", [])))

        for i in range(num_segments):
            input_ids = tokenized["input_ids"][i]
            attention_mask = tokenized["attention_mask"][i]
            offsets = offset_mapping[i]
            sequence_ids = tokenized.sequence_ids(i)

            # CLS index fallback to 0 if not present
            cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0

            has_answer = len(gold_candidates) > 0
            start_positions = cls_index
            end_positions = cls_index
            boundary_info = None
            chosen_answer_text: str = ""
            chosen_answer_char_span: list[int] = [-1, -1]

            if has_answer:
                # Attempt each candidate until one maps; prefer earliest valid mapping
                chosen_start_char: int | None = None
                chosen_end_char: int | None = None
                chosen_true_start: int | None = None

                for raw_answer_text, raw_start in gold_candidates:
                    answer_text = normalize_unicode(raw_answer_text)
                    start_char = raw_start
                    end_char = start_char + len(raw_answer_text)

                    if not validate_answer_positions(context, answer_text, start_char, end_char):
                        fixed_start, fixed_end = fix_answer_positions(context, answer_text, start_char)
                        if validate_answer_positions(context, answer_text, fixed_start, fixed_end):
                            start_char, end_char = fixed_start, fixed_end
                        else:
                            occs = find_all_occurrences(context, answer_text)
                            best = choose_best_occurrence(occs, raw_start)
                            if best:
                                start_char, end_char = best

                    if validate_answer_positions(context, answer_text, start_char, end_char):
                        chosen_start_char = start_char
                        chosen_end_char = end_char
                        chosen_true_start = raw_start
                        break

                # Final fallback if none mapped: try normalized search on first candidate
                if chosen_start_char is None and gold_candidates:
                    fallback_start, fallback_end = find_answer_span_with_normalization(context, gold_candidates[0][0])
                    if fallback_start is not None:
                        chosen_start_char, chosen_end_char = fallback_start, fallback_end
                        chosen_true_start = gold_candidates[0][1]

                # If we have a mapped span, project to token indices if it lies inside this segment window
                if chosen_start_char is not None and chosen_end_char is not None:
                    # Determine context token span within this feature
                    context_token_start = None
                    context_token_end = None
                    for idx, sid in enumerate(sequence_ids):
                        if sid == (1 if pad_on_right else 0):
                            if context_token_start is None:
                                context_token_start = idx
                            context_token_end = idx
                    if context_token_start is None or context_token_end is None:
                        # Should not happen, but guard
                        context_token_start, context_token_end = 0, len(sequence_ids) - 1

                    # Check containment in this window
                    if (
                        offsets[context_token_start][0] <= chosen_start_char
                        and offsets[context_token_end][1] >= chosen_end_char
                    ):
                        # Locate token_start
                        token_start = context_token_start
                        while token_start <= context_token_end and offsets[token_start][0] < chosen_start_char:
                            token_start += 1
                        if (
                            token_start > context_token_start
                            and token_start <= context_token_end
                            and offsets[token_start - 1][0] <= chosen_start_char <= offsets[token_start - 1][1]
                        ):
                            token_start -= 1

                        # Locate token_end
                        token_end = context_token_end
                        while token_end >= context_token_start and offsets[token_end][1] > chosen_end_char:
                            token_end -= 1
                        if (
                            token_end >= context_token_start
                            and token_end < context_token_end
                            and offsets[token_end][0] <= chosen_end_char <= offsets[token_end][1]
                        ):
                            # token_end already correct
                            pass

                        # Defensive: if search overshot and token_start > token_end, treat as no-answer for this segment
                        if token_start > token_end:
                            start_positions = cls_index
                            end_positions = cls_index
                        else:
                            start_positions = max(context_token_start, min(token_start, context_token_end))
                            end_positions = max(context_token_start, min(token_end, context_token_end))

                        # Additional guard: ensure offsets order is non-decreasing; else revert to no-answer
                        try:
                            s_off = offsets[start_positions]
                            e_off = offsets[end_positions]
                            if s_off[0] is None or e_off[1] is None or s_off[0] >= e_off[1]:
                                # Invalid or zero/negative length span -> mark no-answer for this segment
                                start_positions = cls_index
                                end_positions = cls_index
                            else:
                                chosen_answer_text = normalize_unicode(answer_text)
                                chosen_answer_char_span = [chosen_start_char, chosen_end_char]
                        except Exception:
                            start_positions = cls_index
                            end_positions = cls_index

                        # Boundary diagnostics
                        if chosen_true_start is not None:
                            boundary_info = {
                                "delta_start_char": chosen_start_char - chosen_true_start,
                                "boundary_label": characterize_boundary_delta(chosen_start_char, chosen_true_start),
                            }

            # token_type_ids: question tokens 0, context tokens 1
            token_type_ids = [0] * len(input_ids)
            for idx, sid in enumerate(sequence_ids):
                if sid == (1 if pad_on_right else 0):
                    token_type_ids[idx] = 1

            feature: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "start_positions": start_positions,
                "end_positions": end_positions,
                "offset_mapping": offsets,
                "context": context,
                "question": question,
                "cls_index": cls_index,
                "has_answer": has_answer,
                "chosen_answer_text": chosen_answer_text,
                "chosen_answer_char_span": chosen_answer_char_span,
            }
            if boundary_info is not None:
                feature["boundary_info"] = boundary_info

            features.append(feature)

        return features

    def get_document_segments(self, example_id: str) -> list[int]:
        """Get all segment indices for a document.

        Args:
            example_id: Document identifier

        Returns:
            List of feature indices belonging to this document
        """
        result: list[int] = list(self.document_map.get(example_id, []))
        return result

    def get_all_documents(self) -> list[str]:
        """Get all document IDs.

        Returns:
            List of all document identifiers in the dataset
        """
        return list(self.document_map.keys())

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        feature = self.features[idx]
        item = {}

        # Convert tensors
        for key in ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]:
            if key in feature:
                item[key] = torch.tensor(feature[key])

        # Keep metadata as-is (expanded to include new fields)
        metadata_keys = [
            "example_id",
            "segment_index",
            "total_segments",
            "offset_mapping",
            "context",
            "question",
            "cls_index",
            "has_answer",
            "chosen_answer_text",
            "chosen_answer_char_span",
            "boundary_info",
        ]
        for key in metadata_keys:
            if key in feature:
                item[key] = feature[key]

        return item


class ChunkedCacheManager:
    """Manages chunked caching of large datasets for memory-efficient processing.

    This class handles saving and loading processed dataset features in chunks
    to avoid memory issues with large datasets. Each chunk contains a subset
    of processed features that can be loaded independently.

    Now supports incremental caching for streaming processing to avoid
    memory accumulation during large dataset processing.

    Attributes:
        cache_dir: Directory where cache files are stored
        chunk_size: Number of features per cache chunk
    """

    def __init__(self, cache_dir: str, chunk_size: int = 1000):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for storing cache files
            chunk_size: Number of features per chunk file
        """
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_state: dict[str, Any] | None = None  # For incremental caching

    def get_cache_path(self, dataset_name: str, split: str, chunk_id: int) -> str:
        """Get the cache file path for a specific chunk.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split name
            chunk_id: Chunk identifier

        Returns:
            Full path to the cache file
        """
        return os.path.join(self.cache_dir, f"{dataset_name}_{split}_chunk_{chunk_id}.cache")

    def cache_exists(self, dataset_name: str, split: str) -> bool:
        """Check if cached chunks exist for the dataset.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split name

        Returns:
            True if cache exists, False otherwise
        """
        chunk_0_path = self.get_cache_path(dataset_name, split, 0)
        return os.path.exists(chunk_0_path)

    def save_chunk(self, data: list[dict], dataset_name: str, split: str, chunk_id: int):
        """Save a chunk of processed data to cache.

        Args:
            data: List of processed feature dictionaries
            dataset_name: Name of the dataset
            split: Dataset split name
            chunk_id: Chunk identifier
        """
        cache_path = self.get_cache_path(dataset_name, split, chunk_id)
        torch.save(data, cache_path)

    def load_chunk(self, dataset_name: str, split: str, chunk_id: int) -> list[dict[str, Any]]:
        """Load a chunk of processed data from cache.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split name
            chunk_id: Chunk identifier

        Returns:
            List of cached feature dictionaries, empty list if not found
        """
        cache_path = self.get_cache_path(dataset_name, split, chunk_id)
        if os.path.exists(cache_path):
            loaded: list[dict[str, Any]] = torch.load(cache_path)
            return loaded
        return []

    def get_total_chunks(self, dataset_name: str, split: str) -> int:
        """Get the total number of cached chunks for a dataset.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split name

        Returns:
            Total number of cached chunks
        """
        chunk_id = 0
        while os.path.exists(self.get_cache_path(dataset_name, split, chunk_id)):
            chunk_id += 1
        return chunk_id

    def _initialize_incremental_cache(self, dataset_name: str, split: str, **kwargs):
        """Initialize incremental cache saving for streaming processing.

        This method sets up the state needed for incremental saving during
        streaming processing, allowing features to be saved as they are
        processed rather than all at once.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split name
            **kwargs: Additional cache parameters
        """
        # Create cache key from parameters
        cache_key = dataset_name
        if kwargs.get("has_memory_tokens"):
            mem_tokens = kwargs.get("memory_num_tokens", 0)
            if not mem_tokens and kwargs.get("has_memory_tokens"):
                # Estimate from tokenizer size
                cache_key += "_mem"

        # Add versioning
        cache_key += "_v1"  # Match FEATURE_SCHEMA_VERSION

        self._cache_state = {
            "dataset_name": cache_key,
            "split": split,
            "current_chunk_id": 0,
            "temp_features": [],
            "total_features": 0,
            "kwargs": kwargs,
        }

    def _save_features_chunk(self, features: list[dict[str, Any]], dataset_name: str, split: str, **kwargs):
        """Save features incrementally during streaming processing.

        This method accumulates features and saves them in chunks to avoid
        memory accumulation during processing.

        Args:
            features: List of feature dictionaries to save
            dataset_name: Name of the dataset
            split: Dataset split name
            **kwargs: Additional cache parameters
        """
        if self._cache_state is None:
            self._initialize_incremental_cache(dataset_name, split, **kwargs)

        # Type narrowing for mypy - _cache_state is guaranteed to be non-None after initialization
        assert self._cache_state is not None

        # Add features to temporary buffer
        self._cache_state["temp_features"].extend(features)

        # Save complete chunks
        while len(self._cache_state["temp_features"]) >= self.chunk_size:
            chunk_features = self._cache_state["temp_features"][: self.chunk_size]
            self._cache_state["temp_features"] = self._cache_state["temp_features"][self.chunk_size :]

            # Save chunk to disk
            chunk_id = self._cache_state["current_chunk_id"]
            self.save_chunk(chunk_features, self._cache_state["dataset_name"], self._cache_state["split"], chunk_id)

            self._cache_state["current_chunk_id"] += 1
            self._cache_state["total_features"] += len(chunk_features)

    def _finalize_incremental_cache(self, dataset_name: str, split: str, **kwargs):
        """Finalize incremental cache saving.

        This method saves any remaining features and cleans up the cache state.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split name
            **kwargs: Additional cache parameters
        """
        if self._cache_state is None:
            return

        # Type narrowing for mypy - _cache_state is guaranteed to be non-None here
        assert self._cache_state is not None

        # Save any remaining features
        if self._cache_state["temp_features"]:
            chunk_id = self._cache_state["current_chunk_id"]
            self.save_chunk(
                self._cache_state["temp_features"],
                self._cache_state["dataset_name"],
                self._cache_state["split"],
                chunk_id,
            )
            self._cache_state["total_features"] += len(self._cache_state["temp_features"])

        total_features = self._cache_state["total_features"]
        total_chunks = self._cache_state["current_chunk_id"] + (1 if self._cache_state["temp_features"] else 0)

        # Clear cache state
        self._cache_state = None

        print(f"✅ Finalized incremental cache: {total_chunks} chunks, {total_features} features")


def process_and_cache_dataset(
    dataset_name: str,
    split: str,
    cache_dir: str,
    max_examples: int | None,
    max_seq_length: int,
    doc_stride: int,
    streaming_chunk_size: int,
    tokenizer: PreTrainedTokenizerBase | None = None,
    max_n_segs: int | None = None,
    use_streaming: bool = True,
    max_memory_gb: float = 8.0,
) -> int:
    """Process and cache dataset features for memory-efficient loading.

    This function handles the complete preprocessing pipeline:
    1. Load raw dataset from HuggingFace (streaming or regular)
    2. Process into segments with answer span mapping
    3. Cache processed features incrementally for memory efficiency
    4. Handle memory tokens if present in tokenizer

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'squad_v2')
        split: Dataset split ('train', 'validation', etc.)
        cache_dir: Directory for storing cached features
        max_examples: Maximum examples to process (None for all)
        max_seq_length: Maximum tokens per segment
        doc_stride: Overlap between segments in tokens
        streaming_chunk_size: Size of chunks for streaming processing
        tokenizer: Tokenizer to use. If None, loads default xlnet-base-cased.
                  Pass checkpoint tokenizer to use memory tokens properly.
        max_n_segs: Maximum segments per document.
        use_streaming: Whether to use streaming processing for memory efficiency.
                      Automatically enabled for large datasets (>1000 examples).
        max_memory_gb: Maximum memory usage in GB (for streaming mode).

    Returns:
        Number of processed features

    Note:
        The function automatically detects memory-enabled tokenizers
        (those with >32000 tokens) and includes this in the cache key
        to avoid conflicts between standard and memory-enabled processing.

        Streaming mode is recommended for large datasets to avoid OOM errors.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Create cache manager
    cache_manager = ChunkedCacheManager(cache_dir, streaming_chunk_size)

    # For memory-enabled tokenizers, include tokenizer info in cache key
    cache_suffix = ""
    if tokenizer is not None and len(tokenizer) > 32000:  # Has memory tokens
        cache_suffix = f"_mem{len(tokenizer) - 32000}"

    # Schema/versioning: bump when feature fields change (e.g., added question/chosen_answer_text)
    FEATURE_SCHEMA_VERSION = 1  # increment to invalidate old caches
    cache_suffix += f"_v{FEATURE_SCHEMA_VERSION}"

    # Check if already cached (with tokenizer-specific cache)
    modified_dataset_name = f"{dataset_name}{cache_suffix}"
    if cache_manager.cache_exists(modified_dataset_name, split):
        # Count total features from existing cache
        total_features = 0
        total_chunks = cache_manager.get_total_chunks(modified_dataset_name, split)
        for chunk_id in range(total_chunks):
            chunk_data = cache_manager.load_chunk(modified_dataset_name, split, chunk_id)
            total_features += len(chunk_data)
        return total_features

    # If not cached, create dataset and cache features
    try:
        if tokenizer is None:
            from transformers import XLNetTokenizerFast

            tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")

        # Decide whether to use streaming
        # Force streaming for large datasets or when explicitly requested
        force_streaming = max_examples is None or max_examples > 1000
        should_use_streaming = use_streaming or force_streaming

        if should_use_streaming:
            # Use memory-efficient streaming processing
            from memxlnet.data.streaming import StreamingSquadProcessor

            print(f"🌊 Using streaming mode for {dataset_name} ({split})")
            print(f"   Max examples: {max_examples if max_examples else 'all'}")
            print(f"   Streaming chunk size: {streaming_chunk_size}")
            print(f"   Max memory: {max_memory_gb} GB")

            processor = StreamingSquadProcessor(
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                streaming_chunk_size=streaming_chunk_size,
                max_memory_gb=max_memory_gb,
            )

            total_features = processor.process_dataset_streaming(
                dataset_name=dataset_name,
                split=split,
                max_examples=max_examples,
                cache_manager=cache_manager,
                max_n_segs=max_n_segs,
            )

            return total_features

        else:
            # Use traditional in-memory processing for small datasets
            print(f"📚 Using traditional processing for {dataset_name} ({split})")
            print(f"   Max examples: {max_examples}")

            dataset = SquadLikeQADataset(
                split=split,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_examples=max_examples,
                dataset_name=dataset_name,
                max_n_segs=max_n_segs,
            )

            # Save as single chunk for simplicity
            features = [dataset[i] for i in range(len(dataset))]
            cache_manager.save_chunk(features, modified_dataset_name, split, 0)

            return len(features)

    except Exception as e:
        print(f"⚠️ Error during processing: {e}")
        # Fallback: return estimated count
        return max_examples if max_examples is not None else 1000


def create_dataset_from_cache(
    dataset_name: str,
    split: str,
    cache_dir: str,
    max_examples: int | None,
    max_seq_length: int,
    doc_stride: int,
    max_n_segs: int | None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    hub_dataset_id: str | None = None,
    use_hub_dataset: bool = True,
    hub_token: str | None = None,
    use_lazy_loading: bool = False,
):
    """Create (or load) a SquadLikeQADataset using Hub-first, then cache-first strategy.

    Loading priority:
      1. If hub_dataset_id provided and use_hub_dataset=True => try downloading from Hub first
      2. If cached chunk files exist => load from local cache
      3. Else process + cache via `process_and_cache_dataset`, then load
      4. If anything fails, fall back to on-the-fly processing (slow path) with a warning

    Args:
        dataset_name: Base HuggingFace dataset name (e.g. 'squad_v2')
        split: Split name ('train', 'validation')
        cache_dir: Directory holding chunk_*.cache files
        max_examples: Optional cap on number of raw examples processed (honored only when building)
        max_seq_length: Segment length used during (re)processing (for compatibility info only)
        doc_stride: Overlap size used during processing
        max_n_segs: Optional cap on segments per document (used only when building)
        tokenizer: Tokenizer (memory-token aware). Loads default XLNet if None.
        hub_dataset_id: Optional Hub repository ID for preprocessed dataset
        use_hub_dataset: Whether to try loading from Hub first (default: True)
        hub_token: HuggingFace token for Hub access
        use_lazy_loading: Whether to use lazy loading (loads features on-demand).
                         Recommended for large datasets to reduce memory usage.

    Returns:
        SquadLikeQADataset or LazySquadLikeQADataset (from Hub, cache, or freshly processed)
    """
    from pathlib import Path

    # Ensure tokenizer
    if tokenizer is None:
        from transformers import XLNetTokenizerFast

        tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")

    # 0. Try loading from HuggingFace Hub first (if configured)
    if hub_dataset_id and use_hub_dataset:
        print(f"[create_dataset_from_cache] Hub dataset configured: {hub_dataset_id}")
        hub_dataset = load_dataset_from_hub(
            hub_dataset_id=hub_dataset_id,
            split=split,
            hub_token=hub_token,
        )
        if hub_dataset is not None:
            print("[create_dataset_from_cache] ✅ Successfully loaded from Hub!")
            return hub_dataset
        else:
            print("[create_dataset_from_cache] Hub dataset not found, falling back to local processing...")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Memory-token aware suffix so standard & memory-augmented caches don't collide
    cache_suffix = ""
    if len(tokenizer) > 32000:  # heuristic: memory tokens appended beyond base vocab
        cache_suffix = f"_mem{len(tokenizer) - 32000}"
    FEATURE_SCHEMA_VERSION = 1
    cache_suffix += f"_v{FEATURE_SCHEMA_VERSION}"
    cache_key = f"{dataset_name}{cache_suffix}"

    cache_manager = ChunkedCacheManager(str(cache_path))
    print(f"[create_dataset_from_cache] Checking local cache for '{cache_key}' ({split}) in {cache_path} ...")

    def _load_all_chunks() -> SquadLikeQADataset | LazySquadLikeQADataset | None:
        if not cache_manager.cache_exists(cache_key, split):
            return None

        if use_lazy_loading:
            # Use lazy loading for memory efficiency
            print("[create_dataset_from_cache] Using lazy loading mode...")
            ds = LazySquadLikeQADataset(
                cache_manager=cache_manager,
                dataset_name=cache_key,
                split=split,
                cache_params={"max_seq_length": max_seq_length, "doc_stride": doc_stride},
            )
            return ds
        else:
            # Load all features into memory
            total_chunks = cache_manager.get_total_chunks(cache_key, split)
            print(f"[create_dataset_from_cache] Cache hit: {total_chunks} chunk(s) detected. Loading ...")
            features: list[dict[str, Any]] = []
            for cid in range(total_chunks):
                shard = cache_manager.load_chunk(cache_key, split, cid)
                if shard:
                    features.extend(shard)
            if not features:
                print("[create_dataset_from_cache] Warning: chunk files were empty; treating as cache miss.")
                return None
            # _reconstruct_squad_dataset_from_features always returns SquadLikeQADataset
            eager_ds = _reconstruct_squad_dataset_from_features(features)
            print(
                f"[create_dataset_from_cache] Reconstructed dataset with {len(eager_ds.features)} features across {len(eager_ds.document_map)} documents (local cache)."
            )
            return eager_ds

    # 1. Try loading existing local cache
    ds = _load_all_chunks()
    if ds is not None:
        return ds

    # 2. Build cache then load
    print(f"[create_dataset_from_cache] Cache miss for '{cache_key}'. Building now (this may take a while)...")
    try:
        process_and_cache_dataset(
            dataset_name=dataset_name,
            split=split,
            cache_dir=str(cache_path),
            max_examples=max_examples,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            streaming_chunk_size=2000,
            tokenizer=tokenizer,
            max_n_segs=max_n_segs,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[create_dataset_from_cache] ERROR during build: {e}. Falling back to direct processing.")
        return SquadLikeQADataset(
            split=split,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_examples=max_examples,
            dataset_name=dataset_name,
            max_n_segs=max_n_segs,
        )

    # 3. Load again
    ds = _load_all_chunks()
    if ds is not None:
        return ds

    # 4. Final fallback (unexpected)
    print("[create_dataset_from_cache] Fallback: cache build completed but reload failed; processing in-memory.")
    return SquadLikeQADataset(
        split=split,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_examples=max_examples,
        dataset_name=dataset_name,
        max_n_segs=max_n_segs,
    )


def _reconstruct_squad_dataset_from_features(features: list[dict[str, Any]]) -> SquadLikeQADataset:
    """Rebuild a minimal SquadLikeQADataset instance from cached feature dicts.

    The cached features are the per-item outputs of `SquadLikeQADataset.__getitem__`.
    We synthesize a new dataset object without re-tokenizing or reprocessing.

    Args:
        features: List of feature dicts (must include 'example_id').

    Returns:
        An object mimicking a fully processed SquadLikeQADataset, suitable for
        regular and time-step-major dataloaders.
    """
    ds = object.__new__(SquadLikeQADataset)  # type: ignore
    ds.features = features
    document_map: dict[str, list[int]] = {}
    for idx, feat in enumerate(features):
        ex_id = feat.get("example_id")
        if ex_id is None:
            continue
        document_map.setdefault(ex_id, []).append(idx)
    ds.document_map = document_map
    return ds


def _memory_aware_collate_fn(batch, memory_collate_config: MemoryCollateConfig | None = None):
    """Enhanced collate function for time-step-major batching with memory support.

    This function collates batches for both regular and time-step-major processing,
    handling tensor stacking, metadata preservation, and document tracking.

    Args:
        batch: List of feature dictionaries from dataset
        memory_collate_config: Configuration for memory token handling (optional)

    Returns:
        Dictionary containing:
            - Tensor fields: Stacked tensors (input_ids, attention_mask, etc.)
            - Metadata fields: Lists of metadata (example_id, context, etc.)
            - document_mask: Boolean tensor indicating active documents
            - example_ids: List of example identifiers

    Note:
        The function automatically adds document_mask and example_ids for
        time-step-major processing compatibility.
    """
    if not batch:
        return {}

    collated: dict[str, Any] = {}

    # Handle tensor fields
    tensor_keys = ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]
    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([x[key] for x in batch], dim=0)

    # Handle metadata fields
    metadata_keys = [
        "example_id",
        "segment_index",
        "total_segments",
        "offset_mapping",
        "context",
        "question",
        "cls_index",
        "has_answer",
        "chosen_answer_text",
        "chosen_answer_char_span",
        "boundary_info",
    ]
    for key in metadata_keys:
        if key in batch[0]:
            # Check if all items have this key, otherwise use None for missing ones
            collated[key] = [x.get(key, None) for x in batch]

    # Add document_mask for time-step-major processing
    if "example_id" in collated:
        # All examples in a regular batch are active
        collated["document_mask"] = torch.ones(len(batch), dtype=torch.bool)
        collated["example_ids"] = collated["example_id"]

    return collated


class LazySquadLikeQADataset(Dataset):
    """Memory-efficient lazy-loading dataset for SQuAD-like QA data.

    This dataset loads processed features on-demand from cached chunks,
    avoiding the need to load all features into memory at once. This
    enables training on datasets that are larger than available RAM.

    Key Features:
        - On-demand loading from disk cache
        - Minimal memory footprint (only metadata in RAM)
        - Compatible with TimeStepMajorDataLoader
        - Fast random access via chunk-based indexing
        - Full compatibility with memory tokens and all MemXLNet features

    Note:
        This dataset requires that features have already been processed
        and cached using process_and_cache_dataset().
    """

    def __init__(
        self,
        cache_manager: ChunkedCacheManager,
        dataset_name: str,
        split: str,
        cache_params: dict[str, Any],
    ):
        """Initialize lazy-loading dataset from cached features.

        Args:
            cache_manager: ChunkedCacheManager instance for loading features
            dataset_name: Name of the dataset (cache key prefix)
            split: Dataset split name
            cache_params: Additional cache parameters (for cache key generation)
        """
        self.cache_manager = cache_manager
        self.dataset_name = dataset_name
        self.split = split
        self.cache_params = cache_params

        # Build lightweight index: feature_idx -> (chunk_id, position_in_chunk)
        self._feature_index: list[tuple[int, int]] = []
        self.document_map: dict[str, list[int]] = {}

        # Load all chunks to build index
        total_chunks = cache_manager.get_total_chunks(dataset_name, split)
        feature_idx = 0

        print(f"💤 Building lazy dataset index for {dataset_name} ({split})...")

        for chunk_id in range(total_chunks):
            chunk_data = cache_manager.load_chunk(dataset_name, split, chunk_id)

            for pos_in_chunk, feature in enumerate(chunk_data):
                # Record chunk location for this feature
                self._feature_index.append((chunk_id, pos_in_chunk))

                # Build document map
                example_id = feature.get("example_id")
                if example_id:
                    if example_id not in self.document_map:
                        self.document_map[example_id] = []
                    self.document_map[example_id].append(feature_idx)

                feature_idx += 1

        print(f"✅ Lazy dataset ready: {len(self._feature_index)} features, {len(self.document_map)} documents")
        print(f"   Memory footprint: {len(self._feature_index) * 16 / 1024 / 1024:.1f} MB (index only)")

        # Cache for recently accessed chunks
        self._chunk_cache: dict[int, list[dict[str, Any]]] = {}
        self._max_cached_chunks = 3  # Keep last 3 chunks in memory

    def __len__(self) -> int:
        return len(self._feature_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load feature on-demand from cache.

        Args:
            idx: Feature index

        Returns:
            Feature dictionary with tensors and metadata
        """
        chunk_id, pos_in_chunk = self._feature_index[idx]

        # Check if chunk is in cache
        if chunk_id not in self._chunk_cache:
            # Load chunk from disk
            chunk_data = self.cache_manager.load_chunk(self.dataset_name, self.split, chunk_id)
            self._chunk_cache[chunk_id] = chunk_data

            # Limit cache size
            if len(self._chunk_cache) > self._max_cached_chunks:
                # Remove oldest chunk (simple FIFO)
                oldest_chunk_id = next(iter(self._chunk_cache))
                del self._chunk_cache[oldest_chunk_id]

        # Get feature from cached chunk
        feature = self._chunk_cache[chunk_id][pos_in_chunk]

        # Convert to tensor format (same as SquadLikeQADataset.__getitem__)
        item = {}

        # Convert tensor fields
        for key in ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]:
            if key in feature:
                # Feature might already be a tensor (from cache) or a list
                if isinstance(feature[key], torch.Tensor):
                    item[key] = feature[key]
                else:
                    item[key] = torch.tensor(feature[key])

        # Keep metadata as-is
        metadata_keys = [
            "example_id",
            "segment_index",
            "total_segments",
            "offset_mapping",
            "context",
            "question",
            "cls_index",
            "has_answer",
            "chosen_answer_text",
            "chosen_answer_char_span",
            "boundary_info",
        ]
        for key in metadata_keys:
            if key in feature:
                item[key] = feature[key]

        return item

    def get_document_segments(self, example_id: str) -> list[int]:
        """Get all segment indices for a document.

        Args:
            example_id: Document identifier

        Returns:
            List of feature indices belonging to this document
        """
        return list(self.document_map.get(example_id, []))

    def get_all_documents(self) -> list[str]:
        """Get all document IDs.

        Returns:
            List of all document identifiers in the dataset
        """
        return list(self.document_map.keys())


class TimeStepMajorDataLoader:
    """
    DataLoader that reorganizes regular batches into time-step-major format.

    Instead of processing documents sequentially, this processes all first segments together,
    then all second segments together, etc. This enables proper memory state propagation
    across document segments while maintaining batch efficiency.

    Key Concepts:
        - Regular batching: [doc1_seg1, doc2_seg1, doc3_seg1] -> [doc1_seg2, doc2_seg2, doc3_seg2]
        - Time-step-major: [doc1_seg1, doc2_seg1, doc3_seg1] -> [doc1_seg2, doc2_seg2, doc3_seg2]
          where memory states from first batch are propagated to second batch

    Attributes:
        dataset: SquadLikeQADataset with document tracking
        batch_size: Number of documents per batch
        shuffle: Whether to shuffle documents
        max_segments: Maximum segments to process per document
    """

    def __init__(
        self,
        dataset: SquadLikeQADataset | LazySquadLikeQADataset,
        batch_size: int,
        shuffle: bool = False,
        max_segments: int | None = None,
    ):
        """Initialize time-step-major dataloader.

        Args:
            dataset: SquadLikeQADataset or LazySquadLikeQADataset with document tracking
            batch_size: Number of documents to process simultaneously
            shuffle: Whether to shuffle document order
            max_segments: Maximum segments per document (None for all)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_segments = max_segments

        # Organize documents by segment count
        self.documents = dataset.get_all_documents()
        self.document_segments = {}
        self.max_doc_segments = 0

        for doc_id in self.documents:
            segments = dataset.get_document_segments(doc_id)
            self.document_segments[doc_id] = segments
            self.max_doc_segments = max(self.max_doc_segments, len(segments))

        if self.max_segments:
            self.max_doc_segments = min(self.max_doc_segments, self.max_segments)

    def __iter__(self):
        documents = self.documents.copy()
        if self.shuffle:
            import random

            random.shuffle(documents)

        # Process documents in batches
        for batch_start in range(0, len(documents), self.batch_size):
            batch_docs = documents[batch_start : batch_start + self.batch_size]

            # Create time-step-major batches for this document batch
            time_step_batches = []

            for time_step in range(self.max_doc_segments):
                step_batch = []
                step_example_ids = []
                step_document_mask = []

                for doc_id in batch_docs:
                    segments = self.document_segments[doc_id]

                    if time_step < len(segments):
                        # This document has a segment at this time step
                        feature_idx = segments[time_step]
                        feature = self.dataset[feature_idx]
                        step_batch.append(feature)
                        step_example_ids.append(doc_id)
                        step_document_mask.append(True)
                    else:
                        # This document doesn't have a segment at this time step
                        # Add padding entry
                        if step_batch:  # Only if we have at least one real example
                            padding_feature = self._create_padding_feature(step_batch[0])
                            step_batch.append(padding_feature)
                            step_example_ids.append(f"padding_{doc_id}")
                            step_document_mask.append(False)

                if step_batch:
                    # Collate this time step batch
                    collated = _memory_aware_collate_fn(step_batch)
                    collated["example_ids"] = step_example_ids
                    collated["document_mask"] = torch.tensor(step_document_mask, dtype=torch.bool)
                    time_step_batches.append(collated)

            if time_step_batches:
                yield time_step_batches

    def _create_padding_feature(self, template_feature):
        """Create a padding feature based on a template.

        When documents have different numbers of segments, shorter documents
        need padding entries to maintain batch structure.

        Args:
            template_feature: Feature to use as template for tensor shapes

        Returns:
            Padding feature with zero tensors and default metadata
        """
        padding = {}

        for key, value in template_feature.items():
            if isinstance(value, torch.Tensor):
                # Create zero tensor with same shape as a base
                padding[key] = torch.zeros_like(value)
            else:
                # Metadata defaults
                if key == "example_id":
                    padding[key] = "padding"
                elif key == "segment_index":
                    padding[key] = 0
                elif key == "total_segments":
                    padding[key] = 0
                elif key == "context":
                    padding[key] = ""
                elif key == "offset_mapping":
                    padding[key] = []
                elif key == "question":
                    padding[key] = ""
                elif key == "has_answer":
                    padding[key] = False
                elif key == "chosen_answer_text":
                    padding[key] = ""
                elif key == "chosen_answer_char_span":
                    padding[key] = [-1, -1]
                elif key == "boundary_info":
                    padding[key] = None
                elif key == "cls_index":
                    # If template has cls_index keep it; else 0
                    padding[key] = value
                else:
                    padding[key] = value

        # Override start/end to CLS index to avoid accidental label leakage
        cls_index = template_feature.get("cls_index", 0)
        # start_positions / end_positions may be tensors or ints depending on __getitem__ implementation
        if isinstance(template_feature.get("start_positions"), torch.Tensor):
            dtype = template_feature["start_positions"].dtype
            device = template_feature["start_positions"].device
            padding["start_positions"] = torch.tensor(cls_index, dtype=dtype, device=device)
        else:
            padding["start_positions"] = cls_index
        if isinstance(template_feature.get("end_positions"), torch.Tensor):
            dtype = template_feature["end_positions"].dtype
            device = template_feature["end_positions"].device
            padding["end_positions"] = torch.tensor(cls_index, dtype=dtype, device=device)
        else:
            padding["end_positions"] = cls_index

        return padding

    def __len__(self):
        return (len(self.documents) + self.batch_size - 1) // self.batch_size


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    memory_collate_config: MemoryCollateConfig | None = None,
    use_time_step_major: bool = True,
) -> DataLoader[Any] | TimeStepMajorDataLoader:
    """Create dataloader with optional time-step-major batching.

    This function creates different types of dataloaders based on the configuration:
    - TimeStepMajorDataLoader for MA-XLNet (memory-enabled training)
    - Regular PyTorch DataLoader for standard processing

    Args:
        dataset: Dataset to create dataloader for
        batch_size: Batch size for loading
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading (only used for regular DataLoader,
                     ignored when use_time_step_major=True)
        memory_collate_config: Configuration for memory token handling (only used for
                               regular DataLoader, ignored when use_time_step_major=True)
        use_time_step_major: Whether to use time-step-major batching (for MA-XLNet)

    Returns:
        DataLoader instance (time-step-major or regular)

    Note:
        - Time-step-major batching is used for MA-XLNet when use_time_step_major=True
        - Requires SquadLikeQADataset instance for time-step-major batching
        - When using time-step-major mode, num_workers and memory_collate_config are ignored
          as TimeStepMajorDataLoader has its own internal batching mechanism
    """

    # Check if dataset supports TimeStepMajorDataLoader interface
    # (has get_all_documents and get_document_segments methods)
    supports_time_step_major = (
        isinstance(dataset, (SquadLikeQADataset, LazySquadLikeQADataset))
        or hasattr(dataset, "get_all_documents")
        and hasattr(dataset, "get_document_segments")
    )

    if use_time_step_major and supports_time_step_major:
        # Return time-step-major dataloader for memory-enabled training
        # Works with SquadLikeQADataset, LazySquadLikeQADataset, and ChunkedDataset
        # Type cast is safe here because supports_time_step_major verifies compatibility
        return TimeStepMajorDataLoader(
            dataset=cast(SquadLikeQADataset | LazySquadLikeQADataset, dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            max_segments=None,  # Use all segments
        )
    else:
        # Return regular dataloader
        def collate_fn(batch):
            return _memory_aware_collate_fn(batch, memory_collate_config)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


def create_evaluation_dataloader(
    dataset_name: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 384,
    doc_stride: int = 64,
    batch_size: int = 8,
    max_examples: int | None = None,
    max_n_segs: int | None = None,
    cache_dir: str = "./.cache",
    use_time_step_major: bool = True,
) -> tuple[SquadLikeQADataset, DataLoader[Any] | TimeStepMajorDataLoader]:
    """
    Convenience function to create evaluation dataset and dataloader using proper pipeline.

    This is the recommended function for setting up evaluation pipelines. It handles
    the complete workflow from raw data to ready-to-use dataloader, with proper
    caching and memory token support.

    Pipeline Steps:
        1. Process and cache dataset if needed (with memory token awareness)
        2. Create dataset from processed features
        3. Create appropriate dataloader (time-step-major for memory models)

    Args:
        dataset_name: Name of dataset (e.g., "squad_v2")
        split: Dataset split (e.g., "validation")
        tokenizer: Tokenizer to use (should include memory tokens if applicable)
        max_seq_length: Maximum sequence length including special tokens
        doc_stride: Document stride for overlapping segments
        batch_size: Batch size for dataloader
        max_examples: Maximum examples to process (None for all)
        max_n_segs: Maximum segments per document (None for unlimited)
        cache_dir: Directory for caching processed data
        use_time_step_major: Whether to use time-step-major batching

    Returns:
        Tuple of (dataset, dataloader) ready for evaluation

    Example:
        >>> from transformers import XLNetTokenizerFast
        >>> from src.memxlnet_qa import MemXLNetForQA
        >>>
        >>> # Load model and tokenizer
        >>> model = MemXLNetForQA.from_pretrained('path/to/checkpoint')
        >>> tokenizer = XLNetTokenizerFast.from_pretrained('path/to/checkpoint')
        >>>
        >>> # Create evaluation dataloader
        >>> eval_dataset, eval_dataloader = create_evaluation_dataloader(
        ...     dataset_name='squad_v2',
        ...     split='validation',
        ...     tokenizer=tokenizer,
        ...     batch_size=8,
        ...     use_time_step_major=True
        ... )
        >>>
        >>> # Ready for evaluation
        >>> for time_step_batches in eval_dataloader:
        ...     # Process time-step-major batches with memory
        ...     pass

    Note:
        This function automatically detects memory tokens in the tokenizer
        and uses appropriate caching to avoid conflicts between standard
        and memory-enabled processing.
    """
    # Process and cache dataset
    print(f"Processing and caching {dataset_name} {split}...")
    total_features = process_and_cache_dataset(
        dataset_name=dataset_name,
        split=split,
        cache_dir=cache_dir,
        max_examples=max_examples,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        streaming_chunk_size=2000,
        tokenizer=tokenizer,
        max_n_segs=max_n_segs,
    )
    print(f"✓ Processed {total_features} features")

    # Create dataset from cache
    print("Creating dataset from cache...")
    eval_dataset = create_dataset_from_cache(
        dataset_name=dataset_name,
        split=split,
        cache_dir=cache_dir,
        max_examples=max_examples,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_n_segs=max_n_segs,
        tokenizer=tokenizer,
    )
    print(f"✓ Dataset created: {len(eval_dataset)} features")

    # Create dataloader
    print("Creating dataloader...")
    eval_dataloader = create_dataloader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues in notebooks
        use_time_step_major=use_time_step_major,
    )
    print(f"✓ DataLoader created: {len(eval_dataloader)} batches")

    return eval_dataset, eval_dataloader


def upload_processed_dataset_to_hub(
    dataset_name: str,
    split: str,
    hub_dataset_id: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 384,
    doc_stride: int = 64,
    max_examples: int | None = None,
    max_n_segs: int | None = None,
    cache_dir: str = "./.cache",
    hub_token: str | None = None,
    private: bool = False,
) -> str:
    """Process SQuAD-like dataset and upload preprocessed features to HuggingFace Hub.

    This function processes a raw dataset (e.g., SQuAD v2) into segments with
    memory tokens, then uploads the preprocessed features to HuggingFace Hub
    for fast downloading during training. This avoids expensive reprocessing.

    Args:
        dataset_name: Source dataset name (e.g., "squad_v2")
        split: Dataset split (e.g., "train", "validation")
        hub_dataset_id: Target Hub repository (e.g., "username/memxlnet-squad-processed")
        tokenizer: Tokenizer with memory tokens configured
        max_seq_length: Maximum sequence length
        doc_stride: Document stride for overlapping segments
        max_examples: Maximum examples to process (None for all)
        max_n_segs: Maximum segments per document
        cache_dir: Temporary cache directory for processing
        hub_token: HuggingFace token (defaults to HF_TOKEN env var)
        private: Create private repository on Hub

    Returns:
        Hub dataset URL

    Example:
        >>> from transformers import XLNetTokenizerFast
        >>> from memxlnet.data import configure_memory_tokens, upload_processed_dataset_to_hub
        >>>
        >>> # Setup tokenizer with memory tokens
        >>> tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
        >>> configure_memory_tokens(tokenizer, memory_num_tokens=8)
        >>>
        >>> # Process and upload
        >>> url = upload_processed_dataset_to_hub(
        ...     dataset_name="squad_v2",
        ...     split="train",
        ...     hub_dataset_id="username/memxlnet-squad-train-mem8",
        ...     tokenizer=tokenizer,
        ...     max_seq_length=384,
        ...     doc_stride=64,
        ... )
        >>> print(f"Uploaded to: {url}")
    """
    from datasets import Dataset as HFDataset

    print(f"🔄 Processing {dataset_name} {split} for upload to Hub...")
    print(f"   Target repository: {hub_dataset_id}")

    # Process dataset locally first
    print("   Step 1/3: Processing raw dataset...")
    dataset = SquadLikeQADataset(
        split=split,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_examples=max_examples,
        dataset_name=dataset_name,
        max_n_segs=max_n_segs,
    )

    # Convert to HuggingFace Dataset format
    print(f"   Step 2/3: Converting {len(dataset)} features to HuggingFace format...")
    features_list = []
    for i in range(len(dataset)):
        feature = dataset[i]
        # Convert tensors to lists for serialization
        serializable_feature = {}
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                serializable_feature[key] = value.tolist()
            else:
                serializable_feature[key] = value
        features_list.append(serializable_feature)

    # Create HuggingFace Dataset
    hf_dataset = HFDataset.from_list(features_list)

    # Add metadata about preprocessing configuration
    dataset_card = f"""# Preprocessed {dataset_name.upper()} Dataset for MemXLNet

This dataset contains preprocessed features from {dataset_name} ({split} split) for MemXLNet training.

## Preprocessing Configuration

- **Source dataset**: {dataset_name}
- **Split**: {split}
- **Max sequence length**: {max_seq_length}
- **Document stride**: {doc_stride}
- **Max segments**: {max_n_segs if max_n_segs else "unlimited"}
- **Memory tokens**: {len(tokenizer) - 32000 if len(tokenizer) > 32000 else 0}
- **Total features**: {len(features_list)}
- **Total documents**: {len(dataset.document_map)}

## Usage

```python
from datasets import load_dataset
from memxlnet.data import create_dataset_from_cache

# Download preprocessed dataset
dataset = load_dataset("{hub_dataset_id}", split="{split}")

# Or use with MemXLNet training
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

config = TrainingConfig(
    hub_dataset_id="{hub_dataset_id}",
    use_hub_dataset=True,
)
trainer = XLNetRecurrentTrainer(config)
trainer.train()
```

## Benefits

- **No preprocessing required**: Dataset is ready to use
- **Lower RAM requirements**: Skip expensive tokenization step
- **Faster training startup**: Download preprocessed data in minutes
- **Consistent features**: All users get identical preprocessing

Generated with MemXLNet preprocessing pipeline.
"""

    # Upload to Hub
    print(f"   Step 3/3: Uploading to HuggingFace Hub ({hub_dataset_id})...")
    hf_dataset.push_to_hub(
        hub_dataset_id,
        split=split,
        token=hub_token,
        private=private,
    )

    # Upload README separately
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hub_token)
        api.upload_file(
            path_or_fileobj=dataset_card.encode(),
            path_in_repo="README.md",
            repo_id=hub_dataset_id,
            repo_type="dataset",
        )
    except Exception as e:
        print(f"   Warning: Could not upload README: {e}")

    hub_url = f"https://huggingface.co/datasets/{hub_dataset_id}"
    print(f"✅ Successfully uploaded to: {hub_url}")
    return hub_url


def load_dataset_from_hub(
    hub_dataset_id: str,
    split: str,
    hub_token: str | None = None,
) -> SquadLikeQADataset | None:
    """Load preprocessed dataset from HuggingFace Hub.

    Args:
        hub_dataset_id: Hub repository ID (e.g., "username/memxlnet-squad-processed")
        split: Dataset split to load (e.g., "train", "validation")
        hub_token: HuggingFace token (defaults to HF_TOKEN env var)

    Returns:
        SquadLikeQADataset if found on Hub, None if not found

    Example:
        >>> from memxlnet.data import load_dataset_from_hub
        >>>
        >>> dataset = load_dataset_from_hub(
        ...     hub_dataset_id="username/memxlnet-squad-train-mem8",
        ...     split="train",
        ... )
        >>> if dataset:
        ...     print(f"Loaded {len(dataset)} features from Hub")
    """
    try:
        from datasets import load_dataset as hf_load_dataset

        print(f"📥 Downloading preprocessed dataset from Hub: {hub_dataset_id}")
        print(f"   Split: {split}")

        # Download from Hub
        hf_dataset = hf_load_dataset(
            hub_dataset_id,
            split=split,
            token=hub_token,
        )

        print(f"✅ Downloaded {len(hf_dataset)} features from Hub")

        # Convert back to feature format
        features = []
        for item in hf_dataset:
            feature = {}
            for key, value in item.items():
                # Convert lists back to tensors where appropriate
                if key in ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]:
                    feature[key] = torch.tensor(value)
                else:
                    feature[key] = value
            features.append(feature)

        # Reconstruct SquadLikeQADataset
        dataset = _reconstruct_squad_dataset_from_features(features)
        print(f"✅ Reconstructed dataset: {len(dataset.features)} features, {len(dataset.document_map)} documents")

        return dataset

    except Exception as e:
        print(f"⚠️  Could not load dataset from Hub: {e}")
        print(f"   Hub dataset '{hub_dataset_id}' not found or inaccessible")
        print("   Falling back to local processing...")
        return None


__all__ = [
    # Core data structures
    "MemoryCollateConfig",
    "SquadLikeQADataset",
    "LazySquadLikeQADataset",
    "ChunkedCacheManager",
    "TimeStepMajorDataLoader",
    # Memory token integration
    "configure_memory_tokens",
    # Dataset processing pipeline
    "process_and_cache_dataset",
    "create_dataset_from_cache",
    "create_dataloader",
    "create_evaluation_dataloader",
    # HuggingFace Hub integration
    "upload_processed_dataset_to_hub",
    "load_dataset_from_hub",
]
