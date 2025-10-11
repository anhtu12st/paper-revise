"""Memory-efficient streaming processor for SQuAD-like QA datasets.

This module provides streaming capabilities for processing large datasets
that don't fit in memory, while maintaining full compatibility with
MemXLNet's memory token system.
"""

import gc
import logging
from typing import Any

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from memxlnet.data.text_utils import (
    choose_best_occurrence,
    find_all_occurrences,
    find_answer_span_with_normalization,
    fix_answer_positions,
    normalize_unicode,
    validate_answer_positions,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import psutil for memory monitoring (optional)
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.info("psutil not installed - memory monitoring disabled")


class StreamingSquadProcessor:
    """Memory-efficient processor that uses streaming to handle large datasets.

    This processor can handle datasets larger than available RAM by:
    - Loading data in streaming mode from HuggingFace datasets
    - Processing examples in configurable chunks
    - Saving processed chunks incrementally to avoid memory accumulation
    - Supporting adaptive memory management with optional psutil monitoring

    Key features:
    - Full compatibility with memory tokens ([MEM_READ_i], [MEM_WRITE_i])
    - Robust answer span mapping with multiple fallback strategies
    - Unicode normalization for international text support
    - Configurable memory limits with automatic cleanup
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 384,
        doc_stride: int = 64,
        streaming_chunk_size: int = 1000,
        max_memory_gb: float = 8.0,
    ):
        """Initialize streaming processor.

        Args:
            tokenizer: Tokenizer (should include memory tokens if using MemXLNet)
            max_seq_length: Maximum sequence length for segments
            doc_stride: Overlap between consecutive segments
            streaming_chunk_size: Number of examples to process at once
            max_memory_gb: Maximum memory usage in GB before triggering cleanup
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.streaming_chunk_size = streaming_chunk_size
        self.max_memory_gb = max_memory_gb

        # Track if tokenizer has memory tokens
        self.has_memory_tokens = len(tokenizer) > 32000
        if self.has_memory_tokens:
            logger.info(f"✓ Memory tokens detected: {len(tokenizer) - 32000} memory token pairs")

        logger.info("🚀 Initialized StreamingSquadProcessor")
        logger.info(f"   Max sequence length: {max_seq_length}")
        logger.info(f"   Doc stride: {doc_stride}")
        logger.info(f"   Streaming chunk size: {streaming_chunk_size}")
        logger.info(f"   Max memory limit: {max_memory_gb} GB")
        logger.info(f"   Memory monitoring: {'enabled' if PSUTIL_AVAILABLE else 'disabled'}")

    def process_dataset_streaming(
        self,
        dataset_name: str,
        split: str = "train",
        max_examples: int | None = None,
        cache_manager: Any | None = None,
        max_n_segs: int | None = None,
    ) -> int:
        """Process a dataset using streaming mode for memory efficiency.

        Args:
            dataset_name: Name of the HuggingFace dataset (e.g., 'squad_v2')
            split: Dataset split to process ('train', 'validation')
            max_examples: Maximum examples to process (None for all)
            cache_manager: Cache manager for incremental saving
            max_n_segs: Maximum segments per document

        Returns:
            Number of processed features
        """
        logger.info(f"📚 Loading dataset in streaming mode: {dataset_name} ({split})")

        # Load dataset in streaming mode
        dataset = load_dataset(dataset_name, split=split, streaming=True)

        # Initialize incremental cache if available
        if cache_manager:
            cache_params = {
                "max_seq_length": self.max_seq_length,
                "doc_stride": self.doc_stride,
                "max_examples": max_examples,
                "tokenizer_name": self.tokenizer.name_or_path,
                "has_memory_tokens": self.has_memory_tokens,
                "max_n_segs": max_n_segs,
                "streaming": True,
            }
            cache_manager._initialize_incremental_cache(dataset_name, split, **cache_params)

        total_features = 0
        processed_count = 0
        chunk_buffer = []

        # Process dataset in chunks
        for example in dataset:
            chunk_buffer.append(example)

            # Process when chunk is full
            if len(chunk_buffer) >= self.streaming_chunk_size:
                chunk_features = self._process_streaming_chunk(chunk_buffer, max_n_segs)

                # Save chunk immediately to avoid memory accumulation
                if cache_manager:
                    cache_manager._save_features_chunk(chunk_features, dataset_name, split, **cache_params)

                total_features += len(chunk_features)
                processed_count += len(chunk_buffer)

                logger.info(
                    f"🔄 Processed {processed_count} examples, "
                    f"generated {len(chunk_features)} features "
                    f"(total: {total_features} features)"
                )

                # Clear buffer and collected features to free memory
                chunk_buffer = []
                chunk_features = []
                gc.collect()

                # Check memory usage and cleanup if needed
                self._manage_memory()

                # Stop if we've reached max_examples
                if max_examples and processed_count >= max_examples:
                    break

        # Process remaining examples in buffer
        if chunk_buffer:
            chunk_features = self._process_streaming_chunk(chunk_buffer, max_n_segs)

            if cache_manager:
                cache_manager._save_features_chunk(chunk_features, dataset_name, split, **cache_params)

            total_features += len(chunk_features)
            processed_count += len(chunk_buffer)
            logger.info(f"🔄 Processed final chunk: {len(chunk_buffer)} examples")

        # Finalize cache
        if cache_manager:
            cache_manager._finalize_incremental_cache(dataset_name, split, **cache_params)

        logger.info(f"✅ Streaming processing completed: {total_features} features from {processed_count} examples")

        return total_features

    def _process_streaming_chunk(
        self,
        examples: list[dict[str, Any]],
        max_n_segs: int | None = None,
    ) -> list[dict[str, Any]]:
        """Process a chunk of examples.

        Args:
            examples: List of raw examples to process
            max_n_segs: Maximum segments per document

        Returns:
            List of processed features
        """
        chunk_features = []

        for example in examples:
            features = self._process_single_example(example, max_n_segs)
            chunk_features.extend(features)

        return chunk_features

    def _process_single_example(
        self,
        example: dict[str, Any],
        max_n_segs: int | None = None,
    ) -> list[dict[str, Any]]:
        """Process a single QA example into segments.

        This method maintains full compatibility with the existing
        SquadLikeQADataset._process_example logic, including:
        - Robust answer span mapping with multiple fallbacks
        - Unicode normalization
        - Memory token support
        - Proper handling of unanswerable questions

        Args:
            example: Raw example with 'id', 'question', 'context', 'answers'
            max_n_segs: Maximum segments to generate

        Returns:
            List of feature dictionaries for each segment
        """
        # Normalize texts (strip leading spaces from question per SQuAD convention)
        question = normalize_unicode(example["question"].lstrip())
        context = normalize_unicode(example["context"])
        example_id = example["id"]

        # Check if question has answers
        answers = example.get("answers", {})
        has_answers = bool(answers.get("text", []))

        # Tokenize with overflow handling
        pad_on_right = self.tokenizer.padding_side == "right"

        tokenized = self.tokenizer(
            question if pad_on_right else context,
            context if pad_on_right else question,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping", None)  # noqa: F841 (reserved for future)
        offset_mapping = tokenized.pop("offset_mapping")

        features = []

        # Limit segments if specified
        num_segments = len(tokenized["input_ids"])
        if max_n_segs is not None:
            num_segments = min(num_segments, max_n_segs)

        # Get gold answer candidates
        gold_candidates = []
        if has_answers:
            answer_texts = answers.get("text", [])
            answer_starts = answers.get("answer_start", [])
            gold_candidates = list(zip(answer_texts, answer_starts))

        for i in range(num_segments):
            input_ids = tokenized["input_ids"][i]
            attention_mask = tokenized["attention_mask"][i]
            offsets = offset_mapping[i]
            sequence_ids = tokenized.sequence_ids(i)

            # Find CLS index (fallback to 0 if not found)
            cls_index = input_ids.index(self.tokenizer.cls_token_id) if self.tokenizer.cls_token_id in input_ids else 0

            # Default to CLS position for no-answer
            start_positions = cls_index
            end_positions = cls_index
            chosen_answer_text = ""
            chosen_answer_char_span = [-1, -1]
            boundary_info = None

            if has_answers:
                # Try to map answer using multiple fallback strategies
                chosen_start_char = None
                chosen_end_char = None
                chosen_true_start = None

                for raw_answer_text, raw_start in gold_candidates:
                    answer_text = normalize_unicode(raw_answer_text)
                    start_char = raw_start
                    end_char = start_char + len(raw_answer_text)

                    # Strategy 1: Validate current positions
                    if not validate_answer_positions(context, answer_text, start_char, end_char):
                        # Strategy 2: Try fixing positions
                        fixed_start, fixed_end = fix_answer_positions(context, answer_text, start_char)
                        if validate_answer_positions(context, answer_text, fixed_start, fixed_end):
                            start_char, end_char = fixed_start, fixed_end
                        else:
                            # Strategy 3: Find all occurrences and choose best
                            occs = find_all_occurrences(context, answer_text)
                            best = choose_best_occurrence(occs, raw_start)
                            if best:
                                start_char, end_char = best

                    if validate_answer_positions(context, answer_text, start_char, end_char):
                        chosen_start_char = start_char
                        chosen_end_char = end_char
                        chosen_true_start = raw_start
                        break

                # Final fallback: normalized search on first candidate
                if chosen_start_char is None and gold_candidates:
                    fallback_start, fallback_end = find_answer_span_with_normalization(context, gold_candidates[0][0])
                    if fallback_start is not None:
                        chosen_start_char, chosen_end_char = fallback_start, fallback_end
                        chosen_true_start = gold_candidates[0][1]

                # Map character positions to token positions
                if chosen_start_char is not None and chosen_end_char is not None:
                    # Find context token span
                    context_token_start = None
                    context_token_end = None
                    for idx, sid in enumerate(sequence_ids):
                        if sid == (1 if pad_on_right else 0):
                            if context_token_start is None:
                                context_token_start = idx
                            context_token_end = idx

                    if context_token_start is None or context_token_end is None:
                        context_token_start, context_token_end = 0, len(sequence_ids) - 1

                    # Check if answer is in this window
                    if (
                        offsets[context_token_start][0] <= chosen_start_char
                        and offsets[context_token_end][1] >= chosen_end_char
                    ):
                        # Find token start position
                        token_start = context_token_start
                        while token_start <= context_token_end and offsets[token_start][0] < chosen_start_char:
                            token_start += 1
                        if (
                            token_start > context_token_start
                            and token_start <= context_token_end
                            and offsets[token_start - 1][0] <= chosen_start_char <= offsets[token_start - 1][1]
                        ):
                            token_start -= 1

                        # Find token end position
                        token_end = context_token_end
                        while token_end >= context_token_start and offsets[token_end][1] > chosen_end_char:
                            token_end -= 1

                        # Validate span
                        if token_start > token_end:
                            start_positions = cls_index
                            end_positions = cls_index
                        else:
                            start_positions = max(context_token_start, min(token_start, context_token_end))
                            end_positions = max(context_token_start, min(token_end, context_token_end))

                            # Additional validation
                            try:
                                s_off = offsets[start_positions]
                                e_off = offsets[end_positions]
                                if s_off[0] is None or e_off[1] is None or s_off[0] >= e_off[1]:
                                    start_positions = cls_index
                                    end_positions = cls_index
                                else:
                                    chosen_answer_text = normalize_unicode(answer_text)
                                    chosen_answer_char_span = [chosen_start_char, chosen_end_char]
                            except Exception:
                                start_positions = cls_index
                                end_positions = cls_index

                        # Boundary diagnostics (for debugging/analysis)
                        if chosen_true_start is not None and chosen_start_char is not None:
                            from memxlnet.data.text_utils import characterize_boundary_delta

                            boundary_info = {
                                "delta_start_char": chosen_start_char - chosen_true_start,
                                "boundary_label": characterize_boundary_delta(chosen_start_char, chosen_true_start),
                            }

            # Create token type IDs
            token_type_ids = [0] * len(input_ids)
            for idx, sid in enumerate(sequence_ids):
                if sid == (1 if pad_on_right else 0):
                    token_type_ids[idx] = 1

            # Create feature dictionary
            feature = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "start_positions": start_positions,
                "end_positions": end_positions,
                "offset_mapping": offsets,
                "context": context,
                "question": question,
                "cls_index": cls_index,
                "has_answer": has_answers,
                "chosen_answer_text": chosen_answer_text,
                "chosen_answer_char_span": chosen_answer_char_span,
                "example_id": f"doc_{example_id}",
                "segment_index": i,
                "total_segments": num_segments,
            }

            if boundary_info is not None:
                feature["boundary_info"] = boundary_info

            features.append(feature)

        return features

    def _manage_memory(self):
        """Manage memory usage during processing."""
        if PSUTIL_AVAILABLE:
            try:
                # Get current memory usage
                process = psutil.Process()
                memory_gb = process.memory_info().rss / (1024**3)

                if memory_gb > self.max_memory_gb:
                    logger.warning(f"⚠️ Memory usage ({memory_gb:.1f} GB) exceeds limit ({self.max_memory_gb} GB)")
                    logger.info("🧹 Running garbage collection...")

                    # Force garbage collection
                    gc.collect()

                    # Check new memory usage
                    new_memory_gb = process.memory_info().rss / (1024**3)
                    logger.info(f"📉 Memory usage after cleanup: {new_memory_gb:.1f} GB")

            except Exception as e:
                logger.debug(f"Memory monitoring error: {e}")
                # Fall back to simple garbage collection
                gc.collect()
        else:
            # Run periodic garbage collection without monitoring
            gc.collect()


def create_streaming_processor(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 384,
    doc_stride: int = 64,
    streaming_chunk_size: int = 1000,
    max_memory_gb: float = 8.0,
) -> StreamingSquadProcessor:
    """Convenience function to create a streaming processor.

    Args:
        tokenizer: Tokenizer to use (should include memory tokens if applicable)
        max_seq_length: Maximum sequence length for segments
        doc_stride: Overlap between consecutive segments
        streaming_chunk_size: Number of examples to process at once
        max_memory_gb: Maximum memory usage in GB

    Returns:
        StreamingSquadProcessor instance
    """
    return StreamingSquadProcessor(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        streaming_chunk_size=streaming_chunk_size,
        max_memory_gb=max_memory_gb,
    )
