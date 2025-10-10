"""Test suite for multi-segment answer span validation.

This module tests scenarios where long contexts are split into multiple segments,
and answers may appear in different segments (not just the first segment).

Key scenarios tested:
1. Answer in first segment only
2. Answer in middle segment
3. Answer in last segment
4. Multiple occurrences of answer across segments
5. Answer spanning segment boundaries (should not be mapped)
6. Very long contexts with many segments
"""

import tempfile
from unittest.mock import patch

import pytest
import torch
from transformers import XLNetTokenizerFast

from memxlnet.data.dataset import (
    ChunkedCacheManager,
    LazySquadLikeQADataset,
    SquadLikeQADataset,
    configure_memory_tokens,
)
from memxlnet.data.streaming import StreamingSquadProcessor

# ============================================================================
# Helper Functions
# ============================================================================


def extract_answer_from_feature(feature: dict) -> str:
    """Extract answer text from a feature using token positions.

    Args:
        feature: Processed feature dictionary

    Returns:
        Extracted answer text, or empty string if no answer
    """
    start_pos = feature["start_positions"]
    end_pos = feature["end_positions"]

    # Convert tensor to int if needed
    if isinstance(start_pos, torch.Tensor):
        start_pos = start_pos.item()
    if isinstance(end_pos, torch.Tensor):
        end_pos = end_pos.item()

    # Check if this is a no-answer case (points to CLS)
    cls_index = feature.get("cls_index", 0)
    if start_pos == cls_index and end_pos == cls_index:
        return ""

    # Get offset mapping and context
    offsets = feature["offset_mapping"]
    context = feature["context"]

    # Extract character span from offsets
    try:
        start_char = offsets[start_pos][0]
        end_char = offsets[end_pos][1]

        # Handle None offsets (special tokens)
        if start_char is None or end_char is None:
            return ""

        return context[start_char:end_char]
    except (IndexError, TypeError):
        return ""


def validate_multi_segment_answer(
    dataset,
    example_id: str,
    expected_answer: str,
    expected_segment_index: int | None = None,
) -> dict:
    """Validate that an answer is correctly mapped in multi-segment document.

    Args:
        dataset: SquadLikeQADataset instance
        example_id: Document identifier
        expected_answer: Expected answer text
        expected_segment_index: Which segment should contain the answer (None = any)

    Returns:
        Dictionary with validation results
    """
    # Get all segments for this document
    segment_indices = dataset.get_document_segments(example_id)

    # Track which segments have the answer
    segments_with_answer = []
    segments_without_answer = []

    for seg_idx, feature_idx in enumerate(segment_indices):
        feature = dataset[feature_idx]
        extracted = extract_answer_from_feature(feature)

        # Check if this segment has the answer
        # Be lenient: check if expected answer is contained in extracted text
        # This handles cases where tokenization picks up adjacent characters
        extracted_normalized = extracted.strip().lower()
        expected_normalized = expected_answer.strip().lower()

        has_answer_in_segment = (
            extracted_normalized == expected_normalized or expected_normalized in extracted_normalized
        )

        if has_answer_in_segment:
            segments_with_answer.append(
                {
                    "segment_index": seg_idx,
                    "feature_index": feature_idx,
                    "extracted": extracted,
                }
            )
        else:
            segments_without_answer.append(
                {
                    "segment_index": seg_idx,
                    "feature_index": feature_idx,
                    "extracted": extracted,
                }
            )

    # Validation logic
    success = len(segments_with_answer) > 0

    # If expected_segment_index is specified, check that answer is in correct segment
    if expected_segment_index is not None and success:
        segment_indices_with_answer = [s["segment_index"] for s in segments_with_answer]
        success = expected_segment_index in segment_indices_with_answer

    return {
        "success": success,
        "total_segments": len(segment_indices),
        "segments_with_answer": segments_with_answer,
        "segments_without_answer": segments_without_answer,
        "expected_segment": expected_segment_index,
    }


def create_long_context_example(
    answer_text: str,
    answer_position: int,
    context_length_chars: int = 2000,
) -> dict:
    """Create a synthetic example with a long context.

    Args:
        answer_text: The answer text to embed in context
        answer_position: Character position where answer should start
        context_length_chars: Total length of context in characters

    Returns:
        SQuAD-like example dictionary
    """
    # Create filler text before and after answer
    filler = "The quick brown fox jumps over the lazy dog. " * 100

    # Ensure we have enough filler
    while len(filler) < context_length_chars:
        filler += filler

    # Insert answer at specified position with proper word boundaries
    # Find the nearest word boundary before answer_position
    before = filler[:answer_position]
    # Ensure we end at a word boundary
    if before and not before[-1].isspace():
        # Find last space
        last_space = before.rfind(" ")
        if last_space > 0:
            before = before[: last_space + 1]
            answer_position = last_space + 1

    # Add answer with spaces around it to ensure proper tokenization
    answer_with_spaces = f" {answer_text} "

    # Fill the rest
    remaining_length = context_length_chars - len(before) - len(answer_with_spaces)
    after = filler[:remaining_length]

    context = before + answer_with_spaces + after

    return {
        "id": f"long_context_{answer_position}",
        "question": "What is the answer?",
        "context": context,
        "answers": {
            "text": [answer_text],
            "answer_start": [answer_position + 1],  # +1 for leading space
        },
    }


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def tokenizer():
    """Create base XLNet tokenizer for testing."""
    return XLNetTokenizerFast.from_pretrained("xlnet-base-cased")


@pytest.fixture
def tokenizer_with_memory():
    """Create XLNet tokenizer with memory tokens."""
    tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
    configure_memory_tokens(tokenizer, memory_num_tokens=8)
    return tokenizer


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Test Classes
# ============================================================================


class TestAnswerInFirstSegment:
    """Test cases where answer appears in the first segment."""

    def test_answer_in_first_segment_short_context(self, tokenizer):
        """Test answer in first segment of a 2-segment document."""
        example = {
            "id": "test_001",
            "question": "What is mentioned first?",
            "context": (
                "Python is a programming language. " * 10  # Answer here
                + "JavaScript is also a programming language. " * 15  # Extra text to create 2nd segment
            ),
            "answers": {
                "text": ["Python"],
                "answer_start": [0],
            },
        }

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([example])

            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,  # Small to force multiple segments
                doc_stride=64,
                max_examples=1,
                dataset_name="test_dataset",
            )

            # Validate
            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="Python",
                expected_segment_index=0,  # Should be in first segment
            )

            assert result["success"], f"Answer not found in expected segment: {result}"
            assert result["total_segments"] >= 2, f"Should have multiple segments, got {result['total_segments']}"
            assert len(result["segments_with_answer"]) > 0, "No segments contain the answer"

    def test_answer_in_first_segment_long_context(self, tokenizer):
        """Test answer in first segment with very long context (5+ segments)."""
        # Create long context with answer at the beginning
        long_example = create_long_context_example(
            answer_text="ANSWER_TOKEN",
            answer_position=100,  # Near the beginning
            context_length_chars=3000,  # Long enough for 5+ segments
        )

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([long_example])

            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,
                max_examples=1,
                dataset_name="test_dataset",
            )

            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="ANSWER_TOKEN",
                expected_segment_index=0,
            )

            print("\nTest: Answer in first segment (long context)")
            print(f"  Total segments: {result['total_segments']}")
            print(f"  Segments with answer: {len(result['segments_with_answer'])}")

            assert result["success"], f"Answer not found in first segment: {result}"
            assert result["total_segments"] >= 5, f"Expected 5+ segments, got {result['total_segments']}"


class TestAnswerInMiddleSegment:
    """Test cases where answer appears in middle segments."""

    def test_answer_in_second_segment(self, tokenizer):
        """Test answer appearing in the second segment."""
        # Create context where answer is positioned to fall in 2nd segment
        long_example = create_long_context_example(
            answer_text="MIDDLE_ANSWER",
            answer_position=600,  # Position for 2nd segment
            context_length_chars=2000,
        )

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([long_example])

            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,
                max_examples=1,
                dataset_name="test_dataset",
            )

            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="MIDDLE_ANSWER",
            )

            print("\nTest: Answer in second segment")
            print(f"  Total segments: {result['total_segments']}")
            print(f"  Segments with answer: {result['segments_with_answer']}")

            assert result["success"], f"Answer not found in any segment: {result}"
            assert result["total_segments"] >= 3, f"Expected 3+ segments, got {result['total_segments']}"

            # Verify answer is NOT in first segment
            if result["segments_with_answer"]:
                answer_seg_idx = result["segments_with_answer"][0]["segment_index"]
                assert answer_seg_idx > 0, f"Answer should not be in first segment, found in segment {answer_seg_idx}"

    def test_answer_in_third_segment(self, tokenizer):
        """Test answer appearing in the third segment."""
        long_example = create_long_context_example(
            answer_text="THIRD_SEGMENT_ANSWER",
            answer_position=1200,  # Position for 3rd segment
            context_length_chars=2500,
        )

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([long_example])

            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,
                max_examples=1,
                dataset_name="test_dataset",
            )

            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="THIRD_SEGMENT_ANSWER",
            )

            print("\nTest: Answer in third segment")
            print(f"  Total segments: {result['total_segments']}")
            print(f"  Segments with answer: {result['segments_with_answer']}")

            assert result["success"], f"Answer not found: {result}"
            assert result["total_segments"] >= 4, f"Expected 4+ segments, got {result['total_segments']}"


class TestAnswerInLastSegment:
    """Test cases where answer appears in the last segment."""

    def test_answer_in_last_segment(self, tokenizer):
        """Test answer appearing in the final segment."""
        # Create context with answer near the end
        long_example = create_long_context_example(
            answer_text="FINAL_ANSWER",
            answer_position=2400,  # Near the end
            context_length_chars=2600,
        )

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([long_example])

            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,
                max_examples=1,
                dataset_name="test_dataset",
            )

            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="FINAL_ANSWER",
            )

            print("\nTest: Answer in last segment")
            print(f"  Total segments: {result['total_segments']}")
            print(f"  Segments with answer: {result['segments_with_answer']}")

            assert result["success"], f"Answer not found in last segment: {result}"
            assert result["total_segments"] >= 4, f"Expected 4+ segments, got {result['total_segments']}"

            # Verify answer is in one of the last segments
            # Due to document overlap (doc_stride), the answer might appear in the last
            # or second-to-last segment, both are valid
            if result["segments_with_answer"]:
                answer_segment_indices = [s["segment_index"] for s in result["segments_with_answer"]]
                last_segment_idx = result["total_segments"] - 1
                second_to_last_idx = result["total_segments"] - 2

                # Answer should be in last or second-to-last segment
                assert last_segment_idx in answer_segment_indices or second_to_last_idx in answer_segment_indices, (
                    f"Answer should be in last segments ({second_to_last_idx} or {last_segment_idx}), "
                    f"found in segments {answer_segment_indices}"
                )


class TestMultipleOccurrences:
    """Test cases with multiple answer occurrences across segments."""

    def test_answer_appears_in_multiple_segments(self, tokenizer):
        """Test when the same answer text appears in multiple segments."""
        # Create context with answer appearing multiple times
        context = (
            "The word Python appears here. " * 5  # First occurrence
            + "JavaScript and other languages are used. " * 10  # Filler
            + "Python is mentioned again here. " * 5  # Second occurrence
        )

        example = {
            "id": "multi_occurrence",
            "question": "What programming language?",
            "context": context,
            "answers": {
                "text": ["Python"],
                "answer_start": [9],  # First occurrence
            },
        }

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([example])

            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,
                max_examples=1,
                dataset_name="test_dataset",
            )

            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="Python",
            )

            print("\nTest: Multiple occurrences")
            print(f"  Total segments: {result['total_segments']}")
            print(f"  Segments with answer: {len(result['segments_with_answer'])}")

            # At least one segment should have the answer
            assert result["success"], f"Answer not found: {result}"

            # With overlapping segments (doc_stride), we might see answer in multiple segments
            # This is expected behavior


class TestVeryLongDocuments:
    """Test cases with very long documents (10+ segments)."""

    def test_very_long_document_answer_in_middle(self, tokenizer):
        """Test very long document with answer in middle segment."""
        # Create very long context
        long_example = create_long_context_example(
            answer_text="NEEDLE_IN_HAYSTACK",
            answer_position=2500,  # Middle of very long context
            context_length_chars=5000,  # Very long
        )

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([long_example])

            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,
                max_examples=1,
                dataset_name="test_dataset",
            )

            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="NEEDLE_IN_HAYSTACK",
            )

            print("\nTest: Very long document (needle in haystack)")
            print(f"  Total segments: {result['total_segments']}")
            print(f"  Segments with answer: {result['segments_with_answer']}")

            assert result["success"], f"Answer not found in very long document: {result}"
            assert result["total_segments"] >= 8, f"Expected 8+ segments, got {result['total_segments']}"

    def test_max_segments_cap(self, tokenizer):
        """Test that max_n_segs parameter correctly limits segment count."""
        long_example = create_long_context_example(
            answer_text="CAPPED_ANSWER",
            answer_position=500,  # Should be in segment that's kept
            context_length_chars=5000,
        )

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([long_example])

            # Process with max_n_segs cap
            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,
                max_examples=1,
                dataset_name="test_dataset",
                max_n_segs=5,  # Cap at 5 segments
            )

            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="CAPPED_ANSWER",
            )

            print("\nTest: Max segments cap")
            print(f"  Total segments: {result['total_segments']}")
            print("  Max allowed: 5")

            # Should have exactly 5 segments (or fewer if context is shorter)
            assert result["total_segments"] <= 5, f"Exceeded max_n_segs cap: {result['total_segments']}"

            # Answer should still be found (it's early in the document)
            assert result["success"], f"Answer not found with segment cap: {result}"


class TestSegmentOverlap:
    """Test cases verifying segment overlap behavior."""

    def test_answer_in_overlapping_region(self, tokenizer):
        """Test answer that appears in the overlap region between segments."""
        # Create context where answer is likely in overlap
        # This tests that doc_stride creates proper overlap
        example = {
            "id": "overlap_test",
            "question": "What is in the overlap?",
            "context": "A" * 400 + " OVERLAP_ANSWER " + "B" * 400,
            "answers": {
                "text": ["OVERLAP_ANSWER"],
                "answer_start": [400],  # Right at boundary
            },
        }

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([example])

            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,  # 50% overlap
                max_examples=1,
                dataset_name="test_dataset",
            )

            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="OVERLAP_ANSWER",
            )

            print("\nTest: Answer in overlap region")
            print(f"  Total segments: {result['total_segments']}")
            print(f"  Segments with answer: {len(result['segments_with_answer'])}")

            # Answer should be found in at least one segment
            assert result["success"], f"Answer not found in overlapping segments: {result}"

            # Due to overlap, answer might appear in 1 or 2 consecutive segments
            # This is expected and correct behavior


class TestStreamingMultiSegment:
    """Test multi-segment scenarios with streaming processing."""

    def test_streaming_long_document(self, tokenizer, temp_cache_dir):
        """Test streaming processing with long multi-segment document."""
        long_example = create_long_context_example(
            answer_text="STREAMING_ANSWER",
            answer_position=1500,
            context_length_chars=3000,
        )

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        with patch("memxlnet.data.streaming.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([long_example])

            processor = StreamingSquadProcessor(
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,
                streaming_chunk_size=10,
                max_memory_gb=4.0,
            )

            cache_manager = ChunkedCacheManager(temp_cache_dir, chunk_size=10)

            total_features = processor.process_dataset_streaming(
                dataset_name="test_streaming_long",
                split="test",
                max_examples=1,
                cache_manager=cache_manager,
                max_n_segs=None,
            )

            print("\nTest: Streaming long document")
            print(f"  Total features processed: {total_features}")

            assert total_features >= 5, f"Expected 5+ segments, got {total_features}"

            # Load and validate features
            cache_key = "test_streaming_long_v1"
            features = []
            total_chunks = cache_manager.get_total_chunks(cache_key, "test")
            for chunk_id in range(total_chunks):
                chunk_data = cache_manager.load_chunk(cache_key, "test", chunk_id)
                features.extend(chunk_data)

            # Find which segment has the answer
            answer_found = False
            for feat in features:
                extracted = extract_answer_from_feature(feat)
                extracted_normalized = extracted.strip().lower()
                expected_normalized = "streaming_answer"
                if extracted_normalized == expected_normalized or expected_normalized in extracted_normalized:
                    answer_found = True
                    break

            assert answer_found, "Answer not found in streaming processed segments"


class TestLazyLoadingMultiSegment:
    """Test multi-segment scenarios with lazy loading."""

    def test_lazy_loading_long_document(self, tokenizer, temp_cache_dir):
        """Test lazy loading with long multi-segment document."""
        long_example = create_long_context_example(
            answer_text="LAZY_ANSWER",
            answer_position=1800,
            context_length_chars=3500,
        )

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        # First process with streaming
        with patch("memxlnet.data.streaming.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([long_example])

            processor = StreamingSquadProcessor(
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,
                streaming_chunk_size=10,
                max_memory_gb=4.0,
            )

            cache_manager = ChunkedCacheManager(temp_cache_dir, chunk_size=10)

            processor.process_dataset_streaming(
                dataset_name="test_lazy_long",
                split="test",
                max_examples=1,
                cache_manager=cache_manager,
                max_n_segs=None,
            )

        # Create lazy dataset
        cache_key = "test_lazy_long_v1"
        lazy_dataset = LazySquadLikeQADataset(
            cache_manager=cache_manager,
            dataset_name=cache_key,
            split="test",
            cache_params={"max_seq_length": 128, "doc_stride": 64},
        )

        # Print available document IDs for debugging
        available_doc_ids = list(lazy_dataset.document_index.keys()) if hasattr(lazy_dataset, "document_index") else []
        print("\nTest: Lazy loading long document")
        print(f"  Available document IDs: {available_doc_ids}")
        print(f"  Total features in lazy dataset: {len(lazy_dataset)}")

        # Try to find the answer by manually checking all features
        answer_found = False
        answer_segments = []

        for i in range(len(lazy_dataset)):
            feat = lazy_dataset[i]
            extracted = extract_answer_from_feature(feat)
            extracted_normalized = extracted.strip().lower()
            expected_normalized = "lazy_answer"
            if extracted_normalized == expected_normalized or expected_normalized in extracted_normalized:
                answer_found = True
                answer_segments.append(i)

        print(f"  Answer found in {len(answer_segments)} segments: {answer_segments}")

        # Basic validation without using validate_multi_segment_answer
        # since lazy dataset might have different document ID structure
        assert answer_found, "Answer not found with lazy loading"
        assert len(lazy_dataset) >= 5, f"Expected 5+ segments, got {len(lazy_dataset)}"


class TestMemoryTokensMultiSegment:
    """Test multi-segment scenarios with memory tokens."""

    def test_memory_tokens_long_document(self, tokenizer_with_memory):
        """Test that memory tokens don't interfere with multi-segment answers."""
        long_example = create_long_context_example(
            answer_text="MEMORY_ANSWER",
            answer_position=1200,
            context_length_chars=2800,
        )

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([long_example])

            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer_with_memory,
                max_seq_length=128,
                doc_stride=64,
                max_examples=1,
                dataset_name="test_dataset",
            )

            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="MEMORY_ANSWER",
            )

            print("\nTest: Memory tokens with long document")
            print(f"  Total segments: {result['total_segments']}")
            print(f"  Segments with answer: {result['segments_with_answer']}")

            assert result["success"], f"Memory tokens interfered with answer mapping: {result}"
            assert result["total_segments"] >= 4, f"Expected 4+ segments, got {result['total_segments']}"


class TestEdgeCases:
    """Test edge cases for multi-segment processing."""

    def test_answer_at_exact_segment_boundary(self, tokenizer):
        """Test answer that falls exactly at segment boundary."""
        # This is tricky - answers at boundaries might be split or missed
        # Our implementation should handle this gracefully
        example = {
            "id": "boundary_test",
            "question": "What is at the boundary?",
            "context": "X" * 350 + " BOUNDARY " + "Y" * 350,
            "answers": {
                "text": ["BOUNDARY"],
                "answer_start": [351],
            },
        }

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([example])

            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,
                max_examples=1,
                dataset_name="test_dataset",
            )

            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="BOUNDARY",
            )

            print("\nTest: Answer at segment boundary")
            print(f"  Total segments: {result['total_segments']}")
            print(f"  Segments with answer: {result['segments_with_answer']}")

            # Answer should be found in at least one segment (either side of boundary)
            # or in both due to overlap
            assert result["success"], f"Answer at boundary not handled correctly: {result}"

    def test_empty_segments_after_max_n_segs(self, tokenizer):
        """Test that segments after max_n_segs are not processed."""
        long_example = create_long_context_example(
            answer_text="SHOULD_NOT_BE_FOUND",
            answer_position=2800,  # Very late in document
            context_length_chars=3000,
        )

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset([long_example])

            # Cap at 3 segments - answer should not be found
            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=64,
                max_examples=1,
                dataset_name="test_dataset",
                max_n_segs=3,  # Cap early
            )

            result = validate_multi_segment_answer(
                dataset,
                example_id="doc_0",
                expected_answer="SHOULD_NOT_BE_FOUND",
            )

            print("\nTest: Answer beyond max_n_segs cap")
            print(f"  Total segments: {result['total_segments']}")
            print(f"  Segments with answer: {result['segments_with_answer']}")

            # Should have exactly 3 segments
            assert result["total_segments"] == 3, f"Should have 3 segments, got {result['total_segments']}"

            # Answer should NOT be found (it's beyond the cap)
            # This is expected behavior - documents are truncated
            assert not result["success"], "Answer should not be found beyond max_n_segs cap"
