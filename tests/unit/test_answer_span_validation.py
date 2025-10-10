"""Test suite for answer span validation in dataset processing.

This module validates that start/end positions in processed features correctly
correspond to the actual answers in raw data, for both traditional and streaming
processing methods.
"""

import tempfile
from typing import Any

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
from memxlnet.data.text_utils import (
    unicode_answers_match,
)

# ============================================================================
# Helper Functions
# ============================================================================


def extract_answer_from_positions(
    feature: dict[str, Any],
    use_token_positions: bool = True,
) -> str:
    """Extract answer text from feature using start/end positions.

    Args:
        feature: Processed feature dictionary
        use_token_positions: If True, use start/end token positions.
                            If False, use character span directly.

    Returns:
        Extracted answer text
    """
    if use_token_positions:
        # Extract using token positions and offset mapping
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

        # Get offset mapping
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
    else:
        # Extract using character span directly
        char_span = feature.get("chosen_answer_char_span", [-1, -1])
        if char_span[0] == -1 or char_span[1] == -1:
            return ""

        context = feature["context"]
        return context[char_span[0] : char_span[1]]


def validate_single_feature(
    feature: dict[str, Any],
    expected_answer: str | None = None,
) -> dict[str, Any]:
    """Validate a single processed feature.

    Args:
        feature: Processed feature dictionary
        expected_answer: Expected answer text (optional, uses chosen_answer_text if None)

    Returns:
        Dictionary with validation results:
            - valid: Whether validation passed
            - extracted_answer: Answer extracted from positions
            - expected_answer: Expected answer text
            - match_exact: Exact string match
            - match_normalized: Match after normalization
            - error: Error message if validation failed
    """
    result = {
        "valid": False,
        "extracted_answer": "",
        "expected_answer": expected_answer or "",
        "match_exact": False,
        "match_normalized": False,
        "error": None,
    }

    try:
        # Get expected answer
        if expected_answer is None:
            expected_answer = feature.get("chosen_answer_text", "")
        result["expected_answer"] = expected_answer

        # Extract answer from token positions
        extracted = extract_answer_from_positions(feature, use_token_positions=True)
        result["extracted_answer"] = extracted

        # Check for no-answer case
        has_answer = feature.get("has_answer", False)
        if not has_answer:
            # For unanswerable questions, extracted should be empty
            result["valid"] = extracted == ""
            result["match_exact"] = extracted == ""
            result["match_normalized"] = extracted == ""
            if not result["valid"]:
                result["error"] = "Unanswerable question should have empty extraction"
            return result

        # For answerable questions, validate extraction
        if not expected_answer:
            result["error"] = "Has answer but no expected_answer provided"
            return result

        # Check exact match
        result["match_exact"] = extracted == expected_answer

        # Check normalized match (more flexible)
        result["match_normalized"] = unicode_answers_match(expected_answer, extracted)

        # Consider valid if normalized match succeeds
        result["valid"] = result["match_normalized"]

        if not result["valid"]:
            result["error"] = f"Answer mismatch: expected='{expected_answer}', extracted='{extracted}'"

    except Exception as e:
        result["error"] = f"Validation exception: {str(e)}"
        result["valid"] = False

    return result


def validate_dataset_features(
    features: list[dict[str, Any]],
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Validate all features in a processed dataset.

    Args:
        features: List of processed feature dictionaries
        max_examples: Maximum features to validate (None for all)

    Returns:
        Dictionary with validation statistics and failure cases
    """
    total = 0
    valid = 0
    exact_matches = 0
    normalized_matches = 0
    failures = []

    features_to_check = features[:max_examples] if max_examples else features

    for idx, feature in enumerate(features_to_check):
        total += 1
        result = validate_single_feature(feature)

        if result["valid"]:
            valid += 1
        else:
            failures.append(
                {
                    "feature_index": idx,
                    "example_id": feature.get("example_id", "unknown"),
                    "segment_index": feature.get("segment_index", 0),
                    "result": result,
                    "context_preview": feature.get("context", "")[:100],
                }
            )

        if result["match_exact"]:
            exact_matches += 1
        if result["match_normalized"]:
            normalized_matches += 1

    return {
        "total": total,
        "valid": valid,
        "exact_matches": exact_matches,
        "normalized_matches": normalized_matches,
        "success_rate": valid / total if total > 0 else 0.0,
        "exact_match_rate": exact_matches / total if total > 0 else 0.0,
        "normalized_match_rate": normalized_matches / total if total > 0 else 0.0,
        "failures": failures,
    }


def generate_validation_report(stats: dict[str, Any], title: str = "Validation Report") -> str:
    """Generate a human-readable validation report.

    Args:
        stats: Validation statistics from validate_dataset_features()
        title: Report title

    Returns:
        Formatted report string
    """
    lines = [
        f"\n{'=' * 70}",
        f"{title}",
        f"{'=' * 70}",
        f"Total features tested: {stats['total']}",
        f"Valid features: {stats['valid']} ({stats['success_rate']:.1%})",
        f"Exact matches: {stats['exact_matches']} ({stats['exact_match_rate']:.1%})",
        f"Normalized matches: {stats['normalized_matches']} ({stats['normalized_match_rate']:.1%})",
        "",
    ]

    if stats["failures"]:
        lines.append(f"Failed validations: {len(stats['failures'])}")
        lines.append(f"{'-' * 70}")

        # Show first 5 failures
        for i, failure in enumerate(stats["failures"][:5], 1):
            lines.append(f"\nFailure {i}:")
            lines.append(f"  Example ID: {failure['example_id']}")
            lines.append(f"  Segment: {failure['segment_index']}")
            lines.append(f"  Expected: '{failure['result']['expected_answer']}'")
            lines.append(f"  Extracted: '{failure['result']['extracted_answer']}'")
            lines.append(f"  Error: {failure['result']['error']}")
            lines.append(f"  Context: {failure['context_preview']}...")

        if len(stats["failures"]) > 5:
            lines.append(f"\n... and {len(stats['failures']) - 5} more failures")

    lines.append(f"{'=' * 70}\n")

    return "\n".join(lines)


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
    configure_memory_tokens(tokenizer, memory_num_tokens=4)
    return tokenizer


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_squad_examples():
    """Provide curated SQuAD-like examples for testing."""
    return [
        {
            "id": "test_001",
            "question": "What is the capital of France?",
            "context": "Paris is the capital and most populous city of France.",
            "answers": {
                "text": ["Paris"],
                "answer_start": [0],
            },
        },
        {
            "id": "test_002",
            "question": "When did World War II end?",
            "context": "World War II ended in 1945 with the surrender of Germany and Japan.",
            "answers": {
                "text": ["1945"],
                "answer_start": [23],
            },
        },
        {
            "id": "test_003",
            "question": "Who invented the telephone?",
            "context": "Alexander Graham Bell is credited with inventing the telephone in 1876.",
            "answers": {
                "text": ["Alexander Graham Bell"],
                "answer_start": [0],
            },
        },
        # Unicode test case
        {
            "id": "test_004",
            "question": "What is the best café?",
            "context": "The best café in Paris is the Café de Flore.",
            "answers": {
                "text": ["Café de Flore"],
                "answer_start": [31],
            },
        },
        # Unanswerable (SQuAD v2 style)
        {
            "id": "test_005",
            "question": "What is the population of Mars?",
            "context": "Mars is the fourth planet from the Sun and the second-smallest planet.",
            "answers": {
                "text": [],
                "answer_start": [],
            },
        },
    ]


# ============================================================================
# Test Classes
# ============================================================================


class TestAnswerSpanExtraction:
    """Test basic answer extraction logic."""

    def test_extract_from_valid_positions(self, sample_squad_examples):
        """Test extraction from valid token positions."""
        # Create a simple feature manually
        feature = {
            "context": "Paris is the capital of France.",
            "start_positions": 0,
            "end_positions": 0,
            "offset_mapping": [(0, 5), (6, 8), (9, 12), (13, 20), (21, 23), (24, 30), (30, 31)],
            "cls_index": 10,  # Different from answer position
        }

        extracted = extract_answer_from_positions(feature)
        assert extracted == "Paris"

    def test_extract_no_answer(self):
        """Test extraction for unanswerable questions."""
        feature = {
            "context": "Some context text.",
            "start_positions": 5,
            "end_positions": 5,
            "offset_mapping": [(0, 4), (5, 12), (13, 17), (17, 18)],
            "cls_index": 5,  # Same as positions (CLS token)
        }

        extracted = extract_answer_from_positions(feature)
        assert extracted == ""

    def test_validate_single_feature_success(self):
        """Test validation of a correct feature."""
        feature = {
            "context": "Paris is the capital.",
            "start_positions": 0,
            "end_positions": 0,
            "offset_mapping": [(0, 5), (6, 8), (9, 12), (13, 20), (20, 21)],
            "cls_index": 10,
            "has_answer": True,
            "chosen_answer_text": "Paris",
        }

        result = validate_single_feature(feature)
        assert result["valid"]
        assert result["match_normalized"]

    def test_validate_single_feature_failure(self):
        """Test validation of an incorrect feature."""
        feature = {
            "context": "Paris is the capital.",
            "start_positions": 2,  # Wrong position (points to "the")
            "end_positions": 2,
            "offset_mapping": [(0, 5), (6, 8), (9, 12), (13, 20), (20, 21)],
            "cls_index": 10,
            "has_answer": True,
            "chosen_answer_text": "Paris",
        }

        result = validate_single_feature(feature)
        # The extracted text will be "the" which doesn't match "Paris"
        assert not result["valid"] or result["extracted_answer"] != "Paris"
        assert result["extracted_answer"] != "Paris"


class TestTraditionalProcessing:
    """Test answer span validation for traditional SquadLikeQADataset processing."""

    def test_process_small_dataset(self, tokenizer, sample_squad_examples):
        """Test processing a small curated dataset."""
        # Create temporary dataset in memory (mock load_dataset)
        from unittest.mock import patch

        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            # Mock the dataset
            class MockDataset:
                def __init__(self, data):
                    self.data = data

                def select(self, indices):
                    selected_data = [self.data[i] for i in range(len(indices))]
                    return MockDataset(selected_data)

                def __iter__(self):
                    return iter(self.data)

                def __len__(self):
                    return len(self.data)

            mock_load.return_value = MockDataset(sample_squad_examples)

            # Process with SquadLikeQADataset
            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=32,
                max_examples=len(sample_squad_examples),
                dataset_name="test_dataset",
            )

            # Validate all features
            features = [dataset[i] for i in range(len(dataset))]
            stats = validate_dataset_features(features)

            # Print report
            report = generate_validation_report(stats, "Traditional Processing Validation")
            print(report)

            # Assert high success rate
            assert stats["success_rate"] >= 0.8, f"Success rate too low: {stats['success_rate']:.1%}"
            assert stats["total"] > 0, "No features were processed"

    def test_unicode_handling(self, tokenizer):
        """Test processing with Unicode characters."""
        from unittest.mock import patch

        unicode_examples = [
            {
                "id": "unicode_001",
                "question": "What is the best café?",
                "context": "The Café de Flore is a famous café in Paris.",
                "answers": {
                    "text": ["Café de Flore"],
                    "answer_start": [4],
                },
            },
            {
                "id": "unicode_002",
                "question": "Where is Zürich?",
                "context": "Zürich is the largest city in Switzerland.",
                "answers": {
                    "text": ["Zürich"],
                    "answer_start": [0],
                },
            },
        ]

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
            mock_load.return_value = MockDataset(unicode_examples)

            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=32,
                max_examples=len(unicode_examples),
                dataset_name="test_dataset",
            )

            # Validate features
            features = [dataset[i] for i in range(len(dataset))]
            stats = validate_dataset_features(features)

            # Unicode handling should work correctly
            assert stats["success_rate"] >= 0.8, f"Unicode handling failed: {stats['success_rate']:.1%}"


class TestStreamingProcessing:
    """Test answer span validation for StreamingSquadProcessor."""

    def test_streaming_process_small_dataset(self, tokenizer, temp_cache_dir, sample_squad_examples):
        """Test streaming processing with validation."""
        from unittest.mock import patch

        class MockStreamingDataset:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        with patch("memxlnet.data.streaming.load_dataset") as mock_load:
            mock_load.return_value = MockStreamingDataset(sample_squad_examples)

            # Create streaming processor
            processor = StreamingSquadProcessor(
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=32,
                streaming_chunk_size=10,
                max_memory_gb=4.0,
            )

            # Create cache manager
            cache_manager = ChunkedCacheManager(temp_cache_dir, chunk_size=10)

            # Process with streaming
            total_features = processor.process_dataset_streaming(
                dataset_name="test_dataset",
                split="test",
                max_examples=len(sample_squad_examples),
                cache_manager=cache_manager,
                max_n_segs=None,
            )

            assert total_features > 0, "No features were processed"

            # Load processed features from cache
            # Streaming adds "_v1" suffix to dataset name
            cache_key = "test_dataset_v1"
            features = []
            total_chunks = cache_manager.get_total_chunks(cache_key, "test")
            for chunk_id in range(total_chunks):
                chunk_data = cache_manager.load_chunk(cache_key, "test", chunk_id)
                # Convert raw dict features to proper format (add tensors if needed)
                for feat in chunk_data:
                    if not isinstance(feat.get("start_positions"), torch.Tensor):
                        # These are raw dict features from streaming, need minimal conversion
                        pass  # They should still work with our validation
                    features.append(feat)

            # Validate features (should have at least some features)
            assert len(features) > 0, "No features were loaded from cache"
            stats = validate_dataset_features(features)

            # Print report
            report = generate_validation_report(stats, "Streaming Processing Validation")
            print(report)

            # Assert high success rate
            assert stats["success_rate"] >= 0.8, f"Success rate too low: {stats['success_rate']:.1%}"


class TestProcessingConsistency:
    """Test consistency between traditional and streaming processing."""

    def test_traditional_vs_streaming_consistency(self, tokenizer, temp_cache_dir, sample_squad_examples):
        """Test that traditional and streaming produce equivalent results."""
        from unittest.mock import patch

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def select(self, indices):
                return MockDataset([self.data[i] for i in range(len(indices))])

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        # Process with traditional method
        with patch("memxlnet.data.dataset.load_dataset") as mock_load:
            mock_load.return_value = MockDataset(sample_squad_examples)

            traditional_dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=32,
                max_examples=len(sample_squad_examples),
                dataset_name="test_dataset",
            )

            traditional_features = [traditional_dataset[i] for i in range(len(traditional_dataset))]
            traditional_stats = validate_dataset_features(traditional_features)

        # Process with streaming method
        with patch("memxlnet.data.streaming.load_dataset") as mock_load:
            mock_load.return_value = MockDataset(sample_squad_examples)

            processor = StreamingSquadProcessor(
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=32,
                streaming_chunk_size=10,
                max_memory_gb=4.0,
            )

            cache_manager = ChunkedCacheManager(temp_cache_dir, chunk_size=10)

            processor.process_dataset_streaming(
                dataset_name="test_dataset_streaming",
                split="test",
                max_examples=len(sample_squad_examples),
                cache_manager=cache_manager,
                max_n_segs=None,
            )

            # Load streaming features
            # Streaming adds "_v1" suffix to dataset name
            cache_key = "test_dataset_streaming_v1"
            streaming_features = []
            total_chunks = cache_manager.get_total_chunks(cache_key, "test")
            for chunk_id in range(total_chunks):
                chunk_data = cache_manager.load_chunk(cache_key, "test", chunk_id)
                streaming_features.extend(chunk_data)

            streaming_stats = validate_dataset_features(streaming_features)

        # Compare statistics
        print("\nTraditional Processing:")
        print(generate_validation_report(traditional_stats, "Traditional Processing"))

        print("\nStreaming Processing:")
        print(generate_validation_report(streaming_stats, "Streaming Processing"))

        # Both should have similar success rates
        rate_diff = abs(traditional_stats["success_rate"] - streaming_stats["success_rate"])
        assert rate_diff < 0.05, f"Success rates differ by {rate_diff:.1%}"

        # Both should process same number of features
        assert len(traditional_features) == len(streaming_features), (
            f"Feature counts differ: {len(traditional_features)} vs {len(streaming_features)}"
        )


class TestLazyLoadingValidation:
    """Test answer span validation with lazy loading."""

    def test_lazy_dataset_validation(self, tokenizer, temp_cache_dir, sample_squad_examples):
        """Test that lazy-loaded features maintain correct answer spans."""
        from unittest.mock import patch

        class MockDataset:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        # First, process and cache data with streaming
        with patch("memxlnet.data.streaming.load_dataset") as mock_load:
            mock_load.return_value = MockDataset(sample_squad_examples)

            processor = StreamingSquadProcessor(
                tokenizer=tokenizer,
                max_seq_length=128,
                doc_stride=32,
                streaming_chunk_size=10,
                max_memory_gb=4.0,
            )

            cache_manager = ChunkedCacheManager(temp_cache_dir, chunk_size=10)

            processor.process_dataset_streaming(
                dataset_name="test_dataset_lazy",
                split="test",
                max_examples=len(sample_squad_examples),
                cache_manager=cache_manager,
                max_n_segs=None,
            )

        # Create lazy dataset
        # Streaming adds "_v1" suffix to dataset name
        cache_key = "test_dataset_lazy_v1"
        lazy_dataset = LazySquadLikeQADataset(
            cache_manager=cache_manager,
            dataset_name=cache_key,
            split="test",
            cache_params={"max_seq_length": 128, "doc_stride": 32},
        )

        # Validate features loaded lazily
        features = [lazy_dataset[i] for i in range(len(lazy_dataset))]
        stats = validate_dataset_features(features)

        # Print report
        report = generate_validation_report(stats, "Lazy Loading Validation")
        print(report)

        # Lazy loading should maintain correctness
        assert stats["success_rate"] >= 0.8, f"Lazy loading validation failed: {stats['success_rate']:.1%}"


class TestMemoryTokenCompatibility:
    """Test answer span validation with memory tokens enabled."""

    def test_with_memory_tokens(self, tokenizer_with_memory, sample_squad_examples):
        """Test that memory tokens don't interfere with answer span mapping."""
        from unittest.mock import patch

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
            mock_load.return_value = MockDataset(sample_squad_examples)

            # Process with memory-enabled tokenizer
            dataset = SquadLikeQADataset(
                split="test",
                tokenizer=tokenizer_with_memory,
                max_seq_length=128,
                doc_stride=32,
                max_examples=len(sample_squad_examples),
                dataset_name="test_dataset",
            )

            # Validate features
            features = [dataset[i] for i in range(len(dataset))]
            stats = validate_dataset_features(features)

            # Memory tokens should not affect answer span correctness
            assert stats["success_rate"] >= 0.8, f"Memory token compatibility failed: {stats['success_rate']:.1%}"
