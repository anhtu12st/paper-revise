# Testing and Validation Guide

This guide covers comprehensive testing and validation procedures for the MemXLNet QA system, with special focus on Unicode handling and position mapping accuracy.

## Table of Contents

1. [Overview](#overview)
2. [Running the Test Suite](#running-the-test-suite)
3. [Unicode Testing](#unicode-testing)
4. [Position Mapping Validation](#position-mapping-validation)
5. [Data Processing Tests](#data-processing-tests)
6. [Answer Validation Tests](#answer-validation-tests)
7. [Performance Testing](#performance-testing)
8. [Troubleshooting](#troubleshooting)
9. [Custom Test Creation](#custom-test-creation)

## Overview

The testing framework includes comprehensive validation for:

- **Unicode normalization** across multiple languages
- **Character-to-token position mapping** accuracy
- **Answer span validation** with edge cases
- **Data processing pipeline** integrity
- **Memory token integration** functionality
- **Cross-tokenizer compatibility**

## Running the Test Suite

### Quick Start

```bash
# Navigate to project directory
cd /path/to/paper-revise

# Run the comprehensive test notebook
jupyter notebook notebooks/data_processing.ipynb

# Or run specific test cells programmatically
python -c "
import sys, os
sys.path.insert(0, 'src')
from memxlnet.data.text_utils import run_unicode_tests
results = run_unicode_tests()
print(f'Passed: {sum(1 for r in results if r[\"normalized_match\"])}/{len(results)}')
"
```

### Full Test Suite in Notebook

The main test suite is located in `notebooks/data_processing.ipynb` with the following phases:

1. **Phase 1**: Basic cache testing
2. **Phase 2**: Chunked loading implementation
3. **Phase 3**: Answer position validation
4. **Phase 4**: Document segmentation testing
5. **Phase 5**: DataLoader testing
6. **Phase 6**: Memory token integration
7. **Phase 7**: Edge cases and stress testing
8. **Unicode Tests**: Comprehensive Unicode and position mapping tests

### Running Individual Test Phases

```python
# Import test utilities
import sys, os
from memxlnet.data.text_utils import run_unicode_tests, validate_answer_positions

# Run Unicode test suite
unicode_results = run_unicode_tests()
print(f"Unicode tests: {len([r for r in unicode_results if r['normalized_match']])}/{len(unicode_results)} passed")

# Test position validation
context = "Le café est très bon."
answer = "café"
is_valid = validate_answer_positions(context, answer, 3, 7)
print(f"Position validation: {is_valid}")
```

## Unicode Testing

### Supported Languages and Scripts

The Unicode test suite covers:

- **French**: accented characters (é, à, ç, è, ù)
- **German**: umlauts and special characters (ü, ö, ä, ß)
- **Spanish**: accented characters and tildes (ñ, í, ó, ú)
- **Cyrillic**: Russian characters (Москва)
- **CJK**: Chinese characters (北京)
- **Symbols**: emoji (🤖), trademarks (™), mathematical symbols

### Running Unicode Tests

```python
from memxlnet.data.text_utils import run_unicode_tests, UNICODE_TEST_CASES

# Run predefined test cases
results = run_unicode_tests()

# Analyze results
for result in results:
    print(f"Test: {result['name']}")
    print(f"  Context: '{result['context']}'")
    print(f"  Answer: '{result['answer']}'")
    print(f"  Position valid: {result['position_valid']}")
    print(f"  Found correctly: {result['found_correctly']}")
    print(f"  Normalized match: {result['normalized_match']}")
    print()
```

### Custom Unicode Tests

```python
from memxlnet.data.text_utils import (
    normalize_unicode,
    find_answer_span_with_normalization,
    validate_answer_positions
)

# Define custom test cases
custom_tests = [
    {
        'name': 'Portuguese accents',
        'context': 'São Paulo é uma cidade grande.',
        'answer': 'São Paulo',
        'start': 0,
        'end': 9
    },
    {
        'name': 'Italian accents',
        'context': 'Più tardi andrò al università.',
        'answer': 'università',
        'start': 20,
        'end': 30
    }
]

# Run custom tests
for test in custom_tests:
    context = test['context']
    answer = test['answer']
    expected_start = test['start']
    expected_end = test['end']

    # Test position validation
    is_valid = validate_answer_positions(context, answer, expected_start, expected_end)

    # Test answer finding
    found_start, found_end = find_answer_span_with_normalization(context, answer)

    print(f"Test: {test['name']}")
    print(f"  Expected positions valid: {is_valid}")
    print(f"  Found at: ({found_start}, {found_end})")
    print(f"  Match: {found_start == expected_start and found_end == expected_end}")
    print()
```

## Position Mapping Validation

### Testing Answer Position Accuracy

```python
from memxlnet.data.text_utils import validate_answer_mapping
from transformers import XLNetTokenizerFast

# Create tokenizer
tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")

# Test position mapping with various examples
test_examples = [
    {
        'context': 'François Mitterrand était président de la France.',
        'question': 'Qui était président?',
        'answers': {
            'text': ['François Mitterrand'],
            'answer_start': [0]
        }
    },
    {
        'context': 'Le café de la rue Saint-Honoré est délicieux.',
        'question': 'Qu\'est-ce qui est délicieux?',
        'answers': {
            'text': ['Le café'],
            'answer_start': [0]
        }
    }
]

# Process and validate each example
for i, example in enumerate(test_examples):
    print(f"Example {i+1}:")

    # Create a mock feature (simplified)
    # In practice, this would come from SquadLikeQADataset
    feature = {
        'start_positions': 0,  # Would be computed by dataset
        'end_positions': len(example['answers']['text'][0]),
        'input_ids': tokenizer.encode(example['context'], add_special_tokens=True),
        'attention_mask': [1] * len(tokenizer.encode(example['context'], add_special_tokens=True))
    }

    # Validate answer mapping
    validation = validate_answer_mapping(feature, example, tokenizer)

    print(f"  Valid: {validation['valid']}")
    print(f"  Original: {validation['original_answer']}")
    print(f"  Reconstructed: {validation['reconstructed_answer']}")
    print(f"  Exact match: {validation.get('exact_match', False)}")
    print(f"  Normalized match: {validation.get('normalized_match', False)}")

    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    print()
```

### Edge Case Testing

```python
from memxlnet.data.text_utils import validate_answer_positions, fix_answer_positions

# Test edge cases
edge_cases = [
    {
        'name': 'Answer at beginning',
        'context': 'Answer is here in the text.',
        'answer': 'Answer',
        'start': 0,
        'end': 6
    },
    {
        'name': 'Answer at end',
        'context': 'The text ends with answer.',
        'answer': 'answer',
        'start': 20,
        'end': 26
    },
    {
        'name': 'Single character',
        'context': 'a b c d e f g',
        'answer': 'e',
        'start': 8,
        'end': 9
    },
    {
        'name': 'Unicode at boundary',
        'context': 'café thé',
        'answer': 'thé',
        'start': 5,
        'end': 8
    },
    {
        'name': 'Off-by-one error',
        'context': 'The quick brown fox',
        'answer': 'quick',
        'start': 3,  # Should be 4
        'end': 8
    }
]

for case in edge_cases:
    print(f"Testing: {case['name']}")

    # Test original positions
    is_valid = validate_answer_positions(
        case['context'], case['answer'], case['start'], case['end']
    )

    print(f"  Original positions valid: {is_valid}")

    if not is_valid:
        # Try to fix positions
        corrected_start, corrected_end = fix_answer_positions(
            case['context'], case['answer'], case['start']
        )

        is_corrected_valid = validate_answer_positions(
            case['context'], case['answer'], corrected_start, corrected_end
        )

        print(f"  Corrected to: ({corrected_start}, {corrected_end})")
        print(f"  Corrected positions valid: {is_corrected_valid}")

    print()
```

## Data Processing Tests

### Full Pipeline Testing

```python
from memxlnet.data import SquadLikeQADataset, create_dataset_from_cache
from transformers import XLNetTokenizerFast

# Test data processing pipeline
def test_data_processing_pipeline():
    tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")

    # Create dataset with Unicode examples
    dataset = SquadLikeQADataset(
        split='validation',
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_examples=100,  # Small subset for testing
        dataset_name='squad_v2',
        max_n_segs=6
    )

    print(f"Dataset created: {len(dataset)} features")

    # Test random samples
    import random
    sample_indices = random.sample(range(len(dataset)), min(10, len(dataset)))

    validation_issues = []
    for idx in sample_indices:
        feature = dataset[idx]

        # Check feature structure
        required_fields = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
        missing_fields = [field for field in required_fields if field not in feature]

        if missing_fields:
            validation_issues.append(f"Feature {idx}: missing fields {missing_fields}")

        # Check position bounds
        input_length = len(feature['input_ids'])
        start_pos = feature['start_positions']
        end_pos = feature['end_positions']

        if start_pos >= input_length or end_pos >= input_length:
            validation_issues.append(f"Feature {idx}: positions out of bounds")

        if start_pos > end_pos and start_pos != 0:  # Allow CLS token mapping
            validation_issues.append(f"Feature {idx}: start > end positions")

    print(f"Validation issues found: {len(validation_issues)}")
    for issue in validation_issues:
        print(f"  {issue}")

    return len(validation_issues) == 0

# Run test
success = test_data_processing_pipeline()
print(f"Data processing test: {'PASSED' if success else 'FAILED'}")
```

### Cache Integrity Testing

```python
from memxlnet.data import ChunkedCacheManager, process_and_cache_dataset

def test_cache_integrity():
    cache_dir = './test_cache'

    # Create cache
    process_and_cache_dataset(
        dataset_name='squad_v2',
        split='validation',
        cache_dir=cache_dir,
        max_examples=50,
        max_seq_length=384,
        doc_stride=128,
        streaming_chunk_size=100,
        max_memory_gb=8.0,
        use_streaming=False,
        tokenizer=tokenizer,
        max_n_segs=4
    )

    # Test cache loading
    cache_manager = ChunkedCacheManager(cache_dir, 100)
    cache_exists = cache_manager.cache_exists('squad_v2', 'validation')

    if not cache_exists:
        print("Cache creation failed")
        return False

    # Load and verify
    total_chunks = cache_manager.get_total_chunks('squad_v2', 'validation')
    total_features = 0

    for chunk_id in range(total_chunks):
        chunk = cache_manager.load_chunk('squad_v2', 'validation', chunk_id)
        total_features += len(chunk)

        # Verify chunk structure
        if chunk and len(chunk) > 0:
            sample_feature = chunk[0]
            required_fields = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
            missing_fields = [field for field in required_fields if field not in sample_feature]

            if missing_fields:
                print(f"Chunk {chunk_id}: missing fields {missing_fields}")
                return False

    print(f"Cache integrity test: {total_features} features across {total_chunks} chunks")
    return True

# Run test
cache_success = test_cache_integrity()
print(f"Cache integrity test: {'PASSED' if cache_success else 'FAILED'}")
```

## Answer Validation Tests

### Comprehensive Answer Matching

```python
def test_answer_validation_strategies():
    from memxlnet.data.text_utils import (
        normalize_answer_for_comparison,
        compare_answers_fuzzy
    )

    test_pairs = [
        # Exact matches
        ("François", "François", True),
        ("café", "café", True),

        # Case differences
        ("CAFÉ", "café", True),
        ("François Mitterrand", "françois mitterrand", True),

        # Accent differences
        ("cafe", "café", True),  # Should match after normalization
        ("naive", "naïve", True),

        # Punctuation differences
        ("New York City", "new york city", True),
        ("twenty-one", "twenty one", True),

        # Partial matches
        ("François Mitterrand", "François", True),  # Containment
        ("the United States", "United States", True),  # Article removal

        # Non-matches
        ("Paris", "London", False),
        ("cat", "dog", False),
    ]

    results = []
    for answer1, answer2, expected in test_pairs:
        # Test normalized comparison
        norm1 = normalize_answer_for_comparison(answer1)
        norm2 = normalize_answer_for_comparison(answer2)
        normalized_match = norm1 == norm2

        # Test fuzzy comparison
        fuzzy_match = compare_answers_fuzzy(answer1, answer2)

        # Test containment
        containment_match = norm1 in norm2 or norm2 in norm1

        # Overall match
        overall_match = normalized_match or fuzzy_match or containment_match

        result = {
            'answer1': answer1,
            'answer2': answer2,
            'expected': expected,
            'normalized_match': normalized_match,
            'fuzzy_match': fuzzy_match,
            'containment_match': containment_match,
            'overall_match': overall_match,
            'correct': overall_match == expected
        }

        results.append(result)

        if not result['correct']:
            print(f"MISMATCH: '{answer1}' vs '{answer2}' - expected {expected}, got {overall_match}")

    # Summary
    correct_results = sum(1 for r in results if r['correct'])
    total_results = len(results)

    print(f"Answer validation test: {correct_results}/{total_results} correct")
    print(f"Accuracy: {correct_results/total_results*100:.1f}%")

    return correct_results == total_results

# Run test
validation_success = test_answer_validation_strategies()
print(f"Answer validation test: {'PASSED' if validation_success else 'FAILED'}")
```

## Performance Testing

### Timing and Memory Usage

```python
import time
import psutil
import os

def performance_test():
    process = psutil.Process(os.getpid())

    # Test Unicode normalization performance
    from memxlnet.data.text_utils import normalize_unicode, normalize_answer_for_comparison

    test_texts = [
        "Le café de la rue Saint-Honoré est très bon.",
        "François Mitterrand était le président de la France.",
        "Die schöne Stadt Zürich liegt in der Schweiz.",
        "El niño pequeño come mañana por la tarde.",
        "Москва - столица России, большой город."
    ] * 1000  # 5000 texts total

    # Test normalize_unicode
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024

    normalized_texts = [normalize_unicode(text) for text in test_texts]

    unicode_time = time.time() - start_time
    unicode_memory = process.memory_info().rss / 1024 / 1024

    # Test normalize_answer_for_comparison
    start_time = time.time()

    comparison_texts = [normalize_answer_for_comparison(text) for text in test_texts]

    comparison_time = time.time() - start_time
    final_memory = process.memory_info().rss / 1024 / 1024

    print("Performance Test Results:")
    print(f"  Texts processed: {len(test_texts)}")
    print(f"  Unicode normalization: {unicode_time:.3f}s ({len(test_texts)/unicode_time:.0f} texts/sec)")
    print(f"  Answer normalization: {comparison_time:.3f}s ({len(test_texts)/comparison_time:.0f} texts/sec)")
    print(f"  Memory usage: {start_memory:.1f}MB -> {final_memory:.1f}MB (+{final_memory-start_memory:.1f}MB)")

    # Performance benchmarks
    unicode_speed = len(test_texts) / unicode_time
    comparison_speed = len(test_texts) / comparison_time

    # Expect at least 1000 texts/sec for each operation
    unicode_ok = unicode_speed > 1000
    comparison_ok = comparison_speed > 500  # More complex operation
    memory_ok = (final_memory - start_memory) < 100  # Less than 100MB increase

    print(f"  Unicode normalization speed: {'OK' if unicode_ok else 'SLOW'}")
    print(f"  Answer normalization speed: {'OK' if comparison_ok else 'SLOW'}")
    print(f"  Memory usage: {'OK' if memory_ok else 'HIGH'}")

    return unicode_ok and comparison_ok and memory_ok

# Run performance test
perf_success = performance_test()
print(f"Performance test: {'PASSED' if perf_success else 'FAILED'}")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors

```python
# Error: ModuleNotFoundError: No module named 'src.text_utils'
# Solution: Ensure src is in Python path
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from text_utils import normalize_unicode
```

#### 2. Unicode Encoding Issues

```python
# Error: UnicodeDecodeError or UnicodeEncodeError
# Solution: Ensure proper UTF-8 encoding
text = "problematic_text"
try:
    # Ensure text is properly encoded
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    normalized = normalize_unicode(text)
except UnicodeError as e:
    print(f"Unicode error: {e}")
    # Handle gracefully or log for investigation
```

#### 3. Position Validation Failures

```python
# Debug position mapping issues
from memxlnet.data.text_utils import validate_answer_positions, fix_answer_positions

context = "problematic context"
answer = "answer"
start = 10
end = 16

# Check if positions are valid
is_valid = validate_answer_positions(context, answer, start, end)

if not is_valid:
    print(f"Invalid positions: {start}-{end}")
    print(f"Context length: {len(context)}")
    print(f"Extracted: '{context[start:end] if start < len(context) and end <= len(context) else 'OUT_OF_BOUNDS'}'")

    # Try to fix
    corrected_start, corrected_end = fix_answer_positions(context, answer, start)
    print(f"Corrected: {corrected_start}-{corrected_end}")
```

#### 4. Test Failures

```python
# Debug test failures
def debug_test_failure(test_name, expected, actual):
    print(f"Test failure: {test_name}")
    print(f"  Expected: {expected}")
    print(f"  Actual: {actual}")

    # Add detailed debugging
    if hasattr(expected, '__len__') and hasattr(actual, '__len__'):
        print(f"  Lengths: expected={len(expected)}, actual={len(actual)}")

    # Character-by-character comparison for strings
    if isinstance(expected, str) and isinstance(actual, str):
        for i, (e_char, a_char) in enumerate(zip(expected, actual)):
            if e_char != a_char:
                print(f"  First difference at position {i}: '{e_char}' vs '{a_char}'")
                break
```

### Test Environment Verification

```python
def verify_test_environment():
    """Verify that the test environment is properly set up."""
    issues = []

    # Check Python path
    import sys
    if 'src' not in [os.path.basename(p) for p in sys.path]:
        issues.append("src directory not in Python path")

    # Check required modules
    required_modules = ['unicodedata', 'torch', 'transformers', 'datasets']
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            issues.append(f"Missing required module: {module}")

    # Check text_utils availability
    try:
        from memxlnet.data.text_utils import normalize_unicode
    except ImportError:
        issues.append("Cannot import src.text_utils")

    # Check tokenizer availability
    try:
        from transformers import XLNetTokenizerFast
        tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
    except Exception as e:
        issues.append(f"Cannot load XLNet tokenizer: {e}")

    if issues:
        print("Test environment issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Test environment verification: PASSED")
        return True

# Verify environment before running tests
env_ok = verify_test_environment()
```

## Custom Test Creation

### Creating Your Own Tests

```python
def create_custom_test(name, test_function, expected_result):
    """Create a custom test case."""
    def run_test():
        try:
            result = test_function()
            success = result == expected_result
            print(f"Test '{name}': {'PASSED' if success else 'FAILED'}")
            if not success:
                print(f"  Expected: {expected_result}")
                print(f"  Got: {result}")
            return success
        except Exception as e:
            print(f"Test '{name}': ERROR - {e}")
            return False

    return run_test

# Example custom tests
def test_custom_unicode():
    from memxlnet.data.text_utils import normalize_unicode
    # Test your specific Unicode case
    text = "your_unicode_text"
    normalized = normalize_unicode(text)
    return len(normalized) > 0  # Or your specific test condition

def test_custom_position():
    from memxlnet.data.text_utils import validate_answer_positions
    # Test your specific position case
    return validate_answer_positions("your context", "your answer", 0, 10)

# Create and run custom tests
custom_tests = [
    create_custom_test("Custom Unicode", test_custom_unicode, True),
    create_custom_test("Custom Position", test_custom_position, True),
]

for test in custom_tests:
    test()
```

### Test Suite Integration

```python
def run_all_tests():
    """Run the complete test suite."""
    test_results = {}

    # Unicode tests
    print("Running Unicode tests...")
    unicode_results = run_unicode_tests()
    unicode_success = all(r['normalized_match'] for r in unicode_results)
    test_results['unicode'] = unicode_success

    # Data processing tests
    print("Running data processing tests...")
    dp_success = test_data_processing_pipeline()
    test_results['data_processing'] = dp_success

    # Answer validation tests
    print("Running answer validation tests...")
    av_success = test_answer_validation_strategies()
    test_results['answer_validation'] = av_success

    # Performance tests
    print("Running performance tests...")
    perf_success = performance_test()
    test_results['performance'] = perf_success

    # Summary
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    print(f"\nTest Suite Summary: {passed_tests}/{total_tests} passed")
    for test_name, success in test_results.items():
        print(f"  {test_name}: {'PASSED' if success else 'FAILED'}")

    return passed_tests == total_tests

# Run complete test suite
all_tests_passed = run_all_tests()
print(f"\nOverall result: {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}")
```

This comprehensive testing guide ensures robust validation of all Unicode and position mapping improvements in the MemXLNet QA system.