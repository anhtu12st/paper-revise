"""
Text utilities for SQuAD v2 processing with Unicode normalization and answer validation.

This module provides enhanced text processing functions to handle:
- Unicode normalization for accented characters
- Improved answer position mapping
- Robust text comparison for validation
"""

import re
import string
import unicodedata
from typing import Any


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text to handle accents and special characters consistently.

    Args:
        text: Input text that may contain Unicode characters

    Returns:
        Normalized text using NFC normalization for consistent representation

    Example:
        >>> normalize_unicode("café")  # é as combining chars
        'café'  # é as single char
        >>> normalize_unicode("naïve")
        'naïve'
    """
    if not text:
        return text

    # NFD decomposes, then NFC recomposes for consistent representation
    # This handles cases where the same character can be represented differently
    return unicodedata.normalize("NFC", text)


def normalize_answer_for_comparison(s: str) -> str:
    """Enhanced answer normalization with Unicode support for SQuAD evaluation.

    This function applies the standard SQuAD normalization but with proper
    Unicode handling to ensure consistent comparison of answers.

    Args:
        s: Answer string to normalize

    Returns:
        Normalized string suitable for comparison
    """
    if not s:
        return ""

    # First apply Unicode normalization
    s = normalize_unicode(s)

    def remove_articles(text: str) -> str:
        """Remove English articles."""
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        """Fix whitespace by collapsing multiple spaces."""
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        """Remove punctuation marks."""
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        """Convert to lowercase."""
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def find_answer_span_with_normalization(context: str, answer_text: str) -> tuple[int | None, int | None]:
    """Find answer span in context with Unicode normalization.

    This function attempts to find the answer in the context using various
    normalization strategies to handle Unicode differences.

    Args:
        context: The context text to search in
        answer_text: The answer text to find

    Returns:
        Tuple of (start_char, end_char) positions, or (None, None) if not found
    """
    if not answer_text or not context:
        return None, None

    # Normalize both strings
    normalized_context = normalize_unicode(context)
    normalized_answer = normalize_unicode(answer_text)

    # Try exact match first
    start = normalized_context.find(normalized_answer)
    if start != -1:
        return start, start + len(normalized_answer)

    # Try with whitespace normalization
    normalized_answer_ws = " ".join(normalized_answer.split())
    start = normalized_context.find(normalized_answer_ws)
    if start != -1:
        return start, start + len(normalized_answer_ws)

    # Try with case insensitive match
    start = normalized_context.lower().find(normalized_answer.lower())
    if start != -1:
        # Find the actual end position in the original case
        end = start + len(normalized_answer)
        return start, end

    # Try with punctuation removed
    def remove_punct(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    context_no_punct = remove_punct(normalized_context)
    answer_no_punct = remove_punct(normalized_answer)

    if answer_no_punct and context_no_punct:
        start = context_no_punct.find(answer_no_punct)
        if start != -1:
            # This is approximate since we removed punctuation
            return start, start + len(answer_no_punct)

    return None, None


def map_normalized_to_original_position(
    original: str, normalized: str, normalized_pos: int
) -> tuple[int | None, int | None]:
    """Map position in normalized text back to original text.

    This is a complex operation needed when we find an answer in normalized text
    but need to return positions in the original text.

    Args:
        original: Original text
        normalized: Normalized version of the text
        normalized_pos: Position in the normalized text

    Returns:
        Tuple of (start, end) positions in original text
    """
    # For simplicity, this is a basic implementation
    # In practice, this would need more sophisticated character-by-character mapping
    if normalized_pos < len(original):
        return normalized_pos, normalized_pos + 1
    return None, None


def validate_answer_positions(context: str, answer_text: str, start_char: int, end_char: int) -> bool:
    """Validate that the given character positions actually contain the answer.

    Args:
        context: The full context text
        answer_text: The expected answer text
        start_char: Start character position
        end_char: End character position

    Returns:
        True if the positions are valid, False otherwise
    """
    if start_char < 0 or end_char > len(context) or start_char >= end_char:
        return False

    extracted = context[start_char:end_char]

    # Check exact match
    if extracted == answer_text:
        return True

    # Check normalized match
    if normalize_unicode(extracted) == normalize_unicode(answer_text):
        return True

    # Check case-insensitive match
    if extracted.lower() == answer_text.lower():
        return True

    return False


def fix_answer_positions(context: str, answer_text: str, start_char: int) -> tuple[int, int]:
    """Attempt to fix answer positions that may have off-by-one errors.

    Args:
        context: The context text
        answer_text: The answer text
        start_char: The claimed start position

    Returns:
        Tuple of corrected (start_char, end_char) positions
    """
    original_end = start_char + len(answer_text)

    # Check if original positions are correct
    if validate_answer_positions(context, answer_text, start_char, original_end):
        return start_char, original_end

    # Try adjusting by ±1 character
    for offset in [-1, 1, -2, 2]:
        new_start = start_char + offset
        new_end = new_start + len(answer_text)

        if (
            0 <= new_start < len(context)
            and new_end <= len(context)
            and validate_answer_positions(context, answer_text, new_start, new_end)
        ):
            return new_start, new_end

    # Try finding the answer in the context
    found_start, found_end = find_answer_span_with_normalization(context, answer_text)
    if found_start is not None and found_end is not None:
        return found_start, found_end

    # Return original positions if nothing works
    return start_char, original_end


def compare_answers_fuzzy(answer1: str, answer2: str) -> bool:
    """Compare two answers with fuzzy matching for validation.

    Args:
        answer1: First answer string
        answer2: Second answer string

    Returns:
        True if answers are similar enough, False otherwise
    """
    if not answer1 or not answer2:
        return answer1 == answer2

    # Normalize both answers
    norm1 = normalize_answer_for_comparison(answer1)
    norm2 = normalize_answer_for_comparison(answer2)

    # Exact match after normalization
    if norm1 == norm2:
        return True

    # Check if one is contained in the other
    if norm1 in norm2 or norm2 in norm1:
        return True

    # For more sophisticated fuzzy matching, you could add:
    # - Edit distance calculation
    # - Jaccard similarity
    # - Token-based similarity

    return False


# -------- Unicode & Answer Diagnostic Helpers (New) -------- #


def unicode_codepoints(s: str) -> str:
    """Return a compact hex code point representation for diagnostics."""
    return " ".join(f"{ord(c):04X}" for c in s)


def unicode_answers_match(expected: str, extracted: str) -> bool:
    """Unified comparison pipeline for answers with progressive normalization.

    Order of checks:
    1. Exact
    2. NFC-normalized exact
    3. Case-insensitive
    4. Normalized (SQuAD-style)
    5. Fuzzy containment / fuzzy normalization
    """
    if expected == extracted:
        return True
    if normalize_unicode(expected) == normalize_unicode(extracted):
        return True
    if expected.lower() == extracted.lower():
        return True
    if normalize_answer_for_comparison(expected) == normalize_answer_for_comparison(extracted):
        return True
    if compare_answers_fuzzy(expected, extracted):
        return True
    return False


def diagnose_unicode_mismatch(expected: str, extracted: str) -> dict[str, Any]:
    """Produce a structured diagnostic report for a unicode answer mismatch."""
    report: dict[str, Any] = {
        "expected": expected,
        "extracted": extracted,
        "expected_nfc": normalize_unicode(expected),
        "extracted_nfc": normalize_unicode(extracted),
        "expected_codepoints": unicode_codepoints(expected),
        "extracted_codepoints": unicode_codepoints(extracted),
        "expected_norm": normalize_answer_for_comparison(expected),
        "extracted_norm": normalize_answer_for_comparison(extracted),
        "exact_match": expected == extracted,
        "nfc_match": normalize_unicode(expected) == normalize_unicode(extracted),
        "case_match": expected.lower() == extracted.lower(),
        "norm_match": normalize_answer_for_comparison(expected) == normalize_answer_for_comparison(extracted),
        "fuzzy_match": compare_answers_fuzzy(expected, extracted),
    }
    return report


def find_all_occurrences(context: str, answer: str) -> list[tuple[int, int]]:
    """Find all (start,end) occurrences of answer (with unicode normalization) in context."""
    if not answer:
        return []
    norm_context = normalize_unicode(context)
    norm_answer = normalize_unicode(answer)
    occurrences: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = norm_context.find(norm_answer, start)
        if idx == -1:
            break
        occurrences.append((idx, idx + len(norm_answer)))
        start = idx + 1
    return occurrences


def choose_best_occurrence(occurrences: list[tuple[int, int]], original_start: int) -> tuple[int, int] | None:
    """Choose occurrence closest to an original start (min absolute distance)."""
    if not occurrences:
        return None
    best = min(occurrences, key=lambda oc: abs(oc[0] - original_start))
    return best


def characterize_boundary_delta(pred_start: int, true_start: int) -> str:
    """Return a label describing delta pattern (e.g., 'off_by_-1', 'exact')."""
    delta = pred_start - true_start
    if delta == 0:
        return "exact"
    if abs(delta) <= 2:
        return f"off_by_{delta}"
    return f"large_shift_{delta}"


# Test cases for validation
UNICODE_TEST_CASES = [
    {"name": "French accents", "context": "Le café est très bon.", "answer": "café", "start": 3, "end": 7},
    {"name": "German umlauts", "context": "Zürich is in Switzerland.", "answer": "Zürich", "start": 0, "end": 6},
    {"name": "Mixed accents", "context": "François était naïve.", "answer": "François", "start": 0, "end": 8},
    {"name": "Spanish characters", "context": "El niño come mañana.", "answer": "niño", "start": 3, "end": 7},
    {
        "name": "Multiple accents",
        "context": "Café, thé, et café crème.",
        "answer": "café crème",
        "start": 15,
        "end": 26,
    },
]


def run_unicode_tests() -> list[dict[str, Any]]:
    """Run Unicode test cases to validate the functions.

    Returns:
        List of test results
    """
    results: list[dict[str, Any]] = []

    for test_case in UNICODE_TEST_CASES:
        # Extract values with explicit type casting
        name = str(test_case["name"])
        context = str(test_case["context"])
        answer = str(test_case["answer"])
        start_val = test_case["start"]
        end_val = test_case["end"]
        # Handle both int and str values
        expected_start = int(start_val) if isinstance(start_val, (int, str, float)) else 0
        expected_end = int(end_val) if isinstance(end_val, (int, str, float)) else 0

        result: dict[str, Any] = {
            "name": name,
            "context": context,
            "answer": answer,
            "expected_start": expected_start,
            "expected_end": expected_end,
        }

        # Test position validation
        is_valid = validate_answer_positions(context, answer, expected_start, expected_end)
        result["position_valid"] = is_valid

        # Test answer finding
        found_start, found_end = find_answer_span_with_normalization(context, answer)
        result["found_start"] = found_start
        result["found_end"] = found_end
        result["found_correctly"] = found_start == expected_start and found_end == expected_end

        # Test normalization
        extracted = context[expected_start:expected_end]
        normalized_extracted = normalize_answer_for_comparison(extracted)
        normalized_answer = normalize_answer_for_comparison(answer)
        result["normalized_match"] = normalized_extracted == normalized_answer

        results.append(result)

    return results
