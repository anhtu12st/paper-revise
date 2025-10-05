"""Data processing and dataset utilities."""

from memxlnet.data.dataset import (
    ChunkedCacheManager,
    MemoryCollateConfig,
    SquadLikeQADataset,
    TimeStepMajorDataLoader,
    configure_memory_tokens,
    create_dataloader,
    create_dataset_from_cache,
    create_evaluation_dataloader,
    process_and_cache_dataset,
)
from memxlnet.data.text_utils import (
    find_answer_span_with_normalization,
    fix_answer_positions,
    normalize_unicode,
    validate_answer_positions,
)

__all__ = [
    # Dataset classes
    "SquadLikeQADataset",
    "ChunkedCacheManager",
    "TimeStepMajorDataLoader",
    "MemoryCollateConfig",
    # Functions
    "create_dataloader",
    "create_dataset_from_cache",
    "process_and_cache_dataset",
    "configure_memory_tokens",
    "create_evaluation_dataloader",
    # Text utilities
    "normalize_unicode",
    "validate_answer_positions",
    "fix_answer_positions",
    "find_answer_span_with_normalization",
]
