"""MemXLNet: Memory-Augmented XLNet for Long-Context Question Answering."""

__version__ = "0.1.0"

from memxlnet.data.dataset import (
    SquadLikeQADataset,
    configure_memory_tokens,
    create_dataloader,
    process_and_cache_dataset,
)
from memxlnet.models.memory_modules import (
    DifferentiableMemory,
    MemoryController,
)
from memxlnet.models.memxlnet_qa import MemXLNetForQA
from memxlnet.training.trainer import TrainingConfig, XLNetRecurrentTrainer

__all__ = [
    "__version__",
    # Models
    "MemXLNetForQA",
    "DifferentiableMemory",
    "MemoryController",
    # Training
    "TrainingConfig",
    "XLNetRecurrentTrainer",
    # Data
    "SquadLikeQADataset",
    "create_dataloader",
    "process_and_cache_dataset",
    "configure_memory_tokens",
]
