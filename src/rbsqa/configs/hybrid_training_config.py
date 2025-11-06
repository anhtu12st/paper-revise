"""
Hybrid Training Configuration for RBS-QA

This module provides the RBSTrainingConfig class that defines
all hyperparameters and settings for the hybrid SL+RL training pipeline.
"""

import argparse
import json
import os
from dataclasses import dataclass, field, MISSING
from typing import Any, Dict, Optional

@dataclass
class RBSTrainingConfig:
    """Configuration for RBS-QA hybrid training combining supervised and RL learning."""

    # Core training settings
    output_dir: str = "./outputs/rbs_experiment"
    run_name: str = "rbs-experiment"
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # RL training settings
    use_rl_training: bool = True
    rl_start_epoch: int = 2
    rl_weight: float = 0.1
    rl_learning_rate: float = 1e-4
    rl_weight_decay: float = 0.01
    rl_batch_size: int = 8
    lambda_cost: float = 0.01  # Cost coefficient for efficiency reward
    gamma: float = 0.99  # Discount factor for RL
    use_value_baseline: bool = True
    value_weight: float = 0.5

    # Model architecture settings
    memory_num_tokens: int = 16
    num_memory_experts: int = 4
    use_rbs_mode: bool = True
    belief_state_threshold: float = 0.7

    # Data loading settings
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Training stability settings
    warmup_steps: int = 500
    max_steps: int = -1  # -1 for epoch-based training
    gradient_accumulation_steps: int = 1

    # Evaluation and saving
    eval_frequency: int = 1
    save_frequency: int = 2
    keep_best_checkpoints: int = 3
    logging_steps: int = 50

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001

    # Hardware settings
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    mixed_precision: bool = True
    dataloader_prefetch_factor: int = 2

    # Logging settings
    use_wandb: bool = False
    wandb_project: str = "rbs-qa"
    log_level: str = "INFO"

    # Reproducibility
    seed: int = 42

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RBSTrainingConfig":
        """Create config from command line arguments."""
        config_dict = {}

        for field_name, field_def in cls.__dataclass_fields__.items():
            if hasattr(args, field_name):
                config_dict[field_name] = getattr(args, field_name)
            elif field_def.default is not MISSING:
                config_dict[field_name] = field_def.default

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            result[field_name] = value
        return result

    def save(self, save_path: str) -> None:
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, load_path: str) -> "RBSTrainingConfig":
        """Load config from JSON file."""
        with open(load_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate RL settings
        if self.use_rl_training and self.rl_start_epoch >= self.num_epochs:
            raise ValueError(
                f"rl_start_epoch ({self.rl_start_epoch}) must be less than "
                f"num_epochs ({self.num_epochs}) when use_rl_training is True"
            )

        # Validate model settings
        if self.num_memory_experts <= 0 or self.num_memory_experts > 8:
            raise ValueError("num_memory_experts must be between 1 and 8")

        if self.memory_num_tokens <= 0:
            raise ValueError("memory_num_tokens must be positive")

        # Validate training settings
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        # Validate belief state threshold
        if not 0 <= self.belief_state_threshold <= 1:
            raise ValueError("belief_state_threshold must be between 0 and 1")

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        self.validate()


def create_balanced_config(
    num_epochs: int = 10,
    batch_size: int = 8,
    memory_num_tokens: int = 16,
    num_memory_experts: int = 4,
    use_rl_training: bool = True,
    **kwargs
) -> RBSTrainingConfig:
    """
    Create a balanced configuration for RBS-QA training.

    Args:
        num_epochs: Number of training epochs
        batch_size: Training batch size
        memory_num_tokens: Number of memory tokens
        num_memory_experts: Number of memory experts
        use_rl_training: Whether to use RL training
        **kwargs: Additional configuration overrides

    Returns:
        RBSTrainingConfig instance with balanced settings
    """
    config_dict = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "memory_num_tokens": memory_num_tokens,
        "num_memory_experts": num_memory_experts,
        "use_rl_training": use_rl_training,
        "rl_start_epoch": max(2, num_epochs // 3),  # Start RL after 1/3 of training
        "learning_rate": 5e-5,
        "rl_learning_rate": 1e-4,
        "lambda_cost": 0.01,
        "belief_state_threshold": 0.7,
    }

    # Add any overrides
    config_dict.update(kwargs)

    return RBSTrainingConfig(**config_dict)


def create_quick_debug_config(**kwargs) -> RBSTrainingConfig:
    """
    Create a quick configuration for debugging/testing.

    Args:
        **kwargs: Configuration overrides

    Returns:
        RBSTrainingConfig optimized for quick debugging
    """
    config_dict = {
        "num_epochs": 2,
        "batch_size": 2,
        "memory_num_tokens": 4,
        "num_memory_experts": 2,
        "use_rl_training": True,
        "rl_start_epoch": 1,
        "eval_frequency": 1,
        "save_frequency": 1,
        "logging_steps": 1,
        "warmup_steps": 1,
        "dataloader_num_workers": 0,  # Disable multiprocessing for debugging
        "use_wandb": False,
    }

    # Add any overrides
    config_dict.update(kwargs)

    return RBSTrainingConfig(**config_dict)