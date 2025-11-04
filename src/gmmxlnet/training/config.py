"""Training configuration for GMM-XLNet models.

This module extends the base TrainingConfig with GMM-specific parameters
for training models with Gated Memory Mixture architecture.
"""

import json
from dataclasses import dataclass
from typing import Any

from memxlnet.training import TrainingConfig


@dataclass
class GMMTrainingConfig(TrainingConfig):
    """Training configuration for GMM-XLNet models.

    Extends TrainingConfig with parameters specific to Gated Memory Mixture
    architecture, including expert count, routing behavior, and regularization.

    Args:
        use_gmm_memory: Enable GMM memory system (default: False)
        num_memory_experts: Number of memory experts k ∈ [2, 8] (default: 4)
        routing_temperature: Temperature for routing softmax, must be > 0 (default: 1.0)
        routing_mode: Routing mode for read operations, "write-based" or "read-based" (default: "write-based")
        entropy_regularization_weight: Weight for entropy regularization loss, must be >= 0 (default: 0.0)
        load_balance_weight: Weight for load balance loss, must be >= 0 (default: 0.01)
        expert_init_strategies: Initialization strategy per expert (default: ["learned"] * k)
    """

    # GMM-specific parameters
    use_gmm_memory: bool = False
    num_memory_experts: int = 4
    routing_temperature: float = 1.0
    routing_mode: str = "write-based"
    entropy_regularization_weight: float = 0.0
    load_balance_weight: float = 0.01
    expert_init_strategies: list[str] | None = None

    def __post_init__(self):
        """Post-initialization validation for GMM-specific parameters."""
        # Call parent __post_init__ first
        super().__post_init__()

        # Initialize expert_init_strategies if not provided
        if self.expert_init_strategies is None:
            self.expert_init_strategies = ["learned"] * self.num_memory_experts

        # Only validate GMM params if GMM memory is enabled
        if self.use_gmm_memory:
            self._validate_gmm_params()

    def _validate_gmm_params(self):
        """Validate GMM-specific parameters.

        Raises:
            ValueError: If any parameter is invalid with descriptive error message.
        """
        # Validate num_memory_experts
        if not isinstance(self.num_memory_experts, int) or self.num_memory_experts < 2 or self.num_memory_experts > 8:
            raise ValueError(
                f"num_memory_experts must be an integer in [2, 8] for efficiency, got {self.num_memory_experts}"
            )

        # Validate routing_temperature
        if not isinstance(self.routing_temperature, (int, float)) or self.routing_temperature <= 0:
            raise ValueError(f"routing_temperature must be > 0, got {self.routing_temperature}")

        # Validate routing_mode
        valid_routing_modes = ["write-based", "read-based"]
        if self.routing_mode not in valid_routing_modes:
            raise ValueError(f"routing_mode must be one of {valid_routing_modes}, got '{self.routing_mode}'")

        # Validate entropy_regularization_weight
        if not isinstance(self.entropy_regularization_weight, (int, float)) or self.entropy_regularization_weight < 0:
            raise ValueError(f"entropy_regularization_weight must be >= 0, got {self.entropy_regularization_weight}")

        # Validate load_balance_weight
        if not isinstance(self.load_balance_weight, (int, float)) or self.load_balance_weight < 0:
            raise ValueError(f"load_balance_weight must be >= 0, got {self.load_balance_weight}")

        # Validate expert_init_strategies length
        if self.expert_init_strategies is not None:
            if len(self.expert_init_strategies) != self.num_memory_experts:
                raise ValueError(
                    f"expert_init_strategies length must match num_memory_experts "
                    f"({self.num_memory_experts}), got {len(self.expert_init_strategies)}"
                )

            # Validate each strategy is valid
            valid_strategies = ["learned", "zeros"]
            for i, strategy in enumerate(self.expert_init_strategies):
                if strategy not in valid_strategies:
                    raise ValueError(f"expert_init_strategies[{i}] must be one of {valid_strategies}, got '{strategy}'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary.

        Returns:
            Dictionary containing all configuration parameters including GMM-specific ones.
        """
        config_dict = {}

        # Serialize all dataclass fields
        for key, value in self.__dict__.items():
            config_dict[key] = value

        # Add memory_type metadata for GMM models
        if self.use_gmm_memory:
            config_dict["memory_type"] = "gmm"

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "GMMTrainingConfig":
        """Deserialize configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            GMMTrainingConfig instance with parameters from dictionary
        """
        # Remove memory_type metadata if present (not a constructor parameter)
        config_dict = {k: v for k, v in config_dict.items() if k != "memory_type"}

        return cls(**config_dict)

    def to_json(self, filepath: str):
        """Save configuration to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "GMMTrainingConfig":
        """Load configuration from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            GMMTrainingConfig instance loaded from file
        """
        with open(filepath) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Preset configuration factory functions


def gmm_small_config(**kwargs) -> GMMTrainingConfig:
    """Factory for small GMM configuration with k=2 experts.

    This preset is ideal for:
    - Prototyping and testing
    - Limited GPU memory scenarios
    - Quick experiments

    Args:
        **kwargs: Additional configuration parameters to override defaults

    Returns:
        GMMTrainingConfig configured for small-scale GMM training
    """
    # Set defaults, then merge with kwargs (kwargs take precedence)
    defaults: dict[str, Any] = {
        "use_gmm_memory": True,
        "num_memory_experts": 2,
        "routing_temperature": 1.0,
        "routing_mode": "write-based",
        "entropy_regularization_weight": 0.0,
        "load_balance_weight": 0.01,
    }
    defaults.update(kwargs)
    return GMMTrainingConfig(**defaults)


def gmm_balanced_config(**kwargs) -> GMMTrainingConfig:
    """Factory for balanced GMM configuration with k=4 experts.

    This preset is ideal for:
    - Most research experiments
    - Balanced capacity and computational cost
    - Production models with moderate resources

    Args:
        **kwargs: Additional configuration parameters to override defaults

    Returns:
        GMMTrainingConfig configured for balanced GMM training
    """
    # Set defaults, then merge with kwargs (kwargs take precedence)
    defaults: dict[str, Any] = {
        "use_gmm_memory": True,
        "num_memory_experts": 4,
        "routing_temperature": 1.0,
        "routing_mode": "write-based",
        "entropy_regularization_weight": 0.0,
        "load_balance_weight": 0.01,
    }
    defaults.update(kwargs)
    return GMMTrainingConfig(**defaults)


def gmm_large_config(**kwargs) -> GMMTrainingConfig:
    """Factory for large GMM configuration with k=8 experts.

    This preset is ideal for:
    - Production models requiring high capacity
    - Large-scale experiments
    - Maximum expert specialization

    Args:
        **kwargs: Additional configuration parameters to override defaults

    Returns:
        GMMTrainingConfig configured for large-scale GMM training
    """
    # Set defaults, then merge with kwargs (kwargs take precedence)
    defaults: dict[str, Any] = {
        "use_gmm_memory": True,
        "num_memory_experts": 8,
        "routing_temperature": 1.0,
        "routing_mode": "write-based",
        "entropy_regularization_weight": 0.0,
        "load_balance_weight": 0.01,
    }
    defaults.update(kwargs)
    return GMMTrainingConfig(**defaults)
