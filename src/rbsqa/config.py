"""
Training configuration for RBS-QA models.

This module extends GMMTrainingConfig with RBS-specific parameters
for training models with Reasoning Belief State architecture.
"""

import json
from dataclasses import dataclass
from typing import Any

from gmmxlnet.training import GMMTrainingConfig


@dataclass
class RBSTrainingConfig(GMMTrainingConfig):
    """Training configuration for RBS-QA models.

    Extends GMMTrainingConfig with parameters specific to Reasoning Belief State
    architecture, including belief state tracking, non-monotonic reasoning,
    and reinforcement learning-based halting policies.

    Args:
        use_belief_state: Enable belief state tracking (default: True)
        belief_state_threshold: Minimum confidence to consider halting (default: 0.7)
        enable_re_scoring: Enable re-scoring of past spans with new context (default: True)
        confidence_calibration: Enable learnable confidence calibration (default: True)
        re_scoring_method: Method for re-scoring past spans (default: "context_weighted")
        enable_trend_analysis: Enable confidence trend analysis (default: True)
        max_segments: Maximum number of segments to process (default: 32)
        belief_state_memory_limit: Maximum span candidates to keep in history (default: 100)
        halting_patience: Number of segments to wait after confidence threshold (default: 3)
        enable_learnable_re_scoring: Enable learnable re-scoring network (default: False)
        belief_update_frequency: How often to update belief state (default: "every_segment")
        non_monotonic_revision_threshold: Minimum confidence gain for revision (default: 0.05)
        enable_belief_state_visualization: Enable detailed belief tracking for analysis (default: False)

        # Halting policy settings
        use_halting_policy: Enable reinforcement learning halting policy (default: True)
        halting_policy_hidden_dim: Hidden dimension for halting policy network (default: 64)
        halting_policy_layers: Number of layers in halting policy network (default: 2)
        halting_temperature: Temperature for action sampling (default: 1.0)
        halting_exploration_rate: Epsilon-greedy exploration rate (default: 0.1)

        # RL training settings
        rl_weight: Weight for RL loss in total loss (default: 0.1)
        lambda_cost: Cost per processed segment (default: 0.01)
        gamma: Discount factor for future rewards (default: 0.99)
        use_value_baseline: Use value function as baseline (default: True)
        value_weight: Weight for value function loss (default: 0.5)

        # Training schedule
        rl_start_epoch: Epoch to start RL training (default: 2)
        rl_update_frequency: Update policy every N episodes (default: 10)
    """

    # GMM dependency - RBS requires GMM
    use_gmm_memory: bool = True

    # RBS-specific parameters
    use_belief_state: bool = True
    belief_state_threshold: float = 0.7
    enable_re_scoring: bool = True
    confidence_calibration: bool = True
    re_scoring_method: str = "context_weighted"
    enable_trend_analysis: bool = True
    max_segments: int = 32
    belief_state_memory_limit: int = 100
    halting_patience: int = 3
    enable_learnable_re_scoring: bool = False
    belief_update_frequency: str = "every_segment"
    non_monotonic_revision_threshold: float = 0.05
    enable_belief_state_visualization: bool = False

    # Halting policy settings
    use_halting_policy: bool = True
    halting_policy_hidden_dim: int = 64
    halting_policy_layers: int = 2
    halting_temperature: float = 1.0
    halting_exploration_rate: float = 0.1

    # RL training settings
    rl_weight: float = 0.1  # α in L_total = L_QA + α * L_RL
    lambda_cost: float = 0.01  # Cost per segment
    gamma: float = 0.99  # Discount factor
    use_value_baseline: bool = True
    value_weight: float = 0.5

    # Training schedule
    rl_start_epoch: int = 2  # Start RL training after this epoch
    rl_update_frequency: int = 10  # Update policy every N episodes

    def __post_init__(self):
        """Post-initialization validation for RBS-specific parameters."""
        # Call parent __post_init__ first
        super().__post_init__()

        # Only validate RBS params if belief state is enabled
        if self.use_belief_state:
            self._validate_rbs_params()

    def _validate_rbs_params(self):
        """Validate RBS-specific parameters.

        Raises:
            ValueError: If any parameter is invalid with descriptive error message.
        """
        # Validate belief_state_threshold
        if not isinstance(self.belief_state_threshold, (int, float)) or not (0.0 <= self.belief_state_threshold <= 1.0):
            raise ValueError(f"belief_state_threshold must be in [0.0, 1.0], got {self.belief_state_threshold}")

        # Validate re_scoring_method
        valid_re_scoring_methods = ["context_weighted", "learned", "exponential_decay"]
        if self.re_scoring_method not in valid_re_scoring_methods:
            raise ValueError(f"re_scoring_method must be one of {valid_re_scoring_methods}, got '{self.re_scoring_method}'")

        # Validate max_segments
        if not isinstance(self.max_segments, int) or self.max_segments <= 0:
            raise ValueError(f"max_segments must be a positive integer, got {self.max_segments}")

        # Validate belief_state_memory_limit
        if not isinstance(self.belief_state_memory_limit, int) or self.belief_state_memory_limit <= 0:
            raise ValueError(f"belief_state_memory_limit must be a positive integer, got {self.belief_state_memory_limit}")

        # Validate halting_patience
        if not isinstance(self.halting_patience, int) or self.halting_patience < 0:
            raise ValueError(f"halting_patience must be a non-negative integer, got {self.halting_patience}")

        # Validate belief_update_frequency
        valid_frequencies = ["every_segment", "every_other", "every_third", "every_fifth"]
        if self.belief_update_frequency not in valid_frequencies:
            raise ValueError(f"belief_update_frequency must be one of {valid_frequencies}, got '{self.belief_update_frequency}'")

        # Validate non_monotonic_revision_threshold
        if not isinstance(self.non_monotonic_revision_threshold, (int, float)) or self.non_monotonic_revision_threshold < 0:
            raise ValueError(f"non_monotonic_revision_threshold must be >= 0, got {self.non_monotonic_revision_threshold}")

        # Validate consistency with GMM parameters
        if not self.use_gmm_memory:
            raise ValueError("RBS-QA requires GMM memory to be enabled. Set use_gmm_memory=True")

        # Validate re_scoring method consistency
        if self.enable_learnable_re_scoring and self.re_scoring_method != "learned":
            raise ValueError("enable_learnable_re_scoring=True requires re_scoring_method='learned'")

        if self.re_scoring_method == "learned" and not self.enable_learnable_re_scoring:
            raise ValueError("re_scoring_method='learned' requires enable_learnable_re_scoring=True")

        # Validate halting policy parameters
        if self.use_halting_policy:
            self._validate_halting_policy_params()

    def _validate_halting_policy_params(self):
        """Validate halting policy specific parameters.

        Raises:
            ValueError: If any halting policy parameter is invalid.
        """
        # Validate halting_policy_hidden_dim
        if not isinstance(self.halting_policy_hidden_dim, int) or self.halting_policy_hidden_dim <= 0:
            raise ValueError(f"halting_policy_hidden_dim must be a positive integer, got {self.halting_policy_hidden_dim}")

        # Validate halting_policy_layers
        if not isinstance(self.halting_policy_layers, int) or self.halting_policy_layers < 1:
            raise ValueError(f"halting_policy_layers must be a positive integer, got {self.halting_policy_layers}")

        # Validate halting_temperature
        if not isinstance(self.halting_temperature, (int, float)) or self.halting_temperature <= 0:
            raise ValueError(f"halting_temperature must be positive, got {self.halting_temperature}")

        # Validate halting_exploration_rate
        if not isinstance(self.halting_exploration_rate, (int, float)) or not (0.0 <= self.halting_exploration_rate <= 1.0):
            raise ValueError(f"halting_exploration_rate must be in [0.0, 1.0], got {self.halting_exploration_rate}")

        # Validate RL training parameters
        if not isinstance(self.rl_weight, (int, float)) or self.rl_weight < 0:
            raise ValueError(f"rl_weight must be >= 0, got {self.rl_weight}")

        if not isinstance(self.lambda_cost, (int, float)) or self.lambda_cost < 0:
            raise ValueError(f"lambda_cost must be >= 0, got {self.lambda_cost}")

        if not isinstance(self.gamma, (int, float)) or not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0.0, 1.0], got {self.gamma}")

        if not isinstance(self.value_weight, (int, float)) or self.value_weight < 0:
            raise ValueError(f"value_weight must be >= 0, got {self.value_weight}")

        # Validate training schedule parameters
        if not isinstance(self.rl_start_epoch, int) or self.rl_start_epoch < 0:
            raise ValueError(f"rl_start_epoch must be a non-negative integer, got {self.rl_start_epoch}")

        if not isinstance(self.rl_update_frequency, int) or self.rl_update_frequency <= 0:
            raise ValueError(f"rl_update_frequency must be a positive integer, got {self.rl_update_frequency}")

    def get_belief_update_interval(self) -> int:
        """Get the actual interval for belief updates based on frequency setting.

        Returns:
            Number of segments between belief updates
        """
        frequency_map = {
            "every_segment": 1,
            "every_other": 2,
            "every_third": 3,
            "every_fifth": 5
        }
        return frequency_map.get(self.belief_update_frequency, 1)

    def should_enable_belief_features(self) -> dict:
        """Get a dictionary of which belief state features are enabled.

        Returns:
            Dictionary mapping feature names to boolean enabled status
        """
        return {
            "tracking": self.use_belief_state,
            "re_scoring": self.enable_re_scoring and self.use_belief_state,
            "calibration": self.confidence_calibration and self.use_belief_state,
            "trend_analysis": self.enable_trend_analysis and self.use_belief_state,
            "learnable_re_scoring": self.enable_learnable_re_scoring and self.enable_re_scoring,
            "visualization": self.enable_belief_state_visualization and self.use_belief_state
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary.

        Returns:
            Dictionary containing all configuration parameters including RBS-specific ones.
        """
        config_dict = super().to_dict()

        # Add RBS-specific metadata
        if self.use_belief_state:
            config_dict["reasoning_type"] = "rbs"
            config_dict["rbs_version"] = "1.0"

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RBSTrainingConfig":
        """Deserialize configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            RBSTrainingConfig instance with parameters from dictionary
        """
        # Remove metadata fields that are not constructor parameters
        metadata_fields = ["reasoning_type", "rbs_version", "memory_type"]
        config_dict = {k: v for k, v in config_dict.items() if k not in metadata_fields}

        return cls(**config_dict)

    def to_json(self, filepath: str):
        """Save configuration to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "RBSTrainingConfig":
        """Load configuration from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            RBSTrainingConfig instance loaded from file
        """
        with open(filepath) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Preset configuration factory functions


def rbs_lightweight_config(**kwargs) -> RBSTrainingConfig:
    """Factory for lightweight RBS configuration with minimal overhead.

    This preset is ideal for:
    - Fast prototyping and testing
    - Scenarios with minimal computational budget
    - Initial experiments with belief state tracking

    Args:
        **kwargs: Additional configuration parameters to override defaults

    Returns:
        RBSTrainingConfig configured for lightweight RBS training
    """
    # Set defaults, then merge with kwargs (kwargs take precedence)
    defaults: dict[str, Any] = {
        # GMM settings (lightweight)
        "use_gmm_memory": True,
        "num_memory_experts": 2,
        "memory_num_tokens": 8,
        "routing_temperature": 1.2,
        "load_balance_weight": 0.005,

        # RBS settings (lightweight)
        "use_belief_state": True,
        "belief_state_threshold": 0.8,  # Higher threshold to reduce processing
        "enable_re_scoring": True,
        "re_scoring_method": "context_weighted",  # Simple, efficient method
        "confidence_calibration": False,  # Disable for speed
        "enable_trend_analysis": False,  # Disable for speed
        "max_segments": 16,  # Fewer segments
        "belief_state_memory_limit": 50,  # Smaller memory
        "halting_patience": 2,  # Less patience
        "enable_learnable_re_scoring": False,
        "belief_update_frequency": "every_other",  # Update less frequently
        "non_monotonic_revision_threshold": 0.1,  # Higher threshold for revisions
        "enable_belief_state_visualization": False,

        # Halting policy settings (lightweight)
        "use_halting_policy": True,
        "halting_policy_hidden_dim": 32,  # Smaller network
        "halting_policy_layers": 1,  # Single layer
        "halting_temperature": 1.5,  # Higher temperature for more exploration
        "halting_exploration_rate": 0.2,  # More exploration

        # RL training settings (lightweight)
        "rl_weight": 0.05,  # Lower RL weight
        "lambda_cost": 0.02,  # Higher cost to encourage early halting
        "gamma": 0.95,  # Lower discount factor
        "use_value_baseline": True,
        "value_weight": 0.3,

        # Training schedule (lightweight)
        "rl_start_epoch": 1,  # Start RL earlier
        "rl_update_frequency": 5,  # Update more frequently
    }
    defaults.update(kwargs)
    return RBSTrainingConfig(**defaults)


def rbs_balanced_config(**kwargs) -> RBSTrainingConfig:
    """Factory for balanced RBS configuration with good performance tradeoffs.

    This preset is ideal for:
    - Most research experiments
    - Balanced accuracy and computational cost
    - Production models with moderate resources

    Args:
        **kwargs: Additional configuration parameters to override defaults

    Returns:
        RBSTrainingConfig configured for balanced RBS training
    """
    # Set defaults, then merge with kwargs (kwargs take precedence)
    defaults: dict[str, Any] = {
        # GMM settings (balanced)
        "use_gmm_memory": True,
        "num_memory_experts": 4,
        "memory_num_tokens": 16,
        "routing_temperature": 1.0,
        "load_balance_weight": 0.01,

        # RBS settings (balanced)
        "use_belief_state": True,
        "belief_state_threshold": 0.7,
        "enable_re_scoring": True,
        "re_scoring_method": "context_weighted",
        "confidence_calibration": True,
        "enable_trend_analysis": True,
        "max_segments": 32,
        "belief_state_memory_limit": 100,
        "halting_patience": 3,
        "enable_learnable_re_scoring": False,
        "belief_update_frequency": "every_segment",
        "non_monotonic_revision_threshold": 0.05,
        "enable_belief_state_visualization": False,

        # Halting policy settings (balanced)
        "use_halting_policy": True,
        "halting_policy_hidden_dim": 64,  # Default size
        "halting_policy_layers": 2,  # Default layers
        "halting_temperature": 1.0,  # Default temperature
        "halting_exploration_rate": 0.1,  # Default exploration

        # RL training settings (balanced)
        "rl_weight": 0.1,  # Default RL weight
        "lambda_cost": 0.01,  # Default cost
        "gamma": 0.99,  # Default discount factor
        "use_value_baseline": True,
        "value_weight": 0.5,

        # Training schedule (balanced)
        "rl_start_epoch": 2,  # Default start epoch
        "rl_update_frequency": 10,  # Default update frequency
    }
    defaults.update(kwargs)
    return RBSTrainingConfig(**defaults)


def rbs_advanced_config(**kwargs) -> RBSTrainingConfig:
    """Factory for advanced RBS configuration with maximum capabilities.

    This preset is ideal for:
    - Production models requiring maximum accuracy
    - Large-scale experiments with abundant resources
    - Research on belief state dynamics and non-monotonic reasoning

    Args:
        **kwargs: Additional configuration parameters to override defaults

    Returns:
        RBSTrainingConfig configured for advanced RBS training
    """
    # Set defaults, then merge with kwargs (kwargs take precedence)
    defaults: dict[str, Any] = {
        # GMM settings (advanced)
        "use_gmm_memory": True,
        "num_memory_experts": 6,
        "memory_num_tokens": 24,
        "routing_temperature": 0.8,  # Lower for sharper routing
        "entropy_regularization_weight": 0.001,  # Encourage diverse routing
        "load_balance_weight": 0.02,  # Stronger load balancing

        # RBS settings (advanced)
        "use_belief_state": True,
        "belief_state_threshold": 0.6,  # Lower threshold for more thorough processing
        "enable_re_scoring": True,
        "re_scoring_method": "learned",
        "confidence_calibration": True,
        "enable_trend_analysis": True,
        "max_segments": 64,
        "belief_state_memory_limit": 200,
        "halting_patience": 5,  # More patience for better convergence
        "enable_learnable_re_scoring": True,
        "belief_update_frequency": "every_segment",
        "non_monotonic_revision_threshold": 0.02,  # Lower threshold for sensitive revisions
        "enable_belief_state_visualization": True,  # Enable for research analysis

        # Halting policy settings (advanced)
        "use_halting_policy": True,
        "halting_policy_hidden_dim": 96,  # Medium-large network
        "halting_policy_layers": 3,  # Deeper network
        "halting_temperature": 0.9,  # Slightly lower temperature
        "halting_exploration_rate": 0.08,  # Balanced exploration

        # RL training settings (advanced)
        "rl_weight": 0.15,  # Higher RL weight
        "lambda_cost": 0.008,  # Moderate cost
        "gamma": 0.99,  # Standard discount factor
        "use_value_baseline": True,
        "value_weight": 0.6,

        # Training schedule (advanced)
        "rl_start_epoch": 2,  # Standard start
        "rl_update_frequency": 15,  # Moderate update frequency
    }
    defaults.update(kwargs)
    return RBSTrainingConfig(**defaults)


def rbs_research_config(**kwargs) -> RBSTrainingConfig:
    """Factory for research RBS configuration with comprehensive analysis.

    This preset is ideal for:
    - Academic research on belief state mechanisms
    - Ablation studies and component analysis
    - Understanding non-monotonic reasoning dynamics

    Args:
        **kwargs: Additional configuration parameters to override defaults

    Returns:
        RBSTrainingConfig configured for comprehensive RBS research
    """
    # Set defaults, then merge with kwargs (kwargs take precedence)
    defaults: dict[str, Any] = {
        # GMM settings (research)
        "use_gmm_memory": True,
        "num_memory_experts": 8,  # Maximum experts for study
        "memory_num_tokens": 32,
        "routing_temperature": 1.0,
        "entropy_regularization_weight": 0.005,
        "load_balance_weight": 0.015,

        # RBS settings (research - comprehensive)
        "use_belief_state": True,
        "belief_state_threshold": 0.5,  # Very low to observe full dynamics
        "enable_re_scoring": True,
        "re_scoring_method": "learned",
        "confidence_calibration": True,
        "enable_trend_analysis": True,
        "max_segments": 128,  # Large to study long-term dynamics
        "belief_state_memory_limit": 500,  # Large history for analysis
        "halting_patience": 10,  # High patience
        "enable_learnable_re_scoring": True,
        "belief_update_frequency": "every_segment",
        "non_monotonic_revision_threshold": 0.01,  # Very sensitive to revisions
        "enable_belief_state_visualization": True,

        # Halting policy settings (research)
        "use_halting_policy": True,
        "halting_policy_hidden_dim": 128,  # Large network for research
        "halting_policy_layers": 3,  # Deeper network
        "halting_temperature": 0.8,  # Lower temperature for more decisive actions
        "halting_exploration_rate": 0.05,  # Less exploration for stable learning

        # RL training settings (research)
        "rl_weight": 0.2,  # Higher RL weight for emphasis
        "lambda_cost": 0.005,  # Lower cost for more thorough processing
        "gamma": 0.995,  # Higher discount factor for long-term planning
        "use_value_baseline": True,
        "value_weight": 0.7,

        # Training schedule (research)
        "rl_start_epoch": 3,  # Start later for stable belief states
        "rl_update_frequency": 20,  # Update less frequently for stability
    }
    defaults.update(kwargs)
    return RBSTrainingConfig(**defaults)