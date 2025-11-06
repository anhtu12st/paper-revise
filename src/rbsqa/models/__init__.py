"""RBS Model outputs and data structures."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from ..belief_state import BeliefState
from ..halting_policy import HaltingDecision, HaltingStateFeatures
from .rbs_xlnet import RBSXLNetForQA


@dataclass
class RBSModelOutput:
    """Structured output for RBS model."""
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    memory_state: Dict[str, torch.Tensor]
    aggregated_memory: torch.Tensor
    routing_info: Dict[str, torch.Tensor]
    belief_state: Optional[BeliefState] = None
    halting_decision: Optional[HaltingDecision] = None
    segment_info: Optional[Dict] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None

    @property
    def mems(self) -> List[torch.Tensor]:
        """
        Convert memory_state to the format expected by XLNet base trainer.

        This provides backward compatibility by transforming GMM expert memories
        into the sequential memory format expected by XLNet attention mechanism.

        GMM format: {"expert_0": (batch, slots, hidden), "expert_1": (batch, slots, hidden), ...}
        XLNet format: [(memory_length, batch, hidden)] where memory_length = num_experts * slots

        Returns:
            List containing a single memory tensor with shape (memory_length, batch_size, hidden_dim)
        """
        if not self.memory_state:
            return []

        # Extract expert memories in order and stack them
        expert_tensors = []
        num_experts = len(self.memory_state)

        for i in range(num_experts):
            expert_key = f"expert_{i}"
            if expert_key in self.memory_state:
                expert_memory = self.memory_state[expert_key]
                # Validate expert memory shape
                if expert_memory.dim() != 3:
                    raise ValueError(f"Expert memory {expert_key} should be 3D tensor, got {expert_memory.dim()}D")
                expert_tensors.append(expert_memory)
            else:
                # If expert is missing, create zero tensor with same shape as others
                if expert_tensors:  # Use shape from first available expert
                    zero_expert = torch.zeros_like(expert_tensors[0])
                    expert_tensors.append(zero_expert)

        if not expert_tensors:
            return []

        # Validate that all expert memories have the same shape
        first_shape = expert_tensors[0].shape
        for i, tensor in enumerate(expert_tensors[1:], 1):
            if tensor.shape != first_shape:
                raise ValueError(f"Expert memory shapes must match. expert_0: {first_shape}, expert_{i}: {tensor.shape}")

        # Stack experts along first dimension: (num_experts, batch, slots, hidden)
        stacked_experts = torch.stack(expert_tensors, dim=0)

        # Reshape to aggregate all memory slots: (batch, num_experts * slots, hidden)
        batch_size, slots_per_expert, hidden_dim = stacked_experts.shape[1], stacked_experts.shape[2], stacked_experts.shape[3]
        aggregated_memory = stacked_experts.permute(1, 0, 2, 3).contiguous()  # (batch, num_experts, slots, hidden)
        aggregated_memory = aggregated_memory.view(batch_size, -1, hidden_dim)  # (batch, total_slots, hidden)

        # Transpose for XLNet format: (total_slots, batch, hidden)
        xlnet_memory = aggregated_memory.transpose(0, 1).contiguous()

        # Final validation: ensure we have the expected memory length (typically 16)
        expected_memory_length = num_experts * slots_per_expert
        if xlnet_memory.shape[0] != expected_memory_length:
            raise ValueError(f"Memory length mismatch. Expected {expected_memory_length}, got {xlnet_memory.shape[0]}")

        # XLNet expects memory tensors for each transformer layer (12 layers for base model)
        # Duplicate the aggregated memory for all layers to maintain consistency
        num_layers = 12  # xlnet-base-cased has 12 layers
        return [xlnet_memory.clone() for _ in range(num_layers)]

    def to_tuple(self) -> Tuple:
        """Convert to tuple for compatibility with existing code."""
        return (
            self.start_logits,
            self.end_logits,
            self.memory_state,
            self.aggregated_memory,
            self.routing_info,
            self.belief_state,
            self.halting_decision,
            self.segment_info,
            self.hidden_states,
            self.attentions
        )


@dataclass
class RBSInferenceResult:
    """Result of adaptive inference."""
    answer_span: Tuple[int, int]
    confidence: float
    segments_processed: int
    total_segments: int
    belief_history: List[BeliefState]
    halting_history: List[HaltingDecision]
    memory_state: Dict[str, torch.Tensor]
    efficiency_score: float  # Higher is better (more efficient)

    def compute_savings(self) -> float:
        """Compute percentage of segments saved by early stopping."""
        if self.total_segments == 0:
            return 0.0
        return (self.total_segments - self.segments_processed) / self.total_segments * 100.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization."""
        return {
            'answer_span': self.answer_span,
            'confidence': self.confidence,
            'segments_processed': self.segments_processed,
            'total_segments': self.total_segments,
            'savings_percent': self.compute_savings(),
            'efficiency_score': self.efficiency_score,
            'num_revisions': len([b for b in self.belief_history if b.revision_count > 0]) if self.belief_history else 0,
            'final_confidence_trend': self.belief_history[-1].get_trend_analysis() if self.belief_history else None
        }


@dataclass
class RBSModelConfig:
    """Configuration for RBS model."""
    base_model_name: str = "xlnet-base-cased"
    memory_num_tokens: int = 16
    num_memory_experts: int = 4
    use_rbs_mode: bool = True
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    # Belief state configuration
    belief_max_segments: int = 32
    belief_confidence_threshold: float = 0.7
    belief_re_scoring_method: str = "context_weighted"
    belief_enable_trend_analysis: bool = True

    # Halting policy configuration
    halting_hidden_dim: int = 64
    halting_num_layers: int = 2
    halting_temperature: float = 1.0
    halting_exploration_rate: float = 0.1

    # Adaptive inference configuration
    adaptive_max_segments: Optional[int] = None
    adaptive_confidence_threshold: float = 0.8
    adaptive_min_segments: int = 1

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.memory_num_tokens <= 0:
            raise ValueError(f"memory_num_tokens must be positive, got {self.memory_num_tokens}")

        if self.num_memory_experts <= 0:
            raise ValueError(f"num_memory_experts must be positive, got {self.num_memory_experts}")

        if not (0.0 <= self.belief_confidence_threshold <= 1.0):
            raise ValueError(f"belief_confidence_threshold must be in [0.0, 1.0], got {self.belief_confidence_threshold}")

        if self.belief_re_scoring_method not in ["context_weighted", "learned", "exponential_decay"]:
            raise ValueError(f"Unknown belief_re_scoring_method: {self.belief_re_scoring_method}")

        if not (0.0 <= self.adaptive_confidence_threshold <= 1.0):
            raise ValueError(f"adaptive_confidence_threshold must be in [0.0, 1.0], got {self.adaptive_confidence_threshold}")

        if self.adaptive_min_segments < 1:
            raise ValueError(f"adaptive_min_segments must be >= 1, got {self.adaptive_min_segments}")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'base_model_name': self.base_model_name,
            'memory_num_tokens': self.memory_num_tokens,
            'num_memory_experts': self.num_memory_experts,
            'use_rbs_mode': self.use_rbs_mode,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
            'belief_max_segments': self.belief_max_segments,
            'belief_confidence_threshold': self.belief_confidence_threshold,
            'belief_re_scoring_method': self.belief_re_scoring_method,
            'belief_enable_trend_analysis': self.belief_enable_trend_analysis,
            'halting_hidden_dim': self.halting_hidden_dim,
            'halting_num_layers': self.halting_num_layers,
            'halting_temperature': self.halting_temperature,
            'halting_exploration_rate': self.halting_exploration_rate,
            'adaptive_max_segments': self.adaptive_max_segments,
            'adaptive_confidence_threshold': self.adaptive_confidence_threshold,
            'adaptive_min_segments': self.adaptive_min_segments
        }