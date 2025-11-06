"""
Belief State Tracker for RBS-QA

Implements dynamic belief state tracking with non-monotonic reasoning
for long-context question answering using GMM-XLNet.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SpanCandidate:
    """Represents a candidate answer span with associated metadata."""
    span: Tuple[int, int]                     # (start, end) global token indices
    confidence: float                         # Confidence score [0.0, 1.0]
    segment_id: int                           # Origin segment (0-based)
    gmm_context_hash: int                     # GMM state when span was found
    start_logits: float = 0.0                # Raw start logit for debugging
    end_logits: float = 0.0                  # Raw end logit for debugging
    re_scored: bool = False                  # Whether this span has been re-scored
    re_scored_confidence: Optional[float] = None  # Updated confidence after re-scoring

    def __post_init__(self):
        """Validate the span candidate after initialization."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")

        if self.span[0] > self.span[1]:
            raise ValueError(f"Invalid span: start ({self.span[0]}) > end ({self.span[1]})")

        if self.segment_id < 0:
            raise ValueError(f"Segment ID must be non-negative, got {self.segment_id}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'span': self.span,
            'confidence': self.confidence,
            'segment_id': self.segment_id,
            'gmm_context_hash': self.gmm_context_hash,
            'start_logits': self.start_logits,
            'end_logits': self.end_logits,
            're_scored': self.re_scored,
            're_scored_confidence': self.re_scored_confidence
        }


@dataclass
class BeliefState:
    """Tracks the current belief about the best answer span."""
    best_span: Optional[Tuple[int, int]] = None     # Global (start, end) token indices
    confidence: float = 0.0                         # Calibrated confidence score [0.0, 1.0]
    segment_id: int = -1                            # Segment where best span was found
    span_history: List[SpanCandidate] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    revision_count: int = 0                         # Number of non-monotonic revisions
    total_segments: int = 0                         # Total segments processed
    gmm_context_hashes: List[int] = field(default_factory=list)  # Track GMM state changes

    def __post_init__(self):
        """Validate the belief state after initialization."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")

        if self.segment_id < -1:
            raise ValueError(f"Segment ID must be >= -1, got {self.segment_id}")

    def add_candidate(self, candidate: SpanCandidate) -> None:
        """Add a new candidate to the history."""
        self.span_history.append(candidate)
        self.confidence_history.append(candidate.confidence)

    def update_best_span(self, span: Tuple[int, int], confidence: float, segment_id: int) -> None:
        """Update the best span and track if it's a revision."""
        # Check if this is a revision (different segment or significantly different span)
        is_revision = (
            self.best_span is not None and
            (segment_id != self.segment_id or
             abs(span[0] - self.best_span[0]) > 5 or  # Allow small variations
             abs(span[1] - self.best_span[1]) > 5)
        )

        if is_revision:
            self.revision_count += 1
            logger.debug(f"Belief revision #{self.revision_count}: {self.best_span} -> {span}")

        self.best_span = span
        self.confidence = confidence
        self.segment_id = segment_id

    def get_trend_analysis(self) -> dict:
        """Analyze confidence trend across processed segments."""
        if len(self.confidence_history) < 2:
            return {'trend': 'stable', 'slope': 0.0, 'variance': 0.0}

        # Simple linear regression for trend
        x = list(range(len(self.confidence_history)))
        y = self.confidence_history
        n = len(x)

        x_mean, y_mean = sum(x) / n, sum(y) / n
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator > 0 else 0.0
        variance = sum((y[i] - y_mean) ** 2 for i in range(n)) / n

        # Determine trend
        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'

        return {
            'trend': trend,
            'slope': slope,
            'variance': variance,
            'mean_confidence': y_mean,
            'final_confidence': y[-1] if y else 0.0
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'best_span': self.best_span,
            'confidence': self.confidence,
            'segment_id': self.segment_id,
            'span_history': [candidate.to_dict() for candidate in self.span_history],
            'confidence_history': self.confidence_history,
            'revision_count': self.revision_count,
            'total_segments': self.total_segments,
            'gmm_context_hashes': self.gmm_context_hashes,
            'trend_analysis': self.get_trend_analysis()
        }


class BeliefStateTracker(nn.Module):
    """
    Core belief state tracker with non-monotonic reasoning capabilities.

    Tracks the best answer span found across all processed segments and
    enables belief revision when new context becomes available.
    """

    def __init__(self,
                 max_segments: int = 32,
                 confidence_threshold: float = 0.7,
                 re_scoring_method: str = "context_weighted",
                 enable_trend_analysis: bool = True,
                 hidden_dim: int = 768,
                 enable_learnable_re_scoring: bool = False):
        """
        Initialize the belief state tracker.

        Args:
            max_segments: Maximum number of segments to process
            confidence_threshold: Minimum confidence to consider halting
            re_scoring_method: Method for re-scoring past spans
            enable_trend_analysis: Enable confidence trend analysis
            hidden_dim: Hidden dimension for learnable components
            enable_learnable_re_scoring: Enable learnable re-scoring network
        """
        super().__init__()

        # Validate parameters
        if max_segments <= 0:
            raise ValueError(f"max_segments must be positive, got {max_segments}")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}")
        if re_scoring_method not in ["context_weighted", "learned", "exponential_decay"]:
            raise ValueError(f"Unknown re_scoring_method: {re_scoring_method}")

        # Configuration
        self.max_segments = max_segments
        self.confidence_threshold = confidence_threshold
        self.re_scoring_method = re_scoring_method
        self.enable_trend_analysis = enable_trend_analysis
        self.hidden_dim = hidden_dim

        # Learnable parameters for confidence calibration
        self.confidence_scaler = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

        # Optional: Learnable re-scoring network
        if enable_learnable_re_scoring and re_scoring_method == "learned":
            self.re_score_network = nn.Sequential(
                nn.Linear(hidden_dim + 4, 256),  # GMM context + span features
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            self.re_score_network = None

        # Performance tracking
        self.re_scoring_cache = {}  # Cache for re-scoring computations

        # Initialize belief state
        self.reset_belief()

    def reset_belief(self) -> None:
        """Reset belief state for new document."""
        self.belief = BeliefState(
            best_span=None,
            confidence=0.0,
            segment_id=-1,
            span_history=[],
            confidence_history=[],
            revision_count=0,
            total_segments=0,
            gmm_context_hashes=[]
        )
        self.re_scoring_cache.clear()
        logger.debug("Belief state reset for new document")

    def extract_best_span(self,
                         logits: Tuple[torch.Tensor, torch.Tensor],
                         segment_id: int,
                         global_offset: int,
                         max_span_length: int = 30) -> SpanCandidate:
        """
        Extract best span from current segment logits.

        Args:
            logits: (start_logits, end_logits) for current segment
            segment_id: Index of current segment (0-based)
            global_offset: Token offset for global index mapping
            max_span_length: Maximum allowed span length

        Returns:
            SpanCandidate with best span and confidence
        """
        start_logits, end_logits = logits

        # Ensure tensors are on CPU and convert to numpy for efficient processing
        if isinstance(start_logits, torch.Tensor):
            start_logits = start_logits.detach().cpu()
            end_logits = end_logits.detach().cpu()

        # Get best start and end indices
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()

        # Ensure valid span constraints
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        if end_idx - start_idx > max_span_length:
            end_idx = start_idx + max_span_length

        # Convert to global indices
        global_start = start_idx + global_offset
        global_end = end_idx + global_offset

        # Compute confidence
        confidence = self.compute_confidence(
            start_logits, end_logits, start_idx, end_idx
        )

        # Create GMM context hash (simplified - in practice would use actual GMM state)
        gmm_context_hash = hash(f"segment_{segment_id}_gmm_{torch.rand(1).item():.6f}")

        candidate = SpanCandidate(
            span=(global_start, global_end),
            confidence=confidence,
            segment_id=segment_id,
            gmm_context_hash=gmm_context_hash,
            start_logits=start_logits[start_idx].item(),
            end_logits=end_logits[end_idx].item()
        )

        return candidate

    def compute_confidence(self,
                          start_logits: torch.Tensor,
                          end_logits: torch.Tensor,
                          start_idx: int,
                          end_idx: int) -> float:
        """
        Compute calibrated confidence for a span.

        Args:
            start_logits: Start position logits
            end_logits: End position logits
            start_idx: Start position index
            end_idx: End position index

        Returns:
            Calibrated confidence score [0.0, 1.0]
        """
        if isinstance(start_logits, torch.Tensor):
            start_logits = start_logits.detach().cpu()
            end_logits = end_logits.detach().cpu()

        # Get raw logits for the span
        if isinstance(start_logits, torch.Tensor):
            start_logit = start_logits[start_idx].item()
            end_logit = end_logits[end_idx].item()
        else:
            # Assume list/array input
            start_logit = float(start_logits[start_idx])
            end_logit = float(end_logits[end_idx])

        # Compute probability using softmax
        if isinstance(start_logits, torch.Tensor):
            start_probs = F.softmax(start_logits, dim=0)
            end_probs = F.softmax(end_logits, dim=0)
            raw_confidence = (start_probs[start_idx] * end_probs[end_idx]).item()
        else:
            # For list inputs, convert to tensor for softmax computation
            start_tensor = torch.tensor(start_logits, dtype=torch.float32)
            end_tensor = torch.tensor(end_logits, dtype=torch.float32)
            start_probs = F.softmax(start_tensor, dim=0)
            end_probs = F.softmax(end_tensor, dim=0)
            raw_confidence = (start_probs[start_idx] * end_probs[end_idx]).item()

        # Apply learnable calibration
        calibrated_confidence = torch.sigmoid(
            torch.tensor(raw_confidence) * self.confidence_scaler + self.confidence_bias
        ).item()

        return float(calibrated_confidence)

    def re_score_past_spans(self,
                           past_spans: List[SpanCandidate],
                           gmm_context: torch.Tensor) -> List[SpanCandidate]:
        """
        Re-score previously found spans using new GMM context.
        This enables non-monotonic reasoning.

        Args:
            past_spans: List of previously found span candidates
            gmm_context: Aggregated GMM memory context

        Returns:
            List of re-scored span candidates
        """
        re_scored_spans = []

        for candidate in past_spans:
            # Skip if already re-scored with this context
            cache_key = (candidate.span, hash(gmm_context.data_ptr() if isinstance(gmm_context, torch.Tensor) else id(gmm_context)))
            if cache_key in self.re_scoring_cache:
                re_scored_confidence = self.re_scoring_cache[cache_key]
            else:
                re_scored_confidence = self._compute_re_scored_confidence(candidate, gmm_context)
                self.re_scoring_cache[cache_key] = re_scored_confidence

            # Create updated candidate
            updated_candidate = SpanCandidate(
                span=candidate.span,
                confidence=re_scored_confidence,  # Use re-scored confidence
                segment_id=candidate.segment_id,
                gmm_context_hash=candidate.gmm_context_hash,
                start_logits=candidate.start_logits,
                end_logits=candidate.end_logits,
                re_scored=True,
                re_scored_confidence=re_scored_confidence
            )
            re_scored_spans.append(updated_candidate)

        return re_scored_spans

    def _compute_re_scored_confidence(self,
                                     candidate: SpanCandidate,
                                     gmm_context: torch.Tensor) -> float:
        """
        Compute re-scored confidence using the specified method.

        Args:
            candidate: Original span candidate
            gmm_context: Current GMM context

        Returns:
            Re-scored confidence value
        """
        if self.re_scoring_method == "context_weighted":
            # Simple context-weighted adjustment
            context_factor = torch.mean(torch.abs(gmm_context)).item()
            adjusted_confidence = candidate.confidence * (1.0 + 0.1 * context_factor)
            return min(1.0, max(0.0, adjusted_confidence))

        elif self.re_scoring_method == "exponential_decay":
            # Time-based decay
            age = len(self.belief.span_history) - candidate.segment_id
            decay_factor = torch.exp(torch.tensor(-0.1 * age)).item()
            return candidate.confidence * decay_factor

        elif self.re_scoring_method == "learned" and self.re_score_network is not None:
            # Learnable re-scoring (placeholder implementation)
            if isinstance(gmm_context, torch.Tensor):
                # Aggregate context
                context_features = torch.mean(gmm_context, dim=(0, 1))  # [hidden_dim]

                # Span features
                span_features = torch.tensor([
                    float(candidate.span[0]),
                    float(candidate.span[1]),
                    candidate.confidence,
                    float(candidate.segment_id)
                ])

                # Combine features
                combined_features = torch.cat([context_features, span_features])

                # Predict new confidence
                with torch.no_grad():
                    new_confidence = self.re_score_network(combined_features).item()

                return new_confidence

        # Default: return original confidence
        return candidate.confidence

    def should_halt(self) -> bool:
        """
        Determine if current belief warrants halting.

        Returns:
            True if halting criteria are met
        """
        # Primary criterion: confidence threshold
        if self.belief.confidence >= self.confidence_threshold:
            return True

        # Secondary criterion: stability after many revisions
        if (self.belief.total_segments >= self.max_segments and
            self.belief.revision_count == 0):
            return True

        # Tertiary criterion: high confidence with stable trend
        if (self.enable_trend_analysis and
            len(self.belief.confidence_history) >= 3):  # Need sufficient data
            trend = self.belief.get_trend_analysis()
            if (self.belief.confidence >= self.confidence_threshold * 0.8 and
                trend['trend'] == 'stable' and
                trend['variance'] < 0.01):
                return True

        return False

    def update_belief(self,
                     current_logits: Tuple[torch.Tensor, torch.Tensor],
                     current_segment_id: int,
                     gmm_context: torch.Tensor,
                     global_offset: int) -> BeliefState:
        """
        Update belief state with new segment information.

        Args:
            current_logits: (start_logits, end_logits) for current segment
            current_segment_id: Index of current segment (0-based)
            gmm_context: Aggregated GMM memory context
            global_offset: Token offset for global index mapping

        Returns:
            Updated BeliefState with potential non-monotonic revisions
        """
        # Extract best span from current segment
        current_best = self.extract_best_span(
            current_logits, current_segment_id, global_offset
        )

        logger.debug(f"Current best span: {current_best.span}, confidence: {current_best.confidence:.3f}")

        # Update segment tracking
        self.belief.total_segments = current_segment_id + 1

        # Track GMM context changes
        gmm_hash = hash(f"segment_{current_segment_id}_gmm_{torch.mean(gmm_context).item():.6f}")
        self.belief.gmm_context_hashes.append(gmm_hash)

        # If we have a previous belief, re-score it with new context
        if self.belief.best_span is not None:
            # Create list of past spans for re-scoring
            past_candidates = [candidate for candidate in self.belief.span_history
                             if not candidate.re_scored]

            if past_candidates:
                re_scored_spans = self.re_score_past_spans(past_candidates, gmm_context)

                # Find the best re-scored span
                best_re_scored = max(re_scored_spans, key=lambda x: x.confidence)

                logger.debug(f"Best re-scored span: {best_re_scored.span}, confidence: {best_re_scored.confidence:.3f}")

                # Non-monotonic decision: new span vs revised old span
                if best_re_scored.confidence > current_best.confidence:
                    # Old belief confirmed/boosted by new context
                    self.belief.update_best_span(
                        best_re_scored.span,
                        best_re_scored.confidence,
                        best_re_scored.segment_id
                    )
                    logger.debug("Old belief confirmed/boosted by new context")
                else:
                    # New span beats revised old span (belief revision)
                    self.belief.update_best_span(
                        current_best.span,
                        current_best.confidence,
                        current_best.segment_id
                    )
                    logger.debug("Belief revision: new span beats revised old span")
            else:
                # No past spans to re-score, use current best
                self.belief.update_best_span(
                    current_best.span,
                    current_best.confidence,
                    current_best.segment_id
                )
        else:
            # First segment: initialize belief
            self.belief.update_best_span(
                current_best.span,
                current_best.confidence,
                current_best.segment_id
            )
            logger.debug("First segment: belief initialized")

        # Add current candidate to history
        self.belief.add_candidate(current_best)

        # Log status
        if self.enable_trend_analysis:
            trend = self.belief.get_trend_analysis()
            logger.debug(f"Belief state: confidence={self.belief.confidence:.3f}, "
                        f"revisions={self.belief.revision_count}, "
                        f"trend={trend['trend']}")

        return self.belief