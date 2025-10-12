"""
Multi-hop Reasoning Analysis Utilities
=======================================

This module provides tools for analyzing multi-hop reasoning patterns in
memory-augmented models, particularly for questions that require information
from multiple document segments.

Classes:
    HopTracker: Track reasoning hops across document segments
    BridgeEntity: Represents entities that connect multiple segments
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class BridgeEntity:
    """
    Represents an entity that appears in multiple document segments,
    potentially serving as a bridge for multi-hop reasoning.

    Attributes:
        text: The entity text
        segments: List of segment indices where entity appears
        attention_scores: Attention scores for this entity in each segment
        is_answer: Whether this entity is part of the answer
    """

    text: str
    segments: list[int]
    attention_scores: dict[int, float]
    is_answer: bool = False

    @property
    def hop_count(self) -> int:
        """Number of segments this entity bridges."""
        return len(self.segments)

    @property
    def avg_attention(self) -> float:
        """Average attention score across all segments."""
        if not self.attention_scores:
            return 0.0
        return float(np.mean(list(self.attention_scores.values())))


@dataclass
class ReasoningHop:
    """
    Represents a single reasoning hop from one segment to another.

    Attributes:
        from_segment: Source segment index
        to_segment: Target segment index
        bridging_entity: Entity connecting the segments
        attention_flow: Attention weight indicating connection strength
        confidence: Confidence score for this hop (0-1)
    """

    from_segment: int
    to_segment: int
    bridging_entity: str | None = None
    attention_flow: float = 0.0
    confidence: float = 0.0


class HopTracker:
    """
    Track and analyze multi-hop reasoning patterns across document segments.

    This class helps identify when a model performs multi-hop reasoning by:
    1. Tracking entities across segments
    2. Detecting bridge entities (entities in multiple segments)
    3. Analyzing attention patterns to infer reasoning hops
    4. Reconstructing the reasoning chain from question to answer

    Example:
        >>> tracker = HopTracker()
        >>> # Process each segment
        >>> for seg_idx, segment_data in enumerate(document_segments):
        >>>     entities = extract_entities(segment_data["text"])
        >>>     attention = model_outputs["attention_weights"]
        >>>     tracker.track_segment(seg_idx, attention, entities)
        >>>
        >>> # Analyze results
        >>> bridge_entities = tracker.detect_bridge_entities()
        >>> hop_sequence = tracker.get_hop_sequence()
        >>> tracker.export_analysis("output.json")
    """

    def __init__(self, min_attention_threshold: float = 0.1):
        """
        Initialize hop tracker.

        Args:
            min_attention_threshold: Minimum attention score to consider
                                    for hop detection (default: 0.1)
        """
        self.min_attention_threshold = min_attention_threshold

        # Track entities and their occurrences
        self.entity_occurrences: dict[str, list[int]] = defaultdict(list)
        self.segment_entities: dict[int, set[str]] = defaultdict(set)

        # Track attention patterns
        self.segment_attention: dict[int, np.ndarray] = {}
        self.entity_attention: dict[str, dict[int, float]] = defaultdict(dict)

        # Track reasoning hops
        self.detected_hops: list[ReasoningHop] = []

        # Track answer information
        self.answer_entities: set[str] = set()
        self.answer_segment: int | None = None

    def track_segment(
        self,
        segment_idx: int,
        attention_weights: np.ndarray,
        entities: list[str],
        segment_text: str | None = None,
    ):
        """
        Track entities and attention patterns for a document segment.

        Args:
            segment_idx: Index of the segment (0-based)
            attention_weights: Attention weights for this segment
                              Shape: (num_heads, seq_len) or (seq_len,)
            entities: List of entities detected in this segment
            segment_text: Optional full text of the segment
        """
        # Store attention weights
        if attention_weights.ndim == 2:
            # Multi-head: average across heads
            self.segment_attention[segment_idx] = attention_weights.mean(axis=0)
        else:
            # Single head or already averaged
            self.segment_attention[segment_idx] = attention_weights

        # Track entities in this segment
        for entity in entities:
            entity_lower = entity.lower()
            self.entity_occurrences[entity_lower].append(segment_idx)
            self.segment_entities[segment_idx].add(entity_lower)

            # Compute average attention for this entity
            # (simplified - in practice would need token positions)
            avg_attention = float(np.mean(self.segment_attention[segment_idx]))
            self.entity_attention[entity_lower][segment_idx] = avg_attention

    def mark_answer(self, answer_text: str, answer_segment: int):
        """
        Mark the answer text and segment for analysis.

        Args:
            answer_text: The answer text
            answer_segment: Segment index where answer was found
        """
        self.answer_entities = set(answer_text.lower().split())
        self.answer_segment = answer_segment

    def detect_bridge_entities(
        self,
        min_segments: int = 2,
        min_attention: float | None = None,
    ) -> list[BridgeEntity]:
        """
        Detect entities that appear in multiple segments (potential bridges).

        Args:
            min_segments: Minimum number of segments for an entity to be
                         considered a bridge (default: 2)
            min_attention: Minimum average attention score (default: uses
                          min_attention_threshold)

        Returns:
            List of BridgeEntity objects, sorted by hop count (descending)
        """
        if min_attention is None:
            min_attention = self.min_attention_threshold

        bridge_entities = []

        for entity_text, segment_list in self.entity_occurrences.items():
            if len(segment_list) >= min_segments:
                attention_scores = self.entity_attention[entity_text]
                avg_attention = np.mean(list(attention_scores.values()))

                if avg_attention >= min_attention:
                    is_answer = any(word in entity_text for word in self.answer_entities)

                    bridge_entity = BridgeEntity(
                        text=entity_text,
                        segments=sorted(segment_list),
                        attention_scores=attention_scores,
                        is_answer=is_answer,
                    )
                    bridge_entities.append(bridge_entity)

        # Sort by hop count (most segments first)
        bridge_entities.sort(key=lambda e: e.hop_count, reverse=True)
        return bridge_entities

    def detect_hops(
        self,
        attention_threshold: float | None = None,
    ) -> list[ReasoningHop]:
        """
        Detect reasoning hops based on attention patterns and bridge entities.

        Args:
            attention_threshold: Minimum attention for hop detection
                                (default: uses min_attention_threshold)

        Returns:
            List of ReasoningHop objects representing detected hops
        """
        if attention_threshold is None:
            attention_threshold = self.min_attention_threshold

        hops = []
        bridge_entities = self.detect_bridge_entities()

        # For each bridge entity, create hops between segments
        for bridge in bridge_entities:
            segments = bridge.segments
            for i in range(len(segments) - 1):
                from_seg = segments[i]
                to_seg = segments[i + 1]

                # Compute attention flow (simplified)
                attention_flow = (
                    bridge.attention_scores.get(from_seg, 0.0) + bridge.attention_scores.get(to_seg, 0.0)
                ) / 2.0

                if attention_flow >= attention_threshold:
                    hop = ReasoningHop(
                        from_segment=from_seg,
                        to_segment=to_seg,
                        bridging_entity=bridge.text,
                        attention_flow=attention_flow,
                        confidence=min(1.0, attention_flow * 2.0),
                    )
                    hops.append(hop)

        self.detected_hops = hops
        return hops

    def get_hop_sequence(
        self,
        to_answer: bool = True,
    ) -> list[ReasoningHop]:
        """
        Get the sequence of reasoning hops, optionally filtered to answer.

        Args:
            to_answer: If True, only return hops leading to answer segment

        Returns:
            List of ReasoningHop objects in sequence order
        """
        if not self.detected_hops:
            self.detect_hops()

        if not to_answer or self.answer_segment is None:
            # Return all hops, sorted by from_segment
            return sorted(self.detected_hops, key=lambda h: h.from_segment)

        # Filter hops that lead to answer segment
        relevant_hops = []
        current_segments = {0}  # Start from first segment

        while self.answer_segment not in current_segments:
            # Find hops from current segments
            next_hops = [
                hop
                for hop in self.detected_hops
                if hop.from_segment in current_segments and hop.to_segment not in current_segments
            ]

            if not next_hops:
                break

            # Take hop with highest confidence
            best_hop = max(next_hops, key=lambda h: h.confidence)
            relevant_hops.append(best_hop)
            current_segments.add(best_hop.to_segment)

        return relevant_hops

    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics about multi-hop reasoning.

        Returns:
            Dictionary containing:
            - num_segments: Total segments tracked
            - num_entities: Total unique entities
            - num_bridge_entities: Entities appearing in 2+ segments
            - num_hops: Total reasoning hops detected
            - avg_hop_length: Average hop distance
            - max_hop_chain: Longest chain of hops
        """
        bridge_entities = self.detect_bridge_entities()
        hops = self.detected_hops if self.detected_hops else self.detect_hops()

        # Compute hop chain length (simplified)
        avg_hop_length: float
        if hops:
            hop_distances = [abs(hop.to_segment - hop.from_segment) for hop in hops]
            avg_hop_length = float(np.mean(hop_distances))
            max_hop_chain = len(self.get_hop_sequence(to_answer=False))
        else:
            avg_hop_length = 0.0
            max_hop_chain = 0

        return {
            "num_segments": len(self.segment_attention),
            "num_entities": len(self.entity_occurrences),
            "num_bridge_entities": len(bridge_entities),
            "num_hops": len(hops),
            "avg_hop_length": float(avg_hop_length),
            "max_hop_chain": max_hop_chain,
            "has_answer": self.answer_segment is not None,
            "answer_segment": self.answer_segment,
        }

    def export_analysis(self, output_path: str):
        """
        Export complete analysis to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        import json

        bridge_entities = self.detect_bridge_entities()
        hops = self.detected_hops if self.detected_hops else self.detect_hops()
        hop_sequence = self.get_hop_sequence(to_answer=True)

        analysis = {
            "statistics": self.get_statistics(),
            "bridge_entities": [
                {
                    "text": entity.text,
                    "segments": entity.segments,
                    "hop_count": entity.hop_count,
                    "avg_attention": float(entity.avg_attention),
                    "is_answer": entity.is_answer,
                }
                for entity in bridge_entities
            ],
            "all_hops": [
                {
                    "from_segment": hop.from_segment,
                    "to_segment": hop.to_segment,
                    "bridging_entity": hop.bridging_entity,
                    "attention_flow": float(hop.attention_flow),
                    "confidence": float(hop.confidence),
                }
                for hop in hops
            ],
            "hop_sequence_to_answer": [
                {
                    "from_segment": hop.from_segment,
                    "to_segment": hop.to_segment,
                    "bridging_entity": hop.bridging_entity,
                    "confidence": float(hop.confidence),
                }
                for hop in hop_sequence
            ],
        }

        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

    def reset(self):
        """Reset tracker to initial state."""
        self.entity_occurrences.clear()
        self.segment_entities.clear()
        self.segment_attention.clear()
        self.entity_attention.clear()
        self.detected_hops.clear()
        self.answer_entities.clear()
        self.answer_segment = None


def extract_simple_entities(text: str, min_length: int = 3) -> list[str]:
    """
    Simple entity extraction based on capitalization and word length.

    This is a basic implementation. For production, use spaCy or similar.

    Args:
        text: Input text
        min_length: Minimum word length to consider (default: 3)

    Returns:
        List of potential entity strings
    """
    import re

    # Split into words
    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

    # Filter by length
    entities = [w for w in words if len(w) >= min_length]

    return entities
