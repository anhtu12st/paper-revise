"""
Unit tests for multi-hop reasoning analysis utilities.

Tests the HopTracker class and related functionality for analyzing
multi-hop reasoning patterns in memory-augmented models.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from memxlnet.utils.multihop_utils import (
    BridgeEntity,
    HopTracker,
    ReasoningHop,
    extract_simple_entities,
)


class TestBridgeEntity:
    """Test BridgeEntity dataclass."""

    def test_bridge_entity_creation(self):
        """Test creating a bridge entity."""
        entity = BridgeEntity(
            text="Paris", segments=[0, 2, 4], attention_scores={0: 0.8, 2: 0.6, 4: 0.9}, is_answer=True
        )

        assert entity.text == "Paris"
        assert entity.segments == [0, 2, 4]
        assert entity.hop_count == 3
        assert entity.is_answer is True

    def test_hop_count(self):
        """Test hop count calculation."""
        entity = BridgeEntity(text="test", segments=[0, 1], attention_scores={0: 0.5, 1: 0.5})
        assert entity.hop_count == 2

    def test_avg_attention(self):
        """Test average attention calculation."""
        entity = BridgeEntity(text="test", segments=[0, 1, 2], attention_scores={0: 0.3, 1: 0.6, 2: 0.9})
        expected = (0.3 + 0.6 + 0.9) / 3
        assert abs(entity.avg_attention - expected) < 1e-6


class TestReasoningHop:
    """Test ReasoningHop dataclass."""

    def test_reasoning_hop_creation(self):
        """Test creating a reasoning hop."""
        hop = ReasoningHop(from_segment=0, to_segment=2, bridging_entity="Paris", attention_flow=0.75, confidence=0.9)

        assert hop.from_segment == 0
        assert hop.to_segment == 2
        assert hop.bridging_entity == "Paris"
        assert hop.attention_flow == 0.75
        assert hop.confidence == 0.9


class TestHopTracker:
    """Test HopTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a hop tracker instance."""
        return HopTracker(min_attention_threshold=0.1)

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.min_attention_threshold == 0.1
        assert len(tracker.entity_occurrences) == 0
        assert len(tracker.segment_entities) == 0
        assert len(tracker.segment_attention) == 0

    def test_track_segment_single_head(self, tracker):
        """Test tracking a segment with single-head attention."""
        attention = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        entities = ["Paris", "France", "Europe"]

        tracker.track_segment(0, attention, entities)

        assert 0 in tracker.segment_attention
        assert "paris" in tracker.entity_occurrences
        assert "france" in tracker.entity_occurrences
        assert "europe" in tracker.entity_occurrences

    def test_track_segment_multi_head(self, tracker):
        """Test tracking a segment with multi-head attention."""
        # Shape: (num_heads=2, seq_len=5)
        attention = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.5, 0.4, 0.3, 0.2, 0.1],
            ]
        )
        entities = ["Paris", "France"]

        tracker.track_segment(0, attention, entities)

        # Should average across heads
        expected_avg = attention.mean(axis=0)
        np.testing.assert_array_almost_equal(tracker.segment_attention[0], expected_avg)

    def test_track_multiple_segments(self, tracker):
        """Test tracking multiple segments."""
        # Segment 0
        tracker.track_segment(0, np.array([0.5, 0.3, 0.2]), ["Paris", "France"])

        # Segment 1
        tracker.track_segment(1, np.array([0.4, 0.4, 0.2]), ["Paris", "Europe"])

        # Segment 2
        tracker.track_segment(2, np.array([0.6, 0.3, 0.1]), ["France", "Germany"])

        # Check entity occurrences
        assert tracker.entity_occurrences["paris"] == [0, 1]
        assert tracker.entity_occurrences["france"] == [0, 2]
        assert tracker.entity_occurrences["europe"] == [1]

    def test_mark_answer(self, tracker):
        """Test marking answer text and segment."""
        tracker.mark_answer("Paris is the capital", 2)

        assert tracker.answer_segment == 2
        assert "paris" in tracker.answer_entities
        assert "capital" in tracker.answer_entities

    def test_detect_bridge_entities_basic(self, tracker):
        """Test basic bridge entity detection."""
        # Track segments with overlapping entities
        tracker.track_segment(0, np.array([0.5]), ["Paris", "France"])
        tracker.track_segment(1, np.array([0.6]), ["Paris", "Europe"])
        tracker.track_segment(2, np.array([0.4]), ["France", "Germany"])

        bridge_entities = tracker.detect_bridge_entities(min_segments=2)

        # Paris appears in segments 0, 1 -> bridge
        # France appears in segments 0, 2 -> bridge
        assert len(bridge_entities) == 2

        entity_texts = [e.text for e in bridge_entities]
        assert "paris" in entity_texts
        assert "france" in entity_texts

    def test_detect_bridge_entities_with_threshold(self, tracker):
        """Test bridge detection with attention threshold."""
        # Track with different attention levels
        tracker.track_segment(0, np.array([0.9]), ["HighAttention"])
        tracker.track_segment(1, np.array([0.8]), ["HighAttention"])
        tracker.track_segment(0, np.array([0.01]), ["LowAttention"])
        tracker.track_segment(1, np.array([0.01]), ["LowAttention"])

        # With default threshold (0.1), only HighAttention qualifies
        bridge_entities = tracker.detect_bridge_entities(min_attention=0.1)

        entity_texts = [e.text for e in bridge_entities]
        assert "highattention" in entity_texts
        assert "lowattention" not in entity_texts

    def test_detect_bridge_entities_sorted_by_hop_count(self, tracker):
        """Test that bridge entities are sorted by hop count."""
        # Entity in 3 segments
        tracker.track_segment(0, np.array([0.5]), ["ManyHops"])
        tracker.track_segment(1, np.array([0.5]), ["ManyHops"])
        tracker.track_segment(2, np.array([0.5]), ["ManyHops"])

        # Entity in 2 segments
        tracker.track_segment(0, np.array([0.5]), ["FewHops"])
        tracker.track_segment(1, np.array([0.5]), ["FewHops"])

        bridge_entities = tracker.detect_bridge_entities()

        # Should be sorted by hop count descending
        assert bridge_entities[0].text == "manyhops"
        assert bridge_entities[0].hop_count == 3
        assert bridge_entities[1].text == "fewhops"
        assert bridge_entities[1].hop_count == 2

    def test_detect_bridge_entities_marks_answer(self, tracker):
        """Test that answer entities are marked correctly."""
        tracker.mark_answer("Paris", 1)

        tracker.track_segment(0, np.array([0.5]), ["Paris", "France"])
        tracker.track_segment(1, np.array([0.5]), ["Paris", "Germany"])

        bridge_entities = tracker.detect_bridge_entities()

        paris_entity = next(e for e in bridge_entities if e.text == "paris")
        assert paris_entity.is_answer is True

    def test_detect_hops_basic(self, tracker):
        """Test basic hop detection."""
        # Create bridge entity by tracking multiple segments
        tracker.track_segment(0, np.array([0.5]), ["Paris"])
        tracker.track_segment(1, np.array([0.6]), ["Paris"])
        tracker.track_segment(2, np.array([0.7]), ["Paris"])

        hops = tracker.detect_hops()

        # Should detect hops: 0->1, 1->2
        assert len(hops) >= 2

        from_segments = [h.from_segment for h in hops]
        to_segments = [h.to_segment for h in hops]

        assert 0 in from_segments
        assert 1 in from_segments
        assert 1 in to_segments
        assert 2 in to_segments

    def test_detect_hops_with_threshold(self, tracker):
        """Test hop detection respects attention threshold."""
        # High attention entity
        tracker.track_segment(0, np.array([0.9]), ["HighAttention"])
        tracker.track_segment(1, np.array([0.9]), ["HighAttention"])

        # Low attention entity
        tracker.track_segment(0, np.array([0.01]), ["LowAttention"])
        tracker.track_segment(1, np.array([0.01]), ["LowAttention"])

        hops = tracker.detect_hops(attention_threshold=0.5)

        # Only high attention should create hops
        bridging_entities = [h.bridging_entity for h in hops]
        assert "highattention" in bridging_entities
        assert "lowattention" not in bridging_entities

    def test_get_hop_sequence_without_answer(self, tracker):
        """Test getting hop sequence without answer filtering."""
        tracker.track_segment(0, np.array([0.5]), ["Entity1"])
        tracker.track_segment(1, np.array([0.6]), ["Entity1"])
        tracker.track_segment(2, np.array([0.7]), ["Entity2"])
        tracker.track_segment(3, np.array([0.8]), ["Entity2"])

        hop_sequence = tracker.get_hop_sequence(to_answer=False)

        # Should return all hops sorted by from_segment
        assert len(hop_sequence) > 0
        for i in range(len(hop_sequence) - 1):
            assert hop_sequence[i].from_segment <= hop_sequence[i + 1].from_segment

    def test_get_hop_sequence_to_answer(self, tracker):
        """Test getting hop sequence filtered to answer segment."""
        tracker.mark_answer("Target", 2)

        tracker.track_segment(0, np.array([0.5]), ["Bridge"])
        tracker.track_segment(1, np.array([0.6]), ["Bridge"])
        tracker.track_segment(2, np.array([0.7]), ["Bridge", "Target"])

        hop_sequence = tracker.get_hop_sequence(to_answer=True)

        # Should trace path to segment 2 (answer segment)
        if hop_sequence:
            # Last hop should reach answer segment
            assert any(h.to_segment == 2 for h in hop_sequence)

    def test_get_statistics(self, tracker):
        """Test getting tracker statistics."""
        tracker.track_segment(0, np.array([0.5]), ["Entity1", "Entity2"])
        tracker.track_segment(1, np.array([0.6]), ["Entity1", "Entity3"])
        tracker.track_segment(2, np.array([0.7]), ["Entity2"])

        tracker.mark_answer("Entity1", 1)

        stats = tracker.get_statistics()

        assert stats["num_segments"] == 3
        assert stats["num_entities"] == 3
        assert stats["has_answer"] is True
        assert stats["answer_segment"] == 1
        assert "num_bridge_entities" in stats
        assert "num_hops" in stats

    def test_export_analysis(self, tracker):
        """Test exporting analysis to JSON."""
        tracker.track_segment(0, np.array([0.5]), ["Paris", "France"])
        tracker.track_segment(1, np.array([0.6]), ["Paris", "Europe"])

        tracker.mark_answer("Paris", 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "analysis.json"
            tracker.export_analysis(str(output_path))

            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            assert "statistics" in data
            assert "bridge_entities" in data
            assert "all_hops" in data
            assert "hop_sequence_to_answer" in data

    def test_reset(self, tracker):
        """Test resetting tracker state."""
        tracker.track_segment(0, np.array([0.5]), ["Entity"])
        tracker.mark_answer("Answer", 0)

        # Reset
        tracker.reset()

        # All state should be cleared
        assert len(tracker.entity_occurrences) == 0
        assert len(tracker.segment_entities) == 0
        assert len(tracker.segment_attention) == 0
        assert len(tracker.detected_hops) == 0
        assert len(tracker.answer_entities) == 0
        assert tracker.answer_segment is None

    def test_empty_tracker_statistics(self, tracker):
        """Test statistics on empty tracker."""
        stats = tracker.get_statistics()

        assert stats["num_segments"] == 0
        assert stats["num_entities"] == 0
        assert stats["num_bridge_entities"] == 0
        assert stats["num_hops"] == 0


class TestEntityExtraction:
    """Test entity extraction utility."""

    def test_extract_simple_entities_basic(self):
        """Test basic entity extraction."""
        text = "Paris is the capital of France and Europe."
        entities = extract_simple_entities(text)

        assert "Paris" in entities
        assert "France" in entities
        assert "Europe" in entities

    def test_extract_simple_entities_multi_word(self):
        """Test multi-word entity extraction."""
        text = "New York is in the United States of America."
        entities = extract_simple_entities(text)

        # Should extract "New York"
        assert any("New" in e for e in entities)

    def test_extract_simple_entities_min_length(self):
        """Test minimum length filtering."""
        text = "Paris is in France."
        entities = extract_simple_entities(text, min_length=6)

        # "Paris" (5 chars) should be excluded
        # "France" (6 chars) should be included
        assert "Paris" not in entities
        assert "France" in entities

    def test_extract_simple_entities_lowercase_ignored(self):
        """Test that lowercase words are ignored."""
        text = "Paris is the capital."
        entities = extract_simple_entities(text)

        # Only capitalized words
        assert "Paris" in entities
        assert "capital" not in entities  # lowercase
        assert "the" not in entities  # lowercase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
