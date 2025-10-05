"""Multi-hop reasoning utilities for MA-XLNet.

This module provides tools for tracking, visualizing, and analyzing
multi-hop reasoning chains in memory-augmented question answering.
"""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    sns = None  # type: ignore[assignment]
    VISUALIZATION_AVAILABLE = False


@dataclass
class HopInfo:
    """Information about a single reasoning hop."""

    hop_number: int
    question_focus: str  # Part of question being addressed
    memory_read_weights: np.ndarray  # Attention weights for reading
    memory_write_weights: np.ndarray  # Attention weights for writing
    extracted_info: str  # Information extracted in this hop
    confidence: float  # Confidence score for this hop


@dataclass
class ReasoningChain:
    """Complete reasoning chain for a multi-hop question."""

    question: str
    answer: str
    hops: list[HopInfo]
    bridge_entities: list[str]  # Entities that connect hops
    total_confidence: float
    success: bool


class HopTracker:
    """Track reasoning chains through memory operations.

    This class monitors memory read/write operations during inference
    to reconstruct the multi-hop reasoning process.
    """

    def __init__(self, track_attention: bool = True, track_content: bool = True):
        """Initialize hop tracker.

        Args:
            track_attention: Whether to track attention weights
            track_content: Whether to track memory content changes
        """
        self.track_attention = track_attention
        self.track_content = track_content
        self.reset()

    def reset(self):
        """Reset tracking state for new question."""
        self.current_chain = []
        self.memory_history = []
        self.attention_history = []
        self.hop_count = 0

    def record_hop(
        self, memory_info: dict[str, torch.Tensor], question_part: str | None = None, extracted_info: str | None = None
    ):
        """Record information about a reasoning hop.

        Args:
            memory_info: Dictionary with memory operation details
            question_part: Part of question being addressed
            extracted_info: Information extracted in this hop
        """
        self.hop_count += 1

        # Extract attention weights if available
        read_weights = None
        write_weights = None

        if self.track_attention and "read_weights" in memory_info:
            read_weights = memory_info["read_weights"].detach().cpu().numpy()

        if self.track_attention and "write_weights" in memory_info:
            write_weights = memory_info["write_weights"].detach().cpu().numpy()

        # Calculate confidence based on attention sharpness
        confidence = 1.0
        if read_weights is not None:
            # Confidence is higher when attention is focused
            entropy = -np.sum(read_weights * np.log(read_weights + 1e-10))
            max_entropy = np.log(len(read_weights))
            confidence = 1.0 - (entropy / max_entropy)

        hop = HopInfo(
            hop_number=self.hop_count,
            question_focus=question_part or f"Hop {self.hop_count}",
            memory_read_weights=read_weights if read_weights is not None else np.array([]),
            memory_write_weights=write_weights if write_weights is not None else np.array([]),
            extracted_info=extracted_info or "N/A",
            confidence=confidence,
        )

        self.current_chain.append(hop)

        # Track memory state changes
        if self.track_content and "memory_state" in memory_info:
            self.memory_history.append(memory_info["memory_state"].detach().cpu().numpy())

    def get_reasoning_chain(
        self, question: str, answer: str, bridge_entities: list[str] | None = None
    ) -> ReasoningChain:
        """Get the complete reasoning chain.

        Args:
            question: The input question
            answer: The predicted answer
            bridge_entities: Entities that connect reasoning hops

        Returns:
            Complete reasoning chain information
        """
        total_confidence = np.mean([hop.confidence for hop in self.current_chain]) if self.current_chain else 0.0

        return ReasoningChain(
            question=question,
            answer=answer,
            hops=self.current_chain,
            bridge_entities=bridge_entities or [],
            total_confidence=total_confidence,
            success=len(answer) > 0 and total_confidence > 0.5,
        )

    def visualize_attention_flow(self):
        """Visualize attention flow across hops.

        Returns:
            Matplotlib figure showing attention patterns
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization requires matplotlib. Install with: pip install matplotlib")
            return None

        if not self.current_chain:
            return None

        fig, axes = plt.subplots(len(self.current_chain), 2, figsize=(12, 3 * len(self.current_chain)))

        if len(self.current_chain) == 1:
            axes = axes.reshape(1, -1)

        for i, hop in enumerate(self.current_chain):
            # Plot read weights
            if hop.memory_read_weights is not None:
                axes[i, 0].bar(range(len(hop.memory_read_weights[0])), hop.memory_read_weights[0])
                axes[i, 0].set_title(f"Hop {hop.hop_number} - Read Attention")
                axes[i, 0].set_xlabel("Memory Slot")
                axes[i, 0].set_ylabel("Attention Weight")

            # Plot write weights
            if hop.memory_write_weights is not None:
                axes[i, 1].bar(range(len(hop.memory_write_weights[0])), hop.memory_write_weights[0])
                axes[i, 1].set_title(f"Hop {hop.hop_number} - Write Attention")
                axes[i, 1].set_xlabel("Memory Slot")
                axes[i, 1].set_ylabel("Attention Weight")

        plt.tight_layout()
        return fig


class MemoryVisualizer:
    """Visualize memory states and operations.

    Provides utilities for creating heatmaps and visualizations
    of memory content and access patterns.
    """

    @staticmethod
    def plot_memory_heatmap(memory_state: np.ndarray, title: str = "Memory State", cmap: str = "coolwarm"):
        """Create heatmap of memory state.

        Args:
            memory_state: Memory matrix (slots x dimensions)
            title: Title for the plot
            cmap: Colormap to use

        Returns:
            Matplotlib figure with heatmap
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization requires matplotlib and seaborn. Install with: pip install matplotlib seaborn")
            return None

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            memory_state,
            ax=ax,
            cmap=cmap,
            center=0,
            cbar_kws={"label": "Value"},
            xticklabels=False,
            yticklabels=range(memory_state.shape[0]),
        )

        ax.set_title(title)
        ax.set_xlabel("Memory Dimension")
        ax.set_ylabel("Memory Slot")

        return fig

    @staticmethod
    def plot_usage_pattern(usage: np.ndarray, temporal_links: np.ndarray | None = None):
        """Visualize memory usage and temporal patterns.

        Args:
            usage: Usage vector for each memory slot
            temporal_links: Temporal link matrix (optional)

        Returns:
            Matplotlib figure
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization requires matplotlib and seaborn. Install with: pip install matplotlib seaborn")
            return None

        if temporal_links is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))

        # Plot usage
        ax1.bar(range(len(usage)), usage)
        ax1.set_title("Memory Slot Usage")
        ax1.set_xlabel("Memory Slot")
        ax1.set_ylabel("Usage Level")
        ax1.set_ylim([0, 1])

        # Plot temporal links if provided
        if temporal_links is not None:
            sns.heatmap(
                temporal_links,
                ax=ax2,
                cmap="Blues",
                cbar_kws={"label": "Link Strength"},
                xticklabels=range(temporal_links.shape[0]),
                yticklabels=range(temporal_links.shape[1]),
            )
            ax2.set_title("Temporal Links Between Slots")
            ax2.set_xlabel("To Slot")
            ax2.set_ylabel("From Slot")

        plt.tight_layout()
        return fig


class MultiHopMetrics:
    """Metrics for evaluating multi-hop reasoning performance."""

    @staticmethod
    def hop_accuracy(predicted_chains: list[ReasoningChain], gold_chains: list[ReasoningChain]) -> dict[str, float]:
        """Calculate hop-level accuracy metrics.

        Args:
            predicted_chains: Predicted reasoning chains
            gold_chains: Gold standard reasoning chains

        Returns:
            Dictionary with various metrics
        """
        metrics = {
            "exact_match": 0.0,
            "partial_match": 0.0,
            "bridge_entity_recall": 0.0,
            "bridge_entity_precision": 0.0,
            "avg_confidence": 0.0,
            "success_rate": 0.0,
        }

        if not predicted_chains or not gold_chains:
            return metrics

        total_exact = 0
        total_partial = 0
        total_bridge_recall = 0.0
        total_bridge_precision = 0.0
        total_confidence = 0.0
        total_success = 0

        for pred, gold in zip(predicted_chains, gold_chains):
            # Exact match on answer
            if pred.answer == gold.answer:
                total_exact += 1

            # Partial match (at least one correct hop)
            if len(pred.hops) > 0 and len(gold.hops) > 0:
                total_partial += 1

            # Bridge entity metrics
            if gold.bridge_entities:
                pred_entities = set(pred.bridge_entities)
                gold_entities = set(gold.bridge_entities)

                if gold_entities:
                    recall = len(pred_entities & gold_entities) / len(gold_entities)
                    total_bridge_recall += recall

                if pred_entities:
                    precision = len(pred_entities & gold_entities) / len(pred_entities)
                    total_bridge_precision += precision

            # Confidence and success
            total_confidence += pred.total_confidence
            if pred.success:
                total_success += 1

        n = len(predicted_chains)
        metrics["exact_match"] = total_exact / n
        metrics["partial_match"] = total_partial / n
        metrics["bridge_entity_recall"] = total_bridge_recall / n
        metrics["bridge_entity_precision"] = total_bridge_precision / n
        metrics["avg_confidence"] = total_confidence / n
        metrics["success_rate"] = total_success / n

        return metrics

    @staticmethod
    def analyze_error_patterns(failed_chains: list[ReasoningChain]) -> dict[str, Any]:
        """Analyze common error patterns in failed reasoning.

        Args:
            failed_chains: Chains that failed to produce correct answer

        Returns:
            Analysis of error patterns
        """
        error_analysis = {
            "total_failures": len(failed_chains),
            "incomplete_chains": 0,
            "low_confidence": 0,
            "missing_bridge": 0,
            "wrong_first_hop": 0,
            "avg_hops_attempted": 0.0,
        }

        if not failed_chains:
            return error_analysis

        total_hops = 0
        for chain in failed_chains:
            # Incomplete chains
            if len(chain.hops) < 2:
                error_analysis["incomplete_chains"] += 1

            # Low confidence
            if chain.total_confidence < 0.3:
                error_analysis["low_confidence"] += 1

            # Missing bridge entities
            if not chain.bridge_entities:
                error_analysis["missing_bridge"] += 1

            # Wrong first hop (low confidence on first hop)
            if chain.hops and chain.hops[0].confidence < 0.5:
                error_analysis["wrong_first_hop"] += 1

            total_hops += len(chain.hops)

        error_analysis["avg_hops_attempted"] = float(total_hops) / len(failed_chains)

        return error_analysis


class MultiHopDebugger:
    """Debugging tools for multi-hop reasoning."""

    @staticmethod
    def export_reasoning_chain(chain: ReasoningChain, output_path: str):
        """Export reasoning chain to JSON for analysis.

        Args:
            chain: Reasoning chain to export
            output_path: Path to save JSON file
        """
        # Convert chain to serializable format
        hops_list: list[dict[str, Any]] = []

        for hop in chain.hops:
            hop_dict: dict[str, Any] = {
                "hop_number": hop.hop_number,
                "question_focus": hop.question_focus,
                "extracted_info": hop.extracted_info,
                "confidence": float(hop.confidence),
            }
            # Optionally include attention weights
            if hop.memory_read_weights is not None:
                hop_dict["read_weights"] = hop.memory_read_weights.tolist()
            if hop.memory_write_weights is not None:
                hop_dict["write_weights"] = hop.memory_write_weights.tolist()

            hops_list.append(hop_dict)

        chain_dict: dict[str, Any] = {
            "question": chain.question,
            "answer": chain.answer,
            "bridge_entities": chain.bridge_entities,
            "total_confidence": float(chain.total_confidence),
            "success": chain.success,
            "hops": hops_list,
        }

        with open(output_path, "w") as f:
            json.dump(chain_dict, f, indent=2)

        print(f"Reasoning chain exported to {output_path}")
