"""
GMM interpretability and analysis tools.

This module provides the GMMAnalyzer class for tracking routing behavior,
computing expert specialization metrics, and generating visualizations.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage
from torch.utils.data import DataLoader


class GMMAnalyzer:
    """
    Analyzer for GMM expert specialization and routing behavior.

    This class tracks routing probabilities during evaluation, computes
    specialization metrics, and generates visualizations for interpretability.

    Args:
        model: GMMXLNetForQA model to analyze
        device: Device to run analysis on (default: 'cuda' if available)

    Attributes:
        model: The GMM model being analyzed
        device: Computing device
        routing_data: List of routing probability records
        num_experts: Number of experts in the model
    """

    def __init__(
        self,
        model: Any,
        device: str | None = None,
    ):
        """Initialize the GMMAnalyzer."""
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Extract number of experts from model
        if hasattr(model, "memory_mixture"):
            self.num_experts = model.memory_mixture.num_experts
        else:
            raise ValueError("Model does not have memory_mixture attribute")

        # Storage for tracking data
        self.routing_data: list[dict[str, Any]] = []
        self._activation_counts: torch.Tensor | None = None
        self._total_activations: int = 0

    def reset_tracking(self) -> None:
        """Reset all tracked data."""
        self.routing_data = []
        self._activation_counts = None
        self._total_activations = 0

    def track_routing(
        self,
        dataloader: DataLoader,
        max_segments: int | None = None,
    ) -> dict[str, Any]:
        """
        Track routing probabilities across evaluation dataset.

        Args:
            dataloader: DataLoader providing batched inputs
            max_segments: Maximum number of segments to process (None for all)

        Returns:
            Summary statistics of routing behavior
        """
        self.reset_tracking()
        self.model.eval()

        # Initialize activation counters
        self._activation_counts = torch.zeros(self.num_experts, device=self.device)

        segments_processed = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_segments and segments_processed >= max_segments:
                    break

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                # Get memory-related parameters if present
                memory_state = batch.get("memory_state")
                mem_read_ids = batch.get("mem_read_ids")
                mem_write_ids = batch.get("mem_write_ids")

                # Move memory state to device if present
                if memory_state is not None:
                    memory_state = {k: v.to(self.device) for k, v in memory_state.items()}

                # Forward pass to get routing probabilities
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    memory_state=memory_state,
                    mem_read_ids=mem_read_ids,
                    mem_write_ids=mem_write_ids,
                    return_routing_info=True,  # Enable routing tracking
                )

                # Extract routing probabilities
                if "routing_info" in outputs and "routing_probs" in outputs["routing_info"]:
                    routing_probs = outputs["routing_info"]["routing_probs"]  # Shape: (batch_size, num_experts)

                    # Store routing data for each item in batch
                    for idx in range(routing_probs.size(0)):
                        probs = routing_probs[idx].cpu().numpy()
                        self.routing_data.append(
                            {
                                "batch_idx": batch_idx,
                                "item_idx": idx,
                                "segment_idx": segments_processed + idx,
                                "routing_probs": probs.tolist(),
                                "document_id": batch.get("doc_id", [f"doc_{batch_idx}"])[idx]
                                if "doc_id" in batch
                                else f"doc_{batch_idx}",
                            }
                        )

                        # Update activation counts
                        self._activation_counts += routing_probs[idx]

                    segments_processed += routing_probs.size(0)
                    self._total_activations += routing_probs.size(0)
                else:
                    # Model doesn't support routing tracking
                    raise ValueError(
                        "Model does not output routing_info. Ensure return_routing_info=True is supported."
                    )

        # Compute summary statistics
        summary = {
            "segments_processed": segments_processed,
            "num_experts": self.num_experts,
            "mean_routing_entropy": self.compute_routing_entropy(),
            "expert_activation_freq": (self._activation_counts / self._total_activations).cpu().numpy().tolist(),
        }

        return summary

    def export_routing_to_json(self, output_path: str) -> None:
        """
        Export routing data to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_data = {
            "num_experts": self.num_experts,
            "segments_processed": len(self.routing_data),
            "routing_data": self.routing_data,
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

    def compute_expert_activations(self) -> dict[str, float]:
        """
        Compute per-expert activation frequencies.

        Returns:
            Dictionary mapping expert_id to activation frequency
        """
        if self._activation_counts is None:
            raise ValueError("No routing data tracked. Call track_routing() first.")

        activation_freq = (self._activation_counts / self._total_activations).cpu().numpy()

        return {f"expert_{i}": float(freq) for i, freq in enumerate(activation_freq)}

    def compute_routing_entropy(self) -> float:
        """
        Compute mean entropy of routing distribution.

        High entropy indicates uniform routing (less specialization).
        Low entropy indicates peaked routing (more specialization).

        Returns:
            Mean routing entropy across all segments
        """
        if not self.routing_data:
            raise ValueError("No routing data tracked. Call track_routing() first.")

        entropies = []
        for record in self.routing_data:
            probs = torch.tensor(record["routing_probs"])
            # Add epsilon for numerical stability
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            entropies.append(entropy.item())

        return float(np.mean(entropies))

    def compute_expert_diversity(
        self,
        expert_embeddings: torch.Tensor | None = None,
    ) -> float:
        """
        Measure how different experts are from each other.

        Uses average pairwise cosine distance between expert embeddings.
        Higher diversity indicates more specialization.

        Args:
            expert_embeddings: (num_experts, embedding_dim) tensor.
                If None, extracts from model's expert memories.

        Returns:
            Expert diversity score (0 to 1, higher is more diverse)
        """
        if expert_embeddings is None:
            expert_embeddings = self.extract_expert_embeddings()

        # Compute pairwise cosine similarities
        similarities = F.cosine_similarity(
            expert_embeddings.unsqueeze(1),
            expert_embeddings.unsqueeze(0),
            dim=-1,
        )

        # Average off-diagonal elements (exclude self-similarity)
        k = similarities.size(0)
        mask = ~torch.eye(k, dtype=torch.bool, device=similarities.device)
        diversity = 1.0 - similarities[mask].mean().item()

        # Clamp to [0, 1] for numerical stability
        return float(max(0.0, min(1.0, diversity)))

    def compute_utilization_balance(self) -> float:
        """
        Measure how evenly experts are utilized.

        Uses coefficient of variation of activation frequencies.
        Perfect balance = 1.0 (all experts equally used).

        Returns:
            Utilization balance score (0 to 1, higher is more balanced)
        """
        if self._activation_counts is None:
            raise ValueError("No routing data tracked. Call track_routing() first.")

        activation_freq = self._activation_counts / self._total_activations
        std = activation_freq.std().item()
        mean = activation_freq.mean().item()

        # Coefficient of variation
        cv = std / (mean + 1e-10)

        # Normalize to [0, 1] where 1 is perfect balance
        balance = 1.0 / (1.0 + cv)

        return float(balance)

    def compute_specialization_score(
        self,
        expert_embeddings: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Compute composite specialization metrics.

        Returns:
            Dictionary with all specialization metrics
        """
        return {
            "routing_entropy": self.compute_routing_entropy(),
            "expert_diversity": self.compute_expert_diversity(expert_embeddings),
            "utilization_balance": self.compute_utilization_balance(),
        }

    def extract_expert_embeddings(self) -> torch.Tensor:
        """
        Extract expert memory representations from the model.

        Returns:
            (num_experts, embedding_dim) tensor of expert embeddings
        """
        # Get expert memory states from the memory mixture
        if not hasattr(self.model, "memory_mixture"):
            raise ValueError("Model does not have memory_mixture attribute")

        memory_mixture = self.model.memory_mixture

        # Extract embeddings from each expert's memory bank
        # Assuming each expert has a memory tensor we can use as embedding
        embeddings = []
        for expert_idx in range(self.num_experts):
            # Get the expert's memory state
            expert_memory = memory_mixture.expert_memories[expert_idx]
            # Average across memory slots to get a single embedding vector
            expert_embedding = expert_memory.mean(dim=0)  # (hidden_dim,)
            embeddings.append(expert_embedding)

        return torch.stack(embeddings)  # (num_experts, hidden_dim)

    def compute_expert_similarity(
        self,
        expert_i: int,
        expert_j: int,
        expert_embeddings: torch.Tensor | None = None,
    ) -> float:
        """
        Compute cosine similarity between two experts.

        Args:
            expert_i: First expert index
            expert_j: Second expert index
            expert_embeddings: Optional precomputed embeddings

        Returns:
            Cosine similarity score (0 to 1)
        """
        if expert_embeddings is None:
            expert_embeddings = self.extract_expert_embeddings()

        similarity = F.cosine_similarity(
            expert_embeddings[expert_i].unsqueeze(0),
            expert_embeddings[expert_j].unsqueeze(0),
        )

        return float(similarity.item())

    def cluster_experts(
        self,
        expert_embeddings: torch.Tensor | None = None,
        method: str = "ward",
    ) -> np.ndarray:
        """
        Perform hierarchical clustering on expert embeddings.

        Args:
            expert_embeddings: (num_experts, embedding_dim) tensor
            method: Linkage method ('ward', 'average', 'complete')

        Returns:
            Linkage matrix for dendrogram plotting
        """
        if expert_embeddings is None:
            expert_embeddings = self.extract_expert_embeddings()

        # Convert to numpy and compute linkage
        embeddings_np = expert_embeddings.cpu().numpy()
        linkage_matrix: np.ndarray = linkage(embeddings_np, method=method)

        return linkage_matrix

    def generate_analysis_report(
        self,
        output_path: str,
        include_model_id: str | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive analysis report and save to JSON.

        Args:
            output_path: Path to output JSON file
            include_model_id: Optional model identifier
            dataset_name: Optional dataset name

        Returns:
            Analysis report dictionary
        """
        expert_embeddings = self.extract_expert_embeddings()
        specialization_metrics = self.compute_specialization_score(expert_embeddings)

        report = {
            "model_id": include_model_id or "unknown",
            "num_experts": self.num_experts,
            "evaluation_dataset": dataset_name or "unknown",
            "metrics": specialization_metrics,
            "expert_activations": self.compute_expert_activations(),
            "routing_data": self.routing_data,
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        return report
