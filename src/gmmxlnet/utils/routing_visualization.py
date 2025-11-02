"""
Visualization functions for GMM routing and expert specialization.

This module provides matplotlib-based visualization tools for analyzing
expert activation patterns, routing behavior, and memory specialization.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def plot_routing_heatmap(
    routing_data: list[dict[str, Any]],
    output_path: str | None = None,
    figsize: tuple = (12, 8),
    cmap: str = "viridis",
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot routing probability heatmap (segments × experts).

    Args:
        routing_data: List of routing records from GMMAnalyzer
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        cmap: Colormap name
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    # Extract routing probabilities into matrix
    routing_matrix = np.array([record["routing_probs"] for record in routing_data])
    num_segments, num_experts = routing_matrix.shape

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(routing_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Configure axes
    ax.set_xlabel("Expert Index", fontsize=12)
    ax.set_ylabel("Segment Index", fontsize=12)
    ax.set_title("Routing Probability Heatmap", fontsize=14, fontweight="bold")

    # Set ticks
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f"E{i}" for i in range(num_experts)])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Routing Probability", rotation=270, labelpad=20)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")

    return fig


def plot_expert_activation_timeline(
    routing_data: list[dict[str, Any]],
    output_path: str | None = None,
    figsize: tuple = (14, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot expert activation timeline (routing probabilities over segments).

    Args:
        routing_data: List of routing records from GMMAnalyzer
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    # Extract routing probabilities
    routing_matrix = np.array([record["routing_probs"] for record in routing_data])
    num_segments, num_experts = routing_matrix.shape

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot line for each expert
    colors = plt.cm.tab10(np.linspace(0, 1, num_experts))
    for expert_idx in range(num_experts):
        ax.plot(
            range(num_segments),
            routing_matrix[:, expert_idx],
            label=f"Expert {expert_idx}",
            color=colors[expert_idx],
            linewidth=2,
            alpha=0.8,
        )

    # Configure axes
    ax.set_xlabel("Segment Index", fontsize=12)
    ax.set_ylabel("Routing Probability", fontsize=12)
    ax.set_title("Expert Activation Timeline", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")

    return fig


def plot_specialization_dendrogram(
    linkage_matrix: np.ndarray,
    output_path: str | None = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot hierarchical clustering dendrogram of expert embeddings.

    Args:
        linkage_matrix: Linkage matrix from scipy.cluster.hierarchy.linkage
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot dendrogram
    dendrogram(
        linkage_matrix,
        ax=ax,
        labels=[f"Expert {i}" for i in range(linkage_matrix.shape[0] + 1)],
        leaf_font_size=10,
        color_threshold=0,
        above_threshold_color="black",
    )

    # Configure axes
    ax.set_xlabel("Expert", fontsize=12)
    ax.set_ylabel("Distance", fontsize=12)
    ax.set_title("Expert Specialization Dendrogram", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")

    return fig


def plot_expert_utilization_bar(
    activation_freq: dict[str, float],
    output_path: str | None = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot bar chart of expert utilization frequencies.

    Args:
        activation_freq: Dictionary mapping expert_id to activation frequency
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    # Sort experts by index
    experts = sorted(activation_freq.keys(), key=lambda x: int(x.split("_")[1]))
    frequencies = [activation_freq[expert] for expert in experts]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    bars = ax.bar(
        range(len(experts)),
        frequencies,
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(experts))),
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels on bars
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{freq:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Configure axes
    ax.set_xlabel("Expert", fontsize=12)
    ax.set_ylabel("Activation Frequency", fontsize=12)
    ax.set_title("Expert Utilization Balance", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(experts)))
    ax.set_xticklabels([f"E{i}" for i in range(len(experts))])
    ax.set_ylim(0, max(frequencies) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")

    return fig


def plot_routing_entropy_distribution(
    routing_data: list[dict[str, Any]],
    output_path: str | None = None,
    figsize: tuple = (10, 6),
    bins: int = 30,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot histogram of routing entropy distribution.

    Args:
        routing_data: List of routing records from GMMAnalyzer
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        bins: Number of histogram bins
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    # Compute entropy for each segment
    entropies = []
    for record in routing_data:
        probs = np.array(record["routing_probs"])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    entropies = np.array(entropies)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    n, bins_edges, patches = ax.hist(
        entropies,
        bins=bins,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )

    # Add mean line
    mean_entropy = np.mean(entropies)
    ax.axvline(
        mean_entropy,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_entropy:.3f}",
    )

    # Configure axes
    ax.set_xlabel("Routing Entropy", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Routing Entropy Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")

    return fig


def generate_all_visualizations(
    analyzer: Any,
    output_dir: str,
    formats: list[str] = None,
    dpi: int = 300,
) -> dict[str, str]:
    """
    Generate all visualizations and save to output directory.

    Args:
        analyzer: GMMAnalyzer instance with tracked routing data
        output_dir: Directory to save visualizations
        formats: List of output formats (default: ['png', 'pdf'])
        dpi: Resolution for saved figures

    Returns:
        Dictionary mapping visualization type to saved file paths
    """
    if formats is None:
        formats = ["png", "pdf"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. Routing heatmap
    for fmt in formats:
        heatmap_path = output_path / f"routing_heatmap.{fmt}"
        plot_routing_heatmap(analyzer.routing_data, str(heatmap_path), dpi=dpi)
        saved_files[f"heatmap_{fmt}"] = str(heatmap_path)

    # 2. Activation timeline
    for fmt in formats:
        timeline_path = output_path / f"activation_timeline.{fmt}"
        plot_expert_activation_timeline(analyzer.routing_data, str(timeline_path), dpi=dpi)
        saved_files[f"timeline_{fmt}"] = str(timeline_path)

    # 3. Specialization dendrogram
    linkage_matrix = analyzer.cluster_experts()
    for fmt in formats:
        dendrogram_path = output_path / f"specialization_dendrogram.{fmt}"
        plot_specialization_dendrogram(linkage_matrix, str(dendrogram_path), dpi=dpi)
        saved_files[f"dendrogram_{fmt}"] = str(dendrogram_path)

    # 4. Expert utilization bar chart
    activation_freq = analyzer.compute_expert_activations()
    for fmt in formats:
        utilization_path = output_path / f"expert_utilization.{fmt}"
        plot_expert_utilization_bar(activation_freq, str(utilization_path), dpi=dpi)
        saved_files[f"utilization_{fmt}"] = str(utilization_path)

    # 5. Routing entropy distribution
    for fmt in formats:
        entropy_path = output_path / f"routing_entropy_distribution.{fmt}"
        plot_routing_entropy_distribution(analyzer.routing_data, str(entropy_path), dpi=dpi)
        saved_files[f"entropy_{fmt}"] = str(entropy_path)

    # Close all figures to free memory
    plt.close("all")

    return saved_files
