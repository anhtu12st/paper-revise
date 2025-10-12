"""
Memory Visualization Utilities
===============================

This module provides visualization tools for analyzing differentiable memory
attention patterns, usage statistics, and temporal relationships.

Classes:
    MemoryVisualizer: Create visualizations of memory operations
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class MemoryVisualizer:
    """
    Visualize differentiable memory operations and attention patterns.

    This class provides methods to create various visualizations for
    understanding how the memory system operates:
    - Read/write attention heatmaps
    - Memory usage timelines
    - Temporal link graphs
    - Multi-head attention comparisons
    - Animated segment-by-segment progression

    Example:
        >>> visualizer = MemoryVisualizer(output_dir="./visualizations")
        >>>
        >>> # Plot attention weights for a segment
        >>> visualizer.plot_attention_heatmap(
        >>>     weights=memory_info["read_weights"][0],
        >>>     title="Read Attention - Segment 1",
        >>>     save_path="read_attention_seg1.png"
        >>> )
        >>>
        >>> # Plot usage over time
        >>> visualizer.plot_usage_timeline(
        >>>     usage_history=usage_data,
        >>>     save_path="usage_timeline.png"
        >>> )
    """

    def __init__(
        self,
        output_dir: str = "./visualizations",
        figsize: tuple[int, int] = (12, 8),
        dpi: int = 100,
        colormap: str = "viridis",
    ):
        """
        Initialize memory visualizer.

        Args:
            output_dir: Directory to save visualizations
            figsize: Default figure size (width, height) in inches
            dpi: Dots per inch for saved figures
            colormap: Default matplotlib colormap name
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.figsize = figsize
        self.dpi = dpi
        self.colormap = colormap

        # Create custom colormap for attention (white to blue)
        self.attention_cmap = LinearSegmentedColormap.from_list("attention", ["white", "lightblue", "blue", "darkblue"])

    def plot_attention_heatmap(
        self,
        weights: np.ndarray,
        title: str = "Attention Weights",
        save_path: str | None = None,
        xlabel: str = "Memory Slots",
        ylabel: str = "Heads",
        show_values: bool = False,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot attention weights as a heatmap.

        Args:
            weights: Attention weights array
                    Shape: (num_heads, num_slots) or (num_slots,)
            title: Plot title
            save_path: Path to save figure (relative to output_dir)
            xlabel: X-axis label
            ylabel: Y-axis label
            show_values: Whether to show values in cells
            figsize: Figure size override

        Returns:
            matplotlib Figure object
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Ensure 2D array
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
            ylabel = "Single Head"

        # Create heatmap
        im = ax.imshow(
            weights, aspect="auto", cmap=self.attention_cmap, interpolation="nearest", vmin=0, vmax=weights.max()
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight", rotation=270, labelpad=20)

        # Labels and title
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Set ticks
        ax.set_xticks(np.arange(weights.shape[1]))
        ax.set_yticks(np.arange(weights.shape[0]))

        if weights.shape[0] <= 8 and weights.shape[1] <= 64:
            ax.set_xticklabels(np.arange(weights.shape[1]))
            ax.set_yticklabels(np.arange(weights.shape[0]))
        else:
            # Too many slots, use sparse ticks
            ax.set_xticks(np.arange(0, weights.shape[1], max(1, weights.shape[1] // 10)))
            ax.set_yticks(np.arange(weights.shape[0]))

        # Show values in cells if requested and array is small
        if show_values and weights.shape[0] <= 8 and weights.shape[1] <= 32:
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{weights[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if weights[i, j] > weights.max() * 0.5 else "black",
                        fontsize=8,
                    )

        plt.tight_layout()

        # Save if path provided
        if save_path:
            full_path = self.output_dir / save_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(full_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_usage_timeline(
        self,
        usage_history: np.ndarray | list[np.ndarray],
        save_path: str | None = None,
        title: str = "Memory Usage Over Time",
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot memory slot usage over time.

        Args:
            usage_history: Usage values over time
                          Shape: (num_steps, num_slots) or list of arrays
            save_path: Path to save figure
            title: Plot title
            figsize: Figure size override

        Returns:
            matplotlib Figure object
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Convert to numpy array if list
        if isinstance(usage_history, list):
            usage_history = np.array(usage_history)

        # Plot heatmap
        im = ax.imshow(usage_history.T, aspect="auto", cmap=self.colormap, interpolation="nearest", origin="lower")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Usage Score", rotation=270, labelpad=20)

        # Labels
        ax.set_xlabel("Time Step (Segment)", fontsize=12)
        ax.set_ylabel("Memory Slot", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            fig.savefig(full_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_temporal_links(
        self,
        temporal_links: np.ndarray,
        save_path: str | None = None,
        title: str = "Temporal Link Matrix",
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot temporal link matrix showing relationships between memory slots.

        Args:
            temporal_links: Temporal link matrix
                           Shape: (num_slots, num_slots)
            save_path: Path to save figure
            title: Plot title
            figsize: Figure size override

        Returns:
            matplotlib Figure object
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Plot matrix
        im = ax.imshow(temporal_links, cmap="RdYlBu_r", interpolation="nearest")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Link Strength", rotation=270, labelpad=20)

        # Labels
        ax.set_xlabel("Target Memory Slot", fontsize=12)
        ax.set_ylabel("Source Memory Slot", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Grid
        ax.grid(False)

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            fig.savefig(full_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_multi_head_comparison(
        self,
        heads_weights: np.ndarray,
        save_path: str | None = None,
        title: str = "Multi-Head Attention Comparison",
        operation: str = "Read",
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot comparison of attention patterns across multiple heads.

        Args:
            heads_weights: Attention weights for all heads
                          Shape: (num_heads, num_slots)
            save_path: Path to save figure
            title: Plot title
            operation: "Read" or "Write" for labeling
            figsize: Figure size override

        Returns:
            matplotlib Figure object
        """
        # Calculate figsize - ensure it's tuple[int, int]
        calc_figsize: tuple[int, int]
        if figsize is None:
            calc_figsize = (self.figsize[0], int(self.figsize[1] * 1.5))
        else:
            calc_figsize = figsize

        num_heads = heads_weights.shape[0]

        fig, axes_result = plt.subplots(num_heads, 1, figsize=calc_figsize, sharex=True)

        # Convert axes to list for consistent handling
        from matplotlib.axes import Axes

        axes_list: list[Axes]
        if num_heads == 1:
            axes_list = [axes_result]  # type: ignore[list-item]
        else:
            # axes_result is ndarray[Axes] when num_heads > 1
            # Use type: ignore because mypy can't infer the union type correctly
            axes_list = list(axes_result.flatten())  # type: ignore[arg-type, union-attr, attr-defined]

        # Plot each head
        for i, ax in enumerate(axes_list):
            weights = heads_weights[i]

            # Bar plot
            x = np.arange(len(weights))
            cmap = plt.get_cmap("viridis")
            bars = ax.bar(x, weights, color=cmap(i / num_heads))

            ax.set_ylabel(f"Head {i}", fontsize=10)
            ax.set_ylim((0, max(weights.max() * 1.1, 0.01)))
            ax.grid(True, alpha=0.3, axis="y")

            # Highlight top-k slots
            top_k = min(3, len(weights))
            top_indices = np.argsort(weights)[-top_k:]
            for idx in top_indices:
                bars[idx].set_color("red")
                bars[idx].set_alpha(0.7)

        # Common x-label
        axes_list[-1].set_xlabel("Memory Slot", fontsize=12)
        axes_list[-1].set_xticks(np.arange(len(heads_weights[0])))

        # Title
        fig.suptitle(f"{title} - {operation} Attention", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            fig.savefig(full_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_attention_distribution(
        self,
        read_weights: np.ndarray,
        write_weights: np.ndarray,
        save_path: str | None = None,
        title: str = "Read vs Write Attention Distribution",
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot distribution comparison of read and write attention.

        Args:
            read_weights: Read attention weights
                         Shape: (num_heads, num_slots) or (num_slots,)
            write_weights: Write attention weights (same shape)
            save_path: Path to save figure
            title: Plot title
            figsize: Figure size override

        Returns:
            matplotlib Figure object
        """
        figsize = figsize or self.figsize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Flatten to 1D if multi-head
        if read_weights.ndim == 2:
            read_flat = read_weights.flatten()
            write_flat = write_weights.flatten()
        else:
            read_flat = read_weights
            write_flat = write_weights

        # Histograms
        ax1.hist(read_flat, bins=30, alpha=0.7, color="blue", label="Read")
        ax1.set_xlabel("Attention Weight", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("Read Attention Distribution", fontsize=12)
        ax1.grid(True, alpha=0.3)

        ax2.hist(write_flat, bins=30, alpha=0.7, color="green", label="Write")
        ax2.set_xlabel("Attention Weight", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.set_title("Write Attention Distribution", fontsize=12)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            fig.savefig(full_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def create_animation(
        self,
        segment_data: list[dict[str, np.ndarray]],
        output_path: str,
        fps: int = 2,
        title: str = "Memory Evolution",
    ):
        """
        Create an animation showing memory evolution across segments.

        Args:
            segment_data: List of dicts with keys:
                         - "read_weights": (num_heads, num_slots)
                         - "write_weights": (num_heads, num_slots)
                         - "usage": (num_slots,)
            output_path: Path to save animation (e.g., "animation.mp4")
            fps: Frames per second
            title: Animation title
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Initialize plots
        read_ax, write_ax, usage_ax, dist_ax = axes.flatten()

        def update(frame_idx):
            """Update function for animation."""
            data = segment_data[frame_idx]

            # Clear axes
            for ax in axes.flatten():
                ax.clear()

            # Read attention heatmap
            read_weights = data["read_weights"]
            if read_weights.ndim == 1:
                read_weights = read_weights.reshape(1, -1)

            read_ax.imshow(read_weights, aspect="auto", cmap=self.attention_cmap, interpolation="nearest")
            read_ax.set_title(f"Read Attention - Segment {frame_idx}")
            read_ax.set_xlabel("Memory Slots")
            read_ax.set_ylabel("Heads")

            # Write attention heatmap
            write_weights = data["write_weights"]
            if write_weights.ndim == 1:
                write_weights = write_weights.reshape(1, -1)

            write_ax.imshow(write_weights, aspect="auto", cmap=self.attention_cmap, interpolation="nearest")
            write_ax.set_title(f"Write Attention - Segment {frame_idx}")
            write_ax.set_xlabel("Memory Slots")
            write_ax.set_ylabel("Heads")

            # Usage timeline
            if "usage" in data:
                usage = data["usage"]
                usage_ax.bar(np.arange(len(usage)), usage, color="purple", alpha=0.7)
                usage_ax.set_title(f"Memory Usage - Segment {frame_idx}")
                usage_ax.set_xlabel("Memory Slot")
                usage_ax.set_ylabel("Usage Score")
                usage_ax.set_ylim([0, usage.max() * 1.2 if usage.max() > 0 else 1])

            # Distribution comparison
            dist_ax.hist(read_weights.flatten(), bins=20, alpha=0.5, color="blue", label="Read")
            dist_ax.hist(write_weights.flatten(), bins=20, alpha=0.5, color="green", label="Write")
            dist_ax.set_title("Attention Distribution")
            dist_ax.set_xlabel("Weight Value")
            dist_ax.set_ylabel("Frequency")
            dist_ax.legend()

            plt.tight_layout()

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(segment_data), interval=1000 // fps, repeat=True)

        # Save animation
        full_path = self.output_dir / output_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        writer = animation.PillowWriter(fps=fps) if output_path.endswith(".gif") else animation.FFMpegWriter(fps=fps)
        anim.save(full_path, writer=writer)

        plt.close(fig)

    def create_summary_report(
        self,
        memory_data: dict[str, np.ndarray],
        save_path: str = "summary_report",
    ):
        """
        Create a comprehensive summary report with multiple visualizations.

        Args:
            memory_data: Dictionary containing:
                        - "read_weights": Read attention history
                        - "write_weights": Write attention history
                        - "usage": Usage history
                        - "temporal_links": Temporal link matrix (optional)
            save_path: Base path for saving (without extension)
        """
        # Create individual plots
        if "read_weights" in memory_data:
            self.plot_attention_heatmap(
                memory_data["read_weights"][-1],  # Last segment
                title="Final Read Attention",
                save_path=f"{save_path}_read_attention.png",
            )

        if "write_weights" in memory_data:
            self.plot_attention_heatmap(
                memory_data["write_weights"][-1],
                title="Final Write Attention",
                save_path=f"{save_path}_write_attention.png",
            )

        if "usage" in memory_data:
            self.plot_usage_timeline(memory_data["usage"], save_path=f"{save_path}_usage_timeline.png")

        if "temporal_links" in memory_data:
            self.plot_temporal_links(memory_data["temporal_links"][-1], save_path=f"{save_path}_temporal_links.png")

        if "read_weights" in memory_data and "write_weights" in memory_data:
            self.plot_attention_distribution(
                memory_data["read_weights"][-1],
                memory_data["write_weights"][-1],
                save_path=f"{save_path}_distribution.png",
            )

        print(f"✓ Summary report saved to: {self.output_dir / save_path}_*.png")
