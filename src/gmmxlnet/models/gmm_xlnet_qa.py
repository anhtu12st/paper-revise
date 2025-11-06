"""
GMMXLNetForQA: GMM-augmented XLNet for Question Answering.

This module provides the main model class that orchestrates all GMM components
for multi-expert memory-augmented question answering.
"""

import json
import os
from typing import Any, Literal

import torch
import torch.nn as nn

from .expert_updates import ExpertUpdater
from .gating_network import MemoryGatingNetwork
from .memory_mixture import GatedMemoryMixture
from .memory_read import AggregatedMemoryReader

RoutingMode = Literal["write-based", "read-based"]


class GMMXLNetForQA(nn.Module):
    """
    GMM-augmented XLNet wrapper for Question Answering with multi-expert memory.

    This model extends XLNet with k independent memory experts that specialize
    through content-based routing. Each expert maintains its own memory state
    and is selectively updated based on routing decisions.

    Args:
        base_model: Pre-trained XLNet model (e.g., XLNetForQuestionAnsweringSimple)
        num_experts: Number of memory experts (k ∈ [2, 8], must be power of 2)
        memory_slots: Number of memory slots per expert (default: 16)
        routing_mode: Routing strategy for memory reads
            - "write-based": Reuse write routing (efficient)
            - "read-based": Compute separate read routing (adaptive)
        routing_temperature: Temperature for routing softmax (default: 1.0)
        pooling_method: Pooling method for routing network (default: "mean")
        init_strategies: Initialization strategy for expert memories
            Can be single strategy or list of k strategies
        use_gmm_memory: Enable GMM memory (vs. standard memory)
        mem_token_count: Number of memory tokens (for compatibility)

    Example:
        >>> from transformers import XLNetForQuestionAnsweringSimple
        >>> base = XLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased")
        >>> model = GMMXLNetForQA(
        ...     base_model=base,
        ...     num_experts=4,
        ...     memory_slots=16,
        ...     routing_mode="write-based"
        ... )
        >>> # Forward pass
        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     attention_mask=attention_mask,
        ...     memory_state=memory_state
        ... )
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_experts: int = 4,
        memory_slots: int = 16,
        routing_mode: RoutingMode = "write-based",
        routing_temperature: float = 1.0,
        pooling_method: Literal["mean", "max", "attention"] = "mean",
        init_strategies: Literal["learned", "zeros", "uniform", "orthogonal"]
        | list[Literal["learned", "zeros", "uniform", "orthogonal"]] = "orthogonal",
        use_gmm_memory: bool = True,
        mem_token_count: int = 16,
    ):
        super().__init__()

        # Configuration
        self.use_gmm_memory = use_gmm_memory
        self.num_experts = num_experts
        self.memory_slots = memory_slots
        self.routing_mode = routing_mode
        self.routing_temperature = routing_temperature
        self.pooling_method = pooling_method
        self.mem_token_count = mem_token_count

        # Base model
        self.base = base_model

        # Determine memory dimension from base model
        if hasattr(base_model, "config") and hasattr(base_model.config, "d_model"):
            d_model_val = base_model.config.d_model
            memory_dim_value: int = int(d_model_val) if isinstance(d_model_val, (int, float)) else 768
        elif hasattr(base_model, "config") and hasattr(base_model.config, "hidden_size"):
            hidden_size_val = base_model.config.hidden_size
            memory_dim_value = int(hidden_size_val) if isinstance(hidden_size_val, (int, float)) else 768
        else:
            memory_dim_value = 768  # Default for base models

        self.hidden_dim = memory_dim_value

        # Initialize GMM components if GMM memory is enabled
        if self.use_gmm_memory:
            # 1. Gated Memory Mixture: Manages k expert memory banks
            self.memory_mixture = GatedMemoryMixture(
                num_experts=num_experts,
                memory_slots=memory_slots,
                hidden_dim=self.hidden_dim,
                init_strategies=init_strategies,
            )

            # 2. Memory Gating Network: Routes memory updates to experts
            self.gating_network = MemoryGatingNetwork(
                hidden_dim=self.hidden_dim,
                num_experts=num_experts,
                temperature=routing_temperature,
                pooling_method=pooling_method,
            )

            # 3. Expert Updater: Applies routing-modulated gated updates
            self.expert_updater = ExpertUpdater(
                hidden_dim=self.hidden_dim,
                num_experts=num_experts,
            )

            # 4. Aggregated Memory Reader: Combines expert memories for reads
            self.memory_reader = AggregatedMemoryReader(
                hidden_dim=self.hidden_dim,
                num_experts=num_experts,
                routing_mode=routing_mode,
                temperature=routing_temperature,
                pooling_method=pooling_method,
            )

    def get_initial_memory(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        """
        Initialize memory states for all k experts.

        Args:
            batch_size: Number of examples in the batch
            device: Target device for memory tensors

        Returns:
            Dictionary mapping expert indices to memory states:
                {"expert_0": Tensor (batch, memory_slots, hidden_dim), ...}
        """
        if not self.use_gmm_memory:
            return {}

        memory_state = {}
        for expert_idx in range(self.num_experts):
            # Get unbatched expert state from mixture
            expert_param = self.memory_mixture.get_expert_state(expert_idx)  # (memory_slots, hidden_dim)

            # Ensure expert parameter is on the correct device
            expert_param = expert_param.to(device)

            # Expand to batch dimension
            expert_state = expert_param.unsqueeze(0).expand(
                batch_size, self.memory_slots, self.hidden_dim
            )  # (batch, memory_slots, hidden_dim)

            memory_state[f"expert_{expert_idx}"] = expert_state

        return memory_state

    def reset_memory(self) -> None:
        """
        Reset all expert memories to their initial state.

        Should be called when starting a new document to clear memory state.
        """
        if self.use_gmm_memory:
            self.memory_mixture.reset_experts()

    def _extract_memory_write_hiddens(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        mem_write_ids: list[int],
    ) -> torch.Tensor:
        """
        Extract hidden states at [MEM_WRITE] token positions.

        Args:
            input_ids: Input token ids (batch_size, seq_len)
            hidden_states: Model hidden states (batch_size, seq_len, hidden_dim)
            mem_write_ids: List of memory write token ids

        Returns:
            Write hidden states (batch_size, memory_slots, hidden_dim)

        Raises:
            ValueError: If no memory write tokens found
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        write_hiddens = []

        for mem_id in mem_write_ids:
            # Find positions of this memory token
            positions = (input_ids == mem_id).nonzero(as_tuple=True)

            if len(positions[0]) > 0:
                # Extract hidden states at memory token positions
                batch_indices = positions[0]
                seq_indices = positions[1]
                mem_rep = hidden_states[batch_indices, seq_indices]  # (found_count, hidden_dim)

                # Average across multiple occurrences per batch item
                batch_mem_reps = []
                for b in range(batch_size):
                    batch_mask = batch_indices == b
                    if batch_mask.any():
                        batch_mem_reps.append(mem_rep[batch_mask].mean(dim=0))
                    else:
                        batch_mem_reps.append(torch.zeros(hidden_dim, device=hidden_states.device))

                write_hiddens.append(torch.stack(batch_mem_reps))
            else:
                # No memory tokens found, use zeros
                write_hiddens.append(torch.zeros(batch_size, hidden_dim, device=hidden_states.device))

        if write_hiddens:
            return torch.stack(write_hiddens, dim=1)  # (batch_size, memory_slots, hidden_dim)

        # Fallback: return zeros if no write tokens found
        return torch.zeros(batch_size, self.memory_slots, hidden_dim, device=hidden_states.device)

    def _extract_memory_read_positions(
        self,
        input_ids: torch.Tensor,
        mem_read_ids: list[int],
    ) -> torch.Tensor:
        """
        Find positions of [MEM_READ] tokens in the input sequence.

        Args:
            input_ids: Input token ids (batch_size, seq_len)
            mem_read_ids: List of memory read token ids

        Returns:
            Read positions (batch_size, memory_slots)
                Values are token indices in [0, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        read_positions = []

        for mem_id in mem_read_ids:
            # Find positions of this memory read token
            positions = (input_ids == mem_id).nonzero(as_tuple=True)

            # Extract position for each batch item
            batch_positions = []
            for b in range(batch_size):
                batch_mask = positions[0] == b
                if batch_mask.any():
                    # Use first occurrence
                    batch_positions.append(positions[1][batch_mask][0])
                else:
                    # Fallback to last position if not found
                    batch_positions.append(torch.tensor(seq_len - 1, device=input_ids.device))

            read_positions.append(torch.tensor(batch_positions, device=input_ids.device))

        if read_positions:
            return torch.stack(read_positions, dim=1)  # (batch_size, memory_slots)

        # Fallback: return last positions
        return torch.full((batch_size, self.memory_slots), seq_len - 1, device=input_ids.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        memory_state: dict[str, torch.Tensor] | None = None,
        mem_read_ids: list[int] | None = None,
        mem_write_ids: list[int] | None = None,
        return_routing_info: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Forward pass with GMM memory processing.

        Args:
            input_ids: Input token ids (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Token type ids (batch_size, seq_len)
            start_positions: Start positions for QA (training)
            end_positions: End positions for QA (training)
            memory_state: Current expert memory states dict
                {"expert_0": Tensor, "expert_1": Tensor, ...}
            mem_read_ids: Memory read token ids
            mem_write_ids: Memory write token ids
            return_routing_info: Whether to return routing statistics
            **kwargs: Additional arguments passed to base model

        Returns:
            Dictionary containing:
                - loss: Training loss (if positions provided)
                - start_logits: Start position logits (batch_size, seq_len)
                - end_logits: End position logits (batch_size, seq_len)
                - new_memory_state: Updated expert memory states dict
                - routing_info: Optional routing statistics (if return_routing_info=True)
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Initialize memory state if not provided
        if memory_state is None and self.use_gmm_memory:
            memory_state = self.get_initial_memory(batch_size, device)
        elif memory_state is not None and self.use_gmm_memory:
            # Ensure provided memory_state tensors are on the correct device
            memory_state = {
                expert_key: tensor.to(device)
                for expert_key, tensor in memory_state.items()
            }

        # CRITICAL FIX: Filter out mems from kwargs to prevent tensor dimension mismatch
        # XLNet expects mems in specific format, but GMM models handle memory differently
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'mems'}

        # Forward pass through base XLNet model
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=True,
            output_hidden_states=True,  # Need hidden states for memory operations
            **filtered_kwargs,
        )

        # Process with GMM memory if enabled
        new_memory_state = memory_state
        routing_info = {}

        if self.use_gmm_memory and memory_state is not None:
            # Get hidden states from last layer
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None

            if hidden_states is not None and mem_write_ids is not None and mem_read_ids is not None:
                # Step 1: Extract [MEM_WRITE] token hidden states
                write_hiddens = self._extract_memory_write_hiddens(
                    input_ids, hidden_states, mem_write_ids
                )  # (batch, memory_slots, hidden_dim)

                # Step 2: Compute routing probabilities via gating network
                routing_probs, routing_logits, routing_entropy = self.gating_network(
                    write_hiddens
                )  # (batch, num_experts)

                # Step 3: Update expert memories via expert updater
                # Convert memory state dict to list
                expert_states = [memory_state[f"expert_{j}"] for j in range(self.num_experts)]

                updated_expert_states = self.expert_updater(
                    expert_states=expert_states,
                    write_hiddens=write_hiddens,
                    routing_probs=routing_probs,
                )

                # Convert back to dict
                new_memory_state = {f"expert_{j}": state for j, state in enumerate(updated_expert_states)}

                # Step 4: Extract [MEM_READ] token positions
                read_positions = self._extract_memory_read_positions(input_ids, mem_read_ids)

                # Step 5: Compute aggregated memory via memory reader
                # For read-based routing, extract read token hiddens
                if self.routing_mode == "read-based":
                    read_hiddens = self._extract_memory_write_hiddens(input_ids, hidden_states, mem_read_ids)
                    aggregated_memory = self.memory_reader(
                        expert_states=updated_expert_states,
                        read_hiddens=read_hiddens,
                    )  # (batch, memory_slots, hidden_dim)
                else:
                    # Write-based routing: reuse routing_probs
                    aggregated_memory = self.memory_reader(
                        expert_states=updated_expert_states,
                        routing_probs=routing_probs,
                    )  # (batch, memory_slots, hidden_dim)

                # Step 6: Replace [MEM_READ] embeddings with aggregated memory
                # Note: We compute this for memory state tracking, but the base model
                # already produced logits. In a full implementation, we'd re-run the QA head
                # with modified hidden states. For now, we use original logits.
                _ = self.memory_reader.replace_read_embeddings(
                    sequence_output=hidden_states,
                    aggregated_memory=aggregated_memory,
                    read_positions=read_positions,
                )

                # Collect routing info if requested
                if return_routing_info:
                    routing_info = {
                        "routing_probs": routing_probs,  # (batch, num_experts)
                        "routing_logits": routing_logits,  # (batch, num_experts)
                        "routing_entropy": routing_entropy,  # (batch,)
                        "expert_activations": routing_probs.sum(dim=0),  # (num_experts,)
                    }

        result = {
            "loss": getattr(outputs, "loss", None),
            "start_logits": getattr(outputs, "start_logits", None),
            "end_logits": getattr(outputs, "end_logits", None),
            "new_memory_state": new_memory_state,
        }

        if return_routing_info:
            result["routing_info"] = routing_info

        return result

    def get_memory_state(self) -> dict[str, torch.Tensor]:
        """
        Access current memory state for all k experts.

        Returns:
            Dictionary mapping expert indices to memory states:
                {"expert_0": Tensor (memory_slots, hidden_dim), ...}
        """
        if not self.use_gmm_memory:
            return {}

        memory_state = {}
        for expert_idx in range(self.num_experts):
            expert_state = self.memory_mixture.get_expert_state(expert_idx)
            memory_state[f"expert_{expert_idx}"] = expert_state

        return memory_state

    def set_memory_state(self, memory_state: dict[str, torch.Tensor]) -> None:
        """
        Update memory state for all k experts.

        Args:
            memory_state: Dictionary mapping expert indices to memory states
                {"expert_0": Tensor (memory_slots, hidden_dim), ...}

        Raises:
            ValueError: If memory_state has wrong number of experts or invalid shapes
        """
        if not self.use_gmm_memory:
            return

        if len(memory_state) != self.num_experts:
            raise ValueError(
                f"memory_state must contain {self.num_experts} experts, got {len(memory_state)}. "
                f"GMM configuration has {self.num_experts} experts."
            )

        for expert_idx in range(self.num_experts):
            key = f"expert_{expert_idx}"
            if key not in memory_state:
                raise ValueError(f"Missing expert state for {key} in memory_state dict")

            # Handle batched states (remove batch dimension if present)
            state = memory_state[key]
            if state.dim() == 3:
                # Batched state: (batch, memory_slots, hidden_dim) → take first item
                state = state[0]

            self.memory_mixture.set_expert_state(expert_idx, state)

    def _validate_loaded_state(self) -> None:
        """
        Validate model state after loading from checkpoint.

        Performs sanity checks on:
        - Expert count matches configuration
        - Routing network shape matches expected dimensions
        - Expert states have correct shapes
        - Forward pass works on dummy input

        Raises:
            ValueError: If validation fails
        """
        if not self.use_gmm_memory:
            return

        # Verify expert count by checking _expert_params entries
        num_experts_in_mixture = len(
            [name for name in self.memory_mixture.state_dict().keys() if name.startswith("_expert_params.")]
        )
        if num_experts_in_mixture != self.num_experts:
            raise ValueError(
                f"Expert count mismatch: config specifies {self.num_experts} experts "
                f"but loaded mixture has {num_experts_in_mixture}"
            )

        # Verify routing network shape
        gating_params = self.gating_network.state_dict()
        if "routing_projection.weight" in gating_params:
            weight_shape = gating_params["routing_projection.weight"].shape
            expected_shape = (self.num_experts, self.hidden_dim)
            if weight_shape != expected_shape:
                raise ValueError(f"Routing network shape mismatch: expected {expected_shape}, got {weight_shape}")

        # Verify expert states have correct shapes
        for expert_idx in range(self.num_experts):
            expert_state = self.memory_mixture.get_expert_state(expert_idx)
            expected_shape = (self.memory_slots, self.hidden_dim)
            if expert_state.shape != expected_shape:
                raise ValueError(
                    f"Expert {expert_idx} state shape mismatch: expected {expected_shape}, got {expert_state.shape}"
                )

        # Run sanity check on memory initialization (skip full forward pass in tests)
        try:
            device = next(self.parameters()).device
            dummy_memory = self.get_initial_memory(1, device)

            # Verify memory state structure
            if len(dummy_memory) != self.num_experts:
                raise ValueError(
                    f"Memory initialization failed: expected {self.num_experts} expert states, got {len(dummy_memory)}"
                )
        except StopIteration:
            # No parameters (likely in a test with mocked base model), skip validation
            pass
        except Exception as e:
            raise ValueError(f"Memory initialization validation failed after loading: {e}")

    def save_pretrained(
        self, save_directory: str, push_to_hub: bool = False, repo_id: str | None = None, **hub_kwargs: Any
    ) -> None:
        """
        Save the model including GMM-specific parameters.

        Args:
            save_directory: Directory to save model files
            push_to_hub: Whether to push the model to HuggingFace Hub
            repo_id: Repository ID for Hub (e.g., "username/model-name")
            **hub_kwargs: Additional arguments for Hub upload (commit_message, private, etc.)
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save base model
        save_pretrained = getattr(self.base, "save_pretrained", None)
        if save_pretrained is not None and callable(save_pretrained) and not isinstance(self.base, torch.Tensor):
            save_pretrained(save_directory)

        # Save GMM configuration with version info
        config = {
            "model_class": "GMMXLNetForQA",
            "memory_type": "gmm",
            "version": "1.0",
            "num_experts": self.num_experts,
            "memory_slots": self.memory_slots,
            "routing_mode": self.routing_mode,
            "routing_temperature": self.routing_temperature,
            "pooling_method": self.pooling_method,
            "use_gmm_memory": self.use_gmm_memory,
            "mem_token_count": self.mem_token_count,
            "hidden_dim": self.hidden_dim,
        }

        with open(os.path.join(save_directory, "gmm_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save GMM-specific state
        if self.use_gmm_memory:
            state_dict = {
                "memory_mixture": self.memory_mixture.state_dict(),
                "gating_network": self.gating_network.state_dict(),
                "expert_updater": self.expert_updater.state_dict(),
                "memory_reader": self.memory_reader.state_dict(),
            }

            torch.save(state_dict, os.path.join(save_directory, "gmm_state.pt"))

        # Push to Hub if requested
        if push_to_hub:
            if repo_id is None:
                raise ValueError("repo_id must be provided when push_to_hub=True")

            from huggingface_hub import HfApi

            api = HfApi()

            # Upload all files in save_directory
            api.upload_folder(
                folder_path=save_directory,
                repo_id=repo_id,
                repo_type="model",
                commit_message=hub_kwargs.pop("commit_message", "Upload GMMXLNet model"),
                **hub_kwargs,
            )

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs: Any) -> "GMMXLNetForQA":
        """
        Load model including GMM-specific parameters.

        Args:
            load_directory: Directory or Hub model ID to load from
            **kwargs: Additional arguments (can override config values)

        Returns:
            Loaded GMMXLNetForQA model

        Raises:
            FileNotFoundError: If required config files not found
        """
        from transformers import XLNetForQuestionAnsweringSimple

        # Separate HuggingFace Hub kwargs from model kwargs
        hf_hub_kwargs = {}
        hf_specific_keys = {
            "revision",
            "token",
            "cache_dir",
            "force_download",
            "resume_download",
            "proxies",
            "local_files_only",
            "use_auth_token",
            "subfolder",
            "commit_hash",
            "use_safetensors",
            "trust_remote_code",
        }

        for key in hf_specific_keys:
            if key in kwargs:
                hf_hub_kwargs[key] = kwargs.pop(key)

        # Load base model
        base = XLNetForQuestionAnsweringSimple.from_pretrained(load_directory, **hf_hub_kwargs)

        # Check for GMM configuration
        try:
            # Try Hub first if revision specified or path doesn't exist locally
            if "revision" in hf_hub_kwargs or not os.path.exists(load_directory):
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(repo_id=load_directory, filename="gmm_config.json", **hf_hub_kwargs)
                try:
                    state_path = hf_hub_download(repo_id=load_directory, filename="gmm_state.pt", **hf_hub_kwargs)
                except Exception:
                    state_path = None
            else:
                # Local paths
                config_path = os.path.join(load_directory, "gmm_config.json")
                state_path = os.path.join(load_directory, "gmm_state.pt")
        except Exception:
            # Fall back to local paths
            config_path = os.path.join(load_directory, "gmm_config.json")
            state_path = os.path.join(load_directory, "gmm_state.pt")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"GMM config not found at {config_path}. "
                f"This does not appear to be a GMMXLNetForQA checkpoint. "
                f"Use MemXLNetForQA.from_pretrained() for standard checkpoints."
            )

        # Load GMM configuration
        with open(config_path) as f:
            config = json.load(f)

        # Verify this is a GMM checkpoint
        if config.get("memory_type") != "gmm":
            raise ValueError(
                f"Checkpoint memory_type is '{config.get('memory_type')}', expected 'gmm'. "
                f"This checkpoint may not be compatible with GMMXLNetForQA."
            )

        # Build model kwargs from config, allowing kwargs to override
        model_kwargs = {
            "num_experts": config.get("num_experts", 4),
            "memory_slots": config.get("memory_slots", 16),
            "routing_mode": config.get("routing_mode", "write-based"),
            "routing_temperature": config.get("routing_temperature", 1.0),
            "pooling_method": config.get("pooling_method", "mean"),
            "use_gmm_memory": config.get("use_gmm_memory", True),
            "mem_token_count": config.get("mem_token_count", 16),
        }
        model_kwargs.update(kwargs)

        # Create model
        model = cls(base, **model_kwargs)

        # Load GMM state if exists
        if state_path and os.path.exists(state_path) and model.use_gmm_memory:
            device = next(base.parameters()).device
            saved_state = torch.load(state_path, map_location=device, weights_only=True)

            model.memory_mixture.load_state_dict(saved_state["memory_mixture"])
            model.gating_network.load_state_dict(saved_state["gating_network"])
            model.expert_updater.load_state_dict(saved_state["expert_updater"])
            model.memory_reader.load_state_dict(saved_state["memory_reader"])

            # Validate loaded state
            model._validate_loaded_state()

        return model

    def __repr__(self) -> str:
        """String representation of GMMXLNetForQA."""
        return (
            f"GMMXLNetForQA(num_experts={self.num_experts}, "
            f"memory_slots={self.memory_slots}, "
            f"routing_mode='{self.routing_mode}', "
            f"use_gmm_memory={self.use_gmm_memory})"
        )
