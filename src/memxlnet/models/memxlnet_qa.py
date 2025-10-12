import os
from typing import Any

import torch
import torch.nn as nn

# Import enhanced memory modules if available
try:
    from .memory_modules import MemoryController

    DIFFERENTIABLE_MEMORY_AVAILABLE = True
except ImportError:
    DIFFERENTIABLE_MEMORY_AVAILABLE = False


class MemXLNetForQA(nn.Module):
    """Memory-augmented XLNet wrapper for Question Answering with recurrent memory capabilities.

    This implementation provides:
    - Learned or zero-initialized memory states
    - Gated memory updates based on memory tokens
    - Memory state propagation across document segments
    """

    def __init__(
        self,
        base_model: nn.Module,
        mem_token_count: int = 0,
        memory_init: str = "learned",
        memory_update: str = "gated",
        memory_dim: int | None = None,
        # New parameters for enhanced memory
        use_differentiable_memory: bool = False,
        num_memory_heads: int = 1,
        memory_sharpness: float = 1.0,
        enable_usage_tracking: bool = False,
        enable_temporal_links: bool = False,
        memory_slots: int | None = None,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.mem_token_count = mem_token_count
        self.memory_init = memory_init
        self.memory_update = memory_update

        # Enhanced memory parameters
        self.use_differentiable_memory = use_differentiable_memory
        self.num_memory_heads = num_memory_heads
        self.memory_sharpness = memory_sharpness
        self.enable_usage_tracking = enable_usage_tracking
        self.enable_temporal_links = enable_temporal_links

        # Determine memory dimension from model's hidden size
        if memory_dim is None:
            if hasattr(base_model, "config") and hasattr(base_model.config, "d_model"):
                d_model_val = base_model.config.d_model
                memory_dim_value: int = int(d_model_val) if isinstance(d_model_val, (int, float)) else 768
            elif hasattr(base_model, "config") and hasattr(base_model.config, "hidden_size"):
                hidden_size_val = base_model.config.hidden_size
                memory_dim_value = int(hidden_size_val) if isinstance(hidden_size_val, (int, float)) else 768
            else:
                memory_dim_value = 768  # Default for base models
            memory_dim = memory_dim_value

        self.memory_dim: int = memory_dim

        # Set memory slots (default to mem_token_count if not specified)
        if memory_slots is None:
            memory_slots = max(mem_token_count, 16)  # Minimum 16 slots for differentiable memory
        self.memory_slots: int = memory_slots

        # Initialize learnable memory parameters if needed
        if self.mem_token_count > 0 and self.memory_init == "learned":
            self.learned_memory = nn.Parameter(torch.randn(self.mem_token_count, self.memory_dim) * 0.02)

        # Initialize differentiable memory controller if requested
        self.memory_controller = None
        if use_differentiable_memory and DIFFERENTIABLE_MEMORY_AVAILABLE:
            self.memory_controller = MemoryController(
                input_dim=self.memory_dim,
                memory_slots=self.memory_slots,
                memory_dim=self.memory_dim,
                num_heads=num_memory_heads,
                use_temporal_links=enable_temporal_links,
                use_usage_tracking=enable_usage_tracking,
                sharpness=memory_sharpness,
            )
        elif use_differentiable_memory and not DIFFERENTIABLE_MEMORY_AVAILABLE:
            import warnings

            warnings.warn(
                "Differentiable memory requested but memory_modules not available. Falling back to token-based memory.",
                RuntimeWarning,
            )

        # Memory update components for gated updates (token-based)
        if self.mem_token_count > 0 and self.memory_update == "gated" and not use_differentiable_memory:
            self.memory_gate = nn.Linear(self.memory_dim * 2, self.memory_dim)
            self.memory_update_net = nn.Linear(self.memory_dim * 2, self.memory_dim)

    def get_initial_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize memory states for a batch.

        Args:
            batch_size: Number of examples in the batch
            device: Target device for memory tensors

        Returns:
            Memory state tensor of shape (batch_size, mem_token_count or memory_slots, memory_dim)
        """
        # If using differentiable memory, return shape matching memory_slots
        if self.use_differentiable_memory and self.memory_controller is not None:
            # Return zeros with memory_slots dimension
            return torch.zeros(batch_size, self.memory_slots, self.memory_dim, device=device)

        if self.mem_token_count == 0:
            # Return empty tensor if no memory tokens
            return torch.zeros(batch_size, 0, self.memory_dim, device=device)

        if self.memory_init == "learned":
            # Expand learned memory for batch
            memory = self.learned_memory.unsqueeze(0).expand(batch_size, self.mem_token_count, self.memory_dim)
        else:  # "zeros"
            memory = torch.zeros(batch_size, self.mem_token_count, self.memory_dim, device=device)

        return memory

    def _extract_memory_representations(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        mem_read_ids: list[int] | None = None,
        mem_write_ids: list[int] | None = None,
    ) -> torch.Tensor | None:
        """Extract memory token representations from hidden states.

        Args:
            input_ids: Input token ids (batch_size, seq_len)
            hidden_states: Model hidden states (batch_size, seq_len, hidden_dim)
            mem_read_ids: List of memory read token ids
            mem_write_ids: List of memory write token ids

        Returns:
            Memory representations (batch_size, mem_token_count, hidden_dim) or None
        """
        if self.mem_token_count == 0 or mem_write_ids is None:
            return None

        batch_size, seq_len = input_ids.shape
        memory_reps = []

        # Extract representations for each memory write token
        for mem_id in mem_write_ids:
            # Find positions of this memory token in the batch
            positions = (input_ids == mem_id).nonzero(as_tuple=True)

            if len(positions[0]) > 0:
                # Extract hidden states at memory token positions
                batch_indices = positions[0]
                seq_indices = positions[1]
                mem_rep = hidden_states[batch_indices, seq_indices]  # (found_count, hidden_dim)

                # Average across multiple occurrences per example if needed
                batch_mem_reps = []
                for b in range(batch_size):
                    batch_mask = batch_indices == b
                    if batch_mask.any():
                        batch_mem_reps.append(mem_rep[batch_mask].mean(dim=0))
                    else:
                        # No memory token found for this batch element
                        batch_mem_reps.append(torch.zeros(hidden_states.size(-1), device=hidden_states.device))

                memory_reps.append(torch.stack(batch_mem_reps))
            else:
                # No memory tokens found, use zeros
                memory_reps.append(torch.zeros(batch_size, hidden_states.size(-1), device=hidden_states.device))

        if memory_reps:
            return torch.stack(memory_reps, dim=1)  # (batch_size, mem_token_count, hidden_dim)
        return None

    def _update_memory(self, current_memory: torch.Tensor, new_representations: torch.Tensor) -> torch.Tensor:
        """Update memory states using gating mechanism.

        Args:
            current_memory: Current memory state (batch_size, mem_token_count, memory_dim)
            new_representations: New memory representations (batch_size, mem_token_count, memory_dim)

        Returns:
            Updated memory state (batch_size, mem_token_count, memory_dim)
        """
        if self.memory_update == "none":
            return current_memory
        elif self.memory_update == "gated":
            # Concatenate current and new representations
            combined = torch.cat([current_memory, new_representations], dim=-1)

            # Compute gate and update values
            gate = torch.sigmoid(self.memory_gate(combined))
            update = torch.tanh(self.memory_update_net(combined))

            # Apply gated update: new = gate * update + (1 - gate) * current
            new_memory: torch.Tensor = gate * update + (1 - gate) * current_memory
            return new_memory
        else:
            # Default: simple replacement
            return new_representations

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
        mem_read_ids: list[int] | None = None,
        mem_write_ids: list[int] | None = None,
        differentiable_memory_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None | dict[str, Any]]:
        """Forward pass with memory state processing.

        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            token_type_ids: Token type ids
            start_positions: Start positions for QA (training)
            end_positions: End positions for QA (training)
            memory_state: Current memory state
            mem_read_ids: Memory read token ids
            mem_write_ids: Memory write token ids
            differentiable_memory_info: Info for differentiable memory (optional)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - loss: Training loss (if positions provided)
                - start_logits: Start position logits
                - end_logits: End position logits
                - new_memory_state: Updated memory state
                - memory_info: Additional memory info (if using differentiable memory)
        """
        # Handle memory token injection if memory is enabled
        modified_input_ids = input_ids
        modified_attention_mask = attention_mask
        modified_token_type_ids = token_type_ids

        # If memory tokens are enabled and we have memory state, inject memory tokens
        if self.mem_token_count > 0 and memory_state is not None and mem_read_ids is not None and len(mem_read_ids) > 0:
            # For simplicity, we assume memory tokens are already embedded in input_ids
            # A more sophisticated implementation would dynamically inject them
            pass

        # Forward pass through base model
        outputs = self.base(
            input_ids=modified_input_ids,
            attention_mask=modified_attention_mask,
            token_type_ids=modified_token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=True,
            output_hidden_states=True,  # Need hidden states for memory extraction
            **kwargs,
        )

        # Process with differentiable memory if enabled
        memory_info = {}
        if self.memory_controller is not None and self.use_differentiable_memory:
            # Use differentiable memory controller
            hidden_states = (
                outputs.hidden_states[-1]
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None
                else None
            )

            if hidden_states is not None:
                # Use CLS token representation as input to memory controller
                batch_size = hidden_states.size(0)
                cls_positions = []

                # XLNet uses CLS token at the end, get the CLS token ID from tokenizer if available
                cls_token_id = getattr(self.base.config, "cls_token_id", None)
                if cls_token_id is None:
                    # Try to get from tokenizer if attached, otherwise use a common approach
                    # In XLNet, CLS is typically at the end of the sequence
                    for b in range(batch_size):
                        # Use the last non-padding token position
                        seq_len = modified_input_ids.size(1)
                        cls_positions.append(seq_len - 1)
                else:
                    for b in range(batch_size):
                        cls_pos = (modified_input_ids[b] == cls_token_id).nonzero(as_tuple=True)[0]
                        if len(cls_pos) > 0:
                            cls_positions.append(cls_pos[0].item())
                        else:
                            cls_positions.append(modified_input_ids.size(1) - 1)  # Fallback to last position

                # Extract CLS representations
                cls_reps = hidden_states[range(batch_size), cls_positions]

                # Process through memory controller
                memory_output, memory_info = self.memory_controller(
                    cls_reps,
                    prev_read=differentiable_memory_info.get("prev_read") if differentiable_memory_info else None,
                    prev_weights=differentiable_memory_info.get("prev_weights") if differentiable_memory_info else None,
                )

                # The memory_output can be used to modulate the final predictions
                # For now, we keep the original logits but store memory info
                # Get memory state and expand to batch dimension
                memory_state_unbatched = self.memory_controller.get_memory_state()  # [num_slots, memory_dim]
                # Expand to batch dimension - replicate for each batch element
                new_memory_state: torch.Tensor | None = (
                    memory_state_unbatched.unsqueeze(0).expand(batch_size, -1, -1).clone()
                )
                # Shape: [batch_size, memory_slots, memory_dim]
            else:
                new_memory_state = memory_state
        else:
            # Use token-based memory (original implementation)
            new_memory_state = memory_state
            if self.mem_token_count > 0 and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                # Use the last layer's hidden states
                hidden_states = outputs.hidden_states[-1]

                # Extract new memory representations
                new_memory_reps = self._extract_memory_representations(
                    modified_input_ids, hidden_states, mem_read_ids, mem_write_ids
                )

                # Update memory state if we extracted new representations
                if new_memory_reps is not None and memory_state is not None:
                    new_memory_state = self._update_memory(memory_state, new_memory_reps)
                elif new_memory_reps is not None:
                    # If no previous memory state, use new representations
                    new_memory_state = new_memory_reps

        # If no memory state provided, initialize it
        if new_memory_state is None and self.mem_token_count > 0:
            batch_size = input_ids.size(0)
            new_memory_state = self.get_initial_memory(batch_size, input_ids.device)

        result = {
            "loss": getattr(outputs, "loss", None),
            "start_logits": getattr(outputs, "start_logits", None),
            "end_logits": getattr(outputs, "end_logits", None),
            "new_memory_state": new_memory_state,
        }

        # Add memory info if using differentiable memory
        if memory_info:
            result["memory_info"] = memory_info

        return result

    # Hugging Face-like API
    def save_pretrained(self, save_directory: str):
        """Save the model including memory-specific parameters."""
        os.makedirs(save_directory, exist_ok=True)

        # Save base model
        if hasattr(self.base, "save_pretrained") and callable(getattr(self.base, "save_pretrained", None)):
            self.base.save_pretrained(save_directory)  # type: ignore[operator]

        # Save memory wrapper configuration
        config = {
            "wrapper_class": "MemXLNetForQA",
            "mem_token_count": self.mem_token_count,
            "memory_update": self.memory_update,
            "memory_init": self.memory_init,
            "use_differentiable_memory": self.use_differentiable_memory,
            "num_memory_heads": self.num_memory_heads,
            "memory_sharpness": self.memory_sharpness,
            "enable_usage_tracking": self.enable_usage_tracking,
            "enable_temporal_links": self.enable_temporal_links,
            "memory_slots": self.memory_slots,
            "version": 3,  # Updated version for enhanced memory
        }
        with open(os.path.join(save_directory, "memxlnet_config.json"), "w") as f:
            import json

            json.dump(config, f, indent=2)

        # Save memory-specific state
        if self.mem_token_count > 0:
            state_dict: dict[str, Any] = {}

            # Save learned memory if it exists
            if hasattr(self, "learned_memory"):
                state_dict["learned_memory"] = self.learned_memory.data

            # Save memory update components if they exist
            if hasattr(self, "memory_gate"):
                state_dict["memory_gate"] = self.memory_gate.state_dict()

            if hasattr(self, "memory_update_net"):
                state_dict["memory_update_net"] = self.memory_update_net.state_dict()

            # Save differentiable memory controller if exists
            if hasattr(self, "memory_controller") and self.memory_controller is not None:
                state_dict["memory_controller"] = self.memory_controller.state_dict()

            torch.save(state_dict, os.path.join(save_directory, "memxlnet_state.pt"))

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str | None = None,
        private: bool = False,
        token: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Upload the model to the Hugging Face Hub.

        Args:
            repo_id: Repository ID on the Hub (e.g., "username/model-name")
            commit_message: Commit message for the upload
            private: Whether to create a private repository
            token: Hugging Face authentication token (optional, uses cached token if not provided)
            **kwargs: Additional arguments passed to upload_folder

        Returns:
            URL of the uploaded model repository
        """
        import tempfile
        from shutil import rmtree

        from huggingface_hub import HfApi

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Save model to temporary directory
            self.save_pretrained(temp_dir)

            # Initialize HF API
            api = HfApi(token=token)

            # Create repository if it doesn't exist
            api.create_repo(repo_id=repo_id, exist_ok=True, private=private, token=token)

            # Upload the folder
            commit_message = commit_message or "Upload MemXLNetForQA model"
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_id,
                commit_message=commit_message,
                token=token,
                **kwargs,
            )

            # Return the repository URL
            repo_url = f"https://huggingface.co/{repo_id}"
            return repo_url

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                rmtree(temp_dir)

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load model including memory-specific parameters."""
        import json

        from transformers import XLNetForQuestionAnsweringSimple

        # Load base model
        base = XLNetForQuestionAnsweringSimple.from_pretrained(load_directory)

        # Check for memory configuration
        config_path = os.path.join(load_directory, "memxlnet_config.json")
        state_path = os.path.join(load_directory, "memxlnet_state.pt")

        if os.path.exists(config_path):
            # Load wrapper configuration
            with open(config_path) as f:
                config = json.load(f)

            # Use saved config but allow kwargs to override
            wrapper_kwargs = {
                "mem_token_count": config.get("mem_token_count", 0),
                "memory_update": config.get("memory_update", "gated"),
                "memory_init": config.get("memory_init", "learned"),
                "use_differentiable_memory": config.get("use_differentiable_memory", False),
                "num_memory_heads": config.get("num_memory_heads", 1),
                "memory_sharpness": config.get("memory_sharpness", 1.0),
                "enable_usage_tracking": config.get("enable_usage_tracking", False),
                "enable_temporal_links": config.get("enable_temporal_links", False),
                "memory_slots": config.get("memory_slots", None),
            }
            wrapper_kwargs.update(kwargs)

            # Create wrapper
            wrapper = cls(base, **wrapper_kwargs)

            # Load memory state if exists
            if os.path.exists(state_path) and wrapper.mem_token_count > 0:
                device = next(base.parameters()).device
                saved_state = torch.load(state_path, map_location=device, weights_only=True)

                # Handle backward compatibility with old format
                if "version" in config and config["version"] == 1:
                    # Old format compatibility
                    if "init_memory" in saved_state and hasattr(wrapper, "learned_memory"):
                        wrapper.learned_memory.data.copy_(saved_state["init_memory"])

                    # Handle old gating mechanism
                    if "gate" in saved_state and hasattr(wrapper, "memory_gate"):
                        gate_state = saved_state["gate"]
                        if isinstance(gate_state, dict) and "weight" in gate_state:
                            # Old format had single gate, map to new memory_gate
                            wrapper.memory_gate.load_state_dict(gate_state)

                            # Initialize memory_update_net with similar weights if it exists
                            if hasattr(wrapper, "memory_update_net"):
                                with torch.no_grad():
                                    wrapper.memory_update_net.weight.data.copy_(gate_state["weight"])
                                    if "bias" in gate_state:
                                        wrapper.memory_update_net.bias.data.copy_(gate_state["bias"])
                else:
                    # New format - direct loading
                    if "learned_memory" in saved_state and hasattr(wrapper, "learned_memory"):
                        wrapper.learned_memory.data.copy_(saved_state["learned_memory"])

                    if "memory_gate" in saved_state and hasattr(wrapper, "memory_gate"):
                        wrapper.memory_gate.load_state_dict(saved_state["memory_gate"])

                    if "memory_update_net" in saved_state and hasattr(wrapper, "memory_update_net"):
                        wrapper.memory_update_net.load_state_dict(saved_state["memory_update_net"])

                    # Load differentiable memory controller if exists
                    if (
                        "memory_controller" in saved_state
                        and hasattr(wrapper, "memory_controller")
                        and wrapper.memory_controller is not None
                    ):
                        wrapper.memory_controller.load_state_dict(saved_state["memory_controller"])

            return wrapper
        else:
            # No memory config found, create basic wrapper
            return cls(base, **kwargs)
