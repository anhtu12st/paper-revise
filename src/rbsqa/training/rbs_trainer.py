"""
RBSQA Trainer for Reasoning Belief State Question Answering.

This module provides a specialized trainer for RBS-QA models that extends
the XLNetRecurrentTrainer with RBS-specific functionality including:
- Belief state tracking integration
- Halting policy management
- GMM backbone memory handling
- RBS-specific metrics collection
"""

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from memxlnet.training import XLNetRecurrentTrainer
from ..models import RBSXLNetForQA
from ..config import RBSTrainingConfig

logger = logging.getLogger(__name__)


class RBSTrainer(XLNetRecurrentTrainer):
    """
    Specialized trainer for RBS-QA models.

    Extends XLNetRecurrentTrainer with RBS-specific functionality:
    - RBS model initialization with proper configuration
    - GMM backbone memory handling (RBS extends GMM)
    - Belief state and halting policy integration
    - RBS-specific metrics collection and logging
    """

    def __init__(self, config: RBSTrainingConfig):
        """
        Initialize RBS trainer.

        Args:
            config: RBS training configuration with all model and training parameters
        """
        # Call parent constructor
        super().__init__(config)

        # Initialize RBS model if enabled
        if config.use_belief_state:
            self._initialize_rbs_model(config)

    def _initialize_rbs_model(self, config: RBSTrainingConfig) -> None:
        """Initialize RBS-XLNet model with proper configuration."""
        logger.info("🔧 Initializing RBS-XLNet model")

        self.model = RBSXLNetForQA(
            base_model_name=config.model_name,
            memory_num_tokens=16,  # Standard memory slots
            num_memory_experts=4,  # RBS requires GMM experts
            use_rbs_mode=True,
            belief_state_config={
                "max_segments": config.max_segments,
                "confidence_threshold": config.belief_state_threshold,
                "re_scoring_method": config.re_scoring_method,
                "enable_trend_analysis": config.enable_trend_analysis,
            },
            halting_config={
                "hidden_dim": config.halting_policy_hidden_dim,
                "num_layers": config.halting_policy_layers,
                "temperature": config.halting_temperature,
                "exploration_rate": config.halting_exploration_rate,
            },
        )

        logger.info(f"✅ RBSQA model created with belief state threshold {config.belief_state_threshold}")

        # Ensure model is on the correct device
        self.model = self.model.to(self.device)
        logger.info(f"✅ RBSQA model moved to device: {self.device}")

        # Validate model components are on correct device
        if hasattr(self.model, 'belief_state_tracker'):
            belief_device = next(self.model.belief_state_tracker.parameters()).device
            logger.info(f"✅ Belief state tracker device: {belief_device}")
        if hasattr(self.model, 'halting_policy'):
            halting_device = next(self.model.halting_policy.parameters()).device
            logger.info(f"✅ Halting policy device: {halting_device}")

        # Test initial memory creation (RBS uses GMM backbone)
        try:
            test_memory = self.model.gmm_backbone.get_initial_memory(1, self.device)
            logger.info(f"✅ Test initial memory created")
            logger.debug(f"  - Memory device: {next(iter(test_memory.values())).device}")
            logger.debug(f"  - Memory shape: {next(iter(test_memory.values())).shape}")
            logger.info("🔧 RBSQA model initialization validation complete")
        except Exception as e:
            logger.error(f"❌ RBS model initialization validation failed: {e}")
            raise

    def _process_document_batch_with_memory(self, time_step_batches: List, eval_mode: bool = False) -> float:
        """
        Override memory processing for RBS models.

        Args:
            time_step_batches: List of time-step batches for processing
            eval_mode: Whether in evaluation mode

        Returns:
            Computed loss for the document batch
        """
        if hasattr(self.model, 'use_rbs_mode') and self.model.use_rbs_mode:
            return self._process_rbs_memory_batch(time_step_batches, eval_mode)
        else:
            # Non-RBS model: use parent's memory processing
            return super()._process_document_batch_with_memory(time_step_batches, eval_mode)

    def _process_rbs_memory_batch(self, time_step_batches: List, eval_mode: bool) -> float:
        """Process memory batch for RBS models using GMM backbone format."""
        # RBS memory handling
        self.model.train() if not eval_mode else self.model.eval()

        # Initialize per-document memory bank
        if not hasattr(self, "memory_bank"):
            self.memory_bank: Dict[str, Dict[str, torch.Tensor]] = {}

        # Device placement validation
        if not eval_mode:
            logger.debug(f"RBS processing on device: {self.device}")
            logger.debug(f"Model device: {next(self.model.parameters()).device}")

        per_doc_logits_start = {}
        per_doc_logits_end = {}
        per_doc_labels_start = {}
        per_doc_labels_end = {}

        for time_step, batch in enumerate(time_step_batches):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            start_positions = batch["start_positions"].to(self.device)
            end_positions = batch["end_positions"].to(self.device)
            document_mask = batch["document_mask"].to(self.device)

            # Build RBS memory state per batch (RBS uses GMM format)
            memory_state_batch = self._build_rbs_memory_state(batch, document_mask)

            # Validate tensor shapes
            if not eval_mode:
                shapes = [(k, v.shape) for k, v in memory_state_batch.items()]
                logger.debug(f"RBSQA memory state shapes: {shapes}")
            else:
                logger.debug("Evaluation mode - skipping tensor shape validation")

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    memory_state=memory_state_batch,
                    mem_read_ids=self.mem_token_info.get("mem_read_ids") if self.mem_token_info else None,
                    mem_write_ids=self.mem_token_info.get("mem_write_ids") if self.mem_token_info else None,
                )
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    self._handle_device_placement_error(e, input_ids, attention_mask, start_positions,
                                                     end_positions, document_mask, memory_state_batch)
                else:
                    logger.error(f"RBS model runtime error: {e}")
                    logger.error(f"Input shapes: IDs={input_ids.shape}, Attention={attention_mask.shape}")
                    raise

            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
            new_memory_state = outputs["new_memory_state"]

            # Collect logits and labels per document
            self._collect_document_outputs(batch, document_mask, start_logits, end_logits,
                                         new_memory_state, per_doc_logits_start, per_doc_logits_end,
                                         per_doc_labels_start, per_doc_labels_end, eval_mode)

        # Compute loss using collected logits and labels
        return self._compute_document_loss(per_doc_logits_start, per_doc_logits_end,
                                          per_doc_labels_start, per_doc_labels_end)

    def _build_rbs_memory_state(self, batch: Dict, document_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build RBS memory state batch using GMM format."""
        memory_state_batch = {}
        for expert_idx in range(4):  # RBS requires 4 GMM experts
            expert_memories = []

            for ex_id, active in zip(batch["example_ids"], document_mask.tolist()):
                if not active:
                    # New document: use initial memory
                    initial_memory = self.model.gmm_backbone.get_initial_memory(1, self.device)
                    expert_memory = initial_memory[f"expert_{expert_idx}"]
                else:
                    # Existing document: get from memory bank
                    prev = self.memory_bank.get(ex_id)
                    if prev is None:
                        initial_memory = self.model.gmm_backbone.get_initial_memory(1, self.device)
                        expert_memory = initial_memory[f"expert_{expert_idx}"]
                    else:
                        expert_memory = prev[f"expert_{expert_idx}"]

                # Ensure consistent 2D tensor shape for stacking
                if expert_memory.dim() == 3:  # Shape: [1, memory_slots, hidden_dim]
                    expert_memory = expert_memory.squeeze(0)  # -> [memory_slots, hidden_dim]
                elif expert_memory.dim() == 2:  # Shape: [memory_slots, hidden_dim]
                    pass  # Already correct shape
                else:
                    raise ValueError(f"Unexpected expert memory shape: {expert_memory.shape}, expected 2D or 3D tensor")

                expert_memories.append(expert_memory)

            # Stack expert memories across batch
            memory_state_batch[f"expert_{expert_idx}"] = torch.stack(expert_memories, dim=0)

        return memory_state_batch

    def _handle_device_placement_error(self, error: RuntimeError, input_ids: torch.Tensor,
                                     attention_mask: torch.Tensor, start_positions: torch.Tensor,
                                     end_positions: torch.Tensor, document_mask: torch.Tensor,
                                     memory_state_batch: Dict[str, torch.Tensor]) -> None:
        """Handle device placement errors with detailed logging."""
        logger.error("=" * 80)
        logger.error("🚨 RBS DEVICE PLACEMENT ERROR DETECTED")
        logger.error("=" * 80)
        logger.error(f"Root error: {error}")
        logger.error(f"Target device: {self.device}")
        logger.error(f"Input IDs device: {input_ids.device} | Shape: {input_ids.shape}")
        logger.error(f"Attention mask device: {attention_mask.device} | Shape: {attention_mask.shape}")
        logger.error(f"Start positions device: {start_positions.device} | Shape: {start_positions.shape}")
        logger.error(f"End positions device: {end_positions.device} | Shape: {end_positions.shape}")
        logger.error(f"Document mask device: {document_mask.device} | Shape: {document_mask.shape}")

        logger.error("Memory state devices:")
        for expert_key, expert_tensor in memory_state_batch.items():
            logger.error(f"  - {expert_key}: {expert_tensor.device} | Shape: {expert_tensor.shape} | Dtype: {expert_tensor.dtype}")

        logger.error(f"Model parameters device: {next(self.model.parameters()).device}")

        # Check model subcomponents
        if hasattr(self.model, 'belief_state_tracker'):
            logger.error(f"Belief state tracker device: {next(self.model.belief_state_tracker.parameters()).device}")
        if hasattr(self.model, 'halting_policy'):
            logger.error(f"Halting policy device: {next(self.model.halting_policy.parameters()).device}")

        logger.error("=" * 80)
        raise RuntimeError(f"RBS device placement error: {error}. All tensors must be on the same device: {self.device}")

    def _collect_document_outputs(self, batch: Dict, document_mask: torch.Tensor,
                                start_logits: torch.Tensor, end_logits: torch.Tensor,
                                new_memory_state: Dict[str, torch.Tensor],
                                per_doc_logits_start: Dict, per_doc_logits_end: Dict,
                                per_doc_labels_start: Dict, per_doc_labels_end: Dict,
                                eval_mode: bool) -> None:
        """Collect outputs per document and update memory bank."""
        # RBS returns GMM format memory (dict with expert keys)
        if not isinstance(new_memory_state, dict):
            raise ValueError(f"Expected dict memory state from RBS model, got {type(new_memory_state)}")

        for ex_id, active in zip(batch["example_ids"], document_mask.tolist()):
            if active:
                # Get batch index for this document
                example_ids_list = batch["example_ids"].tolist() if hasattr(batch["example_ids"], 'tolist') else batch["example_ids"]
                ex_idx = example_ids_list.index(ex_id)

                # Extract individual memory for each expert and store in bank
                individual_memory = {}
                for expert_key, expert_tensor in new_memory_state.items():
                    if expert_tensor.dim() >= 2:
                        individual_memory[expert_key] = expert_tensor[ex_idx].detach()
                    else:
                        raise ValueError(f"Unexpected expert tensor shape: {expert_tensor.shape}")

                # Store in memory bank (RBS stores dict like GMM)
                self.memory_bank[ex_id] = individual_memory

                # Validate memory bank storage
                self._validate_memory_storage(ex_id, eval_mode)

                # Store logits for this document
                per_doc_logits_start.setdefault(ex_id, []).append(start_logits[ex_idx])
                per_doc_logits_end.setdefault(ex_id, []).append(end_logits[ex_idx])
                per_doc_labels_start.setdefault(ex_id, []).append(start_positions[ex_idx])
                per_doc_labels_end.setdefault(ex_id, []).append(end_positions[ex_idx])

    def _validate_memory_storage(self, ex_id: str, eval_mode: bool) -> None:
        """Validate memory bank storage and handle cleanup."""
        # Validate device consistency for dict memory
        for expert_key, expert_tensor in self.memory_bank[ex_id].items():
            if expert_tensor.device.type != self.device.type:
                logger.debug(f"Memory bank {ex_id} expert {expert_key} device mismatch: {expert_tensor.device} vs {self.device}")
                self.memory_bank[ex_id][expert_key] = expert_tensor.to(self.device)

        # Monitor memory bank size and implement cleanup
        memory_bank_size = len(self.memory_bank)

        if not eval_mode:
            if memory_bank_size > 1000:  # Prevent memory leaks
                logger.warning(f"Memory bank has grown to {memory_bank_size} documents, consider cleanup")
            if memory_bank_size % 100 == 0 and memory_bank_size > 0:
                logger.info(f"Memory bank size: {memory_bank_size} documents")

        # Automatic cleanup (more aggressive for long documents)
        MEMORY_BANK_LIMIT = 2000  # Configurable based on dataset
        if memory_bank_size > MEMORY_BANK_LIMIT:
            self._cleanup_memory_bank(memory_bank_size)

    def _cleanup_memory_bank(self, current_size: int) -> None:
        """Clean up memory bank to prevent OOM errors."""
        # Implement LRU-style cleanup: remove oldest 25% of documents
        docs_to_remove = list(self.memory_bank.keys())[:current_size // 4]
        removed_count = 0
        for doc_id in docs_to_remove:
            if doc_id in self.memory_bank:
                del self.memory_bank[doc_id]
                removed_count += 1

        logger.info(f"🧹 Automatic memory bank cleanup: removed {removed_count} documents "
                   f"(old size: {current_size}, new size: {len(self.memory_bank)})")

        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Log memory usage after cleanup
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            logger.info(f"🧹 Memory after cleanup: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB")

    def _compute_document_loss(self, per_doc_logits_start: Dict, per_doc_logits_end: Dict,
                              per_doc_labels_start: Dict, per_doc_labels_end: Dict) -> float:
        """Compute loss using collected logits and labels."""
        total_loss = 0.0
        total_examples = 0

        for ex_id in per_doc_logits_start:
            ex_logits_start = torch.stack(per_doc_logits_start[ex_id])
            ex_logits_end = torch.stack(per_doc_logits_end[ex_id])
            ex_labels_start = torch.stack(per_doc_labels_start[ex_id])
            ex_labels_end = torch.stack(per_doc_labels_end[ex_id])

            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(ex_logits_start, ex_labels_start)
            end_loss = loss_fct(ex_logits_end, ex_labels_end)
            total_loss += (start_loss + end_loss) / 2
            total_examples += 1

        avg_loss = total_loss / max(total_examples, 1)
        return avg_loss

    def train_one_document_batch(self, time_step_batches: List) -> float:
        """Override train_one_document_batch to handle RBS memory structure."""
        if hasattr(self.model, 'use_rbs_mode') and self.model.use_rbs_mode:
            return self._process_document_batch_with_memory(time_step_batches, eval_mode=False)
        else:
            return super().train_one_document_batch(time_step_batches)

    def _train_single_stage(self, train_dataloader, eval_dataloader, eval_dataset, stage_num: int):
        """Override _train_single_stage to add epoch cleanup for RBS models."""
        if hasattr(self.model, 'use_rbs_mode') and self.model.use_rbs_mode:
            # Call parent implementation
            from memxlnet.training.trainer import XLNetRecurrentTrainer
            parent_method = super()._train_single_stage
            result = parent_method(train_dataloader, eval_dataloader, eval_dataset, stage_num)
            return result
        else:
            return super()._train_single_stage(train_dataloader, eval_dataloader, eval_dataset, stage_num)

    def _clear_memory_bank(self) -> None:
        """Clear the memory bank to prevent memory leaks between epochs."""
        if hasattr(self, "memory_bank"):
            self.memory_bank.clear()
            logger.debug("Cleared RBSQA memory bank")

    def on_epoch_end(self) -> None:
        """Called at the end of each epoch to cleanup resources."""
        if hasattr(self.model, 'use_rbs_mode') and self.model.use_rbs_mode:
            self._clear_memory_bank()
            # Reset RBSQA model memory (uses GMM backbone)
            self.model.gmm_backbone.reset_memory()
            logger.debug("Reset RBSQA model memory at epoch end")

        # Call parent implementation if it exists
        if hasattr(super(), 'on_epoch_end'):
            super().on_epoch_end()

    def log_rbs_metrics(self, metrics: Dict) -> None:
        """Log RBS-specific metrics during training."""
        if hasattr(self.model, 'use_rbs_mode') and self.model.use_rbs_mode:
            # RBS-specific metrics would be collected here
            # For now, log basic RBS configuration info
            logger.info("📊 RBS Model Metrics:")
            logger.info(f"   - Belief state enabled: {self.model.use_rbs_mode}")
            logger.info(f"   - Number of GMM experts: {self.model.config.num_memory_experts}")
            logger.info(f"   - Memory slots per expert: {self.model.config.memory_num_tokens}")

            # Add RBS-specific metrics if available
            if "avg_reasoning_steps" in metrics:
                logger.info(f"   - Avg reasoning steps: {metrics['avg_reasoning_steps']:.2f}")
            if "halting_accuracy" in metrics:
                logger.info(f"   - Halting accuracy: {metrics['halting_accuracy']:.2f}%")
            if "belief_revision_rate" in metrics:
                logger.info(f"   - Belief revision rate: {metrics['belief_revision_rate']:.2f}%")