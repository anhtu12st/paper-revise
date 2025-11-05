#!/usr/bin/env python3
"""
Standard SQuAD v2 - GMM Main (4 Experts, Multi-Expert Memory)
==============================================================

Dataset: squad_v2 (standard)
Max segments: 2
Memory: GMM with 4 experts (multi-expert implementation)

Purpose: Main result on standard benchmark with GMM multi-expert memory.

This is the GMM equivalent of 02b_main_squad_8tokens_differentiable.py that uses
Gated Memory Mixture with 4 experts instead of differentiable memory.

Key differences from 02b_main_squad_8tokens_differentiable.py:
- use_gmm_memory=True instead of use_differentiable_memory=True
- Added GMM parameters: num_memory_experts=4, routing_temperature=1.0
- GMM-specific load balancing and routing regularization
- Multi-expert memory with learned routing

Output: outputs/paper_v2_squad_gmm_main_4experts/
"""

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from gmmxlnet.models import GMMXLNetForQA
from gmmxlnet.training import gmm_balanced_config
from memxlnet.training import XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    return gmm_balanced_config(
        # Base XLNet configuration
        model_name="xlnet-base-cased",
        max_seq_length=384,
        doc_stride=64,
        dataset_name="squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=1000,
        max_eval_samples=200,
        use_lazy_loading=True,
        # Chunked dataset settings (FAST: 2-5 min vs 30-60 min preprocessing)
        use_chunked_dataset=True,
        chunked_dataset_dir="./preprocessed_data/squad_v2",
        chunked_load_mode="streaming",  # Memory-efficient streaming
        progressive_segments=[2],
        max_n_segs=2,
        # Memory configuration
        memory_num_tokens=16,  # ✅ GMM memory requires slots for each expert (16 per expert)
        memory_update="gated",
        memory_init="learned",
        # ✅ GMM: Use multi-expert memory instead of single differentiable memory
        use_gmm_memory=True,
        num_memory_experts=4,
        routing_temperature=1.0,
        routing_mode="write-based",
        load_balance_weight=0.01,
        entropy_regularization_weight=0.0,
        # Global softmax and training settings
        use_global_softmax=False,
        num_epochs=3,
        train_batch_size=4,
        eval_batch_size=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        eval_steps=1000,
        save_steps=10000,
        logging_steps=500,
        output_dir="./outputs/paper_v2_squad_gmm_main_4experts",
        run_name="paper-v2-squad-gmm-main-4experts",
        save_total_limit=3,
        no_answer_threshold=1.5,
        use_any_positive_logic=True,
        device=device,
        fp16=has_cuda,
        # ✅ Enable global softmax immediately (consistent with original)
        warmup_freeze_base_epochs=0,
        warmup_disable_global_softmax_epochs=0,
        warmup_disable_any_positive_epochs=0,
        push_to_hub_on_save=False,
    )


def main():
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT GMM: SQUAD V2 - MAIN (4 EXPERTS, MULTI-EXPERT MEMORY)")
    print("=" * 80 + "\n")
    print("🔧 Configuration details:")
    print("   ✅ memory_impl: 'differentiable' → 'gmm_multi_expert'")
    print("   ✅ use_gmm_memory: False → True")
    print("   ✅ Added: num_memory_experts=4, routing_temperature=1.0")
    print("   ✅ Added: load_balance_weight=0.01 for expert balancing")
    print("   ✅ Memory: 4 independent expert memories with learned routing")
    print()
    print("Expected outcome: F1 > 0% (should achieve ~65-80% F1)")
    print("                  Multi-expert specialization should improve performance")
    print()

    config = create_config()

    print("🔧 GMM Configuration Summary:")
    print(f"   - Number of experts: {config.num_memory_experts}")
    print(f"   - Routing temperature: {config.routing_temperature}")
    print(f"   - Routing mode: {config.routing_mode}")
    print(f"   - Load balance weight: {config.load_balance_weight}")
    print(f"   - Memory slots per expert: {config.memory_num_tokens}")
    print(f"   - Expert initialization: {config.expert_init_strategies}")
    print()

    # Create a custom GMM trainer using the existing infrastructure
    class GMMTrainer(XLNetRecurrentTrainer):
        """Extended trainer that supports GMM memory."""

        def __init__(self, config):
            # Call parent constructor
            super().__init__(config)

            # Override model to use GMM if enabled
            if config.use_gmm_memory:
                logger.info("🔧 Initializing GMM-XLNet model")
                self.model = GMMXLNetForQA(
                    base_model=self.base_model,
                    num_experts=config.num_memory_experts,
                    memory_slots=16,  # Standard memory slots
                    routing_mode=config.routing_mode,
                    routing_temperature=config.routing_temperature,
                    use_gmm_memory=True,
                )
                logger.info(f"✅ GMM model created with {config.num_memory_experts} experts")

                # Ensure GMM model is on the correct device
                self.model = self.model.to(self.device)
                logger.info(f"✅ GMM model moved to device: {self.device}")

                # Validate GMM model components are on correct device
                if hasattr(self.model, 'memory_mixture'):
                    mixture_device = next(self.model.memory_mixture.parameters()).device
                    logger.info(f"✅ Memory mixture device: {mixture_device}")
                if hasattr(self.model, 'gating_network'):
                    gating_device = next(self.model.gating_network.parameters()).device
                    logger.info(f"✅ Gating network device: {gating_device}")

                # Test initial memory creation
                test_memory = self.model.get_initial_memory(1, self.device)
                logger.info(f"✅ Test initial memory created for {len(test_memory)} experts")
                for expert_key, expert_tensor in test_memory.items():
                    logger.debug(f"  - {expert_key}: device={expert_tensor.device}, shape={expert_tensor.shape}")
                logger.info("🔧 GMM model initialization validation complete")

        def _process_document_batch_with_memory(self, time_step_batches, eval_mode=False):
            """Override memory processing for GMM models."""
            if hasattr(self.model, 'use_gmm_memory') and self.model.use_gmm_memory:
                # GMM memory handling
                self.model.train() if not eval_mode else self.model.eval()

                # Initialize per-document memory bank
                if not hasattr(self, "memory_bank"):
                    self.memory_bank: dict[str, dict[str, torch.Tensor]] = {}

                # Device placement validation
                if not eval_mode:
                    logger.debug(f"GMM processing on device: {self.device}")
                    logger.debug(f"Model device: {next(self.model.parameters()).device}")

                per_doc_logits_start = {}
                per_doc_logits_end = {}
                per_doc_labels_start = {}
                per_doc_labels_end = {}

                def normalize_device(device_str):
                    """Normalize device strings to handle cuda vs cuda:0 equivalency"""
                    if isinstance(device_str, torch.device):
                        device_str = str(device_str)
                    if device_str == "cuda":
                        return "cuda:0"  # Normalize cuda to cuda:0
                    return device_str

                for time_step, batch in enumerate(time_step_batches):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    token_type_ids = batch["token_type_ids"].to(self.device)
                    start_positions = batch["start_positions"].to(self.device)
                    end_positions = batch["end_positions"].to(self.device)
                    document_mask = batch["document_mask"].to(self.device)

                    # Build GMM memory state dict per batch ordering with device consistency
                    memory_state_batch = {}
                    for expert_idx in range(self.model.num_experts):
                        expert_memories = []

                        for ex_id, active in zip(batch["example_ids"], document_mask.tolist()):
                            if not active:
                                # New document: use initial memory (already on correct device)
                                initial_memory = self.model.get_initial_memory(1, self.device)
                                expert_memory = initial_memory[f"expert_{expert_idx}"]
                            else:
                                # Existing document: get from memory bank
                                prev = self.memory_bank.get(ex_id)
                                if prev is None:
                                    initial_memory = self.model.get_initial_memory(1, self.device)
                                    expert_memory = initial_memory[f"expert_{expert_idx}"]
                                else:
                                    expert_memory = prev[f"expert_{expert_idx}"].to(self.device)

                            # Ensure consistent tensor shape: remove batch dimension if present
                            if expert_memory.dim() == 3:  # Shape: [1, memory_slots, hidden_dim]
                                expert_memory = expert_memory.squeeze(0)  # -> [memory_slots, hidden_dim]
                            elif expert_memory.dim() == 2:  # Shape: [memory_slots, hidden_dim]
                                pass  # Already correct shape
                            else:
                                raise ValueError(f"Unexpected expert memory shape: {expert_memory.shape}, expected 2D or 3D tensor")

                            expert_memories.append(expert_memory)

                        # Validate all expert memories have the same shape before stacking
                        if expert_memories:
                            expected_shape = expert_memories[0].shape
                            for i, mem in enumerate(expert_memories):
                                if mem.shape != expected_shape:
                                    raise ValueError(f"Expert {expert_idx} memory {i} has shape {mem.shape}, expected {expected_shape}")

                            # Stack expert memories across batch (already on correct device)
                            memory_state_batch[f"expert_{expert_idx}"] = torch.stack(expert_memories, dim=0)
                        else:
                            # Handle empty batch case - create empty tensor with correct shape
                            # Use memory_slots=16, hidden_dim=768 as defaults based on GMM model
                            memory_state_batch[f"expert_{expert_idx}"] = torch.empty(0, 16, 768, device=self.device)

                    # Validate device consistency and tensor shapes before forward pass
                    if not eval_mode:
                        current_device_normalized = normalize_device(self.device)

                        # Validate device consistency
                        for expert_key, expert_tensor in memory_state_batch.items():
                            tensor_device_normalized = normalize_device(expert_tensor.device)
                            if tensor_device_normalized != current_device_normalized:
                                logger.warning(f"Expert {expert_key} device mismatch: {expert_tensor.device} vs {self.device} (normalized: {tensor_device_normalized} vs {current_device_normalized})")
                                memory_state_batch[expert_key] = expert_tensor.to(self.device)
                                logger.debug(f"Fixed {expert_key} device placement to {self.device}")

                        # Validate tensor shapes are consistent across experts
                        if memory_state_batch:
                            shapes = [(k, v.shape) for k, v in memory_state_batch.items()]
                            logger.debug(f"GMM memory state shapes: {shapes}")

                            # Check all experts have the same shape
                            first_shape = None
                            for expert_key, expert_tensor in memory_state_batch.items():
                                if first_shape is None:
                                    first_shape = expert_tensor.shape
                                elif expert_tensor.shape != first_shape:
                                    logger.error(f"Shape mismatch detected: {expert_key} has shape {expert_tensor.shape}, expected {first_shape}")
                                    raise ValueError(f"Expert tensor shapes are inconsistent: {expert_key}: {expert_tensor.shape} vs expected: {first_shape}")

                            logger.debug(f"✅ All expert tensors have consistent shape: {first_shape}")
                        else:
                            logger.warning("Memory state batch is empty - no tensors to validate")

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
                            logger.error("=" * 80)
                            logger.error("🚨 GMM DEVICE PLACEMENT ERROR DETECTED")
                            logger.error("=" * 80)
                            logger.error(f"Root error: {e}")
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
                            if hasattr(self.model, 'memory_mixture'):
                                logger.error(f"Memory mixture device: {next(self.model.memory_mixture.parameters()).device}")
                            if hasattr(self.model, 'gating_network'):
                                logger.error(f"Gating network device: {next(self.model.gating_network.parameters()).device}")

                            logger.error("=" * 80)
                            raise RuntimeError(f"GMM device placement error: {e}. All tensors must be on the same device: {self.device}")
                        else:
                            logger.error(f"GMM model runtime error: {e}")
                            logger.error(f"Input shapes: IDs={input_ids.shape}, Attention={attention_mask.shape}")
                            raise

                    start_logits = outputs["start_logits"]
                    end_logits = outputs["end_logits"]
                    new_memory_state = outputs["new_memory_state"]

                    # Collect logits and labels per document
                    for ex_id, active in zip(batch["example_ids"], document_mask.tolist()):
                        if active:
                            # Update memory bank for next time step (ensure on correct device)
                            self.memory_bank[ex_id] = {
                                expert_key: tensor.detach().to(self.device)  # Use detach() to avoid gradient storage issues
                                for expert_key, tensor in new_memory_state.items()
                            }

                            # Validate memory bank storage consistency
                            if not eval_mode:
                                for expert_key, expert_tensor in self.memory_bank[ex_id].items():
                                    # Use device normalization to handle cuda vs cuda:0 equivalency
                                    tensor_device_normalized = normalize_device(expert_tensor.device)
                                    target_device_normalized = normalize_device(self.device)
                                    if tensor_device_normalized != target_device_normalized:
                                        logger.warning(f"Memory bank {ex_id} expert {expert_key} device mismatch: {expert_tensor.device} vs {self.device} (normalized: {tensor_device_normalized} vs {target_device_normalized})")
                                        self.memory_bank[ex_id][expert_key] = expert_tensor.to(self.device)

                                # Monitor memory bank size
                                if len(self.memory_bank) > 1000:  # Prevent memory leaks
                                    logger.warning(f"Memory bank has grown to {len(self.memory_bank)} documents, consider cleanup")
                                if len(self.memory_bank) % 100 == 0 and len(self.memory_bank) > 0:
                                    logger.info(f"Memory bank size: {len(self.memory_bank)} documents")

                            # Store logits for this document (use first segment for simplicity)
                            # Handle both tensor and list inputs for batch["example_ids"]
                            example_ids_list = batch["example_ids"].tolist() if hasattr(batch["example_ids"], 'tolist') else batch["example_ids"]
                            ex_idx = example_ids_list.index(ex_id)
                            per_doc_logits_start.setdefault(ex_id, []).append(start_logits[ex_idx])
                            per_doc_logits_end.setdefault(ex_id, []).append(end_logits[ex_idx])
                            per_doc_labels_start.setdefault(ex_id, []).append(start_positions[ex_idx])
                            per_doc_labels_end.setdefault(ex_id, []).append(end_positions[ex_idx])

                # Compute loss using collected logits and labels
                total_loss = 0.0
                total_examples = 0

                for ex_id in per_doc_logits_start:
                    ex_logits_start = torch.stack(per_doc_logits_start[ex_id])
                    ex_logits_end = torch.stack(per_doc_logits_end[ex_id])
                    ex_labels_start = torch.stack(per_doc_labels_start[ex_id])
                    ex_labels_end = torch.stack(per_doc_labels_end[ex_id])

                    loss_fct = torch.nn.CrossEntropyLoss()
                    start_loss = loss_fct(ex_logits_start, ex_labels_start)
                    end_loss = loss_fct(ex_logits_end, ex_labels_end)
                    total_loss += (start_loss + end_loss) / 2
                    total_examples += 1

                avg_loss = total_loss / max(total_examples, 1)
                return avg_loss
            else:
                # Non-GMM model: use parent's memory processing
                return super()._process_document_batch_with_memory(time_step_batches, eval_mode)

        def train_one_document_batch(self, time_step_batches):
            """Override train_one_document_batch to handle GMM memory structure."""
            if hasattr(self.model, 'use_gmm_memory') and self.model.use_gmm_memory:
                return self._process_document_batch_with_memory(time_step_batches, eval_mode=False)
            else:
                # Non-GMM model: use parent's implementation
                return super().train_one_document_batch(time_step_batches)

        def _clear_memory_bank(self):
            """Clear the memory bank to prevent memory leaks between epochs."""
            if hasattr(self, 'memory_bank'):
                self.memory_bank.clear()
                logger.debug("Cleared GMM memory bank")

        def on_epoch_end(self):
            """Called at the end of each epoch to cleanup resources."""
            if hasattr(self.model, 'use_gmm_memory') and self.model.use_gmm_memory:
                self._clear_memory_bank()
                # Reset GMM model memory
                self.model.reset_memory()
                logger.debug("Reset GMM model memory at epoch end")

            # Call parent implementation if it exists
            if hasattr(super(), 'on_epoch_end'):
                super().on_epoch_end()

    trainer = GMMTrainer(config)

    try:
        trainer.train()

        # Print final metrics if available
        if hasattr(trainer, "best_metrics"):
            metrics = trainer.best_metrics
            print("\n" + "=" * 80)
            print("📊 FINAL RESULTS")
            print("=" * 80)
            print(f"   F1 Score: {metrics.get('f1', 0.0):.2f}%")
            print(f"   Exact Match: {metrics.get('exact_match', 0.0):.2f}%")
            if "has_answer_f1" in metrics:
                print(f"   Has-Answer F1: {metrics['has_answer_f1']:.2f}%")
                print(f"   No-Answer F1: {metrics['no_answer_f1']:.2f}%")

            # GMM-specific metrics
            if "expert_utilization" in metrics:
                print(f"   Expert Utilization: {metrics['expert_utilization']}")
            if "routing_entropy" in metrics:
                print(f"   Routing Entropy: {metrics['routing_entropy']:.3f}")

            print("=" * 80)

            # Success criteria
            if metrics.get("f1", 0.0) > 0:
                print("\n✅ SUCCESS: GMM model is learning (F1 > 0%)")
                if metrics.get("f1", 0.0) >= 65:
                    print("🎉 EXCELLENT: GMM achieved strong performance (F1 >= 65%)")
                if metrics.get("f1", 0.0) >= 75:
                    print("🏆 OUTSTANDING: GMM expert specialization showing benefits (F1 >= 75%)")
            else:
                print("\n⚠️  WARNING: F1 still 0% - GMM routing investigation needed")

        print(f"\n✅ Completed: {config.output_dir}")

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        raise


if __name__ == "__main__":
    main()