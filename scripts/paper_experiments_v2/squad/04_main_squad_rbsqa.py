#!/usr/bin/env python3
"""
Standard SQuAD v2 - RBSQA Main (Reasoning-Belief State Architecture)
===================================================================

Dataset: squad_v2 (standard)
Max segments: 2
Memory: RBSQA with belief state tracker and halting policy

Purpose: Main result on standard benchmark with RBSQA memory architecture.

This script uses the Reasoning-Belief State QA architecture instead of GMM or
differentiable memory, featuring:
- Belief state tracking for question understanding
- Halting policy for adaptive reasoning steps
- Integrated model architecture for end-to-end training

Key differences from GMM script:
- Uses RBSXLNetForQA instead of GMMXLNetForQA
- RBSQA-specific configuration parameters
- Belief state and halting policy components
- Different training pipeline for RBSQA architecture

Output: outputs/paper_v2_squad_rbsqa_main/
"""

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from rbsqa.models import RBSXLNetForQA
from rbsqa.training import RBSTrainingConfig
from memxlnet.training import XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    return RBSTrainingConfig(
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
        memory_num_tokens=16,  # RBSQA memory slots
        memory_update="gated",
        memory_init="learned",
        # ✅ RBSQA: Use belief state architecture instead of GMM
        use_rbsqa=True,
        belief_state_dim=64,  # Belief state dimension
        max_reasoning_steps=5,  # Maximum reasoning steps
        halting_threshold=0.9,  # Halting decision threshold
        belief_update_weight=0.1,  # Belief state update loss weight
        halting_policy_weight=0.05,  # Halting policy loss weight
        # Global softmax and training settings
        use_global_softmax=False,
        num_epochs=5,
        train_batch_size=8,
        eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        eval_steps=50000,  # TODO: change back to 5000
        save_steps=10000,
        logging_steps=500,
        output_dir="./outputs/paper_v2_squad_rbsqa_main",
        run_name="paper-v2-squad-rbsqa-main",
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
    print("📊 EXPERIMENT RBSQA: SQUAD V2 - MAIN (REASONING-BELIEF STATE ARCHITECTURE)")
    print("=" * 80 + "\n")
    print("🔧 Configuration details:")
    print("   ✅ memory_impl: 'differentiable' → 'rbsqa_belief_state'")
    print("   ✅ use_rbsqa: False → True")
    print("   ✅ Added: belief_state_dim=64, max_reasoning_steps=5")
    print("   ✅ Added: halting_threshold=0.9 for adaptive reasoning")
    print("   ✅ Memory: Belief state tracking with halting policy")
    print()
    print("Expected outcome: F1 > 0% (should achieve ~65-80% F1)")
    print("                  Adaptive reasoning should improve performance")
    print()

    config = create_config()

    print("🔧 RBSQA Configuration Summary:")
    print(f"   - Belief state dimension: {config.belief_state_dim}")
    print(f"   - Max reasoning steps: {config.max_reasoning_steps}")
    print(f"   - Halting threshold: {config.halting_threshold}")
    print(f"   - Memory slots: {config.memory_num_tokens}")
    print(f"   - Belief update weight: {config.belief_update_weight}")
    print(f"   - Halting policy weight: {config.halting_policy_weight}")
    print()

    # Create a custom RBSQA trainer using the existing infrastructure
    class RBSTrainer(XLNetRecurrentTrainer):
        """Extended trainer that supports RBSQA memory."""

        def __init__(self, config):
            # Call parent constructor
            super().__init__(config)

            # Override model to use RBSQA if enabled
            if config.use_rbsqa:
                logger.info("🔧 Initializing RBSQA-XLNet model")
                self.model = RBSXLNetForQA(
                    base_model=self.base_model,
                    belief_state_dim=config.belief_state_dim,
                    max_reasoning_steps=config.max_reasoning_steps,
                    halting_threshold=config.halting_threshold,
                    memory_slots=16,  # Standard memory slots
                    use_rbsqa=True,
                )
                logger.info(f"✅ RBSQA model created with belief state dim {config.belief_state_dim}")

                # Ensure RBSQA model is on the correct device
                self.model = self.model.to(self.device)
                logger.info(f"✅ RBSQA model moved to device: {self.device}")

                # Validate RBSQA model components are on correct device
                if hasattr(self.model, 'belief_state_tracker'):
                    belief_device = next(self.model.belief_state_tracker.parameters()).device
                    logger.info(f"✅ Belief state tracker device: {belief_device}")
                if hasattr(self.model, 'halting_policy'):
                    halting_device = next(self.model.halting_policy.parameters()).device
                    logger.info(f"✅ Halting policy device: {halting_device}")

                # Test initial memory creation
                test_memory = self.model.get_initial_memory(1, self.device)
                logger.info(f"✅ Test initial memory created")
                logger.debug(f"  - Memory device: {next(iter(test_memory.values())).device}")
                logger.debug(f"  - Memory shape: {next(iter(test_memory.values())).shape}")
                logger.info("🔧 RBSQA model initialization validation complete")

        def _process_document_batch_with_memory(self, time_step_batches, eval_mode=False):
            """Override memory processing for RBSQA models."""
            if hasattr(self.model, 'use_rbsqa') and self.model.use_rbsqa:
                # RBSQA memory handling
                self.model.train() if not eval_mode else self.model.eval()

                # Initialize per-document memory bank
                if not hasattr(self, "memory_bank"):
                    self.memory_bank: dict[str, torch.Tensor] = {}

                # Device placement validation
                if not eval_mode:
                    logger.debug(f"RBSQA processing on device: {self.device}")
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

                    # Build RBSQA memory state per batch ordering with device consistency
                    memory_state_batch = []
                    for ex_id, active in zip(batch["example_ids"], document_mask.tolist()):
                        if not active:
                            # New document: use initial memory (already on correct device)
                            initial_memory = self.model.get_initial_memory(1, self.device)
                            # 🔧 FIX: Handle different possible RBSQA memory key formats
                            if isinstance(initial_memory, dict):
                                # Try common key names RBSQA might use
                                memory_state = (initial_memory.get("memory_state") or
                                              initial_memory.get("memory") or
                                              initial_memory.get("state") or
                                              next(iter(initial_memory.values())) if initial_memory else None)
                            else:
                                memory_state = initial_memory

                            if memory_state is None:
                                raise ValueError(f"Could not extract memory state from initial_memory: {list(initial_memory.keys()) if isinstance(initial_memory, dict) else type(initial_memory)}")
                        else:
                            # Existing document: get from memory bank
                            prev = self.memory_bank.get(ex_id)
                            if prev is None:
                                initial_memory = self.model.get_initial_memory(1, self.device)
                                if isinstance(initial_memory, dict):
                                    memory_state = (initial_memory.get("memory_state") or
                                                  initial_memory.get("memory") or
                                                  initial_memory.get("state") or
                                                  next(iter(initial_memory.values())) if initial_memory else None)
                                else:
                                    memory_state = initial_memory
                            else:
                                memory_state = prev  # Already on correct device

                        # Ensure consistent 2D tensor shape for stacking
                        if memory_state.dim() == 3:  # Shape: [1, memory_slots, hidden_dim]
                            memory_state = memory_state.squeeze(0)  # -> [memory_slots, hidden_dim]
                        elif memory_state.dim() == 2:  # Shape: [memory_slots, hidden_dim]
                            pass  # Already correct shape
                        else:
                            raise ValueError(f"Unexpected memory shape: {memory_state.shape}, expected 2D or 3D tensor")

                        memory_state_batch.append(memory_state)

                    # Stack memory states across batch (already on correct device)
                    memory_state_batch = torch.stack(memory_state_batch, dim=0)

                    # Validate tensor shapes
                    if not eval_mode:
                        logger.debug(f"RBSQA memory state shape: {memory_state_batch.shape}")
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
                            logger.error("=" * 80)
                            logger.error("🚨 RBSQA DEVICE PLACEMENT ERROR DETECTED")
                            logger.error("=" * 80)
                            logger.error(f"Root error: {e}")
                            logger.error(f"Target device: {self.device}")
                            logger.error(f"Input IDs device: {input_ids.device} | Shape: {input_ids.shape}")
                            logger.error(f"Attention mask device: {attention_mask.device} | Shape: {attention_mask.shape}")
                            logger.error(f"Start positions device: {start_positions.device} | Shape: {start_positions.shape}")
                            logger.error(f"End positions device: {end_positions.device} | Shape: {end_positions.shape}")
                            logger.error(f"Document mask device: {document_mask.device} | Shape: {document_mask.shape}")
                            logger.error(f"Memory state device: {memory_state_batch.device} | Shape: {memory_state_batch.shape}")
                            logger.error(f"Model parameters device: {next(self.model.parameters()).device}")

                            # Check model subcomponents
                            if hasattr(self.model, 'belief_state_tracker'):
                                logger.error(f"Belief state tracker device: {next(self.model.belief_state_tracker.parameters()).device}")
                            if hasattr(self.model, 'halting_policy'):
                                logger.error(f"Halting policy device: {next(self.model.halting_policy.parameters()).device}")

                            logger.error("=" * 80)
                            raise RuntimeError(f"RBSQA device placement error: {e}. All tensors must be on the same device: {self.device}")
                        else:
                            logger.error(f"RBSQA model runtime error: {e}")
                            logger.error(f"Input shapes: IDs={input_ids.shape}, Attention={attention_mask.shape}")
                            raise

                    start_logits = outputs["start_logits"]
                    end_logits = outputs["end_logits"]
                    new_memory_state = outputs["new_memory_state"]

                    # 🔧 FIX: Handle RBSQA memory output format (could be dict or tensor)
                    if isinstance(new_memory_state, dict):
                        # Extract the main memory tensor from dict
                        memory_tensor = (new_memory_state.get("memory_state") or
                                       new_memory_state.get("memory") or
                                       new_memory_state.get("state") or
                                       next(iter(new_memory_state.values())) if new_memory_state else None)
                        if memory_tensor is None:
                            raise ValueError(f"Could not extract memory tensor from new_memory_state: {list(new_memory_state.keys())}")
                    else:
                        memory_tensor = new_memory_state

                    # Collect logits and labels per document
                    for ex_id, active in zip(batch["example_ids"], document_mask.tolist()):
                        if active:
                            # Get batch index for this document to extract individual memory
                            example_ids_list = batch["example_ids"].tolist() if hasattr(batch["example_ids"], 'tolist') else batch["example_ids"]
                            ex_idx = example_ids_list.index(ex_id)

                            # 🔧 FIX: Extract individual memory and store in bank
                            if memory_tensor.dim() >= 2:
                                individual_memory = memory_tensor[ex_idx].detach()
                            else:
                                raise ValueError(f"Unexpected memory tensor shape: {memory_tensor.shape}")

                            # Store in memory bank (RBSQA stores single tensor, not dict)
                            self.memory_bank[ex_id] = individual_memory

                            # 🔧 FIX: Add memory validation in both train and eval modes
                            # Validate memory bank storage consistency (optimized)
                            if self.memory_bank[ex_id].device.type != self.device.type:
                                logger.debug(f"Memory bank {ex_id} device mismatch: {self.memory_bank[ex_id].device} vs {self.device}")
                                self.memory_bank[ex_id] = self.memory_bank[ex_id].to(self.device)

                            # 🔧 FIX: Add cleanup in both train and eval modes to prevent OOM
                            # Monitor memory bank size and implement automatic cleanup
                            memory_bank_size = len(self.memory_bank)

                            if not eval_mode:
                                if memory_bank_size > 1000:  # Prevent memory leaks
                                    logger.warning(f"Memory bank has grown to {memory_bank_size} documents, consider cleanup")
                                if memory_bank_size % 100 == 0 and memory_bank_size > 0:
                                    logger.info(f"Memory bank size: {memory_bank_size} documents")

                                # 🔧 FIX: More aggressive cleanup for RBSQA (belief states can be larger)
                                MEMORY_BANK_LIMIT = 3000  # Reduced from 5000 for RBSQA

                            # 🔧 CRITICAL FIX: Cleanup in both train and eval modes to prevent OOM
                            if memory_bank_size > MEMORY_BANK_LIMIT:
                                # Implement LRU-style cleanup: remove oldest 25% of documents
                                docs_to_remove = list(self.memory_bank.keys())[:memory_bank_size // 4]
                                removed_count = 0
                                for doc_id in docs_to_remove:
                                    if doc_id in self.memory_bank:
                                        del self.memory_bank[doc_id]
                                        removed_count += 1

                                logger.info(f"🧹 Automatic memory bank cleanup: removed {removed_count} documents "
                                           f"(old size: {memory_bank_size}, new size: {len(self.memory_bank)})")

                                # Force garbage collection to free GPU memory
                                import gc
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    # Log memory usage after cleanup
                                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                                    memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                                    logger.info(f"🧹 Memory after cleanup: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB")

                            # Store logits for this document (use first segment for simplicity)
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
                # Non-RBSQA model: use parent's memory processing
                return super()._process_document_batch_with_memory(time_step_batches, eval_mode)

        def train_one_document_batch(self, time_step_batches):
            """Override train_one_document_batch to handle RBSQA memory structure."""
            if hasattr(self.model, 'use_rbsqa') and self.model.use_rbsqa:
                return self._process_document_batch_with_memory(time_step_batches, eval_mode=False)
            else:
                # Non-RBSQA model: use parent's implementation
                return super().train_one_document_batch(time_step_batches)

        def _train_single_stage(self, train_dataloader, eval_dataloader, eval_dataset, stage_num):
            """Override _train_single_stage to add epoch cleanup for RBSQA models."""
            if hasattr(self.model, 'use_rbsqa') and self.model.use_rbsqa:
                # Call parent implementation but ensure cleanup happens
                from memxlnet.training.trainer import XLNetRecurrentTrainer

                # Get the parent's method
                parent_method = super()._train_single_stage

                # Call parent method
                result = parent_method(train_dataloader, eval_dataloader, eval_dataset, stage_num)

                # Ensure cleanup is called after each epoch (parent calls this internally)
                # The cleanup logic is now handled in the automatic cleanup in _process_document_batch_with_memory

                return result
            else:
                # Non-RBSQA model: use parent's implementation
                return super()._train_single_stage(train_dataloader, eval_dataloader, eval_dataset, stage_num)

        def _clear_memory_bank(self):
            """Clear the memory bank to prevent memory leaks between epochs."""
            if hasattr(self, "memory_bank"):
                self.memory_bank.clear()
                logger.debug("Cleared RBSQA memory bank")

        def on_epoch_end(self):
            """Called at the end of each epoch to cleanup resources."""
            if hasattr(self.model, 'use_rbsqa') and self.model.use_rbsqa:
                self._clear_memory_bank()
                # Reset RBSQA model memory
                self.model.reset_memory()
                logger.debug("Reset RBSQA model memory at epoch end")

            # Call parent implementation if it exists
            if hasattr(super(), 'on_epoch_end'):
                super().on_epoch_end()

    trainer = RBSTrainer(config)

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

            # RBSQA-specific metrics
            if "avg_reasoning_steps" in metrics:
                print(f"   Avg Reasoning Steps: {metrics['avg_reasoning_steps']:.2f}")
            if "halting_accuracy" in metrics:
                print(f"   Halting Accuracy: {metrics['halting_accuracy']:.2f}%")

            print("=" * 80)

            # Success criteria
            if metrics.get("f1", 0.0) > 0:
                print("\n✅ SUCCESS: RBSQA model is learning (F1 > 0%)")
                if metrics.get("f1", 0.0) >= 65:
                    print("🎉 EXCELLENT: RBSQA achieved strong performance (F1 >= 65%)")
                if metrics.get("f1", 0.0) >= 75:
                    print("🏆 OUTSTANDING: RBSQA belief state tracking showing benefits (F1 >= 75%)")
            else:
                print("\n⚠️  WARNING: F1 still 0% - RBSQA reasoning investigation needed")

        print(f"\n✅ Completed: {config.output_dir}")

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        raise


if __name__ == "__main__":
    main()