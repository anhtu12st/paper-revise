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

        def _process_document_batch_with_memory(self, time_step_batches, eval_mode=False):
            """Override memory processing for GMM models."""
            if hasattr(self.model, 'use_gmm_memory') and self.model.use_gmm_memory:
                # GMM memory handling
                self.model.train() if not eval_mode else self.model.eval()

                # Initialize per-document memory bank
                if not hasattr(self, "memory_bank"):
                    self.memory_bank: dict[str, dict[str, torch.Tensor]] = {}

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

                    # Build GMM memory state dict per batch ordering
                    memory_state_batch = {}
                    for expert_idx in range(self.model.num_experts):
                        expert_memories = []

                        for ex_id, active in zip(batch["example_ids"], document_mask.tolist()):
                            if not active:
                                # New document: use initial memory
                                initial_memory = self.model.get_initial_memory(1, self.device)
                                expert_memory = initial_memory[f"expert_{expert_idx}"]
                            else:
                                # Existing document: get from memory bank
                                prev = self.memory_bank.get(ex_id)
                                if prev is None:
                                    initial_memory = self.model.get_initial_memory(1, self.device)
                                    expert_memory = initial_memory[f"expert_{expert_idx}"]
                                else:
                                    expert_memory = prev[f"expert_{expert_idx}"]
                            expert_memories.append(expert_memory.squeeze(0))  # Remove batch dim

                        # Stack expert memories across batch
                        memory_state_batch[f"expert_{expert_idx}"] = torch.stack(expert_memories, dim=0)

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

                    start_logits = outputs["start_logits"]
                    end_logits = outputs["end_logits"]
                    new_memory_state = outputs["new_memory_state"]

                    # Collect logits and labels per document
                    for ex_id, active in zip(batch["example_ids"], document_mask.tolist()):
                        if active and ex_id in self.dataset.example_to_segments:
                            # Update memory bank for next time step
                            self.memory_bank[ex_id] = new_memory_state

                            # Store logits for this document (use first segment for simplicity)
                            ex_idx = batch["example_ids"].tolist().index(ex_id)
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