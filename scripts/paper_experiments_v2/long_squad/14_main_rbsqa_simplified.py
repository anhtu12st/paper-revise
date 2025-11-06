#!/usr/bin/env python3
"""
Long SQuAD v2 - RBSQA Main (Simplified with RBSTrainer)
====================================================

Dataset: huutuan/long_squad_v2 (long documents, 6-12 segments)
Progressive segments: [2, 4, 6]
Memory: RBSQA with belief state tracker and halting policy

Purpose: Main result on long documents with RBSQA memory architecture.
This script uses the dedicated RBSTrainer for cleaner, reusable code.

Output: outputs/paper_v2_long_rbsqa_main/
"""

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from rbsqa.training import RBSTrainer
from rbsqa.config import RBSTrainingConfig

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
        dataset_name="huutuan/long_squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=None,
        max_eval_samples=None,
        use_lazy_loading=False,
        # Chunked dataset settings (FAST: 2-5 min vs 30-60 min preprocessing)
        use_chunked_dataset=True,
        chunked_dataset_dir="./preprocessed_data/huutuan_long_squad_v2",
        chunked_load_mode="streaming",  # Memory-efficient streaming
        progressive_segments=[2, 4, 6],
        max_n_segs=6,
        # Memory configuration
        memory_num_tokens=16,  # RBSQA memory slots
        memory_update="gated",
        memory_init="learned",
        # ✅ RBSQA: Use belief state architecture (RBS extends GMM)
        use_gmm_memory=True,  # RBS requires GMM memory
        # RBS-specific parameters
        use_belief_state=True,
        belief_state_threshold=0.85,  # Lower threshold for more thorough reasoning on long docs
        enable_re_scoring=True,
        confidence_calibration=True,
        re_scoring_method="context_weighted",
        enable_trend_analysis=True,
        max_segments=32,  # Maximum reasoning steps equivalent
        belief_state_memory_limit=100,
        halting_patience=3,
        # Halting policy settings
        use_halting_policy=True,
        halting_policy_hidden_dim=64,  # Belief state dimension equivalent
        halting_policy_layers=2,
        halting_temperature=1.0,
        halting_exploration_rate=0.1,
        # RL training settings
        rl_weight=0.1,  # Belief update weight equivalent
        lambda_cost=0.01,
        gamma=0.99,
        use_value_baseline=True,
        value_weight=0.05,  # Halting policy weight equivalent
        # Global softmax and training settings
        use_global_softmax=True,
        num_epochs=3,
        train_batch_size=8,
        eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        eval_steps=6000,
        save_steps=10000,
        logging_steps=500,
        output_dir="./outputs/paper_v2_long_rbsqa_main",
        run_name="paper-v2-long-rbsqa-main",
        save_total_limit=3,
        no_answer_threshold=1.5,
        use_any_positive_logic=True,
        device=device,
        fp16=has_cuda,
        # ✅ Enable global softmax after 1 epoch (consistent with original for long docs)
        warmup_freeze_base_epochs=0,
        warmup_disable_global_softmax_epochs=1,
        warmup_disable_any_positive_epochs=0,
        push_to_hub_on_save=False,
    )


def main():
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT RBSQA: LONG SQUAD V2 - MAIN (SIMPLIFIED WITH RBS TRAINER)")
    print("=" * 80 + "\n")
    print("🔧 Configuration details:")
    print("   ✅ Using dedicated RBSTrainer for clean, reusable code")
    print("   ✅ RBS extends GMM with belief state tracking and halting policy")
    print("   ✅ Memory: 4 independent expert memories with learned routing")
    print("   ✅ Progressive segments: [2, 4, 6] for long documents")
    print("   ✅ All RBS-specific logic handled by RBSTrainer")
    print()
    print("Expected outcome: F1 > 0% (should achieve ~60-75% F1 on long documents)")
    print("                  Adaptive reasoning should improve long-context performance")
    print()

    config = create_config()

    print("🔧 RBSQA Configuration Summary:")
    print(f"   - Belief state enabled: {config.use_belief_state}")
    print(f"   - Belief state threshold: {config.belief_state_threshold}")
    print(f"   - Max segments: {config.max_segments}")
    print(f"   - Re-scoring method: {config.re_scoring_method}")
    print(f"   - Halting policy enabled: {config.use_halting_policy}")
    print(f"   - Halting hidden dim: {config.halting_policy_hidden_dim}")
    print(f"   - Memory slots: {config.memory_num_tokens}")
    print(f"   - RL weight: {config.rl_weight}")
    print(f"   - Value weight: {config.value_weight}")
    print(f"   - Progressive segments: {config.progressive_segments}")
    print(f"   - Max segments: {config.max_n_segs}")
    print()

    # Create RBSQA trainer (handles all RBS-specific logic internally)
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

            # Log RBS-specific metrics
            trainer.log_rbs_metrics(metrics)

            print("=" * 80)

            # Success criteria
            if metrics.get("f1", 0.0) > 0:
                print("\n✅ SUCCESS: RBSQA model is learning (F1 > 0%)")
                if metrics.get("f1", 0.0) >= 60:
                    print("🎉 EXCELLENT: RBSQA achieved strong performance on long documents (F1 >= 60%)")
                if metrics.get("f1", 0.0) >= 70:
                    print("🏆 OUTSTANDING: RBSQA belief state tracking showing benefits on long context (F1 >= 70%)")
            else:
                print("\n⚠️  WARNING: F1 still 0% - RBSQA reasoning investigation needed")

        print(f"\n✅ Completed: {config.output_dir}")

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        raise


if __name__ == "__main__":
    main()