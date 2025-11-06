#!/usr/bin/env python3
"""
Simple test script to validate RBSQA tensor fixes
"""
import sys
import torch
sys.path.append('src')

from rbsqa.models.rbs_xlnet import RBSXLNetForQA
from rbsqa.config import RBSTrainingConfig

def test_rbs_initialization():
    """Test basic RBS model initialization and tensor shapes"""
    print("🔍 Testing RBS Model Initialization")
    print("=" * 50)

    # Create minimal config using the same pattern as working script
    device = "cpu"  # Force CPU for testing
    config = RBSTrainingConfig(
        # Base XLNet configuration
        model_name="xlnet-base-cased",
        max_seq_length=384,
        doc_stride=64,
        dataset_name="squad_v2",
        train_split="train",
        eval_split="validation",
        cache_dir="./.cache",
        max_train_samples=10,  # Very small for testing
        max_eval_samples=5,
        use_lazy_loading=True,
        # Training settings
        num_epochs=1,
        train_batch_size=2,
        eval_batch_size=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        eval_steps=500,
        save_steps=10000,
        logging_steps=500,
        output_dir="./test_outputs",
        run_name="test-rbs-experiment",
        # RBS-specific settings
        memory_num_tokens=8,  # Reduced memory for testing
        num_memory_experts=4,
        max_segments=2,
        belief_state_threshold=0.9,
        use_halting_policy=False,  # Disable for simpler testing
        device=device
    )

    try:
        # Initialize trainer
        from rbsqa.training.rbs_trainer import RBSTrainer
        trainer = RBSTrainer(config)
        print("✅ RBSTrainer initialized successfully")

        # Test model creation
        model = trainer.model
        print(f"✅ Model created with type: {type(model)}")
        print(f"✅ Model use_rbs_mode: {getattr(model, 'use_rbs_mode', 'NOT SET')}")

        # Test memory state creation
        batch_size = 2
        memory_state = model.gmm_backbone.get_initial_memory(batch_size, "cpu")
        print(f"✅ Initial memory state created with {len(memory_state)} experts")

        for expert_key, expert_tensor in memory_state.items():
            print(f"  {expert_key}: shape={expert_tensor.shape}, dtype={expert_tensor.dtype}")

            # Validate tensor shape
            if expert_tensor.dim() != 3:
                print(f"❌ ERROR: {expert_key} has {expert_tensor.dim()}D tensor, expected 3D")
                return False
            if expert_tensor.size(0) != batch_size:
                print(f"❌ ERROR: {expert_key} batch size mismatch: {expert_tensor.size(0)} != {batch_size}")
                return False

        print("✅ All tensor shapes are correct")
        return True

    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_state_building():
    """Test memory state building logic"""
    print("\n🔍 Testing Memory State Building")
    print("=" * 50)

    try:
        from rbsqa.training.rbs_trainer import RBSTrainer
        device = "cpu"  # Force CPU for testing
        config = RBSTrainingConfig(
            model_name="xlnet-base-cased",
            max_seq_length=384,
            dataset_name="squad_v2",
            max_train_samples=10,
            max_eval_samples=5,
            num_epochs=1,
            train_batch_size=2,
            eval_batch_size=2,
            memory_num_tokens=8,
            max_segments=2,
            device=device
        )
        trainer = RBSTrainer(config)

        # Mock batch data
        batch = {
            "example_ids": torch.tensor([0, 1]),
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "token_type_ids": torch.tensor([[0, 0, 0], [0, 0, 0]]),
            "start_positions": torch.tensor([0, 1]),
            "end_positions": torch.tensor([2, 1]),
        }

        # Test with different document masks
        for mask_desc, document_mask in [
            ("New documents", torch.tensor([False, False])),
            ("Mixed new/existing", torch.tensor([True, False])),
            ("Existing documents", torch.tensor([True, True])),
        ]:
            print(f"\n🧪 Testing: {mask_desc}")
            print(f"  Document mask: {document_mask.tolist()}")

            try:
                memory_state = trainer._build_rbs_memory_state(batch, document_mask)
                print(f"  ✅ Memory state built: {len(memory_state)} experts")

                for expert_key, expert_tensor in memory_state.items():
                    print(f"    {expert_key}: shape={expert_tensor.shape}")

                    # Critical validation
                    if expert_tensor.dim() != 3:
                        print(f"    ❌ ERROR: {expert_key} has {expert_tensor.dim()}D tensor, expected 3D")
                        return False

                    expected_batch = len(batch["example_ids"])
                    actual_batch = expert_tensor.size(0)
                    if actual_batch != expected_batch:
                        print(f"    ❌ ERROR: {expert_key} batch mismatch: {actual_batch} != {expected_batch}")
                        return False

                print(f"  ✅ All validations passed for {mask_desc}")

            except Exception as e:
                print(f"  ❌ ERROR in {mask_desc}: {e}")
                return False

        return True

    except Exception as e:
        print(f"❌ Error in memory state test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting RBSQA Fix Validation Tests")
    print("=" * 60)

    test1_pass = test_rbs_initialization()
    test2_pass = test_memory_state_building()

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Model Initialization Test: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Memory State Building Test: {'✅ PASS' if test2_pass else '❌ FAIL'}")

    if test1_pass and test2_pass:
        print("\n🎉 ALL TESTS PASSED! The RBSQA tensor fixes are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")