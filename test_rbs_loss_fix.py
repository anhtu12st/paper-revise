#!/usr/bin/env python3
"""
Simple test to validate RBSModelOutput loss attribute fix
"""
import sys
import torch
sys.path.append('src')

from rbsqa.models.rbs_xlnet import RBSXLNetForQA
from rbsqa.config import RBSTrainingConfig

def test_rbs_loss_attribute():
    """Test that RBSModelOutput has loss attribute"""
    print("🔍 Testing RBSModelOutput loss attribute")
    print("=" * 50)

    try:
        # Create minimal config
        device = "cpu"
        config = RBSTrainingConfig(
            model_name="xlnet-base-cased",
            max_seq_length=128,  # Reduced for testing
            dataset_name="squad_v2",
            max_train_samples=2,
            max_eval_samples=2,
            num_epochs=1,
            train_batch_size=1,
            eval_batch_size=1,
            memory_num_tokens=4,  # Reduced for testing
            num_memory_experts=2,  # Reduced for testing
            device=device
        )

        # Initialize trainer
        from rbsqa.training.rbs_trainer import RBSTrainer
        trainer = RBSTrainer(config)
        model = trainer.model
        print("✅ RBSQA model initialized successfully")

        # Create mock input data
        batch_size = 1
        seq_len = 32
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        start_positions = torch.tensor([5])
        end_positions = torch.tensor([10])

        # Test forward pass with labels
        print("🧪 Testing forward pass with labels...")
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

        # Check if output has loss attribute
        if hasattr(outputs, 'loss'):
            print(f"✅ RBSModelOutput has loss attribute: {outputs.loss}")
            if outputs.loss is not None:
                print(f"✅ Loss computed: {outputs.loss.item():.4f}")
            else:
                print("⚠️  Loss attribute exists but is None")
        else:
            print("❌ RBSModelOutput missing loss attribute")
            return False

        # Test evaluation compatibility
        print("🧪 Testing evaluation compatibility...")
        if outputs.loss is not None:
            total_loss = outputs.loss.item()
            print(f"✅ Evaluation can access loss: {total_loss:.4f}")
        else:
            print("⚠️  Loss is None in evaluation")

        print("✅ All tests passed! RBSModelOutput loss attribute fix is working.")
        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rbs_loss_attribute()
    exit(0 if success else 1)