#!/usr/bin/env python3
"""
Comprehensive device placement test for GMM XLNet training script.

This test validates the device placement fixes implemented in Story 1.15
to ensure all tensors are on the same device and training runs without
device mismatch errors on both CPU and CUDA environments.

Tests cover:
- GMM model device placement during initialization
- Memory state device consistency throughout training
- Forward pass execution with proper device placement
- CPU/CUDA compatibility validation
- Edge case scenarios for device placement
"""

import sys
import warnings
from pathlib import Path

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gmmxlnet.models import GMMXLNetForQA
from gmmxlnet.training import gmm_balanced_config


@pytest.mark.unit
class TestGMMDevicePlacement:
    """Test suite for GMM device placement validation."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment for device placement tests."""
        # Suppress warnings for cleaner test output
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    @pytest.fixture(params=["cpu", "cuda"])
    def device(self, request):
        """Parametrized device fixture for CPU and CUDA testing."""
        device_str = request.param
        if device_str == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU tests")
        return torch.device(device_str)

    @pytest.fixture
    def gmm_config(self):
        """Create a minimal GMM configuration for testing."""
        return gmm_balanced_config(
            memory_num_tokens=8,  # Reduced for faster testing
            num_epochs=1,
            batch_size=2,         # Small batch for testing
            max_length=128,       # Shorter sequences for testing
        )

    def test_gmm_model_device_initialization(self, device, gmm_config):
        """Test that GMM model components are properly placed on target device."""
        # Create model with device-aware configuration
        model = GMMXLNetForQA.from_config(gmm_config)
        model = model.to(device)

        # Verify model is on correct device
        assert next(model.parameters()).device == device

        # Verify memory mixture is on correct device if it exists
        if hasattr(model, 'memory_mixture') and model.memory_mixture is not None:
            assert model.memory_mixture.expert_states.device == device

        # Verify all model parameters are on the correct device
        for param in model.parameters():
            assert param.device == device, f"Parameter found on {param.device}, expected {device}"

    def test_memory_state_device_consistency(self, device, gmm_config):
        """Test that memory states maintain device consistency during initialization."""
        model = GMMXLNetForQA.from_config(gmm_config)
        model = model.to(device)

        batch_size = 2
        hidden_dim = 768
        memory_slots = gmm_config.memory_num_tokens

        # Initialize memory states
        if hasattr(model, 'initialize_memory_states'):
            memory_states = model.initialize_memory_states(batch_size, device)

            # Verify memory states are on correct device
            assert memory_states.device == device

            # Verify expert memory states maintain device consistency
            if hasattr(model, 'memory_mixture') and model.memory_mixture is not None:
                expert_states = model.memory_mixture.expert_states
                assert expert_states.device == device
                assert expert_states.shape[0] == batch_size
                assert expert_states.shape[1] == model.memory_mixture.num_experts
                assert expert_states.shape[2] == memory_slots
                assert expert_states.shape[3] == hidden_dim

    def test_forward_pass_device_consistency(self, device, gmm_config):
        """Test forward pass maintains device consistency across all tensors."""
        model = GMMXLNetForQA.from_config(gmm_config)
        model = model.to(device)
        model.eval()

        batch_size = 2
        seq_length = 64

        # Create input tensors on target device
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)

        # Test forward pass without memory
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                use_memory=False
            )

            # Verify outputs are on correct device
            assert outputs.logits.device == device
            if hasattr(outputs, 'memory_states') and outputs.memory_states is not None:
                assert outputs.memory_states.device == device

        # Test forward pass with memory
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                use_memory=True
            )

            # Verify outputs and memory states are on correct device
            assert outputs.logits.device == device
            if hasattr(outputs, 'memory_states') and outputs.memory_states is not None:
                assert outputs.memory_states.device == device

    def test_device_mismatch_error_handling(self, gmm_config):
        """Test proper error handling when device mismatches occur."""
        model = GMMXLNetForQA.from_config(gmm_config)
        model = model.to(torch.device("cpu"))

        batch_size = 2
        seq_length = 64

        # Create input tensors on CUDA if available (or different device)
        if torch.cuda.is_available():
            input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to("cuda")
            attention_mask = torch.ones_like(input_ids)
            token_type_ids = torch.zeros_like(input_ids)

            # Should raise an error due to device mismatch
            with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
                with torch.no_grad():
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        use_memory=False
                    )

    def test_memory_state_updates_device_consistency(self, device, gmm_config):
        """Test that memory state updates maintain device consistency."""
        model = GMMXLNetForQA.from_config(gmm_config)
        model = model.to(device)
        model.eval()

        batch_size = 2
        seq_length = 64

        # Create input tensors on target device
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)

        # Initialize memory states
        if hasattr(model, 'initialize_memory_states'):
            memory_states = model.initialize_memory_states(batch_size, device)
            assert memory_states.device == device

        # Perform multiple forward passes to test memory state updates
        with torch.no_grad():
            for _ in range(3):  # Multiple passes to test state updates
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    memory_states=memory_states if hasattr(model, 'initialize_memory_states') else None,
                    use_memory=True
                )

                # Verify outputs remain on correct device
                assert outputs.logits.device == device

                # Update memory states for next iteration
                if hasattr(outputs, 'memory_states') and outputs.memory_states is not None:
                    memory_states = outputs.memory_states
                    assert memory_states.device == device

    def test_expert_memory_device_consistency(self, device, gmm_config):
        """Test that all expert memory banks maintain device consistency."""
        model = GMMXLNetForQA.from_config(gmm_config)
        model = model.to(device)

        # Test memory mixture device placement
        if hasattr(model, 'memory_mixture') and model.memory_mixture is not None:
            memory_mixture = model.memory_mixture

            # Verify expert states are on correct device
            assert memory_mixture.expert_states.device == device

            # Access memory for different experts
            for expert_idx in range(memory_mixture.num_experts):
                expert_memory = memory_mixture.get_expert_memory(expert_idx)
                if expert_memory is not None:
                    assert expert_memory.device == device

    def test_device_transfer_functionality(self, gmm_config):
        """Test that model can be transferred between devices correctly."""
        model = GMMXLNetForQA.from_config(gmm_config)

        # Initially on CPU
        assert next(model.parameters()).device.type == "cpu"

        # Move to CUDA if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            assert next(model.parameters()).device.type == "cuda"

            # Move back to CPU
            model = model.to("cpu")
            assert next(model.parameters()).device.type == "cpu"

    def test_training_script_device_placement_simulation(self, device):
        """Test simulation of training script device placement logic."""
        # This test simulates the device placement validation from the training script

        # Simulate trainer device placement
        trainer_device = device

        # Create model config
        config = gmm_balanced_config(
            memory_num_tokens=8,
            num_epochs=1,
            batch_size=2,
        )

        # Create and place model
        model = GMMXLNetForQA.from_config(config)
        model = model.to(trainer_device)

        # Verify model device placement
        assert next(model.parameters()).device == trainer_device

        # Simulate memory state initialization
        batch_size = 2
        if hasattr(model, 'initialize_memory_states'):
            memory_states = model.initialize_memory_states(batch_size, trainer_device)

            # Device consistency check (similar to training script)
            if memory_states.device != trainer_device:
                raise RuntimeError(f"Memory states device {memory_states.device} != trainer device {trainer_device}")

        # Simulate training loop device validation
        model.eval()
        with torch.no_grad():
            # Create batch on trainer device
            input_ids = torch.randint(0, 1000, (batch_size, 32)).to(trainer_device)
            attention_mask = torch.ones_like(input_ids)

            # Forward pass with device validation
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_memory=True,
                    memory_states=memory_states if hasattr(model, 'initialize_memory_states') else None
                )

                # Verify output device consistency
                assert outputs.logits.device == trainer_device

                if hasattr(outputs, 'memory_states') and outputs.memory_states is not None:
                    assert outputs.memory_states.device == trainer_device

            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    raise RuntimeError(f"Device mismatch detected: {e}")
                else:
                    raise

    @pytest.mark.slow
    def test_extended_training_device_stability(self, device):
        """Test device stability over extended training simulation."""
        config = gmm_balanced_config(
            memory_num_tokens=16,
            num_epochs=1,
            batch_size=4,
        )

        model = GMMXLNetForQA.from_config(config)
        model = model.to(device)
        model.eval()

        batch_size = 4
        seq_length = 64

        # Create consistent input tensors
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)

        memory_states = None

        # Simulate multiple training steps
        for step in range(10):  # 10 steps to test stability
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    memory_states=memory_states,
                    use_memory=True
                )

                # Verify device consistency at each step
                assert outputs.logits.device == device

                if hasattr(outputs, 'memory_states') and outputs.memory_states is not None:
                    assert outputs.memory_states.device == device
                    memory_states = outputs.memory_states

                # Additional device validation every few steps
                if step % 3 == 0:
                    for param in model.parameters():
                        assert param.device == device


def run_device_placement_validation():
    """Run device placement validation as standalone function."""
    print("Running GMM Device Placement Validation...")

    # Setup environment manually
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Test CPU device placement
    print("Testing CPU device placement...")
    try:
        # Create test instance
        test_instance = TestGMMDevicePlacement()

        # Test CPU device placement
        test_instance.test_gmm_model_device_initialization(torch.device("cpu"), gmm_balanced_config())
        test_instance.test_memory_state_device_consistency(torch.device("cpu"), gmm_balanced_config())
        test_instance.test_forward_pass_device_consistency(torch.device("cpu"), gmm_balanced_config())
        test_instance.test_memory_state_updates_device_consistency(torch.device("cpu"), gmm_balanced_config())
        test_instance.test_expert_memory_device_consistency(torch.device("cpu"), gmm_balanced_config())
        test_instance.test_device_transfer_functionality(gmm_balanced_config())
        test_instance.test_training_script_device_placement_simulation(torch.device("cpu"))
        print("✅ CPU device placement tests passed!")
    except Exception as e:
        print(f"❌ CPU device placement test failed: {e}")
        return False

    # Test CUDA device placement if available
    if torch.cuda.is_available():
        print("Testing CUDA device placement...")
        try:
            test_instance.test_gmm_model_device_initialization(torch.device("cuda"), gmm_balanced_config())
            test_instance.test_memory_state_device_consistency(torch.device("cuda"), gmm_balanced_config())
            test_instance.test_forward_pass_device_consistency(torch.device("cuda"), gmm_balanced_config())
            test_instance.test_memory_state_updates_device_consistency(torch.device("cuda"), gmm_balanced_config())
            test_instance.test_expert_memory_device_consistency(torch.device("cuda"), gmm_balanced_config())
            test_instance.test_training_script_device_placement_simulation(torch.device("cuda"))
            print("✅ CUDA device placement tests passed!")
        except Exception as e:
            print(f"❌ CUDA device placement test failed: {e}")
            return False
    else:
        print("⚠️  CUDA not available, skipping GPU tests")

    print("🎉 All device placement validation tests completed successfully!")
    return True


if __name__ == "__main__":
    success = run_device_placement_validation()
    sys.exit(0 if success else 1)