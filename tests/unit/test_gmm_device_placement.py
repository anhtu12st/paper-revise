"""
Unit tests for GMM device placement validation.

Tests cover device placement scenarios and CPU/CUDA compatibility for GMM models.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from gmmxlnet.models import GatedMemoryMixture, GMMXLNetForQA
from gmmxlnet.training import GMMTrainingConfig


@pytest.mark.unit
class TestGMMDevicePlacementScenarios:
    """Test device placement scenarios for GMM components."""

    @pytest.fixture(params=["cpu", "cuda"])
    def device(self, request):
        """Parametrized device fixture."""
        device_str = request.param
        if device_str == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device(device_str)

    @pytest.fixture
    def gmm_config(self):
        """Create GMM configuration for testing."""
        return GMMTrainingConfig(
            memory_num_tokens=8,
            num_memory_experts=4,
            batch_size=2,
            max_length=64,
        )

    def test_gated_memory_mixture_device_placement(self, device):
        """Test GatedMemoryMixture device placement."""
        memory_mixture = GatedMemoryMixture(
            num_experts=4,
            memory_slots=8,
            hidden_dim=64,  # Small for testing
            init_strategies="zeros"
        )
        memory_mixture = memory_mixture.to(device)

        # Verify expert states are on correct device
        assert memory_mixture.expert_states.device == device

        # Test individual expert memory access
        for expert_idx in range(4):
            expert_memory = memory_mixture.get_expert_memory(expert_idx)
            assert expert_memory.device == device

    def test_gmm_model_device_placement(self, device, gmm_config):
        """Test complete GMM model device placement."""
        # Create minimal model for testing
        model = GMMXLNetForQA.from_config(gmm_config)
        model = model.to(device)

        # Verify all parameters are on correct device
        for param in model.parameters():
            assert param.device == device

        # Verify memory mixture if present
        if hasattr(model, 'memory_mixture') and model.memory_mixture is not None:
            assert model.memory_mixture.expert_states.device == device

    def test_device_consistency_across_memory_operations(self, device):
        """Test device consistency across memory operations."""
        memory_mixture = GatedMemoryMixture(
            num_experts=2,
            memory_slots=4,
            hidden_dim=32,
            init_strategies="zeros"
        )
        memory_mixture = memory_mixture.to(device)

        batch_size = 2
        seq_len = 16

        # Create mock hidden states on correct device
        hidden_states = torch.randn(batch_size, seq_len, 32).to(device)

        # Test memory reading maintains device consistency
        read_output = memory_mixture.read_memory(hidden_states)
        assert read_output.device == device

        # Test memory writing maintains device consistency
        write_content = torch.randn(batch_size, seq_len, 32).to(device)
        updated_states = memory_mixture.write_memory(hidden_states, write_content)
        assert updated_states.device == device

    def test_device_transfer_between_models(self, gmm_config):
        """Test transferring models between devices."""
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

    def test_memory_state_device_persistence(self, device, gmm_config):
        """Test that memory states maintain device across operations."""
        model = GMMXLNetForQA.from_config(gmm_config)
        model = model.to(device)

        batch_size = 2

        # Initialize memory states
        if hasattr(model, 'initialize_memory_states'):
            memory_states = model.initialize_memory_states(batch_size, device)
            assert memory_states.device == device

            # Simulate multiple operations
            for _ in range(3):
                # Memory states should maintain device
                assert memory_states.device == device

                # Access expert memories
                if hasattr(model, 'memory_mixture') and model.memory_mixture is not None:
                    for expert_idx in range(model.memory_mixture.num_experts):
                        expert_mem = model.memory_mixture.get_expert_memory(expert_idx)
                        assert expert_mem.device == device

    def test_device_validation_error_scenarios(self, gmm_config):
        """Test error handling for device mismatch scenarios."""
        model = GMMXLNetForQA.from_config(gmm_config)
        model = model.to(torch.device("cpu"))

        # Test with CUDA tensors if available
        if torch.cuda.is_available():
            batch_size = 2
            seq_len = 16

            # Create inputs on wrong device
            input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to("cuda")
            attention_mask = torch.ones_like(input_ids)

            # Should fail due to device mismatch
            with pytest.raises(RuntimeError):
                with torch.no_grad():
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_memory=False
                    )

    def test_expert_routing_device_consistency(self, device):
        """Test that expert routing maintains device consistency."""
        memory_mixture = GatedMemoryMixture(
            num_experts=4,
            memory_slots=8,
            hidden_dim=32,
            init_strategies="learned"
        )
        memory_mixture = memory_mixture.to(device)

        batch_size = 2
        seq_len = 16

        # Create query tensor on correct device
        query = torch.randn(batch_size, seq_len, 32).to(device)

        # Test routing computation
        routing_logits = memory_mixture.compute_routing_logits(query)
        assert routing_logits.device == device

        routing_probs = torch.softmax(routing_logits, dim=-1)
        assert routing_probs.device == device

        # Test that routing decisions maintain device
        expert_assignments = torch.argmax(routing_probs, dim=-1)
        assert expert_assignments.device == device

    def test_memory_bank_device_consistency(self, device):
        """Test memory bank operations maintain device consistency."""
        memory_mixture = GatedMemoryMixture(
            num_experts=2,
            memory_slots=4,
            hidden_dim=16,
            init_strategies="zeros"
        )
        memory_mixture = memory_mixture.to(device)

        # Test initial memory bank device
        assert memory_mixture.expert_states.device == device

        batch_size = 1

        # Simulate memory updates
        for _ in range(5):
            # Create new memory content on correct device
            new_content = torch.randn(batch_size, 4, 16).to(device)

            # Update expert memory (mock operation)
            with torch.no_grad():
                # Simulate memory update
                memory_mixture.expert_states += new_content.unsqueeze(1)  # Add expert dimension

            # Verify device consistency after update
            assert memory_mixture.expert_states.device == device

    def test_batch_device_consistency(self, device):
        """Test device consistency across different batch sizes."""
        memory_mixture = GatedMemoryMixture(
            num_experts=4,
            memory_slots=8,
            hidden_dim=32,
            init_strategies="zeros"
        )
        memory_mixture = memory_mixture.to(device)

        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            # Create batch-specific tensors
            query = torch.randn(batch_size, 16, 32).to(device)

            # Test memory operations
            output = memory_mixture.read_memory(query)
            assert output.device == device
            assert output.shape[0] == batch_size

    def test_gradient_flow_device_consistency(self, device):
        """Test that gradients flow correctly on target device."""
        memory_mixture = GatedMemoryMixture(
            num_experts=2,
            memory_slots=4,
            hidden_dim=16,
            init_strategies="learned"  # Learnable for gradient testing
        )
        memory_mixture = memory_mixture.to(device)

        batch_size = 2
        seq_len = 8

        # Create input requiring gradients
        query = torch.randn(batch_size, seq_len, 16, requires_grad=True).to(device)

        # Forward pass
        output = memory_mixture.read_memory(query)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check gradients are on correct device
        assert query.grad.device == device

        for param in memory_mixture.parameters():
            if param.grad is not None:
                assert param.grad.device == device


@pytest.mark.unit
class TestGMMDeviceCompatibility:
    """Test CPU/CUDA compatibility for GMM models."""

    def test_cpu_only_functionality(self):
        """Test GMM functionality works on CPU-only systems."""
        config = GMMTrainingConfig(
            memory_num_tokens=4,
            num_memory_experts=2,
            batch_size=1,
            max_length=32,
        )

        model = GMMXLNetForQA.from_config(config)
        model = model.to("cpu")

        # Test basic functionality
        batch_size = 1
        seq_len = 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_memory=True
            )

            assert outputs.logits.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_functionality(self):
        """Test GMM functionality works on CUDA."""
        config = GMMTrainingConfig(
            memory_num_tokens=8,
            num_memory_experts=4,
            batch_size=2,
            max_length=64,
        )

        model = GMMXLNetForQA.from_config(config)
        model = model.to("cuda")

        # Test basic functionality
        batch_size = 2
        seq_len = 32

        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to("cuda")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_memory=True
            )

            assert outputs.logits.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_cuda_model_compatibility(self):
        """Test that models can move between CPU and CUDA."""
        config = GMMTrainingConfig(
            memory_num_tokens=4,
            num_memory_experts=2,
            batch_size=1,
            max_length=32,
        )

        model = GMMXLNetForQA.from_config(config)

        # Test CPU functionality
        model = model.to("cpu")
        batch_size = 1
        seq_len = 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_memory=False
            )
            assert outputs.logits.device.type == "cpu"

        # Move to CUDA and test
        model = model.to("cuda")
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_memory=False
            )
            assert outputs.logits.device.type == "cuda"

        # Move back to CPU
        model = model.to("cpu")
        input_ids = input_ids.to("cpu")
        attention_mask = attention_mask.to("cpu")

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_memory=False
            )
            assert outputs.logits.device.type == "cpu"

    def test_device_autodetection(self):
        """Test device autodetection and fallback logic."""
        config = GMMTrainingConfig(
            memory_num_tokens=4,
            num_memory_experts=2,
            batch_size=1,
            max_length=32,
        )

        model = GMMXLNetForQA.from_config(config)

        # Test device selection logic
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = model.to(device)
        assert next(model.parameters()).device == device

    def test_memory_expert_device_independence(self):
        """Test that memory experts maintain independent device states."""
        config = GMMTrainingConfig(
            memory_num_tokens=4,
            num_memory_experts=4,
            batch_size=1,
            max_length=32,
        )

        model = GMMXLNetForQA.from_config(config)
        device = torch.device("cpu")  # Always test on CPU for this test
        model = model.to(device)

        # Test expert independence
        if hasattr(model, 'memory_mixture') and model.memory_mixture is not None:
            memory_mixture = model.memory_mixture

            # Verify all experts are on same device
            for expert_idx in range(memory_mixture.num_experts):
                expert_memory = memory_mixture.get_expert_memory(expert_idx)
                assert expert_memory.device == device

            # Test expert states tensor device consistency
            assert memory_mixture.expert_states.device == device
            assert memory_mixture.expert_states.shape[1] == 4  # 4 experts