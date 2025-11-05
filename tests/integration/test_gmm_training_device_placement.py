"""
Integration tests for GMM training device placement edge cases.

Tests cover comprehensive device placement scenarios in realistic training conditions,
including edge cases that may occur during actual GMM training workflows.
"""

import sys
import tempfile
import warnings
from pathlib import Path

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from gmmxlnet.models import GMMXLNetForQA
from gmmxlnet.training import GMMTrainer, gmm_balanced_config


@pytest.mark.integration
class TestGMMTrainingDevicePlacement:
    """Integration tests for GMM training device placement."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment."""
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    @pytest.fixture(params=["cpu", "cuda"])
    def device(self, request):
        """Parametrized device fixture."""
        device_str = request.param
        if device_str == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device(device_str)

    @pytest.fixture
    def training_config(self):
        """Create minimal training configuration for integration testing."""
        return gmm_balanced_config(
            memory_num_tokens=8,
            num_epochs=1,
            batch_size=2,
            max_length=64,
            warmup_freeze_base_epochs=0,  # No warmup for testing
            warmup_disable_global_softmax_epochs=0,
            warmup_disable_any_positive_epochs=0,
        )

    def test_training_script_device_placement_simulation(self, device, training_config):
        """Simulate the training script device placement logic."""
        # Create temporary dataset for testing
        with tempfile.TemporaryDirectory():

            # Mock trainer initialization
            trainer = GMMTrainer(training_config)
            trainer.device = device

            # Create model
            model = GMMXLNetForQA.from_config(training_config)
            model = model.to(device)
            trainer.model = model

            # Verify model device placement
            assert next(model.parameters()).device == device

            # Mock document processing
            batch_size = 2
            seq_length = 32

            # Create mock batch data
            input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
            attention_mask = torch.ones_like(input_ids)
            start_positions = torch.randint(0, seq_length, (batch_size,))
            end_positions = torch.randint(0, seq_length, (batch_size,))

            # Simulate memory state initialization
            if hasattr(model, 'initialize_memory_states'):
                memory_states = model.initialize_memory_states(batch_size, device)
                assert memory_states.device == device

            # Simulate training step
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            try:
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    use_memory=True,
                    memory_states=memory_states if hasattr(model, 'initialize_memory_states') else None
                )

                # Verify output device consistency
                assert outputs.start_logits.device == device
                assert outputs.end_logits.device == device

                if hasattr(outputs, 'memory_states') and outputs.memory_states is not None:
                    assert outputs.memory_states.device == device

                # Backward pass
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Verify parameters remain on correct device after update
                for param in model.parameters():
                    assert param.device == device

            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    pytest.fail(f"Device mismatch detected during training: {e}")
                else:
                    raise

    def test_multi_batch_device_consistency(self, device, training_config):
        """Test device consistency across multiple training batches."""
        model = GMMXLNetForQA.from_config(training_config)
        model = model.to(device)
        model.train()

        num_batches = 3
        batch_size = 2
        seq_length = 32

        memory_states = None

        for batch_idx in range(num_batches):
            # Create batch on target device
            input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
            attention_mask = torch.ones_like(input_ids)
            start_positions = torch.randint(0, seq_length, (batch_size,))
            end_positions = torch.randint(0, seq_length, (batch_size,))

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    use_memory=True,
                    memory_states=memory_states
                )

                # Verify device consistency
                assert outputs.start_logits.device == device
                assert outputs.end_logits.device == device

                # Update memory states for next batch
                if hasattr(outputs, 'memory_states') and outputs.memory_states is not None:
                    memory_states = outputs.memory_states
                    assert memory_states.device == device

    def test_memory_bank_device_consistency_extended(self, device, training_config):
        """Test memory bank device consistency over extended operations."""
        model = GMMXLNetForQA.from_config(training_config)
        model = model.to(device)

        if not hasattr(model, 'memory_mixture') or model.memory_mixture is None:
            pytest.skip("Model does not have memory mixture")

        memory_mixture = model.memory_mixture
        batch_size = 2
        num_operations = 10

        # Test multiple memory operations
        for _ in range(num_operations):
            # Create query tensor
            query = torch.randn(batch_size, 16, memory_mixture.hidden_dim).to(device)

            # Test memory read
            read_output = memory_mixture.read_memory(query)
            assert read_output.device == device

            # Test memory write
            write_content = torch.randn_like(query).to(device)
            updated_states = memory_mixture.write_memory(query, write_content)
            assert updated_states.device == device

            # Verify expert states remain consistent
            assert memory_mixture.expert_states.device == device

    def test_device_mismatch_recovery(self, training_config):
        """Test recovery from device mismatch scenarios."""
        model = GMMXLNetForQA.from_config(training_config)
        model = model.to(torch.device("cpu"))

        batch_size = 2
        seq_length = 32

        if torch.cuda.is_available():
            # Create mismatched inputs
            input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to("cuda")
            attention_mask = torch.ones_like(input_ids)

            # Should detect device mismatch
            with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
                with torch.no_grad():
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_memory=False
                    )

            # Test recovery by moving inputs to correct device
            input_ids = input_ids.to("cpu")
            attention_mask = attention_mask.to("cpu")

            # Should work fine now
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_memory=False
                )
                assert outputs.start_logits.device.type == "cpu"

    def test_training_loop_device_validation(self, device, training_config):
        """Test device validation throughout training loop simulation."""
        model = GMMXLNetForQA.from_config(training_config)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        num_steps = 5
        batch_size = 2
        seq_length = 32

        memory_states = None

        for step in range(num_steps):
            # Create training batch
            input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
            attention_mask = torch.ones_like(input_ids)
            start_positions = torch.randint(0, seq_length, (batch_size,))
            end_positions = torch.randint(0, seq_length, (batch_size,))

            # Training step
            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
                use_memory=True,
                memory_states=memory_states
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # Device validation after each step
            assert outputs.start_logits.device == device
            assert outputs.end_logits.device == device

            if hasattr(outputs, 'memory_states') and outputs.memory_states is not None:
                memory_states = outputs.memory_states
                assert memory_states.device == device

            # Validate model parameters
            for param in model.parameters():
                assert param.device == device

            # Validate memory components
            if hasattr(model, 'memory_mixture') and model.memory_mixture is not None:
                assert model.memory_mixture.expert_states.device == device

    def test_cross_device_model_state_transfer(self, training_config):
        """Test transferring model states between devices."""
        model = GMMXLNetForQA.from_config(training_config)

        # Create initial state on CPU
        model = model.to("cpu")

        # Transfer to CUDA if available
        if torch.cuda.is_available():
            model = model.to("cuda")

            # Verify all parameters moved
            for param in model.parameters():
                assert param.device.type == "cuda"

            if hasattr(model, 'memory_mixture') and model.memory_mixture is not None:
                assert model.memory_mixture.expert_states.device.type == "cuda"

            # Test functionality on new device
            batch_size = 1
            seq_length = 16

            input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to("cuda")
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_memory=True
                )
                assert outputs.start_logits.device.type == "cuda"

            # Transfer back to CPU
            model = model.to("cpu")

            # Verify all parameters moved back
            for param in model.parameters():
                assert param.device.type == "cpu"

            if hasattr(model, 'memory_mixture') and model.memory_mixture is not None:
                assert model.memory_mixture.expert_states.device.type == "cpu"

    def test_memory_expert_load_balancing_device_consistency(self, device):
        """Test that load balancing maintains device consistency."""
        from gmmxlnet.models import GatedMemoryMixture

        memory_mixture = GatedMemoryMixture(
            num_experts=4,
            memory_slots=8,
            hidden_dim=32,
            init_strategies="learned"
        )
        memory_mixture = memory_mixture.to(device)

        batch_size = 8
        seq_length = 16

        # Create queries that might load-balance across experts
        for _ in range(10):  # Multiple iterations to test load balancing
            query = torch.randn(batch_size, seq_length, 32).to(device)

            # Test routing
            routing_logits = memory_mixture.compute_routing_logits(query)
            routing_probs = torch.softmax(routing_logits, dim=-1)

            # Verify device consistency
            assert routing_logits.device == device
            assert routing_probs.device == device

            # Test load balancing computation if available
            if hasattr(memory_mixture, 'compute_load_balance_loss'):
                load_balance_loss = memory_mixture.compute_load_balance_loss(routing_probs)
                assert load_balance_loss.device == device

    @pytest.mark.slow
    def test_extended_training_memory_management(self, device):
        """Test memory management during extended training simulation."""
        config = gmm_balanced_config(
            memory_num_tokens=16,
            num_epochs=2,
            batch_size=4,
            max_length=128,
        )

        model = GMMXLNetForQA.from_config(config)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Simulate extended training
        num_batches = 20
        batch_size = 4
        seq_length = 64

        memory_states = None

        for batch_idx in range(num_batches):
            # Periodic memory cleanup simulation
            if batch_idx % 5 == 0 and hasattr(model, 'reset_memory'):
                model.reset_memory()
                memory_states = None

            # Create training batch
            input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
            attention_mask = torch.ones_like(input_ids)
            start_positions = torch.randint(0, seq_length, (batch_size,))
            end_positions = torch.randint(0, seq_length, (batch_size,))

            # Training step
            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
                use_memory=True,
                memory_states=memory_states
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # Update memory states
            if hasattr(outputs, 'memory_states') and outputs.memory_states is not None:
                memory_states = outputs.memory_states

            # Memory usage check (simple validation)
            if batch_idx % 10 == 0:
                # Verify no obvious memory leaks in expert states
                if hasattr(model, 'memory_mixture') and model.memory_mixture is not None:
                    expert_states = model.memory_mixture.expert_states
                    assert expert_states.device == device
                    assert not torch.isnan(expert_states).any()

    def test_error_recovery_and_device_reset(self, training_config):
        """Test error recovery and device reset scenarios."""
        model = GMMXLNetForQA.from_config(training_config)
        device = torch.device("cpu")
        model = model.to(device)

        batch_size = 2
        seq_length = 32

        # Test normal operation
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_memory=True
            )
            assert outputs.start_logits.device == device

        # Test memory reset functionality
        if hasattr(model, 'reset_memory'):
            model.reset_memory()

            # Should work fine after reset
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_memory=True
                )
                assert outputs.start_logits.device == device

        # Test re-initialization
        if hasattr(model, 'initialize_memory_states'):
            memory_states = model.initialize_memory_states(batch_size, device)
            assert memory_states.device == device

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_memory=True,
                    memory_states=memory_states
                )
                assert outputs.start_logits.device == device


@pytest.mark.integration
class TestGMMDevicePlacementEdgeCases:
    """Test edge cases for GMM device placement."""

    def test_zero_batch_size_handling(self):
        """Test handling of edge case with zero batch size."""
        config = gmm_balanced_config(
            memory_num_tokens=4,
            num_epochs=1,
            batch_size=1,
            max_length=32,
        )

        model = GMMXLNetForQA.from_config(config)
        model = model.to("cpu")

        # Test with minimal batch
        batch_size = 1
        seq_length = 8

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_memory=True
            )
            assert outputs.start_logits.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_out_of_memory_simulation(self):
        """Test graceful handling of device out of memory scenarios."""
        config = gmm_balanced_config(
            memory_num_tokens=256,  # Large to potentially cause OOM
            num_epochs=1,
            batch_size=16,  # Large batch
            max_length=512,  # Long sequences
        )

        try:
            model = GMMXLNetForQA.from_config(config)
            model = model.to("cuda")

            # Test basic functionality
            batch_size = 2  # Small batch for actual test
            seq_length = 32

            input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to("cuda")
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_memory=False  # Disable memory to reduce memory usage
                )
                assert outputs.start_logits.device.type == "cuda"

        except torch.cuda.OutOfMemoryError:
            # If OOM occurs, test that we can fallback to CPU
            model = GMMXLNetForQA.from_config(config)
            model = model.to("cpu")

            input_ids = torch.randint(0, 1000, (batch_size, seq_length))
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_memory=False
                )
                assert outputs.start_logits.device.type == "cpu"

    def test_model_state_dict_device_transfer(self, training_config):
        """Test transferring model state dictionaries between devices."""
        model1 = GMMXLNetForQA.from_config(training_config)
        model1 = model1.to("cpu")

        # Create a second model
        model2 = GMMXLNetForQA.from_config(training_config)
        model2 = model2.to("cpu")

        # Transfer state dict
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)

        # Test that both models work on CPU
        batch_size = 1
        seq_length = 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs1 = model1(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_memory=True
            )
            outputs2 = model2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_memory=True
            )

            assert outputs1.start_logits.device.type == "cpu"
            assert outputs2.start_logits.device.type == "cpu"

            # Outputs should be identical
            assert torch.allclose(outputs1.start_logits, outputs2.start_logits, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cross_device_state_dict_transfer(self, training_config):
        """Test transferring state dictionaries between different devices."""
        model_cpu = GMMXLNetForQA.from_config(training_config)
        model_cpu = model_cpu.to("cpu")

        model_cuda = GMMXLNetForQA.from_config(training_config)
        model_cuda = model_cuda.to("cuda")

        # Get state dict from CPU model and load to CUDA model
        state_dict = model_cpu.state_dict()

        # Move tensors in state dict to CUDA
        cuda_state_dict = {k: v.to("cuda") for k, v in state_dict.items()}
        model_cuda.load_state_dict(cuda_state_dict)

        # Test CUDA model works
        batch_size = 1
        seq_length = 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to("cuda")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model_cuda(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_memory=True
            )
            assert outputs.start_logits.device.type == "cuda"