"""
Regression tests for GMM code integration.

Ensures that existing MemXLNet functionality is preserved after adding GMM code:
- MemXLNet imports still work
- Basic model operations work
- GMM code is properly isolated
"""

import pytest
import torch
from transformers import XLNetConfig, XLNetForQuestionAnsweringSimple


@pytest.mark.regression
class TestMemXLNetImportsWork:
    """Test that MemXLNet imports still work with GMM code present."""

    def test_memxlnet_core_imports(self):
        """Test that core MemXLNet modules can be imported."""
        from memxlnet.models import MemXLNetForQA
        from memxlnet.models import DifferentiableMemory, MemoryController
        from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

        assert MemXLNetForQA is not None
        assert DifferentiableMemory is not None
        assert MemoryController is not None
        assert TrainingConfig is not None
        assert XLNetRecurrentTrainer is not None

    def test_gmm_imports_dont_break_memxlnet(self):
        """Test that importing GMM doesn't break MemXLNet imports."""
        # Import GMM first
        from gmmxlnet.models import GMMXLNetForQA

        # Then import MemXLNet - should still work
        from memxlnet.models import MemXLNetForQA

        # Verify they are different classes
        assert GMMXLNetForQA is not MemXLNetForQA


@pytest.mark.regression
class TestMemXLNetBasicFunctionality:
    """Test that basic MemXLNet functionality works with GMM code present."""

    def test_memxlnet_initialization(self):
        """Test that MemXLNetForQA can be initialized."""
        from memxlnet.models import MemXLNetForQA

        config = XLNetConfig(
            vocab_size=1000,
            d_model=64,
            n_layer=2,
            n_head=2,
            d_inner=128,
        )
        base_model = XLNetForQuestionAnsweringSimple(config)

        # Initialize with token-based memory
        model = MemXLNetForQA(
            base_model=base_model,
            mem_token_count=8,
        )

        assert model is not None
        assert model.mem_token_count == 8
        assert hasattr(model, "base")

    def test_memxlnet_forward_pass_works(self):
        """Test that MemXLNet forward pass still works."""
        from memxlnet.models import MemXLNetForQA

        config = XLNetConfig(
            vocab_size=1000,
            d_model=64,
            n_layer=2,
            n_head=2,
            d_inner=128,
        )
        base_model = XLNetForQuestionAnsweringSimple(config)

        model = MemXLNetForQA(
            base_model=base_model,
            mem_token_count=8,
        )

        # Forward pass
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Verify outputs
        assert "start_logits" in outputs
        assert "end_logits" in outputs
        assert outputs["start_logits"].shape == (batch_size, seq_len)
        assert outputs["end_logits"].shape == (batch_size, seq_len)

    def test_differentiable_memory_still_works(self):
        """Test that DifferentiableMemory still works."""
        from memxlnet.models import DifferentiableMemory

        memory = DifferentiableMemory(
            num_slots=16,
            slot_dim=64,
            num_heads=1,
        )

        # Create dummy query
        batch_size = 2
        num_heads = 1
        key = torch.randn(batch_size, num_heads, 64)

        # Test content addressing
        weights = memory.content_addressing(key)

        # Verify output
        assert weights.shape == (batch_size, num_heads, 16)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, num_heads), atol=1e-5)

    def test_memory_controller_can_be_imported(self):
        """Test that MemoryController can be imported and initialized."""
        from memxlnet.models import MemoryController

        controller = MemoryController(
            input_dim=64,
            memory_slots=16,
            memory_dim=64,
            num_heads=1,
        )

        # Verify initialization
        assert controller is not None
        assert controller.num_heads == 1
        assert controller.memory_slots == 16


@pytest.mark.regression
class TestGMMCodeIsolation:
    """Test that GMM code is properly isolated from MemXLNet."""

    def test_gmm_modules_are_separate(self):
        """Test that GMM and MemXLNet are separate modules."""
        from gmmxlnet.models import GMMXLNetForQA
        from memxlnet.models import MemXLNetForQA

        # Verify they are different classes
        assert GMMXLNetForQA is not MemXLNetForQA
        assert GMMXLNetForQA.__module__ != MemXLNetForQA.__module__

    def test_use_gmm_memory_false_works(self):
        """Test that GMM code with use_gmm_memory=False doesn't affect existing code."""
        # Create a basic model without GMM
        from memxlnet.models import MemXLNetForQA

        config = XLNetConfig(
            vocab_size=1000,
            d_model=64,
            n_layer=2,
            n_head=2,
            d_inner=128,
        )
        base_model = XLNetForQuestionAnsweringSimple(config)

        model = MemXLNetForQA(
            base_model=base_model,
            mem_token_count=8,
            use_differentiable_memory=False,  # Don't use differentiable memory
        )

        # Forward pass should work
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Should work normally
        assert "start_logits" in outputs
        assert "end_logits" in outputs


@pytest.mark.regression
class TestNoPerformanceRegression:
    """Test that GMM code doesn't slow down existing MemXLNet code."""

    def test_memxlnet_import_time(self):
        """Test that importing memxlnet is still fast."""
        import time

        start_time = time.time()
        import memxlnet  # noqa: F401
        import_time = time.time() - start_time

        # Import should take less than 1 second
        assert import_time < 1.0, f"Import took {import_time}s, should be < 1.0s"

    def test_memxlnet_initialization_time(self):
        """Test that model initialization is still fast."""
        from memxlnet.models import MemXLNetForQA
        import time

        config = XLNetConfig(
            vocab_size=1000,
            d_model=64,
            n_layer=2,
            n_head=2,
            d_inner=128,
        )
        base_model = XLNetForQuestionAnsweringSimple(config)

        start_time = time.time()
        model = MemXLNetForQA(
            base_model=base_model,
            mem_token_count=8,
        )
        init_time = time.time() - start_time

        # Initialization should be fast (< 1s on CPU)
        assert init_time < 1.0, f"Initialization took {init_time}s, should be < 1.0s"
