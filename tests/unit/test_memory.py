"""Tests for enhanced memory modules and MA-XLNet functionality.

This test suite ensures backward compatibility while validating
the new differentiable memory features for multi-hop reasoning.
"""

import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from memxlnet.models.memory_modules import DifferentiableMemory, MemoryController
from memxlnet.models.memxlnet_qa import MemXLNetForQA


class TestDifferentiableMemory(unittest.TestCase):
    """Test the DifferentiableMemory class."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_slots = 16
        self.slot_dim = 128
        self.num_heads = 2
        self.batch_size = 4

        self.memory = DifferentiableMemory(
            num_slots=self.num_slots,
            slot_dim=self.slot_dim,
            num_heads=self.num_heads,
            sharpness=1.0,
            enable_usage_tracking=True,
            enable_temporal_links=True,
        )

    def test_initialization(self):
        """Test memory initialization."""
        self.assertEqual(self.memory.num_slots, self.num_slots)
        self.assertEqual(self.memory.slot_dim, self.slot_dim)
        self.assertEqual(self.memory.num_heads, self.num_heads)
        self.assertEqual(self.memory.memory.shape, (self.num_slots, self.slot_dim))
        self.assertTrue(torch.allclose(self.memory.memory, torch.zeros(self.num_slots, self.slot_dim)))

    def test_content_addressing(self):
        """Test content-based addressing."""
        # Create random key
        key = torch.randn(self.batch_size, self.num_heads, self.slot_dim)

        # Get weights
        weights = self.memory.content_addressing(key)

        # Check shape
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, self.num_slots))

        # Check weights sum to 1
        weight_sums = weights.sum(dim=-1)
        expected_sums = torch.ones(self.batch_size, self.num_heads)
        self.assertTrue(torch.allclose(weight_sums, expected_sums, atol=1e-6))

        # Check weights are positive
        self.assertTrue((weights >= 0).all())

    def test_read_operation(self):
        """Test memory read operation."""
        # Initialize memory with some values
        self.memory.memory = torch.randn(self.num_slots, self.slot_dim)

        # Create weights
        weights = torch.softmax(torch.randn(self.batch_size, self.num_heads, self.num_slots), dim=-1)

        # Read from memory
        read_vectors = self.memory.read(weights)

        # Check shape
        self.assertEqual(read_vectors.shape, (self.batch_size, self.num_heads, self.slot_dim))

        # Verify read is weighted sum
        expected = torch.matmul(weights, self.memory.memory)
        self.assertTrue(torch.allclose(read_vectors, expected, atol=1e-6))

    def test_write_operation(self):
        """Test memory write operation."""
        # Initialize memory
        initial_memory = self.memory.memory.clone()

        # Create write parameters
        weights = torch.softmax(torch.randn(self.batch_size, self.num_heads, self.num_slots), dim=-1)
        write_vector = torch.randn(self.batch_size, self.num_heads, self.slot_dim)

        # Write to memory
        self.memory.write(weights, write_vector)

        # Check memory has changed
        self.assertFalse(torch.allclose(self.memory.memory, initial_memory))

        # Check memory shape is preserved
        self.assertEqual(self.memory.memory.shape, (self.num_slots, self.slot_dim))

    def test_usage_tracking(self):
        """Test memory usage tracking."""
        # Perform some write operations
        for _ in range(3):
            weights = torch.softmax(torch.randn(self.batch_size, self.num_heads, self.num_slots), dim=-1)
            write_vector = torch.randn(self.batch_size, self.num_heads, self.slot_dim)
            self.memory.write(weights, write_vector)

        # Check usage has been updated
        self.assertTrue((self.memory.usage >= 0).all())
        self.assertTrue((self.memory.usage <= 1).all())
        self.assertFalse(torch.allclose(self.memory.usage, torch.zeros(self.num_slots)))

    def test_get_usage_weights(self):
        """Test least-used memory allocation."""
        # Set some usage
        self.memory.usage = torch.tensor([0.9, 0.1, 0.5, 0.2] + [0.0] * (self.num_slots - 4))

        # Get allocation weights
        allocation = self.memory.get_usage_weights()

        # Check least used slot gets highest weight
        self.assertEqual(allocation.argmax().item(), 4)  # First zero usage slot

    def test_reset(self):
        """Test memory reset."""
        # Modify memory state
        self.memory.memory = torch.randn(self.num_slots, self.slot_dim)
        self.memory.usage = torch.rand(self.num_slots)

        # Reset
        self.memory.reset()

        # Check everything is zeroed
        self.assertTrue(torch.allclose(self.memory.memory, torch.zeros(self.num_slots, self.slot_dim)))
        self.assertTrue(torch.allclose(self.memory.usage, torch.zeros(self.num_slots)))


class TestMemoryController(unittest.TestCase):
    """Test the MemoryController class."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 768
        self.memory_slots = 16
        self.memory_dim = 128
        self.num_heads = 2
        self.batch_size = 4

        self.controller = MemoryController(
            input_dim=self.input_dim,
            memory_slots=self.memory_slots,
            memory_dim=self.memory_dim,
            num_heads=self.num_heads,
            use_temporal_links=True,
            use_usage_tracking=True,
            sharpness=1.0,
        )

    def test_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.input_dim, self.input_dim)
        self.assertEqual(self.controller.memory_slots, self.memory_slots)
        self.assertEqual(self.controller.memory_dim, self.memory_dim)
        self.assertEqual(self.controller.num_heads, self.num_heads)
        self.assertIsNotNone(self.controller.memory)

    def test_forward_pass(self):
        """Test forward pass through controller."""
        # Create input
        input_state = torch.randn(self.batch_size, self.input_dim)

        # Forward pass
        output, memory_info = self.controller(input_state)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.input_dim))

        # Check memory_info contains expected keys
        expected_keys = [
            "read_weights",
            "write_weights",
            "read_vectors",
            "write_vectors",
            "memory_state",
            "usage",
            "temporal_links",
        ]
        for key in expected_keys:
            self.assertIn(key, memory_info)

    def test_memory_state_management(self):
        """Test getting and setting memory state."""
        # Get initial state
        initial_state = self.controller.get_memory_state()
        self.assertEqual(initial_state.shape, (self.memory_slots, self.memory_dim))

        # Set new state
        new_state = torch.randn(self.memory_slots, self.memory_dim)
        self.controller.set_memory_state(new_state)

        # Verify state was set
        retrieved_state = self.controller.get_memory_state()
        self.assertTrue(torch.allclose(retrieved_state, new_state))

    def test_reset_memory(self):
        """Test memory reset."""
        # Modify memory
        self.controller.set_memory_state(torch.randn(self.memory_slots, self.memory_dim))

        # Reset
        self.controller.reset_memory()

        # Check memory is zeroed
        state = self.controller.get_memory_state()
        self.assertTrue(torch.allclose(state, torch.zeros(self.memory_slots, self.memory_dim)))

    def test_visualization_data(self):
        """Test visualization data generation."""
        viz_data = self.controller.visualize_memory()

        # Check expected keys
        self.assertIn("memory", viz_data)
        self.assertIn("usage", viz_data)
        self.assertIn("temporal_links", viz_data)

        # Check data types
        self.assertIsInstance(viz_data["memory"], np.ndarray)
        self.assertIsInstance(viz_data["usage"], np.ndarray)
        self.assertIsInstance(viz_data["temporal_links"], np.ndarray)


class TestMemXLNetIntegration(unittest.TestCase):
    """Test integration of enhanced memory with MemXLNetForQA."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a small base model for testing
        self.base_model = nn.Sequential(
            nn.Linear(768, 768),
            nn.Linear(768, 2),  # For start/end logits
        )
        # Mock config
        self.base_model.config = type("config", (), {"d_model": 768, "cls_token_id": 3})()

    def test_backward_compatibility(self):
        """Test that existing token-based memory still works."""
        # Create wrapper with token-based memory (default)
        wrapper = MemXLNetForQA(
            base_model=self.base_model,
            mem_token_count=4,
            memory_init="learned",
            memory_update="gated",
            use_differentiable_memory=False,  # Use token-based
        )

        # Check it doesn't have differentiable memory controller
        self.assertIsNone(wrapper.memory_controller)

        # Check it has token-based components
        self.assertTrue(hasattr(wrapper, "learned_memory"))
        self.assertEqual(wrapper.mem_token_count, 4)

    def test_differentiable_memory_creation(self):
        """Test creation with differentiable memory."""
        # Create wrapper with differentiable memory
        wrapper = MemXLNetForQA(
            base_model=self.base_model,
            mem_token_count=4,
            memory_init="learned",
            memory_update="gated",
            use_differentiable_memory=True,
            num_memory_heads=2,
            memory_sharpness=1.5,
            enable_usage_tracking=True,
            enable_temporal_links=True,
            memory_slots=32,
        )

        # Check it has differentiable memory controller
        self.assertIsNotNone(wrapper.memory_controller)
        self.assertEqual(wrapper.memory_controller.num_heads, 2)
        self.assertEqual(wrapper.memory_controller.memory_slots, 32)

    def test_save_load_with_enhanced_memory(self):
        """Test saving and loading model with enhanced memory."""
        # Create wrapper with differentiable memory
        wrapper = MemXLNetForQA(
            base_model=self.base_model,
            mem_token_count=4,
            use_differentiable_memory=True,
            num_memory_heads=2,
            memory_slots=16,
        )

        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper.save_pretrained(tmpdir)

            # Check files were created
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "memxlnet_config.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "memxlnet_state.pt")))

            # Load model
            # Note: In real scenario, base model would be XLNetForQuestionAnsweringSimple
            # For test, we'll check config loading
            import json

            with open(os.path.join(tmpdir, "memxlnet_config.json")) as f:
                config = json.load(f)

            # Check enhanced memory settings were saved
            self.assertEqual(config["use_differentiable_memory"], True)
            self.assertEqual(config["num_memory_heads"], 2)
            self.assertEqual(config["memory_slots"], 16)
            self.assertEqual(config["version"], 3)  # New version


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency of enhanced memory."""

    def test_memory_scaling(self):
        """Test that memory usage scales as expected."""
        slot_dims = [64, 128, 256]
        num_slots_list = [8, 16, 32]

        for slot_dim in slot_dims:
            for num_slots in num_slots_list:
                memory = DifferentiableMemory(num_slots=num_slots, slot_dim=slot_dim, num_heads=1)

                # Calculate expected memory size
                expected_params = num_slots * slot_dim  # Memory matrix
                if memory.enable_usage_tracking:
                    expected_params += num_slots  # Usage vector
                if memory.enable_temporal_links:
                    expected_params += num_slots * num_slots  # Temporal matrix

                # Count actual parameters
                actual_params = sum(p.numel() for p in memory.buffers())

                # Allow some overhead for other small tensors
                self.assertLessEqual(actual_params, expected_params * 1.1)

    def test_constant_memory_per_step(self):
        """Test that memory usage per step is constant."""
        controller = MemoryController(input_dim=768, memory_slots=16, memory_dim=128, num_heads=2)

        batch_size = 4
        input_state = torch.randn(batch_size, 768)

        # Process multiple steps
        memory_states = []
        for _ in range(5):
            output, info = controller(input_state)
            memory_states.append(info["memory_state"].clone())

        # Check memory dimensions remain constant
        for state in memory_states:
            self.assertEqual(state.shape, (16, 128))


if __name__ == "__main__":
    unittest.main()
