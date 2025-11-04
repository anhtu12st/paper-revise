"""
Integration tests for Memory Gating Network with existing components.

Verifies that the router integrates correctly without affecting existing functionality.
"""

import torch

from gmmxlnet.models.gating_network import MemoryGatingNetwork


class TestRouterIndependence:
    """Test IV1: Router operates independently without affecting existing memory mechanisms."""

    def test_router_operates_independently(self):
        """Test that router can operate without dependencies on other components."""
        # Router should work standalone
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Should be able to process random hiddens without any other components
        hiddens = torch.randn(8, 16, 768)
        routing_probs, routing_logits, routing_entropy = router(hiddens)

        # Verify outputs are valid
        assert routing_probs.shape == (8, 4)
        assert routing_logits.shape == (8, 4)
        assert routing_entropy.shape == (8,)

        # Verify probabilities are valid distributions
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(8), atol=1e-6)
        assert torch.all(routing_probs >= 0.0)
        assert torch.all(routing_probs <= 1.0)

    def test_router_does_not_modify_input(self):
        """Test that router does not modify input tensors."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Create input and clone it
        hiddens = torch.randn(8, 16, 768)
        hiddens_original = hiddens.clone()

        # Run router
        router(hiddens)

        # Input should be unchanged
        assert torch.allclose(hiddens, hiddens_original)

    def test_router_multiple_calls_consistent(self):
        """Test that router produces consistent results across multiple calls."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Same input should produce same output
        hiddens = torch.randn(8, 16, 768)

        probs1, logits1, entropy1 = router(hiddens)
        probs2, logits2, entropy2 = router(hiddens)

        assert torch.allclose(probs1, probs2)
        assert torch.allclose(logits1, logits2)
        assert torch.allclose(entropy1, entropy2)


class TestOptimizerParameterIdentification:
    """Test IV2: Routing network parameters correctly identified by optimizer groups."""

    def test_routing_parameters_identifiable(self):
        """Test that routing parameters can be identified for optimizer groups."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Get all parameters
        all_params = list(router.parameters())

        # Should have weight and bias from nn.Linear
        assert len(all_params) == 2  # weight + bias

        # Check shapes
        assert all_params[0].shape == (4, 768)  # weight: (num_experts, hidden_dim)
        assert all_params[1].shape == (4,)  # bias: (num_experts,)

        # All should require gradients
        for param in all_params:
            assert param.requires_grad

    def test_routing_parameters_optimizer_grouping(self):
        """Test that routing parameters can be grouped for optimizer."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Simulate creating optimizer groups
        routing_params = [p for p in router.parameters() if p.requires_grad]

        # Should be able to create parameter groups
        param_groups = [
            {"params": routing_params, "lr": 1e-4, "name": "routing"},
        ]

        # Verify parameter group structure
        assert len(param_groups) == 1
        assert param_groups[0]["lr"] == 1e-4
        assert param_groups[0]["name"] == "routing"
        assert len(param_groups[0]["params"]) == 2

    def test_routing_parameters_named(self):
        """Test that routing parameters have proper names."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Get named parameters
        named_params = dict(router.named_parameters())

        # Should have routing_projection.weight and routing_projection.bias
        assert "routing_projection.weight" in named_params
        assert "routing_projection.bias" in named_params

        # Check shapes
        assert named_params["routing_projection.weight"].shape == (4, 768)
        assert named_params["routing_projection.bias"].shape == (4,)


class TestParameterCount:
    """Test IV3: Model parameter count increases by expected amount."""

    def test_parameter_count_k4_d768(self):
        """Test parameter count for k=4, d=768 configuration."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Count parameters
        total_params = sum(p.numel() for p in router.parameters())

        # Expected: (k × d) + k = (4 × 768) + 4 = 3072 + 4 = 3076
        expected_params = (4 * 768) + 4
        assert total_params == expected_params

        # Should be << 100K
        assert total_params < 100_000
        print(f"Total parameters for k=4, d=768: {total_params:,}")

    def test_parameter_count_k2_d768(self):
        """Test parameter count for k=2, d=768 configuration."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=2, temperature=1.0)

        # Count parameters
        total_params = sum(p.numel() for p in router.parameters())

        # Expected: (k × d) + k = (2 × 768) + 2 = 1536 + 2 = 1538
        expected_params = (2 * 768) + 2
        assert total_params == expected_params

        # Should be << 100K
        assert total_params < 100_000

    def test_parameter_count_k8_d768(self):
        """Test parameter count for k=8, d=768 configuration."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=8, temperature=1.0)

        # Count parameters
        total_params = sum(p.numel() for p in router.parameters())

        # Expected: (k × d) + k = (8 × 768) + 8 = 6144 + 8 = 6152
        expected_params = (8 * 768) + 8
        assert total_params == expected_params

        # Should be << 100K
        assert total_params < 100_000

    def test_parameter_count_scaling(self):
        """Test that parameter count scales linearly with num_experts."""
        hidden_dim = 768

        # Test different expert counts
        param_counts = {}
        for num_experts in [2, 4, 8]:
            router = MemoryGatingNetwork(hidden_dim=hidden_dim, num_experts=num_experts, temperature=1.0)
            param_counts[num_experts] = sum(p.numel() for p in router.parameters())

        # Verify linear scaling
        # Doubling experts should approximately double parameters
        ratio_2_to_4 = param_counts[4] / param_counts[2]
        ratio_4_to_8 = param_counts[8] / param_counts[4]

        # Should be close to 2x (within 10% tolerance due to bias term)
        assert 1.9 < ratio_2_to_4 < 2.1
        assert 1.9 < ratio_4_to_8 < 2.1


class TestGradientFlowIntegration:
    """Test gradient flow through router in integration scenarios."""

    def test_gradient_flow_to_inputs(self):
        """Test that gradients flow back to input hiddens."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)
        hiddens = torch.randn(8, 16, 768, requires_grad=True)

        # Forward pass
        routing_probs, routing_logits, routing_entropy = router(hiddens)

        # Compute loss and backward
        loss = routing_probs.sum() + routing_entropy.sum()
        loss.backward()

        # Check gradients exist and are valid
        assert hiddens.grad is not None
        assert not torch.isnan(hiddens.grad).any()
        assert not torch.isinf(hiddens.grad).any()

    def test_gradient_flow_to_router_parameters(self):
        """Test that gradients flow to router parameters."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)
        hiddens = torch.randn(8, 16, 768, requires_grad=True)

        # Forward pass
        routing_probs, _, _ = router(hiddens)

        # Compute loss and backward
        loss = routing_probs.sum()
        loss.backward()

        # Check router parameters have gradients
        for name, param in router.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"

    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly across multiple backward passes."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # First backward pass
        hiddens1 = torch.randn(8, 16, 768)
        probs1, _, _ = router(hiddens1)
        loss1 = probs1.sum()
        loss1.backward()

        # Save gradients
        grad1 = router.routing_projection.weight.grad.clone()

        # Second backward pass (accumulate)
        hiddens2 = torch.randn(8, 16, 768)
        probs2, _, _ = router(hiddens2)
        loss2 = probs2.sum()
        loss2.backward()

        # Gradients should have accumulated
        grad2 = router.routing_projection.weight.grad

        # grad2 should be different from grad1 (accumulated)
        assert not torch.allclose(grad1, grad2)


class TestMemoryEfficiency:
    """Test memory efficiency and overhead of router."""

    def test_router_memory_footprint_small(self):
        """Test that router has small memory footprint."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Count parameter memory (float32)
        total_params = sum(p.numel() for p in router.parameters())
        memory_bytes = total_params * 4  # 4 bytes per float32

        # Should be < 100KB for parameters
        assert memory_bytes < 100_000

        print(f"Router memory footprint: {memory_bytes:,} bytes ({memory_bytes / 1024:.2f} KB)")

    def test_router_forward_pass_efficient(self):
        """Test that router forward pass is efficient (no memory leaks)."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Run multiple forward passes
        for _ in range(10):
            hiddens = torch.randn(8, 16, 768)
            routing_probs, routing_logits, routing_entropy = router(hiddens)

            # Clean up
            del routing_probs, routing_logits, routing_entropy, hiddens

        # If we get here without OOM, the router is memory-efficient


class TestRouterConfiguration:
    """Test router configuration options."""

    def test_different_hidden_dims(self):
        """Test router with different hidden dimensions."""
        for hidden_dim in [512, 768, 1024]:
            router = MemoryGatingNetwork(hidden_dim=hidden_dim, num_experts=4, temperature=1.0)
            hiddens = torch.randn(8, 16, hidden_dim)
            probs, _, _ = router(hiddens)

            assert probs.shape == (8, 4)
            assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-6)

    def test_different_temperatures(self):
        """Test router with different temperature settings."""
        for temperature in [0.1, 0.5, 1.0, 2.0, 5.0]:
            router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=temperature)
            hiddens = torch.randn(8, 16, 768)
            probs, _, _ = router(hiddens)

            assert probs.shape == (8, 4)
            assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-6)

    def test_different_num_experts(self):
        """Test router with different expert counts."""
        for num_experts in [2, 4, 8]:
            router = MemoryGatingNetwork(hidden_dim=768, num_experts=num_experts, temperature=1.0)
            hiddens = torch.randn(8, 16, 768)
            probs, _, _ = router(hiddens)

            assert probs.shape == (8, num_experts)
            assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-6)
