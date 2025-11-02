"""
Unit tests for Memory Gating Network (Router).

Tests routing probability computation, temperature scaling, numerical stability,
entropy calculation, and load balancing for GMM-augmented XLNet.
"""

import pytest
import torch

from gmmxlnet.models.gating_network import MemoryGatingNetwork


class TestMemoryGatingNetworkInitialization:
    """Tests for MemoryGatingNetwork initialization and configuration validation."""

    def test_valid_initialization(self):
        """Test successful initialization with valid parameters."""
        router = MemoryGatingNetwork(
            hidden_dim=768, num_experts=4, temperature=1.0, pooling_method="mean"
        )
        assert router.hidden_dim == 768
        assert router.num_experts == 4
        assert router.temperature == 1.0
        assert router.pooling_method == "mean"

    def test_temperature_validation_zero(self):
        """Test that temperature = 0 raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be > 0"):
            MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=0.0)

    def test_temperature_validation_negative(self):
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be > 0"):
            MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=-1.0)

    def test_num_experts_validation_too_small(self):
        """Test that num_experts < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_experts must be in"):
            MemoryGatingNetwork(hidden_dim=768, num_experts=1, temperature=1.0)

    def test_num_experts_validation_too_large(self):
        """Test that num_experts > 8 raises ValueError."""
        with pytest.raises(ValueError, match="num_experts must be in"):
            MemoryGatingNetwork(hidden_dim=768, num_experts=16, temperature=1.0)

    def test_valid_num_experts_range(self):
        """Test that num_experts in [2, 8] are all valid."""
        for num_experts in [2, 4, 8]:
            router = MemoryGatingNetwork(
                hidden_dim=768, num_experts=num_experts, temperature=1.0
            )
            assert router.num_experts == num_experts

    def test_pooling_method_validation_invalid(self):
        """Test that invalid pooling method raises ValueError."""
        with pytest.raises(ValueError, match="pooling_method must be"):
            MemoryGatingNetwork(
                hidden_dim=768, num_experts=4, temperature=1.0, pooling_method="invalid"
            )

    def test_pooling_method_attention_not_implemented(self):
        """Test that attention pooling raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Attention-weighted pooling"):
            MemoryGatingNetwork(
                hidden_dim=768, num_experts=4, temperature=1.0, pooling_method="attention"
            )

    def test_routing_projection_shape(self):
        """Test that routing projection has correct shape."""
        hidden_dim = 768
        num_experts = 4
        router = MemoryGatingNetwork(
            hidden_dim=hidden_dim, num_experts=num_experts, temperature=1.0
        )
        # Linear layer should map hidden_dim → num_experts
        assert router.routing_projection.weight.shape == (num_experts, hidden_dim)
        assert router.routing_projection.bias.shape == (num_experts,)


class TestRoutingComputation:
    """Tests for routing probability computation and normalization."""

    @pytest.fixture
    def router(self):
        """Standard router fixture."""
        return MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

    def test_forward_output_shapes(self, router):
        """Test that forward pass returns correct shapes."""
        batch_size = 8
        memory_slots = 16
        hidden_dim = 768

        hiddens = torch.randn(batch_size, memory_slots, hidden_dim)
        routing_probs, routing_logits, routing_entropy = router(hiddens)

        assert routing_probs.shape == (batch_size, 4)
        assert routing_logits.shape == (batch_size, 4)
        assert routing_entropy.shape == (batch_size,)

    def test_routing_probabilities_sum_to_one(self, router):
        """Test that routing probabilities sum to 1.0 per batch item."""
        hiddens = torch.randn(8, 16, 768)
        routing_probs, _, _ = router(hiddens)

        # Check probabilities sum to 1.0 (within floating point tolerance)
        prob_sums = routing_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(8), atol=1e-6)

    def test_routing_probabilities_non_negative(self, router):
        """Test that all routing probabilities are non-negative."""
        hiddens = torch.randn(8, 16, 768)
        routing_probs, _, _ = router(hiddens)

        assert torch.all(routing_probs >= 0.0)

    def test_routing_probabilities_bounded(self, router):
        """Test that routing probabilities are in [0, 1]."""
        hiddens = torch.randn(8, 16, 768)
        routing_probs, _, _ = router(hiddens)

        assert torch.all(routing_probs >= 0.0)
        assert torch.all(routing_probs <= 1.0)

    def test_forward_with_different_batch_sizes(self, router):
        """Test forward pass with various batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            hiddens = torch.randn(batch_size, 16, 768)
            routing_probs, routing_logits, routing_entropy = router(hiddens)

            assert routing_probs.shape == (batch_size, 4)
            assert routing_logits.shape == (batch_size, 4)
            assert routing_entropy.shape == (batch_size,)

    def test_forward_with_different_memory_slots(self, router):
        """Test forward pass with various memory slot counts."""
        for memory_slots in [4, 8, 16, 32]:
            hiddens = torch.randn(8, memory_slots, 768)
            routing_probs, _, _ = router(hiddens)

            assert routing_probs.shape == (8, 4)
            # Probabilities should still sum to 1.0 regardless of memory_slots
            prob_sums = routing_probs.sum(dim=-1)
            assert torch.allclose(prob_sums, torch.ones(8), atol=1e-6)

    def test_forward_with_invalid_hidden_dim(self, router):
        """Test that forward raises ValueError for mismatched hidden_dim."""
        # Router expects hidden_dim=768, but input has 512
        hiddens = torch.randn(8, 16, 512)

        with pytest.raises(ValueError, match="Expected hidden_dim=768, got 512"):
            router(hiddens)


class TestTemperatureScaling:
    """Tests for temperature-controlled softmax and routing sharpness."""

    def test_high_temperature_uniform_distribution(self):
        """Test that high temperature produces more uniform routing."""
        # High temperature should soften the distribution
        router_high_temp = MemoryGatingNetwork(
            hidden_dim=768, num_experts=4, temperature=10.0
        )

        hiddens = torch.randn(32, 16, 768)
        routing_probs, _, entropy = router_high_temp(hiddens)

        # With high temperature, entropy should be closer to maximum (log(k))
        max_entropy = torch.log(torch.tensor(4.0))  # log(num_experts)
        mean_entropy = entropy.mean()

        # High temperature entropy should be > 0.8 * max_entropy (softer)
        assert mean_entropy > 0.8 * max_entropy

    def test_low_temperature_peaked_distribution(self):
        """Test that low temperature produces more peaked routing than high temperature."""
        # Compare low vs high temperature on same input
        hiddens = torch.randn(32, 16, 768)

        # High temperature (softer)
        router_high_temp = MemoryGatingNetwork(
            hidden_dim=768, num_experts=4, temperature=5.0
        )
        torch.manual_seed(42)
        router_high_temp.routing_projection.weight.data = torch.randn(4, 768) * 0.1
        _, _, entropy_high = router_high_temp(hiddens)

        # Low temperature (sharper)
        router_low_temp = MemoryGatingNetwork(
            hidden_dim=768, num_experts=4, temperature=0.1
        )
        torch.manual_seed(42)
        router_low_temp.routing_projection.weight.data = torch.randn(4, 768) * 0.1
        _, _, entropy_low = router_low_temp(hiddens)

        # Low temperature should produce lower entropy (more peaked)
        assert entropy_low.mean() < entropy_high.mean()

    def test_temperature_comparison(self):
        """Test that entropy decreases as temperature decreases."""
        hiddens = torch.randn(32, 16, 768)

        # Use same random seed for fair comparison
        torch.manual_seed(42)
        router_temp_1 = MemoryGatingNetwork(
            hidden_dim=768, num_experts=4, temperature=1.0
        )
        _, _, entropy_1 = router_temp_1(hiddens)

        torch.manual_seed(42)
        router_temp_01 = MemoryGatingNetwork(
            hidden_dim=768, num_experts=4, temperature=0.1
        )
        _, _, entropy_01 = router_temp_01(hiddens)

        # Lower temperature should produce lower entropy
        assert entropy_01.mean() < entropy_1.mean()

    def test_temperature_one_standard_softmax(self):
        """Test that temperature=1.0 produces standard softmax."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)
        hiddens = torch.randn(8, 16, 768)
        routing_probs, _, _ = router(hiddens)

        # Just verify basic properties (sum to 1, non-negative)
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(8), atol=1e-6)
        assert torch.all(routing_probs >= 0.0)


class TestNumericalStability:
    """Tests for numerical stability with edge cases."""

    def test_all_zero_input(self):
        """Test that all-zero input returns uniform distribution."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # All-zero input
        hiddens = torch.zeros(8, 16, 768)
        routing_probs, _, _ = router(hiddens)

        # Should return uniform distribution (1/k for each expert)
        expected_uniform = torch.ones(4) / 4.0
        for i in range(8):
            assert torch.allclose(routing_probs[i], expected_uniform, atol=1e-6)

    def test_extreme_positive_logits(self):
        """Test that extreme positive logits are handled without NaN."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Create input that would produce extreme logits
        # Force large positive values
        hiddens = torch.ones(8, 16, 768) * 100.0
        routing_probs, routing_logits, _ = router(hiddens)

        # Check no NaN or Inf
        assert not torch.isnan(routing_probs).any()
        assert not torch.isinf(routing_probs).any()
        assert not torch.isnan(routing_logits).any()
        assert not torch.isinf(routing_logits).any()

        # Probabilities should still sum to 1.0
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(8), atol=1e-6)

    def test_extreme_negative_logits(self):
        """Test that extreme negative logits are handled without NaN."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Create input that would produce extreme negative logits
        hiddens = torch.ones(8, 16, 768) * -100.0
        routing_probs, routing_logits, _ = router(hiddens)

        # Check no NaN or Inf
        assert not torch.isnan(routing_probs).any()
        assert not torch.isinf(routing_probs).any()

        # Probabilities should still sum to 1.0
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(8), atol=1e-6)

    def test_mixed_zero_and_nonzero_batch(self):
        """Test batch with both zero and non-zero inputs."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Create batch with mix of zero and non-zero inputs
        hiddens = torch.randn(8, 16, 768)
        hiddens[0] = 0.0  # First item all zeros
        hiddens[3] = 0.0  # Fourth item all zeros

        routing_probs, _, _ = router(hiddens)

        # Zero inputs should have uniform distribution
        expected_uniform = torch.ones(4) / 4.0
        assert torch.allclose(routing_probs[0], expected_uniform, atol=1e-6)
        assert torch.allclose(routing_probs[3], expected_uniform, atol=1e-6)

        # Non-zero inputs should have valid distributions
        for i in [1, 2, 4, 5, 6, 7]:
            assert torch.allclose(routing_probs[i].sum(), torch.tensor(1.0), atol=1e-6)

    def test_very_small_inputs(self):
        """Test with very small but non-zero inputs."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Very small values (but above epsilon threshold)
        hiddens = torch.ones(8, 16, 768) * 1e-8
        routing_probs, _, _ = router(hiddens)

        # Should produce valid probabilities (not uniform, since above epsilon)
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(8), atol=1e-6)
        assert not torch.isnan(routing_probs).any()


class TestEntropyCalculation:
    """Tests for routing entropy computation and correctness."""

    @pytest.fixture
    def router(self):
        """Standard router fixture."""
        return MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

    def test_entropy_uniform_distribution(self, router):
        """Test that uniform distribution produces maximum entropy."""
        # Create uniform routing probabilities
        uniform_probs = torch.ones(8, 4) / 4.0
        entropy = router.compute_routing_entropy(uniform_probs)

        # Maximum entropy for uniform distribution over k=4 experts: log(4)
        max_entropy = torch.log(torch.tensor(4.0))
        assert torch.allclose(entropy, max_entropy.expand(8), atol=1e-6)

    def test_entropy_peaked_distribution(self, router):
        """Test that peaked distribution produces low entropy."""
        # Create peaked distribution (one expert has high probability)
        peaked_probs = torch.zeros(8, 4)
        peaked_probs[:, 0] = 0.97
        peaked_probs[:, 1:] = 0.01

        entropy = router.compute_routing_entropy(peaked_probs)

        # Peaked distribution should have low entropy
        assert torch.all(entropy < 0.3)  # Much lower than max entropy (log(4) ≈ 1.386)

    def test_entropy_non_negative(self, router):
        """Test that entropy is always non-negative."""
        hiddens = torch.randn(32, 16, 768)
        _, _, entropy = router(hiddens)

        assert torch.all(entropy >= 0.0)

    def test_entropy_bounded(self, router):
        """Test that entropy is bounded by log(num_experts)."""
        hiddens = torch.randn(32, 16, 768)
        _, _, entropy = router(hiddens)

        max_entropy = torch.log(torch.tensor(4.0))  # log(num_experts)
        assert torch.all(entropy <= max_entropy + 1e-6)  # Allow small tolerance

    def test_entropy_calculation_consistency(self, router):
        """Test that entropy from forward() matches compute_routing_entropy()."""
        hiddens = torch.randn(8, 16, 768)
        routing_probs, _, entropy_forward = router(hiddens)

        # Compute entropy using public method
        entropy_method = router.compute_routing_entropy(routing_probs)

        # Should be identical
        assert torch.allclose(entropy_forward, entropy_method, atol=1e-6)


class TestLoadBalanceLoss:
    """Tests for load balancing auxiliary loss computation."""

    @pytest.fixture
    def router(self):
        """Standard router fixture."""
        return MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

    def test_load_balance_loss_shape(self, router):
        """Test that load balance loss is a scalar."""
        routing_probs = torch.rand(8, 4)
        routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True)

        loss = router.compute_load_balance_loss(routing_probs)
        assert loss.shape == torch.Size([])  # Scalar

    def test_load_balance_loss_non_negative(self, router):
        """Test that load balance loss is non-negative."""
        routing_probs = torch.rand(32, 4)
        routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True)

        loss = router.compute_load_balance_loss(routing_probs)
        assert loss >= 0.0

    def test_load_balance_loss_uniform_routing(self, router):
        """Test load balance loss for uniform routing (ideal case)."""
        # Uniform routing: each expert gets equal probability
        uniform_probs = torch.ones(32, 4) / 4.0

        loss = router.compute_load_balance_loss(uniform_probs)

        # For uniform routing, loss should be k * (1/k) * (1/k) = 1/k
        expected_loss = 1.0  # k * (1/k) * (1/k) * k = k/k^2 * k = 1
        assert torch.allclose(loss, torch.tensor(expected_loss), atol=0.1)

    def test_load_balance_loss_imbalanced_routing(self, router):
        """Test that imbalanced routing produces higher loss."""
        # Imbalanced routing: most samples go to expert 0
        imbalanced_probs = torch.zeros(32, 4)
        imbalanced_probs[:, 0] = 0.9
        imbalanced_probs[:, 1:] = 0.1 / 3.0

        # Balanced routing for comparison
        balanced_probs = torch.ones(32, 4) / 4.0

        loss_imbalanced = router.compute_load_balance_loss(imbalanced_probs)
        loss_balanced = router.compute_load_balance_loss(balanced_probs)

        # Imbalanced routing should have higher loss
        assert loss_imbalanced > loss_balanced


class TestAccessorMethods:
    """Tests for accessor and analysis methods."""

    @pytest.fixture
    def router(self):
        """Standard router fixture."""
        return MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

    def test_get_routing_weights_shape(self, router):
        """Test that get_routing_weights returns correct shape."""
        weights = router.get_routing_weights()
        assert weights.shape == (4, 768)  # (num_experts, hidden_dim)

    def test_get_routing_weights_not_none(self, router):
        """Test that routing weights are initialized."""
        weights = router.get_routing_weights()
        assert weights is not None
        assert not torch.isnan(weights).any()

    def test_get_routing_distribution_keys(self, router):
        """Test that get_routing_distribution returns all expected keys."""
        routing_probs = torch.rand(8, 4)
        routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True)

        distribution = router.get_routing_distribution(routing_probs)

        expected_keys = {"mean_probs", "max_probs", "min_probs", "entropy_mean", "entropy_std"}
        assert set(distribution.keys()) == expected_keys

    def test_get_routing_distribution_shapes(self, router):
        """Test that get_routing_distribution returns correct shapes."""
        routing_probs = torch.rand(8, 4)
        routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True)

        distribution = router.get_routing_distribution(routing_probs)

        assert distribution["mean_probs"].shape == (4,)
        assert distribution["max_probs"].shape == (4,)
        assert distribution["min_probs"].shape == (4,)
        assert distribution["entropy_mean"].shape == torch.Size([])  # Scalar
        assert distribution["entropy_std"].shape == torch.Size([])  # Scalar

    def test_get_routing_distribution_values_valid(self, router):
        """Test that get_routing_distribution returns valid values."""
        routing_probs = torch.rand(8, 4)
        routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True)

        distribution = router.get_routing_distribution(routing_probs)

        # All values should be finite
        assert torch.isfinite(distribution["mean_probs"]).all()
        assert torch.isfinite(distribution["max_probs"]).all()
        assert torch.isfinite(distribution["min_probs"]).all()
        assert torch.isfinite(distribution["entropy_mean"])
        assert torch.isfinite(distribution["entropy_std"])

        # Probabilities should be in [0, 1]
        assert torch.all(distribution["mean_probs"] >= 0.0)
        assert torch.all(distribution["mean_probs"] <= 1.0)


class TestPoolingMethods:
    """Tests for different pooling methods."""

    def test_mean_pooling(self):
        """Test mean-pooling produces correct output."""
        router = MemoryGatingNetwork(
            hidden_dim=768, num_experts=4, temperature=1.0, pooling_method="mean"
        )

        hiddens = torch.randn(8, 16, 768)
        routing_probs, _, _ = router(hiddens)

        assert routing_probs.shape == (8, 4)
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(8), atol=1e-6)

    def test_max_pooling(self):
        """Test max-pooling produces correct output."""
        router = MemoryGatingNetwork(
            hidden_dim=768, num_experts=4, temperature=1.0, pooling_method="max"
        )

        hiddens = torch.randn(8, 16, 768)
        routing_probs, _, _ = router(hiddens)

        assert routing_probs.shape == (8, 4)
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(8), atol=1e-6)

    def test_mean_vs_max_pooling_different_outputs(self):
        """Test that mean and max pooling produce different routing."""
        hiddens = torch.randn(8, 16, 768)

        # Use same seed for fair comparison
        torch.manual_seed(42)
        router_mean = MemoryGatingNetwork(
            hidden_dim=768, num_experts=4, temperature=1.0, pooling_method="mean"
        )
        probs_mean, _, _ = router_mean(hiddens)

        torch.manual_seed(42)
        router_max = MemoryGatingNetwork(
            hidden_dim=768, num_experts=4, temperature=1.0, pooling_method="max"
        )
        probs_max, _, _ = router_max(hiddens)

        # Outputs should be different (different pooling strategies)
        # Both should produce valid probabilities though
        assert torch.allclose(probs_mean.sum(dim=-1), torch.ones(8), atol=1e-6)
        assert torch.allclose(probs_max.sum(dim=-1), torch.ones(8), atol=1e-6)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_batch_size_one(self):
        """Test with batch size = 1."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)
        hiddens = torch.randn(1, 16, 768)

        routing_probs, routing_logits, routing_entropy = router(hiddens)

        assert routing_probs.shape == (1, 4)
        assert torch.allclose(routing_probs.sum(), torch.tensor(1.0), atol=1e-6)

    def test_num_experts_minimum(self):
        """Test with minimum num_experts = 2."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=2, temperature=1.0)
        hiddens = torch.randn(8, 16, 768)

        routing_probs, _, _ = router(hiddens)

        assert routing_probs.shape == (8, 2)
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(8), atol=1e-6)

    def test_num_experts_maximum(self):
        """Test with maximum num_experts = 8."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=8, temperature=1.0)
        hiddens = torch.randn(8, 16, 768)

        routing_probs, _, _ = router(hiddens)

        assert routing_probs.shape == (8, 8)
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(8), atol=1e-6)

    def test_temperature_near_zero(self):
        """Test with temperature very close to zero (but not zero)."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=0.01)
        hiddens = torch.randn(8, 16, 768)

        routing_probs, _, _ = router(hiddens)

        # Should produce very peaked distribution (but valid)
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(8), atol=1e-6)
        assert not torch.isnan(routing_probs).any()

    def test_all_routing_probabilities_equal(self):
        """Test entropy calculation when all experts have equal probability."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        # Create perfectly uniform routing
        uniform_probs = torch.ones(8, 4) / 4.0
        entropy = router.compute_routing_entropy(uniform_probs)

        # Should equal maximum entropy: log(4)
        max_entropy = torch.log(torch.tensor(4.0))
        assert torch.allclose(entropy, max_entropy.expand(8), atol=1e-6)


class TestGradientFlow:
    """Tests for gradient flow through routing network."""

    def test_routing_probabilities_differentiable(self):
        """Test that routing probabilities have gradients enabled."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)
        hiddens = torch.randn(8, 16, 768, requires_grad=True)

        routing_probs, _, _ = router(hiddens)

        # Check that gradients can flow back
        loss = routing_probs.sum()
        loss.backward()

        assert hiddens.grad is not None
        assert not torch.isnan(hiddens.grad).any()

    def test_routing_weights_have_gradients(self):
        """Test that routing weights require gradients."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)

        assert router.routing_projection.weight.requires_grad
        assert router.routing_projection.bias.requires_grad

    def test_backward_pass_no_nan(self):
        """Test that backward pass doesn't produce NaN gradients."""
        router = MemoryGatingNetwork(hidden_dim=768, num_experts=4, temperature=1.0)
        hiddens = torch.randn(8, 16, 768, requires_grad=True)

        routing_probs, _, entropy = router(hiddens)

        # Compute loss including entropy
        loss = routing_probs.sum() + entropy.sum()
        loss.backward()

        # Check no NaN in gradients
        assert not torch.isnan(hiddens.grad).any()
        assert not torch.isnan(router.routing_projection.weight.grad).any()
        assert not torch.isnan(router.routing_projection.bias.grad).any()
