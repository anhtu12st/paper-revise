"""
Unit tests for Halting Policy Network.

Tests the REINFORCE-based halting policy implementation including
forward pass, action selection, reward computation, and loss calculation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from rbsqa.halting_policy import HaltingStateFeatures, HaltingPolicyNetwork


class TestHaltingStateFeatures:
    """Test HaltingStateFeatures dataclass."""

    def test_halt_state_features_creation(self):
        """Test creating valid HaltingStateFeatures."""
        features = HaltingStateFeatures(
            current_confidence=0.8,
            confidence_trend=[0.7, 0.75, 0.8],
            confidence_variance=0.01,
            revision_count=2,
            segments_processed=5,
            segments_remaining=10,
            processing_time=2.5,
            routing_entropy=1.2,
            expert_utilization=[0.3, 0.4, 0.2, 0.1],
            context_quality_score=0.7,
            document_length=15,
            question_complexity=0.6,
            segment_relevance_score=0.8
        )

        assert features.current_confidence == 0.8
        assert len(features.confidence_trend) == 3
        assert features.revision_count == 2
        assert features.segments_processed == 5

    def test_halt_state_features_empty_trend(self):
        """Test HaltingStateFeatures with empty confidence trend."""
        features = HaltingStateFeatures(
            current_confidence=0.5,
            confidence_trend=[],
            confidence_variance=0.0,
            revision_count=0,
            segments_processed=1,
            segments_remaining=9,
            processing_time=0.5,
            routing_entropy=0.8,
            expert_utilization=[],
            context_quality_score=0.5,
            document_length=10,
            question_complexity=0.4,
            segment_relevance_score=0.6
        )

        tensor = features.to_tensor(torch.device('cpu'))
        assert tensor.shape == (12,)
        assert tensor[1] == 0.0  # Empty trend should default to 0.0

    def test_halt_state_features_validation(self):
        """Test HaltingStateFeatures validation."""
        # Test invalid confidence
        with pytest.raises(ValueError, match="current_confidence must be in \\[0.0, 1.0\\]"):
            features = HaltingStateFeatures(
                current_confidence=1.5,
                confidence_trend=[],
                confidence_variance=0.0,
                revision_count=0,
                segments_processed=1,
                segments_remaining=9,
                processing_time=0.5,
                routing_entropy=0.8,
                expert_utilization=[],
                context_quality_score=0.5,
                document_length=10,
                question_complexity=0.4,
                segment_relevance_score=0.6
            )
            features.validate()

        # Test negative segments processed
        with pytest.raises(ValueError, match="segments_processed must be >= 0"):
            features = HaltingStateFeatures(
                current_confidence=0.5,
                confidence_trend=[],
                confidence_variance=0.0,
                revision_count=0,
                segments_processed=-1,
                segments_remaining=9,
                processing_time=0.5,
                routing_entropy=0.8,
                expert_utilization=[],
                context_quality_score=0.5,
                document_length=10,
                question_complexity=0.4,
                segment_relevance_score=0.6
            )
            features.validate()

    def test_to_tensor_conversion(self):
        """Test converting features to tensor."""
        features = HaltingStateFeatures(
            current_confidence=0.8,
            confidence_trend=[0.7, 0.75, 0.8],
            confidence_variance=0.01,
            revision_count=2,
            segments_processed=5,
            segments_remaining=10,
            processing_time=2.5,
            routing_entropy=1.2,
            expert_utilization=[0.3, 0.4, 0.2, 0.1],
            context_quality_score=0.7,
            document_length=15,
            question_complexity=0.6,
            segment_relevance_score=0.8
        )

        device = torch.device('cpu')
        tensor = features.to_tensor(device)

        assert tensor.shape == (12,)
        assert tensor.dtype == torch.float32
        assert tensor.device == device

        # Check some key values
        assert tensor[0] == 0.8  # current_confidence
        assert tensor[1] == 0.75  # mean confidence trend
        assert tensor[2] == 0.01  # confidence variance
        assert tensor[3] == 2.0  # revision count


class TestHaltingPolicyNetwork:
    """Test HaltingPolicyNetwork implementation."""

    @pytest.fixture
    def policy_network(self):
        """Create a halting policy network for testing."""
        return HaltingPolicyNetwork(
            input_dim=12,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            temperature=1.0,
            exploration_rate=0.1
        )

    @pytest.fixture
    def sample_features(self):
        """Create sample halting state features for testing."""
        return HaltingStateFeatures(
            current_confidence=0.8,
            confidence_trend=[0.7, 0.75, 0.8],
            confidence_variance=0.01,
            revision_count=2,
            segments_processed=5,
            segments_remaining=10,
            processing_time=2.5,
            routing_entropy=1.2,
            expert_utilization=[0.3, 0.4, 0.2, 0.1],
            context_quality_score=0.7,
            document_length=15,
            question_complexity=0.6,
            segment_relevance_score=0.8
        )

    def test_network_initialization(self):
        """Test network initialization with valid parameters."""
        network = HaltingPolicyNetwork(
            input_dim=12,
            hidden_dim=64,
            num_layers=3,
            dropout=0.2,
            temperature=1.5,
            exploration_rate=0.15
        )

        assert network.input_dim == 12
        assert network.hidden_dim == 64
        assert network.num_layers == 3
        assert network.temperature == 1.5
        assert network.exploration_rate == 0.15
        assert len(network.current_episode) == 0
        assert len(network.training_episodes) == 0

    def test_network_initialization_invalid_params(self):
        """Test network initialization with invalid parameters."""
        # Invalid input_dim
        with pytest.raises(ValueError, match="input_dim must be positive"):
            HaltingPolicyNetwork(input_dim=0)

        # Invalid temperature
        with pytest.raises(ValueError, match="temperature must be positive"):
            HaltingPolicyNetwork(temperature=-1.0)

        # Invalid exploration_rate
        with pytest.raises(ValueError, match="exploration_rate must be in \\[0.0, 1.0\\]"):
            HaltingPolicyNetwork(exploration_rate=1.5)

    def test_forward_pass(self, policy_network, sample_features):
        """Test forward pass through the network."""
        policy_logits, value_estimate = policy_network.forward(sample_features)

        assert policy_logits.shape == (2,)  # CONTINUE, HALT
        assert value_estimate.shape == ()  # Scalar
        assert policy_logits.dtype == torch.float32
        assert value_estimate.dtype == torch.float32

    def test_select_action_training(self, policy_network, sample_features):
        """Test action selection during training."""
        # Test without exploration
        policy_network.exploration_rate = 0.0
        action, log_prob, value = policy_network.select_action(sample_features, training=True)

        assert action in ["CONTINUE", "HALT"]
        assert isinstance(log_prob, torch.Tensor)
        assert log_prob.shape == ()
        assert isinstance(value, torch.Tensor)
        assert value.shape == ()
        assert len(policy_network.current_episode) == 1

    def test_select_action_exploration(self, policy_network, sample_features):
        """Test action selection with exploration."""
        # Force exploration by mocking random choice
        with patch('torch.rand') as mock_rand:
            mock_rand.return_value = torch.tensor([0.05])  # Force exploration

            action, log_prob, value = policy_network.select_action(sample_features, training=True)

            assert action in ["CONTINUE", "HALT"]
            step = policy_network.current_episode[-1]
            assert step['exploration'] is True

    def test_select_action_eval_mode(self, policy_network, sample_features):
        """Test action selection during evaluation."""
        action, log_prob, value = policy_network.select_action(sample_features, training=False)

        assert action in ["CONTINUE", "HALT"]
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        # Should not store episode during evaluation
        assert len(policy_network.current_episode) == 0

    def test_f1_score_computation(self, policy_network):
        """Test F1 score computation for reward calculation."""
        # Perfect match
        f1 = policy_network._compute_f1_score((10, 15), (10, 15))
        assert f1 == 1.0

        # Partial overlap
        f1 = policy_network._compute_f1_score((10, 15), (12, 18))
        assert 0.0 < f1 < 1.0

        # No overlap
        f1 = policy_network._compute_f1_score((10, 15), (20, 25))
        assert f1 == 0.0

        # Empty spans
        f1 = policy_network._compute_f1_score((0, -1), (0, -1))
        assert f1 == 1.0

    def test_compute_rewards(self, policy_network, sample_features):
        """Test reward computation for episodes."""
        # Create a simple episode
        episode = [
            {
                'features': sample_features,
                'action': 'CONTINUE',
                'predicted_span': (0, 0)
            },
            {
                'features': sample_features,
                'action': 'HALT',
                'predicted_span': (10, 15)
            }
        ]

        ground_truth_spans = [(12, 18)]  # Partial overlap with prediction

        rewards = policy_network.compute_rewards([episode], ground_truth_spans, lambda_cost=0.01)

        assert len(rewards) == 1
        assert len(rewards[0]) == 2  # Two actions in episode

        # CONTINUE action should have negative reward (cost)
        assert rewards[0][0] == -0.01

        # HALT action should have F1 score minus total cost
        expected_f1 = policy_network._compute_f1_score((10, 15), (12, 18))
        expected_reward = expected_f1 - (0.01 * 2)  # 2 segments processed
        assert abs(rewards[0][1] - expected_reward) < 1e-6

    def test_policy_gradient_loss(self, policy_network, sample_features):
        """Test policy gradient loss computation."""
        # Create simple episode with log_probs and values
        episode = [
            {
                'log_prob': torch.tensor([-0.5]),
                'value_estimate': torch.tensor([0.1])
            },
            {
                'log_prob': torch.tensor([-0.3]),
                'value_estimate': torch.tensor([0.2])
            }
        ]

        rewards = [[-0.01, 0.8]]  # CONTINUE cost, HALT reward

        loss = policy_network.policy_gradient_loss(
            [episode], rewards, gamma=0.99, use_baseline=True
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.requires_grad

    def test_value_loss(self, policy_network, sample_features):
        """Test value function loss computation."""
        episode = [
            {'value_estimate': torch.tensor([0.1])},
            {'value_estimate': torch.tensor([0.2])}
        ]

        rewards = [[-0.01, 0.8]]

        loss = policy_network.value_loss([episode], rewards, gamma=0.99)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.requires_grad

    def test_episode_management(self, policy_network, sample_features):
        """Test episode start/end management."""
        # Start new episode
        assert len(policy_network.current_episode) == 0

        # Add steps to episode
        policy_network.select_action(sample_features, training=True)
        policy_network.select_action(sample_features, training=True)

        assert len(policy_network.current_episode) == 2

        # End episode
        episode_data = policy_network.end_episode()

        assert episode_data['length'] == 2
        assert len(episode_data['actions']) == 2
        assert len(policy_network.current_episode) == 0  # Should be reset
        assert len(policy_network.training_episodes) == 1

    def test_training_state_reset(self, policy_network):
        """Test training state reset."""
        # Add some training data
        policy_network.training_episodes.append([{'action': 'CONTINUE'}])
        policy_network.current_episode.append({'action': 'HALT'})

        # Reset state
        policy_network.reset_training_state()

        assert len(policy_network.training_episodes) == 0
        assert len(policy_network.current_episode) == 0

    def test_training_statistics(self, policy_network, sample_features):
        """Test training statistics computation."""
        # Empty state
        stats = policy_network.get_training_stats()
        assert stats['total_episodes'] == 0
        assert stats['avg_episode_length'] == 0.0

        # Add some episodes
        policy_network.select_action(sample_features, training=True)
        policy_network.end_episode()

        policy_network.select_action(sample_features, training=True)
        policy_network.select_action(sample_features, training=True)
        policy_network.end_episode()

        stats = policy_network.get_training_stats()
        assert stats['total_episodes'] == 2
        assert stats['total_steps'] == 3
        assert stats['avg_episode_length'] == 1.5

    def test_exploration_rate_update(self, policy_network):
        """Test exploration rate updates."""
        original_rate = policy_network.exploration_rate

        # Update to valid rate
        policy_network.set_exploration_rate(0.2)
        assert policy_network.exploration_rate == 0.2

        # Try invalid rate
        with pytest.raises(ValueError, match="exploration_rate must be in \\[0.0, 1.0\\]"):
            policy_network.set_exploration_rate(1.5)

    def test_temperature_update(self, policy_network):
        """Test temperature updates."""
        original_temp = policy_network.temperature

        # Update to valid temperature
        policy_network.set_temperature(1.5)
        assert policy_network.temperature == 1.5

        # Try invalid temperature
        with pytest.raises(ValueError, match="temperature must be positive"):
            policy_network.set_temperature(0.0)

    def test_gradient_flow(self, policy_network, sample_features):
        """Test that gradients flow properly through the network."""
        # Create episode with actions
        policy_network.select_action(sample_features, training=True)
        policy_network.end_episode()

        # Compute loss
        rewards = [[0.5]]  # Simple reward
        episodes = [policy_network.training_episodes[0]]

        loss = policy_network.policy_gradient_loss(episodes, rewards)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for param in policy_network.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break
        else:
            pytest.fail("No gradients found in network parameters")

    def test_device_consistency(self, policy_network, sample_features):
        """Test device consistency across tensors."""
        # Move network to CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            policy_network = policy_network.to(device)
        else:
            device = torch.device('cpu')

        # Forward pass
        policy_logits, value_estimate = policy_network.forward(sample_features)

        assert policy_logits.device == device
        assert value_estimate.device == device

        # Action selection
        action, log_prob, value = policy_network.select_action(sample_features, training=True)

        assert log_prob.device == device
        assert value.device == device


class TestIntegration:
    """Integration tests for halting policy components."""

    def test_end_to_end_training_step(self):
        """Test a complete training step."""
        # Setup
        policy = HaltingPolicyNetwork(
            input_dim=12,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            temperature=1.0,
            exploration_rate=0.1
        )

        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        # Create episode
        features = HaltingStateFeatures(
            current_confidence=0.7,
            confidence_trend=[0.6, 0.65, 0.7],
            confidence_variance=0.02,
            revision_count=1,
            segments_processed=3,
            segments_remaining=7,
            processing_time=1.5,
            routing_entropy=0.9,
            expert_utilization=[0.25, 0.35, 0.25, 0.15],
            context_quality_score=0.6,
            document_length=10,
            question_complexity=0.5,
            segment_relevance_score=0.7
        )

        # Get initial parameter values
        initial_params = [p.clone() for p in policy.parameters()]

        # Simulate training episode
        for _ in range(3):
            policy.select_action(features, training=True)

        episode_data = policy.end_episode()

        # Compute rewards and loss
        rewards = policy.compute_rewards(
            [policy.training_episodes[-1]],
            [(5, 10)],  # Ground truth span
            lambda_cost=0.01
        )

        policy_loss = policy.policy_gradient_loss(
            policy.training_episodes[-1:],
            rewards[-1:]
        )

        # Backward pass
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Check that at least some parameters changed
        params_changed = False
        for initial_param, current_param in zip(initial_params, policy.parameters()):
            if not torch.allclose(initial_param, current_param, atol=1e-6):
                params_changed = True
                break

        assert params_changed, "No parameters were updated after optimization step"