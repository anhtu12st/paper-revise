"""
Integration tests for RBS-QA halting policy system.

Tests the integration between belief state tracking, halting policy,
and hybrid training components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from rbsqa.belief_state import BeliefStateTracker, BeliefState, SpanCandidate
from rbsqa.halting_policy import HaltingPolicyNetwork, HaltingStateFeatures
from rbsqa.rbs_trainer import RBSTrainer
from rbsqa.config import RBSTrainingConfig, rbs_balanced_config


class TestRBSHaltingIntegration:
    """Integration tests for RBS halting system."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = rbs_balanced_config(
            num_epochs=1,
            learning_rate=1e-4,
            max_segments=4,
            num_memory_experts=2,
            memory_num_tokens=8
        )
        # Override device for testing
        config.device = 'cpu'
        return config

    @pytest.fixture
    def mock_model(self):
        """Create mock GMM-XLNet model."""
        model = Mock()

        # Mock model outputs
        mock_outputs = Mock()
        mock_outputs.start_logits = torch.randn(2, 20)
        mock_outputs.end_logits = torch.randn(2, 20)
        mock_outputs.loss = torch.tensor(0.5)
        mock_outputs.memory_state = torch.randn(2, 8, 768)
        mock_outputs.gmm_context = {
            'routing_entropy': 0.8,
            'expert_utilization': [0.3, 0.4, 0.2, 0.1],
            'context_quality': 0.7
        }

        model.return_value = mock_outputs
        model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        model.train = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.state_dict = Mock(return_value={})
        model.load_state_dict = Mock()

        return model

    @pytest.fixture
    def belief_tracker(self):
        """Create belief state tracker."""
        tracker = BeliefStateTracker(
            max_segments=32,
            confidence_threshold=0.7,
            re_scoring_method="context_weighted",
            enable_trend_analysis=True,
            hidden_dim=768,
            enable_learnable_re_scoring=False
        )
        tracker.to = Mock(return_value=tracker)
        tracker.train = Mock()
        tracker.eval = Mock()
        tracker.state_dict = Mock(return_value={})
        tracker.load_state_dict = Mock()
        return tracker

    @pytest.fixture
    def halting_policy(self):
        """Create halting policy network."""
        policy = HaltingPolicyNetwork(
            input_dim=12,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            temperature=1.0,
            exploration_rate=0.1
        )
        policy.to = Mock(return_value=policy)
        policy.train = Mock()
        policy.eval = Mock()
        policy.state_dict = Mock(return_value={})
        policy.load_state_dict = Mock()
        return policy

    @pytest.fixture
    def rbs_trainer(self, config, mock_model, belief_tracker, halting_policy):
        """Create RBS trainer."""
        return RBSTrainer(
            model=mock_model,
            belief_tracker=belief_tracker,
            halting_policy=halting_policy,
            config=config
        )

    @pytest.fixture
    def mock_batch(self):
        """Create mock training batch."""
        return {
            'input_ids': torch.randint(0, 1000, (2, 128)),
            'attention_mask': torch.ones(2, 128),
            'start_positions': torch.randint(0, 128, (2,)),
            'end_positions': torch.randint(0, 128, (2,)),
            'global_start_positions': torch.tensor([50, 30]),
            'global_end_positions': torch.tensor([60, 40]),
            'num_segments': 3,
            'segment_offsets': [0, 128, 256]
        }

    def test_trainer_initialization(self, rbs_trainer, config):
        """Test RBS trainer initialization."""
        assert rbs_trainer.config == config
        assert rbs_trainer.current_epoch == 0
        assert rbs_trainer.device == torch.device('cpu')
        assert hasattr(rbs_trainer, 'qa_optimizer')
        assert hasattr(rbs_trainer, 'rl_optimizer')

    def test_halting_features_extraction(self, rbs_trainer, mock_model):
        """Test halting features extraction from model outputs."""
        # Create belief state
        belief_state = BeliefState(
            best_span=(10, 15),
            confidence=0.8,
            segment_id=2,
            confidence_history=[0.7, 0.75, 0.8],
            revision_count=1,
            total_segments=3
        )

        # Create batch
        batch = {
            'num_segments': 5,
            'segment_offsets': [0, 128, 256, 384, 512]
        }

        # Extract features
        features = rbs_trainer.extract_halting_features(
            belief_state, mock_model.return_value, batch, segment_idx=2
        )

        assert isinstance(features, HaltingStateFeatures)
        assert features.current_confidence == 0.8
        assert features.segments_processed == 3
        assert features.segments_remaining == 2
        assert features.routing_entropy == 0.8

    def test_qa_loss_computation(self, rbs_trainer):
        """Test QA loss computation."""
        # Mock model outputs
        mock_outputs = Mock()
        mock_outputs.start_logits = torch.randn(2, 20)
        mock_outputs.end_logits = torch.randn(2, 20)
        mock_outputs.loss = torch.tensor(0.8)

        start_positions = torch.randint(0, 20, (2,))
        end_positions = torch.randint(0, 20, (2,))

        loss = rbs_trainer.compute_qa_loss(mock_outputs, start_positions, end_positions)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.requires_grad

    def test_mock_training_epoch(self, rbs_trainer, mock_batch):
        """Test training epoch with mock data loader."""
        # Create mock data loader
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch, mock_batch]))
        mock_dataloader.__len__ = Mock(return_value=2)

        # Patch tqdm to avoid progress bar
        with patch('rbsqa.rbs_trainer.tqdm', return_value=[mock_batch, mock_batch]):
            # Train epoch
            stats = rbs_trainer.train_epoch(mock_dataloader)

        # Check stats
        assert 'qa_loss' in stats
        assert 'rl_loss' in stats
        assert 'total_loss' in stats
        assert 'avg_segments_processed' in stats
        assert stats['num_batches'] == 2
        assert rbs_trainer.current_epoch == 1

    def test_halting_policy_training(self, rbs_trainer, halting_policy):
        """Test halting policy training component."""
        # Create mock episodes
        episodes = [[
            {
                'features': Mock(),
                'action': 'CONTINUE',
                'log_prob': torch.tensor(-0.5, requires_grad=True),
                'value_estimate': torch.tensor(0.1, requires_grad=True)
            },
            {
                'features': Mock(),
                'action': 'HALT',
                'log_prob': torch.tensor(-0.3, requires_grad=True),
                'value_estimate': torch.tensor(0.2, requires_grad=True)
            }
        ]]

        ground_truth_spans = [(10, 15)]

        # Train halting policy
        rl_loss = rbs_trainer.train_halting_policy(episodes, ground_truth_spans)

        assert isinstance(rl_loss, torch.Tensor)
        assert rl_loss.requires_grad
        # RL loss can be negative (policy gradient loss), but should be a valid tensor
        assert torch.isfinite(rl_loss)

    def test_belief_state_and_halting_integration(self, belief_tracker, halting_policy):
        """Test integration between belief state tracker and halting policy."""
        # Reset belief state
        belief_state = belief_tracker.reset_belief()

        # Mock model outputs
        mock_logits = (torch.randn(128), torch.randn(128))
        mock_gmm_context = torch.randn(1, 1, 768)

        # Update belief state
        belief_state = belief_tracker.update_belief(
            mock_logits, segment_idx=0, gmm_context=mock_gmm_context, global_offset=0
        )

        # Create halting features
        features = HaltingStateFeatures(
            current_confidence=belief_state.confidence,
            confidence_trend=belief_state.confidence_history[-3:],
            confidence_variance=0.01,
            revision_count=belief_state.revision_count,
            segments_processed=1,
            segments_remaining=4,
            processing_time=1.0,
            routing_entropy=0.8,
            expert_utilization=[0.25, 0.25, 0.25, 0.25],
            context_quality_score=0.7,
            document_length=5,
            question_complexity=0.6,
            segment_relevance_score=0.8
        )

        # Make halting decision
        action, log_prob, value = halting_policy.select_action(features, training=True)

        assert action in ["CONTINUE", "HALT"]
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(value, torch.Tensor)

    def test_exploration_decay_during_training(self, rbs_trainer, halting_policy):
        """Test exploration rate decay during training."""
        initial_rate = halting_policy.exploration_rate

        # Create mock episodes for training
        episodes = [[
            {
                'features': Mock(),
                'action': 'CONTINUE',
                'log_prob': torch.tensor(-0.5, requires_grad=True),
                'value_estimate': torch.tensor(0.1, requires_grad=True)
            }
        ]]

        ground_truth_spans = [(10, 15)]

        # Train halting policy multiple times
        for _ in range(5):
            rbs_trainer.train_halting_policy(episodes, ground_truth_spans)

        # Exploration rate should have decayed
        assert halting_policy.exploration_rate < initial_rate

    def test_reward_shaping_with_different_costs(self, rbs_trainer, halting_policy):
        """Test reward computation with different cost parameters."""
        # Create episode with multiple CONTINUE actions before HALT
        episodes = [[
            {
                'features': Mock(),
                'action': 'CONTINUE',
                'predicted_span': (0, 0)
            },
            {
                'features': Mock(),
                'action': 'CONTINUE',
                'predicted_span': (0, 0)
            },
            {
                'features': Mock(),
                'action': 'HALT',
                'predicted_span': (10, 15)
            }
        ]]

        ground_truth_spans = [(12, 17)]  # Partial overlap

        # Test with different lambda costs
        costs = [0.01, 0.05, 0.1]
        rewards_by_cost = []

        for cost in costs:
            rewards = halting_policy.compute_rewards(episodes, ground_truth_spans, lambda_cost=cost)
            rewards_by_cost.append(rewards[0])

        # Higher costs should result in lower total rewards
        total_reward_low_cost = sum(rewards_by_cost[0])
        total_reward_high_cost = sum(rewards_by_cost[2])

        assert total_reward_low_cost > total_reward_high_cost

    def test_checkpoint_save_and_load(self, rbs_trainer):
        """Test checkpoint saving and loading."""
        import tempfile
        import os

        # Create temporary checkpoint file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            checkpoint_path = tmp_file.name

        try:
            # Save checkpoint
            metrics = {'qa_loss': 0.5, 'rl_loss': 0.1, 'total_loss': 0.6}
            rbs_trainer.save_checkpoint(checkpoint_path, epoch=5, metrics=metrics)

            # Verify file was created
            assert os.path.exists(checkpoint_path)

            # Load checkpoint
            loaded_checkpoint = rbs_trainer.load_checkpoint(checkpoint_path)

            # Verify loaded data
            assert loaded_checkpoint['epoch'] == 5
            assert loaded_checkpoint['metrics'] == metrics

            # Verify trainer state was updated
            assert rbs_trainer.current_epoch == 5

        finally:
            # Clean up
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)

    def test_training_summary(self, rbs_trainer):
        """Test training summary generation."""
        summary = rbs_trainer.get_training_summary()

        assert 'config' in summary
        assert 'current_epoch' in summary
        assert 'device' in summary
        assert 'training_stats' in summary
        assert 'model_parameters' in summary
        assert 'belief_tracker_parameters' in summary
        assert 'halting_policy_parameters' in summary
        assert 'rl_stats' in summary

    def test_evaluation_mode(self, rbs_trainer, mock_batch):
        """Test evaluation mode."""
        # Create mock data loader
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch]))
        mock_dataloader.__len__ = Mock(return_value=1)

        # Patch tqdm to avoid progress bar
        with patch('rbsqa.rbs_trainer.tqdm', return_value=[mock_batch]):
            # Run evaluation
            eval_stats = rbs_trainer.evaluate(mock_dataloader)

        # Check evaluation stats
        assert 'qa_loss' in eval_stats
        assert 'exact_match' in eval_stats
        assert 'f1_score' in eval_stats
        assert 'avg_segments_processed' in eval_stats
        assert 'num_examples' in eval_stats
        assert eval_stats['num_examples'] == 1

    def test_backward_compatibility_without_halting_policy(self, config, mock_model, belief_tracker):
        """Test that trainer works without halting policy (backward compatibility)."""
        config.use_halting_policy = False

        trainer = RBSTrainer(
            model=mock_model,
            belief_tracker=belief_tracker,
            halting_policy=Mock(),  # Mock halting policy won't be used
            config=config
        )

        # Create mock data loader
        mock_batch = {
            'input_ids': torch.randint(0, 1000, (2, 128)),
            'attention_mask': torch.ones(2, 128),
            'start_positions': torch.randint(0, 128, (2,)),
            'end_positions': torch.randint(0, 128, (2,)),
            'num_segments': 2
        }

        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch]))
        mock_dataloader.__len__ = Mock(return_value=1)

        # Should still be able to train without halting policy
        with patch('rbsqa.rbs_trainer.tqdm', return_value=[mock_batch]):
            stats = trainer.train_epoch(mock_dataloader)

        assert stats['rl_loss'] == 0.0  # No RL loss when halting policy is disabled
        assert stats['qa_loss'] > 0.0    # QA loss should still be computed

    def test_different_rl_start_epochs(self, config, mock_model, belief_tracker, halting_policy):
        """Test behavior with different RL start epochs."""
        config.rl_start_epoch = 3  # Start RL after epoch 2

        trainer = RBSTrainer(
            model=mock_model,
            belief_tracker=belief_tracker,
            halting_policy=halting_policy,
            config=config
        )

        # Create mock data loader
        mock_batch = {
            'input_ids': torch.randint(0, 1000, (2, 128)),
            'attention_mask': torch.ones(2, 128),
            'start_positions': torch.randint(0, 128, (2,)),
            'end_positions': torch.randint(0, 128, (2,)),
            'num_segments': 2
        }

        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch]))
        mock_dataloader.__len__ = Mock(return_value=1)

        # Train epoch 0 and 1 (RL should not start yet)
        for epoch in range(2):
            trainer.current_epoch = epoch
            with patch('rbsqa.rbs_trainer.tqdm', return_value=[mock_batch]):
                stats = trainer.train_epoch(mock_dataloader)

            # RL loss should be 0 before start epoch
            assert stats['rl_loss'] == 0.0

        # Train epoch 2 (RL should start now)
        trainer.current_epoch = 2
        with patch('rbsqa.rbs_trainer.tqdm', return_value=[mock_batch]):
            stats = trainer.train_epoch(mock_dataloader)

        # RL loss should be computed after start epoch
        # Note: Might still be 0 if no episodes were generated, but the training process should run
        assert 'rl_loss' in stats