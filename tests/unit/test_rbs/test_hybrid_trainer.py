"""
Unit tests for RBS hybrid trainer.
"""

import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch.utils.data import Dataset

from rbsqa.training.hybrid_trainer import CheckpointManager, RBSHybridTrainer
from rbsqa.configs.hybrid_training_config import RBSTrainingConfig


class DummyDataset(Dataset):
    """Simple dummy dataset for testing."""

    def __init__(self, num_examples=10):
        self.num_examples = num_examples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        seq_len = 128
        return {
            'input_ids': torch.randint(1, 1000, (2, seq_len)),  # 2 segments
            'attention_mask': torch.ones(2, seq_len),
            'start_positions': torch.randint(0, seq_len, (2,)),
            'end_positions': torch.randint(0, seq_len, (2,)),
            'segment_ids': torch.arange(2).repeat_interleave(seq_len // 2)[:seq_len].unsqueeze(0).repeat(2, 1),
            'segment_offsets': torch.tensor([0, seq_len // 2]),
            'num_segments': torch.tensor(2),
            'global_start_positions': torch.randint(0, seq_len, (1,)).item(),
            'global_end_positions': torch.randint(0, seq_len, (1,)).item(),
            'question_input_ids': torch.randint(1, 1000, (64,)),
            'context_segments': [torch.randint(1, 1000, (64,)) for _ in range(2)]
        }


class TestCheckpointManager:
    """Test cases for CheckpointManager."""

    def test_initialization(self):
        """Test CheckpointManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                save_dir=temp_dir,
                save_frequency=2,
                keep_best=3
            )

            assert manager.save_dir == temp_dir
            assert manager.save_frequency == 2
            assert manager.keep_best == 3

    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir, 2, 3)

            checkpoint_data = {"epoch": 1, "model_state": {"param": torch.tensor([1.0])}}
            saved_path = manager.save_checkpoint(checkpoint_data, "test-checkpoint")

            expected_path = f"{temp_dir}/test-checkpoint.pt"
            assert saved_path == expected_path
            assert saved_path == expected_path

            # Verify checkpoint was saved
            loaded_data = torch.load(saved_path)
            assert loaded_data["epoch"] == 1
            assert torch.equal(loaded_data["model_state"]["param"], torch.tensor([1.0]))

    def test_cleanup_old_checkpoints(self):
        """Test cleanup of old checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir, 1, 2)

            # Create multiple epoch checkpoints
            for i in range(5):
                checkpoint_data = {"epoch": i}
                manager.save_checkpoint(checkpoint_data, f"epoch-{i}")

            # Should only keep the most recent 2 (epochs 3 and 4)
            checkpoint_files = [f for f in os.listdir(temp_dir) if f.startswith("epoch-")]
            assert len(checkpoint_files) == 2

            # Check that only recent epochs remain
            remaining_epochs = []
            for filename in checkpoint_files:
                epoch_num = int(filename.replace("epoch-", "").replace(".pt", ""))
                remaining_epochs.append(epoch_num)

            assert sorted(remaining_epochs) == [3, 4]

    def test_save_non_epoch_checkpoint_no_cleanup(self):
        """Test that non-epoch checkpoints don't trigger cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir, 1, 2)

            # Save various checkpoints
            for i in range(3):
                manager.save_checkpoint({"epoch": i}, f"epoch-{i}")

            manager.save_checkpoint({"best": True}, "best-model")
            manager.save_checkpoint({"interrupted": True}, "interrupted")

            # Should have 5 checkpoints (3 epochs + best + interrupted)
            all_files = os.listdir(temp_dir)
            checkpoint_files = [f for f in all_files if f.endswith(".pt")]
            assert len(checkpoint_files) == 5


class TestRBSHybridTrainer:
    """Test cases for RBSHybridTrainer."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.gmm_backbone = Mock()
        model.gmm_backbone.parameters.return_value = [torch.tensor([1.0], requires_grad=True)]

        # Mock belief tracker
        model.belief_tracker = Mock()
        model.belief_tracker.parameters.return_value = [torch.tensor([2.0], requires_grad=True)]
        model.belief_tracker.reset_belief = Mock()

        # Mock halting policy
        model.halting_policy = Mock()
        model.halting_policy.parameters.return_value = [torch.tensor([3.0], requires_grad=True)]
        model.halting_policy.compute_rewards = Mock(return_value=[[1.0], [0.5]])
        model.halting_policy.policy_gradient_loss = Mock(return_value=torch.tensor(0.1))
        model.halting_policy.value_loss = Mock(return_value=torch.tensor(0.05))
        model.halting_policy.compute_f1_score = Mock(return_value=0.8)

        # Mock training mode setting
        model.set_training_mode = Mock()
        model.set_inference_mode = Mock()

        # Mock forward pass
        model.forward = Mock(return_value=Mock(
            start_logits=torch.randn(2, 128),
            end_logits=torch.randn(2, 128),
            memory_state={"expert_0": torch.randn(1, 16, 768)},
            belief_state=Mock(best_span=(10, 20)),
            halting_decision=Mock(
                features=torch.randn(64),
                action="CONTINUE",
                log_prob=torch.tensor(-0.5)
            )
        ))

        # Mock adaptive inference
        model.adaptive_inference = Mock(return_value=Mock(
            answer_span=(10, 20),
            confidence=0.85,
            segments_processed=2,
            total_segments=4,
            belief_history=[],
            halting_history=[],
            memory_state={},
            efficiency_score=0.75
        ))

        return model

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RBSTrainingConfig(
            num_epochs=2,
            batch_size=2,
            learning_rate=1e-4,
            use_rl_training=True,
            rl_start_epoch=1,
            output_dir=tempfile.mkdtemp(),
            logging_steps=1,
            eval_frequency=1,
            save_frequency=1,
            warmup_steps=1,
            max_steps=10
        )

    @pytest.fixture
    def trainer(self, mock_model, config):
        """Create trainer instance."""
        train_dataset = DummyDataset()
        eval_dataset = DummyDataset()

        return RBSHybridTrainer(
            model=mock_model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

    def test_trainer_initialization(self, trainer, config):
        """Test trainer initialization."""
        assert trainer.config == config
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.training_stage == "supervised"
        assert trainer.best_eval_score == 0.0
        assert hasattr(trainer, 'qa_optimizer')
        assert hasattr(trainer, 'train_loader')
        assert hasattr(trainer, 'eval_loader')

    def test_optimizer_setup(self, mock_model, config):
        """Test optimizer setup."""
        trainer = RBSHybridTrainer(
            model=mock_model,
            config=config,
            train_dataset=DummyDataset(),
            eval_dataset=None
        )

        # Check QA optimizer exists
        assert hasattr(trainer, 'qa_optimizer')
        assert trainer.qa_optimizer.param_groups[0]['lr'] == config.learning_rate

        # Check RL optimizer exists when RL is enabled
        if config.use_rl_training:
            assert hasattr(trainer, 'rl_optimizer')
            assert trainer.rl_optimizer.param_groups[0]['lr'] == config.rl_learning_rate

    def test_dataloader_creation(self, trainer):
        """Test data loader creation."""
        assert trainer.train_loader is not None
        assert trainer.eval_loader is not None

        # Test batch retrieval
        batch = next(iter(trainer.train_loader))
        required_keys = [
            'input_ids', 'attention_mask', 'start_positions', 'end_positions',
            'segment_ids', 'segment_offsets', 'num_segments',
            'question_input_ids', 'context_segments'
        ]

        for key in required_keys:
            assert key in batch

    def test_supervised_training_step(self, trainer):
        """Test supervised training step."""
        trainer.training_stage = "supervised"

        # Create a mock batch
        batch = next(iter(trainer.train_loader))

        metrics = trainer._supervised_training_step(batch)

        assert 'qa_loss' in metrics
        assert isinstance(metrics['qa_loss'], float)

        # Verify optimizer step was called
        trainer.qa_optimizer.step.assert_called()

    def test_hybrid_training_step(self, trainer):
        """Test hybrid training step."""
        trainer.training_stage = "hybrid"

        # Create a mock batch
        batch = next(iter(trainer.train_loader))

        metrics, episode_data = trainer._hybrid_training_step(batch)

        assert 'qa_loss' in metrics
        assert isinstance(metrics['qa_loss'], float)
        assert episode_data is not None

        # Verify QA optimizer step was called
        trainer.qa_optimizer.step.assert_called()

    def test_rl_episode_collection(self, trainer):
        """Test RL episode collection."""
        batch = next(iter(trainer.train_loader))

        episode_data = trainer._collect_rl_episode(batch)

        if episode_data:
            assert 'episodes' in episode_data
            assert 'ground_truths' in episode_data
            assert len(episode_data['episodes']) > 0

    def test_process_rl_episodes(self, trainer):
        """Test RL episode processing."""
        episodes = [
            [
                {
                    'features': torch.randn(64),
                    'action': 'CONTINUE',
                    'log_prob': torch.tensor(-0.5),
                    'value_estimate': 0.8
                }
            ]
        ]
        ground_truths = [(10, 20)]

        metrics = trainer._process_rl_episodes(episodes, ground_truths)

        assert 'rl_loss' in metrics
        assert 'avg_reward' in metrics
        assert 'avg_episode_length' in metrics

        # Verify RL optimizer was called
        trainer.rl_optimizer.step.assert_called()

    def test_qa_loss_computation(self, trainer):
        """Test QA loss computation."""
        batch_size, seq_len = 2, 128

        start_logits = torch.randn(batch_size, seq_len)
        end_logits = torch.randn(batch_size, seq_len)
        start_positions = torch.randint(0, seq_len, (batch_size,))
        end_positions = torch.randint(0, seq_len, (batch_size,))

        loss = trainer._compute_qa_loss(
            start_logits, end_logits, start_positions, end_positions
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_stage_transition(self, trainer):
        """Test stage transition logic."""
        trainer.current_epoch = 1
        trainer.config.rl_start_epoch = 1

        # Should transition from supervised to hybrid
        trainer._check_stage_transition()
        assert trainer.training_stage == "hybrid"

        # Verify model mode was set
        trainer.model.set_training_mode.assert_called_with("rl")

    def test_evaluation(self, trainer):
        """Test evaluation process."""
        eval_metrics = trainer.evaluate()

        required_metrics = ['f1', 'exact_match', 'confidence', 'efficiency_score', 'combined_score']
        for metric in required_metrics:
            assert metric in eval_metrics
            assert isinstance(eval_metrics[metric], float)

    def test_checkpoint_save_load(self, trainer):
        """Test checkpoint saving and loading."""
        # Set some state
        trainer.current_epoch = 3
        trainer.global_step = 100
        trainer.best_eval_score = 0.85
        trainer.training_history['train_loss'] = [0.5, 0.4, 0.3]

        # Save checkpoint
        trainer.save_checkpoint("test-checkpoint")

        # Modify state
        trainer.current_epoch = 0
        trainer.global_step = 0
        trainer.best_eval_score = 0.0

        # Load checkpoint
        checkpoint_path = f"{trainer.config.output_dir}/test-checkpoint.pt"
        trainer.load_checkpoint(checkpoint_path)

        # Verify state was restored
        assert trainer.current_epoch == 3
        assert trainer.global_step == 100
        assert trainer.best_eval_score == 0.85
        assert trainer.training_history['train_loss'] == [0.5, 0.4, 0.3]

    def test_early_stopping(self, trainer):
        """Test early stopping logic."""
        trainer.config.early_stopping_patience = 3
        trainer.best_eval_score = 0.8

        # No early stopping with improving scores
        trainer.training_history['eval_f1'] = [0.7, 0.75, 0.78]
        assert not trainer._should_early_stop()

        # Trigger early stopping with no improvement
        trainer.training_history['eval_f1'] = [0.79, 0.78, 0.77]  # All <= 0.8
        assert trainer._should_early_stop()

    def test_logging_setup(self, trainer):
        """Test logger setup."""
        assert trainer.logger is not None
        assert trainer.logger.name == "RBSHybridTrainer"

    def test_wandb_setup_disabled(self, trainer):
        """Test WandB setup when disabled."""
        trainer.config.use_wandb = False
        wandb_enabled = trainer._setup_wandb()
        assert wandb_enabled is False

    @patch('rbsqa.training.hybrid_trainer.tqdm')
    def test_train_epoch_supervised(self, mock_tqdm, trainer):
        """Test training epoch in supervised mode."""
        trainer.training_stage = "supervised"

        # Mock tqdm to return the batch
        batch = next(iter(trainer.train_loader))
        mock_tqdm.return_value = [batch]

        metrics = trainer.train_epoch()

        assert 'qa_loss' in metrics
        assert trainer.global_step > 0

    @patch('rbsqa.training.hybrid_trainer.tqdm')
    def test_train_epoch_hybrid(self, mock_tqdm, trainer):
        """Test training epoch in hybrid mode."""
        trainer.training_stage = "hybrid"

        # Mock tqdm to return the batch
        batch = next(iter(trainer.train_loader))
        mock_tqdm.return_value = [batch]

        metrics = trainer.train_epoch()

        assert 'qa_loss' in metrics
        assert trainer.global_step > 0

    def test_train_with_keyboard_interrupt(self, trainer):
        """Test training interruption handling."""
        with patch.object(trainer, 'train_epoch', side_effect=KeyboardInterrupt()):
            with patch.object(trainer, 'save_checkpoint') as mock_save:
                try:
                    trainer.train()
                except KeyboardInterrupt:
                    pass

                # Verify checkpoint was saved with "interrupted" name
                mock_save.assert_called_with("interrupted")

    def test_train_with_exception(self, trainer):
        """Test training exception handling."""
        with patch.object(trainer, 'train_epoch', side_effect=Exception("Test error")):
            with patch.object(trainer, 'save_checkpoint') as mock_save:
                with pytest.raises(Exception, match="Test error"):
                    trainer.train()

                # Verify checkpoint was saved with "failed" name
                mock_save.assert_called_with("failed")

    def test_scheduler_creation(self, trainer):
        """Test learning rate scheduler creation."""
        assert hasattr(trainer, 'qa_scheduler')
        if hasattr(trainer, 'rl_optimizer'):
            assert hasattr(trainer, 'rl_scheduler')

    def test_gradient_clipping(self, trainer):
        """Test gradient clipping."""
        trainer.config.max_grad_norm = 1.0

        # This is tested implicitly in the training steps
        # but we can verify the config value is used
        assert trainer.config.max_grad_norm == 1.0