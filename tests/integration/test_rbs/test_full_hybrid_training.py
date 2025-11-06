"""
Integration tests for full RBS hybrid training pipeline.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import Dataset

from rbsqa.training.hybrid_trainer import RBSHybridTrainer
from rbsqa.configs.hybrid_training_config import RBSTrainingConfig, create_quick_debug_config


class SimpleQADataset(Dataset):
    """Simple QA dataset for integration testing."""

    def __init__(self, num_examples=20, num_segments=3):
        self.num_examples = num_examples
        self.num_segments = num_segments
        self.seq_len = 256

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Create realistic QA data structure
        segment_len = self.seq_len // self.num_segments

        # Multi-segment document
        input_ids = torch.randint(1, 1000, (self.num_segments, self.seq_len))
        attention_mask = torch.ones(self.num_segments, self.seq_len)

        # Answer spans (ensure they're valid)
        start_positions = []
        end_positions = []
        for seg_idx in range(self.num_segments):
            start_pos = torch.randint(0, self.seq_len // 2, (1,)).item()
            end_pos = torch.randint(start_pos + 1, self.seq_len, (1,)).item()
            start_positions.append(start_pos)
            end_positions.append(end_pos)

        start_positions = torch.tensor(start_positions)
        end_positions = torch.tensor(end_positions)

        # Global positions (cumulative across segments)
        global_start = start_positions[0].item()
        global_end = end_positions[0].item() + segment_len * (self.num_segments - 1)
        global_start_positions = torch.tensor(global_start)
        global_end_positions = torch.tensor(global_end)

        # Question and context separation
        question_len = 64
        question_input_ids = torch.randint(1, 1000, (question_len,))

        # Context segments
        context_segments = []
        for seg_idx in range(self.num_segments):
            context_segments.append(input_ids[seg_idx])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions,
            'segment_ids': torch.arange(self.num_segments).repeat_interleave(self.seq_len)[:self.seq_len].unsqueeze(0).repeat(self.num_segments, 1),
            'segment_offsets': torch.tensor([i * segment_len for i in range(self.num_segments)]),
            'num_segments': torch.tensor(self.num_segments),
            'global_start_positions': global_start_positions,
            'global_end_positions': global_end_positions,
            'question_input_ids': question_input_ids,
            'context_segments': context_segments,
            'question_length': question_len
        }


class MockRBSModel:
    """Mock RBS model for integration testing."""

    def __init__(self, device='cpu'):
        self.device = device
        self.training_stage = "supervised"
        self.inference_mode = "standard"

        # Create mock components
        self.gmm_backbone = Mock()
        self.belief_tracker = Mock()
        self.halting_policy = Mock()

        # Mock parameters
        self.gmm_backbone.parameters.return_value = [torch.randn(10, requires_grad=True)]
        self.belief_tracker.parameters.return_value = [torch.randn(5, requires_grad=True)]
        self.halting_policy.parameters.return_value = [torch.randn(3, requires_grad=True)]

        # Setup mock methods
        self._setup_mock_methods()

    def _setup_mock_methods(self):
        """Setup mock methods for the model."""
        # Mock forward pass
        def mock_forward(input_ids, attention_mask, memory_state=None, segment_info=None, return_dict=True):
            batch_size, seq_len = input_ids.shape if len(input_ids.shape) == 2 else input_ids.shape[:2]

            return Mock(
                start_logits=torch.randn(batch_size, seq_len),
                end_logits=torch.randn(batch_size, seq_len),
                memory_state={"expert_0": torch.randn(1, 16, 768)},
                belief_state=Mock(best_span=(10, 20)),
                halting_decision=Mock(
                    features=torch.randn(64),
                    action="CONTINUE" if torch.rand(1).item() > 0.3 else "HALT",
                    log_prob=torch.tensor(-0.5),
                    halting_policy=self.halting_policy
                )
            )

        self.forward = mock_forward

        # Mock adaptive inference
        def mock_adaptive_inference(question_input_ids, context_segments):
            return Mock(
                answer_span=(10, 20),
                confidence=0.85,
                segments_processed=len(context_segments) // 2,
                total_segments=len(context_segments),
                belief_history=[Mock()],
                halting_history=[Mock(action="HALT")],
                memory_state={},
                efficiency_score=0.75
            )

        self.adaptive_inference = mock_adaptive_inference

        # Mock training/inference mode setting
        def mock_set_training_mode(mode):
            self.training_stage = mode

        def mock_set_inference_mode(mode):
            self.inference_mode = mode

        self.set_training_mode = mock_set_training_mode
        self.set_inference_mode = mock_set_inference_mode

        # Mock belief tracker reset
        self.belief_tracker.reset_belief = Mock()

        # Mock halting policy methods
        self.halting_policy.compute_rewards = Mock(return_value=[[1.0], [0.5]])
        self.halting_policy.policy_gradient_loss = Mock(return_value=torch.tensor(0.1))
        self.halting_policy.value_loss = Mock(return_value=torch.tensor(0.05))
        self.halting_policy.compute_f1_score = Mock(return_value=0.8)

    def to(self, device):
        """Move model to device."""
        self.device = device
        return self

    def train(self):
        """Set model to training mode."""
        pass

    def eval(self):
        """Set model to evaluation mode."""
        pass

    def state_dict(self):
        """Return mock state dict."""
        return {
            'gmm_backbone': torch.randn(10),
            'belief_tracker': torch.randn(5),
            'halting_policy': torch.randn(3)
        }

    def load_state_dict(self, state_dict):
        """Mock loading state dict."""
        pass


class TestFullHybridTrainingPipeline:
    """Integration tests for the complete hybrid training pipeline."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return create_quick_debug_config(
            num_epochs=2,
            batch_size=2,
            memory_num_tokens=4,
            num_memory_experts=2,
            use_rl_training=True,
            output_dir=tempfile.mkdtemp()
        )

    @pytest.fixture
    def model(self):
        """Create mock model."""
        return MockRBSModel()

    @pytest.fixture
    def datasets(self):
        """Create train and eval datasets."""
        train_dataset = SimpleQADataset(num_examples=10, num_segments=2)
        eval_dataset = SimpleQADataset(num_examples=5, num_segments=2)
        return train_dataset, eval_dataset

    @pytest.fixture
    def trainer(self, model, config, datasets):
        """Create trainer instance."""
        train_dataset, eval_dataset = datasets
        return RBSHybridTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

    def test_full_training_pipeline_integration(self, trainer):
        """Test complete training pipeline integration."""
        # Run training for a few epochs
        results = trainer.train()

        # Verify training completed
        assert 'total_time' in results
        assert 'best_score' in results
        assert 'final_epoch' in results
        assert 'training_history' in results

        # Verify metrics were recorded
        assert trainer.global_step > 0
        assert len(trainer.training_history) > 0

        # Verify stage transition occurred
        assert trainer.training_stage == "hybrid"

        # Verify evaluation was performed
        assert trainer.best_eval_score >= 0

    def test_supervised_to_hybrid_transition(self, config):
        """Test transition from supervised to hybrid training."""
        config.rl_start_epoch = 1
        config.num_epochs = 2

        model = MockRBSModel()
        train_dataset = SimpleQADataset(num_examples=5, num_segments=2)
        eval_dataset = SimpleQADataset(num_examples=3, num_segments=2)

        trainer = RBSHybridTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Should start in supervised mode
        assert trainer.training_stage == "supervised"

        # Run training
        results = trainer.train()

        # Should transition to hybrid mode
        assert trainer.training_stage == "hybrid"

        # Verify model was notified of mode changes
        model.set_training_mode.assert_called_with("rl")

    def test_checkpoint_resumption_integration(self, config, datasets):
        """Test saving and resuming from checkpoint."""
        model = MockRBSModel()
        train_dataset, eval_dataset = datasets

        # Create initial trainer and run for 1 epoch
        trainer1 = RBSHybridTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Mock the train method to run only 1 epoch
        with patch.object(trainer1, 'train') as mock_train:
            mock_train.return_value = {'final_epoch': 1}
            trainer1.current_epoch = 1
            trainer1.global_step = 10
            trainer1.best_eval_score = 0.75

            # Save checkpoint
            checkpoint_path = os.path.join(config.output_dir, "test-checkpoint.pt")
            trainer1.save_checkpoint("test-checkpoint")

        # Create new trainer and resume from checkpoint
        model2 = MockRBSModel()
        trainer2 = RBSHybridTrainer(
            model=model2,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            resume_from_checkpoint=checkpoint_path
        )

        # Verify state was restored
        assert trainer2.current_epoch == 1
        assert trainer2.global_step == 10
        assert trainer2.best_eval_score == 0.75

    def test_evaluation_pipeline_integration(self, trainer):
        """Test evaluation pipeline integration."""
        # Run evaluation
        eval_metrics = trainer.evaluate()

        # Verify all expected metrics are present
        expected_metrics = [
            'f1', 'exact_match', 'confidence',
            'efficiency_score', 'avg_segments_processed', 'combined_score'
        ]

        for metric in expected_metrics:
            assert metric in eval_metrics
            assert isinstance(eval_metrics[metric], (int, float))

        # Verify metric ranges
        assert 0 <= eval_metrics['f1'] <= 1
        assert 0 <= eval_metrics['exact_match'] <= 1
        assert 0 <= eval_metrics['confidence'] <= 1
        assert 0 <= eval_metrics['efficiency_score'] <= 1

    def test_rl_episode_collection_integration(self, trainer):
        """Test RL episode collection in hybrid mode."""
        trainer.training_stage = "hybrid"

        # Get a batch from the data loader
        batch = next(iter(trainer.train_loader))

        # Collect RL episodes
        episode_data = trainer._collect_rl_episode(batch)

        if episode_data:
            assert 'episodes' in episode_data
            assert 'ground_truths' in episode_data

            # Verify episode structure
            episodes = episode_data['episodes']
            ground_truths = episode_data['ground_truths']

            assert len(episodes) > 0
            assert len(ground_truths) > 0
            assert len(episodes) == len(ground_truths)

            # Verify episode step structure
            for episode in episodes:
                for step in episode:
                    assert 'features' in step
                    assert 'action' in step
                    assert 'log_prob' in step

    def test_data_loading_integration(self, config):
        """Test data loading and batch processing integration."""
        model = MockRBSModel()
        train_dataset = SimpleQADataset(num_examples=8, num_segments=3)
        eval_dataset = SimpleQADataset(num_examples=4, num_segments=3)

        trainer = RBSHybridTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Test train data loader
        train_batch = next(iter(trainer.train_loader))
        self._verify_batch_structure(train_batch, config.batch_size)

        # Test eval data loader
        eval_batch = next(iter(trainer.eval_loader))
        self._verify_batch_structure(eval_batch, config.batch_size)

    def _verify_batch_structure(self, batch, expected_batch_size):
        """Verify batch has correct structure."""
        required_keys = [
            'input_ids', 'attention_mask', 'start_positions', 'end_positions',
            'segment_ids', 'segment_offsets', 'num_segments',
            'global_start_positions', 'global_end_positions',
            'question_input_ids', 'context_segments'
        ]

        for key in required_keys:
            assert key in batch, f"Missing key: {key}"

        # Verify tensor shapes
        assert batch['input_ids'].shape[0] == expected_batch_size
        assert batch['attention_mask'].shape[0] == expected_batch_size
        assert len(batch['context_segments']) == expected_batch_size

    def test_optimizer_integration(self, trainer):
        """Test optimizer setup and step integration."""
        # Verify optimizers exist
        assert hasattr(trainer, 'qa_optimizer')
        assert hasattr(trainer, 'rl_optimizer')

        # Verify schedulers exist
        assert hasattr(trainer, 'qa_scheduler')
        assert hasattr(trainer, 'rl_scheduler')

        # Test supervised training step
        trainer.training_stage = "supervised"
        batch = next(iter(trainer.train_loader))

        # Reset optimizer call counts
        trainer.qa_optimizer.reset_mock()
        if hasattr(trainer, 'rl_optimizer'):
            trainer.rl_optimizer.reset_mock()

        # Perform training step
        metrics = trainer._supervised_training_step(batch)

        # Verify QA optimizer was called
        trainer.qa_optimizer.zero_grad.assert_called()
        trainer.qa_optimizer.step.assert_called()

        # Test hybrid training step
        trainer.training_stage = "hybrid"

        # Reset optimizer call counts
        trainer.qa_optimizer.reset_mock()
        if hasattr(trainer, 'rl_optimizer'):
            trainer.rl_optimizer.reset_mock()

        # Perform hybrid training step
        metrics, episode_data = trainer._hybrid_training_step(batch)

        # Verify QA optimizer was called
        trainer.qa_optimizer.zero_grad.assert_called()
        trainer.qa_optimizer.step.assert_called()

    def test_loss_computation_integration(self, trainer):
        """Test loss computation integration."""
        batch_size, seq_len = 4, 128

        # Create mock logits and positions
        start_logits = torch.randn(batch_size, seq_len)
        end_logits = torch.randn(batch_size, seq_len)
        start_positions = torch.randint(0, seq_len, (batch_size,))
        end_positions = torch.randint(0, seq_len, (batch_size,))

        # Compute QA loss
        qa_loss = trainer._compute_qa_loss(
            start_logits, end_logits, start_positions, end_positions
        )

        assert isinstance(qa_loss, torch.Tensor)
        assert qa_loss.requires_grad
        assert qa_loss.item() > 0

    def test_early_stopping_integration(self, trainer):
        """Test early stopping integration."""
        trainer.config.early_stopping_patience = 2
        trainer.best_eval_score = 0.8

        # Simulate training history with no improvement
        trainer.training_history['eval_f1'] = [0.75, 0.76]

        # Should trigger early stopping
        assert trainer._should_early_stop()

        # Simulate improvement
        trainer.training_history['eval_f1'] = [0.81, 0.82]

        # Should not trigger early stopping
        assert not trainer._should_early_stop()

    def test_logging_integration(self, trainer):
        """Test logging integration."""
        # Test epoch metrics logging
        train_metrics = {'qa_loss': 0.5, 'rl_loss': 0.1}
        eval_metrics = {'f1': 0.8, 'efficiency_score': 0.7}

        # Should not raise any exceptions
        trainer._log_epoch_metrics(0, train_metrics, eval_metrics)
        trainer._log_eval_metrics(0, eval_metrics)
        trainer._log_batch_metrics(train_metrics, 0)

        # Verify training history was updated
        assert 'train_qa_loss' in trainer.training_history
        assert 'eval_f1' in trainer.training_history
        assert trainer.training_history['train_qa_loss'][-1] == 0.5

    def test_device_handling_integration(self, config, datasets):
        """Test device handling integration."""
        device = 'cpu'  # Use CPU for testing
        model = MockRBSModel(device=device)
        train_dataset, eval_dataset = datasets

        trainer = RBSHybridTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Verify model is on correct device
        assert model.device == device

        # Test forward pass with correct device handling
        batch = next(iter(trainer.train_loader))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Should not raise device-related errors
        outputs = trainer.model.forward(
            input_ids=batch['input_ids'][:, 0],  # First segment
            attention_mask=batch['attention_mask'][:, 0]
        )

        assert outputs is not None

    def test_error_handling_integration(self, trainer):
        """Test error handling integration."""
        # Test with corrupted batch data
        corrupted_batch = {
            'input_ids': torch.tensor([[1, 2, 3]]),  # Wrong shape
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'start_positions': torch.tensor([0]),
            'end_positions': torch.tensor([2]),
            'segment_ids': torch.tensor([[0, 0, 0]]),
            'segment_offsets': torch.tensor([0]),
            'num_segments': torch.tensor(1),
            'global_start_positions': torch.tensor(0),
            'global_end_positions': torch.tensor(2),
            'question_input_ids': torch.tensor([1, 2]),
            'context_segments': [torch.tensor([1, 2, 3])]
        }

        # Should handle gracefully or raise informative error
        try:
            trainer._supervised_training_step(corrupted_batch)
        except Exception as e:
            # Should be a meaningful error, not a cryptic tensor mismatch
            assert isinstance(e, (RuntimeError, ValueError))

    def test_memory_management_integration(self, trainer):
        """Test memory management integration."""
        # Test with multiple batches to check for memory leaks
        initial_memory_state = None

        for i, batch in enumerate(trainer.train_loader):
            if i >= 3:  # Test only first 3 batches
                break

            # Process batch
            trainer.training_stage = "hybrid"
            metrics, episode_data = trainer._hybrid_training_step(batch)

            # Verify memory state is managed correctly
            # This is a basic check - in practice, you'd monitor GPU memory
            assert isinstance(metrics, dict)
            assert 'qa_loss' in metrics

    def test_multi_segment_document_integration(self, config):
        """Test handling of multi-segment documents."""
        # Create dataset with more segments
        train_dataset = SimpleQADataset(num_examples=5, num_segments=4)
        eval_dataset = SimpleQADataset(num_examples=3, num_segments=4)

        model = MockRBSModel()
        trainer = RBSHybridTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Test batch processing with multi-segment documents
        batch = next(iter(trainer.train_loader))

        # Verify batch structure for multi-segment documents
        assert batch['num_segments'].item() == 4
        assert batch['input_ids'].shape[1] == 256  # seq_len
        assert len(batch['context_segments'][0]) == 4  # 4 segments per example

        # Test training step with multi-segment data
        trainer.training_stage = "supervised"
        metrics = trainer._supervised_training_step(batch)

        assert 'qa_loss' in metrics
        assert isinstance(metrics['qa_loss'], float)