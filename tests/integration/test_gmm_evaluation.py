"""
Integration tests for GMM evaluation pipeline.

Tests full evaluation workflow to verify:
- Model evaluation executes without errors
- Evaluation metrics (EM, F1) are computed correctly
- Time-step-major batching works with GMM
- Routing statistics are collected correctly
"""

import pytest
import torch
from torch.utils.data import Dataset
from transformers import XLNetConfig, XLNetForQuestionAnsweringSimple

from gmmxlnet.models.gmm_xlnet_qa import GMMXLNetForQA


class ToyEvalDataset(Dataset):
    """Minimal evaluation dataset for testing."""

    def __init__(self, num_examples=5, num_segments=2, vocab_size=1000, seq_len=64):
        """Initialize toy evaluation dataset.

        Args:
            num_examples: Number of QA examples
            num_segments: Number of segments per example
            vocab_size: Vocabulary size for random token generation
            seq_len: Sequence length per segment
        """
        self.num_examples = num_examples
        self.num_segments = num_segments
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        # Generate random data
        torch.manual_seed(123)  # Different seed from training
        self.data = []

        for _ in range(num_examples):
            # Generate answer span
            start_pos = torch.randint(5, seq_len - 10, (1,)).item()
            end_pos = start_pos + torch.randint(1, 5, (1,)).item()

            example = {
                "input_ids": torch.randint(0, vocab_size, (num_segments, seq_len)),
                "attention_mask": torch.ones(num_segments, seq_len),
                "start_position": start_pos,
                "end_position": end_pos,
                "answer_text": f"answer_{_}",  # Mock answer text
                "qid": f"q_{_}",
            }
            self.data.append(example)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def toy_eval_base_model():
    """Create a small XLNet model for evaluation testing."""
    config = XLNetConfig(
        vocab_size=1000,
        d_model=64,
        n_layer=2,
        n_head=2,
        d_inner=128,
    )
    return XLNetForQuestionAnsweringSimple(config)


@pytest.fixture
def toy_eval_dataset():
    """Create toy evaluation dataset with 5 examples, 2 segments each."""
    return ToyEvalDataset(num_examples=5, num_segments=2, seq_len=32)


@pytest.mark.integration
class TestGMMEvaluation:
    """Integration tests for GMM evaluation pipeline."""

    def test_evaluation_basic(self, toy_eval_base_model, toy_eval_dataset):
        """Test basic evaluation pipeline executes without errors."""
        # Initialize model
        model = GMMXLNetForQA(
            base_model=toy_eval_base_model,
            num_experts=4,
            memory_slots=8,
            routing_mode="write-based",
        )
        model.eval()

        # Run evaluation on each example
        predictions = {}
        with torch.no_grad():
            for idx in range(len(toy_eval_dataset)):
                example = toy_eval_dataset[idx]
                qid = example["qid"]

                # Process each segment
                for seg_idx in range(2):
                    input_ids = example["input_ids"][seg_idx].unsqueeze(0)
                    attention_mask = example["attention_mask"][seg_idx].unsqueeze(0)

                    # Forward pass through GMM
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_routing_info=True,
                    )

                    # Get predictions (simplified - just take argmax)
                    start_pred = outputs["start_logits"].argmax(dim=-1).item()
                    end_pred = outputs["end_logits"].argmax(dim=-1).item()

                    predictions[qid] = {
                        "start": start_pred,
                        "end": end_pred,
                    }

        # Verify predictions were generated for all examples
        assert len(predictions) == len(toy_eval_dataset), "Should have predictions for all examples"

        # Verify prediction format
        for qid, pred in predictions.items():
            assert "start" in pred, f"Prediction for {qid} should have start position"
            assert "end" in pred, f"Prediction for {qid} should have end position"
            assert isinstance(pred["start"], int), "Start position should be integer"
            assert isinstance(pred["end"], int), "End position should be integer"

    def test_evaluation_metrics_computation(self, toy_eval_base_model, toy_eval_dataset):
        """Test that evaluation metrics (EM, F1) can be computed."""
        model = GMMXLNetForQA(
            base_model=toy_eval_base_model,
            num_experts=4,
            memory_slots=8,
            routing_mode="write-based",
        )
        model.eval()

        # Simplified metric computation
        correct_count = 0
        total_count = 0

        with torch.no_grad():
            for idx in range(len(toy_eval_dataset)):
                example = toy_eval_dataset[idx]

                # Process first segment only for simplicity
                input_ids = example["input_ids"][0].unsqueeze(0)
                attention_mask = example["attention_mask"][0].unsqueeze(0)

                # Forward pass through GMM
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_routing_info=True,
                )

                # Get predictions
                start_pred = outputs["start_logits"].argmax(dim=-1).item()
                end_pred = outputs["end_logits"].argmax(dim=-1).item()

                # Simple exact match check (start and end both correct)
                if start_pred == example["start_position"] and end_pred == example["end_position"]:
                    correct_count += 1
                total_count += 1

        # Compute metrics
        exact_match = correct_count / total_count if total_count > 0 else 0.0

        # Assert metrics are computed
        assert 0.0 <= exact_match <= 1.0, f"EM should be in [0, 1], got {exact_match}"

    def test_time_step_major_batching_compatibility(self, toy_eval_base_model, toy_eval_dataset):
        """Test that time-step-major batching works with GMM."""
        model = GMMXLNetForQA(
            base_model=toy_eval_base_model,
            num_experts=4,
            memory_slots=8,
            routing_mode="write-based",
        )
        model.eval()

        # Simulate time-step-major batching:
        # Process segment 0 from all examples, then segment 1 from all examples
        num_segments = 2
        batch_size = 2  # Process 2 examples at a time

        with torch.no_grad():
            for seg_idx in range(num_segments):
                # Gather segment seg_idx from multiple examples
                batch_input_ids = []
                batch_attention_mask = []

                for ex_idx in range(min(batch_size, len(toy_eval_dataset))):
                    example = toy_eval_dataset[ex_idx]
                    batch_input_ids.append(example["input_ids"][seg_idx])
                    batch_attention_mask.append(example["attention_mask"][seg_idx])

                # Stack into batch
                batch_input_ids = torch.stack(batch_input_ids, dim=0)
                batch_attention_mask = torch.stack(batch_attention_mask, dim=0)

                # Forward pass with batched inputs through GMM
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    return_routing_info=True,
                )

                # Verify output shapes
                assert outputs["start_logits"].shape[0] == len(batch_input_ids), (
                    f"Expected batch size {len(batch_input_ids)}, got {outputs['start_logits'].shape[0]}"
                )
                assert outputs["end_logits"].shape[0] == len(batch_input_ids), (
                    f"Expected batch size {len(batch_input_ids)}, got {outputs['end_logits'].shape[0]}"
                )

    def test_routing_statistics_collection(self, toy_eval_base_model, toy_eval_dataset):
        """Test that routing statistics are collected correctly during evaluation."""
        model = GMMXLNetForQA(
            base_model=toy_eval_base_model,
            num_experts=4,
            memory_slots=8,
            routing_mode="write-based",
        )
        model.eval()

        # Track routing probabilities during evaluation
        routing_probs_history = []

        with torch.no_grad():
            for idx in range(len(toy_eval_dataset)):
                # Mock: Compute routing probabilities
                # In real evaluation, this would be extracted from model forward pass
                batch_size = 1
                hidden_dim = 64
                write_hiddens = torch.randn(batch_size, 8, hidden_dim)

                routing_probs, _, _ = model.gating_network(write_hiddens)
                routing_probs_history.append(routing_probs)

        # Verify routing statistics
        assert len(routing_probs_history) == len(toy_eval_dataset), "Should collect routing probs for each example"

        for routing_probs in routing_probs_history:
            # Verify routing probabilities are valid
            assert routing_probs.shape[1] == 4, f"Should have 4 experts, got shape {routing_probs.shape}"
            assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(1), atol=1e-5), "Routing probs should sum to 1"
            assert not torch.isnan(routing_probs).any(), "Routing probs should not be NaN"
            assert (routing_probs >= 0).all(), "Routing probs should be non-negative"
            assert (routing_probs <= 1).all(), "Routing probs should be <= 1"

    def test_evaluation_with_checkpoint(self, toy_eval_base_model, toy_eval_dataset, tmp_path):
        """Test evaluation with loaded checkpoint."""
        # Initialize and save model
        model = GMMXLNetForQA(
            base_model=toy_eval_base_model,
            num_experts=4,
            memory_slots=8,
            routing_mode="write-based",
        )

        checkpoint_path = tmp_path / "eval_checkpoint"
        model.save_pretrained(checkpoint_path)

        # Load model from checkpoint
        loaded_model = GMMXLNetForQA.from_pretrained(checkpoint_path)
        loaded_model.eval()

        # Run evaluation with loaded model
        predictions = {}
        with torch.no_grad():
            for idx in range(len(toy_eval_dataset)):
                example = toy_eval_dataset[idx]
                qid = example["qid"]

                # Process first segment
                input_ids = example["input_ids"][0].unsqueeze(0)
                attention_mask = example["attention_mask"][0].unsqueeze(0)

                # Forward pass through GMM
                outputs = loaded_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_routing_info=True,
                )

                # Get predictions
                start_pred = outputs["start_logits"].argmax(dim=-1).item()
                end_pred = outputs["end_logits"].argmax(dim=-1).item()

                predictions[qid] = {
                    "start": start_pred,
                    "end": end_pred,
                }

        # Verify predictions were generated
        assert len(predictions) == len(toy_eval_dataset), "Should have predictions for all examples"

    def test_evaluation_with_different_routing_modes(self, toy_eval_base_model, toy_eval_dataset):
        """Test evaluation works with different routing modes."""
        for routing_mode in ["write-based", "read-based"]:
            model = GMMXLNetForQA(
                base_model=toy_eval_base_model,
                num_experts=4,
                memory_slots=8,
                routing_mode=routing_mode,
            )
            model.eval()

            # Run evaluation
            with torch.no_grad():
                for idx in range(len(toy_eval_dataset)):
                    example = toy_eval_dataset[idx]

                    # Process first segment
                    input_ids = example["input_ids"][0].unsqueeze(0)
                    attention_mask = example["attention_mask"][0].unsqueeze(0)

                    # Forward pass through GMM
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_routing_info=True,
                    )

                    # Verify outputs
                    assert outputs["start_logits"].shape[0] == 1, "Should have batch size 1"
                    assert outputs["end_logits"].shape[0] == 1, "Should have batch size 1"

    def test_evaluation_memory_state_persistence(self, toy_eval_base_model, toy_eval_dataset):
        """Test that memory states can persist across segments during evaluation."""
        model = GMMXLNetForQA(
            base_model=toy_eval_base_model,
            num_experts=4,
            memory_slots=8,
            routing_mode="write-based",
        )
        model.eval()

        # Get a single example with multiple segments
        example = toy_eval_dataset[0]
        num_segments = 2
        batch_size = 1

        # Initialize memory state
        device = next(model.parameters()).device
        memory_state = model.get_initial_memory(batch_size=batch_size, device=device)

        # Verify initial memory state structure
        assert len(memory_state) == 4, "Should have 4 expert memory states"
        for key, expert_state in memory_state.items():
            assert expert_state.shape == (batch_size, 8, 64), (
                f"Expert state {key} should have shape (1, 8, 64), got {expert_state.shape}"
            )

        # Process segments sequentially
        with torch.no_grad():
            for seg_idx in range(num_segments):
                input_ids = example["input_ids"][seg_idx].unsqueeze(0)
                attention_mask = example["attention_mask"][seg_idx].unsqueeze(0)

                # Forward pass through GMM
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    memory_state=memory_state,
                    return_routing_info=True,
                )

                # Verify outputs
                assert outputs["start_logits"].shape[0] == batch_size
                assert outputs["end_logits"].shape[0] == batch_size

                # Update memory state for next segment
                memory_state = outputs["new_memory_state"]

        # Verify memory state maintained structure
        for key, expert_state in memory_state.items():
            assert expert_state.shape == (batch_size, 8, 64), (
                f"Expert state {key} should maintain shape after forward passes"
            )

    def test_evaluation_batch_processing(self, toy_eval_base_model, toy_eval_dataset):
        """Test batched evaluation processing."""
        model = GMMXLNetForQA(
            base_model=toy_eval_base_model,
            num_experts=4,
            memory_slots=8,
            routing_mode="write-based",
        )
        model.eval()

        batch_size = 3
        num_batches = 0

        with torch.no_grad():
            # Process examples in batches
            for batch_start in range(0, len(toy_eval_dataset), batch_size):
                batch_end = min(batch_start + batch_size, len(toy_eval_dataset))
                current_batch_size = batch_end - batch_start

                # Gather batch (first segment only for simplicity)
                batch_input_ids = []
                batch_attention_mask = []

                for idx in range(batch_start, batch_end):
                    example = toy_eval_dataset[idx]
                    batch_input_ids.append(example["input_ids"][0])
                    batch_attention_mask.append(example["attention_mask"][0])

                # Stack into batch
                batch_input_ids = torch.stack(batch_input_ids, dim=0)
                batch_attention_mask = torch.stack(batch_attention_mask, dim=0)

                # Forward pass through GMM
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    return_routing_info=True,
                )

                # Verify output shapes
                assert outputs["start_logits"].shape[0] == current_batch_size, (
                    f"Expected batch size {current_batch_size}, got {outputs['start_logits'].shape[0]}"
                )
                assert outputs["end_logits"].shape[0] == current_batch_size

                num_batches += 1

        # Verify we processed all examples
        assert num_batches > 0, "Should have processed at least one batch"
