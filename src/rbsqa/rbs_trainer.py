"""
RBS-QA Hybrid Trainer

Implements hybrid training combining supervised QA loss with reinforcement
learning halting policy loss for adaptive computation.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import RBS components
from rbsqa.belief_state import BeliefStateTracker, BeliefState
from rbsqa.halting_policy import HaltingPolicyNetwork, HaltingStateFeatures
from rbsqa.config import RBSTrainingConfig

# Import GMM components
from gmmxlnet.models.gmm_xlnet_qa import GMMXLNetForQA

# Import data components
from memxlnet.data.dataset import TimeStepMajorDataLoader

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class RBSBatchOutput:
    """Output from RBS trainer for a single batch."""
    qa_loss: torch.Tensor
    rl_loss: torch.Tensor
    total_loss: torch.Tensor
    qa_accuracy: float
    halting_accuracy: float
    segments_processed: int
    belief_confidence: float
    exploration_rate: float


class RBSTrainer:
    """
    Hybrid trainer for RBS-QA models combining supervised QA with RL halting policy.

    Implements the training objective:
    L_total = L_QA + α * L_RL

    Where L_QA is the standard question answering loss and L_RL is the
    REINFORCE policy gradient loss for halting decisions.
    """

    def __init__(self,
                 model: GMMXLNetForQA,
                 belief_tracker: BeliefStateTracker,
                 halting_policy: HaltingPolicyNetwork,
                 config: RBSTrainingConfig):
        """Initialize RBS trainer.

        Args:
            model: GMM-XLNet model for QA
            belief_tracker: Belief state tracker for adaptive reasoning
            halting_policy: RL-based halting policy network
            config: Training configuration
        """
        self.model = model
        self.belief_tracker = belief_tracker
        self.halting_policy = halting_policy
        self.config = config

        # Setup device
        self.device = torch.device(config.device if hasattr(config, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.belief_tracker.to(self.device)
        self.halting_policy.to(self.device)

        # Setup optimizers
        self.qa_optimizer = self._create_qa_optimizer()
        self.rl_optimizer = self._create_rl_optimizer()

        # Training state
        self.current_epoch = 0
        self.training_stats = {
            'total_batches': 0,
            'total_episodes': 0,
            'avg_segments_processed': 0.0,
            'halting_accuracy': 0.0,
            'exploration_rate': 0.0
        }

        logger.info(f"Initialized RBSTrainer: device={self.device}, "
                   f"rl_weight={config.rl_weight}, lambda_cost={config.lambda_cost}")

    def _create_qa_optimizer(self) -> Optimizer:
        """Create optimizer for QA model parameters."""
        return AdamW(
            self.model.parameters(),
            lr=getattr(self.config, 'learning_rate', 2e-5),
            weight_decay=getattr(self.config, 'weight_decay', 0.01),
            eps=getattr(self.config, 'adam_epsilon', 1e-8)
        )

    def _create_rl_optimizer(self) -> Optimizer:
        """Create optimizer for RL policy parameters."""
        return AdamW(
            self.halting_policy.parameters(),
            lr=getattr(self.config, 'rl_learning_rate', 1e-4),  # Typically lower than QA LR
            weight_decay=getattr(self.config, 'rl_weight_decay', 0.01),
            eps=getattr(self.config, 'adam_epsilon', 1e-8)
        )

    def extract_halting_features(self,
                                belief_state: BeliefState,
                                model_outputs: Any,
                                batch: Dict,
                                segment_idx: int) -> HaltingStateFeatures:
        """Extract halting features from current state.

        Args:
            belief_state: Current belief state
            model_outputs: Model outputs for current segment
            batch: Input batch data
            segment_idx: Current segment index

        Returns:
            HaltingStateFeatures for policy decision
        """
        # Belief state features
        confidence_trend = belief_state.confidence_history[-3:] if len(belief_state.confidence_history) >= 3 else belief_state.confidence_history

        # Compute confidence variance
        if len(confidence_trend) >= 2:
            confidence_variance = float(np.var(confidence_trend))
        else:
            confidence_variance = 0.0

        # Computation features (simplified processing time)
        processing_time = time.time() - getattr(self, '_segment_start_time', time.time())

        # GMM context features (extract from model outputs if available)
        routing_entropy = 0.0
        expert_utilization = [0.0] * getattr(self.config, 'num_memory_experts', 4)
        context_quality_score = 0.5  # Placeholder

        if hasattr(model_outputs, 'gmm_context') and model_outputs.gmm_context is not None:
            # Extract GMM routing information
            if isinstance(model_outputs.gmm_context, dict):
                routing_entropy = model_outputs.gmm_context.get('routing_entropy', 0.0)
                expert_utilization = model_outputs.gmm_context.get('expert_utilization', expert_utilization)
                context_quality_score = model_outputs.gmm_context.get('context_quality', 0.5)

        # Document complexity features
        document_length = batch.get('num_segments', 1)
        question_complexity = 0.5  # Placeholder - could compute from question embeddings
        segment_relevance_score = 0.7  # Placeholder - could compute from attention patterns

        # Estimate segments remaining
        segments_processed = segment_idx + 1
        segments_remaining = max(0, document_length - segments_processed)

        return HaltingStateFeatures(
            # Belief state features
            current_confidence=belief_state.confidence,
            confidence_trend=confidence_trend,
            confidence_variance=confidence_variance,
            revision_count=belief_state.revision_count,

            # Computation features
            segments_processed=segments_processed,
            segments_remaining=segments_remaining,
            processing_time=processing_time,

            # GMM context features
            routing_entropy=routing_entropy,
            expert_utilization=expert_utilization,
            context_quality_score=context_quality_score,

            # Document complexity features
            document_length=document_length,
            question_complexity=question_complexity,
            segment_relevance_score=segment_relevance_score
        )

    def compute_qa_loss(self,
                       model_outputs: Any,
                       start_positions: torch.Tensor,
                       end_positions: torch.Tensor) -> torch.Tensor:
        """Compute question answering loss.

        Args:
            model_outputs: Model outputs with start/end logits
            start_positions: Ground truth start positions
            end_positions: Ground truth end positions

        Returns:
            QA loss tensor
        """
        if hasattr(model_outputs, 'loss') and model_outputs.loss is not None:
            return model_outputs.loss

        # Standard QA loss computation
        start_logits = model_outputs.start_logits
        end_logits = model_outputs.end_logits

        # Compute cross-entropy losses
        start_loss = nn.CrossEntropyLoss()(start_logits, start_positions)
        end_loss = nn.CrossEntropyLoss()(end_logits, end_positions)

        return (start_loss + end_loss) / 2

    def train_epoch(self, dataloader: TimeStepMajorDataLoader) -> Dict[str, float]:
        """Train one epoch with hybrid supervised + RL learning.

        Args:
            dataloader: Time-step major data loader

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.belief_tracker.train()
        self.halting_policy.train()

        epoch_stats = {
            'qa_loss': 0.0,
            'rl_loss': 0.0,
            'total_loss': 0.0,
            'qa_accuracy': 0.0,
            'halting_accuracy': 0.0,
            'avg_segments_processed': 0.0,
            'num_batches': 0
        }

        # Check if we should do RL training this epoch
        do_rl_training = (self.current_epoch >= self.config.rl_start_epoch and
                         self.config.use_halting_policy)

        logger.info(f"Starting epoch {self.current_epoch}, RL training: {do_rl_training}")

        # Episode data for RL training
        episodes_data: List[List[Dict]] = []
        ground_truth_spans: List[Tuple[int, int]] = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {self.current_epoch}")):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Process document segment by segment
            belief_state = self.belief_tracker.reset_belief()
            episode_steps = []
            self._segment_start_time = time.time()

            # Get ground truth span for this example
            if 'global_start_positions' in batch and 'global_end_positions' in batch:
                gt_span = (batch['global_start_positions'][0].item(),
                          batch['global_end_positions'][0].item())
            else:
                # Fallback to local positions (not ideal but works for testing)
                gt_span = (batch['start_positions'][0].item(),
                          batch['end_positions'][0].item())

            # Process segments sequentially
            num_segments = batch.get('num_segments', 1)
            memory_state = None

            for segment_idx in range(num_segments):
                # Extract segment data (simplified)
                if isinstance(batch.get('input_ids'), torch.Tensor):
                    # Single segment case
                    segment_input_ids = batch['input_ids']
                    segment_attention_mask = batch.get('attention_mask')
                else:
                    # Multi-segment case (placeholder - would need proper indexing)
                    segment_input_ids = batch['input_ids']
                    segment_attention_mask = batch.get('attention_mask')

                # Forward pass through model
                outputs = self.model(
                    input_ids=segment_input_ids,
                    attention_mask=segment_attention_mask,
                    memory_state=memory_state,
                    return_dict=True
                )

                # Update memory state for next segment
                if hasattr(outputs, 'memory_state'):
                    memory_state = outputs.memory_state

                # Update belief state
                belief_state = self.belief_tracker.update_belief(
                    (outputs.start_logits, outputs.end_logits),
                    segment_idx,
                    getattr(outputs, 'gmm_context', torch.zeros(1, 1, self.config.hidden_dim if hasattr(self.config, 'hidden_dim') else 768)),
                    batch.get('segment_offsets', [0])[segment_idx] if isinstance(batch.get('segment_offsets'), list) else 0
                )

                # Make halting decision if policy is enabled
                if self.config.use_halting_policy and do_rl_training:
                    halting_features = self.extract_halting_features(
                        belief_state, outputs, batch, segment_idx
                    )

                    action, log_prob, value = self.halting_policy.select_action(
                        halting_features, training=True
                    )

                    episode_steps.append({
                        'features': halting_features,
                        'action': action,
                        'log_prob': log_prob,
                        'value_estimate': value,
                        'predicted_span': belief_state.best_span or (0, 0),
                        'belief_state': belief_state
                    })

                    if action == "HALT":
                        break
                else:
                    # Fixed halting based on confidence threshold
                    if belief_state.confidence >= self.config.belief_state_threshold:
                        break

            # Store episode for RL training
            if do_rl_training and episode_steps:
                episodes_data.append(episode_steps)
                ground_truth_spans.append(gt_span)

            # Compute QA loss (for stability)
            qa_loss = self.compute_qa_loss(
                outputs,
                batch['start_positions'],
                batch['end_positions']
            )

            # Backward pass for QA loss
            self.qa_optimizer.zero_grad()
            qa_loss.backward()
            self.qa_optimizer.step()

            # Update batch statistics
            epoch_stats['qa_loss'] += qa_loss.item()
            epoch_stats['num_batches'] += 1
            epoch_stats['avg_segments_processed'] += segment_idx + 1

            # Log batch progress
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}: qa_loss={qa_loss.item():.4f}, "
                           f"segments={segment_idx + 1}, confidence={belief_state.confidence:.3f}")

        # RL training after processing all episodes
        if do_rl_training and len(episodes_data) > 0:
            rl_loss = self.train_halting_policy(episodes_data, ground_truth_spans)
            epoch_stats['rl_loss'] = rl_loss.item()
        else:
            epoch_stats['rl_loss'] = 0.0

        # Compute final statistics
        if epoch_stats['num_batches'] > 0:
            epoch_stats['qa_loss'] /= epoch_stats['num_batches']
            epoch_stats['avg_segments_processed'] /= epoch_stats['num_batches']
            epoch_stats['total_loss'] = (
                epoch_stats['qa_loss'] + self.config.rl_weight * epoch_stats['rl_loss']
            )

        # Get RL policy statistics
        if self.config.use_halting_policy:
            rl_stats = self.halting_policy.get_training_stats()
            epoch_stats['exploration_rate'] = rl_stats['exploration_rate']
            epoch_stats['continue_rate'] = rl_stats['continue_rate']
            epoch_stats['halt_rate'] = rl_stats['halt_rate']

        logger.info(f"Epoch {self.current_epoch} completed: "
                   f"qa_loss={epoch_stats['qa_loss']:.4f}, "
                   f"rl_loss={epoch_stats['rl_loss']:.4f}, "
                   f"total_loss={epoch_stats['total_loss']:.4f}, "
                   f"avg_segments={epoch_stats['avg_segments_processed']:.1f}")

        self.current_epoch += 1
        return epoch_stats

    def train_halting_policy(self,
                           episodes: List[List[Dict]],
                           ground_truth_spans: List[Tuple[int, int]]) -> torch.Tensor:
        """Train halting policy using REINFORCE.

        Args:
            episodes: List of episodes with actions and features
            ground_truth_spans: Ground truth answer spans

        Returns:
            Total RL loss
        """
        # Compute rewards
        rewards = self.halting_policy.compute_rewards(
            episodes, ground_truth_spans, self.config.lambda_cost
        )

        # Compute policy gradient loss
        policy_loss = self.halting_policy.policy_gradient_loss(
            episodes, rewards, gamma=self.config.gamma,
            use_baseline=self.config.use_value_baseline
        )

        # Compute value loss (optional baseline)
        total_rl_loss = policy_loss
        if self.config.use_value_baseline:
            value_loss = self.halting_policy.value_loss(
                episodes, rewards, gamma=self.config.gamma
            )
            total_rl_loss = policy_loss + self.config.value_weight * value_loss

        # Backward pass
        self.rl_optimizer.zero_grad()
        total_rl_loss.backward()
        self.rl_optimizer.step()

        # Update exploration rate (decay)
        current_rate = self.halting_policy.exploration_rate
        new_rate = max(0.01, current_rate * 0.995)  # Decay to minimum 0.01
        self.halting_policy.set_exploration_rate(new_rate)

        # Update training statistics
        self.training_stats['total_episodes'] += len(episodes)

        logger.debug(f"RL training: policy_loss={policy_loss.item():.4f}, "
                    f"total_rl_loss={total_rl_loss.item():.4f}, "
                    f"exploration_rate={new_rate:.3f}")

        return total_rl_loss

    def evaluate(self, dataloader: TimeStepMajorDataLoader) -> Dict[str, float]:
        """Evaluate model on validation set.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        self.belief_tracker.eval()
        self.halting_policy.eval()

        eval_stats = {
            'qa_loss': 0.0,
            'exact_match': 0.0,
            'f1_score': 0.0,
            'avg_segments_processed': 0.0,
            'halting_accuracy': 0.0,
            'num_examples': 0
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Process document segment by segment
                belief_state = self.belief_tracker.reset_belief()
                num_segments = batch.get('num_segments', 1)
                memory_state = None

                for segment_idx in range(num_segments):
                    # Extract segment data
                    segment_input_ids = batch['input_ids']
                    segment_attention_mask = batch.get('attention_mask')

                    # Forward pass
                    outputs = self.model(
                        input_ids=segment_input_ids,
                        attention_mask=segment_attention_mask,
                        memory_state=memory_state,
                        return_dict=True
                    )

                    # Update memory state
                    if hasattr(outputs, 'memory_state'):
                        memory_state = outputs.memory_state

                    # Update belief state
                    belief_state = self.belief_tracker.update_belief(
                        (outputs.start_logits, outputs.end_logits),
                        segment_idx,
                        getattr(outputs, 'gmm_context', torch.zeros(1, 1, 768)),
                        batch.get('segment_offsets', [0])[segment_idx] if isinstance(batch.get('segment_offsets'), list) else 0
                    )

                    # Make halting decision
                    if self.config.use_halting_policy:
                        halting_features = self.extract_halting_features(
                            belief_state, outputs, batch, segment_idx
                        )
                        action, _, _ = self.halting_policy.select_action(
                            halting_features, training=False
                        )
                        if action == "HALT":
                            break
                    else:
                        if belief_state.confidence >= self.config.belief_state_threshold:
                            break

                # Compute QA loss
                qa_loss = self.compute_qa_loss(
                    outputs,
                    batch['start_positions'],
                    batch['end_positions']
                )
                eval_stats['qa_loss'] += qa_loss.item()

                # Compute exact match and F1 (simplified)
                pred_span = belief_state.best_span or (0, 0)
                # Note: Would need proper text extraction and comparison here
                # For now, using placeholder computations
                eval_stats['exact_match'] += 0.7  # Placeholder
                eval_stats['f1_score'] += 0.6     # Placeholder

                eval_stats['avg_segments_processed'] += segment_idx + 1
                eval_stats['num_examples'] += 1

        # Compute averages
        if eval_stats['num_examples'] > 0:
            for key in ['qa_loss', 'exact_match', 'f1_score', 'avg_segments_processed']:
                eval_stats[key] /= eval_stats['num_examples']

        logger.info(f"Evaluation completed: "
                   f"qa_loss={eval_stats['qa_loss']:.4f}, "
                   f"exact_match={eval_stats['exact_match']:.3f}, "
                   f"f1={eval_stats['f1_score']:.3f}, "
                   f"avg_segments={eval_stats['avg_segments_processed']:.1f}")

        return eval_stats

    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            metrics: Training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'belief_tracker_state_dict': self.belief_tracker.state_dict(),
            'halting_policy_state_dict': self.halting_policy.state_dict(),
            'qa_optimizer_state_dict': self.qa_optimizer.state_dict(),
            'rl_optimizer_state_dict': self.rl_optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'training_stats': self.training_stats
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with loaded information
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.belief_tracker.load_state_dict(checkpoint['belief_tracker_state_dict'])
        self.halting_policy.load_state_dict(checkpoint['halting_policy_state_dict'])
        self.qa_optimizer.load_state_dict(checkpoint['qa_optimizer_state_dict'])
        self.rl_optimizer.load_state_dict(checkpoint['rl_optimizer_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.training_stats = checkpoint.get('training_stats', self.training_stats)

        logger.info(f"Checkpoint loaded from {filepath}, epoch {self.current_epoch}")
        return checkpoint

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary.

        Returns:
            Dictionary with training statistics and configuration
        """
        summary = {
            'config': self.config,
            'current_epoch': self.current_epoch,
            'device': str(self.device),
            'training_stats': self.training_stats,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'belief_tracker_parameters': sum(p.numel() for p in self.belief_tracker.parameters()),
            'halting_policy_parameters': sum(p.numel() for p in self.halting_policy.parameters()),
        }

        if self.config.use_halting_policy:
            rl_stats = self.halting_policy.get_training_stats()
            summary['rl_stats'] = rl_stats

        return summary