"""
Hybrid Training Pipeline for RBS-QA

This module implements the RBSHybridTrainer class that combines
supervised QA learning with reinforcement learning for computational efficiency.
"""

import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..configs.hybrid_training_config import RBSTrainingConfig
from ..models.rbs_xlnet import RBSXLNetForQA


class CheckpointManager:
    """Manages checkpoint saving and cleanup."""

    def __init__(self, save_dir: str, save_frequency: int, keep_best: int = 3):
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        self.keep_best = keep_best

    def save_checkpoint(self, checkpoint_data: Dict, name: str) -> str:
        """Save checkpoint and manage cleanup."""
        checkpoint_path = os.path.join(self.save_dir, f"{name}.pt")
        torch.save(checkpoint_data, checkpoint_path)

        # Cleanup old checkpoints
        if name.startswith("epoch-"):
            self._cleanup_old_checkpoints()

        return checkpoint_path

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old epoch checkpoints to save space."""
        checkpoints = []
        for filename in os.listdir(self.save_dir):
            if filename.startswith("epoch-") and filename.endswith(".pt"):
                epoch_num = int(filename.replace("epoch-", "").replace(".pt", ""))
                checkpoints.append((epoch_num, filename))

        # Sort by epoch number
        checkpoints.sort(key=lambda x: x[0])

        # Keep only the most recent ones
        if len(checkpoints) > self.keep_best:
            for epoch_num, filename in checkpoints[:-self.keep_best]:
                os.remove(os.path.join(self.save_dir, filename))


class RBSHybridTrainer:
    """
    Hybrid trainer for RBS-QA combining supervised QA learning with RL halting policy training.

    Stage 1: Pure supervised pre-training of GMM backbone and belief state tracker
    Stage 2: Hybrid fine-tuning with combined SL (stability) + RL (efficiency) losses
    """

    def __init__(self,
                 model: RBSXLNetForQA,
                 config: RBSTrainingConfig,
                 train_dataset: Dataset,
                 eval_dataset: Optional[Dataset] = None,
                 resume_from_checkpoint: Optional[str] = None):

        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Initialize data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.eval_loader = self._create_dataloader(eval_dataset, shuffle=False) if eval_dataset else None

        # Initialize optimizers
        self._setup_optimizers()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_stage = "supervised"  # "supervised" or "hybrid"
        self.best_eval_score = 0.0
        self.training_history = defaultdict(list)

        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.config.output_dir,
            save_frequency=self.config.save_frequency,
            keep_best=self.config.keep_best_checkpoints
        )

        # Logging
        self.logger = self._setup_logger()
        self.wandb_enabled = self._setup_wandb()

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

    def _setup_optimizers(self) -> None:
        """Setup separate optimizers for different components."""

        # QA optimizer (GMM backbone + belief state tracker)
        qa_params = []
        qa_params.extend(self.model.gmm_backbone.parameters())
        if hasattr(self.model, 'belief_tracker') and self.model.belief_tracker:
            qa_params.extend(self.model.belief_tracker.parameters())

        self.qa_optimizer = torch.optim.AdamW(
            qa_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_epsilon
        )

        # RL optimizer (halting policy network only)
        if hasattr(self.model, 'halting_policy') and self.model.halting_policy and self.config.use_rl_training:
            self.rl_optimizer = torch.optim.Adam(
                self.model.halting_policy.parameters(),
                lr=self.config.rl_learning_rate,
                weight_decay=self.config.rl_weight_decay
            )

        # Learning rate schedulers
        self.qa_scheduler = self._create_scheduler(self.qa_optimizer, "qa")
        if hasattr(self, 'rl_optimizer'):
            self.rl_scheduler = self._create_scheduler(self.rl_optimizer, "rl")

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, optimizer_type: str) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        # Simple linear warmup with cosine decay
        if optimizer_type == "qa":
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = max(100, self.config.warmup_steps // 4)  # Less warmup for RL

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            # Cosine decay
            progress = (current_step - warmup_steps) / max(1, self.config.max_steps - warmup_steps) if self.config.max_steps > 0 else 0
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Create data loader with proper collation for RBS training."""

        def rbs_collate_fn(batch):
            """Custom collate function for RBS batch processing."""

            # Standard QA fields
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])

            # Segment information for adaptive processing
            if 'segment_ids' in batch[0]:
                segment_ids = torch.stack([item['segment_ids'] for item in batch])
                segment_offsets = torch.stack([item['segment_offsets'] for item in batch])
                num_segments = torch.stack([item['num_segments'] for item in batch])
            else:
                # Legacy single-segment format
                segment_ids = input_ids.unsqueeze(1)  # Add segment dimension
                segment_offsets = torch.zeros(input_ids.size(0), 1, dtype=torch.long)
                num_segments = torch.ones(input_ids.size(0), dtype=torch.long)

            # Answer positions
            start_positions = torch.stack([item['start_positions'] for item in batch])
            end_positions = torch.stack([item['end_positions'] for item in batch])

            # Global positions for multi-segment documents
            if 'global_start_positions' in batch[0]:
                global_start_positions = torch.stack([item['global_start_positions'] for item in batch])
                global_end_positions = torch.stack([item['global_end_positions'] for item in batch])
            else:
                # Single-segment fallback
                global_start_positions = start_positions
                global_end_positions = end_positions

            # Question and context separation (for adaptive inference)
            question_input_ids = []
            context_segments = []

            for item in batch:
                if 'question_input_ids' in item:
                    question_input_ids.append(item['question_input_ids'])
                    context_segments.append(item['context_segments'])
                else:
                    # Split input_ids into question + context if not pre-separated
                    sep_pos = item.get('question_length', item['input_ids'].size(0) // 2)
                    question_input_ids.append(item['input_ids'][:sep_pos])
                    context_segments.append([item['input_ids'][sep_pos:]])

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'segment_ids': segment_ids,
                'segment_offsets': segment_offsets,
                'num_segments': num_segments,
                'start_positions': start_positions,
                'end_positions': end_positions,
                'global_start_positions': global_start_positions,
                'global_end_positions': global_end_positions,
                'question_input_ids': question_input_ids,
                'context_segments': context_segments
            }

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=rbs_collate_fn,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory
        )

    def train(self) -> Dict[str, Any]:
        """Main training loop with automatic stage transitions."""

        self.logger.info("Starting RBS-QA hybrid training...")
        self.logger.info(f"Configuration: {json.dumps(self.config.to_dict(), indent=2)}")

        total_training_time = 0
        training_start_time = time.time()

        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                epoch_start_time = time.time()
                self.current_epoch = epoch

                # Check for stage transition
                self._check_stage_transition()

                # Train one epoch
                epoch_metrics = self.train_epoch()
                epoch_time = time.time() - epoch_start_time
                total_training_time += epoch_time

                # Update learning rate schedulers
                self.qa_scheduler.step()
                if hasattr(self, 'rl_scheduler'):
                    self.rl_scheduler.step()

                # Evaluation
                if self.eval_loader and (epoch + 1) % self.config.eval_frequency == 0:
                    eval_metrics = self.evaluate()
                    self._log_eval_metrics(epoch, eval_metrics)

                    # Check for best model
                    current_score = eval_metrics.get('combined_score', eval_metrics.get('f1', 0.0))
                    if current_score > self.best_eval_score:
                        self.best_eval_score = current_score
                        self.save_checkpoint("best-model")
                        self.logger.info(f"New best model: {current_score:.4f}")

                # Log epoch metrics
                self._log_epoch_metrics(epoch, epoch_metrics, eval_metrics if self.eval_loader else None)

                # Save checkpoint
                if (epoch + 1) % self.config.save_frequency == 0:
                    self.save_checkpoint(f"epoch-{epoch + 1}")

                # Early stopping
                if self._should_early_stop():
                    self.logger.info("Early stopping triggered")
                    break

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.save_checkpoint("interrupted")
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            self.save_checkpoint("failed")
            raise

        total_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")

        return {
            'total_time': total_time,
            'best_score': self.best_eval_score,
            'final_epoch': self.current_epoch,
            'training_history': dict(self.training_history)
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch with appropriate loss computation."""

        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0

        # RL episode collection (only in hybrid stage)
        rl_episodes = []
        rl_ground_truths = []

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")):

            if self.training_stage == "supervised":
                batch_metrics = self._supervised_training_step(batch)
            else:
                batch_metrics, episode_data = self._hybrid_training_step(batch)

                if episode_data:
                    rl_episodes.extend(episode_data['episodes'])
                    rl_ground_truths.extend(episode_data['ground_truths'])

            # Accumulate metrics
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value

            num_batches += 1
            self.global_step += 1

            # Log batch metrics (less frequently)
            if self.global_step % self.config.logging_steps == 0:
                self._log_batch_metrics(batch_metrics, batch_idx)

        # Process collected RL episodes
        if self.training_stage == "hybrid" and rl_episodes:
            rl_metrics = self._process_rl_episodes(rl_episodes, rl_ground_truths)
            for key, value in rl_metrics.items():
                epoch_metrics[key] += value

        # Average metrics
        return {key: value / max(num_batches, 1) for key, value in epoch_metrics.items()}

    def _supervised_training_step(self, batch: Dict) -> Dict[str, float]:
        """Supervised training step (Stage 1)."""

        self.qa_optimizer.zero_grad()

        # Forward pass through all segments (full document processing)
        total_qa_loss = 0.0
        memory_state = None

        num_segments = batch['num_segments'][0].item()

        for segment_idx in range(num_segments):
            # Extract segment data
            segment_input_ids = batch['input_ids'][:, segment_idx]
            segment_attention_mask = batch['attention_mask'][:, segment_idx]

            # Forward pass (legacy mode)
            outputs = self.model.forward(
                input_ids=segment_input_ids,
                attention_mask=segment_attention_mask,
                memory_state=memory_state,
                return_dict=True
            )

            # Compute QA loss for this segment
            qa_loss = self._compute_qa_loss(
                outputs.start_logits,
                outputs.end_logits,
                batch['start_positions'][:, segment_idx],
                batch['end_positions'][:, segment_idx]
            )

            total_qa_loss += qa_loss
            memory_state = outputs.memory_state

        # Average loss across segments
        avg_qa_loss = total_qa_loss / num_segments

        # Backward pass
        avg_qa_loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            qa_params = list(self.model.gmm_backbone.parameters())
            if hasattr(self.model, 'belief_tracker') and self.model.belief_tracker:
                qa_params.extend(self.model.belief_tracker.parameters())
            torch.nn.utils.clip_grad_norm_(qa_params, self.config.max_grad_norm)

        self.qa_optimizer.step()

        return {'qa_loss': avg_qa_loss.item()}

    def _hybrid_training_step(self, batch: Dict) -> Tuple[Dict[str, float], Optional[Dict]]:
        """Hybrid training step combining SL and RL (Stage 2)."""

        # QA loss computation (stability component)
        self.qa_optimizer.zero_grad()

        total_qa_loss = 0.0
        memory_state = None

        # Process all segments for QA loss (full document)
        num_segments = batch['num_segments'][0].item()

        for segment_idx in range(num_segments):
            segment_input_ids = batch['input_ids'][:, segment_idx]
            segment_attention_mask = batch['attention_mask'][:, segment_idx]

            outputs = self.model.forward(
                input_ids=segment_input_ids,
                attention_mask=segment_attention_mask,
                memory_state=memory_state,
                return_dict=True
            )

            qa_loss = self._compute_qa_loss(
                outputs.start_logits,
                outputs.end_logits,
                batch['start_positions'][:, segment_idx],
                batch['end_positions'][:, segment_idx]
            )

            total_qa_loss += qa_loss
            memory_state = outputs.memory_state

        avg_qa_loss = total_qa_loss / num_segments

        # Backward pass for QA loss
        avg_qa_loss.backward(retain_graph=True)

        # RL episode collection
        episode_data = self._collect_rl_episode(batch)

        # Update QA parameters
        if self.config.max_grad_norm > 0:
            qa_params = list(self.model.gmm_backbone.parameters())
            if hasattr(self.model, 'belief_tracker') and self.model.belief_tracker:
                qa_params.extend(self.model.belief_tracker.parameters())
            torch.nn.utils.clip_grad_norm_(qa_params, self.config.max_grad_norm)

        self.qa_optimizer.step()

        return {'qa_loss': avg_qa_loss.item()}, episode_data

    def _collect_rl_episode(self, batch: Dict) -> Optional[Dict]:
        """Collect RL episodes by processing documents adaptively."""

        episodes = []
        ground_truths = []

        for batch_idx in range(batch['input_ids'].size(0)):
            # Reset belief state for new document
            if hasattr(self.model, 'belief_tracker') and self.model.belief_tracker:
                self.model.belief_tracker.reset_belief()
            memory_state = None

            episode_steps = []
            document_segments = batch['context_segments'][batch_idx]
            question_ids = batch['question_input_ids'][batch_idx]

            # Process segments adaptively
            for segment_idx, segment_ids in enumerate(document_segments):
                # Combine question and current segment
                input_ids = torch.cat([question_ids.unsqueeze(0), segment_ids.unsqueeze(0)], dim=-1)
                attention_mask = torch.ones_like(input_ids)

                segment_info = {
                    'segment_id': segment_idx,
                    'global_offset': sum(s.size(-1) for s in document_segments[:segment_idx]),
                    'total_segments': len(document_segments)
                }

                # Forward pass in RBS mode
                self.model.eval()  # Set to eval for episode collection
                with torch.no_grad():
                    outputs = self.model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        memory_state=memory_state,
                        segment_info=segment_info,
                        return_dict=True
                    )
                self.model.train()

                # Store episode step
                if hasattr(outputs, 'halting_decision') and outputs.halting_decision:
                    episode_steps.append({
                        'features': outputs.halting_decision.features,
                        'action': outputs.halting_decision.action,
                        'log_prob': outputs.halting_decision.log_prob,
                        'value_estimate': outputs.halting_policy.value_estimate if hasattr(outputs.halting_policy, 'value_estimate') else 0.0,
                        'predicted_span': outputs.belief_state.best_span if outputs.belief_state else (0, 0)
                    })

                    # Check halting decision
                    if outputs.halting_decision.action == "HALT":
                        break

                memory_state = outputs.memory_state

            # Store episode and ground truth
            if episode_steps:
                episodes.append(episode_steps)
                ground_truths.append((
                    batch['global_start_positions'][batch_idx].item(),
                    batch['global_end_positions'][batch_idx].item()
                ))

        if episodes:
            return {'episodes': episodes, 'ground_truths': ground_truths}
        else:
            return None

    def _process_rl_episodes(self,
                           episodes: List[List[Dict]],
                           ground_truths: List[Tuple[int, int]]) -> Dict[str, float]:
        """Process collected RL episodes and update halting policy."""

        if not episodes or not hasattr(self.model, 'halting_policy') or not self.model.halting_policy:
            return {'rl_loss': 0.0}

        self.rl_optimizer.zero_grad()

        # Compute rewards
        rewards = self.model.halting_policy.compute_rewards(
            episodes, ground_truths, self.config.lambda_cost
        )

        # Compute policy gradient loss
        policy_loss = self.model.halting_policy.policy_gradient_loss(
            episodes, rewards, gamma=self.config.gamma
        )

        # Optional value loss
        total_rl_loss = policy_loss
        if self.config.use_value_baseline:
            value_loss = self.model.halting_policy.value_loss(
                episodes, rewards, gamma=self.config.gamma
            )
            total_rl_loss = policy_loss + self.config.value_weight * value_loss

        # Backward pass
        total_rl_loss.backward()

        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.halting_policy.parameters(),
                self.config.max_grad_norm
            )

        self.rl_optimizer.step()

        return {
            'rl_loss': total_rl_loss.item(),
            'avg_reward': np.mean([np.mean(episode_rewards) for episode_rewards in rewards]),
            'avg_episode_length': np.mean([len(episode) for episode in episodes])
        }

    def _compute_qa_loss(self,
                        start_logits: torch.Tensor,
                        end_logits: torch.Tensor,
                        start_positions: torch.Tensor,
                        end_positions: torch.Tensor) -> torch.Tensor:
        """Compute standard QA loss with cross-entropy."""

        # Ignore indices where position is -1 (padding/invalid)
        start_mask = start_positions != -1
        end_mask = end_positions != -1

        if start_mask.sum() == 0 or end_mask.sum() == 0:
            return torch.tensor(0.0, device=start_logits.device, requires_grad=True)

        start_loss = F.cross_entropy(
            start_logits[start_mask],
            start_positions[start_mask]
        )
        end_loss = F.cross_entropy(
            end_logits[end_mask],
            end_positions[end_mask]
        )

        return (start_loss + end_loss) / 2

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model with comprehensive metrics."""

        self.model.eval()
        if hasattr(self.model, 'set_inference_mode'):
            self.model.set_inference_mode("adaptive")

        eval_metrics = defaultdict(float)
        num_examples = 0

        all_predictions = []
        all_ground_truths = []
        efficiency_scores = []

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):

                # Adaptive inference for each example
                for batch_idx in range(batch['input_ids'].size(0)):
                    question_ids = batch['question_input_ids'][batch_idx]
                    context_segments = batch['context_segments'][batch_idx]

                    result = self.model.adaptive_inference(
                        question_input_ids=question_ids,
                        context_segments=context_segments
                    )

                    # Ground truth
                    gt_span = (
                        batch['global_start_positions'][batch_idx].item(),
                        batch['global_end_positions'][batch_idx].item()
                    )

                    # Compute metrics
                    f1 = self.model.halting_policy.compute_f1_score(
                        result.answer_span, gt_span
                    ) if hasattr(self.model.halting_policy, 'compute_f1_score') else 0.0
                    exact_match = (result.answer_span[0] == gt_span[0] and
                                 result.answer_span[1] == gt_span[1])

                    eval_metrics['f1'] += f1
                    eval_metrics['exact_match'] += float(exact_match)
                    eval_metrics['confidence'] += result.confidence
                    efficiency_scores.append(result.efficiency_score)

                    all_predictions.append(result.answer_span)
                    all_ground_truths.append(gt_span)
                    num_examples += 1

        # Average metrics
        for key in eval_metrics:
            eval_metrics[key] /= max(num_examples, 1)

        eval_metrics['efficiency_score'] = np.mean(efficiency_scores)
        eval_metrics['avg_segments_processed'] = np.mean([len(pred) for pred in all_predictions]) if all_predictions else 0

        # Combined score (weighted accuracy + efficiency)
        eval_metrics['combined_score'] = (
            eval_metrics['f1'] * 0.7 + eval_metrics['efficiency_score'] * 0.3
        )

        return dict(eval_metrics)

    def _check_stage_transition(self) -> None:
        """Check and perform stage transition from supervised to hybrid."""

        if (self.training_stage == "supervised" and
            self.current_epoch >= self.config.rl_start_epoch and
            self.config.use_rl_training):

            self.training_stage = "hybrid"
            if hasattr(self.model, 'set_training_mode'):
                self.model.set_training_mode("rl")

            self.logger.info(f"Transitioning to hybrid SL+RL training at epoch {self.current_epoch}")
            self.logger.info(f"QA optimizer LR: {self.qa_scheduler.get_last_lr()[0]:.2e}")
            if hasattr(self, 'rl_scheduler'):
                self.logger.info(f"RL optimizer LR: {self.rl_scheduler.get_last_lr()[0]:.2e}")

    def _should_early_stop(self) -> bool:
        """Check if early stopping should be triggered."""

        if not self.config.early_stopping_patience:
            return False

        # Check if eval F1 hasn't improved in patience epochs
        recent_f1_scores = self.training_history['eval_f1'][-self.config.early_stopping_patience:]

        if len(recent_f1_scores) < self.config.early_stopping_patience:
            return False

        max_recent = max(recent_f1_scores)
        if max_recent <= self.best_eval_score:
            self.logger.info(f"Early stopping: no improvement in {self.config.early_stopping_patience} epochs")
            return True

        return False

    def save_checkpoint(self, checkpoint_name: str) -> None:
        """Save training checkpoint."""

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'qa_optimizer_state_dict': self.qa_optimizer.state_dict(),
            'qa_scheduler_state_dict': self.qa_scheduler.state_dict(),
            'training_stage': self.training_stage,
            'best_eval_score': self.best_eval_score,
            'training_history': dict(self.training_history),
            'config': self.config.to_dict()
        }

        if hasattr(self, 'rl_optimizer'):
            checkpoint['rl_optimizer_state_dict'] = self.rl_optimizer.state_dict()
        if hasattr(self, 'rl_scheduler'):
            checkpoint['rl_scheduler_state_dict'] = self.rl_scheduler.state_dict()

        checkpoint_path = os.path.join(self.config.output_dir, f"{checkpoint_name}.pt")
        torch.save(checkpoint, checkpoint_path)

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""

        self.logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.training_stage = checkpoint['training_stage']
        self.best_eval_score = checkpoint['best_eval_score']
        self.training_history = defaultdict(list, checkpoint['training_history'])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.qa_optimizer.load_state_dict(checkpoint['qa_optimizer_state_dict'])
        self.qa_scheduler.load_state_dict(checkpoint['qa_scheduler_state_dict'])

        if 'rl_optimizer_state_dict' in checkpoint and hasattr(self, 'rl_optimizer'):
            self.rl_optimizer.load_state_dict(checkpoint['rl_optimizer_state_dict'])
        if 'rl_scheduler_state_dict' in checkpoint and hasattr(self, 'rl_scheduler'):
            self.rl_scheduler.load_state_dict(checkpoint['rl_scheduler_state_dict'])

        if hasattr(self.model, 'set_training_mode'):
            self.model.set_training_mode(self.training_stage)

        self.logger.info(f"Resumed from epoch {self.current_epoch}, stage {self.training_stage}")

    def _setup_logger(self) -> logging.Logger:
        """Setup training logger."""

        logger = logging.getLogger("RBSHybridTrainer")
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # File handler
        os.makedirs(self.config.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(self.config.output_dir, "training.log")
        )
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _setup_wandb(self) -> bool:
        """Setup Weights & Biases logging if configured."""

        if not self.config.use_wandb:
            return False

        try:
            import wandb

            wandb.init(
                project=self.config.wandb_project,
                name=self.config.run_name,
                config=self.config.to_dict(),
                dir=self.config.output_dir
            )

            self.logger.info("WandB logging initialized")
            return True

        except ImportError:
            self.logger.warning("WandB not available, skipping wandb logging")
            return False
        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {str(e)}")
            return False

    def _log_epoch_metrics(self,
                          epoch: int,
                          train_metrics: Dict[str, float],
                          eval_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log epoch-level metrics."""

        # Update training history
        for key, value in train_metrics.items():
            self.training_history[f'train_{key}'].append(value)

        if eval_metrics:
            for key, value in eval_metrics.items():
                self.training_history[f'eval_{key}'].append(value)

        # Log to console
        log_str = f"Epoch {epoch}: "
        log_str += ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])

        if eval_metrics:
            log_str += " | Eval: "
            log_str += ", ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items()])

        self.logger.info(log_str)

        # Log to WandB
        if self.wandb_enabled:
            import wandb

            wandb_log_data = {f"train/{k}": v for k, v in train_metrics.items()}

            if eval_metrics:
                wandb_log_data.update({f"eval/{k}": v for k, v in eval_metrics.items()})

            wandb_log_data.update({
                "epoch": epoch,
                "global_step": self.global_step,
                "learning_rate_qa": self.qa_scheduler.get_last_lr()[0],
                "training_stage": self.training_stage
            })

            if hasattr(self, 'rl_scheduler'):
                wandb_log_data["learning_rate_rl"] = self.rl_scheduler.get_last_lr()[0]

            wandb.log(wandb_log_data)

    def _log_batch_metrics(self, metrics: Dict[str, float], batch_idx: int) -> None:
        """Log batch-level metrics (less frequently)."""

        if self.wandb_enabled and self.global_step % self.config.logging_steps == 0:
            import wandb
            wandb.log({
                f"batch/{k}": v for k, v in metrics.items()
            }, step=self.global_step)

    def _log_eval_metrics(self, epoch: int, eval_metrics: Dict[str, float]) -> None:
        """Log evaluation metrics with emphasis."""

        self.logger.info(f"Evaluation Results - Epoch {epoch}:")
        for key, value in eval_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")

        if self.wandb_enabled:
            import wandb
            wandb.log({
                f"eval_final/{k}": v for k, v in eval_metrics.items()
            }, step=self.global_step)