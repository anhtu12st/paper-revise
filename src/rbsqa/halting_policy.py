"""
Halting Policy Network for RBS-QA

Implements reinforcement learning-based halting decisions for adaptive
computation in long-context question answering using REINFORCE.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class HaltingDecision:
    """Halting policy decision."""
    action: str  # "CONTINUE" or "HALT"
    log_prob: torch.Tensor
    value_estimate: torch.Tensor
    features: 'HaltingStateFeatures'


@dataclass
class HaltingStateFeatures:
    """Features for halting policy decision making.

    Encodes all relevant information for deciding whether to CONTINUE
    reading more segments or HALT with current belief.
    """
    # Belief state features
    current_confidence: float              # Best span confidence [0.0, 1.0]
    confidence_trend: List[float]          # Last 3 confidence values
    confidence_variance: float              # Confidence stability measure
    revision_count: int                    # Number of belief revisions

    # Computation features
    segments_processed: int                # How many segments read so far
    segments_remaining: int                # Estimated segments remaining
    processing_time: float                 # Time spent so far

    # GMM context features
    routing_entropy: float                 # Memory routing diversity
    expert_utilization: List[float]        # Expert activation patterns
    context_quality_score: float           # GMM context coherence

    # Document complexity features
    document_length: int                   # Total segments in document
    question_complexity: float             # Question embedding complexity
    segment_relevance_score: float         # Current segment relevance

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert features to tensor for network input.

        Args:
            device: PyTorch device for tensor placement

        Returns:
            Feature tensor of shape (12,)
        """
        features = [
            self.current_confidence,
            np.mean(self.confidence_trend) if self.confidence_trend else 0.0,
            self.confidence_variance,
            float(self.revision_count),
            self.segments_processed / max(self.segments_remaining, 1),
            self.processing_time,
            self.routing_entropy,
            np.mean(self.expert_utilization) if self.expert_utilization else 0.0,
            self.context_quality_score,
            self.segments_processed / max(self.document_length, 1),
            self.question_complexity,
            self.segment_relevance_score
        ]
        return torch.tensor(features, dtype=torch.float32, device=device)

    def validate(self) -> None:
        """Validate feature values."""
        if not (0.0 <= self.current_confidence <= 1.0):
            raise ValueError(f"current_confidence must be in [0.0, 1.0], got {self.current_confidence}")

        if self.segments_processed < 0:
            raise ValueError(f"segments_processed must be >= 0, got {self.segments_processed}")

        if self.segments_remaining < 0:
            raise ValueError(f"segments_remaining must be >= 0, got {self.segments_remaining}")

        if self.revision_count < 0:
            raise ValueError(f"revision_count must be >= 0, got {self.revision_count}")

        if self.document_length <= 0:
            raise ValueError(f"document_length must be > 0, got {self.document_length}")


class HaltingPolicyNetwork(nn.Module):
    """
    Halting Policy Network for adaptive computation using REINFORCE.

    Learns to make optimal CONTINUE/HALT decisions based on belief state
    features, GMM context, and computation progress. Uses policy gradient
    methods with value baseline for stable training.
    """

    def __init__(self,
                 input_dim: int = 12,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 exploration_rate: float = 0.1):
        """Initialize halting policy network.

        Args:
            input_dim: Dimension of input features (default: 12)
            hidden_dim: Hidden dimension for policy/value networks
            num_layers: Number of hidden layers
            dropout: Dropout rate for regularization
            temperature: Temperature for action sampling softmax
            exploration_rate: Epsilon-greedy exploration rate
        """
        super().__init__()

        # Validate parameters
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0], got {dropout}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not (0.0 <= exploration_rate <= 1.0):
            raise ValueError(f"exploration_rate must be in [0.0, 1.0], got {exploration_rate}")

        # Configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temperature = temperature
        self.exploration_rate = exploration_rate

        # Policy network layers
        layers = []
        prev_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))  # CONTINUE, HALT logits
        self.policy_net = nn.Sequential(*layers)

        # Value network for baseline (optional, reduces variance)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Training state
        self.training_episodes: List[List[Dict]] = []
        self.current_episode: List[Dict] = []
        self.device = torch.device('cpu')  # Will be updated on first forward pass

        logger.info(f"Initialized HaltingPolicyNetwork: input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, layers={num_layers}")

    def forward(self, features: HaltingStateFeatures) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of halting policy network.

        Args:
            features: HaltingStateFeatures for current decision point

        Returns:
            (policy_logits, value_estimate) for action selection and baseline
        """
        # Validate features
        features.validate()

        # Update device on first forward pass
        if not hasattr(self, '_device_set'):
            self.device = next(self.parameters()).device
            self._device_set = True

        feature_tensor = features.to_tensor(self.device)

        # Ensure feature tensor has correct shape
        if feature_tensor.shape[-1] != self.input_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.input_dim}, "
                           f"got {feature_tensor.shape[-1]}")

        # Policy logits
        policy_logits = self.policy_net(feature_tensor)
        policy_probs = F.softmax(policy_logits / self.temperature, dim=-1)

        # Value estimate (baseline for REINFORCE)
        value_estimate = self.value_net(feature_tensor).squeeze(-1)

        # Log debugging info
        if logger.isEnabledFor(logging.DEBUG):
            continue_prob = policy_probs[0].item()
            halt_prob = policy_probs[1].item()
            logger.debug(f"Policy forward: continue_prob={continue_prob:.3f}, "
                        f"halt_prob={halt_prob:.3f}, value={value_estimate.item():.3f}")

        return policy_logits, value_estimate

    def select_action(self, features: HaltingStateFeatures, training: bool = True) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Select CONTINUE or HALT action.

        Args:
            features: Current state features
            training: Whether to use exploration

        Returns:
            (action, log_prob, value_estimate)
        """
        policy_logits, value_estimate = self.forward(features)
        policy_probs = F.softmax(policy_logits / self.temperature, dim=-1)

        if training and torch.rand(1).item() < self.exploration_rate:
            # Epsilon-greedy exploration
            action_idx = torch.randint(0, 2, (1,)).item()
            exploration = True
        else:
            # Sample from policy distribution
            action_idx = torch.multinomial(policy_probs, 1).item()
            exploration = False

        action = "HALT" if action_idx == 1 else "CONTINUE"
        log_prob = torch.log(policy_probs[action_idx] + 1e-8)

        # Store for training
        if training:
            self.current_episode.append({
                'features': features,
                'action': action,
                'action_idx': action_idx,
                'log_prob': log_prob,
                'value_estimate': value_estimate,
                'policy_probs': policy_probs.detach(),
                'exploration': exploration,
                'timestamp': time.time()
            })

        # Log action selection
        if logger.isEnabledFor(logging.DEBUG):
            continue_prob = policy_probs[0].item()
            halt_prob = policy_probs[1].item()
            logger.debug(f"Selected action: {action} (idx={action_idx}), "
                        f"probs=[{continue_prob:.3f}, {halt_prob:.3f}], "
                        f"exploration={exploration}")

        return action, log_prob, value_estimate

    def compute_rewards(self,
                       episodes: List[List[Dict]],
                       ground_truth_spans: List[Tuple[int, int]],
                       lambda_cost: float = 0.01) -> List[List[float]]:
        """Compute rewards for each action in episode using F1 score - cost.

        Args:
            episodes: List of episodes with actions and belief states
            ground_truth_spans: True answer spans for each episode
            lambda_cost: Cost per segment processed

        Returns:
            List of reward sequences for each episode
        """
        if len(episodes) != len(ground_truth_spans):
            raise ValueError(f"Episodes and ground truth spans length mismatch: "
                           f"{len(episodes)} vs {len(ground_truth_spans)}")

        all_rewards = []

        for episode_idx, (episode, gt_span) in enumerate(zip(episodes, ground_truth_spans)):
            episode_rewards = []
            segments_processed = len(episode)

            for step_idx, step in enumerate(episode):
                action = step['action']
                belief_confidence = step['features'].current_confidence

                if action == "CONTINUE":
                    # Small negative reward for reading cost
                    reward = -lambda_cost
                else:  # HALT
                    # Final reward: F1 score - total cost
                    predicted_span = step.get('predicted_span', (0, 0))
                    f1_score = self._compute_f1_score(predicted_span, gt_span)
                    total_cost = lambda_cost * segments_processed
                    reward = f1_score - total_cost

                episode_rewards.append(reward)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Episode {episode_idx}, step {step_idx}: "
                                f"action={action}, reward={reward:.4f}, "
                                f"confidence={belief_confidence:.3f}")

            all_rewards.append(episode_rewards)

        return all_rewards

    def _compute_f1_score(self, pred_span: Tuple[int, int], true_span: Tuple[int, int]) -> float:
        """Compute F1 score between predicted and true spans.

        Args:
            pred_span: Predicted (start, end) token indices
            true_span: Ground truth (start, end) token indices

        Returns:
            F1 score in [0.0, 1.0]
        """
        pred_start, pred_end = pred_span
        true_start, true_end = true_span

        # Convert to token sets
        pred_tokens = set(range(pred_start, pred_end + 1))
        true_tokens = set(range(true_start, true_end + 1))

        # Handle edge cases
        if len(pred_tokens) == 0 and len(true_tokens) == 0:
            return 1.0
        elif len(pred_tokens) == 0 or len(true_tokens) == 0:
            return 0.0

        # Compute precision and recall
        intersection = len(pred_tokens & true_tokens)
        precision = intersection / len(pred_tokens)
        recall = intersection / len(true_tokens)

        # Compute F1 score
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def policy_gradient_loss(self,
                           episodes: List[List[Dict]],
                           rewards: List[List[float]],
                           gamma: float = 0.99,
                           use_baseline: bool = True) -> torch.Tensor:
        """Compute REINFORCE policy gradient loss.

        Args:
            episodes: List of episodes with actions and log_probs
            rewards: List of reward sequences for each episode
            gamma: Discount factor for future rewards
            use_baseline: Whether to use value estimates as baseline

        Returns:
            Policy gradient loss tensor
        """
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0.0, 1.0], got {gamma}")

        if len(episodes) != len(rewards):
            raise ValueError(f"Episodes and rewards length mismatch: {len(episodes)} vs {len(rewards)}")

        policy_losses = []

        for episode_idx, (episode, episode_rewards) in enumerate(zip(episodes, rewards)):
            if len(episode) == 0:
                continue

            # Compute discounted returns
            returns = []
            R = 0
            for r in reversed(episode_rewards):
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns, device=self.device, dtype=torch.float32)

            # Compute advantages
            if use_baseline:
                values = torch.stack([step['value_estimate'] for step in episode])
                advantages = returns - values
            else:
                advantages = returns

            # Compute policy loss
            for step, advantage in zip(episode, advantages):
                policy_loss = -step['log_prob'] * advantage.detach()
                policy_losses.append(policy_loss)

            # Log episode statistics
            if logger.isEnabledFor(logging.DEBUG):
                total_return = returns.sum().item()
                mean_advantage = advantages.mean().item()
                logger.debug(f"Episode {episode_idx}: total_return={total_return:.4f}, "
                            f"mean_advantage={mean_advantage:.4f}, "
                            f"steps={len(episode)}")

        if not policy_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = torch.stack(policy_losses).mean()
        # Ensure requires_grad is True
        if not total_loss.requires_grad:
            total_loss.requires_grad_(True)
        return total_loss

    def value_loss(self,
                  episodes: List[List[Dict]],
                  rewards: List[List[float]],
                  gamma: float = 0.99) -> torch.Tensor:
        """Compute value function loss (MSE).

        Args:
            episodes: List of episodes with value estimates
            rewards: List of reward sequences for each episode
            gamma: Discount factor for future rewards

        Returns:
            Value function loss tensor
        """
        if len(episodes) != len(rewards):
            raise ValueError(f"Episodes and rewards length mismatch: {len(episodes)} vs {len(rewards)}")

        value_losses = []

        for episode_idx, (episode, episode_rewards) in enumerate(zip(episodes, rewards)):
            if len(episode) == 0:
                continue

            # Compute discounted returns
            returns = []
            R = 0
            for r in reversed(episode_rewards):
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
            values = torch.stack([step['value_estimate'] for step in episode])

            # Ensure same shape for MSE loss
            if values.dim() > returns.dim():
                values = values.squeeze()
            elif returns.dim() > values.dim():
                returns = returns.squeeze()

            # MSE loss between predicted and actual returns
            mse_loss = F.mse_loss(values, returns)
            value_losses.append(mse_loss)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Episode {episode_idx}: value_loss={mse_loss.item():.4f}, "
                            f"steps={len(episode)}")

        if not value_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = torch.stack(value_losses).mean()
        # Ensure requires_grad is True
        if not total_loss.requires_grad:
            total_loss.requires_grad_(True)
        return total_loss

    def end_episode(self) -> Dict:
        """Mark end of current episode and return episode data.

        Returns:
            Dictionary containing episode summary
        """
        episode_data = {
            'steps': self.current_episode.copy(),
            'length': len(self.current_episode),
            'actions': [step['action'] for step in self.current_episode],
            'exploration_steps': sum(1 for step in self.current_episode if step.get('exploration', False))
        }

        # Add to training history
        self.training_episodes.append(self.current_episode.copy())
        self.current_episode = []

        # Log episode summary
        if logger.isEnabledFor(logging.DEBUG):
            continue_count = episode_data['actions'].count('CONTINUE')
            halt_count = episode_data['actions'].count('HALT')
            logger.debug(f"Episode ended: {episode_data['length']} steps, "
                        f"{continue_count} CONTINUE, {halt_count} HALT, "
                        f"{episode_data['exploration_steps']} exploration")

        return episode_data

    def reset_training_state(self) -> None:
        """Reset training state for new training session."""
        self.training_episodes = []
        self.current_episode = []
        logger.debug("Halting policy training state reset")

    def get_training_stats(self) -> Dict:
        """Get training statistics.

        Returns:
            Dictionary with training statistics
        """
        if not self.training_episodes:
            return {
                'total_episodes': 0,
                'avg_episode_length': 0.0,
                'continue_rate': 0.0,
                'halt_rate': 0.0,
                'exploration_rate': 0.0
            }

        total_episodes = len(self.training_episodes)
        total_steps = sum(len(ep) for ep in self.training_episodes)
        all_actions = [step for episode in self.training_episodes for step in episode]

        continue_count = sum(1 for step in all_actions if step['action'] == 'CONTINUE')
        halt_count = sum(1 for step in all_actions if step['action'] == 'HALT')
        exploration_count = sum(1 for step in all_actions if step.get('exploration', False))

        return {
            'total_episodes': total_episodes,
            'total_steps': total_steps,
            'avg_episode_length': total_steps / total_episodes if total_episodes > 0 else 0.0,
            'continue_rate': continue_count / len(all_actions) if all_actions else 0.0,
            'halt_rate': halt_count / len(all_actions) if all_actions else 0.0,
            'exploration_rate': exploration_count / len(all_actions) if all_actions else 0.0,
            'episodes_in_memory': total_episodes
        }

    def set_exploration_rate(self, new_rate: float) -> None:
        """Update exploration rate.

        Args:
            new_rate: New exploration rate in [0.0, 1.0]
        """
        if not (0.0 <= new_rate <= 1.0):
            raise ValueError(f"exploration_rate must be in [0.0, 1.0], got {new_rate}")

        old_rate = self.exploration_rate
        self.exploration_rate = new_rate
        logger.debug(f"Exploration rate updated: {old_rate:.3f} -> {new_rate:.3f}")

    def set_temperature(self, new_temperature: float) -> None:
        """Update action sampling temperature.

        Args:
            new_temperature: New temperature > 0
        """
        if new_temperature <= 0:
            raise ValueError(f"temperature must be positive, got {new_temperature}")

        old_temp = self.temperature
        self.temperature = new_temperature
        logger.debug(f"Temperature updated: {old_temp:.3f} -> {new_temperature:.3f}")