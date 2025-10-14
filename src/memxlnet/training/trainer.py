import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import XLNetForQuestionAnsweringSimple, XLNetTokenizerFast, get_linear_schedule_with_warmup

# Optional import for experiment tracking
try:
    import wandb  # type: ignore

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

from memxlnet.data.dataset import (
    TimeStepMajorDataLoader,
    configure_memory_tokens,
    create_dataloader,
    create_dataset_from_cache,
    process_and_cache_dataset,
)
from memxlnet.data.text_utils import normalize_answer_for_comparison
from memxlnet.models.memxlnet_qa import MemXLNetForQA

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer_for_comparison(prediction).split()
    ground_truth_tokens = normalize_answer_for_comparison(ground_truth).split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Calculate exact match score between prediction and ground truth."""
    return normalize_answer_for_comparison(prediction) == normalize_answer_for_comparison(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Calculate maximum metric over all ground truths."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def find_best_threshold(predictions_with_scores, references, thresholds=None):
    """
    Find the best no-answer threshold by trying multiple values.

    Args:
        predictions_with_scores: Dict mapping qid to (prediction, score_diff)
        references: Dict with question_ids as keys and reference data as values
        thresholds: List of thresholds to try (default: -3.0 to 3.0 in steps of 0.1)

    Returns:
        Dict with best_threshold, best_f1, and best_exact
    """
    if thresholds is None:
        thresholds = [i * 0.1 for i in range(-30, 31)]  # -3.0 to 3.0 in steps of 0.1

    best_f1 = 0.0
    best_exact = 0.0
    best_f1_threshold = 0.0
    best_exact_threshold = 0.0

    for threshold in thresholds:
        # Apply threshold to predictions
        thresholded_predictions = {}
        for qid, (pred, score_diff) in predictions_with_scores.items():
            # If score_diff < threshold, predict no-answer
            if score_diff < threshold:
                thresholded_predictions[qid] = ""
            else:
                thresholded_predictions[qid] = pred

        # Evaluate with this threshold
        metrics = evaluate_squad_v2(thresholded_predictions, references)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_f1_threshold = threshold

        if metrics["exact"] > best_exact:
            best_exact = metrics["exact"]
            best_exact_threshold = threshold

    return {
        "best_f1": best_f1,
        "best_f1_threshold": best_f1_threshold,
        "best_exact": best_exact,
        "best_exact_threshold": best_exact_threshold,
    }


def evaluate_squad_v2(predictions, references, no_answer_threshold=0.0):
    """
    Evaluate SQuAD v2 predictions with official metrics.

    Args:
        predictions: Dict with question_ids as keys and predictions as values
        references: Dict with question_ids as keys and reference data as values
        no_answer_threshold: Threshold for no-answer predictions (not used, kept for compatibility)

    Returns:
        Dict with evaluation metrics
    """
    f1_sum = 0.0
    em_sum = 0.0
    has_ans_f1_sum = 0.0
    has_ans_em_sum = 0.0
    no_ans_f1_sum = 0.0
    no_ans_em_sum = 0.0
    has_ans_count = 0
    no_ans_count = 0

    for qid, prediction in predictions.items():
        if qid not in references:
            continue

        reference = references[qid]
        ground_truths = reference["answers"]

        # Check if question has answer
        has_answer = len(ground_truths) > 0 and ground_truths[0] != ""

        if has_answer:
            has_ans_count += 1
            # Calculate F1 and EM for questions with answers
            f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
            em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
            has_ans_f1_sum += f1
            has_ans_em_sum += em
        else:
            no_ans_count += 1
            # For no-answer questions, correct prediction is empty string
            f1 = 1.0 if prediction == "" else 0.0
            em = 1.0 if prediction == "" else 0.0
            no_ans_f1_sum += f1
            no_ans_em_sum += em

        f1_sum += f1
        em_sum += em

    total_count = has_ans_count + no_ans_count

    if total_count == 0:
        return {
            "exact": 0.0,
            "f1": 0.0,
            "HasAns_exact": 0.0,
            "HasAns_f1": 0.0,
            "NoAns_exact": 0.0,
            "NoAns_f1": 0.0,
            "HasAns_total": 0,
            "NoAns_total": 0,
        }

    results = {
        "exact": 100.0 * em_sum / total_count,
        "f1": 100.0 * f1_sum / total_count,
        "HasAns_total": has_ans_count,
        "NoAns_total": no_ans_count,
    }

    if has_ans_count > 0:
        results["HasAns_exact"] = 100.0 * has_ans_em_sum / has_ans_count
        results["HasAns_f1"] = 100.0 * has_ans_f1_sum / has_ans_count
    else:
        results["HasAns_exact"] = 0.0
        results["HasAns_f1"] = 0.0

    if no_ans_count > 0:
        results["NoAns_exact"] = 100.0 * no_ans_em_sum / no_ans_count
        results["NoAns_f1"] = 100.0 * no_ans_f1_sum / no_ans_count
    else:
        results["NoAns_exact"] = 0.0
        results["NoAns_f1"] = 0.0

    return results


@dataclass
class TrainingConfig:
    """Training configuration with all hyperparameters and settings."""

    # Model settings
    model_name: str = "xlnet-base-cased"
    max_seq_length: int = 384
    doc_stride: int = 128

    # Dataset settings
    dataset_name: str = "squad_v2"
    train_split: str = "train"
    eval_split: str = "validation"
    cache_dir: str = "./.cache"
    max_train_samples: int | None = None  # Set to limit training samples
    max_eval_samples: int | None = None  # Limit eval for faster validation
    use_lazy_loading: bool = True  # Use lazy loading for large datasets

    # Progressive segment training settings
    max_n_segs: int | None = None  # Maximum number of segments per document
    progressive_segments: list[int] | None = None  # List of segment counts for progressive training

    # Data processing settings
    streaming_chunk_size: int = 1000
    max_memory_gb: float = 8.0
    use_streaming: bool = True

    # Training hyperparameters
    num_epochs: int = 3
    train_batch_size: int = 4  # Number of documents processed in parallel
    eval_batch_size: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Memory and performance
    gradient_accumulation_steps: int = 2
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    # Output settings
    output_dir: str = "./outputs"
    run_name: str = "xlnet-long-qa"
    save_total_limit: int = 3

    # Evaluation settings
    no_answer_threshold: float = 0.0  # Will be tuned during evaluation
    use_any_positive_logic: bool = True  # Use "any positive" logic for multi-segment predictions

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "xlnet-long-qa"

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = torch.cuda.is_available()

    # Memory-augmented model settings
    memory_num_tokens: int = 32  # per read/write set
    memory_update: str = "gated"  # or "none"

    # Enhanced memory settings
    use_differentiable_memory: bool = False  # Enable differentiable memory
    num_memory_heads: int = 1  # Number of read/write heads
    memory_sharpness: float = 1.0  # Attention sharpening factor
    enable_usage_tracking: bool = False  # Track memory slot usage
    enable_temporal_links: bool = False  # Enable temporal link matrix
    memory_slots: int | None = None  # Number of memory slots (default: max(mem_token_count, 16))
    memory_init: str = "learned"  # or "zeros"
    memory_impl: str = "token"  # or "prefix" (phase-2)
    use_global_softmax: bool = True

    # Phase-2 warmup controls
    warmup_freeze_base_epochs: int = 1  # freeze base transformer for first N epochs
    warmup_disable_global_softmax_epochs: int = 1  # disable doc-level softmax initially
    warmup_disable_any_positive_epochs: int = 1  # disable any-positive extraction initially

    # HuggingFace Hub integration - Model uploads
    hub_model_id: str | None = None  # Repository ID on HuggingFace Hub (e.g., "username/model-name")
    push_to_hub_on_save: bool = False  # Automatically push to Hub when saving checkpoints
    hub_private: bool = True  # 🔒 Create PRIVATE repository on Hub (change to False for public)
    hub_token: str | None = None  # HuggingFace token (defaults to HF_TOKEN environment variable)
    hub_strategy: str = "every_save"  # Push strategy: "every_save", "best_only", "end"

    # HuggingFace Hub integration - Preprocessed datasets
    hub_dataset_id: str | None = (
        None  # Repository ID for preprocessed dataset (e.g., "username/memxlnet-squad-processed")
    )
    use_hub_dataset: bool = True  # Try loading preprocessed dataset from Hub first (faster, lower RAM)
    force_reprocess: bool = False  # Force reprocessing even if Hub/cache exists (for debugging)

    # Chunked dataset settings (GitHub-friendly preprocessing)
    use_chunked_dataset: bool = False  # Enable chunked dataset loading (faster startup, 2-5 min vs 30-60 min)
    chunked_dataset_dir: str | None = None  # Directory with chunked data (e.g., "./preprocessed_data/squad_v2")
    chunked_load_mode: str = "streaming"  # Loading mode: "streaming", "first_n", "chunks", "full"
    chunked_num_examples: int | None = None  # Number of examples to load (for "first_n" mode)
    chunked_chunk_indices: list[int] | None = None  # Specific chunks to load (for "chunks" mode)

    # Smart segment selection for progressive training
    segment_selection_strategy: str = "answer_centered"  # Segment selection strategy:
    # - "answer_centered": Place answer segment near middle (default, recommended)
    # - "random_continuous": Random N continuous segments
    # - "first_n": Use first N segments (original behavior, for comparison)
    segment_selection_seed: int = 42  # Random seed for reproducible segment selection (fixed per experiment)

    def __post_init__(self):
        """Post-initialization setup."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)


class XLNetRecurrentTrainer:
    """
    Trainer that implements the correct XLNet recurrent memory approach.

    This trainer processes documents as streams of chunks, maintaining
    memory state across chunks within each document.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Determine model source (local path or Hub)
        model_source = self._resolve_model_source(config.model_name)

        # Initialize model and tokenizer
        logger.info(f"🚀 Loading XLNet model: {model_source}")
        self.base_model = XLNetForQuestionAnsweringSimple.from_pretrained(model_source)
        self.tokenizer = XLNetTokenizerFast.from_pretrained(model_source)

        # Configure memory tokens if enabled and resize embeddings
        self.mem_token_info = None
        if self.config.memory_num_tokens and self.config.memory_num_tokens > 0:
            try:
                self.mem_token_info = configure_memory_tokens(self.tokenizer, self.config.memory_num_tokens)
                # Resize model embeddings only if tokens were added
                if self.mem_token_info:
                    self.base_model.resize_token_embeddings(len(self.tokenizer))
                    logger.info(
                        f"🔢 Added memory tokens: R={len(self.mem_token_info['mem_read_ids'])}, W={len(self.mem_token_info['mem_write_ids'])}"
                    )
            except Exception as e:
                logger.warning(f"Failed to configure memory tokens: {e}")

        # Initialize model (Memory-Augmented XLNet)
        logger.info("🏗️ Initializing MA-XLNet architecture")

        # Use MemXLNetForQA wrapper when ANY memory features are enabled:
        # - Token-based memory (memory_impl == "token" with memory_num_tokens > 0)
        # - Differentiable memory (use_differentiable_memory == True)
        should_use_wrapper = (
            self.config.memory_num_tokens and self.config.memory_num_tokens > 0 and self.config.memory_impl == "token"
        ) or self.config.use_differentiable_memory

        if should_use_wrapper:
            # If loading from a checkpoint that already has MemXLNet state, load via wrapper
            try:
                self.model = MemXLNetForQA.from_pretrained(
                    model_source,
                    mem_token_count=self.config.memory_num_tokens,
                    memory_init=self.config.memory_init,
                    memory_update=self.config.memory_update,
                    # Enhanced memory parameters
                    use_differentiable_memory=self.config.use_differentiable_memory,
                    num_memory_heads=self.config.num_memory_heads,
                    memory_sharpness=self.config.memory_sharpness,
                    enable_usage_tracking=self.config.enable_usage_tracking,
                    enable_temporal_links=self.config.enable_temporal_links,
                    memory_slots=self.config.memory_slots,
                )
            except Exception:
                # Fallback: wrap the freshly loaded base model
                self.model = MemXLNetForQA(
                    base_model=self.base_model,
                    mem_token_count=self.config.memory_num_tokens,
                    memory_init=self.config.memory_init,
                    memory_update=self.config.memory_update,
                    # Enhanced memory parameters
                    use_differentiable_memory=self.config.use_differentiable_memory,
                    num_memory_heads=self.config.num_memory_heads,
                    memory_sharpness=self.config.memory_sharpness,
                    enable_usage_tracking=self.config.enable_usage_tracking,
                    enable_temporal_links=self.config.enable_temporal_links,
                    memory_slots=self.config.memory_slots,
                )
            # IMPORTANT: ensure base embeddings match tokenizer (memory tokens added)
            try:
                self.model.base.resize_token_embeddings(len(self.tokenizer))
            except Exception as e:
                logger.warning(f"Could not resize embeddings after wrapping: {e}")
        else:
            # No memory enabled, use base model
            self.model = self.base_model

        self.model.to(self.device)

        # Initialize optimizer and scheduler (will be set in prepare_training)
        self.optimizer: AdamW | None = None
        self.scheduler: Any | None = None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_score = 0.0

        # Setup experiment tracking
        if config.use_wandb:
            if HAS_WANDB:
                wandb.init(project=config.wandb_project, name=config.run_name, config=config.__dict__)
            else:
                logger.warning("⚠️ wandb not installed. Install with: pip install wandb")
                config.use_wandb = False

        # Log memory-related settings
        logger.info(
            f"🧠 Memory settings | tokens={self.config.memory_num_tokens}, update={self.config.memory_update}, "
            f"impl={self.config.memory_impl}, global_softmax={self.config.use_global_softmax}"
        )

        # Write minimal run metadata for reproducibility
        try:
            self._write_run_metadata()
        except Exception as e:
            logger.debug(f"Could not write run metadata: {e}")

        # Epoch-scoped toggles (set each epoch in _train_single_stage)
        self._use_global_softmax_epoch = self.config.use_global_softmax
        self._use_any_positive_epoch = self.config.use_any_positive_logic

    def _resolve_model_source(self, model_name: str) -> str:
        """Resolve model source, preferring Hub if configured and exists.

        Args:
            model_name: Original model name/path from config

        Returns:
            Model source to use (local path or Hub ID)
        """
        # If hub_model_id is configured and model_name is not a local path, try Hub first
        if self.config.hub_model_id:
            # Check if model_name is a local path that exists
            is_local_path = os.path.exists(model_name) or os.path.isabs(model_name)

            if not is_local_path:
                # Check if Hub model actually exists
                try:
                    from huggingface_hub import repo_exists

                    logger.info(f"🔍 Checking HuggingFace Hub for model: {self.config.hub_model_id}")

                    # Check if repository exists on Hub
                    if repo_exists(self.config.hub_model_id, repo_type="model", token=self.config.hub_token):
                        logger.info(f"✅ Found existing model on Hub: {self.config.hub_model_id}")
                        logger.info("📥 Will resume training from checkpoint")
                        return self.config.hub_model_id
                    else:
                        logger.info(f"⚠️  Model not found on Hub: {self.config.hub_model_id}")
                        logger.info(f"📁 Using base model: {model_name}")
                        logger.info("🆕 Will train from scratch and push to Hub when ready")

                except Exception as e:
                    logger.warning(f"⚠️  Could not check Hub model {self.config.hub_model_id}: {e}")
                    logger.info(f"📁 Falling back to base model: {model_name}")

        return model_name

    def _freeze_base_transformer(self, freeze: bool = True):
        """Freeze or unfreeze the base transformer weights."""
        model = self.model
        try:
            if hasattr(model, "base") and hasattr(model.base, "transformer"):
                for p in model.base.transformer.parameters():
                    p.requires_grad = not freeze
            elif hasattr(model, "transformer"):
                for p in model.transformer.parameters():
                    p.requires_grad = not freeze
            logger.info(f"🧊 Base transformer {'frozen' if freeze else 'unfrozen'}")
        except Exception as e:
            logger.warning(f"Could not {'freeze' if freeze else 'unfreeze'} base transformer: {e}")

    def _write_run_metadata(self):
        """Write a small metadata file capturing key run parameters."""
        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "run_name": self.config.run_name,
            "model_name": self.config.model_name,
            "dataset_name": self.config.dataset_name,
            "max_seq_length": self.config.max_seq_length,
            "doc_stride": self.config.doc_stride,
            "progressive_segments": self.config.progressive_segments,
            "memory": {
                "memory_num_tokens": self.config.memory_num_tokens,
                "memory_update": self.config.memory_update,
                "memory_init": self.config.memory_init,
                "memory_impl": self.config.memory_impl,
                "use_global_softmax": self.config.use_global_softmax,
            },
            "device": self.config.device,
            "fp16": self.config.fp16,
        }
        out = os.path.join(self.config.output_dir, "run_metadata.json")
        with open(out, "w") as f:
            json.dump(meta, f, indent=2)

    def prepare_data(
        self,
        override_max_segments: int | None = None,
    ) -> tuple[DataLoader[Any] | TimeStepMajorDataLoader, DataLoader[Any] | TimeStepMajorDataLoader, Any]:
        """Prepare training and evaluation datasets with memory-efficient loading.

        Args:
            override_max_segments: Override max_segments for progressive training stages.
                If None, uses all segments from the dataset.
        """
        logger.info("📚 Preparing datasets...")
        if override_max_segments is not None:
            logger.info(f"📊 Progressive training: limiting to {override_max_segments} segments per document")

        # FAST PATH: Check for chunked datasets first (2-5 min vs 30-60 min)
        if self.config.use_chunked_dataset and self.config.chunked_dataset_dir:
            logger.info("🚀 Using chunked dataset (fast loading enabled)")
            logger.info(f"📁 Chunked dataset directory: {self.config.chunked_dataset_dir}")
            logger.info(f"📊 Load mode: {self.config.chunked_load_mode}")

            from memxlnet.data import load_chunked_dataset

            # Load training dataset from chunks
            train_dataset = load_chunked_dataset(
                dataset_dir=self.config.chunked_dataset_dir,
                split=self.config.train_split,
                mode=self.config.chunked_load_mode,
                num_examples=self.config.chunked_num_examples or self.config.max_train_samples,
                chunk_indices=self.config.chunked_chunk_indices,
                max_n_segs=self.config.max_n_segs,
            )

            # Load evaluation dataset from chunks
            # For eval: use same num_examples as training if in first_n mode, or max_eval_samples if set
            eval_num_examples = None
            if self.config.chunked_load_mode == "first_n":
                eval_num_examples = self.config.chunked_num_examples or self.config.max_eval_samples

            eval_dataset = load_chunked_dataset(
                dataset_dir=self.config.chunked_dataset_dir,
                split=self.config.eval_split,
                mode=self.config.chunked_load_mode,
                num_examples=eval_num_examples,
                chunk_indices=None,  # Use all chunks for eval
                max_n_segs=self.config.max_n_segs,
            )

            logger.info("✅ Loaded chunked datasets:")
            logger.info(f"   Training documents: {len(train_dataset)}")
            logger.info(f"   Evaluation documents: {len(eval_dataset)}")

            # Create memory collate config only if memory enabled AND wrapper is active
            memory_collate_cfg = None
            if self.mem_token_info and hasattr(self.model, "get_initial_memory"):
                from memxlnet.data import MemoryCollateConfig

                memory_collate_cfg = MemoryCollateConfig(
                    enable=True,
                    mem_read_ids=self.mem_token_info["mem_read_ids"],
                    mem_write_ids=self.mem_token_info["mem_write_ids"],
                    max_seq_length=self.config.max_seq_length,
                    cls_token_id=self.tokenizer.cls_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            train_dataloader = create_dataloader(
                train_dataset,  # type: ignore[arg-type]  # ChunkedDataset is compatible with Dataset protocol
                batch_size=self.config.train_batch_size,
                shuffle=True,
                num_workers=0,  # ChunkedDataset doesn't work well with multiprocessing
                memory_collate_config=memory_collate_cfg,
                use_time_step_major=True,
                max_segments=override_max_segments,  # Progressive training override or None for all
                segment_selection_strategy=self.config.segment_selection_strategy,
                epoch_seed=self.config.segment_selection_seed,
            )

            eval_dataloader = create_dataloader(
                eval_dataset,  # type: ignore[arg-type]  # ChunkedDataset is compatible with Dataset protocol
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                num_workers=0,  # ChunkedDataset doesn't work well with multiprocessing
                memory_collate_config=memory_collate_cfg,
                use_time_step_major=True,
                max_segments=override_max_segments,  # Use same limit for evaluation
                segment_selection_strategy=self.config.segment_selection_strategy,
                epoch_seed=self.config.segment_selection_seed,
            )

            logger.info("✅ Chunked dataset loading complete (fast path)")
            return train_dataloader, eval_dataloader, eval_dataset

        # Only preprocess locally if NOT using Hub datasets
        # (create_dataset_from_cache handles Hub → local cache → process pipeline)
        if not (self.config.use_hub_dataset and self.config.hub_dataset_id):
            logger.info("⚠️  Hub datasets disabled - will preprocess locally")

            # Process and cache training data
            logger.info(f"Processing training data: {self.config.dataset_name}")
            train_features = process_and_cache_dataset(
                dataset_name=self.config.dataset_name,
                split=self.config.train_split,
                cache_dir=self.config.cache_dir,
                max_examples=self.config.max_train_samples,
                max_seq_length=self.config.max_seq_length,
                doc_stride=self.config.doc_stride,
                streaming_chunk_size=self.config.streaming_chunk_size,
                tokenizer=self.tokenizer,
            )
            logger.info(f"✅ Training data: {train_features} features cached")

            # Process and cache evaluation data
            logger.info(f"Processing evaluation data: {self.config.dataset_name}")
            eval_features = process_and_cache_dataset(
                dataset_name=self.config.dataset_name,
                split=self.config.eval_split,
                cache_dir=self.config.cache_dir,
                max_examples=self.config.max_eval_samples,
                max_seq_length=self.config.max_seq_length,
                doc_stride=self.config.doc_stride,
                streaming_chunk_size=self.config.streaming_chunk_size,
                tokenizer=self.tokenizer,
            )
            logger.info(f"✅ Evaluation data: {eval_features} features cached")
        else:
            logger.info(f"🚀 Hub datasets enabled: {self.config.hub_dataset_id}")
            logger.info("   Skipping local preprocessing - will try loading from Hub first")

        # Create datasets (with Hub support - handles Hub → local cache → process automatically)
        train_dataset = create_dataset_from_cache(
            dataset_name=self.config.dataset_name,
            split=self.config.train_split,
            cache_dir=self.config.cache_dir,
            max_examples=self.config.max_train_samples,
            max_seq_length=self.config.max_seq_length,
            doc_stride=self.config.doc_stride,
            max_n_segs=self.config.max_n_segs,
            tokenizer=self.tokenizer,
            hub_dataset_id=self.config.hub_dataset_id,
            use_hub_dataset=self.config.use_hub_dataset,
            hub_token=self.config.hub_token,
        )

        eval_dataset = create_dataset_from_cache(
            dataset_name=self.config.dataset_name,
            split=self.config.eval_split,
            cache_dir=self.config.cache_dir,
            max_examples=self.config.max_eval_samples,
            max_seq_length=self.config.max_seq_length,
            doc_stride=self.config.doc_stride,
            max_n_segs=self.config.max_n_segs,
            tokenizer=self.tokenizer,
            hub_dataset_id=self.config.hub_dataset_id,
            use_hub_dataset=self.config.use_hub_dataset,
            hub_token=self.config.hub_token,
        )

        # Create memory collate config only if memory enabled AND wrapper is active
        memory_collate_cfg = None
        if self.mem_token_info and hasattr(self.model, "get_initial_memory"):
            from memxlnet.data import MemoryCollateConfig

            memory_collate_cfg = MemoryCollateConfig(
                enable=True,
                mem_read_ids=self.mem_token_info["mem_read_ids"],
                mem_write_ids=self.mem_token_info["mem_write_ids"],
                max_seq_length=self.config.max_seq_length,
                cls_token_id=self.tokenizer.cls_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Create dataloaders with time-step-major batching for MA-XLNet
        logger.info("📊 Creating time-step-major dataloaders for MA-XLNet")
        train_dataloader = create_dataloader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=12,
            memory_collate_config=memory_collate_cfg,
            use_time_step_major=True,
            max_segments=override_max_segments,  # Progressive training override or None for all
            segment_selection_strategy=self.config.segment_selection_strategy,
            epoch_seed=self.config.segment_selection_seed,
        )

        eval_dataloader = create_dataloader(
            eval_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=12,
            memory_collate_config=memory_collate_cfg,
            use_time_step_major=True,
            max_segments=override_max_segments,  # Use same limit for evaluation
            segment_selection_strategy=self.config.segment_selection_strategy,
            epoch_seed=self.config.segment_selection_seed,
        )

        logger.info(f"📊 Training documents: {len(train_dataset)}")
        logger.info(f"📊 Evaluation documents: {len(eval_dataset)}")

        return train_dataloader, eval_dataloader, eval_dataset

    def prepare_training(self, train_dataloader: DataLoader):
        """Prepare optimizer and scheduler."""
        # Calculate total training steps
        num_training_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        logger.info("🔧 Training setup:")
        logger.info(f"   Total training steps: {num_training_steps}")
        logger.info(f"   Warmup steps: {num_warmup_steps}")
        logger.info(f"   Gradient accumulation steps: {self.config.gradient_accumulation_steps}")

    def train_one_document_batch(self, time_step_batches: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Train on one batch of documents using recurrent memory approach.

        This is the core implementation of the recurrent streaming model:
        - Processes documents as streams of chunks
        - Maintains memory (mems) across chunks within each document
        - Calculates loss after processing entire documents

        Args:
            time_step_batches: List of batches, one for each time step

        Returns:
            Average loss for this document batch
        """
        self.model.train()

        # Branch: wrapper with explicit memory vs original XLNet mems
        if hasattr(self.model, "get_initial_memory"):
            # Initialize per-document memory bank
            if not hasattr(self, "memory_bank"):
                self.memory_bank: dict[str, torch.Tensor] = {}

            per_doc_logits_start: dict[str, list] = defaultdict(list)
            per_doc_logits_end: dict[str, list] = defaultdict(list)
            per_doc_labels_start: dict[str, list] = defaultdict(list)
            per_doc_labels_end: dict[str, list] = defaultdict(list)

            for time_step, batch in enumerate(time_step_batches):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions = batch["end_positions"].to(self.device)
                document_mask = batch["document_mask"].to(self.device)

                # Build memory_state tensor per batch ordering
                memory_states = []
                for ex_id, active in zip(batch["example_ids"], document_mask.tolist()):
                    if not active:
                        memory_states.append(self.model.get_initial_memory(1, device=self.device)[0])
                    else:
                        prev = self.memory_bank.get(ex_id)
                        if prev is None:
                            prev = self.model.get_initial_memory(1, device=self.device)[0]
                        memory_states.append(prev)
                memory_state_batch = torch.stack(memory_states, dim=0)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    memory_state=memory_state_batch,
                    mem_read_ids=self.mem_token_info.get("mem_read_ids") if self.mem_token_info else None,
                    mem_write_ids=self.mem_token_info.get("mem_write_ids") if self.mem_token_info else None,
                )

                start_logits = outputs["start_logits"]
                end_logits = outputs["end_logits"]
                new_memory_state = outputs["new_memory_state"]

                # Store updated memory for active docs
                for i, (ex_id, active) in enumerate(zip(batch["example_ids"], document_mask.tolist())):
                    if active:
                        self.memory_bank[ex_id] = new_memory_state[i].detach()

                # Append logits/labels to per-doc buffers for active docs
                # NOTE: Keep gradients attached - needed for loss.backward()
                for i, active in enumerate(document_mask.tolist()):
                    ex_id_raw = batch["example_ids"][i]
                    ex_id = str(ex_id_raw)  # Ensure string key for dict
                    if not active:
                        continue
                    per_doc_logits_start[ex_id].append(start_logits[i])
                    per_doc_logits_end[ex_id].append(end_logits[i])
                    per_doc_labels_start[ex_id].append(start_positions[i])
                    per_doc_labels_end[ex_id].append(end_positions[i])

            # Compute loss per document; use document-level global concat if enabled
            loss_fct = nn.CrossEntropyLoss()
            total_loss = torch.tensor(0.0, device=self.device)
            doc_count = 0
            for ex_id in per_doc_logits_start.keys():
                starts = per_doc_logits_start[ex_id]
                ends = per_doc_logits_end[ex_id]
                y_s = per_doc_labels_start[ex_id]
                y_e = per_doc_labels_end[ex_id]
                if not starts:
                    continue

                if self._use_global_softmax_epoch:
                    # Document-level global softmax: flatten logits across segments
                    # Positive segment if start/end labels differ (real span)
                    seg_lengths = [t.size(-1) for t in starts]
                    offsets = []
                    cum = 0
                    for L in seg_lengths:
                        offsets.append(cum)
                        cum += L

                    pos_index = None
                    for i, (ls, le) in enumerate(zip(y_s, y_e)):
                        if int(ls.item()) != int(le.item()):
                            pos_index = i
                            break

                    if pos_index is not None:
                        target_s = offsets[pos_index] + int(y_s[pos_index].item())
                        target_e = offsets[pos_index] + int(y_e[pos_index].item())
                    else:
                        # No positive: fall back to last segment's (use its label index)
                        target_s = offsets[-1] + int(y_s[-1].item())
                        target_e = offsets[-1] + int(y_e[-1].item())

                    logits_s_flat = torch.cat(starts, dim=0).unsqueeze(0)  # [1, sum_L]
                    logits_e_flat = torch.cat(ends, dim=0).unsqueeze(0)  # [1, sum_L]
                    labels_s = torch.tensor([target_s], device=logits_s_flat.device)
                    labels_e = torch.tensor([target_e], device=logits_e_flat.device)
                else:
                    # Per-segment CE averaged
                    logits_s = torch.stack(starts, dim=0).view(-1, starts[0].size(-1))
                    logits_e = torch.stack(ends, dim=0).view(-1, ends[0].size(-1))
                    labels_s = torch.stack(y_s, dim=0).view(-1)
                    labels_e = torch.stack(y_e, dim=0).view(-1)

                start_loss = loss_fct(logits_s_flat if self._use_global_softmax_epoch else logits_s, labels_s)
                end_loss = loss_fct(logits_e_flat if self._use_global_softmax_epoch else logits_e, labels_e)
                total_loss += 0.5 * (start_loss + end_loss)
                doc_count += 1

            # Cleanup memory for documents completed in this batch to avoid growth
            for ex_id in list(per_doc_logits_start.keys()):
                if hasattr(self, "memory_bank") and ex_id in self.memory_bank:
                    self.memory_bank.pop(ex_id, None)

            if doc_count == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            avg_loss: torch.Tensor = total_loss / doc_count
            return avg_loss

        # Fallback: original XLNet mems path (no explicit memory tokens)
        mems = None
        all_start_logits = []
        all_end_logits = []
        all_start_labels = []
        all_end_labels = []

        for time_step, batch in enumerate(time_step_batches):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            start_positions = batch["start_positions"].to(self.device)
            end_positions = batch["end_positions"].to(self.device)
            document_mask = batch["document_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mems=mems
            )
            mems = outputs.mems

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            active = document_mask.bool()
            if active.any():
                # NOTE: Keep gradients attached - needed for loss.backward()
                all_start_logits.append(start_logits[active])
                all_end_logits.append(end_logits[active])
                all_start_labels.append(start_positions[active])
                all_end_labels.append(end_positions[active])

        if all_start_logits:
            combined_start_logits = torch.cat(all_start_logits, dim=0)
            combined_end_logits = torch.cat(all_end_logits, dim=0)
            combined_start_labels = torch.cat(all_start_labels, dim=0)
            combined_end_labels = torch.cat(all_end_labels, dim=0)
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(combined_start_logits, combined_start_labels)
            end_loss = loss_fct(combined_end_logits, combined_end_labels)
            total_loss_val: torch.Tensor = (start_loss + end_loss) / 2
            return total_loss_val
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def extract_predictions_from_logits(
        self,
        all_start_logits: list[torch.Tensor],
        all_end_logits: list[torch.Tensor],
        all_example_ids: list[str],
        all_offset_mappings: list[list[tuple[int, int]]],
        all_contexts: list[str],
        all_input_ids: list[list[int]] | None = None,
        all_token_type_ids: list[list[int]] | None = None,
        all_cls_indices: list[int] | None = None,
        no_answer_threshold: float = 0.0,
        max_answer_length: int = 30,
        n_best_size: int = 20,
        return_score_diff: bool = False,
    ) -> dict[str, str | tuple[str, float]]:
        """
        Extract predictions from start and end logits using recurrent memory approach.

        This implements Phase 3 from the research document: extracting final answers
        from the collection of logits produced across all chunks.

        Uses "any positive" logic when use_any_positive_logic=True (default):
        - If ANY segment confidently predicts an answer, use it
        - Only predict no-answer when ALL segments predict no-answer
        - Prevents bias toward no-answer in long documents with many segments
        """
        predictions = {}

        # Load the original dataset to get actual contexts
        try:
            from datasets import load_dataset

            squad_dataset = load_dataset(self.config.dataset_name, split=self.config.eval_split)
            # Create mapping from example_id to context
            id_to_context = {example["id"]: example["context"] for example in squad_dataset}
        except Exception as e:
            logger.warning(f"Could not load original dataset for contexts: {e}")
            id_to_context = {}

        # Group logits by example_id (document)
        logits_by_example = defaultdict(list)

        # Debug: Check array lengths
        logger.debug(
            f"Array lengths - logits: {len(all_start_logits)}, example_ids: {len(all_example_ids)}, "
            f"offset_mappings: {len(all_offset_mappings)}, contexts: {len(all_contexts)}"
        )

        # Ensure all arrays have the same length
        lens = [len(all_start_logits), len(all_example_ids), len(all_offset_mappings), len(all_contexts)]
        if all_input_ids is not None:
            lens.append(len(all_input_ids))
        if all_token_type_ids is not None:
            lens.append(len(all_token_type_ids))
        if all_cls_indices is not None:
            lens.append(len(all_cls_indices))
        min_length = min(lens)

        if min_length < len(all_start_logits):
            logger.warning(f"Trimming arrays to common length {min_length} (was {len(all_start_logits)})")
            all_start_logits = all_start_logits[:min_length]
            all_end_logits = all_end_logits[:min_length]
            all_example_ids = all_example_ids[:min_length]
            all_offset_mappings = all_offset_mappings[:min_length]
            all_contexts = all_contexts[:min_length]
            if all_input_ids is not None:
                all_input_ids = all_input_ids[:min_length]
            if all_token_type_ids is not None:
                all_token_type_ids = all_token_type_ids[:min_length]
            if all_cls_indices is not None:
                all_cls_indices = all_cls_indices[:min_length]

        for i, example_id in enumerate(all_example_ids):
            # Use actual context from dataset if available, otherwise use provided context
            actual_context = id_to_context.get(example_id, all_contexts[i] if i < len(all_contexts) else "")

            entry = {
                "start_logits": all_start_logits[i],
                "end_logits": all_end_logits[i],
                "offset_mapping": all_offset_mappings[i] if i < len(all_offset_mappings) else [],
                "context": actual_context,
            }
            if all_input_ids is not None and i < len(all_input_ids):
                entry["input_ids"] = all_input_ids[i]
            if all_token_type_ids is not None and i < len(all_token_type_ids):
                entry["token_type_ids"] = all_token_type_ids[i]
            if all_cls_indices is not None and i < len(all_cls_indices):
                entry["cls_index"] = all_cls_indices[i]
            logits_by_example[example_id].append(entry)

        for example_id, example_logits in logits_by_example.items():
            if self.config.use_any_positive_logic:
                # NEW APPROACH: "Any Positive" Logic
                # If ANY segment confidently predicts an answer, use it
                # Only predict no-answer when ALL segments predict no-answer
                result = self._extract_prediction_any_positive(
                    example_id, example_logits, no_answer_threshold, max_answer_length, return_score_diff
                )
            else:
                # ORIGINAL APPROACH: Best score across all segments
                result = self._extract_prediction_best_score(
                    example_id, example_logits, no_answer_threshold, max_answer_length, return_score_diff
                )
            predictions[example_id] = result

        return predictions

    def _extract_prediction_any_positive(
        self,
        example_id: str,
        example_logits: list[dict],
        no_answer_threshold: float,
        max_answer_length: int,
        return_score_diff: bool = False,
    ) -> str | tuple[str, float]:
        """Extract prediction using 'any positive' logic for multi-segment documents."""
        confident_answer_candidates = []
        segment_decisions = []

        for chunk_idx, chunk_data in enumerate(example_logits):
            start_logits = chunk_data["start_logits"].cpu().numpy()
            end_logits = chunk_data["end_logits"].cpu().numpy()
            offset_mapping = chunk_data["offset_mapping"]
            context = chunk_data["context"]
            tts = chunk_data.get("token_type_ids", None)
            ids = chunk_data.get("input_ids", None)
            # Note: offset_mapping filtering removed - span filtering logic handles this

            # Ensure arrays are 1D
            if start_logits.ndim > 1:
                start_logits = start_logits.flatten()
            if end_logits.ndim > 1:
                end_logits = end_logits.flatten()

            logger.debug(f"Processing chunk {chunk_idx} for {example_id} with start_logits shape: {start_logits.shape}")

            # Robust no-answer score: consider CLS, last token, and last non-context idx
            def _robust_no_answer_score(start_l, end_l, ids_list, tts_list, offsets):
                candidates = []
                # CLS position if present
                try:
                    if ids_list is not None and isinstance(ids_list, list):
                        cls_id = self.tokenizer.cls_token_id
                        cls_pos = [i for i, tok in enumerate(ids_list) if tok == cls_id]
                        if cls_pos:
                            candidates.append(cls_pos[-1])
                except Exception:
                    pass
                # Last index
                candidates.append(len(start_l) - 1)
                # Last non-context token (token_type_ids != 1 or offset (0,0))
                if tts_list is not None and offsets:
                    for i in range(len(tts_list) - 1, -1, -1):
                        if tts_list[i] != 1 or (i < len(offsets) and offsets[i] == (0, 0)):
                            candidates.append(i)
                            break
                # Deduplicate and clamp
                cand = sorted({max(0, min(len(start_l) - 1, c)) for c in candidates})
                return (
                    max(float(start_l[c]) + float(end_l[c]) for c in cand)
                    if cand
                    else float(start_l[-1]) + float(end_l[-1])
                )

            # ✅ FIXED: Use actual CLS index from batch (like test_evaluation.py)
            cls_index = chunk_data.get("cls_index", None)

            # Fallback: If cls_index not in batch, search for it (legacy compatibility)
            if cls_index is None:
                if ids is not None and isinstance(ids, list):
                    cls_token_id = self.tokenizer.cls_token_id
                    cls_positions = [i for i, tok in enumerate(ids) if tok == cls_token_id]
                    if cls_positions:
                        cls_index = cls_positions[-1]

            segment_no_answer_score = _robust_no_answer_score(start_logits, end_logits, ids, tts, offset_mapping)

            # Find best answer candidate in THIS segment only
            segment_best_score = None
            segment_best_answer = ""
            segment_best_candidate = None

            for start_idx in range(len(start_logits)):
                for end_idx in range(start_idx, min(start_idx + max_answer_length, len(end_logits))):
                    # Skip non-context tokens (offset (0,0)) and CLS index
                    if (
                        start_idx < len(offset_mapping)
                        and end_idx < len(offset_mapping)
                        and offset_mapping[start_idx][0] == 0
                        and offset_mapping[start_idx][1] == 0
                    ):
                        continue
                    if (
                        end_idx < len(offset_mapping)
                        and offset_mapping[end_idx][0] == 0
                        and offset_mapping[end_idx][1] == 0
                    ):
                        continue
                    if cls_index is not None and (end_idx == cls_index or start_idx == cls_index):
                        continue

                    score = float(start_logits[start_idx]) + float(end_logits[end_idx])

                    # Extract answer text using offset mapping
                    if start_idx < len(offset_mapping) and end_idx < len(offset_mapping) and context:
                        start_char = offset_mapping[start_idx][0]
                        end_char = offset_mapping[end_idx][1]

                        # Validate span before extraction
                        if start_char >= end_char or start_char < 0 or end_char > len(context):
                            continue  # Invalid character span
                        if end_idx - start_idx > 60:  # Token-level length check
                            continue  # Span too long (SQuAD answers typically < 30 tokens)

                        answer_text = context[start_char:end_char].strip()

                        # Validate extracted text
                        if len(answer_text) == 0 or len(answer_text) > 300:  # Character-level length check
                            continue  # Empty or suspiciously long answer

                        if segment_best_score is None or score > segment_best_score:
                            segment_best_score = score
                            segment_best_answer = answer_text
                            segment_best_candidate = {
                                "score": score,
                                "text": answer_text,
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                                "chunk_idx": chunk_idx,
                            }

            # SEGMENT-LEVEL DECISION: Does this segment predict an answer?
            segment_predicts_answer = False
            if (
                segment_best_score is not None
                and segment_best_score > segment_no_answer_score + no_answer_threshold
                and segment_best_answer.strip() != ""
                and segment_best_candidate is not None
            ):
                segment_predicts_answer = True
                confident_answer_candidates.append(segment_best_candidate)

            segment_decisions.append(
                {
                    "chunk_idx": chunk_idx,
                    "predicts_answer": segment_predicts_answer,
                    "best_score": segment_best_score,
                    "no_answer_score": segment_no_answer_score,
                    "best_answer": segment_best_answer,
                }
            )

            # Format the score properly for logging
            score_str = f"{segment_best_score:.3f}" if segment_best_score is not None else "None"
            logger.debug(
                f"Segment {chunk_idx}: answer={segment_predicts_answer}, "
                f"best_score={score_str}, "
                f"no_answer_score={segment_no_answer_score:.3f}, "
                f"text='{segment_best_answer[:50]}...'"
            )

        # DOCUMENT-LEVEL DECISION: Any positive logic
        if confident_answer_candidates:
            # At least one segment confidently predicts an answer
            # Choose the best answer among confident predictions
            confident_answer_candidates.sort(key=lambda x: x["score"], reverse=True)
            best_answer_text = str(confident_answer_candidates[0]["text"])
            best_answer_score = confident_answer_candidates[0]["score"]

            # Calculate best no-answer score across all segments
            max_no_answer_score = max(d["no_answer_score"] for d in segment_decisions)
            score_diff = best_answer_score - max_no_answer_score

            logger.debug(
                f"Document {example_id}: ANSWER from segment {confident_answer_candidates[0]['chunk_idx']} "
                f"(score={best_answer_score:.3f}, score_diff={score_diff:.3f}): '{best_answer_text[:50]}...'"
            )

            if return_score_diff:
                return (best_answer_text, score_diff)
            else:
                return best_answer_text
        else:
            # ALL segments predict no-answer
            logger.debug(f"Document {example_id}: NO ANSWER (all {len(segment_decisions)} segments predict no-answer)")
            if return_score_diff:
                # Negative score diff means no-answer is preferred
                return ("", -10.0)
            else:
                return ""

    def _extract_prediction_best_score(
        self,
        example_id: str,
        example_logits: list[dict],
        no_answer_threshold: float,
        max_answer_length: int,
        return_score_diff: bool = False,
    ) -> str | tuple[str, float]:
        """Extract prediction using original 'best score' logic (for comparison)."""
        # Find best answer span across all chunks for this document
        best_score = None
        best_answer = ""
        no_answer_score = None

        all_candidates = []

        for chunk_data in example_logits:
            start_logits = chunk_data["start_logits"].cpu().numpy()
            end_logits = chunk_data["end_logits"].cpu().numpy()
            offset_mapping = chunk_data["offset_mapping"]
            context = chunk_data["context"]
            tts = chunk_data.get("token_type_ids", None)
            ids = chunk_data.get("input_ids", None)
            if tts is not None and offset_mapping:
                offset_mapping = [
                    (s, e) if (i < len(tts) and tts[i] == 1) else (0, 0) for i, (s, e) in enumerate(offset_mapping)
                ]

            # Ensure arrays are 1D
            if start_logits.ndim > 1:
                start_logits = start_logits.flatten()
            if end_logits.ndim > 1:
                end_logits = end_logits.flatten()

            # ✅ FIXED: Use actual CLS index from batch (like test_evaluation.py)
            cls_index = chunk_data.get("cls_index", None)

            # Fallback: If cls_index not in batch, search for it (legacy compatibility)
            if cls_index is None:
                try:
                    if ids is not None and isinstance(ids, list):
                        cls_token_id = self.tokenizer.cls_token_id
                        cls_positions = [i for i, tok in enumerate(ids) if tok == cls_token_id]
                        if cls_positions:
                            cls_index = cls_positions[-1]
                    if cls_index is None:
                        cls_index = len(start_logits) - 1
                except Exception:
                    cls_index = len(start_logits) - 1

            chunk_no_answer_score = float(start_logits[cls_index]) + float(end_logits[cls_index])
            if no_answer_score is None or chunk_no_answer_score > no_answer_score:
                no_answer_score = chunk_no_answer_score

            # Find best answer candidates in this chunk
            for start_idx in range(len(start_logits)):
                for end_idx in range(start_idx, min(start_idx + max_answer_length, len(end_logits))):
                    # Skip non-context tokens (offset (0,0)) and CLS index
                    if (
                        start_idx < len(offset_mapping)
                        and end_idx < len(offset_mapping)
                        and offset_mapping[start_idx][0] == 0
                        and offset_mapping[start_idx][1] == 0
                    ):
                        continue
                    if (
                        end_idx < len(offset_mapping)
                        and offset_mapping[end_idx][0] == 0
                        and offset_mapping[end_idx][1] == 0
                    ):
                        continue
                    if end_idx == cls_index or start_idx == cls_index:
                        continue

                    score = float(start_logits[start_idx]) + float(end_logits[end_idx])

                    # Extract answer text using offset mapping
                    if start_idx < len(offset_mapping) and end_idx < len(offset_mapping) and context:
                        start_char = offset_mapping[start_idx][0]
                        end_char = offset_mapping[end_idx][1]
                        answer_text = context[start_char:end_char].strip()

                        all_candidates.append(
                            {"score": score, "text": answer_text, "start_idx": start_idx, "end_idx": end_idx}
                        )

        # Sort candidates by score and take the best
        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        if all_candidates:
            best_candidate = all_candidates[0]
            best_score = best_candidate["score"]
            best_answer = best_candidate["text"]

        # Calculate score difference
        if best_score is not None and no_answer_score is not None:
            score_diff = best_score - no_answer_score
        else:
            score_diff = -10.0 if not best_answer else 10.0

        # Decide between best answer and no answer
        if (
            no_answer_score is not None
            and best_score is not None
            and no_answer_score > best_score + no_answer_threshold
        ):
            if return_score_diff:
                return ("", score_diff)
            else:
                return ""
        else:
            if return_score_diff:
                return (best_answer if best_answer else "", score_diff)
            else:
                return best_answer if best_answer else ""

    def get_references_from_dataset(self, dataset) -> dict[str, dict[str, Any]]:
        """Extract reference answers from the dataset for evaluation.

        IMPORTANT: This function now maps references using the preprocessed dataset's
        internal IDs (doc_0, doc_1, etc.) to match the prediction IDs. Previously,
        it used SQuAD's original IDs which caused 0% F1 due to ID mismatch.

        This function tries two approaches:
        1. First, try to extract answers from preprocessed features (if available)
        2. Fallback to reloading raw SQuAD dataset
        """
        references = {}

        # Option 1: Try to get answers from preprocessed features (preferred)
        # This uses the all_valid_answers field if available
        try:
            logger.info("📊 Attempting to extract references from preprocessed features...")

            # Get all document IDs from dataset
            if hasattr(dataset, 'get_all_documents'):
                document_ids = dataset.get_all_documents()

                for doc_id in document_ids:
                    # Get first segment for this document
                    segments = dataset.get_document_segments(doc_id)
                    if segments:
                        first_segment_idx = segments[0]
                        first_segment = dataset[first_segment_idx]

                        # Check if all_valid_answers is available
                        if "all_valid_answers" in first_segment and first_segment["all_valid_answers"]:
                            references[doc_id] = {
                                "answers": first_segment["all_valid_answers"],
                                "question": first_segment.get("question", ""),
                                "context": first_segment.get("context", ""),
                            }

                if references:
                    logger.info(f"✅ Extracted {len(references)} references from preprocessed features")
                    return references
                else:
                    logger.info("⚠️  No all_valid_answers found in features, falling back to raw dataset")
        except Exception as e:
            logger.info(f"⚠️  Could not extract from features: {e}, falling back to raw dataset")

        # Option 2: Fallback to loading raw SQuAD dataset
        try:
            # Load the original dataset to get ground truth answers
            from datasets import load_dataset

            logger.info("📊 Loading references from raw SQuAD v2 dataset...")
            squad_dataset = load_dataset(self.config.dataset_name, split=self.config.eval_split)

            # ✅ FIX: Map using internal doc_idx to match prediction IDs
            # The preprocessed dataset uses "doc_0", "doc_1", etc. as example_id
            # Predictions are keyed by these internal IDs, not SQuAD's original IDs
            for example_idx, example in enumerate(squad_dataset):
                context = example["context"]
                question = example["question"]
                answers = example["answers"]["text"] if example["answers"]["text"] else [""]
                original_squad_id = example["id"]

                # Use internal ID format to match predictions
                internal_id = f"doc_{example_idx}"

                # Store reference data by INTERNAL ID (not original SQuAD ID)
                references[internal_id] = {
                    "answers": answers,
                    "question": question,
                    "context": context,
                    "original_squad_id": original_squad_id,  # Keep for reference
                }

            logger.info(f"✅ Loaded {len(references)} references from raw dataset")

        except Exception as e:
            logger.warning(f"Could not load ground truth dataset: {e}")
            # Fallback to placeholder data for basic functionality
            for doc_idx in range(len(dataset)):
                doc_segments = dataset[doc_idx]
                if doc_segments:
                    first_segment = doc_segments[0]
                    example_id = first_segment.get("example_id", f"doc_{doc_idx}")

                    references[example_id] = {
                        "answers": [""],  # Placeholder - no ground truth available
                        "question": "",
                        "context": "",
                    }

        return references

    def evaluate(self, eval_dataloader: DataLoader[Any] | TimeStepMajorDataLoader, dataset=None) -> dict[str, float]:
        """
        Evaluate the model using recurrent memory approach with comprehensive SQuAD v2 metrics.

        This implements the evaluation phase using the recurrent memory methodology
        from the research document, now including proper QA metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Collect all predictions and logits
        all_start_logits = []
        all_end_logits = []
        all_example_ids = []
        all_offset_mappings = []
        all_contexts = []
        all_input_ids = []
        all_token_type_ids = []
        all_cls_indices = []

        logger.info("🔍 Starting evaluation...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader):
                # For evaluation, also use recurrent memory
                doc_batch = batch  # Assuming batch contains one document's chunks
                doc_loss, doc_predictions = self._evaluate_one_document(doc_batch)
                total_loss += doc_loss
                num_batches += 1

                # Collect prediction data
                all_start_logits.extend(doc_predictions["start_logits"])
                all_end_logits.extend(doc_predictions["end_logits"])
                all_example_ids.extend(doc_predictions["example_ids"])
                all_offset_mappings.extend(doc_predictions["offset_mappings"])
                all_contexts.extend(doc_predictions["contexts"])
                if "input_ids" in doc_predictions:
                    all_input_ids.extend(doc_predictions["input_ids"])
                if "token_type_ids" in doc_predictions:
                    all_token_type_ids.extend(doc_predictions["token_type_ids"])
                if "cls_indices" in doc_predictions:
                    all_cls_indices.extend(doc_predictions["cls_indices"])

                if batch_idx % 10 == 0:
                    logger.info(f"Evaluated {batch_idx + 1}/{len(eval_dataloader)} batches")

        # Calculate basic loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = {"eval_loss": avg_loss}

        logger.info(f"📊 Collected {len(all_start_logits)} predictions from {num_batches} batches")
        logger.info(f"📊 Example IDs sample: {all_example_ids[:5] if all_example_ids else 'None'}")

        # Debug: Check for consistency in collected data
        logger.info("📊 Collected data lengths:")
        logger.info(f"  Start logits: {len(all_start_logits)}")
        logger.info(f"  End logits: {len(all_end_logits)}")
        logger.info(f"  Example IDs: {len(all_example_ids)}")
        logger.info(f"  Offset mappings: {len(all_offset_mappings)}")
        logger.info(f"  Contexts: {len(all_contexts)}")
        logger.info(f"  Input IDs: {len(all_input_ids) if all_input_ids else 'Not collected'}")
        logger.info(f"  Token type IDs: {len(all_token_type_ids) if all_token_type_ids else 'Not collected'}")
        logger.info(f"  CLS indices: {len(all_cls_indices) if all_cls_indices else 'Not collected'}")

        # If we have the dataset, calculate SQuAD v2 metrics
        if dataset is not None and all_start_logits:
            try:
                logger.info("🎯 Extracting predictions from logits...")

                # Extract predictions from logits
                # Temporarily override any-positive behavior during warmup
                orig_any_pos = self.config.use_any_positive_logic
                self.config.use_any_positive_logic = self._use_any_positive_epoch
                predictions = self.extract_predictions_from_logits(
                    all_start_logits,
                    all_end_logits,
                    all_example_ids,
                    all_offset_mappings,
                    all_contexts,
                    all_input_ids if all_input_ids else None,
                    all_token_type_ids if all_token_type_ids else None,
                    all_cls_indices if all_cls_indices else None,
                    no_answer_threshold=self.config.no_answer_threshold,
                )
                self.config.use_any_positive_logic = orig_any_pos

                logger.info(f"✅ Extracted {len(predictions)} predictions")

                # Analyze prediction distribution
                # Note: predictions could be dict[str, str] or dict[str, tuple[str, float]]
                # but here we know it's dict[str, str] since return_score_diff=False (default)
                empty_predictions = sum(
                    1 for p in predictions.values() if (p == "" if isinstance(p, str) else p[0] == "")
                )
                non_empty_predictions = len(predictions) - empty_predictions
                logger.info("📊 Prediction distribution:")
                logger.info(
                    f"   - Empty (no-answer): {empty_predictions} ({100 * empty_predictions / len(predictions):.1f}%)"
                )
                logger.info(
                    f"   - Non-empty (has-answer): {non_empty_predictions} ({100 * non_empty_predictions / len(predictions):.1f}%)"
                )

                # Get reference answers
                logger.info("📚 Loading reference answers...")
                references = self.get_references_from_dataset(dataset)

                logger.info(f"✅ Loaded {len(references)} reference answers")

                # ✅ DIAGNOSTIC: Verify ID matching
                sample_pred_ids = list(predictions.keys())[:3]
                sample_ref_ids = list(references.keys())[:3]
                logger.info("🔍 ID Matching Verification:")
                logger.info(f"   Sample prediction IDs: {sample_pred_ids}")
                logger.info(f"   Sample reference IDs: {sample_ref_ids}")

                # Check how many predictions have matching references
                matched_ids = sum(1 for pred_id in predictions.keys() if pred_id in references)
                match_rate = 100 * matched_ids / len(predictions) if predictions else 0
                logger.info(
                    f"   ID match rate: {matched_ids}/{len(predictions)} ({match_rate:.1f}%)"
                )

                if match_rate < 50:
                    logger.warning(
                        "⚠️  LOW ID MATCH RATE! Predictions and references may be using different ID schemes."
                    )

                # Analyze ground truth distribution
                has_answer_refs = sum(1 for r in references.values() if r["answers"] and r["answers"][0] != "")
                no_answer_refs = len(references) - has_answer_refs
                logger.info("📊 Ground truth distribution:")
                logger.info(f"   - Has answer: {has_answer_refs} ({100 * has_answer_refs / len(references):.1f}%)")
                logger.info(f"   - No answer: {no_answer_refs} ({100 * no_answer_refs / len(references):.1f}%)")

                # Print some sample predictions for debugging
                sample_predictions = list(predictions.items())[:5]
                logger.info("📝 Sample predictions:")
                for example_id, pred_value in sample_predictions:
                    # Handle both str and tuple[str, float] cases
                    pred = pred_value if isinstance(pred_value, str) else pred_value[0]
                    pred_display = pred[:80] + "..." if len(pred) > 80 else pred
                    pred_display = pred_display if pred else "[EMPTY/NO-ANSWER]"
                    logger.info(f"  ID: {example_id}")
                    logger.info(f"    Prediction: {pred_display}")
                    if example_id in references:
                        ref_answers = references[example_id]["answers"]
                        ref_display = (
                            ref_answers[0][:80] + "..."
                            if ref_answers and len(ref_answers[0]) > 80
                            else (ref_answers[0] if ref_answers else "[NO-ANSWER]")
                        )
                        logger.info(f"    Ground truth: {ref_display}")

                # Calculate SQuAD v2 metrics
                logger.info("🧮 Calculating SQuAD v2 metrics...")
                # Ensure predictions are strings (extract from tuple if needed)
                predictions_str: dict[str, str] = {
                    qid: (pred if isinstance(pred, str) else pred[0]) for qid, pred in predictions.items()
                }
                squad_metrics = evaluate_squad_v2(predictions_str, references)
                metrics.update(squad_metrics)

                logger.info("✅ Evaluation complete:")
                logger.info(f"   Loss: {avg_loss:.4f}")
                logger.info(f"   Exact Match: {squad_metrics.get('exact', 0):.2f}%")
                logger.info(f"   F1 Score: {squad_metrics.get('f1', 0):.2f}%")
                logger.info(f"   HasAns EM: {squad_metrics.get('HasAns_exact', 0):.2f}%")
                logger.info(f"   HasAns F1: {squad_metrics.get('HasAns_f1', 0):.2f}%")
                logger.info(f"   NoAns EM: {squad_metrics.get('NoAns_exact', 0):.2f}%")
                logger.info(f"   NoAns F1: {squad_metrics.get('NoAns_f1', 0):.2f}%")

            except Exception as e:
                logger.error(f"❌ Could not calculate SQuAD metrics: {e}")
                logger.error("Continuing with loss-only evaluation...")
                # Add debug information
                logger.debug(f"Number of predictions: {len(all_start_logits) if all_start_logits else 0}")
                logger.debug(f"Dataset available: {dataset is not None}")
                import traceback

                logger.debug(f"Full error traceback: {traceback.format_exc()}")

                # Try to provide more specific error information
                try:
                    logger.debug("Debugging collected data:")
                    if all_start_logits:
                        logger.debug(f"First start_logits shape: {all_start_logits[0].shape}")
                    if all_end_logits:
                        logger.debug(f"First end_logits shape: {all_end_logits[0].shape}")
                    if all_example_ids:
                        logger.debug(f"Example IDs type: {type(all_example_ids[0])}")
                    if all_offset_mappings:
                        logger.debug(f"First offset_mapping length: {len(all_offset_mappings[0])}")
                except Exception as debug_e:
                    logger.debug(f"Could not debug data: {debug_e}")

        return metrics

    def _evaluate_one_document(self, doc_batch: list[dict[str, torch.Tensor]]) -> tuple[float, dict]:
        """Evaluate time-step-major batches, maintaining memory across chunks."""
        total_loss = 0.0
        num_chunks = len(doc_batch)
        loss_fct = nn.CrossEntropyLoss()

        # For prediction extraction
        doc_start_logits = []
        doc_end_logits = []
        doc_example_ids = []
        doc_offset_mappings = []
        doc_contexts = []
        doc_input_ids = []
        doc_token_type_ids = []
        doc_cls_indices = []

        # 🔍 DIAGNOSTIC: Log which evaluation path is taken
        if hasattr(self.model, "get_initial_memory"):
            logger.debug("✅ Evaluation using wrapper path (differentiable memory)")
        else:
            logger.debug("⚠️  Evaluation using fallback path (XLNet mems)")

        # Wrapper path with explicit memory
        if hasattr(self.model, "get_initial_memory"):
            eval_memory_bank: dict[str, torch.Tensor] = {}
            for chunk in doc_batch:
                input_ids = chunk["input_ids"].to(self.device)
                attention_mask = chunk["attention_mask"].to(self.device)
                token_type_ids = chunk.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                start_positions = chunk["start_positions"].to(self.device)
                end_positions = chunk["end_positions"].to(self.device)
                document_mask = chunk["document_mask"].to(self.device)
                example_ids = chunk.get("example_ids", [f"ex_{i}" for i in range(input_ids.size(0))])

                memory_states = []
                for ex_id, active in zip(example_ids, document_mask.tolist()):
                    if not active:
                        memory_states.append(self.model.get_initial_memory(1, device=self.device)[0])
                    else:
                        prev = eval_memory_bank.get(ex_id)
                        if prev is None:
                            prev = self.model.get_initial_memory(1, device=self.device)[0]
                        memory_states.append(prev)
                memory_state_batch = torch.stack(memory_states, dim=0)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    memory_state=memory_state_batch,
                    mem_read_ids=self.mem_token_info.get("mem_read_ids") if self.mem_token_info else None,
                    mem_write_ids=self.mem_token_info.get("mem_write_ids") if self.mem_token_info else None,
                )

                start_logits_all = outputs["start_logits"]
                end_logits_all = outputs["end_logits"]
                new_memory_state = outputs["new_memory_state"]

                active = document_mask.bool()
                if active.any():
                    sl = start_logits_all[active]
                    el = end_logits_all[active]
                    sp = start_positions[active]
                    ep = end_positions[active]
                    start_loss = loss_fct(sl, sp)
                    end_loss = loss_fct(el, ep)
                    total_loss += float(0.5 * (start_loss + end_loss))

                    doc_start_logits.extend([log for log in sl.cpu()])
                    doc_end_logits.extend([log for log in el.cpu()])
                    # Metadata alignment: only collect for active examples
                    act_list = active.tolist()
                    doc_example_ids.extend([example_ids[i] for i, a in enumerate(act_list) if a])
                    # Get metadata, handling both Tensor and list types
                    # Fixed: use singular form to match collate function
                    offsets_raw: Any = chunk.get("offset_mapping", [])
                    chunk_offsets: list[Any] = (
                        offsets_raw.tolist()
                        if isinstance(offsets_raw, torch.Tensor)
                        else (offsets_raw if isinstance(offsets_raw, list) else [])
                    )
                    contexts_raw: Any = chunk.get("context", [])
                    chunk_contexts: list[str] = (
                        contexts_raw.tolist()
                        if isinstance(contexts_raw, torch.Tensor)
                        else (contexts_raw if isinstance(contexts_raw, list) else [])
                    )
                    chunk_token_type_ids = chunk.get("token_type_ids", None)
                    ids_cpu = input_ids.cpu().tolist()

                    # ✅ Get CLS indices from batch (like test_evaluation.py)
                    cls_indices_raw: Any = chunk.get("cls_index", [])
                    chunk_cls_indices: list[int] = (
                        cls_indices_raw.tolist()
                        if isinstance(cls_indices_raw, torch.Tensor)
                        else (cls_indices_raw if isinstance(cls_indices_raw, list) else [])
                    )

                    if chunk_offsets:
                        doc_offset_mappings.extend([chunk_offsets[i] for i, a in enumerate(act_list) if a])
                    if chunk_contexts:
                        doc_contexts.extend([chunk_contexts[i] for i, a in enumerate(act_list) if a])
                    # Keep input_ids for potential CLS position checks in downstream logic
                    doc_input_ids.extend([ids_cpu[i] for i, a in enumerate(act_list) if a])
                    if chunk_token_type_ids is not None:
                        tts_cpu = chunk_token_type_ids.cpu().tolist()
                        doc_token_type_ids.extend([tts_cpu[i] for i, a in enumerate(act_list) if a])
                    if chunk_cls_indices:
                        doc_cls_indices.extend([chunk_cls_indices[i] for i, a in enumerate(act_list) if a])

                for i, (ex_id, a) in enumerate(zip(example_ids, document_mask.tolist())):
                    if a:
                        eval_memory_bank[ex_id] = new_memory_state[i].detach()

            avg_loss = total_loss / num_chunks if num_chunks > 0 else 0.0
            predictions_data = {
                "start_logits": doc_start_logits,
                "end_logits": doc_end_logits,
                "example_ids": doc_example_ids,
                "offset_mappings": doc_offset_mappings,
                "contexts": doc_contexts,
                "input_ids": doc_input_ids,
                "token_type_ids": doc_token_type_ids,
                "cls_indices": doc_cls_indices,
            }
            return avg_loss, predictions_data

        # Fallback: original mems path
        mems = None
        prev_batch_size = None  # Track batch size changes for mems compatibility

        for chunk in doc_batch:
            input_ids = chunk["input_ids"].to(self.device)
            attention_mask = chunk["attention_mask"].to(self.device)
            token_type_ids = chunk.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            start_positions = chunk["start_positions"].to(self.device)
            end_positions = chunk["end_positions"].to(self.device)

            # 🔧 FIX: Track batch size and reset mems if it changes
            batch_size = input_ids.size(0)
            if prev_batch_size is not None and batch_size != prev_batch_size:
                logger.debug(
                    f"⚠️  Batch size changed from {prev_batch_size} to {batch_size}, resetting mems "
                    f"(XLNet requires consistent batch size for memory concatenation)"
                )
                mems = None
            prev_batch_size = batch_size

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions,
                mems=mems,
            )
            if hasattr(outputs, "mems") and outputs.mems is not None:
                mems = [mem.detach() for mem in outputs.mems]

            if outputs.loss is not None:
                total_loss += outputs.loss.item()

            if hasattr(outputs, "start_logits") and hasattr(outputs, "end_logits"):
                batch_size = outputs.start_logits.size(0)
                for i in range(batch_size):
                    # Detach before moving to CPU to avoid memory leak
                    doc_start_logits.append(outputs.start_logits[i : i + 1].detach().cpu())
                    doc_end_logits.append(outputs.end_logits[i : i + 1].detach().cpu())

                # Get metadata, handling both Tensor and list types
                ids_raw: Any = chunk.get("example_ids", [])
                batch_example_ids: list[str] = (
                    ids_raw.tolist()
                    if isinstance(ids_raw, torch.Tensor)
                    else (ids_raw if isinstance(ids_raw, list) else [])
                )
                offsets_raw_2: Any = chunk.get("offset_mappings", [])
                batch_offset_mappings: list[Any] = (
                    offsets_raw_2.tolist()
                    if isinstance(offsets_raw_2, torch.Tensor)
                    else (offsets_raw_2 if isinstance(offsets_raw_2, list) else [])
                )
                contexts_raw_2: Any = chunk.get("contexts", [])
                batch_contexts: list[str] = (
                    contexts_raw_2.tolist()
                    if isinstance(contexts_raw_2, torch.Tensor)
                    else (contexts_raw_2 if isinstance(contexts_raw_2, list) else [])
                )
                for i in range(batch_size):
                    doc_example_ids.append(
                        batch_example_ids[i] if i < len(batch_example_ids) else f"doc_{len(doc_example_ids)}"
                    )
                    doc_offset_mappings.append(batch_offset_mappings[i] if i < len(batch_offset_mappings) else [])
                    doc_contexts.append(batch_contexts[i] if i < len(batch_contexts) else "")

        predictions_data = {
            "start_logits": doc_start_logits,
            "end_logits": doc_end_logits,
            "example_ids": doc_example_ids,
            "offset_mappings": doc_offset_mappings,
            "contexts": doc_contexts,
        }
        avg_loss = total_loss / num_chunks if num_chunks > 0 else 0.0
        return avg_loss, predictions_data

    def save_model(self, output_dir: str, step: int | None = None, is_best: bool = False):
        """Save model checkpoint and optionally push to HuggingFace Hub.

        Args:
            output_dir: Directory to save the model
            step: Training step number (for checkpoint naming)
            is_best: Whether this is the best model so far
        """
        if step is not None:
            save_dir = os.path.join(output_dir, f"checkpoint-{step}")
        else:
            save_dir = output_dir

        os.makedirs(save_dir, exist_ok=True)

        # Save model and tokenizer (support MemXLNet wrapper or plain HF model)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_dir)
        else:
            # Should not happen, but keep defensive branch
            torch.save(self.model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(save_dir)

        # Save training configuration
        try:
            with open(os.path.join(save_dir, "training_config.json"), "w") as f:
                json.dump(self.config.__dict__, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save training_config.json: {e}")

        # Save training state
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_eval_score": self.best_eval_score,
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            },
            os.path.join(save_dir, "training_state.pt"),
        )

        logger.info(f"💾 Model saved to {save_dir}")

        # Push to HuggingFace Hub if configured
        if self.config.push_to_hub_on_save and self.config.hub_model_id:
            should_push = False

            if self.config.hub_strategy == "every_save":
                should_push = True
            elif self.config.hub_strategy == "best_only" and is_best:
                should_push = True
            elif self.config.hub_strategy == "end" and step is None:
                should_push = True

            if should_push:
                self._push_to_hub(save_dir, step, is_best)

    def _push_to_hub(self, save_dir: str, step: int | None = None, is_best: bool = False):
        """Push model to HuggingFace Hub.

        Args:
            save_dir: Directory containing the saved model
            step: Training step number (for commit message)
            is_best: Whether this is the best model so far
        """
        try:
            # Get token from config or environment
            token = self.config.hub_token or os.getenv("HF_TOKEN")

            # Build commit message
            commit_parts = []
            if is_best:
                commit_parts.append("🏆 Best model")
            if step is not None:
                commit_parts.append(f"Step {step}")
            if self.epoch is not None:
                commit_parts.append(f"Epoch {self.epoch + 1}")

            commit_message = " | ".join(commit_parts) if commit_parts else "Update model"

            # Build revision/tag name for versioning
            revision = None
            if step is not None:
                revision = f"step-{step}"
            elif is_best:
                revision = "best"

            logger.info(f"📤 Pushing to HuggingFace Hub: {self.config.hub_model_id}")
            if revision:
                logger.info(f"   Revision: {revision}")

            # Push model using the model's push_to_hub method
            if hasattr(self.model, "push_to_hub"):
                repo_url = self.model.push_to_hub(
                    repo_id=self.config.hub_model_id,
                    commit_message=commit_message,
                    private=self.config.hub_private,
                    token=token,
                )
                logger.info(f"✅ Model pushed to Hub: {repo_url}")
            else:
                logger.warning("⚠️ Model does not support push_to_hub, skipping Hub upload")

            # Also push tokenizer
            try:
                self.tokenizer.push_to_hub(
                    repo_id=self.config.hub_model_id,
                    commit_message=f"Update tokenizer | {commit_message}",
                    private=self.config.hub_private,
                    token=token,
                )
                logger.info("✅ Tokenizer pushed to Hub")
            except Exception as e:
                logger.warning(f"⚠️ Could not push tokenizer to Hub: {e}")

        except Exception as e:
            logger.error(f"❌ Failed to push to HuggingFace Hub: {e}")
            logger.error("Continuing training without Hub upload...")

    def train_progressive_segments(self):
        """
        Progressive training with increasing segment counts.

        Train with progressively more segments per document to gradually
        learn long-context processing.
        """
        if not self.config.progressive_segments:
            logger.warning(
                "Progressive training requested but no progressive_segments specified. Using regular training."
            )
            return self.train()

        logger.info("🚀 Starting progressive segment training...")
        logger.info(f"📈 Segment progression: {self.config.progressive_segments}")

        original_max_n_segs = self.config.max_n_segs
        original_output_dir = self.config.output_dir

        for stage, max_segs in enumerate(self.config.progressive_segments):
            logger.info(
                f"\n🚀 STAGE {stage + 1}/{len(self.config.progressive_segments)}: Training with max {max_segs} segments"
            )

            # Update configuration for this stage (output paths only, NOT max_n_segs)
            # max_n_segs no longer used for truncation - segments selected dynamically at dataloader level
            self.config.output_dir = os.path.join(original_output_dir, f"stage_{stage + 1}_segs_{max_segs}")
            self.config.run_name = f"{original_output_dir.split('/')[-1]}_stage_{stage + 1}_segs_{max_segs}"

            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)

            # Load model from previous stage if available
            if stage > 0:
                prev_stage_dir = os.path.join(
                    original_output_dir, f"stage_{stage}_segs_{self.config.progressive_segments[stage - 1]}"
                )
                best_model_path = os.path.join(prev_stage_dir, "best_model")

                # Determine model source for previous stage
                model_load_path = None
                if os.path.exists(best_model_path):
                    model_load_path = best_model_path
                    logger.info(f"🔄 Loading model from previous stage (local): {best_model_path}")
                elif self.config.hub_model_id:
                    # Try loading from Hub with stage tag
                    hub_stage_revision = f"stage-{stage}"
                    model_load_path = self.config.hub_model_id
                    logger.info(
                        f"🔄 Loading model from Hub: {self.config.hub_model_id} (revision: {hub_stage_revision})"
                    )
                    # Note: HuggingFace from_pretrained supports revision parameter, but we'll try without it first

                if model_load_path:
                    # Preserve MemXLNet wrapper when memory is enabled
                    if (
                        self.config.memory_num_tokens
                        and self.config.memory_num_tokens > 0
                        and self.config.memory_impl == "token"
                    ):
                        try:
                            self.model = MemXLNetForQA.from_pretrained(
                                model_load_path,
                                mem_token_count=self.config.memory_num_tokens,
                                memory_init=self.config.memory_init,
                                memory_update=self.config.memory_update,
                            )
                            # Ensure embeddings sized to tokenizer (memory tokens)
                            try:
                                self.model.base.resize_token_embeddings(len(self.tokenizer))
                            except Exception as e:
                                logger.warning(f"Could not resize embeddings after stage load: {e}")
                        except Exception as e:
                            logger.warning(f"Falling back to base model load (MemXLNet state not found): {e}")
                            self.model = XLNetForQuestionAnsweringSimple.from_pretrained(model_load_path)
                            try:
                                self.model.resize_token_embeddings(len(self.tokenizer))
                            except Exception:
                                pass
                    else:
                        self.model = XLNetForQuestionAnsweringSimple.from_pretrained(model_load_path)
                    self.model.to(self.device)
                else:
                    logger.warning("⚠️ Previous stage model not found locally or on Hub")

            # Prepare data with segment limit for this stage
            # Segments selected dynamically at dataloader level using smart selection
            logger.info(f"📚 Preparing data with max {max_segs} segments per document (smart selection enabled)")
            train_dataloader, eval_dataloader, eval_dataset = self.prepare_data(override_max_segments=max_segs)

            # Reset training state for new stage
            self.global_step = 0
            self.epoch = 0
            self.best_eval_score = 0.0

            # Prepare training components
            self.prepare_training(train_dataloader)

            # Train for this stage
            logger.info(f"🎯 Starting training for stage {stage + 1}")
            self._train_single_stage(train_dataloader, eval_dataloader, eval_dataset, stage + 1)

            logger.info(f"✅ Stage {stage + 1} completed!")

        # Restore original configuration
        self.config.max_n_segs = original_max_n_segs
        self.config.output_dir = original_output_dir

        logger.info("🎉 Progressive training completed!")
        logger.info(f"📁 All stage results available in: {original_output_dir}")

    def _train_single_stage(self, train_dataloader, eval_dataloader, eval_dataset, stage_num):
        """Train a single stage of progressive training."""

        logger.info(f"📊 Stage {stage_num} Training Info:")
        logger.info(f"   Training documents: {len(train_dataloader.dataset)}")
        logger.info(f"   Training batches: {len(train_dataloader)}")
        logger.info(f"   Max segments per document: {self.config.max_n_segs}")

        # self.evaluate(eval_dataloader, eval_dataset)

        # Training loop for this stage
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"📚 Stage {stage_num}, Epoch {epoch + 1}/{self.config.num_epochs}")

            # Training phase
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            progress_bar = tqdm(train_dataloader, desc=f"Stage {stage_num} Epoch {epoch + 1}")

            # Per-epoch warmup behavior
            self._use_global_softmax_epoch = (
                False if epoch < self.config.warmup_disable_global_softmax_epochs else self.config.use_global_softmax
            )
            self._use_any_positive_epoch = (
                False if epoch < self.config.warmup_disable_any_positive_epochs else self.config.use_any_positive_logic
            )
            # Freeze base for early epochs if configured
            if epoch < self.config.warmup_freeze_base_epochs:
                self._freeze_base_transformer(True)
            elif epoch == self.config.warmup_freeze_base_epochs:
                self._freeze_base_transformer(False)

            for batch_idx, time_step_batches in enumerate(progress_bar):
                # Process one batch of documents with recurrent memory
                loss = self.train_one_document_batch(time_step_batches)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                epoch_loss += loss.item()
                epoch_steps += 1

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    # Update weights
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Clear CUDA cache to prevent memory fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        learning_rate = self.scheduler.get_last_lr()[0]

                        logger.info(
                            f"Stage {stage_num}, Step {self.global_step}: loss={avg_loss:.4f}, lr={learning_rate:.2e}"
                        )

                        if self.config.use_wandb and HAS_WANDB:
                            wandb.log(
                                {
                                    "train_loss": avg_loss,
                                    "learning_rate": learning_rate,
                                    "global_step": self.global_step,
                                    "stage": stage_num,
                                    "max_segments": self.config.max_n_segs,
                                }
                            )

                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        logger.info(f"🔍 Stage {stage_num}, Step {self.global_step} evaluation")
                        eval_results = self.evaluate(eval_dataloader, eval_dataset)

                        if self.config.use_wandb and HAS_WANDB:
                            log_dict = {
                                "eval_loss": eval_results["eval_loss"],
                                "global_step": self.global_step,
                                "stage": stage_num,
                                "max_segments": self.config.max_n_segs,
                            }
                            for key in ["exact", "f1", "HasAns_f1", "NoAns_f1"]:
                                if key in eval_results:
                                    log_dict[f"eval_{key}"] = eval_results[key]
                            wandb.log(log_dict)

                        # Save best model based on F1 score
                        eval_metric = eval_results.get("f1", -eval_results["eval_loss"])
                        if eval_metric > self.best_eval_score:
                            self.best_eval_score = eval_metric
                            self.save_model(os.path.join(self.config.output_dir, "best_model"), is_best=True)
                            logger.info(f"🏆 Stage {stage_num}: New best model! Metric: {eval_metric:.4f}")

                        self.model.train()

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_model(self.config.output_dir, self.global_step)

                # Update progress bar
                progress_bar.set_postfix(
                    {"loss": f"{epoch_loss / epoch_steps:.4f}", "step": self.global_step, "stage": stage_num}
                )

            # End of epoch evaluation
            logger.info(f"📊 Stage {stage_num}, End of epoch {epoch + 1} evaluation")
            eval_results = self.evaluate(eval_dataloader, eval_dataset)

            if self.config.use_wandb and HAS_WANDB:
                log_dict = {
                    "epoch": epoch + 1,
                    "stage": stage_num,
                    "max_segments": self.config.max_n_segs,
                    "epoch_eval_loss": eval_results["eval_loss"],
                }
                for key in ["exact", "f1", "HasAns_f1", "NoAns_f1"]:
                    if key in eval_results:
                        log_dict[f"epoch_eval_{key}"] = eval_results[key]
                wandb.log(log_dict)

        # Save final model for this stage
        self.save_model(os.path.join(self.config.output_dir, "final_model"), is_best=False)
        logger.info(f"💾 Stage {stage_num} final model saved")

    def train(self):
        """Main training loop implementing recurrent memory approach."""
        # Check if progressive training is requested
        if self.config.progressive_segments:
            return self.train_progressive_segments()

        logger.info("🚀 Starting XLNet recurrent memory training...")

        # Prepare data
        train_dataloader, eval_dataloader, eval_dataset = self.prepare_data()

        # Prepare training components
        self.prepare_training(train_dataloader)

        # Single stage training
        self._train_single_stage(train_dataloader, eval_dataloader, eval_dataset, 1)


def main():
    """Main function to run training with default configuration."""

    # Create training configuration
    config = TrainingConfig(
        # Model settings
        model_name="xlnet-base-cased",
        max_seq_length=384,
        doc_stride=128,
        # Dataset settings - CONFIGURE THESE FOR YOUR NEEDS
        dataset_name="squad_v2",
        max_train_samples=1000,  # Set to None for full dataset, or limit for testing
        max_eval_samples=200,  # Limit evaluation for faster feedback
        # Training settings
        num_epochs=2,  # Reduced for initial testing
        train_batch_size=2,  # Small batch size for memory efficiency
        eval_batch_size=2,
        learning_rate=3e-5,
        # Memory settings
        max_memory_gb=8.0,  # Adjust based on your system
        streaming_chunk_size=500,
        # Output settings
        output_dir="./outputs/xlnet-long-qa",
        run_name="xlnet-recurrent-memory",
        # Experiment tracking
        use_wandb=False,  # Set to True if you want to use Weights & Biases
        # Performance settings
        gradient_accumulation_steps=4,  # Effective batch size = train_batch_size * gradient_accumulation_steps
        eval_steps=10,  # Evaluate every 10 steps for faster feedback
        save_steps=200,  # Save checkpoint every 200 steps
        logging_steps=5,  # Log every 5 steps
    )

    # Print configuration
    logger.info("🔧 Training Configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"   {key}: {value}")

    # Initialize trainer
    trainer = XLNetRecurrentTrainer(config)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
