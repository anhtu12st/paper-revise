#!/usr/bin/env python3
"""Example: Training GMM-XLNet with Multi-Expert Memory.

This script demonstrates how to train a GMM-XLNet model with multi-expert memory
on the SQuAD v2 dataset. It showcases:
- Configuration of GMM-specific parameters
- Multi-expert memory initialization
- Routing behavior monitoring
- Expert specialization analysis

Usage:
    # Basic training
    python examples/train_with_gmm_memory.py

    # Custom configuration
    python examples/train_with_gmm_memory.py \\
        --num-experts 4 \\
        --memory-slots 16 \\
        --epochs 3 \\
        --batch-size 4

    # With Hub integration
    python examples/train_with_gmm_memory.py \\
        --hub-model-id username/gmm-xlnet-squad \\
        --push-to-hub
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import XLNetForQuestionAnsweringSimple, XLNetTokenizerFast

from gmmxlnet.models import GMMXLNetForQA
from gmmxlnet.training import GMMTrainingConfig, gmm_balanced_config
from gmmxlnet.utils import GMMAnalyzer
from memxlnet.data import ChunkedSquadDataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GMM-XLNet model with multi-expert memory"
    )

    # Model architecture
    parser.add_argument(
        "--base-model",
        type=str,
        default="xlnet-base-cased",
        help="Base XLNet model (default: xlnet-base-cased)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 6, 7, 8],
        help="Number of memory experts (default: 4)",
    )
    parser.add_argument(
        "--memory-slots",
        type=int,
        default=16,
        help="Number of memory slots per expert (default: 16)",
    )

    # GMM-specific parameters
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="write-based",
        choices=["write-based", "read-based"],
        help="Routing mode for memory reads (default: write-based)",
    )
    parser.add_argument(
        "--routing-temperature",
        type=float,
        default=1.0,
        help="Temperature for routing softmax (default: 1.0)",
    )
    parser.add_argument(
        "--load-balance-weight",
        type=float,
        default=0.01,
        help="Weight for load balance loss (default: 0.01)",
    )
    parser.add_argument(
        "--entropy-reg-weight",
        type=float,
        default=0.0,
        help="Weight for entropy regularization (default: 0.0)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="Maximum training examples (default: 1000 for demo)",
    )

    # Output and Hub
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gmm-xlnet-demo",
        help="Output directory for checkpoints (default: outputs/gmm-xlnet-demo)",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID (optional)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to HuggingFace Hub after training",
    )

    # Analysis
    parser.add_argument(
        "--analyze-routing",
        action="store_true",
        help="Analyze routing behavior after training",
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Skip training, only run analysis (requires existing checkpoint)",
    )

    return parser.parse_args()


def setup_tokenizer(base_model_name, memory_slots):
    """Initialize tokenizer with memory tokens."""
    print("\n[1/7] Setting up tokenizer with memory tokens...")

    tokenizer = XLNetTokenizerFast.from_pretrained(base_model_name)

    # Add memory tokens
    memory_tokens = []
    for i in range(memory_slots):
        memory_tokens.append(f"[MEM_READ_{i}]")
    for i in range(memory_slots):
        memory_tokens.append(f"[MEM_WRITE_{i}]")

    tokenizer.add_special_tokens({"additional_special_tokens": memory_tokens})

    print(f"✓ Tokenizer initialized")
    print(f"  - Base model: {base_model_name}")
    print(f"  - Total vocabulary size: {len(tokenizer)}")
    print(f"  - Memory tokens added: {len(memory_tokens)}")

    return tokenizer


def setup_model(args, tokenizer):
    """Initialize GMM-XLNet model."""
    print("\n[2/7] Initializing GMM-XLNet model...")

    # Load base XLNet model
    base_model = XLNetForQuestionAnsweringSimple.from_pretrained(args.base_model)

    # Resize embeddings to account for memory tokens
    base_model.resize_token_embeddings(len(tokenizer))

    # Wrap with GMM
    model = GMMXLNetForQA(
        base_model=base_model,
        num_experts=args.num_experts,
        memory_slots=args.memory_slots,
        routing_mode=args.routing_mode,
        routing_temperature=args.routing_temperature,
        pooling_method="mean",
        init_strategies="orthogonal",  # Use orthogonal initialization for diversity
        use_gmm_memory=True,
    )

    print(f"✓ GMM-XLNet model created")
    print(f"  - Number of experts: {model.num_experts}")
    print(f"  - Memory slots per expert: {model.memory_slots}")
    print(f"  - Routing mode: {model.routing_mode}")
    print(f"  - Routing temperature: {model.routing_temperature}")
    print(f"  - Hidden dimension: {model.hidden_dim}")

    return model


def load_datasets(args, tokenizer):
    """Load training and validation datasets."""
    print("\n[3/7] Loading SQuAD v2 dataset...")

    # Training dataset
    train_dataset = ChunkedSquadDataset(
        split="train",
        tokenizer=tokenizer,
        max_seq_length=512,
        doc_stride=128,
        max_examples=args.max_examples,
        max_n_segs=4,  # Maximum 4 segments per document
    )

    # Validation dataset (smaller for demo)
    val_dataset = ChunkedSquadDataset(
        split="validation",
        tokenizer=tokenizer,
        max_seq_length=512,
        doc_stride=128,
        max_examples=200,  # Small validation set for demo
        max_n_segs=4,
    )

    print(f"✓ Datasets loaded")
    print(f"  - Training examples: {len(train_dataset)}")
    print(f"  - Validation examples: {len(val_dataset)}")
    print(f"  - Documents in training set: {len(train_dataset.get_all_documents())}")

    return train_dataset, val_dataset


def create_config(args):
    """Create GMM training configuration."""
    print("\n[4/7] Creating training configuration...")

    config = GMMTrainingConfig(
        # GMM-specific
        use_gmm_memory=True,
        num_memory_experts=args.num_experts,
        routing_temperature=args.routing_temperature,
        routing_mode=args.routing_mode,
        entropy_regularization_weight=args.entropy_reg_weight,
        load_balance_weight=args.load_balance_weight,
        # Memory
        memory_num_tokens=args.memory_slots,
        # Training
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        # Progressive training
        progressive_segments=[2, 4],
        # Output
        output_dir=args.output_dir,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        # Hub (optional)
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )

    print(f"✓ Configuration created")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Progressive segments: {config.progressive_segments}")
    print(f"  - Load balance weight: {config.load_balance_weight}")

    # Save configuration
    os.makedirs(args.output_dir, exist_ok=True)
    config.to_json(os.path.join(args.output_dir, "config.json"))
    print(f"  - Config saved to: {args.output_dir}/config.json")

    return config


def train_model(model, train_dataset, val_dataset, config):
    """Train the GMM-XLNet model (simplified training loop for demo)."""
    print("\n[5/7] Training model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training metrics
    total_loss = 0.0
    routing_stats = {"expert_utilization": [0.0] * model.num_experts, "entropy_sum": 0.0, "steps": 0}

    print(f"✓ Starting training on {device}")
    print(f"  - Training batches: {len(train_loader)}")

    # Simple training loop (for demonstration)
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            # Initialize memory if needed
            if batch_idx == 0 or "memory_state" not in locals():
                memory_state = model.get_initial_memory(
                    batch_size=input_ids.size(0), device=device
                )

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                memory_state=memory_state,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update memory state
            memory_state = outputs["memory_state"]

            # Track routing statistics
            if "routing_probs" in outputs:
                routing_probs = outputs["routing_probs"].detach().cpu()
                expert_util = routing_probs.mean(dim=0).numpy()

                for i in range(model.num_experts):
                    routing_stats["expert_utilization"][i] += expert_util[i]

                # Compute entropy: H = -Σ p_j log(p_j)
                entropy = -(routing_probs * torch.log(routing_probs + 1e-10)).sum(dim=1).mean().item()
                routing_stats["entropy_sum"] += entropy
                routing_stats["steps"] += 1

            total_loss += loss.item()

            # Logging
            if (batch_idx + 1) % config.logging_steps == 0:
                avg_loss = total_loss / config.logging_steps
                avg_routing = [u / routing_stats["steps"] for u in routing_stats["expert_utilization"]]
                avg_entropy = routing_stats["entropy_sum"] / routing_stats["steps"]

                print(f"  Step {batch_idx + 1}/{len(train_loader)}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Entropy={avg_entropy:.3f}, "
                      f"Expert Util={[f'{u:.2f}' for u in avg_routing]}")

                total_loss = 0.0

    print("\n✓ Training completed")

    # Save final model
    save_path = Path(config.output_dir) / "final"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    print(f"  - Model saved to: {save_path}")

    return model


def analyze_routing(model, val_dataset):
    """Analyze expert routing behavior."""
    print("\n[6/7] Analyzing routing behavior...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create analyzer
    analyzer = GMMAnalyzer(model=model, device=device)

    # Create validation dataloader
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Track routing
    routing_stats = analyzer.track_routing(
        dataloader=val_loader,
        max_segments=100,  # Analyze first 100 segments
    )

    print(f"✓ Routing analysis completed")
    print(f"\n  Expert Utilization:")
    for i, util in enumerate(routing_stats["expert_utilization"]):
        print(f"    Expert {i}: {util:.2%}")

    print(f"\n  Routing Entropy:")
    print(f"    Mean: {routing_stats['mean_entropy']:.3f}")
    print(f"    Std:  {routing_stats['std_entropy']:.3f}")

    # Compute additional metrics
    diversity = analyzer.compute_expert_diversity()
    consistency = analyzer.compute_routing_consistency()

    print(f"\n  Expert Diversity (cosine similarity):")
    print(f"    Average off-diagonal: {diversity[~torch.eye(len(diversity), dtype=bool)].mean():.3f}")
    print(f"    (< 0.5 indicates good specialization)")

    print(f"\n  Routing Consistency: {consistency:.3f}")
    print(f"    (< 0.3: context-dependent, > 0.8: may indicate collapse)")

    return routing_stats, analyzer


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 70)
    print("GMM-XLNet Training Example")
    print("Multi-Expert Memory-Augmented Question Answering")
    print("=" * 70)

    # Setup
    tokenizer = setup_tokenizer(args.base_model, args.memory_slots)
    model = setup_model(args, tokenizer)

    # Load data
    train_dataset, val_dataset = load_datasets(args, tokenizer)

    # Create configuration
    config = create_config(args)

    # Train (unless skipped)
    if not args.no_training:
        model = train_model(model, train_dataset, val_dataset, config)
    else:
        print("\n[5/7] Skipping training (--no-training flag set)")
        # Load existing model
        checkpoint_path = Path(args.output_dir) / "final"
        if checkpoint_path.exists():
            model = GMMXLNetForQA.from_pretrained(str(checkpoint_path))
            print(f"✓ Loaded model from: {checkpoint_path}")
        else:
            print(f"✗ No checkpoint found at: {checkpoint_path}")
            return

    # Analyze routing (if requested)
    if args.analyze_routing:
        routing_stats, analyzer = analyze_routing(model, val_dataset)
    else:
        print("\n[6/7] Skipping routing analysis (use --analyze-routing to enable)")

    # Push to Hub (if requested)
    if args.push_to_hub and args.hub_model_id:
        print("\n[7/7] Pushing model to HuggingFace Hub...")
        # This requires huggingface_hub to be installed and HF_TOKEN to be set
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.upload_folder(
                folder_path=str(Path(args.output_dir) / "final"),
                repo_id=args.hub_model_id,
                repo_type="model",
            )
            print(f"✓ Model pushed to: https://huggingface.co/{args.hub_model_id}")
        except Exception as e:
            print(f"✗ Failed to push to Hub: {e}")
    else:
        print("\n[7/7] Skipping Hub upload (use --push-to-hub to enable)")

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print(f"\nModel saved to: {args.output_dir}/final")
    if args.hub_model_id and args.push_to_hub:
        print(f"Model uploaded to: https://huggingface.co/{args.hub_model_id}")
    print("\nNext steps:")
    print("  1. Evaluate the model: python scripts/evaluate_cls_fix.py --model-path outputs/gmm-xlnet-demo/final")
    print("  2. Analyze experts: python examples/analyze_gmm_experts.py --model-path outputs/gmm-xlnet-demo/final")
    print("  3. Try different configurations: python examples/train_with_gmm_memory.py --num-experts 8")


if __name__ == "__main__":
    main()
