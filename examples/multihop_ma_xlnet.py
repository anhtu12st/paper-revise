#!/usr/bin/env python3
"""Example script demonstrating enhanced MA-XLNet for multi-hop QA.

This script shows how to:
1. Configure and train MA-XLNet with differentiable memory
2. Track multi-hop reasoning chains
3. Visualize memory operations
4. Evaluate performance on multi-hop questions
"""

import torch
from transformers import XLNetForQuestionAnsweringSimple, XLNetTokenizerFast

from memxlnet.data import create_dataloader
from memxlnet.models import MemXLNetForQA
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer
from memxlnet.utils import HopTracker, MemoryVisualizer


def create_enhanced_model(config: TrainingConfig):
    """Create MA-XLNet model with enhanced memory.

    Args:
        config: Training configuration

    Returns:
        Model and tokenizer
    """
    print("🚀 Creating enhanced MA-XLNet model...")

    # Load base model and tokenizer
    base_model = XLNetForQuestionAnsweringSimple.from_pretrained(config.model_name)
    tokenizer = XLNetTokenizerFast.from_pretrained(config.model_name)

    # Wrap with enhanced memory
    model = MemXLNetForQA(
        base_model=base_model,
        mem_token_count=config.memory_num_tokens,
        memory_init=config.memory_init,
        memory_update=config.memory_update,
        # Enhanced memory parameters
        use_differentiable_memory=config.use_differentiable_memory,
        num_memory_heads=config.num_memory_heads,
        memory_sharpness=config.memory_sharpness,
        enable_usage_tracking=config.enable_usage_tracking,
        enable_temporal_links=config.enable_temporal_links,
        memory_slots=config.memory_slots,
    )

    print("✅ Model created with:")
    print(f"   - Differentiable memory: {config.use_differentiable_memory}")
    print(f"   - Memory heads: {config.num_memory_heads}")
    print(f"   - Memory slots: {config.memory_slots or 'auto'}")
    print(f"   - Usage tracking: {config.enable_usage_tracking}")
    print(f"   - Temporal links: {config.enable_temporal_links}")

    return model, tokenizer


def demonstrate_multihop_reasoning(model, tokenizer):
    """Demonstrate multi-hop reasoning with example questions.

    Args:
        model: The MA-XLNet model
        tokenizer: The tokenizer
    """
    print("\n📊 Demonstrating Multi-Hop Reasoning...")

    # Example multi-hop question
    context = """
    The Eiffel Tower is located in Paris, France. It was completed in 1889 and
    stands 324 meters tall. Paris is the capital city of France and has a
    population of over 2 million people. France is a country in Western Europe
    with a total population of 67 million. The current President of France is
    Emmanuel Macron, who took office in 2017.
    """

    questions = [
        "What is the population of the city where the Eiffel Tower is located?",
        "Who is the president of the country containing the Eiffel Tower?",
        "In what year did the current French president take office?",
    ]

    # Initialize hop tracker
    tracker = HopTracker(track_attention=True, track_content=True)

    for question in questions:
        print(f"\n❓ Question: {question}")

        # Tokenize input
        inputs = tokenizer(question, context, max_length=384, truncation=True, padding=True, return_tensors="pt")

        # Reset tracker for new question
        tracker.reset()

        # Forward pass with memory tracking
        with torch.no_grad():
            outputs = model(**inputs)

            # Track memory operations if using differentiable memory
            if hasattr(outputs, "memory_info") and outputs.get("memory_info"):
                tracker.record_hop(
                    outputs["memory_info"], question_part="Initial hop", extracted_info="Processing question"
                )

        # Get answer span
        start_idx = outputs["start_logits"].argmax().item()
        end_idx = outputs["end_logits"].argmax().item()

        # Decode answer
        answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        print(f"✅ Answer: {answer}")

        # Get reasoning chain
        chain = tracker.get_reasoning_chain(question, answer)
        print(f"   Confidence: {chain.total_confidence:.2f}")
        print(f"   Hops used: {len(chain.hops)}")


def train_with_enhanced_memory():
    """Train MA-XLNet with enhanced memory configuration."""
    print("\n🎯 Training MA-XLNet with Enhanced Memory...")

    # Create configuration with enhanced memory
    config = TrainingConfig(
        model_name="xlnet-base-cased",
        dataset_name="squad_v2",
        # Standard memory settings
        memory_num_tokens=32,
        memory_update="gated",
        memory_init="learned",
        # Enhanced memory settings
        use_differentiable_memory=True,  # Enable differentiable memory
        num_memory_heads=4,  # Multi-head attention
        memory_sharpness=2.0,  # Sharper attention
        enable_usage_tracking=True,  # Track slot usage
        enable_temporal_links=True,  # Track temporal patterns
        memory_slots=64,  # More memory slots
        # Training settings
        num_epochs=1,  # Quick demo
        max_train_samples=100,  # Small subset for demo
        max_eval_samples=50,
        train_batch_size=2,
        eval_batch_size=2,
        output_dir="./outputs/enhanced_ma_xlnet_demo",
        run_name="enhanced_ma_xlnet_demo",
    )

    # Create trainer
    trainer = XLNetRecurrentTrainer(config)

    # Prepare data
    print("\n📚 Preparing data...")
    train_dataloader = create_dataloader(
        dataset_name=config.dataset_name,
        tokenizer=trainer.tokenizer,
        split=config.train_split,
        max_seq_length=config.max_seq_length,
        doc_stride=config.doc_stride,
        batch_size=config.train_batch_size,
        max_samples=config.max_train_samples,
        cache_dir=config.cache_dir,
        is_training=True,
    )

    eval_dataloader = create_dataloader(
        dataset_name=config.dataset_name,
        tokenizer=trainer.tokenizer,
        split=config.eval_split,
        max_seq_length=config.max_seq_length,
        doc_stride=config.doc_stride,
        batch_size=config.eval_batch_size,
        max_samples=config.max_eval_samples,
        cache_dir=config.cache_dir,
        is_training=False,
    )

    # Train model
    print("\n🏋️ Training model...")
    trainer.train(train_dataloader, eval_dataloader)

    print("\n✅ Training complete!")
    return trainer.model


def visualize_memory_operations(model):
    """Visualize memory operations during inference.

    Args:
        model: The MA-XLNet model with differentiable memory
    """
    print("\n🎨 Visualizing Memory Operations...")

    if not model.use_differentiable_memory or model.memory_controller is None:
        print("⚠️ Model doesn't use differentiable memory. Skipping visualization.")
        return

    # Get memory visualization data
    viz_data = model.memory_controller.visualize_memory()

    # Create visualizer
    visualizer = MemoryVisualizer()

    # Plot memory state heatmap
    print("📊 Creating memory state heatmap...")
    fig = visualizer.plot_memory_heatmap(viz_data["memory"], title="Current Memory State")
    # Save figure
    fig.savefig("memory_state.png", dpi=150, bbox_inches="tight")
    print("   Saved to memory_state.png")

    # Plot usage pattern if available
    if "usage" in viz_data:
        print("📊 Creating usage pattern plot...")
        fig = visualizer.plot_usage_pattern(viz_data["usage"], viz_data.get("temporal_links"))
        fig.savefig("memory_usage.png", dpi=150, bbox_inches="tight")
        print("   Saved to memory_usage.png")


def evaluate_multihop_performance(model, tokenizer, eval_dataloader):
    """Evaluate model performance on multi-hop questions.

    Args:
        model: The MA-XLNet model
        tokenizer: The tokenizer
        eval_dataloader: Evaluation data loader
    """
    print("\n📈 Evaluating Multi-Hop Performance...")

    model.eval()
    tracker = HopTracker()
    predicted_chains = []

    # Process a few examples
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if i >= 5:  # Just a few examples for demo
                break

            # Move batch to device
            batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)

            # Track reasoning
            if hasattr(outputs, "memory_info") and outputs.get("memory_info"):
                tracker.record_hop(outputs["memory_info"])

            # Get predictions
            start_idx = outputs["start_logits"].argmax(dim=-1)

            # Create reasoning chains
            for j in range(len(start_idx)):
                chain = tracker.get_reasoning_chain(
                    question="Example question",  # Would get from batch
                    answer="Example answer",  # Would decode from predictions
                )
                predicted_chains.append(chain)

    # Calculate metrics
    if predicted_chains:
        avg_confidence = sum(c.total_confidence for c in predicted_chains) / len(predicted_chains)
        avg_hops = sum(len(c.hops) for c in predicted_chains) / len(predicted_chains)

        print("\n📊 Results:")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Average hops: {avg_hops:.1f}")
        print(f"   Success rate: {sum(c.success for c in predicted_chains) / len(predicted_chains):.1%}")


def main():
    """Main demonstration script."""
    print("=" * 60)
    print("MA-XLNet Enhanced Memory Multi-Hop QA Demonstration")
    print("=" * 60)

    # Configuration options
    print("\n🔧 Configuration Options:")
    print("1. Token-based memory (backward compatible)")
    print("2. Differentiable memory (enhanced)")
    print("3. Hybrid mode (both systems)")

    choice = input("\nSelect configuration (1-3): ").strip()

    # Create configuration
    config = TrainingConfig()

    if choice == "1":
        # Token-based (existing)
        config.use_differentiable_memory = False
        config.memory_num_tokens = 32
        print("\n✅ Using token-based memory (backward compatible)")

    elif choice == "2":
        # Differentiable memory
        config.use_differentiable_memory = True
        config.num_memory_heads = 4
        config.memory_sharpness = 2.0
        config.enable_usage_tracking = True
        config.enable_temporal_links = True
        config.memory_slots = 64
        print("\n✅ Using differentiable memory (enhanced)")

    elif choice == "3":
        # Hybrid (both)
        config.memory_num_tokens = 16
        config.use_differentiable_memory = True
        config.num_memory_heads = 2
        config.memory_slots = 32
        print("\n✅ Using hybrid configuration")

    else:
        print("Invalid choice. Using default configuration.")

    # Create model
    model, tokenizer = create_enhanced_model(config)

    # Demonstrate reasoning
    demonstrate_multihop_reasoning(model, tokenizer)

    # Optionally train
    train_choice = input("\n🏋️ Train the model? (y/n): ").strip().lower()
    if train_choice == "y":
        model = train_with_enhanced_memory()

    # Visualize memory if using differentiable memory
    if config.use_differentiable_memory:
        try:
            visualize_memory_operations(model)
        except ImportError:
            print("⚠️ Visualization requires matplotlib and seaborn. Skipping.")

    print("\n🎉 Demonstration complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Backward compatible token-based memory")
    print("✅ Enhanced differentiable memory with content-based addressing")
    print("✅ Multi-head read/write operations")
    print("✅ Memory usage tracking and temporal links")
    print("✅ Reasoning chain tracking and visualization")
    print("✅ Multi-hop question answering")


if __name__ == "__main__":
    main()
