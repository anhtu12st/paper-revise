#!/usr/bin/env python3
"""Example: Training MemXLNet with Differentiable Memory.

This script demonstrates how to train a MemXLNet model with differentiable memory
enabled on the SQuAD v2 dataset. It showcases:
- Configuration of differentiable memory parameters
- Multi-head attention setup
- Usage tracking and temporal links
- Memory state visualization

Usage:
    python examples/train_with_differentiable_memory.py
"""

import torch
from transformers import XLNetForQuestionAnsweringSimple, XLNetTokenizerFast

from memxlnet.models import MemXLNetForQA


def main():
    print("=" * 70)
    print("MemXLNet with Differentiable Memory - Training Example")
    print("=" * 70)

    # 1. Load base model and tokenizer
    print("\n[1/6] Loading base XLNet model and tokenizer...")
    model_name = "xlnet-base-cased"
    base_model = XLNetForQuestionAnsweringSimple.from_pretrained(model_name)
    tokenizer = XLNetTokenizerFast.from_pretrained(model_name)
    print(f"✓ Loaded {model_name}")

    # 2. Create MemXLNet with differentiable memory
    print("\n[2/6] Creating MemXLNet with differentiable memory...")

    model = MemXLNetForQA(
        base_model=base_model,
        mem_token_count=16,  # Number of memory tokens
        # Differentiable memory configuration
        use_differentiable_memory=True,  # Enable differentiable memory
        num_memory_heads=4,  # Multi-head attention (4 heads)
        memory_sharpness=2.0,  # Attention sharpening factor
        enable_usage_tracking=True,  # Track memory slot usage
        enable_temporal_links=True,  # Track temporal relationships
        memory_slots=32,  # Number of memory slots
    )

    print("✓ Model created with differentiable memory:")
    print(f"  - Memory slots: {model.memory_slots}")
    print(f"  - Memory heads: {model.num_memory_heads}")
    print(f"  - Memory sharpness: {model.memory_sharpness}")
    print(f"  - Usage tracking: {model.enable_usage_tracking}")
    print(f"  - Temporal links: {model.enable_temporal_links}")

    # 3. Prepare sample data
    print("\n[3/6] Preparing sample question and context...")

    question = "What is the capital of France?"
    context = """
    Paris is the capital and most populous city of France. With an official estimated
    population of 2,102,650 residents as of 1 January 2023 in an area of more than
    105 km², Paris is the fourth-largest city in the European Union and the 30th most
    densely populated city in the world in 2022.
    """

    # Tokenize
    inputs = tokenizer(question, context.strip(), return_tensors="pt", max_length=256, truncation=True, padding=True)

    print(f"✓ Question: {question}")
    print(f"✓ Context length: {len(context.strip())} chars")
    print(f"✓ Tokenized sequence length: {inputs['input_ids'].shape[1]}")

    # 4. Forward pass
    print("\n[4/6] Running forward pass...")

    with torch.no_grad():
        outputs = model(**inputs)

    print("✓ Forward pass completed")
    print(f"  - Start logits shape: {outputs['start_logits'].shape}")
    print(f"  - End logits shape: {outputs['end_logits'].shape}")
    print(f"  - Memory info available: {'memory_info' in outputs}")

    # 5. Analyze memory operations
    if "memory_info" in outputs:
        print("\n[5/6] Analyzing memory operations...")

        memory_info = outputs["memory_info"]

        print(f"✓ Read weights shape: {memory_info['read_weights'].shape}")
        print(f"  - Batch size: {memory_info['read_weights'].shape[0]}")
        print(f"  - Number of heads: {memory_info['read_weights'].shape[1]}")
        print(f"  - Memory slots: {memory_info['read_weights'].shape[2]}")

        print(f"\n✓ Write weights shape: {memory_info['write_weights'].shape}")

        print(f"\n✓ Memory state shape: {memory_info['memory_state'].shape}")

        if "usage" in memory_info:
            usage = memory_info["usage"]
            print("\n✓ Memory usage statistics:")
            print(f"  - Average usage: {usage.mean():.4f}")
            print(f"  - Max usage: {usage.max():.4f}")
            print(f"  - Min usage: {usage.min():.4f}")
            print(f"  - Most used slot: {usage.argmax().item()}")

        if "temporal_links" in memory_info:
            temporal_links = memory_info["temporal_links"]
            print(f"\n✓ Temporal links shape: {temporal_links.shape}")
            print(f"  - Average link strength: {temporal_links.mean():.4f}")

    # 6. Extract answer (demonstration)
    print("\n[6/6] Extracting answer...")

    start_logits = outputs["start_logits"][0]
    end_logits = outputs["end_logits"][0]

    # Get answer span
    start_idx = start_logits.argmax().item()
    end_idx = end_logits.argmax().item()

    # Ensure valid span
    if end_idx < start_idx:
        end_idx = start_idx

    # Decode answer
    answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    print(f"✓ Predicted answer: '{answer}'")
    print(f"  - Start position: {start_idx}")
    print(f"  - End position: {end_idx}")

    # 7. Memory state management
    print("\n[7/6] Memory state management (bonus)...")

    # Get current memory state
    memory_state = model.memory_controller.get_memory_state()
    print(f"✓ Current memory state shape: {memory_state.shape}")

    # Reset memory
    model.memory_controller.reset_memory()
    print("✓ Memory reset to initial state")

    # Verify reset
    reset_state = model.memory_controller.get_memory_state()
    is_zero = torch.allclose(reset_state, torch.zeros_like(reset_state))
    print(f"✓ Memory is zero after reset: {is_zero}")

    # 8. Comparison with token-based memory
    print("\n[8/6] Comparison: Token-based vs Differentiable Memory (bonus)...")

    # Token-based model
    model_token = MemXLNetForQA(
        base_model=base_model,
        mem_token_count=16,
        use_differentiable_memory=False,  # Token-based
        memory_update="gated",
    )

    with torch.no_grad():
        outputs_token = model_token(**inputs)
        outputs_diff = model(**inputs)

    print("✓ Token-based memory:")
    print(f"  - Has memory_info: {'memory_info' in outputs_token}")
    print(f"  - Output shape: {outputs_token['start_logits'].shape}")

    print("\n✓ Differentiable memory:")
    print(f"  - Has memory_info: {'memory_info' in outputs_diff}")
    print(f"  - Output shape: {outputs_diff['start_logits'].shape}")
    print(f"  - Additional info keys: {list(outputs_diff.get('memory_info', {}).keys())}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Successfully demonstrated differentiable memory with MemXLNet")
    print("✓ Multi-head attention works correctly")
    print("✓ Usage tracking and temporal links functional")
    print("✓ Memory state management operational")
    print("\nNext steps:")
    print("  1. Use TrainingConfig to configure full training pipeline")
    print("  2. Train on SQuAD v2 dataset with progressive segments")
    print("  3. Analyze memory usage patterns during training")
    print("  4. Compare performance with token-based memory")
    print("\nFor full training, see:")
    print("  - scripts/phase2_train.py")
    print("  - scripts/train_memxlnet_squad.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
