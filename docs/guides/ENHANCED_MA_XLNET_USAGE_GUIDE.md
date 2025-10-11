# Enhanced MA-XLNet Usage Guide

> **✅ Feature Availability Notice (Updated January 2025):**
>
> This guide describes both **implemented** and **planned** features for MemXLNet-QA.
>
> **✅ Currently Available:**
> - Token-based memory
> - **Differentiable memory with MemoryController** ✅ **NEW!**
> - **Multi-head attention** ✅ **NEW!**
> - **Usage tracking and temporal links** ✅ **NEW!**
> - Basic training and configuration
> - Progressive training
>
> **🚧 Planned Features (Not Yet Implemented):**
> - HopTracker and multi-hop reasoning utilities
> - MemoryVisualizer and attention visualization
> - Advanced memory features (adaptive allocation, compression, etc.)
>
> See [PLANNED_FEATURES.md](../PLANNED_FEATURES.md) for complete implementation status.

## Overview

This guide covers how to use the Memory-Augmented XLNet (MA-XLNet) for question answering. It includes documentation for both currently implemented features and planned enhancements for complex reasoning tasks.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Options](#configuration-options)
3. [Basic Usage Examples](#basic-usage-examples)
4. [Advanced Features](#advanced-features)
5. [Multi-Hop Reasoning](#multi-hop-reasoning)
6. [Memory Visualization](#memory-visualization)
7. [Training Guide](#training-guide)
8. [Migration from Token-Based Memory](#migration-from-token-based-memory)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation Requirements

```bash
# Core dependencies
pip install torch transformers datasets

# Optional for visualization
pip install matplotlib seaborn

# Optional for development
pip install pytest
```

### Basic Setup (✅ Available Now)

```python
from memxlnet.training import TrainingConfig
from memxlnet.models import MemXLNetForQA
from transformers import XLNetForQuestionAnsweringSimple, XLNetTokenizerFast

# Load base model and tokenizer
base_model = XLNetForQuestionAnsweringSimple.from_pretrained('xlnet-base-cased')
tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')

# Create model with token-based memory (✅ Available)
model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,
    memory_init="learned",
    memory_update="gated"
)
```

### Enhanced Setup with Differentiable Memory (✅ Available Now!)

```python
# Differentiable memory is now fully implemented and tested!

model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,
    use_differentiable_memory=True,  # ✅ Fully functional
    num_memory_heads=4,               # ✅ Multi-head attention working
    memory_sharpness=2.0,             # ✅ Attention sharpening working
    enable_usage_tracking=True,       # ✅ Usage tracking operational
    enable_temporal_links=True,       # ✅ Temporal links functional
    memory_slots=64                   # ✅ Configurable memory slots
)

# See examples/train_with_differentiable_memory.py for a complete working example
```

## Configuration Options

### Memory Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_differentiable_memory` | bool | `False` | Enable differentiable memory (vs token-based) |
| `num_memory_heads` | int | `1` | Number of read/write attention heads |
| `memory_sharpness` | float | `1.0` | Temperature for attention sharpening |
| `enable_usage_tracking` | bool | `False` | Track memory slot usage for allocation |
| `enable_temporal_links` | bool | `False` | Track temporal relationships between slots |
| `memory_slots` | int | `None` | Number of memory slots (auto if None) |

### TrainingConfig Integration

```python
from memxlnet.training import TrainingConfig

# Enhanced memory configuration
config = TrainingConfig(
    model_name="xlnet-base-cased",

    # Standard memory settings
    memory_num_tokens=32,
    memory_update="gated",
    memory_init="learned",

    # Enhanced memory settings
    use_differentiable_memory=True,
    num_memory_heads=4,
    memory_sharpness=2.0,
    enable_usage_tracking=True,
    enable_temporal_links=True,
    memory_slots=64,

    # Training settings
    num_epochs=3,
    train_batch_size=4,
    learning_rate=3e-5
)
```

## Basic Usage Examples

### 1. Token-Based Memory (Backward Compatible)

```python
# Create traditional token-based model
model_token = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=16,
    memory_init="learned",
    memory_update="gated",
    use_differentiable_memory=False  # Traditional approach
)

# Usage is identical to existing code
inputs = tokenizer("What is the capital of France?",
                  "Paris is the capital of France.",
                  return_tensors="pt")

outputs = model_token(**inputs)
# Output: {'loss': None, 'start_logits': ..., 'end_logits': ..., 'new_memory_state': ...}
```

### 2. Enhanced Differentiable Memory

```python
# Create enhanced model
model_enhanced = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=16,
    use_differentiable_memory=True,
    num_memory_heads=2,
    memory_sharpness=1.5,
    enable_usage_tracking=True
)

# Enhanced usage with memory info
outputs = model_enhanced(**inputs)
# Output includes additional 'memory_info' key with detailed memory operations

if 'memory_info' in outputs:
    memory_info = outputs['memory_info']
    print(f"Read weights shape: {memory_info['read_weights'].shape}")
    print(f"Memory usage: {memory_info['usage']}")
```

### 3. Multi-Hop Question Example

```python
# Multi-hop context
context = """
Paris is the capital of France. The Eiffel Tower is located in Paris and was
completed in 1889. France has a population of 67 million people. Emmanuel Macron
is the current President of France, taking office in 2017.
"""

questions = [
    "What is the population of the country where the Eiffel Tower is located?",
    "When did the current French president take office?",
    "In which city is the Eiffel Tower located?"
]

# Process multiple related questions
for question in questions:
    inputs = tokenizer(question, context, return_tensors="pt",
                      max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model_enhanced(**inputs)

        # Extract answer
        start_idx = outputs['start_logits'].argmax().item()
        end_idx = outputs['end_logits'].argmax().item()

        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        print(f"Q: {question}")
        print(f"A: {answer}")

        # Analyze memory usage if available
        if 'memory_info' in outputs:
            usage = outputs['memory_info']['usage'].numpy()
            print(f"Memory utilization: {usage.mean():.3f}")
        print("-" * 50)
```

## Advanced Features

### 1. Memory State Management

```python
# Get current memory state
memory_state = model_enhanced.memory_controller.get_memory_state()
print(f"Memory state shape: {memory_state.shape}")

# Set custom memory state
custom_memory = torch.randn_like(memory_state)
model_enhanced.memory_controller.set_memory_state(custom_memory)

# Reset memory to initial state
model_enhanced.memory_controller.reset_memory()
```

### 2. Multi-Head Memory Configuration

```python
# Configure different head specializations
model_multihead = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=16,
    use_differentiable_memory=True,
    num_memory_heads=8,  # More heads for complex reasoning
    memory_sharpness=3.0,  # Sharper attention
    memory_slots=128  # More memory capacity
)

# Each head can specialize in different types of information
# Head 1: Entity tracking
# Head 2: Temporal relationships
# Head 3: Causal connections
# etc.
```

### 3. Dynamic Memory Allocation

```python
# Model with usage tracking for optimal slot allocation
model_dynamic = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,
    use_differentiable_memory=True,
    enable_usage_tracking=True,  # Track which slots are used
    enable_temporal_links=True,  # Track slot relationships
    memory_slots=64
)

# The model will automatically allocate less-used slots for new information
```

## Multi-Hop Reasoning (🚧 Planned - Not Yet Implemented)

> **⚠️ Note:** The utilities described in this section are planned but not yet implemented.
> The imports will fail with `ModuleNotFoundError`. See [PLANNED_FEATURES.md](../PLANNED_FEATURES.md).

### 1. Reasoning Chain Tracking (🚧 Planned)

```python
# 🚧 NOT YET AVAILABLE - This will fail with ImportError
from memxlnet.utils.multihop_utils import HopTracker

# Initialize hop tracker
tracker = HopTracker(track_attention=True, track_content=True)

# Process question with tracking
tracker.reset()

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model_enhanced(**inputs)

# Record reasoning hop
if 'memory_info' in outputs:
    tracker.record_hop(
        outputs['memory_info'],
        question_part="Identifying bridge entity",
        extracted_info="Found Eiffel Tower in Paris"
    )

# Get complete reasoning chain
chain = tracker.get_reasoning_chain(question, answer)
print(f"Reasoning confidence: {chain.total_confidence:.3f}")
print(f"Number of hops: {len(chain.hops)}")
```

### 2. Bridge Entity Detection

```python
# Example of tracking bridge entities through memory
def track_bridge_entities(model, tokenizer, question, context):
    """Track entities that bridge different pieces of information."""

    inputs = tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

        if 'memory_info' in outputs:
            # Analyze attention patterns to identify bridge entities
            read_weights = outputs['memory_info']['read_weights']
            write_weights = outputs['memory_info']['write_weights']

            # High attention on both read and write suggests bridge entity
            bridge_scores = (read_weights * write_weights).sum(dim=1)

            return bridge_scores

    return None

# Usage
bridge_scores = track_bridge_entities(model_enhanced, tokenizer, question, context)
if bridge_scores is not None:
    print(f"Bridge entity scores: {bridge_scores}")
```

## Memory Visualization (🚧 Planned - Not Yet Implemented)

> **⚠️ Note:** The visualization utilities described in this section are planned but not yet implemented.
> The imports will fail with `ModuleNotFoundError`. See [PLANNED_FEATURES.md](../PLANNED_FEATURES.md).

### 1. Memory State Visualization (🚧 Planned)

```python
# 🚧 NOT YET AVAILABLE - This will fail with ImportError
from memxlnet.utils.multihop_utils import MemoryVisualizer

# Create visualizer
visualizer = MemoryVisualizer()

# Get memory data
if model_enhanced.memory_controller:
    viz_data = model_enhanced.memory_controller.visualize_memory()

    # Create memory heatmap
    fig = visualizer.plot_memory_heatmap(
        viz_data['memory'],
        title="Current Memory State"
    )
    if fig:
        fig.savefig("memory_state.png")

    # Plot usage patterns
    if 'usage' in viz_data:
        fig = visualizer.plot_usage_pattern(
            viz_data['usage'],
            viz_data.get('temporal_links')
        )
        if fig:
            fig.savefig("memory_usage.png")
```

### 2. Attention Flow Visualization

```python
# Visualize attention flow across reasoning hops
if 'memory_info' in outputs:
    tracker.record_hop(outputs['memory_info'])

    # Generate attention flow visualization
    fig = tracker.visualize_attention_flow()
    if fig:
        fig.savefig("attention_flow.png")
```

## Training Guide

### 1. Basic Training Setup

```python
from memxlnet.training import XLNetRecurrentTrainer

# Create enhanced configuration
config = TrainingConfig(
    model_name="xlnet-base-cased",
    dataset_name="squad_v2",

    # Enhanced memory settings
    use_differentiable_memory=True,
    num_memory_heads=4,
    memory_sharpness=2.0,
    enable_usage_tracking=True,
    memory_slots=64,

    # Training parameters
    num_epochs=3,
    train_batch_size=4,
    eval_batch_size=4,
    learning_rate=3e-5,
    output_dir="./outputs/enhanced_ma_xlnet"
)

# Create trainer
trainer = XLNetRecurrentTrainer(config)

# Training data will automatically use enhanced memory features
```

### 2. Progressive Memory Training

```python
# Start with simple memory, gradually increase complexity
configs = [
    # Stage 1: Basic differentiable memory
    TrainingConfig(
        use_differentiable_memory=True,
        num_memory_heads=1,
        memory_slots=16,
        num_epochs=1
    ),

    # Stage 2: Multi-head attention
    TrainingConfig(
        use_differentiable_memory=True,
        num_memory_heads=4,
        memory_slots=32,
        num_epochs=2
    ),

    # Stage 3: Full features
    TrainingConfig(
        use_differentiable_memory=True,
        num_memory_heads=8,
        memory_slots=64,
        enable_usage_tracking=True,
        enable_temporal_links=True,
        num_epochs=3
    )
]

# Train progressively
for i, config in enumerate(configs):
    print(f"Training stage {i+1}")
    trainer = XLNetRecurrentTrainer(config)
    # trainer.train(train_loader, eval_loader)
```

### 3. Custom Memory Initialization

```python
# Initialize memory with domain-specific knowledge
def init_domain_memory(model, domain_embeddings):
    """Initialize memory with domain-specific embeddings."""
    if model.memory_controller:
        # Set initial memory state
        memory_state = domain_embeddings  # Shape: (memory_slots, memory_dim)
        model.memory_controller.set_memory_state(memory_state)

# Example usage
domain_embeddings = torch.randn(64, 768)  # 64 slots, 768 dimensions
init_domain_memory(model_enhanced, domain_embeddings)
```

## Migration from Token-Based Memory

### 1. Backward Compatibility

```python
# Existing code continues to work unchanged
old_model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=16,
    memory_init="learned",
    memory_update="gated"
    # No new parameters - uses defaults (backward compatible)
)

# Same API, same outputs
outputs = old_model(**inputs)
# Output: {'loss': None, 'start_logits': ..., 'end_logits': ..., 'new_memory_state': ...}
```

### 2. Gradual Migration

```python
# Step 1: Enable differentiable memory with minimal changes
model_step1 = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=16,
    use_differentiable_memory=True  # Only change this
)

# Step 2: Add multi-head attention
model_step2 = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=16,
    use_differentiable_memory=True,
    num_memory_heads=2  # Add multiple heads
)

# Step 3: Full enhancement
model_step3 = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=16,
    use_differentiable_memory=True,
    num_memory_heads=4,
    memory_sharpness=2.0,
    enable_usage_tracking=True,
    enable_temporal_links=True
)
```

### 3. A/B Testing

```python
# Compare token-based vs differentiable memory
def compare_models(question, context):
    """Compare token-based and differentiable memory models."""

    inputs = tokenizer(question, context, return_tensors="pt")

    # Token-based model
    with torch.no_grad():
        outputs_token = model_token(**inputs)

    # Differentiable memory model
    with torch.no_grad():
        outputs_diff = model_enhanced(**inputs)

    # Compare results
    return {
        'token_based': {
            'start_logits': outputs_token['start_logits'],
            'end_logits': outputs_token['end_logits']
        },
        'differentiable': {
            'start_logits': outputs_diff['start_logits'],
            'end_logits': outputs_diff['end_logits'],
            'memory_info': outputs_diff.get('memory_info')
        }
    }
```

## Performance Tuning

### 1. Memory Configuration Tuning

```python
# Memory slots vs performance trade-off
configs = [
    {'memory_slots': 16, 'description': 'Minimal memory, fastest'},
    {'memory_slots': 32, 'description': 'Balanced'},
    {'memory_slots': 64, 'description': 'High capacity'},
    {'memory_slots': 128, 'description': 'Maximum capacity, slower'}
]

for config in configs:
    model = MemXLNetForQA(
        base_model=base_model,
        use_differentiable_memory=True,
        memory_slots=config['memory_slots']
    )
    # Benchmark performance
```

### 2. Attention Sharpness Tuning

```python
# Tune attention sharpness for your task
sharpness_values = [0.5, 1.0, 1.5, 2.0, 3.0]

for sharpness in sharpness_values:
    model = MemXLNetForQA(
        base_model=base_model,
        use_differentiable_memory=True,
        memory_sharpness=sharpness
    )

    # Test on validation set
    # Higher sharpness = more focused attention
    # Lower sharpness = more distributed attention
```

### 3. Multi-Head Configuration

```python
# Optimize number of heads for your task complexity
head_configs = [
    {'heads': 1, 'use_case': 'Simple QA'},
    {'heads': 2, 'use_case': 'Basic multi-hop'},
    {'heads': 4, 'use_case': 'Complex multi-hop'},
    {'heads': 8, 'use_case': 'Very complex reasoning'}
]

for config in head_configs:
    model = MemXLNetForQA(
        base_model=base_model,
        use_differentiable_memory=True,
        num_memory_heads=config['heads']
    )
    # Test and compare
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Information Not Available

**Problem:** `memory_info` not in outputs
```python
# Check if differentiable memory is enabled
if not model.use_differentiable_memory:
    print("Enable differentiable memory: use_differentiable_memory=True")

# Check if memory controller exists
if model.memory_controller is None:
    print("Memory controller not initialized. Check configuration.")
```

#### 2. CUDA Memory Issues

**Problem:** Out of memory errors
```python
# Reduce memory slots
model = MemXLNetForQA(
    base_model=base_model,
    use_differentiable_memory=True,
    memory_slots=16,  # Reduce from 64
    num_memory_heads=2  # Reduce from 4
)

# Use gradient checkpointing
config.gradient_checkpointing = True
```

#### 3. Slow Training

**Problem:** Training is slower than expected
```python
# Profile memory operations
import time

start_time = time.time()
outputs = model(**inputs)
end_time = time.time()

print(f"Forward pass time: {end_time - start_time:.3f}s")

# Consider reducing complexity
model = MemXLNetForQA(
    base_model=base_model,
    use_differentiable_memory=True,
    num_memory_heads=2,  # Reduce heads
    memory_slots=32,     # Reduce slots
    enable_usage_tracking=False,  # Disable if not needed
    enable_temporal_links=False   # Disable if not needed
)
```

#### 4. Visualization Issues

**Problem:** Visualization functions not working
```python
# Install visualization dependencies
# pip install matplotlib seaborn

# Check if available
from memxlnet.utils.multihop_utils import VISUALIZATION_AVAILABLE
if not VISUALIZATION_AVAILABLE:
    print("Install visualization dependencies: pip install matplotlib seaborn")
```

#### 5. Save/Load Issues

**Problem:** Model not loading correctly
```python
# Check saved configuration
import json
with open("model_dir/memxlnet_config.json", "r") as f:
    config = json.load(f)
    print(f"Saved version: {config.get('version')}")
    print(f"Differentiable memory: {config.get('use_differentiable_memory')}")

# Load with explicit parameters
model = MemXLNetForQA.from_pretrained(
    "model_dir",
    use_differentiable_memory=True,  # Override if needed
    num_memory_heads=4
)
```

## Best Practices

### 1. Configuration Selection

- **Simple QA tasks:** Use token-based memory or minimal differentiable memory
- **Multi-hop QA:** Use differentiable memory with 2-4 heads
- **Complex reasoning:** Use full features with usage tracking and temporal links
- **Resource-constrained:** Start with 16-32 memory slots

### 2. Training Strategy

- **Progressive training:** Start simple, gradually add complexity
- **Curriculum learning:** Train on simple examples first
- **Memory warm-up:** Allow model to learn memory usage patterns

### 3. Evaluation

- **Compare with baseline:** Always compare against token-based memory
- **Memory utilization:** Monitor memory usage patterns
- **Reasoning analysis:** Use hop tracking to understand model behavior

### 4. Production Deployment

- **A/B testing:** Gradually roll out enhanced memory
- **Performance monitoring:** Track inference speed and memory usage
- **Fallback strategy:** Keep token-based models as backup

## Conclusion

The enhanced MA-XLNet provides powerful new capabilities for multi-hop question answering while maintaining full backward compatibility. Start with basic configurations and gradually explore advanced features as needed for your specific use case.

For more examples and usage patterns, see:
- `examples/validate_answer_spans.py` - Answer span validation examples
- `tests/unit/test_answer_span_validation.py` - Answer span validation tests
- `tests/unit/test_multi_segment_answers.py` - Multi-segment answer tests