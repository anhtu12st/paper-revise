# Enhanced MA-XLNet Quick Reference

> **⚠️ Feature Status:** This guide includes both ✅ **available** and 🚧 **planned** features.
> See [PLANNED_FEATURES.md](../PLANNED_FEATURES.md) for details.

## Cheat Sheet for Developers

### 🚀 Quick Setup (✅ Available)

```python
from memxlnet.models import MemXLNetForQA
from transformers import XLNetForQuestionAnsweringSimple

# Load base model
base = XLNetForQuestionAnsweringSimple.from_pretrained('xlnet-base-cased')

# Token-based model (✅ Available)
model = MemXLNetForQA(
    base_model=base,
    mem_token_count=32,
    memory_init="learned",
    memory_update="gated"
)
```

### 🔮 Enhanced Setup (✅ Available - Phase 1 & 2 Complete)

```python
# Fully implemented differentiable memory features
model = MemXLNetForQA(
    base_model=base,
    mem_token_count=32,
    use_differentiable_memory=True,  # ✅ Available
    num_memory_heads=4,               # ✅ Available
    memory_sharpness=2.0,             # ✅ Available
    memory_slots=64                   # ✅ Available
)
```

### 📋 Configuration Cheat Sheet

| Use Case | Config | Status |
|----------|--------|--------|
| **Standard (Token-Based)** | `mem_token_count=32, memory_update="gated"` | ✅ Available |
| **Progressive Training** | `progressive_segments=[2,4,6]` | ✅ Available |
| **Lazy Loading** | Use `LazySquadLikeQADataset` | ✅ Available |
| **Streaming** | Use `StreamingSquadProcessor` | ✅ Available |
| **Basic Enhanced** | `use_differentiable_memory=True, num_memory_heads=2` | ✅ Available |
| **Multi-hop QA** | `use_differentiable_memory=True, num_memory_heads=4` | ✅ Available |

### 🔧 Training Configuration

```python
from memxlnet.training import TrainingConfig

config = TrainingConfig(
    # Model
    model_name="xlnet-base-cased",

    # Enhanced Memory (NEW)
    use_differentiable_memory=True,
    num_memory_heads=4,
    memory_sharpness=2.0,
    memory_slots=64,

    # Standard Settings
    memory_num_tokens=32,
    num_epochs=3,
    train_batch_size=4
)
```

### 🧠 Memory Features Quick Reference

| Feature | Parameter | Description |
|---------|-----------|-------------|
| **Differentiable Memory** | `use_differentiable_memory=True` | Content-based addressing |
| **Multi-head Attention** | `num_memory_heads=4` | Parallel memory operations |
| **Attention Sharpening** | `memory_sharpness=2.0` | Focus attention (1.0-3.0) |
| **Usage Tracking** | `enable_usage_tracking=True` | Track slot allocation |
| **Temporal Links** | `enable_temporal_links=True` | Sequential relationships |
| **Memory Capacity** | `memory_slots=64` | Number of memory slots |

### 📊 Output Structure

```python
# Token-based output
{
    'loss': tensor,
    'start_logits': tensor,
    'end_logits': tensor,
    'new_memory_state': tensor
}

# Enhanced memory output (additional)
{
    'loss': tensor,
    'start_logits': tensor,
    'end_logits': tensor,
    'new_memory_state': tensor,
    'memory_info': {
        'read_weights': tensor,
        'write_weights': tensor,
        'read_vectors': tensor,
        'write_vectors': tensor,
        'memory_state': tensor,
        'usage': tensor,           # if usage_tracking=True
        'temporal_links': tensor   # if temporal_links=True
    }
}
```

### 🔍 Multi-hop Reasoning

```python
from memxlnet.utils.multihop_utils import HopTracker

# Track reasoning chains
tracker = HopTracker()
tracker.record_hop(outputs['memory_info'])
chain = tracker.get_reasoning_chain(question, answer)

print(f"Confidence: {chain.total_confidence:.3f}")
print(f"Hops: {len(chain.hops)}")
```

### 📈 Performance Guidelines

| Memory Slots | Use Case | Performance |
|--------------|----------|-------------|
| 16 | Simple QA | Fastest |
| 32 | Basic multi-hop | Balanced |
| 64 | Complex reasoning | Good |
| 128+ | Very complex | Slower |

| Memory Heads | Complexity | When to Use |
|--------------|------------|-------------|
| 1 | Simple | Single-hop questions |
| 2-4 | Medium | Multi-hop questions |
| 4-8 | High | Complex reasoning |
| 8+ | Very High | Research/experimentation |

### 🛠️ Common Patterns

#### Pattern 1: Basic Usage
```python
# Simple enhanced model
model = MemXLNetForQA(
    base_model=base,
    use_differentiable_memory=True
)
```

#### Pattern 2: Production Ready
```python
# Balanced for production
model = MemXLNetForQA(
    base_model=base,
    mem_token_count=32,
    use_differentiable_memory=True,
    num_memory_heads=4,
    memory_slots=64
)
```

#### Pattern 3: Research/Experimentation
```python
# Full features for research
model = MemXLNetForQA(
    base_model=base,
    mem_token_count=32,
    use_differentiable_memory=True,
    num_memory_heads=8,
    memory_sharpness=2.5,
    enable_usage_tracking=True,
    enable_temporal_links=True,
    memory_slots=128
)
```

### 🔧 Migration Examples

#### From Token-based to Enhanced
```python
# Before (token-based)
model_old = MemXLNetForQA(
    base_model=base,
    mem_token_count=16,
    memory_update="gated"
)

# After (enhanced)
model_new = MemXLNetForQA(
    base_model=base,
    mem_token_count=16,
    memory_update="gated",
    use_differentiable_memory=True,  # Add this
    num_memory_heads=2               # Add this
)
```

### 📊 Memory Visualization

```python
from memxlnet.utils.multihop_utils import MemoryVisualizer

if model.memory_controller:
    viz_data = model.memory_controller.visualize_memory()

    visualizer = MemoryVisualizer()
    fig = visualizer.plot_memory_heatmap(viz_data['memory'])
    fig.savefig('memory.png')
```

### ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| No `memory_info` in output | Set `use_differentiable_memory=True` |
| CUDA OOM | Reduce `memory_slots` and `num_memory_heads` |
| Slow training | Disable `usage_tracking` and `temporal_links` |
| Import errors | Install: `pip install matplotlib seaborn` |

### 🧪 Quick Tests

```python
# Test if enhanced memory is working
outputs = model(input_ids=torch.randint(0, 1000, (1, 10)))
assert 'memory_info' in outputs, "Enhanced memory not working"
assert outputs['memory_info']['read_weights'].shape[1] == num_memory_heads

# Test backward compatibility
model_compat = MemXLNetForQA(base_model=base, mem_token_count=4)
outputs_compat = model_compat(input_ids=torch.randint(0, 1000, (1, 10)))
assert 'memory_info' not in outputs_compat, "Should not have memory_info"
```

### 📝 Key Files

| File | Purpose | Status |
|------|---------|--------|
| `src/memxlnet/models/memory_modules.py` | Core memory implementation | ✅ Available |
| `src/memxlnet/models/memxlnet_qa.py` | Enhanced model wrapper | ✅ Available |
| `src/memxlnet/data/streaming.py` | Streaming processor | ✅ Available |
| `examples/validate_answer_spans.py` | Usage examples | ✅ Available |
| `tests/unit/test_answer_span_validation.py` | Test suite | ✅ Available |

### 🎯 Quick Validation

```bash
# Run tests
pytest tests/unit/

# Test example
python examples/validate_answer_spans.py

# Check imports
python -c "from memxlnet.models import MemXLNetForQA; print('✅ Ready')"

# Test streaming
python -c "from memxlnet.data.streaming import StreamingSquadProcessor; print('✅ Streaming Ready')"
```