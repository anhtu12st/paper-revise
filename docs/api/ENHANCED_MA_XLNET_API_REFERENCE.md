# Enhanced MA-XLNet API Reference

> **✅ FEATURE STATUS UPDATE (January 2025)**
>
> **Phase 1 & 2 Complete!** All enhanced memory features are now fully implemented and tested.
>
> **✅ What Works:**
> - ✅ MemoryController with content-based addressing
> - ✅ Multi-head attention (1-8 heads)
> - ✅ Usage tracking and temporal links
> - ✅ Memory state management (get/set/reset)
> - ✅ HopTracker for multi-hop reasoning analysis
> - ✅ MemoryVisualizer for attention visualization
>
> **Test Coverage:**
> - 26 unit tests (memory_modules)
> - 13 integration tests (differentiable_memory_training)
> - 35 unit tests (multihop_utils)
> - 25 unit tests (memory_visualization)
>
> **Validation:** Run `python scripts/validate_differentiable_memory.py` to verify (~30 min)
>
> **For planned future features:** See [PLANNED_FEATURES.md](../PLANNED_FEATURES.md).

---

## Overview

Enhanced memory features provide advanced mechanisms for complex multi-hop reasoning tasks, including content-based memory addressing, multi-head attention, and comprehensive analysis tools.

## ✅ Implementation Status Check

**Verify Enhanced Memory Availability:**
```python
from memxlnet.models.memxlnet_qa import DIFFERENTIABLE_MEMORY_AVAILABLE

if DIFFERENTIABLE_MEMORY_AVAILABLE:
    print("✅ Enhanced memory available")
else:
    print("❌ memory_modules.py not found - check installation")
```

**Expected Output:**
```
✅ Enhanced memory available
```

---

## API Documentation

All features documented below are fully implemented and tested.

### Configuration Parameters

```python
from memxlnet.models import MemXLNetForQA

model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,

    # Enhanced Memory Parameters (✅ All Functional)
    use_differentiable_memory=True,    # Enable differentiable memory
    num_memory_heads=4,                # Number of attention heads (1-8 supported)
    memory_sharpness=2.0,              # Attention temperature (>0)
    enable_usage_tracking=True,        # Track slot usage for allocation
    enable_temporal_links=True,        # Track temporal relationships
    memory_slots=64                    # Number of memory slots
)
```

**Parameter Details:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_differentiable_memory` | bool | `False` | Enable differentiable memory (vs token-based) |
| `num_memory_heads` | int | `1` | Number of read/write attention heads (1-8) |
| `memory_sharpness` | float | `1.0` | Temperature for attention sharpening (higher = sharper) |
| `enable_usage_tracking` | bool | `False` | Track memory slot usage for optimal allocation |
| `enable_temporal_links` | bool | `False` | Track temporal relationships between memory accesses |
| `memory_slots` | int | `None` | Number of memory slots (defaults to max(mem_token_count, 16)) |

### MemoryController Class

**Status:** ✅ Fully implemented and tested

**Complete Interface:**
```python
class MemoryController(nn.Module):
    def __init__(
        self,
        input_dim: int,
        memory_slots: int,
        memory_dim: int,
        num_heads: int = 1,
        use_temporal_links: bool = False,
        use_usage_tracking: bool = False,
        sharpness: float = 1.0
    ):
        """Initialize memory controller.

        Args:
            input_dim: Dimension of input vectors from model
            memory_slots: Number of memory slots
            memory_dim: Dimension of memory vectors
            num_heads: Number of attention heads
            use_temporal_links: Enable temporal link tracking
            use_usage_tracking: Enable usage tracking
            sharpness: Attention sharpness parameter
        """

    def forward(
        self,
        input_state: torch.Tensor,
        prev_read: Optional[torch.Tensor] = None,
        prev_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process input through memory controller.

        Args:
            input_state: Input from model (batch_size, input_dim)
            prev_read: Previous read vectors (optional)
            prev_weights: Previous attention weights (optional)

        Returns:
            output: Processed output (batch_size, input_dim)
            memory_info: Dict with read_weights, write_weights, memory_state, usage, etc.
        """

    def get_memory_state(self) -> torch.Tensor:
        """Get current memory state (num_slots, memory_dim)."""

    def set_memory_state(self, state: torch.Tensor):
        """Set memory state."""

    def reset_memory(self):
        """Reset memory to initial state."""
```

**See:** [PLANNED_FEATURES.md](../PLANNED_FEATURES.md) for detailed API and examples.

---

## Migration Path

### Token-Based Memory (Traditional) ✅ Available

```python
# Backward-compatible traditional approach
model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,
    memory_init="learned",
    memory_update="gated"
)
# Production-ready, fully tested
```

### Differentiable Memory (Enhanced) ✅ Available

```python
# New differentiable memory with multi-head attention
model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,
    use_differentiable_memory=True,  # ✅ Fully functional
    num_memory_heads=4,               # ✅ Multi-head attention working
    memory_sharpness=2.0,             # ✅ Attention sharpening operational
    enable_usage_tracking=True,       # ✅ Usage tracking functional
    enable_temporal_links=True        # ✅ Temporal links working
)
# Production-ready since January 2025, comprehensive test coverage
```

### Gradual Migration Strategy

```python
# Step 1: Enable differentiable memory (minimal change)
model_step1 = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,
    use_differentiable_memory=True
)

# Step 2: Add multi-head attention
model_step2 = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,
    use_differentiable_memory=True,
    num_memory_heads=2
)

# Step 3: Full enhancement
model_final = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,
    use_differentiable_memory=True,
    num_memory_heads=4,
    memory_sharpness=2.0,
    enable_usage_tracking=True,
    enable_temporal_links=True
)
```

---

## Usage Examples

### Basic Usage with Memory Info

```python
from memxlnet.models import MemXLNetForQA
from transformers import XLNetForQuestionAnsweringSimple, XLNetTokenizerFast

# Load base model
base_model = XLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased")
tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")

# Create enhanced model
model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=16,
    use_differentiable_memory=True,
    num_memory_heads=4,
    enable_usage_tracking=True
)

# Process input
question = "What is the capital of France?"
context = "Paris is the capital of France."
inputs = tokenizer(question, context, return_tensors="pt")

# Get outputs with memory information
outputs = model(**inputs)

# Access memory information
if "memory_info" in outputs:
    memory_info = outputs["memory_info"]
    print(f"Read weights: {memory_info['read_weights'].shape}")
    print(f"Write weights: {memory_info['write_weights'].shape}")
    print(f"Memory state: {memory_info['memory_state'].shape}")
    if "usage" in memory_info:
        print(f"Slot usage: {memory_info['usage']}")
```

### Training with Differentiable Memory

```python
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

config = TrainingConfig(
    model_name="xlnet-base-cased",
    dataset_name="squad_v2",

    # Enable differentiable memory
    use_differentiable_memory=True,
    num_memory_heads=4,
    memory_sharpness=2.0,
    enable_usage_tracking=True,
    memory_slots=64,

    # Training settings
    num_epochs=3,
    train_batch_size=4,
    eval_batch_size=4,
    output_dir="./outputs/diff-memory"
)

trainer = XLNetRecurrentTrainer(config)
trainer.train()
```

### Validation Script

```bash
# Quick validation (~30 minutes)
python scripts/validate_differentiable_memory.py

# Expected output: F1 > 15%, all validation checks passed
```

---

## Recent Updates (January 2025)

### Shape Compatibility Fix
Fixed memory state shape mismatch between token-based and differentiable memory:
- **Issue**: Trainer expected batched memory `[batch_size, num_slots, memory_dim]`
- **Fix**: Updated `get_initial_memory()` and `forward()` in `memxlnet_qa.py`
- **Status**: ✅ Resolved, end-to-end training working

### Trainer Wrapper Detection
Improved wrapper detection to support differentiable memory:
- **Issue**: Trainer only wrapped model for `memory_impl == "token"`
- **Fix**: Updated `should_use_wrapper` condition in `trainer.py`
- **Status**: ✅ Resolved, differentiable memory properly initialized

---

## Contributing

Interested in contributing to future enhancements? See:
- [PLANNED_FEATURES.md](../PLANNED_FEATURES.md) - Roadmap (Phase 3+)
- `src/memxlnet/models/memory_modules.py` - Full implementation
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Contribution guidelines

---

**Document Status:** ✅ Production Ready
**Last Updated:** January 2025
**Implementation Status:** Phase 1 & 2 Complete - See [PLANNED_FEATURES.md](../PLANNED_FEATURES.md)
