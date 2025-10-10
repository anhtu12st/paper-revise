# Enhanced MA-XLNet API Reference

> **🚧 IMPORTANT - PLANNED FEATURES DOCUMENTATION**
>
> This document describes **planned enhancements** to MemXLNet-QA that are **NOT yet fully implemented**.
>
> **Current Status:**
> - ✅ **Configuration Interface**: Model accepts enhanced memory parameters
> - 🔨 **Partial Implementation**: Some components partially implemented in `memory_modules.py`
> - 🚧 **Planned Features**: Full differentiable memory, visualization, multi-hop tracking
>
> **What Works:**
> - Setting `use_differentiable_memory=True` (falls back to token-based with warning)
> - All other enhanced memory parameters (accepted but not functional)
> - Token-based memory (fully functional - see main [API Reference](API_REFERENCE.md))
>
> **What Doesn't Work:**
> - `MemoryController` operations
> - `HopTracker` and `MemoryVisualizer` (not yet implemented)
> - Multi-head attention
> - Usage tracking and temporal links
>
> **For Working Features:** See the main [API Reference](API_REFERENCE.md).
>
> **Implementation Roadmap:** See [PLANNED_FEATURES.md](../PLANNED_FEATURES.md).

---

## Overview

This document describes the planned API for enhanced memory features in MemXLNet-QA. These features are designed to provide advanced memory mechanisms for complex multi-hop reasoning tasks.

## ⚠️ Before Using This API

**Check Implementation Status:**
```python
from memxlnet.models.memxlnet_qa import DIFFERENTIABLE_MEMORY_AVAILABLE

if DIFFERENTIABLE_MEMORY_AVAILABLE:
    print("✅ Enhanced memory available")
else:
    print("🚧 Enhanced memory not available - using token-based memory")
```

**Expected Output (Current):**
```
🚧 Enhanced memory not available - using token-based memory
```

---

## Planned API Documentation

*The following sections describe the intended API once features are fully implemented.*

### Configuration Parameters (Planned)

```python
from memxlnet.models import MemXLNetForQA

model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,

    # Enhanced Memory Parameters (🚧 Planned)
    use_differentiable_memory=True,    # Enable differentiable memory
    num_memory_heads=4,                # Number of attention heads
    memory_sharpness=2.0,              # Attention temperature
    enable_usage_tracking=True,        # Track slot usage
    enable_temporal_links=True,        # Track temporal relationships
    memory_slots=64                    # Number of memory slots
)
```

### MemoryController Class (Planned)

**Status:** 🔨 Partially implemented, not functional

**Planned Interface:**
```python
class MemoryController:
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
            input_dim: Dimension of input vectors
            memory_slots: Number of memory slots
            memory_dim: Dimension of memory vectors
            num_heads: Number of attention heads
            use_temporal_links: Enable temporal link tracking
            use_usage_tracking: Enable usage tracking
            sharpness: Attention sharpness parameter
        """
        pass  # Implementation pending
```

For the complete planned API, see [PLANNED_FEATURES.md](../PLANNED_FEATURES.md).

---

## Migration Path

### Current (Token-Based Memory) ✅ Available

```python
model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,
    memory_init="learned",
    memory_update="gated"
)
# This works and is production-ready
```

### Planned (Differentiable Memory) 🚧 Not Available

```python
model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,
    use_differentiable_memory=True,  # Accepted but not functional
    num_memory_heads=4                # Parameters accepted but ignored
)
# This configuration is accepted but falls back to token-based memory
```

---

## Contributing

Interested in implementing these features? See:
- [PLANNED_FEATURES.md](../PLANNED_FEATURES.md) - Implementation roadmap
- `src/memxlnet/models/memory_modules.py` - Partial implementation
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Contribution guidelines

---

**Document Status:** 🚧 Planned Features Documentation
**Last Updated:** January 2025
**Implementation Status:** See [PLANNED_FEATURES.md](../PLANNED_FEATURES.md)
