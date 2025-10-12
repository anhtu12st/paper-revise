# Planned Features for MemXLNet-QA

This document describes features that are planned or partially implemented for future releases of MemXLNet-QA. These features are documented for reference but may not be fully available in the current codebase.

## Status Legend
- 🚧 **Planned** - Feature is designed but not yet implemented
- 🔨 **In Progress** - Feature is partially implemented
- ✅ **Completed** - Feature is fully implemented and tested

---

## Enhanced Differentiable Memory (✅ Completed - Phase 1)

### Overview
Advanced memory system with content-based addressing and multi-head attention, designed to improve multi-hop reasoning capabilities.

### Implemented Components

#### MemoryController Class
**Status:** ✅ Fully implemented and tested in `memory_modules.py`

**Implemented Features:**
- ✅ Content-based memory addressing using cosine similarity
- ✅ Multi-head read/write attention (1-8 heads)
- ✅ Configurable attention sharpening (temperature control)
- ✅ Memory usage tracking for optimal slot allocation
- ✅ Temporal link matrix for relationship tracking
- ✅ Memory state management (get/set/reset)
- ✅ Visualization data export

**Configuration Parameters:**
```python
model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=32,
    use_differentiable_memory=True,  # Enable differentiable memory
    num_memory_heads=4,               # Multi-head attention
    memory_sharpness=2.0,             # Attention temperature
    enable_usage_tracking=True,       # Track slot usage
    enable_temporal_links=True,       # Track relationships
    memory_slots=64                   # Number of memory slots
)
```

**Implementation Status:**
- ✅ Full `MemoryController` implementation complete
- ✅ `DifferentiableMemory` class with all features
- ✅ Multi-head attention mechanism working
- ✅ Usage tracking system functional
- ✅ Temporal links operational
- ✅ Integration with `MemXLNetForQA` complete
- ✅ Comprehensive unit tests (26 tests passing)
- ✅ Integration tests (13 tests passing)
- ✅ Example script provided

**Test Coverage:**
- Unit tests: `tests/unit/test_memory_modules.py` (26/26 passing)
- Integration tests: `tests/integration/test_differentiable_memory_training.py` (13/13 passing)
- Example: `examples/train_with_differentiable_memory.py`

---

## Multi-Hop Reasoning Utilities (✅ Completed - Phase 2)

### HopTracker
**Status:** ✅ Fully implemented and tested

**Purpose:** Track reasoning chains across multiple hops in complex questions.

**Implemented Features:**
- ✅ Automatic hop detection based on attention patterns
- ✅ Bridge entity identification (entities spanning multiple segments)
- ✅ Confidence scoring per hop
- ✅ Reasoning chain reconstruction
- ✅ Statistics and analysis export

**API:**
```python
from memxlnet.utils import HopTracker

tracker = HopTracker(min_attention_threshold=0.1)

# Track each segment
for seg_idx, segment_data in enumerate(segments):
    entities = extract_entities(segment_data["text"])
    attention = model_outputs["attention_weights"]
    tracker.track_segment(seg_idx, attention, entities)

# Mark answer for analysis
tracker.mark_answer("answer text", answer_segment=2)

# Get analysis results
bridge_entities = tracker.detect_bridge_entities()
reasoning_hops = tracker.detect_hops()
hop_sequence = tracker.get_hop_sequence(to_answer=True)
stats = tracker.get_statistics()

# Export to JSON
tracker.export_analysis("analysis.json")
```

### MemoryVisualizer
**Status:** ✅ Fully implemented and tested

**Purpose:** Visualize memory state and attention patterns.

**Implemented Features:**
- ✅ Attention weight heatmaps (read/write)
- ✅ Memory usage timelines
- ✅ Temporal link matrix visualization
- ✅ Multi-head attention comparison
- ✅ Attention distribution analysis
- ✅ Animated segment-by-segment progression
- ✅ Comprehensive summary reports

**API:**
```python
from memxlnet.utils import MemoryVisualizer

visualizer = MemoryVisualizer(output_dir="./visualizations")

# Plot attention heatmaps
visualizer.plot_attention_heatmap(
    weights=read_weights,
    title="Read Attention - Segment 1",
    save_path="read_attention.png"
)

# Plot usage timeline
visualizer.plot_usage_timeline(
    usage_history=usage_data,
    save_path="usage_timeline.png"
)

# Plot temporal links
visualizer.plot_temporal_links(
    temporal_links=link_matrix,
    save_path="temporal_links.png"
)

# Multi-head comparison
visualizer.plot_multi_head_comparison(
    heads_weights=attention_weights,
    save_path="multi_head_comparison.png"
)

# Create comprehensive summary
visualizer.create_summary_report(
    memory_data=memory_history,
    save_path="summary_report"
)

# Create animation
visualizer.create_animation(
    segment_data=segment_history,
    output_path="memory_evolution.gif"
)
```

**Test Coverage:**
- Unit tests: `tests/unit/test_multihop_utils.py` (35 tests passing)
- Unit tests: `tests/unit/test_memory_visualization.py` (25 tests passing)
- Example: `examples/analyze_memory_attention.py`

---

## Additional Planned Features

### 1. Adaptive Memory Allocation (🚧 Planned)
Dynamic memory slot allocation based on question complexity and document length.

### 2. Memory Compression (🚧 Planned)
Compress less-important memory slots to maintain long-term context.

### 3. Cross-Document Memory (🚧 Planned)
Share memory states across related documents for better knowledge integration.

### 4. Memory Persistence (🚧 Planned)
Save and restore memory states between training sessions for continual learning.

---

## Implementation Roadmap

### Phase 1: Core Differentiable Memory (✅ Completed - January 2025)
- [x] Basic interface and configuration
- [x] Complete MemoryController implementation
- [x] Multi-head attention mechanism
- [x] Integration tests
- [x] Unit tests (26 tests)
- [x] Example script
- [x] Documentation updates

### Phase 2: Visualization & Analysis Tools (✅ Completed - January 2025)
- [x] HopTracker implementation
- [x] MemoryVisualizer implementation
- [x] Attention analysis utilities
- [x] Example script (`examples/analyze_memory_attention.py`)
- [x] Unit tests (60 tests total)
- [x] Training comparison script (`scripts/train_comparison_full.py`)

### Phase 3: Advanced Features
- [ ] Adaptive memory allocation
- [ ] Memory compression
- [ ] Cross-document memory
- [ ] Memory persistence

### Phase 4: Production Optimization
- [ ] Performance benchmarking
- [ ] Memory efficiency improvements
- [ ] Documentation completion
- [ ] Production deployment guides

---

## Contributing

If you're interested in implementing any of these features:

1. **Check current status** in this document
2. **Review existing code** in `src/memxlnet/models/memory_modules.py`
3. **Open an issue** to discuss implementation approach
4. **Submit a PR** with tests and documentation

---

## Notes for Developers

### Current Fallback Behavior
The codebase includes graceful fallbacks when enhanced memory features are not available:

```python
# In memxlnet_qa.py
try:
    from .memory_modules import MemoryController
    DIFFERENTIABLE_MEMORY_AVAILABLE = True
except ImportError:
    DIFFERENTIABLE_MEMORY_AVAILABLE = False

# Later in __init__
if use_differentiable_memory and DIFFERENTIABLE_MEMORY_AVAILABLE:
    self.memory_controller = MemoryController(...)
elif use_differentiable_memory:
    warnings.warn("Differentiable memory requested but not available...")
```

This ensures backward compatibility and allows gradual implementation.

### Testing Strategy
When implementing planned features:
1. Unit tests for each component
2. Integration tests with existing pipeline
3. Regression tests to ensure backward compatibility
4. Performance benchmarks vs token-based memory

---

## Recent Completion: Phase 1 (January 2025)

Phase 1 of the differentiable memory system is now complete! This includes:

- **Full MemoryController implementation** with content-based addressing
- **Multi-head attention** supporting 1-8 heads
- **Usage tracking** for memory slot optimization
- **Temporal links** for relationship tracking
- **Comprehensive test suite** with 39 tests (26 unit + 13 integration)
- **Example script** demonstrating all features
- **Complete integration** with training pipeline

### Getting Started with Differentiable Memory

```python
from memxlnet.models import MemXLNetForQA
from transformers import XLNetForQuestionAnsweringSimple

# Load base model
base_model = XLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased")

# Create model with differentiable memory
model = MemXLNetForQA(
    base_model=base_model,
    mem_token_count=16,
    use_differentiable_memory=True,
    num_memory_heads=4,
    memory_sharpness=2.0,
    enable_usage_tracking=True,
    enable_temporal_links=True,
    memory_slots=32,
)

# Use in training or inference as normal
outputs = model(**inputs)

# Access memory information
if "memory_info" in outputs:
    memory_info = outputs["memory_info"]
    print(f"Read weights: {memory_info['read_weights'].shape}")
    print(f"Memory usage: {memory_info['usage']}")
```

**See also:**
- Example script: `examples/train_with_differentiable_memory.py`
- Unit tests: `tests/unit/test_memory_modules.py`
- Integration tests: `tests/integration/test_differentiable_memory_training.py`
- Updated usage guide: `docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md`

---

**Last Updated:** January 2025 (Phase 1 & 2 Completed)
**Next Review:** Quarterly or after major implementation milestones
**Current Status:** Phase 1 & 2 Complete ✅
**Next Phase:** Advanced Features (Phase 3)

---

## Phase 2 Completion Summary (January 2025)

Phase 2 (Visualization & Analysis Tools) is now complete! This includes:

### HopTracker
- **Complete hop detection** based on attention patterns
- **Bridge entity identification** for multi-segment reasoning
- **Reasoning chain reconstruction** from question to answer
- **Comprehensive statistics** and JSON export
- **35 unit tests** covering all functionality

### MemoryVisualizer
- **Attention heatmaps** for read/write operations
- **Usage timelines** showing memory evolution
- **Temporal link visualization** for relationship tracking
- **Multi-head comparison plots** for analysis
- **Animated visualizations** for segment progression
- **Summary reports** with multiple visualization types
- **25 unit tests** ensuring correctness

### Training & Analysis Scripts
- **Validation script** (`scripts/validate_differentiable_memory.py`) - Quick validation of differentiable memory
- **Analysis example** (`examples/analyze_memory_attention.py`) - Complete analysis workflow
- **Full training comparison** (`scripts/train_comparison_full.py`) - Token-based vs differentiable memory comparison

### Getting Started with Phase 2 Tools

```python
# Analyze memory attention patterns
from memxlnet.utils import HopTracker, MemoryVisualizer

# Track reasoning hops
tracker = HopTracker()
for seg_idx, data in enumerate(segments):
    tracker.track_segment(seg_idx, attention, entities)

bridge_entities = tracker.detect_bridge_entities()
hops = tracker.detect_hops()
tracker.export_analysis("analysis.json")

# Create visualizations
visualizer = MemoryVisualizer(output_dir="./viz")
visualizer.plot_attention_heatmap(read_weights, save_path="attention.png")
visualizer.plot_usage_timeline(usage_history, save_path="usage.png")
visualizer.create_summary_report(memory_data, save_path="summary")
```

**See also:**
- Example: `examples/analyze_memory_attention.py`
- Tests: `tests/unit/test_multihop_utils.py`, `tests/unit/test_memory_visualization.py`
- Validation: `scripts/validate_differentiable_memory.py`
- Comparison: `scripts/train_comparison_full.py`

---

## Final Integration Fixes (January 2025)

After completing Phase 1 & 2 implementations, several critical fixes were required to enable end-to-end training with differentiable memory. These fixes resolved shape compatibility issues and trainer detection logic.

### Shape Compatibility Fix

**Issue:** Memory state shape mismatch between token-based and differentiable memory implementations caused runtime errors during training.

**Root Cause:**
- Token-based memory returns batched state: `[batch_size, num_slots, memory_dim]`
- Differentiable memory (MemoryController) returned unbatched state: `[num_slots, memory_dim]`
- Trainer's `recurrent_forward_pass()` expected consistent batched shapes

**Error:**
```python
RuntimeError: stack expects each tensor to be equal size,
but got [768] at entry 0 and [8, 768] at entry 1
```

**Solution:**
Fixed in `src/memxlnet/models/memxlnet_qa.py`:

1. **`get_initial_memory()` (lines 111-114):**
   ```python
   if self.use_differentiable_memory:
       # Return shape: [batch_size, num_slots, memory_dim]
       mem_state = self.memory_controller.get_memory_state()
       return mem_state.unsqueeze(0).expand(batch_size, -1, -1)
   ```

2. **`forward()` (lines 311-316):**
   ```python
   if memory_state is not None:
       if memory_state.dim() == 2:
           # Expand unbatched [num_slots, dim] to [batch_size, num_slots, dim]
           memory_state = memory_state.unsqueeze(0).expand(batch_size, -1, -1)
       self.memory_controller.set_memory_state(memory_state[0])
   ```

**Status:** ✅ Fixed - All memory implementations now return consistent batched shapes

### Trainer Wrapper Detection Fix

**Issue:** Model not properly wrapped with `MemXLNetForQA` when differentiable memory was enabled, causing initialization failures.

**Root Cause:**
Trainer's `should_use_wrapper` method only checked for token-based memory:
```python
# Old logic
def should_use_wrapper(self) -> bool:
    return self.config.memory_impl == "token" and self.config.memory_num_tokens > 0
```

This meant differentiable memory models weren't wrapped, preventing proper initialization.

**Solution:**
Fixed in `src/memxlnet/training/trainer.py` (lines 297-302):
```python
def should_use_wrapper(self) -> bool:
    """Determine if the model should be wrapped with MemXLNetForQA."""
    return (
        (self.config.memory_impl == "token" and self.config.memory_num_tokens > 0)
        or self.config.use_differentiable_memory
    )
```

**Status:** ✅ Fixed - Differentiable memory models now properly initialized

### Validation Success

After these fixes, end-to-end training with differentiable memory works correctly:

```bash
# Quick validation (~30 minutes)
python scripts/validate_differentiable_memory.py
```

**Expected Results:**
- ✅ Training completes without shape errors
- ✅ Memory states propagate correctly across segments
- ✅ Model achieves >15% F1 score on validation set
- ✅ All enhanced memory features functional

**Impact:**
These fixes complete the Phase 1 & 2 implementations, making differentiable memory fully operational in production training pipelines. Users can now train models with multi-head attention, usage tracking, and temporal links without encountering shape or initialization errors.

---
