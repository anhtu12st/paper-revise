# Enhanced MA-XLNet API Reference

## Overview

This document provides a complete API reference for the enhanced Memory-Augmented XLNet (MA-XLNet) implementation, focusing on the new differentiable memory features and multi-hop reasoning capabilities.

## Core Classes

### MemXLNetForQA (Enhanced)

The main wrapper class for XLNet with enhanced memory capabilities.

#### Constructor

```python
MemXLNetForQA(
    base_model: nn.Module,
    mem_token_count: int = 0,
    memory_init: str = "learned",
    memory_update: str = "gated",
    memory_dim: Optional[int] = None,
    # New enhanced parameters
    use_differentiable_memory: bool = False,
    num_memory_heads: int = 1,
    memory_sharpness: float = 1.0,
    enable_usage_tracking: bool = False,
    enable_temporal_links: bool = False,
    memory_slots: Optional[int] = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | `nn.Module` | Required | Base XLNet model |
| `mem_token_count` | `int` | `0` | Number of memory tokens |
| `memory_init` | `str` | `"learned"` | Memory initialization (`"learned"` or `"zeros"`) |
| `memory_update` | `str` | `"gated"` | Memory update mechanism (`"gated"` or `"none"`) |
| `memory_dim` | `Optional[int]` | `None` | Memory dimension (auto-detected if None) |
| `use_differentiable_memory` | `bool` | `False` | Enable differentiable memory |
| `num_memory_heads` | `int` | `1` | Number of read/write heads |
| `memory_sharpness` | `float` | `1.0` | Attention sharpening factor |
| `enable_usage_tracking` | `bool` | `False` | Track memory slot usage |
| `enable_temporal_links` | `bool` | `False` | Track temporal relationships |
| `memory_slots` | `Optional[int]` | `None` | Number of memory slots |

#### Methods

##### `forward()`

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    start_positions: Optional[torch.Tensor] = None,
    end_positions: Optional[torch.Tensor] = None,
    memory_state: Optional[torch.Tensor] = None,
    mem_read_ids: Optional[List[int]] = None,
    mem_write_ids: Optional[List[int]] = None,
    differentiable_memory_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]
```

**Returns:**
```python
{
    'loss': Optional[torch.Tensor],
    'start_logits': torch.Tensor,
    'end_logits': torch.Tensor,
    'new_memory_state': Optional[torch.Tensor],
    'memory_info': Optional[Dict[str, torch.Tensor]]  # Only if use_differentiable_memory=True
}
```

**Memory Info Structure:**
```python
'memory_info': {
    'read_weights': torch.Tensor,      # (batch_size, num_heads, memory_slots)
    'write_weights': torch.Tensor,     # (batch_size, num_heads, memory_slots)
    'read_vectors': torch.Tensor,      # (batch_size, num_heads, memory_dim)
    'write_vectors': torch.Tensor,     # (batch_size, num_heads, memory_dim)
    'memory_state': torch.Tensor,      # (memory_slots, memory_dim)
    'usage': torch.Tensor,             # (memory_slots,) - if usage_tracking=True
    'temporal_links': torch.Tensor,    # (memory_slots, memory_slots) - if temporal_links=True
}
```

##### `save_pretrained()`

```python
def save_pretrained(self, save_directory: str)
```

Saves the model including enhanced memory parameters.

##### `from_pretrained()`

```python
@classmethod
def from_pretrained(cls, load_directory: str, **kwargs)
```

Loads model with enhanced memory parameters.

### DifferentiableMemory

Core differentiable memory implementation with content-based addressing.

#### Constructor

```python
DifferentiableMemory(
    num_slots: int,
    slot_dim: int,
    num_heads: int = 1,
    sharpness: float = 1.0,
    enable_usage_tracking: bool = True,
    enable_temporal_links: bool = False,
)
```

#### Methods

##### `content_addressing()`

```python
def content_addressing(
    self,
    key: torch.Tensor,
    beta: Optional[torch.Tensor] = None
) -> torch.Tensor
```

Compute content-based addressing weights using cosine similarity.

**Parameters:**
- `key`: Query key tensor `(batch_size, num_heads, slot_dim)`
- `beta`: Key strength `(batch_size, num_heads, 1)`

**Returns:**
- Content-based weights `(batch_size, num_heads, num_slots)`

##### `read()`

```python
def read(self, weights: torch.Tensor) -> torch.Tensor
```

Read from memory using attention weights.

**Parameters:**
- `weights`: Attention weights `(batch_size, num_heads, num_slots)`

**Returns:**
- Read vectors `(batch_size, num_heads, slot_dim)`

##### `write()`

```python
def write(
    self,
    weights: torch.Tensor,
    write_vector: torch.Tensor,
    erase_vector: Optional[torch.Tensor] = None
)
```

Write to memory using attention weights.

**Parameters:**
- `weights`: Attention weights `(batch_size, num_heads, num_slots)`
- `write_vector`: Vectors to write `(batch_size, num_heads, slot_dim)`
- `erase_vector`: Erase factors `(batch_size, num_heads, slot_dim)`

##### `get_usage_weights()`

```python
def get_usage_weights(self) -> torch.Tensor
```

Get least-used memory slots for allocation.

**Returns:**
- Allocation weights `(num_slots,)`

##### `reset()`

```python
def reset(self)
```

Reset memory to initial state.

### MemoryController

High-level controller for memory operations.

#### Constructor

```python
MemoryController(
    input_dim: int,
    memory_slots: int,
    memory_dim: int,
    num_heads: int = 1,
    use_temporal_links: bool = False,
    use_usage_tracking: bool = True,
    sharpness: float = 1.0,
)
```

#### Methods

##### `forward()`

```python
def forward(
    self,
    input_state: torch.Tensor,
    prev_read: Optional[torch.Tensor] = None,
    prev_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
```

**Parameters:**
- `input_state`: Input from model `(batch_size, input_dim)`
- `prev_read`: Previous read vectors
- `prev_weights`: Previous attention weights

**Returns:**
- `output`: Processed output `(batch_size, input_dim)`
- `memory_info`: Dictionary with memory operation details

##### `get_memory_state()`

```python
def get_memory_state(self) -> torch.Tensor
```

Get current memory state.

**Returns:**
- Memory matrix `(num_slots, memory_dim)`

##### `set_memory_state()`

```python
def set_memory_state(self, state: torch.Tensor)
```

Set memory state.

##### `reset_memory()`

```python
def reset_memory(self)
```

Reset memory to initial state.

##### `visualize_memory()`

```python
def visualize_memory(self) -> Dict[str, np.ndarray]
```

Generate visualization data for memory state.

## Multi-Hop Utilities

### HopTracker

Track reasoning chains through memory operations.

#### Constructor

```python
HopTracker(track_attention: bool = True, track_content: bool = True)
```

#### Methods

##### `record_hop()`

```python
def record_hop(
    self,
    memory_info: Dict[str, torch.Tensor],
    question_part: Optional[str] = None,
    extracted_info: Optional[str] = None
)
```

Record information about a reasoning hop.

##### `get_reasoning_chain()`

```python
def get_reasoning_chain(
    self,
    question: str,
    answer: str,
    bridge_entities: Optional[List[str]] = None
) -> ReasoningChain
```

Get the complete reasoning chain.

##### `visualize_attention_flow()`

```python
def visualize_attention_flow(self) -> Optional[plt.Figure]
```

Visualize attention flow across hops.

##### `reset()`

```python
def reset(self)
```

Reset tracking state for new question.

### MemoryVisualizer

Visualize memory states and operations.

#### Static Methods

##### `plot_memory_heatmap()`

```python
@staticmethod
def plot_memory_heatmap(
    memory_state: np.ndarray,
    title: str = "Memory State",
    cmap: str = "coolwarm"
) -> Optional[plt.Figure]
```

Create heatmap of memory state.

##### `plot_usage_pattern()`

```python
@staticmethod
def plot_usage_pattern(
    usage: np.ndarray,
    temporal_links: Optional[np.ndarray] = None
) -> Optional[plt.Figure]
```

Visualize memory usage and temporal patterns.

### MultiHopMetrics

Metrics for evaluating multi-hop reasoning performance.

#### Static Methods

##### `hop_accuracy()`

```python
@staticmethod
def hop_accuracy(
    predicted_chains: List[ReasoningChain],
    gold_chains: List[ReasoningChain]
) -> Dict[str, float]
```

Calculate hop-level accuracy metrics.

**Returns:**
```python
{
    'exact_match': float,
    'partial_match': float,
    'bridge_entity_recall': float,
    'bridge_entity_precision': float,
    'avg_confidence': float,
    'success_rate': float
}
```

##### `analyze_error_patterns()`

```python
@staticmethod
def analyze_error_patterns(
    failed_chains: List[ReasoningChain]
) -> Dict[str, Any]
```

Analyze common error patterns in failed reasoning.

### MultiHopDebugger

Debugging tools for multi-hop reasoning.

#### Static Methods

##### `trace_reasoning()`

```python
@staticmethod
def trace_reasoning(
    model,
    tokenizer,
    question: str,
    context: str,
    verbose: bool = True
) -> Dict[str, Any]
```

Trace the reasoning process step by step.

##### `export_reasoning_chain()`

```python
@staticmethod
def export_reasoning_chain(
    chain: ReasoningChain,
    output_path: str
)
```

Export reasoning chain to JSON for analysis.

## Data Classes

### HopInfo

Information about a single reasoning hop.

```python
@dataclass
class HopInfo:
    hop_number: int
    question_focus: str
    memory_read_weights: np.ndarray
    memory_write_weights: np.ndarray
    extracted_info: str
    confidence: float
```

### ReasoningChain

Complete reasoning chain for a multi-hop question.

```python
@dataclass
class ReasoningChain:
    question: str
    answer: str
    hops: List[HopInfo]
    bridge_entities: List[str]
    total_confidence: float
    success: bool
```

## Configuration Classes

### TrainingConfig (Enhanced)

Enhanced training configuration with new memory parameters.

#### New Parameters

```python
# Enhanced memory settings (MA-XLNet multi-hop)
use_differentiable_memory: bool = False
num_memory_heads: int = 1
memory_sharpness: float = 1.0
enable_usage_tracking: bool = False
enable_temporal_links: bool = False
memory_slots: Optional[int] = None
```

## Error Handling

### Common Exceptions

#### Configuration Errors

```python
# Missing dependencies
RuntimeWarning: "Differentiable memory requested but memory_modules not available"

# Invalid parameters
ValueError: "memory_slots must be positive integer"
AttributeError: "XLNetConfig object has no attribute 'cls_token_id'"
```

#### Memory Errors

```python
# CUDA out of memory
RuntimeError: "CUDA out of memory"
# Solution: Reduce memory_slots or num_memory_heads

# Shape mismatch
RuntimeError: "Expected tensor shape mismatch"
# Solution: Check input dimensions and memory configuration
```

## Constants and Defaults

### Default Values

```python
DEFAULT_MEMORY_CONFIG = {
    'use_differentiable_memory': False,
    'num_memory_heads': 1,
    'memory_sharpness': 1.0,
    'enable_usage_tracking': False,
    'enable_temporal_links': False,
    'memory_slots': None,  # Auto-calculated
    'memory_init': 'learned',
    'memory_update': 'gated'
}
```

### Version Information

```python
MEMXLNET_CONFIG_VERSION = 3  # Current config version
BACKWARD_COMPATIBLE_VERSIONS = [1, 2, 3]  # Supported versions
```

## Type Hints

### Common Types

```python
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import numpy as np

MemoryState = torch.Tensor  # (memory_slots, memory_dim)
AttentionWeights = torch.Tensor  # (batch_size, num_heads, memory_slots)
MemoryInfo = Dict[str, torch.Tensor]
ReasoningChainList = List[ReasoningChain]
VisualizationData = Dict[str, np.ndarray]
```

## Usage Examples

See the [Enhanced MA-XLNet Usage Guide](ENHANCED_MA_XLNET_USAGE_GUIDE.md) for comprehensive usage examples and the [Quick Reference](ENHANCED_MA_XLNET_QUICK_REFERENCE.md) for common patterns.