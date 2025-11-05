# GMM-XLNet Guide

This guide provides comprehensive documentation for the Gated Memory Mixture (GMM) extension to MemXLNet-QA, enabling multi-expert memory-augmented question answering for long-context tasks.

## Table of Contents

1. [Overview and Motivation](#overview-and-motivation)
2. [Architecture Overview](#architecture-overview)
3. [Configuration Options](#configuration-options)
4. [Training GMM Models](#training-gmm-models)
5. [Evaluating GMM Models](#evaluating-gmm-models)
6. [Interpretability Analysis](#interpretability-analysis)
7. [Troubleshooting](#troubleshooting)
8. [References](#references)

## Overview and Motivation

### The Flat Memory Problem

In the base MemXLNet-QA system, a **single monolithic memory** stores information across document segments. This flat memory architecture faces several limitations:

1. **Capacity Bottleneck**: A single memory bank must encode all document information, limiting representation capacity
2. **Interference**: Different types of information (entities, events, reasoning) compete for the same memory slots
3. **Lack of Specialization**: The model cannot develop specialized memory subsystems for different aspects of understanding

### Mixture-of-Experts Solution

GMM-XLNet addresses these limitations by introducing **k independent memory experts**, each specialized for different types of information:

```
Single Memory (Base)          Multi-Expert Memory (GMM)
┌─────────────────┐          ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│                 │          │ Expert 1 │ │ Expert 2 │ │ Expert 3 │ │ Expert 4 │
│   Flat Memory   │   →      │ (Entity) │ │ (Event)  │ │(Temporal)│ │(Spatial) │
│                 │          │  Memory  │ │  Memory  │ │  Memory  │ │  Memory  │
└─────────────────┘          └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**Key Benefits**:
- **Increased Capacity**: k experts provide k× memory capacity
- **Reduced Interference**: Experts specialize to different information types
- **Emergent Specialization**: Routing network learns to direct information optimally
- **Computational Efficiency**: Mixture-of-Experts architecture scales sub-linearly

### Comparison to Other Approaches

| Feature | Token-Based Memory | Differentiable Memory | GMM Memory |
|---------|-------------------|----------------------|------------|
| **Memory Type** | Explicit tokens | Single learnable bank | k expert banks |
| **Capacity** | Fixed by token count | Fixed by slot count | k× slot count |
| **Specialization** | None | None | Automatic via routing |
| **Computational Cost** | O(T·M) | O(T·M) | O(T·M + k) |
| **Interpretability** | High (token-level) | Medium (attention weights) | High (routing + attention) |
| **Training Stability** | High | Medium | High (with load balancing) |

**Legend**: T = sequence length, M = memory slots, k = number of experts

### Research Motivation

GMM-XLNet is designed to test the hypothesis:

> **"Multi-expert memory architectures can improve long-context QA by enabling specialized memory subsystems that reduce interference and increase capacity."**

This architecture enables research into:
- **Expert specialization patterns**: What information does each expert learn to store?
- **Routing dynamics**: How does the gating network allocate information across experts?
- **Capacity-performance trade-offs**: Does k-expert memory scale performance linearly with k?
- **Transfer learning**: Can GMM models generalize routing strategies across domains?

## Architecture Overview

### System Components

GMM-XLNet extends the base XLNet model with four key components:

```
┌──────────────────────────────────────────────────────────┐
│                     GMMXLNetForQA                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. GatedMemoryMixture  ←──┐                           │
│     • k Expert Memory Banks │                           │
│     • Shape: (k, M, D)      │                           │
│                             │                           │
│  2. MemoryGatingNetwork ────┤ Routing                   │
│     • Content-based Router  │ Probabilities             │
│     • Output: (batch, k)    │                           │
│                             │                           │
│  3. ExpertUpdater ──────────┘                           │
│     • Routing-modulated Gates                            │
│     • Per-expert Updates                                 │
│                                                          │
│  4. AggregatedMemoryReader                              │
│     • Weighted Expert Combination                        │
│     • Output: (batch, M, D)                             │
│                                                          │
│  Base XLNet Model                                        │
│  • XLNet Encoder                                         │
│  • QA Prediction Head                                    │
└──────────────────────────────────────────────────────────┘
```

**Legend**: k = num experts, M = memory slots, D = hidden dimension

### 1. Multiple Memory Experts

The `GatedMemoryMixture` maintains k independent memory banks:

```python
# Shape: (num_experts, memory_slots, hidden_dim)
expert_memories = [
    expert_0,  # (M, D) - e.g., entity-focused
    expert_1,  # (M, D) - e.g., event-focused
    expert_2,  # (M, D) - e.g., temporal-focused
    expert_3,  # (M, D) - e.g., spatial-focused
]
```

**Initialization Strategies**:
- `orthogonal`: Orthogonal initialization for diverse expert states
- `learned`: Learnable parameters (default)
- `zeros`: Zero initialization (experts differentiate through training)

### 2. Learnable Gating Network

The `MemoryGatingNetwork` computes routing probabilities based on write token representations:

```python
# Input: Write token hidden states (batch, num_mem_tokens, hidden_dim)
# Output: Routing probabilities (batch, num_experts)

routing_probs = softmax(router_network(pooled_write_states) / temperature)
```

**Pooling Methods**:
- `mean`: Average pooling over write tokens (efficient, default)
- `max`: Max pooling (emphasizes salient features)
- `attention`: Learned attention pooling (adaptive, higher capacity)

**Temperature Scaling**:
- **High T (> 1.0)**: Soft routing, multiple experts activated (exploration)
- **Low T (< 1.0)**: Sharp routing, single expert activated (exploitation)
- **T = 1.0**: Balanced (default)

### 3. Routing-Modulated Updates

The `ExpertUpdater` applies gated updates to experts, modulated by routing probabilities:

```python
for expert_j in range(num_experts):
    # Compute expert-specific gate
    gate_j = sigmoid(gate_network_j(write_states))

    # Modulate by routing probability
    effective_gate_j = routing_probs[j] * gate_j

    # Gated update
    expert_j_new = (1 - effective_gate_j) * expert_j_old + effective_gate_j * write_states
```

**Key Properties**:
- **Soft Gating**: Gradual memory updates prevent catastrophic forgetting
- **Routing Modulation**: Low-probability experts receive minimal updates (preserves specialization)
- **Per-Expert Gates**: Each expert learns specialized update behavior

### 4. Aggregated Memory Reads

The `AggregatedMemoryReader` combines expert memories for read operations:

**Write-Based Routing** (default, efficient):
```python
# Reuse write routing probabilities
read_memory = sum(routing_probs[j] * expert_j for j in range(k))
```

**Read-Based Routing** (adaptive):
```python
# Compute separate routing based on query (read tokens)
read_routing_probs = softmax(router_network(pooled_read_states) / temperature)
read_memory = sum(read_routing_probs[j] * expert_j for j in range(k))
```

### Component Diagram

```
Input Segment (XLNet Format)
    ↓
[Context] [MEM_WRITE_0..N-1] [SEP] [MEM_READ_0..N-1] [Question] [SEP] [CLS]
    ↓                                     ↓
    └──→ XLNet Encoder ←──────────────────┘
            ↓           ↓
         Write States  Read States
            ↓
    ┌───────────────────────────┐
    │  MemoryGatingNetwork      │
    │  • Pools write states     │
    │  • Computes routing_probs │
    └───────────────────────────┘
            ↓
    routing_probs (batch, k)
            ↓
    ┌───────────────────────────┐
    │  ExpertUpdater            │
    │  • For each expert_j:     │
    │    - Compute gate_j       │
    │    - Modulate by p_j      │
    │    - Update expert_j      │
    └───────────────────────────┘
            ↓
    Updated Expert Memories
            ↓
    ┌───────────────────────────┐
    │  AggregatedMemoryReader   │
    │  • Weights experts by p   │
    │  • Returns combined state │
    └───────────────────────────┘
            ↓
    Read Memory → Replace [MEM_READ_*] Embeddings
            ↓
    QA Prediction Head
            ↓
    Start/End Logits
```

## Configuration Options

### GMMTrainingConfig Parameters

GMM-XLNet training is configured via `GMMTrainingConfig`, which extends the base `TrainingConfig`:

```python
from gmmxlnet.training import GMMTrainingConfig

config = GMMTrainingConfig(
    # GMM-specific parameters
    use_gmm_memory=True,                      # Enable GMM (required)
    num_memory_experts=4,                     # Number of experts k ∈ [2, 8]
    routing_temperature=1.0,                  # Routing softmax temperature
    routing_mode="write-based",               # "write-based" or "read-based"
    entropy_regularization_weight=0.0,        # Entropy regularization (≥ 0)
    load_balance_weight=0.01,                 # Load balance loss weight (≥ 0)
    expert_init_strategies=["learned"] * 4,   # Initialization per expert

    # Base MemXLNet parameters (inherited)
    memory_num_tokens=16,                     # Memory slots per expert
    progressive_segments=[2, 4, 6],           # Progressive training schedule
    num_epochs=3,                             # Training epochs
    batch_size=4,                             # Batch size
    learning_rate=5e-5,                       # Learning rate
    # ... (see API Reference for full list)
)
```

### Parameter Details

#### `use_gmm_memory` (bool, default=False)
- **Purpose**: Enable GMM memory system
- **Effect**: Instantiates GMM components (GatedMemoryMixture, MemoryGatingNetwork, etc.)
- **Required**: Must be `True` for GMM training

#### `num_memory_experts` (int, default=4)
- **Purpose**: Number of memory experts k
- **Valid Range**: 2 to 8 (recommended: 2, 4, 8 for computational efficiency)
- **Trade-offs**:
  - **k=2**: Low overhead, limited specialization
  - **k=4**: Balanced (recommended for most use cases)
  - **k=8**: Maximum capacity, higher computational cost

#### `routing_temperature` (float, default=1.0)
- **Purpose**: Temperature for routing softmax (controls sharpness)
- **Valid Range**: > 0.0
- **Guidance**:
  - **T = 1.0**: Balanced routing (default)
  - **T = 0.5**: Sharp routing, encourages specialization
  - **T = 2.0**: Soft routing, ensemble-like behavior
- **Typical Range**: 0.5 to 2.0

#### `routing_mode` (str, default="write-based")
- **Purpose**: Routing strategy for memory reads
- **Options**:
  - `"write-based"`: Reuse write routing probabilities (efficient, default)
  - `"read-based"`: Compute separate read routing (adaptive, slower)
- **Guidance**: Use `"write-based"` unless you observe significant performance gaps

#### `entropy_regularization_weight` (float, default=0.0)
- **Purpose**: Weight for entropy regularization loss (encourages exploration)
- **Valid Range**: ≥ 0.0
- **Effect**: Penalizes low-entropy routing (prevents premature specialization)
- **Guidance**:
  - **0.0**: No regularization (default, usually sufficient with load balancing)
  - **0.001-0.01**: Light regularization (if routing collapses to single expert)
  - **> 0.01**: Strong regularization (may hurt specialization)

#### `load_balance_weight` (float, default=0.01)
- **Purpose**: Weight for load balance loss (prevents expert collapse)
- **Valid Range**: ≥ 0.0
- **Effect**: Encourages uniform expert utilization
- **Guidance**:
  - **0.0**: No load balancing (may cause expert collapse)
  - **0.01**: Standard (recommended)
  - **0.1**: Strong (use if experts are severely imbalanced)

#### `expert_init_strategies` (list[str], default=None)
- **Purpose**: Initialization strategy for each expert
- **Valid Strategies**: `"orthogonal"`, `"learned"`, `"zeros"`
- **Default**: `["learned"] * k` if `None`
- **Example**: `["orthogonal", "orthogonal", "learned", "learned"]`

#### `memory_num_tokens` (int, inherited from TrainingConfig)
- **Purpose**: Number of memory slots per expert
- **Valid Range for GMM**: 1 to 32 (recommended: 16 for most use cases)
- **GMM Requirements**: **Must be > 0 when `use_gmm_memory=True`**
- **Key Difference from Differentiable Memory**:
  - **GMM Memory**: Requires `memory_num_tokens > 0` (recommended: 16)
  - **Differentiable Memory**: Uses `memory_num_tokens = 0` (no special tokens needed)
- **Guidance**:
  - **16 slots**: Standard for most applications (recommended)
  - **8 slots**: Smaller models, limited GPU memory
  - **32 slots**: Large models requiring maximum capacity
- **Validation**: GMM configuration will raise `ValueError` if `memory_num_tokens <= 0` when GMM is enabled

### Memory Token Requirements for GMM

Unlike differentiable memory (which uses `memory_num_tokens=0`), GMM memory requires
`memory_num_tokens > 0` to provide memory slots for each expert:

```python
# GMM Memory (CORRECT)
config = GMMTrainingConfig(
    use_gmm_memory=True,
    memory_num_tokens=16,  # Provides 16 memory slots per expert
    num_memory_experts=4,  # 4 experts × 16 slots = 64 total memory slots
)

# Differentiable Memory (CORRECT)
config = TrainingConfig(
    use_differentiable_memory=True,
    memory_num_tokens=0,  # No special memory tokens needed
)

# GMM Memory (INCORRECT - will raise ValueError)
config = GMMTrainingConfig(
    use_gmm_memory=True,
    memory_num_tokens=0,  # ❌ GMM experts need memory slots!
)
```

**Why GMM Requires Memory Slots:**
- Each expert maintains independent memory states
- Memory slots store expert-specific information and learned patterns
- Unlike differentiable memory which uses content-based addressing only
- GMM experts need explicit memory locations for read/write operations

### Preset Configurations

For quick experimentation, use preset factory functions:

#### Small Configuration (k=2)
```python
from gmmxlnet.training import gmm_small_config

config = gmm_small_config(
    num_epochs=3,
    batch_size=4,
    # ... override other params as needed
)
```

**Ideal For**:
- Prototyping and debugging
- Limited GPU memory (< 16GB)
- Quick experiments

#### Balanced Configuration (k=4, Recommended)
```python
from gmmxlnet.training import gmm_balanced_config

config = gmm_balanced_config(
    num_epochs=3,
    progressive_segments=[2, 4, 6],
    # ... override other params as needed
)
```

**Ideal For**:
- Production models
- Research experiments
- Balanced capacity and cost

#### Large Configuration (k=8)
```python
from gmmxlnet.training import gmm_large_config

config = gmm_large_config(
    num_epochs=3,
    batch_size=2,  # Reduce batch size for larger k
    # ... override other params as needed
)
```

**Ideal For**:
- High-capacity models
- Large-scale experiments
- Maximum expert specialization

### Configuration Examples

#### Example 1: Basic GMM Training
```python
from gmmxlnet.training import GMMTrainingConfig

config = GMMTrainingConfig(
    use_gmm_memory=True,
    num_memory_experts=4,
    memory_num_tokens=16,
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    output_dir="outputs/gmm-xlnet-squad",
)
```

#### Example 2: Sharp Routing with Entropy Regularization
```python
config = GMMTrainingConfig(
    use_gmm_memory=True,
    num_memory_experts=4,
    routing_temperature=0.5,              # Sharp routing
    entropy_regularization_weight=0.005,  # Prevent collapse
    load_balance_weight=0.01,
    memory_num_tokens=16,
    num_epochs=3,
)
```

#### Example 3: High-Capacity Model with Progressive Training
```python
config = GMMTrainingConfig(
    use_gmm_memory=True,
    num_memory_experts=8,
    memory_num_tokens=32,                 # More slots per expert
    progressive_segments=[2, 4, 6, 8],    # Longer curriculum
    routing_mode="read-based",            # Adaptive routing
    num_epochs=5,
    batch_size=2,                         # Smaller batch for larger k
)
```

## Training GMM Models

### Step-by-Step Workflow

#### Step 1: Prepare Environment
```bash
# Install dependencies
uv sync

# Set HuggingFace token (optional, for Hub integration)
export HF_TOKEN='your_token_here'
```

#### Step 2: Configure Training
```python
from gmmxlnet.training import GMMTrainingConfig

config = GMMTrainingConfig(
    use_gmm_memory=True,
    num_memory_experts=4,
    memory_num_tokens=16,
    progressive_segments=[2, 4],
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    output_dir="outputs/gmm-xlnet-squad",
    hub_model_id="username/gmm-xlnet-squad",  # Optional: auto-push to Hub
    hub_dataset_id="username/memxlnet-squad-mem16",  # Optional: use Hub dataset
)

# Save configuration
config.to_json("gmm_config.json")
```

#### Step 3: Initialize Model
```python
from gmmxlnet.models import GMMXLNetForQA
from transformers import XLNetForQuestionAnsweringSimple

# Load base XLNet model
base_model = XLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased")

# Wrap with GMM
model = GMMXLNetForQA(
    base_model=base_model,
    num_experts=config.num_memory_experts,
    memory_slots=config.memory_num_tokens,
    routing_mode=config.routing_mode,
    routing_temperature=config.routing_temperature,
)
```

#### Step 4: Load Dataset
```python
from memxlnet.data import ChunkedSquadDataset
from transformers import XLNetTokenizerFast

# Initialize tokenizer with memory tokens
tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
tokenizer.add_special_tokens({
    "additional_special_tokens": [
        "[MEM_READ_0]", "[MEM_READ_1]", ...,
        "[MEM_WRITE_0]", "[MEM_WRITE_1]", ...
    ]
})

# Load chunked dataset
train_dataset = ChunkedSquadDataset(
    split="train",
    tokenizer=tokenizer,
    max_seq_length=512,
    max_n_segs=max(config.progressive_segments),
)
```

#### Step 5: Train Model
```python
from gmmxlnet.training import GMMTrainer

# Initialize trainer
trainer = GMMTrainer(model=model, config=config)

# Train
trainer.train(train_dataset)

# Model checkpoints saved to config.output_dir
```

#### Step 6: Save and Upload
```python
# Save final checkpoint
model.save_pretrained("outputs/gmm-xlnet-squad/final")

# Optional: Upload to HuggingFace Hub
from scripts.upload_checkpoint_to_hub import upload_checkpoint

upload_checkpoint(
    checkpoint_path="outputs/gmm-xlnet-squad/final",
    hub_id="username/gmm-xlnet-squad",
    token=os.environ["HF_TOKEN"],
)
```

### Configuration Best Practices

#### Memory and Compute Optimization

**GPU Memory Constraints**:
```python
# 16GB GPU
config = GMMTrainingConfig(
    num_memory_experts=4,
    memory_num_tokens=16,
    batch_size=4,
    gradient_accumulation_steps=2,  # Effective batch size = 8
)

# 8GB GPU
config = GMMTrainingConfig(
    num_memory_experts=2,
    memory_num_tokens=8,
    batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
)
```

**Training Speed**:
- **Faster**: `routing_mode="write-based"`, `num_memory_experts=2`
- **Slower but Higher Quality**: `routing_mode="read-based"`, `num_memory_experts=8`

#### Hyperparameter Tuning Guidance

**Routing Temperature**:
1. Start with `T=1.0` (default)
2. If routing collapses to single expert → increase to `T=1.5` or `T=2.0`
3. If routing is too diffuse (no specialization) → decrease to `T=0.7` or `T=0.5`

**Load Balance Weight**:
1. Start with `load_balance_weight=0.01` (default)
2. Monitor expert utilization during training
3. If experts are severely imbalanced → increase to `0.05` or `0.1`

**Number of Experts**:
- Start with `k=4` (balanced)
- If performance plateaus → try `k=8` (more capacity)
- If overfitting → try `k=2` (less capacity)

### Common Training Issues

#### Issue 1: Expert Collapse (All Routing to Single Expert)

**Symptoms**:
- Routing probabilities: `[0.95, 0.02, 0.02, 0.01]` (one expert dominates)
- Load balance loss remains high

**Solutions**:
```python
# Solution 1: Increase load balance weight
config.load_balance_weight = 0.05  # Up from 0.01

# Solution 2: Increase temperature
config.routing_temperature = 1.5  # Up from 1.0

# Solution 3: Add entropy regularization
config.entropy_regularization_weight = 0.01
```

#### Issue 2: Out of Memory (OOM)

**Symptoms**:
- CUDA OOM errors during training

**Solutions**:
```python
# Solution 1: Reduce batch size
config.batch_size = 2  # Down from 4
config.gradient_accumulation_steps = 4  # Maintain effective batch size

# Solution 2: Reduce number of experts
config.num_memory_experts = 2  # Down from 4

# Solution 3: Reduce memory slots
config.memory_num_tokens = 8  # Down from 16
```

#### Issue 3: Slow Training

**Symptoms**:
- Training iterations are slow

**Solutions**:
```python
# Solution 1: Use write-based routing
config.routing_mode = "write-based"  # Instead of "read-based"

# Solution 2: Reduce experts
config.num_memory_experts = 2  # Down from 4 or 8

# Solution 3: Use faster pooling
# In model initialization:
model = GMMXLNetForQA(
    ...,
    pooling_method="mean",  # Instead of "attention"
)
```

## Evaluating GMM Models

### Evaluation Workflow

#### Step 1: Load Model from Checkpoint or Hub

**From Local Checkpoint**:
```python
from gmmxlnet.models import GMMXLNetForQA

model = GMMXLNetForQA.from_pretrained("outputs/gmm-xlnet-squad/final")
model.eval()
```

**From HuggingFace Hub**:
```python
model = GMMXLNetForQA.from_pretrained(
    "username/gmm-xlnet-squad",
    revision="best-model",  # Optional: specify revision
)
model.eval()
```

#### Step 2: Load Evaluation Dataset
```python
from memxlnet.data import ChunkedSquadDataset
from transformers import XLNetTokenizerFast

tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
# Add memory tokens to tokenizer (same as training)
tokenizer.add_special_tokens({...})

eval_dataset = ChunkedSquadDataset(
    split="validation",
    tokenizer=tokenizer,
    max_seq_length=512,
    max_n_segs=6,  # Use max segments from training
)
```

#### Step 3: Run Evaluation
```python
from memxlnet.evaluation import QAEvaluator

evaluator = QAEvaluator(model=model, tokenizer=tokenizer)
results = evaluator.evaluate(eval_dataset)

print(f"Exact Match: {results['exact_match']:.2f}")
print(f"F1 Score: {results['f1']:.2f}")
```

#### Step 4: Analyze Routing (Optional)
```python
from gmmxlnet.utils import GMMAnalyzer

analyzer = GMMAnalyzer(model=model)
routing_stats = analyzer.track_routing(eval_dataloader)

print(f"Expert Utilization: {routing_stats['expert_utilization']}")
print(f"Routing Entropy: {routing_stats['mean_entropy']:.3f}")
```

### Model Loading from Hub

#### List Available Models
```bash
# List your GMM models
huggingface-cli repo list -o username | grep gmm-xlnet
```

#### Load Specific Revision
```python
# Load best checkpoint
model = GMMXLNetForQA.from_pretrained(
    "username/gmm-xlnet-squad",
    revision="best-model",
)

# Load stage-specific checkpoint
model = GMMXLNetForQA.from_pretrained(
    "username/gmm-xlnet-squad",
    revision="stage-2-segs-4",
)
```

#### Check Model Metadata
```python
import json
from pathlib import Path

# Load config from Hub model
config_path = Path("outputs/gmm-xlnet-squad/final/config.json")
with open(config_path) as f:
    config = json.load(f)

print(f"Number of experts: {config['num_experts']}")
print(f"Memory slots: {config['memory_slots']}")
print(f"Routing mode: {config['routing_mode']}")
```

### Metrics Interpretation

#### Standard QA Metrics

**Exact Match (EM)**:
- **Definition**: Percentage of predictions exactly matching ground truth
- **Typical Range**: 70-85% on SQuAD v2
- **Interpretation**:
  - **EM < 70%**: Model may not be fully trained or has issues
  - **EM 75-80%**: Reasonable performance
  - **EM > 80%**: Strong performance

**F1 Score**:
- **Definition**: Token-level overlap between prediction and ground truth
- **Typical Range**: 75-90% on SQuAD v2
- **Interpretation**:
  - **F1 < 75%**: Model may need more training
  - **F1 80-85%**: Good performance
  - **F1 > 85%**: Very strong performance

**Gap (F1 - EM)**:
- **Definition**: Difference between F1 and EM
- **Typical Range**: 3-8 points
- **Interpretation**:
  - **Gap < 3**: Model is very precise (exact predictions)
  - **Gap 5-8**: Normal (partial overlaps)
  - **Gap > 10**: Model may be imprecise (noisy boundaries)

#### GMM-Specific Metrics

**Expert Utilization**:
- **Definition**: Percentage of activations per expert
- **Ideal**: Roughly uniform (25% per expert for k=4)
- **Interpretation**:
  - **Balanced (20-30% each)**: Healthy specialization
  - **Imbalanced (>50% to one expert)**: Expert collapse (increase load_balance_weight)
  - **Perfectly uniform (25% each)**: May indicate insufficient specialization

**Routing Entropy**:
- **Definition**: Shannon entropy of routing distribution: H = -Σ p_j log(p_j)
- **Range**: 0 (deterministic) to log(k) (uniform)
- **Interpretation** (k=4, max entropy = 1.39):
  - **H < 0.5**: Sharp routing, strong specialization
  - **H 0.8-1.2**: Balanced
  - **H > 1.3**: Diffuse routing, weak specialization

**Load Balance Loss**:
- **Definition**: Variance of expert utilization
- **Ideal**: Close to 0
- **Interpretation**:
  - **< 0.01**: Well-balanced
  - **0.01-0.05**: Acceptable
  - **> 0.05**: Imbalanced (tune load_balance_weight)

## Interpretability Analysis

GMM-XLNet provides rich interpretability tools for understanding expert specialization and routing behavior.

### Using GMMAnalyzer

#### Basic Analysis Workflow

```python
from gmmxlnet.utils import GMMAnalyzer
from torch.utils.data import DataLoader

# Initialize analyzer
analyzer = GMMAnalyzer(model=model, device="cuda")

# Create dataloader
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=8,
    shuffle=False,
)

# Track routing across dataset
routing_stats = analyzer.track_routing(
    dataloader=eval_dataloader,
    max_segments=500,  # Optional: limit for speed
)

# View summary statistics
print(routing_stats)
```

**Output**:
```python
{
    'expert_utilization': [0.27, 0.24, 0.26, 0.23],  # Per-expert activation rates
    'mean_entropy': 1.15,                             # Mean routing entropy
    'std_entropy': 0.32,                              # Std dev of entropy
    'total_segments': 500,                            # Segments processed
}
```

### Computing Specialization Metrics

#### Expert Diversity (Cosine Similarity)

```python
# Compute pairwise cosine similarity between expert memories
diversity_matrix = analyzer.compute_expert_diversity()

print("Expert Diversity Matrix:")
print(diversity_matrix)
```

**Interpretation**:
```
       Expert 0  Expert 1  Expert 2  Expert 3
Expert 0   1.00      0.42      0.38      0.51
Expert 1   0.42      1.00      0.55      0.43
Expert 2   0.38      0.55      1.00      0.47
Expert 3   0.51      0.43      0.47      1.00
```

- **Diagonal = 1.0**: Self-similarity
- **Off-diagonal < 0.5**: Diverse experts (good specialization)
- **Off-diagonal > 0.8**: Similar experts (weak specialization)

#### Routing Consistency

```python
# Compute routing consistency (how often same expert is chosen)
consistency = analyzer.compute_routing_consistency()

print(f"Routing Consistency: {consistency:.2f}")
```

**Interpretation**:
- **< 0.3**: Highly variable routing (context-dependent)
- **0.5-0.7**: Moderate consistency
- **> 0.8**: Very consistent routing (may indicate collapse)

### Generating Visualizations

#### 1. Routing Heatmap

```python
from gmmxlnet.utils import plot_routing_heatmap

# Generate heatmap of routing probabilities over segments
fig = plot_routing_heatmap(
    routing_data=analyzer.routing_data,
    max_segments=100,
    figsize=(12, 4),
)
fig.savefig("routing_heatmap.png")
```

**Output**: Heatmap showing which experts are activated for each segment

#### 2. Expert Utilization Bar Chart

```python
from gmmxlnet.utils import plot_expert_utilization

fig = plot_expert_utilization(
    utilization=routing_stats['expert_utilization'],
    figsize=(8, 5),
)
fig.savefig("expert_utilization.png")
```

#### 3. Routing Entropy Timeline

```python
from gmmxlnet.utils import plot_entropy_timeline

fig = plot_entropy_timeline(
    routing_data=analyzer.routing_data,
    figsize=(12, 4),
)
fig.savefig("entropy_timeline.png")
```

### Interpreting Routing Patterns

#### Pattern 1: Specialized Experts (Ideal)

```
Expert Utilization: [0.25, 0.26, 0.24, 0.25]  # Balanced
Mean Entropy: 1.05                             # Moderate
Expert Diversity: Off-diagonal < 0.5           # Diverse
```

**Interpretation**: Experts have learned distinct specializations with balanced utilization. This is the ideal scenario.

#### Pattern 2: Expert Collapse (Problem)

```
Expert Utilization: [0.85, 0.05, 0.05, 0.05]  # Imbalanced
Mean Entropy: 0.25                             # Low
Expert Diversity: N/A (unused experts)
```

**Interpretation**: Model relies primarily on one expert. **Solution**: Increase `load_balance_weight` or `routing_temperature`.

#### Pattern 3: Diffuse Routing (Problem)

```
Expert Utilization: [0.25, 0.25, 0.25, 0.25]  # Perfectly uniform
Mean Entropy: 1.38                             # Max entropy
Expert Diversity: Off-diagonal > 0.8           # Similar
```

**Interpretation**: Routing is too soft, experts are similar. **Solution**: Decrease `routing_temperature` or reduce `entropy_regularization_weight`.

#### Pattern 4: Context-Dependent Routing (Good)

```
Routing Consistency: 0.45                      # Moderate
Entropy Std Dev: 0.45                          # High variance
```

**Interpretation**: Model adapts routing based on context, indicating learned specialization strategies.

## Troubleshooting

### Common Errors

#### Error 1: `num_memory_experts must be an integer in [2, 8]`

**Cause**: Invalid number of experts specified

**Solution**:
```python
config = GMMTrainingConfig(
    num_memory_experts=4,  # Must be in [2, 8]
)
```

#### Error 2: `routing_temperature must be > 0`

**Cause**: Invalid temperature value

**Solution**:
```python
config = GMMTrainingConfig(
    routing_temperature=1.0,  # Must be positive
)
```

#### Error 3: `Model does not have memory_mixture attribute`

**Cause**: Loading non-GMM checkpoint with GMM analyzer

**Solution**:
```python
# Ensure model has use_gmm_memory=True
model = GMMXLNetForQA(
    base_model=base,
    use_gmm_memory=True,  # Required
    num_experts=4,
)
```

#### Error 4: `Checkpoint was trained with memory_type="gmm" but loading into standard model`

**Cause**: Attempting to load GMM checkpoint into non-GMM model

**Solution**:
```python
# Use GMMXLNetForQA instead of MemXLNetForQA
from gmmxlnet.models import GMMXLNetForQA

model = GMMXLNetForQA.from_pretrained("path/to/gmm/checkpoint")
```

### Performance Optimization

#### Slow Training

**Diagnostics**:
```python
import time
from torch.profiler import profile, ProfilerActivity

# Profile training step
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

print(prof.key_averages().table())
```

**Solutions**:
1. Use `routing_mode="write-based"` (faster than `"read-based"`)
2. Reduce `num_memory_experts` (2 or 4 instead of 8)
3. Use `pooling_method="mean"` (faster than `"attention"`)
4. Enable mixed precision training:
   ```python
   config.fp16 = True  # If supported
   ```

#### High Memory Usage

**Diagnostics**:
```python
import torch

# Check peak memory
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

**Solutions**:
1. Reduce batch size: `config.batch_size = 2`
2. Reduce number of experts: `config.num_memory_experts = 2`
3. Reduce memory slots: `config.memory_num_tokens = 8`
4. Enable gradient checkpointing (if available)

#### Poor Performance (Low F1/EM)

**Diagnostics**:
```python
# Check routing behavior
routing_stats = analyzer.track_routing(eval_dataloader)
print(f"Expert Utilization: {routing_stats['expert_utilization']}")
print(f"Mean Entropy: {routing_stats['mean_entropy']}")
```

**Solutions**:

**If Expert Collapse** (one expert dominates):
```python
config.load_balance_weight = 0.05  # Increase from 0.01
config.routing_temperature = 1.5   # Increase from 1.0
```

**If Diffuse Routing** (no specialization):
```python
config.routing_temperature = 0.7   # Decrease from 1.0
config.entropy_regularization_weight = 0.0  # Disable if enabled
```

**If Underfitting**:
```python
config.num_memory_experts = 8      # Increase capacity
config.memory_num_tokens = 32      # More slots
config.num_epochs = 5              # More training
```

### Debugging Routing Issues

#### Enable Detailed Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("gmmxlnet")
logger.setLevel(logging.DEBUG)
```

#### Inspect Routing Probabilities During Training

```python
# In training loop (custom script)
for batch in train_dataloader:
    outputs = model(**batch)

    # Access routing probabilities
    if hasattr(model, 'last_routing_probs'):
        routing_probs = model.last_routing_probs
        print(f"Routing probs: {routing_probs.mean(dim=0).cpu()}")

    loss = outputs.loss
    loss.backward()
```

#### Visualize Expert Memory States

```python
import matplotlib.pyplot as plt
import numpy as np

# Get expert memories
expert_memories = model.memory_mixture.expert_memories

# Plot memory norms for each expert
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for expert_idx in range(4):
    memory = expert_memories[expert_idx].detach().cpu().numpy()
    norms = np.linalg.norm(memory, axis=-1)  # (memory_slots,)

    axes[expert_idx].bar(range(len(norms)), norms)
    axes[expert_idx].set_title(f"Expert {expert_idx} Memory Norms")
    axes[expert_idx].set_xlabel("Memory Slot")
    axes[expert_idx].set_ylabel("L2 Norm")

plt.tight_layout()
plt.savefig("expert_memory_norms.png")
```

### FAQ

**Q: How many experts should I use?**

A: Start with `k=4` (balanced). Use `k=2` for prototyping or limited resources. Use `k=8` for maximum capacity if you have sufficient GPU memory (24GB+).

**Q: Which routing mode is better: write-based or read-based?**

A: **Write-based** is recommended for most use cases (faster, usually sufficient). Use **read-based** if you observe significant performance gaps between training and evaluation.

**Q: How do I know if my experts are specializing properly?**

A: Check three metrics:
1. **Expert Utilization**: Should be roughly balanced (e.g., [0.25, 0.24, 0.26, 0.25] for k=4)
2. **Routing Entropy**: Should be moderate (0.8-1.2 for k=4)
3. **Expert Diversity**: Off-diagonal cosine similarities should be < 0.6

**Q: Can I use GMM with differentiable memory (from Story 1.2)?**

A: No, GMM is a standalone memory architecture. Use either GMM (multi-expert) or differentiable memory (single expert with content-based addressing), not both simultaneously.

**Q: How do I transfer a trained GMM model to a new domain?**

A: Load the model checkpoint and fine-tune on the new domain. The routing network will adapt to the new data distribution:
```python
model = GMMXLNetForQA.from_pretrained("username/gmm-xlnet-squad")
# Fine-tune on new dataset
trainer.train(new_domain_dataset)
```

**Q: Can I visualize what each expert has learned?**

A: Yes, use the `GMMAnalyzer` to track routing patterns and correlate expert activations with question/context types. See [Interpretability Analysis](#interpretability-analysis) for details.

**Q: My model has expert collapse. What should I do?**

A: Increase `load_balance_weight` from 0.01 to 0.05 or 0.1, and/or increase `routing_temperature` from 1.0 to 1.5. See [Common Training Issues](#common-training-issues) for details.

## References

### Internal Documentation

- **[API Reference](../api/API_REFERENCE.md)**: Complete API documentation for all GMM classes
- **[Memory Tokens Guide](MEMORY_TOKENS_GUIDE.md)**: Detailed guide on memory token system (base architecture)
- **[Usage Guide](ENHANCED_MA_XLNET_USAGE_GUIDE.md)**: General usage guide for MemXLNet-QA
- **[Testing Guide](TESTING_VALIDATION_GUIDE.md)**: Testing and validation guidelines

### Code Examples

- **[examples/train_with_gmm_memory.py](../../examples/train_with_gmm_memory.py)**: Complete GMM training example
- **[examples/analyze_gmm_experts.py](../../examples/analyze_gmm_experts.py)**: Expert analysis and visualization

### External Resources

- **Mixture-of-Experts**: Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- **XLNet**: Yang et al., "XLNet: Generalized Autoregressive Pretraining for Language Understanding" (2019)
- **SQuAD v2**: Rajpurkar et al., "Know What You Don't Know: Unanswerable Questions for SQuAD" (2018)

### Support

For issues, questions, or contributions:
- GitHub Issues: [repository URL]
- Documentation: [docs/ directory]
- Examples: [examples/ directory]

---

**Last Updated**: 2025-11-02
**Version**: 1.0
