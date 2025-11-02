# Data Models and Schema Changes

## Existing Data Models (Preserved)

The GMM enhancement does not modify existing data models. All current data structures remain intact:

- **QA Examples:** Question, context, answers (from SQuAD v2)
- **Chunked Documents:** Segmented long documents for time-step-major batching
- **Memory States:** Hidden states associated with memory tokens
- **Model Outputs:** Start logits, end logits, answer spans

## New Data Models (GMM-Specific)

### Expert Memory State

**Purpose:** Represent the state of a single memory expert within the GMM system

**Key Attributes:**
- `expert_id`: int - Expert index (0 to k-1)
- `memory_state`: Tensor (m × d) - Memory bank content for this expert
- `usage_stats`: Dict[str, float] - Activation frequency and routing statistics
- `initialization_strategy`: str - How this expert was initialized (learned, zeros, uniform, orthogonal)

**Integration:** Replaces single memory state in forward pass; each segment maintains k expert states

**Relationships:**
- **With Existing:** Conforms to same shape constraints as current memory states (m × d per expert)
- **With New:** k experts collectively replace single monolithic memory

### Routing Probabilities

**Purpose:** Track probability distribution over memory experts computed by gating network

**Key Attributes:**
- `segment_id`: int - Document segment index
- `routing_logits`: Tensor (batch_size × k) - Pre-softmax routing scores
- `routing_probs`: Tensor (batch_size × k) - Softmax-normalized expert selection probabilities
- `entropy`: Tensor (batch_size) - Routing entropy (measure of specialization)
- `top_expert_idx`: Tensor (batch_size) - Most activated expert for each example

**Integration:** Computed during forward pass; used for both write and read operations

**Relationships:**
- **With Existing:** Same batch dimension as current memory states
- **With New:** Controls expert update modulation and aggregated reads

### GMM Configuration

**Purpose:** Encapsulate all GMM-specific hyperparameters and settings

**Key Attributes:**
- `num_memory_experts`: int - Number of expert memory banks (k ∈ [2, 8])
- `routing_temperature`: float - Temperature for softmax routing (default: 1.0)
- `routing_mode`: str - "write-based" or "read-based" routing strategy
- `entropy_regularization_weight`: float - Regularization coefficient for routing entropy
- `load_balance_weight`: float - Load balancing loss weight (prevent expert collapse)
- `expert_init_strategies`: List[str] - Per-expert initialization methods

**Recommended Default Values:**
```python
# Recommended defaults for GMMConfig
num_memory_experts: int = 4                    # Balanced: enough specialization, manageable overhead
routing_temperature: float = 1.0               # Standard softmax (increase for more uniform routing)
routing_mode: str = "write-based"              # Reuse write routing for reads (lower overhead)
entropy_regularization_weight: float = 0.01    # Encourage routing diversity without overwhelming task loss
load_balance_weight: float = 0.01              # Prevent expert collapse; tuned from Switch Transformer
expert_init_strategies: List[str] = ["learned"] * 4  # Learned init for all experts (simplest)
```

**Preset Configurations:**
- **gmm-small:** k=2, minimal overhead, good for experimentation
- **gmm-balanced:** k=4 (default), good trade-off between specialization and efficiency
- **gmm-large:** k=8, maximum specialization, higher compute/memory cost

**Integration:** Extends existing `TrainingConfig` via inheritance; serialized with checkpoints

**Relationships:**
- **With Existing:** Inherits from `TrainingConfig`; compatible with existing training infrastructure
- **With New:** Core configuration for all GMM components

## Schema Integration Strategy

**Database Changes Required:** N/A (no database used)

**Checkpoint Format Changes:**
- **New Tables:** N/A
- **Modified Checkpoint Structure:**
  - Add `gmm_memory_experts` key containing all expert states
  - Add `gmm_gating_network` key containing router parameters
  - Add `gmm_config` key containing GMM-specific configuration
  - Preserve existing checkpoint structure for backward compatibility

**Migration Strategy:**
- **Forward Compatibility:** Old checkpoints load as single-expert GMM (k=1) when using GMM code
- **Backward Compatibility:** GMM checkpoints include metadata flag; loading into old code raises clear error with upgrade instructions
- **Version Detection:** Checkpoint metadata includes `memory_type: "token" | "differentiable" | "gmm"`

**Backward Compatibility:**
- Existing checkpoints remain valid; no migration required
- New `GMMXLNetForQA.from_pretrained()` method detects checkpoint type
- Clear error messages guide users when attempting incompatible loads

---
