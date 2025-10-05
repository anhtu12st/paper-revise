# Memory Tokens Guide for MemXLNet-QA

This guide provides comprehensive documentation of the memory token system in MemXLNet-QA, covering architecture, implementation, usage, and optimization strategies.

## Table of Contents

1. [Overview](#overview)
2. [Memory Token Architecture](#memory-token-architecture)
3. [Token Integration Process](#token-integration-process)
4. [Memory State Management](#memory-state-management)
5. [Training with Memory Tokens](#training-with-memory-tokens)
6. [Evaluation with Memory Tokens](#evaluation-with-memory-tokens)
7. [Configuration and Tuning](#configuration-and-tuning)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)
10. [Best Practices](#best-practices)

## Overview

The memory token system enables MemXLNet-QA to maintain persistent state across document segments, allowing the model to:

- **Remember context** from previous segments within the same document
- **Accumulate information** as it processes longer documents
- **Make informed decisions** based on full document understanding
- **Handle long-range dependencies** that span multiple segments

### Key Concepts

**Memory Tokens**: Special vocabulary tokens that interface with the model's recurrent memory
- `[MEM_READ_i]`: Tokens whose embeddings are replaced with memory state
- `[MEM_WRITE_i]`: Tokens whose hidden states update the memory state

**Memory State**: A learned representation that persists across document segments
- Shape: `(batch_size, memory_num_tokens, hidden_dim)`
- Updated via gated mechanisms for stable learning

**Document-Level Processing**: Memory is maintained per document and reset between documents

## Memory Token Architecture

### Token Types and Functions

```python
# Memory Read Tokens - Input Interface
READ_TOKENS = [
    "[MEM_READ_0]",    # ID: 32000
    "[MEM_READ_1]",    # ID: 32001
    "[MEM_READ_2]",    # ID: 32002
    ...
    "[MEM_READ_N-1]"   # ID: 32000 + N - 1
]

# Memory Write Tokens - Output Interface
WRITE_TOKENS = [
    "[MEM_WRITE_0]",   # ID: 32000 + N
    "[MEM_WRITE_1]",   # ID: 32000 + N + 1
    "[MEM_WRITE_2]",   # ID: 32000 + N + 2
    ...
    "[MEM_WRITE_N-1]"  # ID: 32000 + 2*N - 1
]
```

### Memory Token Placement Strategy

#### Standard Sequence Format
```
[CLS] question_tokens [SEP] context_tokens [SEP] [PAD] [PAD] [PAD]
```

#### Memory-Enhanced Sequence Format
```
[CLS] [MEM_READ_0] [MEM_READ_1] ... [MEM_READ_N-1] question_tokens [SEP]
context_tokens [MEM_WRITE_0] [MEM_WRITE_1] ... [MEM_WRITE_N-1] [SEP] [PAD] [PAD]
```

#### Placement Rationale

1. **READ tokens after CLS**: Early access to memory for question processing
2. **WRITE tokens in context**: Memory updates based on context understanding
3. **Symmetric design**: Same number of read and write tokens for balanced flow

### Memory Dimension Compatibility

```python
# Memory dimension must match model hidden dimension
model_config = XLNetConfig.from_pretrained("xlnet-base-cased")
memory_dim = model_config.d_model  # 768 for base, 1024 for large

# Memory state shape
memory_state = torch.zeros(batch_size, memory_num_tokens, memory_dim)
```

## Token Integration Process

### Step 1: Vocabulary Expansion

```python
def configure_memory_tokens(tokenizer: PreTrainedTokenizerBase,
                          memory_num_tokens: int) -> Dict[str, Any]:
    """
    Add memory tokens to tokenizer vocabulary.

    Before: ~32,000 tokens (standard XLNet)
    After:  ~32,000 + 2 * memory_num_tokens
    """
    # Generate token strings
    mem_read_tokens = [f"[MEM_READ_{i}]" for i in range(memory_num_tokens)]
    mem_write_tokens = [f"[MEM_WRITE_{i}]" for i in range(memory_num_tokens)]

    # Add to tokenizer vocabulary (modifies tokenizer in-place)
    all_tokens = mem_read_tokens + mem_write_tokens
    tokenizer.add_special_tokens({"additional_special_tokens": all_tokens})

    # Get token IDs for model configuration
    mem_read_ids = tokenizer.convert_tokens_to_ids(mem_read_tokens)
    mem_write_ids = tokenizer.convert_tokens_to_ids(mem_write_tokens)

    return {
        "mem_read_ids": mem_read_ids,     # [32000, 32001, 32002, ...]
        "mem_write_ids": mem_write_ids,   # [32004, 32005, 32006, ...]
        "total_added": len(all_tokens)    # 2 * memory_num_tokens
    }
```

### Step 2: Model Embedding Resize

```python
# After adding tokens, resize model embeddings
model = XLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased")
model.resize_token_embeddings(len(tokenizer))

# Memory wrapper handles token processing
memory_model = MemXLNetForQA(
    base_model=model,
    mem_token_count=memory_num_tokens,
    memory_init="learned",    # or "zeros"
    memory_update="gated"     # or "none"
)
```

### Step 3: Token Injection During Processing

```python
# In SquadLikeQADataset._process_example()
def inject_memory_tokens(self, input_ids, mem_read_ids, mem_write_ids):
    """
    Inject memory tokens into tokenized sequences.
    This is conceptual - actual implementation varies.
    """
    # Find insertion points
    cls_pos = 0  # After CLS
    sep_pos = input_ids.index(tokenizer.sep_token_id)  # Before first SEP

    # Insert READ tokens after CLS
    modified_ids = input_ids[:cls_pos+1] + mem_read_ids + input_ids[cls_pos+1:]

    # Insert WRITE tokens before SEP (adjust position after READ insertion)
    sep_pos += len(mem_read_ids)
    modified_ids = modified_ids[:sep_pos] + mem_write_ids + modified_ids[sep_pos:]

    return modified_ids
```

## Memory State Management

### Memory Initialization

The memory state is initialized when processing the first segment of a document:

```python
def get_initial_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
    """Initialize memory states for a batch."""
    if self.memory_init == "learned":
        # Use learned parameters
        memory = self.learned_memory.unsqueeze(0).expand(
            batch_size, self.mem_token_count, self.memory_dim
        )
    elif self.memory_init == "zeros":
        # Zero initialization
        memory = torch.zeros(
            batch_size, self.mem_token_count, self.memory_dim,
            device=device
        )
    else:
        raise ValueError(f"Unknown memory_init: {self.memory_init}")

    return memory
```

### Memory State Flow Across Segments

```python
# Processing pipeline for a document with 3 segments
document_segments = [segment_1, segment_2, segment_3]
memory_state = model.get_initial_memory(batch_size=1, device=device)

for i, segment in enumerate(document_segments):
    print(f"Processing segment {i+1}")

    # Forward pass with current memory
    outputs = model(
        input_ids=segment['input_ids'],
        attention_mask=segment['attention_mask'],
        token_type_ids=segment['token_type_ids'],
        memory_state=memory_state,
        mem_read_ids=mem_config['mem_read_ids'],
        mem_write_ids=mem_config['mem_write_ids']
    )

    # Extract predictions and updated memory
    start_logits = outputs['start_logits']
    end_logits = outputs['end_logits']
    memory_state = outputs['new_memory_state']  # Use for next segment

    print(f"Memory state shape: {memory_state.shape}")
    print(f"Memory state norm: {memory_state.norm().item():.4f}")
```

### Memory Update Mechanisms

#### Gated Memory Updates (Recommended)

```python
def _update_memory(self, current_memory: torch.Tensor,
                  new_representations: torch.Tensor) -> torch.Tensor:
    """
    Update memory using gated mechanism for stable learning.

    Args:
        current_memory: (batch_size, mem_tokens, hidden_dim)
        new_representations: (batch_size, mem_tokens, hidden_dim)

    Returns:
        updated_memory: (batch_size, mem_tokens, hidden_dim)
    """
    # Concatenate current and new information
    combined = torch.cat([current_memory, new_representations], dim=-1)
    # Shape: (batch_size, mem_tokens, 2 * hidden_dim)

    # Compute gate values (what to keep from current memory)
    gate = torch.sigmoid(self.memory_gate(combined))
    # Shape: (batch_size, mem_tokens, hidden_dim)

    # Compute update values (what to add from new information)
    update = torch.tanh(self.memory_update_net(combined))
    # Shape: (batch_size, mem_tokens, hidden_dim)

    # Gated combination
    new_memory = gate * update + (1 - gate) * current_memory
    # new_memory = how_much_new * new_info + how_much_old * old_info

    return new_memory
```

#### Simple Replacement Updates

```python
def _update_memory_simple(self, current_memory: torch.Tensor,
                         new_representations: torch.Tensor) -> torch.Tensor:
    """Simple replacement - just use new representations."""
    return new_representations
```

#### No Updates (Memory Disabled)

```python
def _update_memory_none(self, current_memory: torch.Tensor,
                       new_representations: torch.Tensor) -> torch.Tensor:
    """No updates - memory stays constant."""
    return current_memory
```

### Memory Extraction from Hidden States

```python
def _extract_memory_representations(self, input_ids: torch.Tensor,
                                   hidden_states: torch.Tensor,
                                   mem_write_ids: List[int]) -> torch.Tensor:
    """
    Extract memory representations from WRITE token hidden states.

    Args:
        input_ids: (batch_size, seq_len)
        hidden_states: (batch_size, seq_len, hidden_dim)
        mem_write_ids: List of WRITE token IDs

    Returns:
        memory_reps: (batch_size, mem_token_count, hidden_dim)
    """
    batch_size, seq_len = input_ids.shape
    memory_reps = []

    # Extract hidden states for each WRITE token
    for mem_id in mem_write_ids:
        # Find positions of this token in the batch
        positions = (input_ids == mem_id).nonzero(as_tuple=True)

        if len(positions[0]) > 0:
            # Extract hidden states at token positions
            batch_indices = positions[0]  # Which example in batch
            seq_indices = positions[1]    # Which position in sequence
            mem_rep = hidden_states[batch_indices, seq_indices]

            # Handle multiple occurrences per example (average)
            batch_mem_reps = []
            for b in range(batch_size):
                batch_mask = (batch_indices == b)
                if batch_mask.any():
                    batch_mem_reps.append(mem_rep[batch_mask].mean(dim=0))
                else:
                    # No token found - use zero vector
                    batch_mem_reps.append(
                        torch.zeros(hidden_states.size(-1), device=hidden_states.device)
                    )

            memory_reps.append(torch.stack(batch_mem_reps))
        else:
            # No tokens found - use zero vectors
            memory_reps.append(
                torch.zeros(batch_size, hidden_states.size(-1), device=hidden_states.device)
            )

    return torch.stack(memory_reps, dim=1)  # (batch_size, mem_count, hidden_dim)
```

## Training with Memory Tokens

### Memory-Aware Training Loop

```python
def train_with_memory(model, train_dataloader, optimizer, config):
    """Training loop with memory state management."""
    model.train()

    for epoch in range(config.num_epochs):
        for batch_idx, time_step_batches in enumerate(train_dataloader):
            # Memory bank for this document batch
            memory_bank = {}

            # Process each time step
            for time_step, batch in enumerate(time_step_batches):
                # Get memory states for this batch
                batch_memory = []
                for example_id in batch['example_ids']:
                    if example_id in memory_bank:
                        batch_memory.append(memory_bank[example_id])
                    else:
                        # Initialize memory for new document
                        init_mem = model.get_initial_memory(1, device)
                        batch_memory.append(init_mem[0])

                memory_state = torch.stack(batch_memory, dim=0)

                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions'],
                    memory_state=memory_state
                )

                loss = outputs['loss']
                new_memory = outputs['new_memory_state']

                # Update memory bank
                for i, example_id in enumerate(batch['example_ids']):
                    if batch['document_mask'][i]:  # Only active documents
                        memory_bank[example_id] = new_memory[i].detach()

                # Backpropagation
                loss.backward()

            # Optimizer step after processing all time steps
            optimizer.step()
            optimizer.zero_grad()
```

### Progressive Training Strategy

```python
# Progressive training with increasing segments
class ProgressiveTraining:
    def __init__(self, config):
        self.progressive_segments = config.progressive_segments  # [1, 2, 4, 6]
        self.current_stage = 0

    def should_progress(self, epoch):
        """Check if should move to next stage."""
        epochs_per_stage = self.total_epochs // len(self.progressive_segments)
        return epoch > 0 and epoch % epochs_per_stage == 0

    def get_max_segments(self):
        """Get current maximum segments."""
        if self.current_stage < len(self.progressive_segments):
            return self.progressive_segments[self.current_stage]
        return self.progressive_segments[-1]

    def progress_stage(self):
        """Move to next stage."""
        self.current_stage = min(self.current_stage + 1,
                               len(self.progressive_segments) - 1)
```

### Memory-Specific Training Considerations

#### Gradient Handling
```python
# Memory states should not accumulate gradients across documents
memory_state = memory_state.detach()  # Break gradient flow between documents

# But allow gradients within document segments
memory_state.requires_grad_(True)
```

#### Learning Rate Scheduling
```python
# Memory components may need different learning rates
optimizer = AdamW([
    {'params': model.base.parameters(), 'lr': 3e-5},           # Base model
    {'params': model.learned_memory, 'lr': 1e-4},             # Memory init
    {'params': model.memory_gate.parameters(), 'lr': 5e-5},   # Gate network
    {'params': model.memory_update_net.parameters(), 'lr': 5e-5}  # Update network
])
```

#### Warmup Strategy
```python
# Warm up memory components gradually
def memory_warmup_schedule(epoch, config):
    if epoch < config.warmup_freeze_base_epochs:
        # Freeze base model, train only memory components
        for param in model.base.parameters():
            param.requires_grad = False
        for param in model.memory_parameters():
            param.requires_grad = True
    else:
        # Train all parameters
        for param in model.parameters():
            param.requires_grad = True
```

## Evaluation with Memory Tokens

### Memory-Aware Evaluation Pipeline

```python
def evaluate_with_memory(model, eval_dataloader, eval_dataset, device):
    """Evaluation with proper memory state tracking."""
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_idx, time_step_batches in enumerate(eval_dataloader):
            memory_bank = {}

            # Process each time step
            for time_step, batch in enumerate(time_step_batches):
                # Prepare memory states
                memory_states = []
                for example_id in batch['example_ids']:
                    if example_id.startswith('padding'):
                        # Use initial memory for padding
                        memory_states.append(model.get_initial_memory(1, device)[0])
                    else:
                        # Get or initialize memory for real documents
                        if example_id in memory_bank:
                            memory_states.append(memory_bank[example_id])
                        else:
                            memory_states.append(model.get_initial_memory(1, device)[0])

                memory_state = torch.stack(memory_states, dim=0)

                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'],
                    memory_state=memory_state
                )

                # Update memory bank for active documents
                new_memory = outputs['new_memory_state']
                for i, example_id in enumerate(batch['example_ids']):
                    if (batch['document_mask'][i] and
                        not example_id.startswith('padding')):
                        memory_bank[example_id] = new_memory[i]

                # Extract predictions
                predictions = extract_predictions(outputs, batch)
                all_predictions.extend(predictions)

    return aggregate_predictions(all_predictions, eval_dataset)
```

### Prediction Aggregation

```python
def aggregate_predictions(predictions, eval_dataset):
    """Aggregate predictions across document segments."""
    from collections import defaultdict

    # Group by document
    doc_predictions = defaultdict(list)
    for pred in predictions:
        doc_predictions[pred['example_id']].append(pred)

    # Select best prediction per document
    final_predictions = []
    for i in range(len(eval_dataset)):
        doc_id = f"doc_{i}"

        if doc_id in doc_predictions:
            # Choose prediction with highest confidence
            best_pred = max(doc_predictions[doc_id],
                          key=lambda x: x['confidence'])
            final_predictions.append(best_pred['text'])
        else:
            final_predictions.append("")  # No prediction

    return final_predictions
```

## Configuration and Tuning

### Memory Configuration Options

```python
@dataclass
class MemoryConfig:
    # Core memory settings
    memory_num_tokens: int = 16          # Number of READ/WRITE token pairs
    memory_update: str = "gated"         # "gated", "simple", "none"
    memory_init: str = "learned"         # "learned", "zeros"
    memory_impl: str = "token"           # Implementation type

    # Training settings
    warmup_freeze_base_epochs: int = 1   # Freeze base during warmup

    # Memory-specific hyperparameters
    memory_gate_dropout: float = 0.1     # Dropout in gate networks
    memory_init_std: float = 0.02        # Std for learned memory init

    def validate(self):
        assert self.memory_num_tokens > 0, "Must have positive memory tokens"
        assert self.memory_update in ["gated", "simple", "none"]
        assert self.memory_init in ["learned", "zeros"]
```

### Hyperparameter Tuning Guidelines

#### Memory Token Count
```python
# Guidelines for memory_num_tokens selection
memory_configs = [
    {"tokens": 4,  "use_case": "Small experiments, fast training"},
    {"tokens": 8,  "use_case": "Balanced performance/speed"},
    {"tokens": 16, "use_case": "Good performance, recommended"},
    {"tokens": 32, "use_case": "Maximum performance, slower training"},
    {"tokens": 64, "use_case": "Research experiments only"}
]

# Rule of thumb: Start with 16, tune based on performance
```

#### Memory Update Strategy
```python
# Update mechanism comparison
update_strategies = {
    "gated": {
        "performance": "Best",
        "stability": "High",
        "training_time": "Longer",
        "parameters": "More (gate + update networks)"
    },
    "simple": {
        "performance": "Good",
        "stability": "Medium",
        "training_time": "Medium",
        "parameters": "Fewer"
    },
    "none": {
        "performance": "Baseline",
        "stability": "High",
        "training_time": "Fastest",
        "parameters": "Fewest (no updates)"
    }
}
```

#### Learning Rate Sensitivity
```python
# Memory components often need different learning rates
lr_recommendations = {
    "base_model": 3e-5,           # Standard transformer LR
    "memory_init": 1e-4,          # Slightly higher for initialization
    "memory_gates": 5e-5,         # Conservative for gates
    "memory_updates": 5e-5        # Conservative for updates
}
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Memory States Growing Unboundedly

**Symptoms**: Memory state norms increasing exponentially during training

**Diagnosis**:
```python
def monitor_memory_norms(memory_state):
    norms = memory_state.norm(dim=-1).mean().item()
    if norms > 10.0:  # Threshold
        print(f"Warning: Large memory norms: {norms:.2f}")
    return norms
```

**Solutions**:
```python
# 1. Add memory state normalization
memory_state = F.layer_norm(memory_state, memory_state.shape[-1:])

# 2. Clip memory state norms
max_norm = 5.0
memory_state = torch.renorm(memory_state, p=2, dim=-1, maxnorm=max_norm)

# 3. Reduce learning rates for memory components
```

#### Issue 2: Memory Tokens Not Found in Sequences

**Symptoms**: `_extract_memory_representations` returns zeros

**Diagnosis**:
```python
def debug_memory_tokens(input_ids, mem_write_ids):
    for i, mem_id in enumerate(mem_write_ids):
        count = (input_ids == mem_id).sum().item()
        print(f"MEM_WRITE_{i} (ID {mem_id}): {count} occurrences")
```

**Solutions**:
```python
# 1. Verify tokenizer configuration
assert len(tokenizer) > 32000, "Memory tokens not added to tokenizer"

# 2. Check token injection in dataset processing
# 3. Ensure model vocab size matches tokenizer
model.resize_token_embeddings(len(tokenizer))
```

#### Issue 3: Poor Memory Utilization

**Symptoms**: Memory states remain close to initialization

**Diagnosis**:
```python
def analyze_memory_utilization(initial_memory, current_memory):
    diff = (current_memory - initial_memory).norm().item()
    print(f"Memory change from init: {diff:.4f}")
    return diff
```

**Solutions**:
```python
# 1. Increase learning rates for memory components
# 2. Check gate values are not saturated
gate_values = torch.sigmoid(self.memory_gate(combined))
print(f"Gate mean: {gate_values.mean():.3f}, std: {gate_values.std():.3f}")

# 3. Add memory regularization loss
memory_utilization_loss = -memory_state.var(dim=-1).mean()
total_loss = qa_loss + 0.01 * memory_utilization_loss
```

#### Issue 4: Time-Step-Major Batching Errors

**Symptoms**: Shape mismatches in memory bank operations

**Diagnosis**:
```python
def debug_time_step_batch(batch, memory_bank):
    print(f"Batch example_ids: {batch['example_ids']}")
    print(f"Document mask: {batch['document_mask']}")
    print(f"Memory bank keys: {list(memory_bank.keys())}")

    for example_id in batch['example_ids']:
        if example_id in memory_bank:
            mem_shape = memory_bank[example_id].shape
            print(f"{example_id}: memory shape {mem_shape}")
```

**Solutions**:
```python
# 1. Ensure consistent memory dimensions
assert all(mem.shape == expected_shape for mem in memory_bank.values())

# 2. Handle padding documents correctly
if not example_id.startswith('padding'):
    memory_bank[example_id] = new_memory[i]

# 3. Validate document masks
assert batch['document_mask'].sum() <= len(batch['example_ids'])
```

## Advanced Topics

### Memory Initialization Strategies

#### Learned Initialization with Constraints
```python
class ConstrainedMemoryInit(nn.Module):
    def __init__(self, mem_count, mem_dim, constraint="sphere"):
        super().__init__()
        self.constraint = constraint
        self.raw_memory = nn.Parameter(torch.randn(mem_count, mem_dim) * 0.02)

    def forward(self):
        if self.constraint == "sphere":
            # Normalize to unit sphere
            return F.normalize(self.raw_memory, p=2, dim=-1)
        elif self.constraint == "bounded":
            # Bound to [-1, 1]
            return torch.tanh(self.raw_memory)
        else:
            return self.raw_memory
```

#### Task-Specific Initialization
```python
def initialize_memory_for_qa(model, qa_examples):
    """Initialize memory based on representative QA examples."""
    # Process representative examples
    rep_features = []
    for example in qa_examples[:100]:  # Use subset
        features = process_example(example)
        rep_features.append(features)

    # Compute average context representations
    context_reps = []
    for features in rep_features:
        # Extract context hidden states
        with torch.no_grad():
            outputs = model.base(features['input_ids'],
                                features['attention_mask'])
            context_mask = features['token_type_ids'] == 1
            context_hidden = outputs.last_hidden_state[context_mask]
            context_reps.append(context_hidden.mean(0))

    # Use k-means to find representative vectors
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=model.mem_token_count)
    cluster_centers = kmeans.fit(torch.stack(context_reps)).cluster_centers_

    # Initialize memory with cluster centers
    model.learned_memory.data.copy_(torch.from_numpy(cluster_centers))
```

### Memory Compression Techniques

#### Attention-Based Memory Updates
```python
class AttentionMemoryUpdate(nn.Module):
    def __init__(self, hidden_dim, mem_count):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.memory_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, current_memory, new_representations):
        # Use attention to selectively update memory
        attended_updates, _ = self.attention(
            query=current_memory.transpose(0, 1),     # (mem_count, batch, dim)
            key=new_representations.transpose(0, 1),   # (mem_count, batch, dim)
            value=new_representations.transpose(0, 1)  # (mem_count, batch, dim)
        )

        # Project and combine
        updated_memory = current_memory + self.memory_projection(
            attended_updates.transpose(0, 1)
        )

        return updated_memory
```

#### Multi-Level Memory
```python
class MultiLevelMemory(nn.Module):
    """Memory organized into multiple levels with different granularities.

    Note: This is a future enhancement concept, not currently implemented.
    """
    def __init__(self, mem_count, hidden_dim, num_levels=2):
        super().__init__()
        self.num_levels = num_levels
        self.memories = nn.ModuleList([
            nn.Parameter(torch.randn(mem_count // (2**i), hidden_dim) * 0.02)
            for i in range(num_levels)
        ])

    def get_memory_at_level(self, level):
        """Get memory at specific level."""
        return self.memories[level]

    def compress_to_next_level(self, memory, level):
        """Compress memory to next level."""
        if level + 1 < self.num_levels:
            # Simple compression: average pairs
            compressed = memory.view(*memory.shape[:-2], -1, 2, memory.shape[-1])
            return compressed.mean(dim=-2)
        return memory
```

### Memory Persistence Across Sessions

#### Memory State Serialization
```python
def save_memory_bank(memory_bank, filepath):
    """Save memory bank for resuming training."""
    torch.save({
        'memory_states': {k: v.cpu() for k, v in memory_bank.items()},
        'timestamp': time.time(),
        'version': '1.0'
    }, filepath)

def load_memory_bank(filepath, device):
    """Load memory bank for resuming training."""
    checkpoint = torch.load(filepath, map_location=device)
    memory_bank = {
        k: v.to(device) for k, v in checkpoint['memory_states'].items()
    }
    return memory_bank
```

#### Cross-Document Memory Transfer
```python
class CrossDocumentMemory:
    def __init__(self, similarity_threshold=0.8):
        self.global_memory_pool = {}
        self.similarity_threshold = similarity_threshold

    def get_similar_memory(self, document_embedding):
        """Find similar document memory for initialization."""
        best_similarity = 0
        best_memory = None

        for doc_emb, memory in self.global_memory_pool.items():
            similarity = F.cosine_similarity(document_embedding, doc_emb, dim=0)
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_memory = memory

        return best_memory

    def update_global_pool(self, document_embedding, final_memory):
        """Add document memory to global pool."""
        # Simple replacement strategy
        self.global_memory_pool[document_embedding] = final_memory
```

## Best Practices

### Development Guidelines

#### 1. Start Simple, Scale Gradually
```python
# Development progression
configs = [
    {"mem_tokens": 4,  "update": "simple", "segments": 1},  # Baseline
    {"mem_tokens": 8,  "update": "gated",  "segments": 2},  # Intermediate
    {"mem_tokens": 16, "update": "gated",  "segments": 4},  # Production
]
```

#### 2. Monitor Memory Health
```python
class MemoryMonitor:
    def __init__(self):
        self.memory_norms = []
        self.memory_changes = []
        self.gate_activations = []

    def log_memory_state(self, memory_state, prev_memory=None, gates=None):
        # Track memory norms
        norm = memory_state.norm(dim=-1).mean().item()
        self.memory_norms.append(norm)

        # Track memory changes
        if prev_memory is not None:
            change = (memory_state - prev_memory).norm().item()
            self.memory_changes.append(change)

        # Track gate activations
        if gates is not None:
            activation = gates.mean().item()
            self.gate_activations.append(activation)

    def report(self):
        print(f"Avg memory norm: {np.mean(self.memory_norms):.3f}")
        print(f"Avg memory change: {np.mean(self.memory_changes):.3f}")
        print(f"Avg gate activation: {np.mean(self.gate_activations):.3f}")
```

#### 3. Memory-Aware Evaluation
```python
def comprehensive_memory_evaluation(model, eval_data):
    """Evaluate model with detailed memory analysis."""
    results = {}

    # Standard evaluation
    standard_metrics = evaluate_model(model, eval_data)
    results['standard'] = standard_metrics

    # Memory utilization analysis
    memory_stats = analyze_memory_utilization(model, eval_data)
    results['memory'] = memory_stats

    # Ablation: disable memory
    model_no_memory = disable_memory(model)
    no_memory_metrics = evaluate_model(model_no_memory, eval_data)
    results['no_memory'] = no_memory_metrics

    # Memory contribution
    results['memory_contribution'] = {
        'f1_improvement': standard_metrics['f1'] - no_memory_metrics['f1'],
        'em_improvement': standard_metrics['em'] - no_memory_metrics['em']
    }

    return results
```

#### 4. Robust Training Practices
```python
def robust_memory_training(model, train_data, config):
    """Training with memory-specific robustness measures."""

    # Gradient clipping for memory components
    memory_params = [p for name, p in model.named_parameters()
                    if 'memory' in name]

    # Custom optimizer groups
    optimizer = AdamW([
        {'params': [p for p in model.parameters() if p not in memory_params],
         'lr': config.learning_rate},
        {'params': memory_params, 'lr': config.memory_learning_rate}
    ])

    # Memory state validation
    def validate_memory_state(memory_state):
        if torch.isnan(memory_state).any():
            print("Warning: NaN in memory state")
            return model.get_initial_memory(memory_state.shape[0], memory_state.device)

        norm = memory_state.norm(dim=-1)
        if (norm > 10.0).any():
            print("Warning: Large memory norms, clipping")
            memory_state = torch.renorm(memory_state, p=2, dim=-1, maxnorm=5.0)

        return memory_state

    # Training loop with validation
    for batch in train_data:
        memory_state = validate_memory_state(memory_state)
        outputs = model(batch, memory_state=memory_state)
        # ... rest of training
```

### Performance Optimization

#### Memory-Efficient Implementation
```python
# Use gradient checkpointing for long sequences
model.gradient_checkpointing_enable()

# Optimize memory allocation
def optimize_memory_allocation(batch_size, mem_tokens, hidden_dim):
    # Pre-allocate memory tensors
    memory_cache = torch.zeros(
        batch_size, mem_tokens, hidden_dim,
        device=device, requires_grad=False
    )
    return memory_cache

# Use mixed precision for memory operations
with torch.cuda.amp.autocast():
    memory_state = model.update_memory(current_memory, new_reps)
```

#### Efficient Token Processing
```python
# Cache memory token positions to avoid repeated searches
class MemoryTokenCache:
    def __init__(self, mem_read_ids, mem_write_ids):
        self.mem_read_ids = set(mem_read_ids)
        self.mem_write_ids = set(mem_write_ids)
        self.position_cache = {}

    def get_memory_positions(self, input_ids):
        # Use tuple for hashable key
        key = tuple(input_ids.tolist())
        if key not in self.position_cache:
            positions = self._compute_positions(input_ids)
            self.position_cache[key] = positions
        return self.position_cache[key]
```

This comprehensive guide provides the foundation for understanding and implementing the memory token system in MemXLNet-QA. The memory mechanism is crucial for the model's ability to handle long documents effectively while maintaining state across segments.