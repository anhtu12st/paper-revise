# MemXLNet-QA Brownfield Enhancement PRD

**Version**: 1.0
**Date**: 2025-11-02
**Author**: Sarah (Product Owner)
**Enhancement**: Gated Memory Mixture (GMM-XLNet)

---

## Intro Project Analysis and Context

### Analysis Source

**Source**: IDE-based fresh analysis of existing project structure and documentation

The analysis was conducted through direct examination of:
- Project codebase structure (`src/memxlnet/`)
- Technical documentation (19 comprehensive markdown documents)
- Configuration files (`pyproject.toml`, `.gitignore`)
- Existing implementation files

### Current Project State

**Project Name**: MemXLNet-QA

**Primary Purpose**: Enable XLNet to handle long-context question answering through recurrent memory states that persist across document segments, specifically targeting SQuAD v2 and similar datasets.

**Current System Overview**:

MemXLNet-QA is a mature, production-ready research system for memory-augmented question answering on long documents. The project implements:

- **Token-based memory system** with explicit `[MEM_READ]` and `[MEM_WRITE]` tokens
- **Differentiable memory module** (optional, backward-compatible enhancement)
- **Time-step-major batching** for proper memory state propagation across document segments
- **Progressive training** with curriculum learning (increasing segment counts)
- **Comprehensive infrastructure**: 19 technical docs, complete test suite, HuggingFace Hub integration

**Current Tech Stack**: Python 3.12+, PyTorch 2.8+, Transformers 4.56.2+, Datasets 4.1+, comprehensive testing with pytest

**Architecture Style**: Modular memory-augmented transformer with explicit memory token interface

**Deployment Method**: Local GPU training (16GB+ VRAM), HuggingFace Hub for model/dataset storage

### Available Documentation Analysis

**Documentation Status**: ✅ Comprehensive documentation coverage

The project has excellent documentation across all critical areas:

**Available Documentation**:

- ✓ **Tech Stack Documentation** - `pyproject.toml`, `CLAUDE.md`
- ✓ **Source Tree/Architecture** - `MA_XLNET_IMPLEMENTATION.md`
- ✓ **Coding Standards** - ruff, mypy configured in `pyproject.toml`
- ✓ **API Documentation** - `API_REFERENCE.md`, `ENHANCED_MA_XLNET_API_REFERENCE.md`
- ✓ **Memory System Documentation** - `MEMORY_TOKENS_GUIDE.md` (1010 lines, comprehensive)
- ✓ **Data Processing Documentation** - `DATA_PROCESSING.md`, `DATA_FLOW_DIAGRAMS.md`
- ✓ **Testing Documentation** - `TESTING_VALIDATION_GUIDE.md`
- ✓ **Usage Guides** - `ENHANCED_MA_XLNET_USAGE_GUIDE.md`, `USAGE_EXAMPLES.md`
- ✓ **Technical Details** - `CLS_INDEX_FIX.md`, `UNICODE_AND_POSITION_MAPPING.md`
- ✓ **Feature Roadmap** - `PLANNED_FEATURES.md`

**Assessment**: No additional documentation project required. Existing documentation provides comprehensive foundation for enhancement planning.

### Enhancement Scope Definition

#### Enhancement Type

- ☑ **New Feature Addition** (primary)
- ☑ **Major Feature Modification** (extends existing memory system)

#### Enhancement Description

Implement **Gated Memory Mixture (GMM-XLNet)** - a novel Mixture-of-Experts (MoE) approach to memory management. Instead of a single monolithic memory state, the system will maintain multiple specialized memory "experts" (k memory banks), each learning to store different types of information.

**Key Innovation**: A learnable gating network (router) dynamically determines which expert(s) to update based on incoming information, allowing selective memory updates and protecting irrelevant experts from being overwritten.

**Technical Approach**:
- Replace single memory state M ∈ ℝ^(m×d) with k expert memories {M₁, M₂, ..., Mₖ}
- Implement gating network computing probability distribution over experts: p = softmax(W_gate · pool(M_new))
- Modulate LSTM-style gates with router probabilities: M_j^(i) = (p_j · g_j) ⊙ u_j + (1 - p_j · g_j) ⊙ M_j^(i-1)
- Aggregate experts for reads: M_context = Σ(p_j · M_j)

**Research Significance**: Creates a structured, learnable memory hierarchy with high interpretability potential (e.g., one expert for numerical data, another for named entities). Novel application of MoE to recurrent memory states.

#### Impact Assessment

- ☑ **Significant Impact** (substantial existing code changes)

**Rationale**: Requires modifications to:
- Core memory modules (`src/memxlnet/models/memory_modules.py`)
- Model architecture (`src/memxlnet/models/memxlnet_qa.py`)
- Training configuration (`src/memxlnet/training/trainer.py`)
- Evaluation pipeline
- Documentation and testing infrastructure

However, design will maintain **full backward compatibility** with existing token-based and differentiable memory implementations through configuration flags.

### Goals and Background Context

#### Goals

- Implement MoE-based memory routing with k specialized memory experts (k=2 to k=8)
- Add learnable gating network for dynamic memory slot selection
- Extend gated memory updates with router probability modulation
- Implement aggregated memory read with weighted expert averaging
- Provide memory specialization analysis tools for interpretability research
- Maintain full backward compatibility with existing memory implementations
- Enable publishable research with compelling interpretability analysis showing expert specialization patterns

#### Background Context

**Problem Statement**:

Current MemXLNet-QA uses "flat" memory architectures - either token-based or differentiable memory - where all memory slots are treated uniformly. While this works well for basic long-context QA, it has limitations:

1. **No learned specialization** of memory slots
2. **Limited filtering ability** - cannot protect specific information across long contexts
3. **Uniform updates** - all memory slots equally affected by each segment
4. **Limited interpretability** - difficult to understand what information is stored where

**Research Gap**:

Existing long-context architectures face the "flat memory problem":
- **Recurrent Memory Transformer (RMT)**: Flat memory with limited filtering
- **Hierarchical Memory Transformer (HMT)**: Fixed bio-inspired hierarchy (sensory → short-term → long-term)
- **Transformer-XL**: No explicit memory management, relies on hidden state recurrence

**Solution Approach**:

GMM-XLNet introduces a novel application of Mixture-of-Experts to recurrent memory states. Unlike HMT's fixed hierarchy, GMM allows **dynamic, task-learned specialization**. Unlike standard MoE (used for feed-forward scaling), GMM applies expert routing to stateful memory management.

**Research Value**:

This enhancement is designed for publication in top-tier ML conferences (ICML, NeurIPS, ACL). Key contributions:
- Novel architecture addressing known limitations
- Flexible, learnable approach vs. fixed hierarchies
- High interpretability through routing probability analysis
- Maintains efficiency of existing MemXLNet-QA system

**Integration with Existing System**:

GMM will be the third memory implementation option alongside token-based and differentiable memory, maintaining the project's philosophy of backward-compatible, opt-in enhancements.

### Change Log

| Change | Date | Version | Description | Author |
|--------|------|---------|-------------|--------|
| Initial PRD | 2025-11-02 | 1.0 | GMM-XLNet enhancement PRD created | Sarah (PO) |

---

## Requirements

### Functional Requirements

**FR1**: The system SHALL implement a Mixture-of-Experts memory architecture with configurable number of memory experts (k=2 to k=8).

**FR2**: The system SHALL implement a learnable gating network (router) that computes probability distributions over memory experts based on memory write proposals.

**FR3**: The gating network SHALL use mean-pooling over memory write token representations followed by a learnable weight matrix to generate routing logits.

**FR4**: The system SHALL implement gated memory updates where each expert's update is modulated by both the global routing probability (p_j) and the local LSTM-style gate (g_j).

**FR5**: The system SHALL implement aggregated memory read operations that compute weighted averages of all memory experts based on routing probabilities.

**FR6**: The system SHALL support two routing modes: "write-based" (routing probabilities from previous write) and "read-based" (new routing probabilities computed for each read).

**FR7**: The system SHALL initialize each memory expert independently with optional different initialization strategies per expert.

**FR8**: The system SHALL provide memory specialization analysis tools that track which experts activate for different types of input content.

**FR9**: The system SHALL maintain full backward compatibility with existing token-based and differentiable memory implementations through configuration flags.

**FR10**: The system SHALL support saving and loading GMM-XLNet models with all expert states and gating network parameters.

**FR11**: The system SHALL provide visualization tools for routing probabilities across document segments to enable interpretability analysis.

**FR12**: The system SHALL integrate with existing progressive training and warmup strategies without requiring new training infrastructure.

### Non-Functional Requirements

**NFR1**: Memory operations SHALL maintain O(k × memory_slots × memory_dim × num_heads) computational complexity where k is the number of experts.

**NFR2**: Training with GMM SHALL not increase GPU memory usage by more than 50% compared to single-expert differentiable memory at the same total memory capacity.

**NFR3**: The gating network SHALL be implemented as a lightweight module with fewer than 100K additional parameters for typical configurations (k≤8, hidden_dim=768).

**NFR4**: Model inference latency SHALL not increase by more than 30% compared to single-expert memory when using k=4 experts.

**NFR5**: The implementation SHALL pass all existing unit tests and integration tests in the test suite.

**NFR6**: All new code SHALL maintain >= 80% test coverage and follow existing code quality standards (ruff, mypy).

**NFR7**: Documentation SHALL include comprehensive guides for GMM configuration, training, and interpretability analysis.

**NFR8**: The implementation SHALL support gradient checkpointing to enable training on GPUs with 16GB VRAM using reasonable batch sizes.

**NFR9**: Routing probability computations SHALL be numerically stable with automatic handling of edge cases (all-zero probabilities, NaN values).

**NFR10**: The system SHALL provide clear error messages when GMM-specific configuration errors occur (e.g., mismatched expert counts during loading).

### Compatibility Requirements

**CR1**: Existing checkpoints trained with token-based or differentiable memory SHALL load and evaluate correctly without any modifications.

**CR2**: The `MemXLNetForQA` class interface SHALL remain unchanged; GMM functionality SHALL be accessed through new configuration parameters only.

**CR3**: All existing training scripts (`phase2_train.py`, `train.py`, `train_comparison_full.py`) SHALL work with GMM by adding configuration flags without code changes.

**CR4**: The time-step-major batching system SHALL work with GMM memory updates without modifications to the data loader.

**CR5**: Evaluation scripts SHALL support GMM models through the same interfaces used for existing memory implementations.

**CR6**: Model serialization format SHALL remain compatible with HuggingFace Hub upload/download infrastructure.

**CR7**: Memory visualization tools SHALL be extended (not replaced) to support GMM routing visualization alongside existing memory visualizations.

---

## Technical Constraints and Integration Requirements

### Existing Technology Stack

**Languages**: Python 3.12+

**Frameworks**:
- PyTorch 2.8.0+
- Transformers 4.56.2+
- Datasets 4.1.1+
- HuggingFace Hub 0.20.0+

**Development Tools**:
- pytest 7.4.0+ (testing)
- ruff 0.1.0+ (linting, line length 120)
- mypy 1.7.0+ (type checking)
- uv (package management)

**Infrastructure**:
- Local GPU training (16GB+ VRAM recommended)
- HuggingFace Hub for model/dataset storage
- Git for version control

**External Dependencies**:
- SQuAD v2 dataset (via HuggingFace Datasets)
- XLNet base models (xlnet-base-cased, xlnet-large-cased)

**Constraints**:
- Must support Python 3.12+ (specified in pyproject.toml)
- Must maintain compatibility with PyTorch 2.8+ and Transformers 4.56+
- Code must pass ruff linting with 120-char line limit
- All type hints must pass mypy validation

### Integration Approach

**Database Integration Strategy**: N/A - No database used; preprocessed data cached locally

**API Integration Strategy**:
- GMM will integrate as new memory module type in `src/memxlnet/models/memory_modules.py`
- Exposed through `MemXLNetForQA` via new config parameters: `use_gmm_memory=True`, `num_memory_experts=4`
- No changes to external API surface; backward compatible flag-based activation
- HuggingFace Hub integration maintained for model upload/download

**Frontend Integration Strategy**: N/A - Research system with CLI/script interface

**Testing Integration Strategy**:
- New GMM tests added to `tests/unit/test_memory.py` alongside existing memory tests
- Integration tests added to `tests/integration/test_gmm_training.py`
- Reuse existing test fixtures from `tests/fixtures/` and `tests/conftest.py`
- All existing tests must continue to pass (backward compatibility verification)

### Code Organization and Standards

**File Structure Approach**:

```
src/memxlnet/models/
├── memory_modules.py         # Add GMMMemory, MemoryGatingNetwork classes
├── memxlnet_qa.py           # Extend forward() to support GMM routing
└── __init__.py              # Export new GMM classes

src/memxlnet/utils/
├── gmm_analysis.py          # NEW: GMM-specific analysis tools
└── memory_visualization.py  # EXTEND: Add routing visualization

tests/unit/
├── test_memory.py           # EXTEND: Add GMM unit tests
└── test_gmm_routing.py      # NEW: Routing-specific tests

tests/integration/
└── test_gmm_training.py     # NEW: End-to-end GMM training test

docs/guides/
└── GMM_XLNET_GUIDE.md       # NEW: GMM usage guide
```

**Naming Conventions**:
- Classes: PascalCase (e.g., `GMMMemory`, `MemoryGatingNetwork`)
- Functions/methods: snake_case (e.g., `compute_routing_probabilities`)
- Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_NUM_EXPERTS`)
- Private methods: `_leading_underscore` (e.g., `_update_expert_memory`)

**Coding Standards**:
- Follow existing patterns in `memory_modules.py` (DifferentiableMemory as reference)
- Use type hints for all public methods (mypy compliance)
- Docstrings in Google format with Args/Returns/Raises sections
- Maximum line length: 120 characters (ruff configured)
- Use `torch.nn.Module` for all learnable components

**Documentation Standards**:
- Comprehensive docstrings for all GMM classes and public methods
- Inline comments for complex routing logic
- Update MEMORY_TOKENS_GUIDE.md with GMM section
- Create standalone GMM_XLNET_GUIDE.md following existing guide structure
- Update API_REFERENCE.md with GMM configuration parameters

### Deployment and Operations

**Build Process Integration**:
- No changes required; existing `pyproject.toml` build configuration sufficient
- New GMM code automatically included via `packages = ["src/memxlnet"]`

**Deployment Strategy**:
- Local training: `uv run python scripts/phase2_train.py` with GMM config flags
- Model checkpoints saved to `outputs/` directory (existing pattern)
- HuggingFace Hub upload via `scripts/upload_checkpoint_to_hub.py` (existing script)
- No infrastructure changes required

**Monitoring and Logging**:
- Extend existing training logs to include GMM-specific metrics:
  - Routing entropy (measure of expert specialization)
  - Per-expert activation frequencies
  - Expert diversity scores
- Reuse existing logging infrastructure (Python logging module)
- Add TensorBoard support for routing probability visualization (optional enhancement)

**Configuration Management**:
- GMM parameters added to `TrainingConfig` dataclass in `src/memxlnet/training/trainer.py`
- Configuration serialized to JSON alongside model checkpoints (existing pattern)
- Environment variables: Reuse existing `HF_TOKEN` for Hub operations

### Risk Assessment and Mitigation

**Technical Risks**:

1. **Routing probability collapse** - All routing probability mass concentrates on single expert
   - *Mitigation*: Add entropy regularization term to training loss
   - *Mitigation*: Implement load balancing loss (as in Switch Transformer)

2. **Numerical instability in softmax routing** - Extreme logit values cause NaN
   - *Mitigation*: Apply temperature scaling to routing logits
   - *Mitigation*: Implement robust softmax with max normalization

3. **Expert specialization failure** - Experts learn redundant representations
   - *Mitigation*: Initialize experts with different random seeds
   - *Mitigation*: Add diversity regularization (minimize expert similarity)

4. **Increased memory consumption** - k experts × memory size could exceed GPU limits
   - *Mitigation*: Implement expert capacity limits
   - *Mitigation*: Support gradient checkpointing for memory efficiency

**Integration Risks**:

1. **Breaking backward compatibility** - Existing models fail to load
   - *Mitigation*: Comprehensive unit tests for model loading from old checkpoints
   - *Mitigation*: Version checking in `from_pretrained()` method

2. **Time-step-major batching incompatibility** - Expert state tracking errors
   - *Mitigation*: Thorough testing with multi-segment evaluation pipeline
   - *Mitigation*: Clear documentation of expert state management across segments

3. **Configuration complexity** - Too many hyperparameters confuse users
   - *Mitigation*: Provide sensible defaults (k=4, load_balance_weight=0.01)
   - *Mitigation*: Create preset configurations (gmm-small, gmm-balanced, gmm-large)

**Deployment Risks**:

1. **Checkpoint size increase** - k experts increase model size significantly
   - *Mitigation*: Support sparse expert saving (only save active experts)
   - *Mitigation*: Document checkpoint size expectations

2. **Inference latency increase** - Routing adds computational overhead
   - *Mitigation*: Implement cached routing (reuse routing from write for read)
   - *Mitigation*: Profile and optimize routing network forward pass

**Mitigation Strategies Summary**:
- **Phased Implementation**: Implement core GMM first, add optimizations incrementally
- **Extensive Testing**: Unit tests for each component, integration tests for full pipeline
- **Monitoring**: Track routing statistics during training to detect issues early
- **Documentation**: Clear troubleshooting guide with common issues and solutions
- **Fallback Options**: Always allow users to disable GMM and fall back to standard memory

---

## Epic and Story Structure

### Epic Approach

**Epic Structure Decision**: Single comprehensive epic

**Rationale**: The GMM-XLNet enhancement is a cohesive feature that modifies a single subsystem (memory management) within the existing architecture. While substantial, it has clear boundaries and dependencies that flow naturally from core infrastructure → integration → tooling. A single epic with well-sequenced stories provides:

1. **Clear dependency chain**: Each story builds on previous work (e.g., router needs GMM experts, visualization needs router)
2. **Unified testing strategy**: All stories contribute to comprehensive GMM test coverage
3. **Cohesive documentation**: Single narrative arc from implementation to usage
4. **Simplified project tracking**: Single epic clearly scopes the enhancement

Multiple epics would create artificial boundaries in a naturally connected implementation sequence.

---

## Epic 1: Gated Memory Mixture (GMM-XLNet) Implementation

**Epic Goal**: Implement a Mixture-of-Experts memory architecture for MemXLNet-QA that enables specialized memory banks with learnable routing, improving memory utilization and providing interpretable memory specialization patterns, while maintaining full backward compatibility with existing memory implementations.

**Integration Requirements**:
- Maintain compatibility with existing token-based and differentiable memory systems
- Integrate with time-step-major batching without data loader modifications
- Support existing progressive training and warmup strategies
- Enable HuggingFace Hub upload/download for GMM checkpoints
- Preserve all existing training scripts and evaluation pipelines

---

### Story 1.1: Implement Core GMM Memory Expert Infrastructure

As a **research engineer**,
I want **to implement the core multi-expert memory infrastructure**,
so that **the system can maintain multiple independent memory banks instead of a single monolithic memory state**.

#### Acceptance Criteria

1. **GMMMemory class created** in `src/memxlnet/models/memory_modules.py` with support for k=2 to k=8 experts
2. **Independent expert initialization** with configurable per-expert initialization strategies (learned, zeros, uniform)
3. **Expert state management** methods: `get_expert_state(expert_idx)`, `set_expert_state(expert_idx, state)`, `reset_experts()`
4. **Shape validation** ensuring each expert has shape `(batch_size, memory_slots, hidden_dim)`
5. **Configuration parameters** added: `num_memory_experts`, `memory_expert_init_strategies`
6. **Unit tests** covering expert initialization, state access, and shape validation

#### Integration Verification

**IV1**: Existing `DifferentiableMemory` class continues to pass all unit tests without modification
**IV2**: Existing `MemoryController` interface remains unchanged and functional
**IV3**: Module imports (`from memxlnet.models import MemXLNetForQA`) work without errors

---

### Story 1.2: Implement Memory Gating Network (Router)

As a **research engineer**,
I want **to implement a learnable gating network that routes memory updates to appropriate experts**,
so that **the model can selectively update memory experts based on information content**.

#### Acceptance Criteria

1. **MemoryGatingNetwork class created** with learnable weight matrix `W_gate ∈ R^(k × d)`
2. **Routing computation** implemented: mean-pool memory write proposals → linear projection → softmax
3. **Temperature-controlled softmax** to prevent probability collapse (configurable temperature parameter)
4. **Numerical stability** handling for edge cases (all-zero inputs, extreme logits)
5. **Entropy regularization** hook for preventing routing collapse (optional, configurable)
6. **Unit tests** for routing probability computation, temperature scaling, numerical stability

#### Integration Verification

**IV1**: Router operates independently without affecting existing memory update mechanisms
**IV2**: Routing network parameters correctly identified by optimizer groups
**IV3**: Model parameter count increases by expected amount (<100K for k=4, d=768)

---

### Story 1.3: Implement Gated Expert Updates with Router Modulation

As a **research engineer**,
I want **to implement memory update logic that modulates LSTM-style gates with router probabilities**,
so that **experts are selectively updated based on routing decisions while maintaining stable learning**.

#### Acceptance Criteria

1. **Expert-specific gated updates** implemented: `M_j^(i) = (p_j · g_j) ⊙ u_j + (1 - p_j · g_j) ⊙ M_j^(i-1)`
2. **Per-expert gate networks** created with separate parameters for each expert
3. **Routing modulation** correctly applied to gate activations before memory update
4. **Gradient flow** verified through routing probabilities to gate networks
5. **Memory protection mechanism** ensuring low-probability experts preserve their state
6. **Unit tests** for update computation, routing modulation, gradient flow

#### Integration Verification

**IV1**: Standard gated updates (non-GMM) continue to work with existing tests passing
**IV2**: Memory state shapes remain consistent across update operations
**IV3**: No memory leaks or GPU memory growth during repeated updates

---

### Story 1.4: Implement Aggregated Memory Read Operations

As a **research engineer**,
I want **to implement weighted aggregation of expert memories for read operations**,
so that **memory read tokens have access to collective knowledge from all specialized experts**.

#### Acceptance Criteria

1. **Weighted aggregation** implemented: `M_context = Σ(p_j · M_j)` for j=1 to k
2. **Routing mode support**: "write-based" (reuse write routing) and "read-based" (compute new routing)
3. **Read-specific routing** option with separate gating network for read operations
4. **Efficient computation** minimizing redundant routing calculations
5. **Memory replacement** logic to swap `[MEM_READ]` token embeddings with aggregated memory
6. **Unit tests** for aggregation computation, both routing modes, embedding replacement

#### Integration Verification

**IV1**: Existing memory read operations (token-based, differentiable) work without modification
**IV2**: Read operation latency increases by less than 30% for k=4 experts
**IV3**: Aggregated memory shapes match expected dimensions for downstream processing

---

### Story 1.5: Integrate GMM with MemXLNetForQA

As a **research engineer**,
I want **to extend MemXLNetForQA to support GMM memory operations in its forward pass**,
so that **users can enable GMM through configuration flags without code changes**.

#### Acceptance Criteria

1. **Configuration flags** added to `MemXLNetForQA.__init__`: `use_gmm_memory`, `num_memory_experts`, `gmm_routing_mode`
2. **Forward pass routing** logic integrated to switch between token-based, differentiable, and GMM memory
3. **Memory state initialization** extended to initialize expert banks when GMM enabled
4. **Output compatibility** ensuring GMM outputs match expected format (start_logits, end_logits, new_memory_state)
5. **Type hints and docstrings** updated for GMM parameters
6. **Integration test** demonstrating full forward pass with GMM enabled

#### Integration Verification

**IV1**: Models with `use_gmm_memory=False` produce identical outputs to existing models
**IV2**: All existing training scripts work without modification when GMM disabled
**IV3**: Model loading with `from_pretrained()` correctly detects memory type from config

---

### Story 1.6: Extend Training Configuration for GMM Support

As a **research engineer**,
I want **to add GMM-specific configuration parameters to TrainingConfig**,
so that **users can train GMM models through standard training scripts with configuration changes only**.

#### Acceptance Criteria

1. **TrainingConfig parameters added**: `use_gmm_memory`, `num_memory_experts`, `routing_temperature`, `entropy_regularization_weight`, `load_balance_weight`
2. **Default configurations** provided: gmm-small (k=2), gmm-balanced (k=4), gmm-large (k=8)
3. **Validation logic** ensuring parameter compatibility (e.g., k must be power of 2, temperature > 0)
4. **Warmup strategy compatibility** verified with `warmup_freeze_base_epochs`
5. **Configuration serialization** extended to save/load GMM parameters
6. **Integration test** training GMM model for 1 epoch with minimal data

#### Integration Verification

**IV1**: Existing training configurations work without modification (backward compatibility)
**IV2**: Training with GMM enabled completes without errors on toy dataset
**IV3**: Configuration JSON saved alongside checkpoint contains GMM parameters

---

### Story 1.7: Implement GMM Model Serialization and Loading

As a **research engineer**,
I want **to implement proper save/load mechanisms for GMM models**,
so that **trained GMM models can be checkpointed, resumed, and shared via HuggingFace Hub**.

#### Acceptance Criteria

1. **Save method extended** to serialize all expert states, gating network parameters, and GMM config
2. **Load method extended** to deserialize experts and reconstruct routing networks
3. **Version detection** to identify GMM vs non-GMM checkpoints during loading
4. **Backward compatibility** ensuring old checkpoints load without GMM parameters
5. **HuggingFace Hub compatibility** verified with upload/download round-trip
6. **Unit tests** for save/load with various expert counts and configurations

#### Integration Verification

**IV1**: Existing non-GMM checkpoints load correctly and evaluate with expected metrics
**IV2**: GMM checkpoints uploaded to Hub download successfully on different machines
**IV3**: Loaded GMM models produce deterministic outputs matching saved checkpoint

---

### Story 1.8: Implement GMM Interpretability and Visualization Tools

As a **research engineer**,
I want **to implement analysis tools that reveal expert specialization patterns**,
so that **researchers can understand what types of information each expert learns to store**.

#### Acceptance Criteria

1. **GMMAnalyzer class created** in `src/memxlnet/utils/gmm_analysis.py`
2. **Routing probability tracking** across document segments with export to JSON
3. **Expert activation frequency analysis** identifying which experts activate for which inputs
4. **Memory specialization metrics**: entropy, diversity, expert utilization balance
5. **Visualization functions**: routing heatmaps, expert activation timelines, specialization dendrograms
6. **Example analysis script** demonstrating interpretability workflow on sample documents

#### Integration Verification

**IV1**: Existing `MemoryVisualizer` continues to work for non-GMM models
**IV2**: Visualization functions produce valid matplotlib figures without errors
**IV3**: Analysis tools work with time-step-major batched evaluation pipeline

---

### Story 1.9: Create Comprehensive GMM Test Suite

As a **research engineer**,
I want **to implement thorough unit and integration tests for GMM functionality**,
so that **GMM implementation is robust, reliable, and maintains quality standards**.

#### Acceptance Criteria

1. **Unit tests for GMMMemory**: expert initialization, state management, shape validation
2. **Unit tests for MemoryGatingNetwork**: routing computation, numerical stability, temperature scaling
3. **Unit tests for gated updates**: routing modulation, gradient flow, memory protection
4. **Integration test**: full training loop with GMM on toy dataset (10 examples, 2 epochs)
5. **Integration test**: evaluation pipeline with GMM memory state propagation
6. **Regression test**: verify all existing tests still pass with GMM code added
7. **Test coverage**: ≥80% coverage for all GMM-related code

#### Integration Verification

**IV1**: Entire test suite passes with GMM code present but disabled (`use_gmm_memory=False`)
**IV2**: No test execution time increase >10% for non-GMM tests
**IV3**: CI/CD pipeline (if present) successfully runs all tests

---

### Story 1.10: Create GMM Documentation and Usage Guide

As a **technical writer / research engineer**,
I want **to create comprehensive documentation for GMM functionality**,
so that **users understand how to configure, train, evaluate, and interpret GMM models**.

#### Acceptance Criteria

1. **GMM_XLNET_GUIDE.md created** following existing guide structure with sections:
   - Overview and motivation
   - Configuration options
   - Training GMM models
   - Evaluating GMM models
   - Interpretability analysis
   - Troubleshooting
2. **API_REFERENCE.md updated** with GMM configuration parameters
3. **MEMORY_TOKENS_GUIDE.md extended** with GMM architecture section
4. **Example script** created: `examples/train_with_gmm_memory.py`
5. **CLAUDE.md updated** with GMM quick reference
6. **README.md updated** mentioning GMM as advanced memory option

#### Integration Verification

**IV1**: All existing documentation remains accurate with GMM code present
**IV2**: Documentation examples run without errors when copy-pasted
**IV3**: Links between documents remain valid (no broken references)

---

## Story Sequencing Rationale

This story sequence minimizes risk to the existing system through:

1. **Foundation first** (Stories 1.1-1.4): Core GMM infrastructure built independently before integration
2. **Isolated integration** (Story 1.5): Main model integration isolated as single story with clear rollback point
3. **Gradual enhancement** (Stories 1.6-1.7): Training and serialization added after core functionality proven
4. **Quality assurance** (Stories 1.8-1.10): Testing and documentation ensure robustness

**Dependencies**: Each story depends only on previous stories, enabling sequential execution by AI agents without parallel coordination requirements.

**Risk Mitigation**: Integration Verification criteria in each story ensure existing functionality remains intact throughout implementation.

---

**End of PRD**
