# Testing Strategy

## Integration with Existing Tests

**Existing Test Framework:**
- **Framework:** pytest 7.4.0+ with pytest-cov for coverage
- **Organization:** `tests/unit/` for module-level tests, `tests/integration/` for end-to-end workflows
- **Markers:** `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- **Coverage Target:** >= 80% line coverage (maintained with pytest-cov)

**Test Organization:**
- **Shared fixtures:** `tests/conftest.py` provides model configs, sample data, device fixtures
- **Existing memory tests:** `tests/unit/test_memory.py` covers DifferentiableMemory (unchanged)
- **Integration tests:** `tests/integration/test_training.py` covers full training loop (unchanged)

**Coverage Requirements:**
- **Existing tests:** All must continue to pass without modifications (zero regressions)
- **GMM tests:** New GMM code must achieve >= 80% coverage independently

## New Testing Requirements

### Unit Tests for New Components

**tests/unit/test_gmm_memory.py:**
- Test `GatedMemoryMixture` expert initialization (all strategies: learned, zeros, uniform, orthogonal)
- Test expert state access (`get_expert_state`, `set_expert_state`)
- Test expert reset functionality
- Test shape validation for different batch sizes and expert counts
- Test independent expert state management (updates don't cross-contaminate)

**Coverage Target:** >= 90% (core component, critical for correctness)

---

**tests/unit/test_gmm_routing.py:**
- Test `MemoryGatingNetwork` routing computation (softmax, temperature scaling)
- Test routing probability normalization (sum to 1.0 per batch item)
- Test numerical stability (extreme logits, all-zero inputs)
- Test entropy calculation (correctness, edge cases)
- Test load balance loss computation
- Test temperature parameter effects on routing sharpness

**Coverage Target:** >= 90% (routing is critical for expert specialization)

---

**tests/unit/test_gmm_expert_updates.py:**
- Test `ExpertUpdater` per-expert gate computation (g_j, u_j)
- Test routing modulation application (p_j × g_j)
- Test memory protection (low-probability experts preserve state)
- Test gradient flow through routing probabilities
- Test update consistency across different expert counts

**Coverage Target:** >= 85%

---

**tests/unit/test_gmm_memory_read.py:**
- Test `AggregatedMemoryReader` weighted aggregation
- Test write-based vs read-based routing modes
- Test memory embedding replacement logic
- Test batched aggregation correctness

**Coverage Target:** >= 85%

---

**tests/unit/test_gmm_xlnet_qa.py:**
- Test `GMMXLNetForQA` initialization with various configurations
- Test forward pass output shapes (start_logits, end_logits, new_memory_state)
- Test memory state propagation across segments
- Test checkpoint save/load round-trip
- Test `from_pretrained` compatibility with HuggingFace Hub

**Coverage Target:** >= 80%

---

### Integration Tests

**tests/integration/test_gmm_training.py:**
- **Scope:** End-to-end training loop with GMM model on tiny dataset (10 examples, 2 epochs)
- **Validation:**
  - Training completes without errors
  - Loss decreases across epochs
  - Expert routing probabilities are valid (sum to 1, non-NaN)
  - Memory states propagate correctly across segments
  - Checkpoint save/load works
- **Infrastructure:** Reuses existing test fixtures for SQuAD data
- **Runtime:** < 2 minutes on CPU (marked with `@pytest.mark.integration`)

**Coverage Target:** Validates integration of all GMM components

---

**tests/integration/test_gmm_evaluation.py:**
- **Scope:** Full evaluation pipeline with GMM model on validation set
- **Validation:**
  - Evaluation metrics computed correctly (EM, F1)
  - Time-step-major batching works with GMM memory
  - Routing statistics collected across all examples
- **Infrastructure:** Uses cached preprocessed validation data

**Coverage Target:** Validates GMM integration with evaluation infrastructure

---

### Regression Tests

**Regression Testing Approach:**
- **Automated:** All existing tests in `tests/unit/test_memory.py` and `tests/integration/test_training.py` must pass unchanged
- **CI Integration:** pytest run on every commit; GMM tests isolated with markers
- **Coverage Tracking:** pytest-cov reports separate coverage for `memxlnet` (unchanged) and `gmmxlnet` (new)

**Existing Feature Verification:**
- **Token-based memory:** Original MemXLNet-QA tests verify no functionality changes
- **Differentiable memory:** Existing memory module tests confirm compatibility
- **Training infrastructure:** Phase2Trainer tests validate no regressions

**Manual Testing Requirements:**
- **Performance benchmarking:** Compare training time and inference latency (GMM vs baseline)
- **Memory profiling:** Monitor GPU memory usage with different expert counts
- **Routing analysis:** Visual inspection of routing heatmaps for sensible specialization patterns

---
