# Coding Standards

## Existing Standards Compliance

GMM code must adhere to all existing MemXLNet-QA coding standards:

**Code Style:**
- **Linter:** ruff 0.1.0+ with existing configuration from `pyproject.toml`
- **Line Length:** 120 characters maximum (enforced by ruff)
- **Import Sorting:** isort via ruff (first-party: `memxlnet`, `gmmxlnet`)
- **Error Rules:** pycodestyle (E), pyflakes (F), pyupgrade (UP)

**Linting Rules:**
```toml
# From pyproject.toml (applied to GMM code)
[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
ignore = ["E501"]  # Line too long (handled by formatter)
```

**Testing Patterns:**
- **Framework:** pytest 7.4.0+ with existing configuration
- **Test Discovery:** `test_*.py` files in `tests/unit/` and `tests/integration/`
- **Naming:** `test_<functionality>` functions, `Test<Class>` classes
- **Fixtures:** Shared fixtures in `tests/conftest.py`, GMM-specific in test files
- **Markers:** `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

**Documentation Style:**
- **Docstrings:** Google-style format with Args/Returns/Raises sections
- **Type Hints:** Full type annotations for all public methods (mypy compliant)
- **Inline Comments:** Explain complex routing logic and expert update mechanics
- **Module Docstrings:** High-level overview at top of each file

**Type Checking:**
```toml
# From pyproject.toml (applied to GMM code)
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true  # For third-party libraries
```

## Enhancement-Specific Standards

### GMM-Specific Coding Rules

- **Expert Indexing:** Always use 0-based indexing for experts (j ∈ [0, k-1]); include assertions for bounds checking
- **Routing Stability:** Apply temperature scaling to all routing softmax operations; never use raw logits
- **Memory Shapes:** All expert memories maintain identical shape (batch_size, memory_slots, hidden_dim); validate in forward pass
- **Gradient Flow:** Ensure routing probabilities are differentiable (no detach() unless explicitly intended for analysis)
- **Numerical Safety:** Use torch.clamp() for routing weights to prevent NaN; add epsilon to denominators in entropy calculations
- **State Management:** Always return new memory states (never mutate in-place); facilitates time-step-major batching
- **Error Messages:** Include expert count, routing mode, and configuration in all GMM-specific error messages

### Naming Conventions (GMM-Specific)

| Element | Convention | Example |
|---------|-----------|---------|
| **Classes** | PascalCase with GMM prefix | `GMMXLNetForQA`, `MemoryGatingNetwork` |
| **Functions** | snake_case | `compute_routing_probabilities`, `update_expert_memory` |
| **Constants** | UPPER_SNAKE_CASE | `DEFAULT_NUM_EXPERTS`, `MIN_ROUTING_TEMPERATURE` |
| **Private Methods** | _leading_underscore | `_validate_expert_shapes`, `_compute_load_balance_loss` |
| **Expert Variables** | `expert_*` or `*_j` | `expert_states`, `routing_prob_j` |
| **Routing Variables** | `routing_*` | `routing_logits`, `routing_probs`, `routing_entropy` |

## Critical Integration Rules

- **Backward Compatibility:** Never modify existing `memxlnet` imports or APIs; always create new GMM-specific classes
- **Data Reuse:** Import data loading from `memxlnet.data` directly; do not duplicate tokenization or batching logic
- **Error Handling:** Raise clear exceptions when loading GMM checkpoint into non-GMM model (vice versa)
- **Configuration Validation:** Validate `num_experts` is power of 2 in [2, 8] at initialization; fail fast with helpful message
- **Memory Management:** Clear expert states between documents; reuse `reset_memory()` patterns from existing code
- **Checkpoint Compatibility:** Include `memory_type="gmm"` metadata in all saved checkpoints; check on load
- **Test Isolation:** GMM tests must not import or depend on existing MemXLNet tests (can share fixtures from conftest.py)

---
