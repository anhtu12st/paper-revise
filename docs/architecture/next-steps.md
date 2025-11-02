# Next Steps

## Story Manager Handoff

**Handoff to Product Owner / Story Manager:**

This GMM-XLNet brownfield enhancement architecture is ready for story breakdown and implementation planning. Key points for story management:

**Architecture Reference:**
- Use this document (`docs/gmm-architecture.md`) as the definitive technical blueprint
- Reference existing MemXLNet-QA docs (especially `docs/guides/MEMORY_TOKENS_GUIDE.md`) for context on token-based memory system
- PRD available at `docs/prd.md` with detailed epic and story structure already defined

**Key Integration Requirements (Validated):**
1. **Zero modifications to existing code:** All new functionality in `src/gmmxlnet/` module
2. **Reuse via imports:** GMM code imports from `memxlnet.data`, `memxlnet.evaluation` directly
3. **Parallel testing:** New GMM tests in separate files; existing tests must pass unchanged
4. **Backward compatibility:** Existing checkpoints load and evaluate correctly without modifications
5. **Hub compatibility:** GMM checkpoints use same serialization format (with metadata flag)

**Existing System Constraints (Based on Project Analysis):**
- **Python 3.12+ required:** No lower versions supported
- **PyTorch 2.8+ dependency:** GMM must use compatible tensor operations
- **120-char line limit:** Enforced by ruff linter (from `pyproject.toml`)
- **80% test coverage:** Required for all new GMM code
- **Type hints mandatory:** All public methods must pass mypy validation
- **Time-step-major batching:** GMM memory state propagation must respect existing batching strategy
- **CLS at end:** XLNet places CLS token at sequence end (not position 0); GMM must respect this

**First Story to Implement:**
**Story 1.1: Implement Core GMM Memory Expert Infrastructure** (from PRD Epic 1)

**Recommended Implementation Sequence (from PRD):**
1. Core infrastructure (Stories 1.1-1.4): Build GMM components independently
2. Model integration (Story 1.5): Integrate into `GMMXLNetForQA`
3. Training integration (Stories 1.6-1.7): Extend training config and serialization
4. Tooling (Stories 1.8-1.10): Add analysis tools, tests, documentation

**Clear Integration Checkpoints:**
- After Story 1.1: Verify `GatedMemoryMixture` unit tests pass
- After Story 1.4: Verify all component unit tests pass independently
- After Story 1.5: Verify `GMMXLNetForQA` forward pass works on toy data
- After Story 1.7: Verify checkpoint save/load round-trip
- After Story 1.9: Verify all existing tests still pass (zero regressions)

**Emphasis on Maintaining Existing System Integrity:**
- Every story includes "Integration Verification" acceptance criteria ensuring backward compatibility
- Regression tests must pass at each checkpoint (existing functionality preserved)
- Clear separation: GMM code lives in `src/gmmxlnet/`, imports from `src/memxlnet/` (never vice versa)

---

## Developer Handoff

**Handoff to Development Agent:**

This architecture is ready for implementation following the story sequence defined in the PRD. Key technical guidance:

**Architecture and Standards Reference:**
- **Primary:** This document (`docs/gmm-architecture.md`)
- **Coding Standards:** Section "Coding Standards" above + existing ruff/mypy config in `pyproject.toml`
- **Existing Patterns:** Study `src/memxlnet/models/memory_modules.py` (DifferentiableMemory) as reference for memory management patterns
- **Type Hints:** All public methods must have complete type annotations (mypy configured in `pyproject.toml`)

**Integration Requirements (Validated with Project Analysis):**
- **Import pattern:** `from memxlnet.data import ChunkedSquadDataset` (reuse existing data loading)
- **No modifications:** Zero changes to any file in `src/memxlnet/` (absolute requirement)
- **Parallel module:** All GMM code in `src/gmmxlnet/` with parallel structure to `src/memxlnet/`
- **Testing isolation:** GMM tests in separate files (`test_gmm_*.py`); existing tests must not be modified

**Key Technical Decisions (Based on Real Project Constraints):**
- **PyTorch 2.8+:** Use native PyTorch operations (no custom CUDA kernels)
- **Shape conventions:** Expert memories are `(batch_size, memory_slots, hidden_dim)` matching existing memory shape
- **Routing computation:** Softmax over logits with temperature scaling (default temp=1.0)
- **Gating formula:** `M_j^(i) = (p_j * g_j) ⊙ u_j + (1 - p_j * g_j) ⊙ M_j^(i-1)` (from PRD mathematical spec)
- **Aggregated read:** `M_context = Σ(p_j * M_j)` for j in [0, k-1]

**Existing System Compatibility (with Specific Verification Steps):**
1. **Before starting:** Run `pytest tests/` to establish baseline (all tests passing)
2. **After each component:** Run `pytest tests/unit/test_gmm_*.py` to verify new tests pass
3. **After integration:** Run `pytest tests/` again to verify zero regressions (all existing tests still pass)
4. **Before PR:** Run `ruff check src/gmmxlnet/` and `mypy src/gmmxlnet/` to validate code quality

**Clear Sequencing (Minimize Risk to Existing Functionality):**
1. **Phase 1 (Stories 1.1-1.4):** Implement GMM components in isolation (no integration with existing code)
   - Test each component independently
   - No risk to existing functionality (new code not imported anywhere)
2. **Phase 2 (Story 1.5):** Create `GMMXLNetForQA` that imports but doesn't modify existing code
   - Test forward pass on toy data
   - Existing code untouched (GMM is separate model class)
3. **Phase 3 (Stories 1.6-1.7):** Extend training infrastructure via inheritance
   - `GMMTrainingConfig` extends existing `TrainingConfig`
   - Existing training scripts work unchanged
4. **Phase 4 (Stories 1.8-1.10):** Add tooling and documentation
   - No code changes, only additions (tests, docs, analysis scripts)

**Critical Validation Points:**
- **Memory shapes:** Assert expert memory shapes match expected dimensions in forward pass
- **Routing probabilities:** Assert routing probs sum to 1.0 and are non-negative
- **Gradient flow:** Verify gradients flow through routing network (no accidental detach)
- **Checkpoint compatibility:** Test save/load round-trip produces identical model outputs

---

**Architecture Status:** ✅ Complete and ready for validation

**Next Action:** Run architect checklist validation (Section "Checklist Results Report")
