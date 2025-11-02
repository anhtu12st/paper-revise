# Source Tree

## Existing Project Structure (Preserved)

The current MemXLNet-QA structure remains completely untouched:

```
paper-revise/
├── src/
│   └── memxlnet/                          # EXISTING - NO MODIFICATIONS
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── memory_modules.py          # DifferentiableMemory, MemoryController
│       │   └── memxlnet_qa.py             # MemXLNetForQA (original)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py                 # Time-step-major batching
│       │   ├── chunked_dataset.py         # Document segmentation
│       │   ├── streaming.py               # Streaming data loading
│       │   └── text_utils.py              # Text processing utilities
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py                 # Phase2Trainer, TrainingConfig
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── evaluator.py               # QA evaluation metrics
│       └── utils/
│           ├── __init__.py
│           ├── memory_visualization.py    # Memory analysis plots
│           └── multihop_utils.py          # Multi-hop QA utilities
├── scripts/                                # EXISTING - PRESERVED
│   ├── phase2_train.py                    # Main training script (original)
│   ├── evaluate_cls_fix.py                # Evaluation script (extended)
│   └── upload_checkpoint_to_hub.py        # Hub upload (works with both)
├── tests/                                  # EXISTING - EXTENDED
│   ├── conftest.py                        # Shared test fixtures
│   ├── fixtures/                          # Test data
│   ├── unit/
│   │   └── test_memory.py                 # Memory module tests (extended)
│   └── integration/
│       └── test_training.py               # Training integration tests
├── docs/                                   # EXISTING - EXTENDED
│   ├── prd.md                             # GMM enhancement PRD
│   ├── gmm-architecture.md                # This document
│   ├── api/
│   │   └── API_REFERENCE.md               # Extended with GMM APIs
│   ├── guides/
│   │   ├── MEMORY_TOKENS_GUIDE.md         # Extended with GMM section
│   │   └── GMM_XLNET_GUIDE.md             # NEW - GMM usage guide
│   └── technical/
│       └── MA_XLNET_IMPLEMENTATION.md     # Original architecture reference
├── pyproject.toml                          # EXISTING - NO CHANGES TO DEPS
├── CLAUDE.md                               # EXISTING - UPDATED WITH GMM REF
└── README.md                               # EXISTING - UPDATED WITH GMM MENTION
```

## New File Organization (GMM Module)

The GMM enhancement creates a parallel module structure:

```
paper-revise/
├── src/
│   ├── memxlnet/                          # EXISTING (unchanged)
│   └── gmmxlnet/                          # ✨ NEW GMM MODULE
│       ├── __init__.py                    # Export GMMXLNetForQA, GMMConfig
│       ├── models/
│       │   ├── __init__.py                # Export all GMM model components
│       │   ├── config.py                  # GMMConfig class
│       │   ├── memory_mixture.py          # GatedMemoryMixture (k expert banks)
│       │   ├── gating_network.py          # MemoryGatingNetwork (router)
│       │   ├── expert_updates.py          # ExpertUpdater (modulated gates)
│       │   ├── memory_read.py             # AggregatedMemoryReader
│       │   └── gmm_xlnet_qa.py            # GMMXLNetForQA (main model)
│       ├── data/                          # SYMLINK OR IMPORT WRAPPER
│       │   ├── __init__.py                # Re-export from memxlnet.data
│       │   └── README.md                  # "See memxlnet.data for data loading"
│       ├── training/
│       │   ├── __init__.py                # Export GMMTrainingConfig, GMMTrainer
│       │   ├── config.py                  # GMMTrainingConfig (extends TrainingConfig)
│       │   └── trainer.py                 # GMMTrainer (extends Phase2Trainer)
│       ├── evaluation/                    # SYMLINK OR IMPORT WRAPPER
│       │   ├── __init__.py                # Re-export from memxlnet.evaluation
│       │   └── README.md                  # "See memxlnet.evaluation"
│       └── utils/
│           ├── __init__.py                # Export GMM-specific utilities
│           ├── gmm_analysis.py            # Expert specialization analysis
│           └── routing_visualization.py   # Routing heatmaps, timelines
├── scripts/
│   ├── phase2_train.py                    # EXISTING (unchanged)
│   ├── phase2_train_gmm.py                # ✨ NEW - GMM training script
│   ├── evaluate_cls_fix.py                # EXTENDED - Auto-detect GMM models
│   └── analyze_gmm_routing.py             # ✨ NEW - Routing analysis script
├── tests/
│   ├── unit/
│   │   ├── test_memory.py                 # EXISTING (unchanged)
│   │   ├── test_gmm_memory.py             # ✨ NEW - GMMMemory tests
│   │   ├── test_gmm_routing.py            # ✨ NEW - Router tests
│   │   └── test_gmm_expert_updates.py     # ✨ NEW - Expert update tests
│   └── integration/
│       ├── test_training.py               # EXISTING (unchanged)
│       └── test_gmm_training.py           # ✨ NEW - End-to-end GMM test
├── examples/
│   └── train_with_gmm_memory.py           # ✨ NEW - GMM training example
└── docs/
    └── guides/
        └── GMM_XLNET_GUIDE.md             # ✨ NEW - Comprehensive GMM guide
```

## Integration Guidelines

**File Naming:**
- **Existing files:** Unchanged (maintain current conventions)
- **New GMM files:** Use descriptive names with `gmm_` prefix where ambiguous (e.g., `gmm_xlnet_qa.py`, `gmm_analysis.py`)
- **Class naming:** Prefix with `GMM` for new classes (e.g., `GMMXLNetForQA`, `GMMConfig`)

**Folder Organization:**
- **Parallel structure:** `src/gmmxlnet/` mirrors `src/memxlnet/` layout for discoverability
- **Reuse via imports:** GMM code imports from `memxlnet.data`, `memxlnet.evaluation` directly (no duplication)
- **Separate utilities:** GMM-specific analysis tools in `gmmxlnet/utils/` to avoid polluting existing utils

**Import/Export Patterns:**
- **From GMM code:** `from memxlnet.data import ChunkedSquadDataset` (reuse existing)
- **From existing code:** No imports of GMM (maintains isolation)
- **User-facing imports:** `from gmmxlnet import GMMXLNetForQA` (clean public API)

---
