# Tech Stack

## Existing Technology Stack

The GMM enhancement will build upon the existing, well-established technology stack without introducing new dependencies.

| Category | Current Technology | Version | Usage in Enhancement | Notes |
|----------|-------------------|---------|---------------------|-------|
| **Language** | Python | 3.12+ | All GMM code | Maintain existing version requirement |
| **Deep Learning Framework** | PyTorch | 2.8.0+ | GMM memory experts, gating network | Core framework for all learnable components |
| **Transformers** | HuggingFace Transformers | 4.56.2+ | XLNet base model integration | Reuse existing model loading patterns |
| **Datasets** | HuggingFace Datasets | 4.1.1+ | Data loading via existing pipelines | No changes to data processing |
| **Model Hub** | HuggingFace Hub | 0.20.0+ | GMM checkpoint distribution | Same serialization format |
| **Testing Framework** | pytest | 7.4.0+ | GMM unit and integration tests | Extend existing test suite |
| **Test Coverage** | pytest-cov | 4.1.0+ | GMM test coverage tracking | Maintain >= 80% coverage |
| **Linter** | ruff | 0.1.0+ | GMM code quality | 120-char line limit |
| **Type Checker** | mypy | 1.7.0+ | GMM type validation | Strict type hints required |
| **Package Manager** | uv | Latest | Development environment | Fast dependency resolution |
| **Visualization** | matplotlib | 3.10.6+ | GMM routing visualization | Extend existing memory visualization |
| **System Monitoring** | psutil | 7.1.0+ | Memory profiling | Monitor GMM overhead |
| **Environment Management** | python-dotenv | 1.1.1+ | Configuration | Reuse existing patterns |
| **Notebook Support** | ipykernel | 6.30.1+ | GMM analysis notebooks | Optional for research |

## New Technology Additions

**No new external dependencies required.** All GMM functionality will be implemented using the existing tech stack (primarily PyTorch for the gating network and expert memory management).

**Rationale:** Minimizing dependencies reduces integration risk, maintains backward compatibility, and simplifies deployment. PyTorch provides all necessary primitives for MoE-style routing and multi-expert memory management.

---
