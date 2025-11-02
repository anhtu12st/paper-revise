# Infrastructure and Deployment Integration

## Existing Infrastructure

**Current Deployment:**
- **Training:** Local GPU execution via `uv run python scripts/phase2_train.py`
- **Checkpointing:** Local filesystem (`outputs/` directory) with incremental saves
- **Distribution:** HuggingFace Hub for trained model sharing
- **Evaluation:** Local execution with cached datasets

**Infrastructure Tools:**
- **Package Management:** uv (for fast dependency resolution and environment management)
- **Version Control:** Git (with .gitignore for outputs, cache directories)
- **Model Registry:** HuggingFace Hub (username/model-name naming convention)

**Environments:**
- **Development:** Local machine with GPU (16GB+ VRAM)
- **Production:** N/A (research system; models shared via Hub)

## Enhancement Deployment Strategy

**Deployment Approach:**
- **Identical to existing:** GMM follows exact same deployment patterns as original MemXLNet-QA
- **No infrastructure changes required:** Reuse existing training scripts, Hub integration, evaluation pipelines

**Infrastructure Changes:**
- **None required:** GMM leverages existing infrastructure completely
- **Optional enhancement:** TensorBoard integration for routing visualization (backward compatible)

**Pipeline Integration:**
- **Training script:** New `scripts/phase2_train_gmm.py` follows same structure as `scripts/phase2_train.py`
- **Evaluation:** Existing `scripts/evaluate_cls_fix.py` extended with auto-detection of GMM checkpoints
- **Hub upload:** Existing `scripts/upload_checkpoint_to_hub.py` works with GMM checkpoints unchanged

## Rollback Strategy

**Rollback Method:**
- **No rollback needed:** GMM is additive (doesn't replace existing functionality)
- **User choice:** Users can continue using original MemXLNet-QA by not installing/using GMM module

**Risk Mitigation:**
- **Parallel installation:** Both `memxlnet` and `gmmxlnet` installed simultaneously
- **Separate checkpoints:** GMM models saved with distinct naming convention (e.g., `memxlnet-gmm-k4-*`)
- **Clear documentation:** Users informed of differences and use cases

**Monitoring:**
- **Training metrics:** Existing logging infrastructure captures GMM-specific metrics (routing entropy, expert utilization)
- **Memory profiling:** psutil monitoring to track GPU memory overhead
- **Performance tracking:** Inference latency measurements to validate < 30% overhead target

**Load Balancing Thresholds:**
- **Target Expert Utilization:** 1/k ± 0.2 for k experts
  - Example: For k=4, each expert should activate ~25% ± 5% (i.e., 20-30% of the time)
  - Example: For k=8, each expert should activate ~12.5% ± 2.5% (i.e., 10-15% of the time)
- **Imbalance Warning Threshold:** Trigger warning if any expert activates < 10% or > 50% across validation set
- **Critical Imbalance:** Consider routing collapse if single expert > 80% activation (increase entropy_regularization_weight)
- **Monitoring Frequency:** Log expert utilization statistics every 100 training steps and full validation pass
- **Load Balance Loss Scaling:** Increase `load_balance_weight` from 0.01 → 0.05 if imbalance persists after 1 epoch

---
