# MemXLNet-QA Examples

This directory contains example scripts demonstrating various features of MemXLNet-QA, including base memory-augmented models and advanced GMM multi-expert models.

## Quick Reference

| Example | Description | GMM | Difficulty |
|---------|-------------|-----|------------|
| `train_with_differentiable_memory.py` | Train with content-based memory | ❌ | Beginner |
| **`train_with_gmm_memory.py`** | **Train with multi-expert memory** | ✅ | **Intermediate** |
| **`evaluate_gmm_model.py`** | **Evaluate GMM model** | ✅ | **Beginner** |
| **`analyze_gmm_experts.py`** | **Analyze expert specialization** | ✅ | **Advanced** |
| `analyze_memory_attention.py` | Visualize memory attention | ❌ | Advanced |
| `validate_answer_spans.py` | Debug answer span extraction | ❌ | Beginner |

## Base MemXLNet Examples

### 1. Training with Differentiable Memory

Demonstrates training with content-based memory using multi-head attention.

```bash
python examples/train_with_differentiable_memory.py
```

**Features**:
- Differentiable memory with content-based addressing
- Multi-head attention for memory reads
- Usage tracking and temporal links
- Memory state visualization

**Best for**: Understanding content-based memory systems

---

### 2. Memory Attention Analysis

Visualize and analyze memory attention patterns.

```bash
python examples/analyze_memory_attention.py \\
    --model-path outputs/memxlnet-model \\
    --output-dir attention_analysis
```

**Features**:
- Attention weight visualization
- Memory slot utilization analysis
- Temporal attention patterns
- Per-head attention breakdown

**Best for**: Debugging memory behavior

---

### 3. Answer Span Validation

Debug and validate answer span extraction.

```bash
python examples/validate_answer_spans.py \\
    --dataset squad_v2 \\
    --split validation
```

**Features**:
- Answer span verification
- CLS token position checking
- Token alignment validation
- Context-answer overlap analysis

**Best for**: Troubleshooting QA extraction issues

---

## GMM Multi-Expert Examples

### 4. Training with GMM Memory (★ Recommended)

Complete training workflow with multi-expert memory.

```bash
# Basic training (4 experts, 3 epochs)
python examples/train_with_gmm_memory.py \\
    --num-experts 4 \\
    --epochs 3

# High-capacity training (8 experts, longer training)
python examples/train_with_gmm_memory.py \\
    --num-experts 8 \\
    --memory-slots 32 \\
    --epochs 5 \\
    --batch-size 2

# With routing analysis
python examples/train_with_gmm_memory.py \\
    --num-experts 4 \\
    --analyze-routing
```

**Features**:
- Multi-expert memory initialization
- Routing-modulated updates
- Load balancing and entropy regularization
- Real-time routing monitoring
- Automatic checkpoint saving

**Best for**: Training production GMM models

**Command-Line Options**:
- `--num-experts`: Number of memory experts (2-8, default: 4)
- `--memory-slots`: Memory slots per expert (default: 16)
- `--routing-temperature`: Routing softmax temperature (default: 1.0)
- `--routing-mode`: "write-based" or "read-based" (default: "write-based")
- `--epochs`: Training epochs (default: 3)
- `--analyze-routing`: Enable routing analysis

**Output**:
- Model checkpoint: `outputs/gmm-xlnet-demo/final/`
- Configuration: `outputs/gmm-xlnet-demo/config.json`
- Training logs with routing statistics

---

### 5. Evaluating GMM Models

Evaluate trained GMM models and analyze performance.

```bash
# Evaluate local checkpoint
python examples/evaluate_gmm_model.py \\
    --model-path outputs/gmm-xlnet-squad/final

# Evaluate from HuggingFace Hub
python examples/evaluate_gmm_model.py \\
    --model-id username/gmm-xlnet-squad \\
    --from-hub

# With detailed routing analysis
python examples/evaluate_gmm_model.py \\
    --model-path outputs/gmm-xlnet-squad/final \\
    --analyze-routing \\
    --output-dir evaluation_results
```

**Features**:
- QA metrics computation (EM, F1)
- Routing behavior monitoring
- Expert utilization tracking
- Load balancing verification
- Results export to JSON

**Best for**: Quick model evaluation and routing sanity checks

**Command-Line Options**:
- `--model-path`: Path to local checkpoint
- `--model-id`: HuggingFace Hub model ID
- `--from-hub`: Load from Hub instead of local
- `--analyze-routing`: Compute detailed routing metrics
- `--max-examples`: Limit evaluation examples

**Output**:
- Evaluation results: `evaluation_results/evaluation_results.json`
- Routing statistics: Expert utilization, entropy, diversity

---

### 6. Analyzing Expert Specialization (★ Advanced)

Comprehensive interpretability analysis of expert behavior.

```bash
# Full analysis with visualizations
python examples/analyze_gmm_experts.py \\
    --model-path outputs/gmm-xlnet-squad/final \\
    --data-path validation \\
    --output-dir expert_analysis \\
    --max-segments 500

# Quick routing check
python examples/analyze_gmm_experts.py \\
    --model-path outputs/gmm-xlnet-squad/final \\
    --max-segments 100 \\
    --quick
```

**Features**:
- Routing probability tracking
- Expert specialization metrics
- Diversity and consistency analysis
- Comprehensive visualizations:
  - Routing heatmaps
  - Expert utilization bar charts
  - Entropy timelines
  - Similarity matrices
- Detailed HTML/JSON reports

**Best for**: Understanding what experts have learned

**Visualizations Generated**:
1. **Routing Heatmap** - Shows which experts are activated for each segment
2. **Expert Utilization** - Bar chart of expert activation frequencies
3. **Entropy Timeline** - Routing entropy over time
4. **Diversity Matrix** - Expert memory similarity (cosine)

**Metrics Computed**:
- **Expert Utilization**: Per-expert activation rates (should be balanced)
- **Routing Entropy**: H = -Σ p_j log(p_j) (moderate is good)
- **Expert Diversity**: Cosine similarity between expert memories (low is good)
- **Routing Consistency**: How often same expert is chosen (moderate is good)
- **Load Balance Loss**: Variance of expert utilization (low is good)

**Output**:
- Visualizations: `expert_analysis/*.png`
- Metrics report: `expert_analysis/analysis_report.json`

---

## Common Use Cases

### Training a GMM Model from Scratch

```bash
# Step 1: Train model
python examples/train_with_gmm_memory.py \\
    --num-experts 4 \\
    --memory-slots 16 \\
    --epochs 3 \\
    --output-dir outputs/my-gmm-model

# Step 2: Evaluate model
python examples/evaluate_gmm_model.py \\
    --model-path outputs/my-gmm-model/final \\
    --analyze-routing

# Step 3: Analyze experts
python examples/analyze_gmm_experts.py \\
    --model-path outputs/my-gmm-model/final \\
    --output-dir analysis/my-gmm-model
```

### Debugging Expert Collapse

If you observe expert collapse (one expert dominates), use analysis tools:

```bash
# 1. Check routing behavior
python examples/evaluate_gmm_model.py \\
    --model-path outputs/gmm-model/final \\
    --analyze-routing

# Look for expert utilization like [0.85, 0.05, 0.05, 0.05] (bad)
# Healthy: [0.27, 0.24, 0.26, 0.23] (good)

# 2. Visualize routing patterns
python examples/analyze_gmm_experts.py \\
    --model-path outputs/gmm-model/final \\
    --output-dir debug_analysis

# 3. Retrain with adjusted hyperparameters
python examples/train_with_gmm_memory.py \\
    --load-balance-weight 0.05 \\  # Increase from 0.01
    --routing-temperature 1.5      # Increase from 1.0
```

### Comparing Different Expert Counts

```bash
# Train models with different k
for k in 2 4 8; do
    python examples/train_with_gmm_memory.py \\
        --num-experts $k \\
        --output-dir outputs/gmm-k$k \\
        --epochs 3

    python examples/evaluate_gmm_model.py \\
        --model-path outputs/gmm-k$k/final \\
        --analyze-routing \\
        --output-dir results/gmm-k$k
done

# Compare results
cat results/gmm-k*/evaluation_results.json
```

---

## Tips and Best Practices

### For Training

1. **Start Small**: Begin with k=2 or k=4 for faster iteration
2. **Monitor Routing**: Watch for expert collapse (one expert dominating)
3. **Balance GPU Memory**:
   - k=4: 16GB GPU
   - k=8: 24GB+ GPU
4. **Tune Temperature**: If routing collapses, increase temperature (1.0 → 1.5)

### For Evaluation

1. **Use Validation Set**: Don't evaluate on training data
2. **Check Routing Health**: Expert utilization should be balanced
3. **Compare Baselines**: Evaluate base MemXLNet model for comparison
4. **Save Results**: Always export results to JSON for tracking

### For Analysis

1. **Analyze After Training**: Understanding routing helps debugging
2. **Visualize Early**: Check routing patterns after 1 epoch
3. **Compare Checkpoints**: Analyze multiple training stages
4. **Document Findings**: Save analysis reports for reproducibility

---

## Dependencies

All examples require the base MemXLNet-QA installation:

```bash
uv sync
```

Additional dependencies for specific examples:
- `matplotlib` - For visualizations (analyze examples)
- `scipy` - For hierarchical clustering (GMM analysis)
- `huggingface_hub` - For Hub model loading

---

## Troubleshooting

### Import Errors

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python examples/your_example.py
```

### CUDA Out of Memory

Reduce batch size or number of experts:
```bash
python examples/train_with_gmm_memory.py --batch-size 2 --num-experts 2
```

### Slow Training

Use write-based routing (faster than read-based):
```bash
python examples/train_with_gmm_memory.py --routing-mode write-based
```

### Expert Collapse

Increase load balance weight or temperature:
```bash
python examples/train_with_gmm_memory.py \\
    --load-balance-weight 0.05 \\
    --routing-temperature 1.5
```

---

## Further Documentation

- **[GMM XLNet Guide](../docs/guides/GMM_XLNET_GUIDE.md)** - Complete GMM documentation
- **[API Reference](../docs/api/API_REFERENCE.md)** - Full API documentation
- **[Memory Tokens Guide](../docs/guides/MEMORY_TOKENS_GUIDE.md)** - Memory system details
- **[Usage Guide](../docs/guides/ENHANCED_MA_XLNET_USAGE_GUIDE.md)** - General usage guide

---

## Contributing Examples

To contribute a new example:

1. Create a well-documented script in `examples/`
2. Add detailed docstring with usage examples
3. Include command-line arguments with `argparse`
4. Add entry to this README
5. Test the example end-to-end

See existing examples for reference patterns.
