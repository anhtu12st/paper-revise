# Hybrid Experiment Suite for MemXLNet-QA Paper

This directory contains a comprehensive experiment suite for evaluating MemXLNet-QA across both **Standard SQuAD v2** (short documents, 1-2 segments) and **Long SQuAD v2** (long documents, 6-12 segments).

## Overview

The hybrid approach demonstrates that:
1. **Standard SQuAD v2**: MemXLNet-QA is competitive on standard-length documents
2. **Long SQuAD v2**: Memory benefits scale dramatically with document length

This provides a complete picture of the model's performance across different document lengths.

## Directory Structure

```
paper_experiments_v2/
├── squad/                      # Standard SQuAD v2 experiments (01-06)
│   ├── 01_baseline_squad_no_memory.py
│   ├── 02_main_squad_8tokens.py
│   ├── 03_ablation_no_gating.py
│   ├── 04_ablation_4tokens.py
│   ├── 05_ablation_16tokens.py
│   └── 06_ablation_32tokens.py
│
├── long_squad/                 # Long SQuAD v2 experiments (07-12)
│   ├── 07_baseline_no_memory.py
│   ├── 08_main_8tokens.py
│   ├── 09_ablation_no_progressive.py
│   ├── 10_ablation_no_gating.py
│   ├── 11_segments_4seg.py
│   └── 12_segments_6seg.py
│
├── evaluate_all.py             # Evaluate all experiments
├── analyze_results.py          # Generate publication figures
├── run_all_experiments.sh      # Master script (runs everything)
└── README.md                   # This file
```

## Experiments

### Standard SQuAD v2 Experiments (01-06)

**Dataset**: Standard SQuAD v2 (squad_v2)
**Document Length**: 1-2 segments (most documents fit in single segment)
**Configuration**: `max_n_segs=2`, `progressive_segments=[2]`

| ID | Name | Memory Tokens | Purpose |
|----|------|---------------|---------|
| 01 | Baseline (No Memory) | 0 | Baseline performance |
| 02 | Main (8 tokens) | 8 | Main contribution |
| 03 | Ablation (No Gating) | 8 | Test gating importance |
| 04 | Ablation (4 tokens) | 4 | Lower memory bound |
| 05 | Ablation (16 tokens) | 16 | Higher memory bound |
| 06 | Ablation (32 tokens) | 32 | Upper memory bound |

**Key Questions:**
- Is MemXLNet-QA competitive on standard-length documents?
- What is the optimal number of memory tokens for short documents?
- Does gating improve performance even on short documents?

### Long SQuAD v2 Experiments (07-12)

**Dataset**: Long SQuAD v2 (huutuan/long_squad_v2)
**Document Length**: 6-12 segments (truly long documents)
**Configuration**: `max_n_segs=6`, `progressive_segments=[2,4,6]`

| ID | Name | Memory Tokens | Purpose |
|----|------|---------------|---------|
| 07 | Baseline (No Memory) | 0 | Baseline on long docs |
| 08 | Main (8 tokens) | 8 | Main contribution |
| 09 | Ablation (No Progressive) | 8 | Test progressive training |
| 10 | Ablation (No Gating) | 8 | Test gating importance |
| 11 | 4 Segments | 8 | Medium-length docs |
| 12 | 6 Segments | 8 | Long docs |

**Key Questions:**
- How much do memory benefits scale with document length?
- Is progressive training necessary for long documents?
- How does performance vary with segment count?

## Quick Start

### Option 1: Run Everything (Recommended)

```bash
# Run all 12 experiments, evaluate, and generate figures
bash scripts/paper_experiments_v2/run_all_experiments.sh
```

**Time estimate**: 24-48 hours on GPU (depends on hardware)

### Option 2: Run Specific Experiments

```bash
# Run only experiments 01 and 02
bash scripts/paper_experiments_v2/run_all_experiments.sh --experiments "01 02"

# Run only evaluation and analysis (skip training)
bash scripts/paper_experiments_v2/run_all_experiments.sh --skip-training
```

### Option 3: Manual Execution

```bash
# 1. Run individual experiments
python scripts/paper_experiments_v2/squad/01_baseline_squad_no_memory.py
python scripts/paper_experiments_v2/squad/02_main_squad_8tokens.py
# ... (run all experiments)

# 2. Evaluate all experiments
python scripts/paper_experiments_v2/evaluate_all.py

# 3. Generate figures
python scripts/paper_experiments_v2/analyze_results.py
```

## Output Structure

After running experiments, outputs are organized as follows:

```
outputs/
├── paper_v2_squad_01_baseline_no_memory/
│   ├── training_config.json
│   ├── best_model/
│   ├── evaluation_results.json
│   └── trainer_state.json
│
├── paper_v2_squad_02_main_8tokens/
│   └── ...
│
├── paper_v2_long_07_baseline_no_memory/
│   └── ...
│
├── paper_v2_evaluation_results/
│   ├── all_results.json              # Combined results
│   ├── squad_v2_comparison.csv       # Standard SQuAD v2 table
│   ├── long_squad_v2_comparison.csv  # Long SQuAD v2 table
│   ├── combined_comparison.csv       # All experiments
│   ├── squad_v2_table.tex            # LaTeX table
│   ├── long_squad_v2_table.tex       # LaTeX table
│   └── figures/
│       ├── squad_v2_comparison.{pdf,png}
│       ├── long_squad_v2_comparison.{pdf,png}
│       ├── memory_scaling.{pdf,png}
│       ├── segment_analysis.{pdf,png}
│       ├── ablation_study.{pdf,png}
│       └── combined_improvement.{pdf,png}
│
└── paper_v2_logs/
    ├── 01_baseline_squad_no_memory.log
    ├── 02_main_squad_8tokens.log
    └── ...
```

## Generated Figures

The `analyze_results.py` script generates publication-quality figures:

1. **squad_v2_comparison.{pdf,png}**
   - Bar chart comparing all Standard SQuAD v2 experiments
   - Shows F1 and EM scores for each configuration

2. **long_squad_v2_comparison.{pdf,png}**
   - Bar chart comparing all Long SQuAD v2 experiments
   - Demonstrates memory scaling on long documents

3. **memory_scaling.{pdf,png}**
   - Line plot showing performance vs memory token count
   - Helps identify optimal memory size

4. **segment_analysis.{pdf,png}**
   - Line plot showing performance vs document length
   - Demonstrates how memory benefits scale with segments

5. **ablation_study.{pdf,png}**
   - Bar chart comparing ablation configurations
   - Shows importance of gating and progressive training

6. **combined_improvement.{pdf,png}** ⭐ **Key Finding**
   - Bar chart showing F1 improvement over baseline for both datasets
   - Highlights how memory benefits scale with document length

## Expected Results

### Standard SQuAD v2 (Short Documents)

| Experiment | Expected F1 | Notes |
|------------|-------------|-------|
| Baseline (No Memory) | ~75-77% | Competitive baseline |
| Main (8 tokens) | ~77-79% | **+2-4% improvement** |
| No Gating | ~76-78% | Gating helps even on short docs |
| Memory Scaling | Varies | Optimal around 8-16 tokens |

**Key Insight**: MemXLNet-QA is competitive on standard benchmarks while adding minimal overhead.

### Long SQuAD v2 (Long Documents)

| Experiment | Expected F1 | Notes |
|------------|-------------|-------|
| Baseline (No Memory) | ~65-70% | Baseline struggles on long docs |
| Main (8 tokens) | ~75-80% | **+10-15% improvement** |
| No Progressive | ~72-75% | Progressive training helps |
| Segment Analysis | Varies | Benefits scale with length |

**Key Insight**: Memory benefits scale dramatically with document length, achieving 2-3x larger improvements on long documents compared to short documents.

## Configuration Details

### Standard SQuAD v2 Configuration

```python
TrainingConfig(
    dataset_name="squad_v2",
    train_split="train",
    eval_split="validation",

    # Realistic segment configuration
    max_n_segs=2,                    # Most docs fit in 1-2 segments
    progressive_segments=[2],         # No progressive needed

    # Memory configuration (varies by experiment)
    memory_num_tokens=8,
    memory_update="gated",
    memory_init="learned",

    # Training settings
    num_epochs=3,
    train_batch_size=16,
    eval_batch_size=32,
    learning_rate=3e-5,

    # Phase 2 warmup
    warmup_freeze_base_epochs=0,
    warmup_disable_global_softmax_epochs=1,
)
```

### Long SQuAD v2 Configuration

```python
TrainingConfig(
    dataset_name="huutuan/long_squad_v2",
    train_split="train",
    eval_split="validation",

    # Long document configuration
    max_n_segs=6,                     # Support 6-12 segment docs
    progressive_segments=[2, 4, 6],   # Progressive curriculum

    # Memory configuration (varies by experiment)
    memory_num_tokens=8,
    memory_update="gated",
    memory_init="learned",

    # Training settings
    num_epochs=3,
    train_batch_size=16,
    eval_batch_size=32,
    learning_rate=3e-5,

    # Phase 2 warmup
    warmup_freeze_base_epochs=0,
    warmup_disable_global_softmax_epochs=1,
)
```

## Troubleshooting

### Out of Memory Errors

**Problem**: GPU OOM during training
**Solutions**:
```python
# Reduce batch size
train_batch_size=8  # or 4

# Reduce gradient accumulation
gradient_accumulation_steps=1

# Reduce memory tokens
memory_num_tokens=4  # or 0 for baseline
```

### Training Takes Too Long

**Problem**: Each experiment takes 12+ hours
**Solutions**:
```bash
# Run fewer experiments
bash run_all_experiments.sh --experiments "01 02 07 08"

# Use max_train_samples for testing
# Edit experiment scripts to add:
max_train_samples=1000  # Quick test
```

### Evaluation Fails

**Problem**: `evaluate_all.py` fails
**Solutions**:
```bash
# Check if training completed
ls outputs/paper_v2_*/training_config.json

# Re-run failed experiments manually
python scripts/paper_experiments_v2/squad/01_baseline_squad_no_memory.py

# Skip failed experiments in evaluation
# (evaluate_all.py handles missing results gracefully)
```

### Figures Look Wrong

**Problem**: Generated figures have missing data
**Solutions**:
```bash
# Ensure all experiments completed
python scripts/paper_experiments_v2/evaluate_all.py

# Check evaluation results
cat outputs/paper_v2_evaluation_results/all_results.json

# Regenerate figures
python scripts/paper_experiments_v2/analyze_results.py
```

## Paper Integration

### Tables

Use the generated LaTeX tables in your paper:

```latex
% Standard SQuAD v2 results
\input{outputs/paper_v2_evaluation_results/squad_v2_table.tex}

% Long SQuAD v2 results
\input{outputs/paper_v2_evaluation_results/long_squad_v2_table.tex}
```

### Figures

Include the generated figures:

```latex
\begin{figure}
  \centering
  \includegraphics[width=\linewidth]{outputs/paper_v2_evaluation_results/figures/combined_improvement.pdf}
  \caption{Memory benefits scale with document length. MemXLNet-QA achieves
           modest improvements (+2-4\%) on standard documents but dramatic
           improvements (+10-15\%) on long documents.}
  \label{fig:combined_improvement}
\end{figure}
```

### Key Claims Supported

1. **Competitive on Standard Benchmarks**: Standard SQuAD v2 results (Exp 01-06)
2. **Scales with Document Length**: Combined improvement plot
3. **Memory Token Efficiency**: Memory scaling analysis
4. **Importance of Design Choices**: Ablation studies (gating, progressive training)

## Next Steps

After completing experiments:

1. **Review Results**: Check `outputs/paper_v2_evaluation_results/`
2. **Verify Key Findings**: Ensure improvements match expectations
3. **Select Figures**: Choose best figures for paper (recommend combined_improvement.pdf)
4. **Write Results Section**: Use tables and figures to support claims
5. **Update README**: Add actual results to this file

## Differences from Original Plan

This hybrid approach differs from the initial `paper_experiments/` in key ways:

| Aspect | Original | Hybrid (v2) |
|--------|----------|-------------|
| Datasets | Only SQuAD v2 | SQuAD v2 + Long SQuAD v2 |
| Max Segments | 6 (unrealistic) | 2 (realistic) for standard, 6 for long |
| Progressive | Always [2,4,6] | [2] for standard, [2,4,6] for long |
| Segment Analysis | All experiments | Only long experiments |
| Key Finding | Modest improvements | **Scaling with document length** |

## Contact & Support

For issues or questions:
1. Check experiment logs in `outputs/paper_v2_logs/`
2. Review configuration in `training_config.json`
3. See main project documentation in `docs/`

## License

Part of the MemXLNet-QA project. See repository root for license information.
