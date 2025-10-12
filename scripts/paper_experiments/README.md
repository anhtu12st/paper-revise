# Paper Experiments

This directory contains all experimental scripts for the MemXLNet-QA paper.

## 📊 Overview

**Total Experiments**: 10 training runs + evaluation + analysis
**Purpose**: Comprehensive evaluation of memory-augmented XLNet for paper publication
**Estimated Time**: 3-5 days on GPU (can run experiments in parallel)

## 🧪 Experiments

### Group 1: Baselines
- **01_baseline_no_memory.py** - Standard XLNet without memory tokens

### Group 2: Main Contributions
- **02_main_memory_8tokens.py** - Main configuration (8 memory tokens)
- **03_main_memory_16tokens.py** - Scalability test (16 memory tokens)

### Group 3: Ablation Studies
- **04_ablation_no_progressive.py** - Without progressive training
- **05_ablation_no_gating.py** - Without gated memory updates
- **06_ablation_4tokens.py** - Fewer memory tokens (4)
- **07_ablation_32tokens.py** - More memory tokens (32)

### Group 4: Segment Analysis
- **08_segments_2seg.py** - Short documents (2 segments max)
- **09_segments_4seg.py** - Medium documents (4 segments max)
- **10_segments_6seg.py** - Long documents (6 segments max)

### Evaluation & Analysis
- **evaluate_all.py** - Evaluate all trained models
- **analyze_results.py** - Generate plots and tables for paper

## 🚀 Quick Start

### Option 1: Run All Experiments (Recommended)

```bash
# Run all experiments sequentially
./scripts/run_all_experiments.sh

# This will:
# 1. Train all 10 models
# 2. Evaluate them on full SQuAD v2
# 3. Generate comparison tables and plots
```

### Option 2: Run Individual Experiments

```bash
# Run a specific experiment
uv run python scripts/paper_experiments/01_baseline_no_memory.py

# Or run multiple experiments in parallel (if you have multiple GPUs)
uv run python scripts/paper_experiments/01_baseline_no_memory.py &
uv run python scripts/paper_experiments/02_main_memory_8tokens.py &
wait
```

### Option 3: Quick Test Mode

```bash
# Test with small dataset (1000 samples)
# Good for debugging before full run
./scripts/run_all_experiments.sh --quick-test
```

## 📁 Output Structure

```
outputs/
├── paper_exp_01_baseline_no_memory/
│   └── stage_1_segs_*/
│       ├── best_model/
│       ├── final_model/
│       └── training_config.json
├── paper_exp_02_main_memory_8tokens/
│   └── ...
└── ...

results/
└── paper_experiments/
    ├── 01_baseline_no_memory_results.json
    ├── 02_main_memory_8tokens_results.json
    ├── ...
    ├── all_results.json
    ├── comparison_table.json
    ├── comparison_table.csv
    ├── comparison_table.tex
    ├── summary_statistics.json
    └── figures/
        ├── memory_tokens_comparison.pdf
        ├── ablation_comparison.pdf
        └── segment_analysis.pdf
```

## 📊 Evaluation

After training, evaluate all models:

```bash
uv run python scripts/paper_experiments/evaluate_all.py
```

This generates:
- Individual result files for each experiment
- `all_results.json` - Complete results
- `comparison_table.json/csv/tex` - Ready for paper

## 📈 Analysis

Generate publication-ready figures:

```bash
uv run python scripts/paper_experiments/analyze_results.py
```

This creates:
- **memory_tokens_comparison.pdf** - F1 vs token count
- **ablation_comparison.pdf** - Ablation study results
- **segment_analysis.pdf** - Performance vs document length
- **summary_statistics.json** - Statistical summary

## 🎯 Expected Results

Based on preliminary testing, expected F1 scores on SQuAD v2:

| Experiment | Expected F1 |
|------------|-------------|
| Baseline (No Memory) | ~72-74% |
| Main (8 tokens) | ~76-78% |
| Main (16 tokens) | ~76-79% |
| No Progressive | ~72-75% |
| No Gating | ~74-76% |
| 4 tokens | ~74-76% |
| 32 tokens | ~76-78% |

## 💡 Tips for Paper Writing

### Key Claims Supported by These Experiments

1. **Memory tokens improve performance** (Compare 01 vs 02)
2. **Progressive training helps** (Compare 02 vs 04)
3. **Gated updates are important** (Compare 02 vs 05)
4. **Optimal memory size is 8-16 tokens** (Compare 02, 03, 06, 07)
5. **Scales to long documents** (Compare 08, 09, 10)

### Paper Sections

**Results Tables**:
- Use `comparison_table.tex` directly in LaTeX
- Or format `comparison_table.csv` as needed

**Figures**:
- All figures generated as PDF (vector) and PNG (raster)
- Ready for publication in most venues

**Ablation Study**:
- Group experiments 04-07 in ablation section
- Show impact of each component

**Analysis**:
- Use segment analysis (08-10) to show scalability
- Memory token comparison (02, 03, 06, 07) for capacity analysis

## ⚙️ Configuration

All experiments use consistent hyperparameters:
- **Batch size**: 16 (reduce if OOM)
- **Learning rate**: 3e-5
- **Epochs**: 3
- **Dataset**: Full SQuAD v2
- **Optimizer**: AdamW with warmup

To modify for quick testing, edit each script's `create_config()` function:
```python
max_train_samples=1000,  # Limit training samples
num_epochs=1,            # Fewer epochs
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

Reduce batch size in experiment scripts:
```python
train_batch_size=8,  # Instead of 16
```

Or enable gradient accumulation:
```python
train_batch_size=8,
gradient_accumulation_steps=2,  # Effective batch size = 16
```

### Experiment Failed

Check the log file:
```bash
tail -100 paper_experiments.log
```

Resume from a specific experiment:
```bash
# Run remaining experiments individually
uv run python scripts/paper_experiments/05_ablation_no_gating.py
uv run python scripts/paper_experiments/06_ablation_4tokens.py
# ...
```

### Missing Dependencies

Install required packages:
```bash
uv sync
# Or
pip install -e .
```

## 📋 Checklist for Paper Submission

- [ ] Run all 10 training experiments
- [ ] Evaluate all models (`evaluate_all.py`)
- [ ] Generate figures (`analyze_results.py`)
- [ ] Verify results make sense (no 0% F1, etc.)
- [ ] Check for statistical significance
- [ ] Include comparison table in paper
- [ ] Include key figures (ablation, memory tokens)
- [ ] Report training time and resources
- [ ] Archive trained models for reproducibility

## 🔗 Related Files

- **Main README**: `../../README.md`
- **Training config**: `../../src/memxlnet/training/config.py`
- **Evaluation code**: `../../src/memxlnet/evaluation/`
- **Model code**: `../../src/memxlnet/models/`

## 📞 Support

If you encounter issues:
1. Check the experiment logs
2. Verify GPU availability: `nvidia-smi`
3. Check disk space: `df -h`
4. Review configuration in each script

## 📄 Citation

If you use these experiments in your research, please cite:

```bibtex
@article{memxlnet-qa,
  title={MemXLNet-QA: Memory-Augmented XLNet for Long-Context Question Answering},
  author={Your Name},
  year={2025}
}
```

---

**Last Updated**: 2025-01-12
**Status**: Ready for paper experiments
**Questions**: Check main README or open an issue
