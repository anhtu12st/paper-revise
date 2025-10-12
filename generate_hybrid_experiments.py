#!/usr/bin/env python3
"""Generate all 12 hybrid experiment scripts."""

from pathlib import Path

# Template for Standard SQuAD v2 experiments
SQUAD_TEMPLATE = '''#!/usr/bin/env python3
"""
Standard SQuAD v2 - {title}
{separator}

Dataset: squad_v2 (standard, short documents
Max segments: 2 (realistic)
Memory: {memory_desc}

Purpose: {purpose}

Output: outputs/paper_v2_squad_{exp_num}_{name}/
"""

import logging, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO)

def create_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    return TrainingConfig(
        model_name="xlnet-base-cased", max_seq_length=384, doc_stride=64,
        dataset_name="squad_v2", train_split="train", eval_split="validation",
        cache_dir="./.cache", max_train_samples=None, max_eval_samples=None,
        use_lazy_loading=False, progressive_segments=[2], max_n_segs=2,
        memory_num_tokens={mem_tokens}, memory_update="{mem_update}",
        memory_init="{mem_init}", memory_impl="token",
        use_global_softmax={global_softmax},
        num_epochs=3, train_batch_size=16, eval_batch_size=32,
        learning_rate=3e-5, weight_decay=0.01, warmup_ratio=0.1,
        max_grad_norm=1.0, gradient_accumulation_steps=1,
        eval_steps=6000, save_steps=10000, logging_steps=500,
        output_dir="./outputs/paper_v2_squad_{exp_num}_{name}",
        run_name="paper-v2-squad-{name}", save_total_limit=3,
        no_answer_threshold=1.5, use_any_positive_logic=True,
        device=device, fp16=has_cuda,
        warmup_freeze_base_epochs=0, warmup_disable_global_softmax_epochs={warmup_gs},
        warmup_disable_any_positive_epochs=0, push_to_hub_on_save=False,
    )

def main():
    print("\\n" + "=" * 80)
    print("📊 EXPERIMENT {exp_num}: SQUAD V2 - {title_upper}")
    print("=" * 80 + "\\n")
    config = create_config()
    trainer = XLNetRecurrentTrainer(config)
    try:
        trainer.train()
        print(f"\\n✅ Completed: {{config.output_dir}}")
    except Exception as e:
        print(f"\\n❌ Failed: {{e}}")
        raise

if __name__ == "__main__":
    main()
'''

# Template for Long SQuAD v2 experiments
LONG_SQUAD_TEMPLATE = '''#!/usr/bin/env python3
"""
Long SQuAD v2 - {title}
{separator}

Dataset: huutuan/long_squad_v2 (long documents, 6-12 segments)
Progressive segments: {progressive}
Memory: {memory_desc}

Purpose: {purpose}

Output: outputs/paper_v2_long_{exp_num}_{name}/
"""

import logging, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from memxlnet.training import TrainingConfig, XLNetRecurrentTrainer

logging.basicConfig(level=logging.INFO)

def create_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_cuda = torch.cuda.is_available()

    return TrainingConfig(
        model_name="xlnet-base-cased", max_seq_length=384, doc_stride=64,
        dataset_name="huutuan/long_squad_v2", train_split="train", eval_split="validation",
        cache_dir="./.cache", max_train_samples=None, max_eval_samples=None,
        use_lazy_loading=False, progressive_segments={progressive}, max_n_segs={max_segs},
        memory_num_tokens={mem_tokens}, memory_update="{mem_update}",
        memory_init="{mem_init}", memory_impl="token",
        use_global_softmax={global_softmax},
        num_epochs=3, train_batch_size=16, eval_batch_size=32,
        learning_rate=3e-5, weight_decay=0.01, warmup_ratio=0.1,
        max_grad_norm=1.0, gradient_accumulation_steps=1,
        eval_steps=6000, save_steps=10000, logging_steps=500,
        output_dir="./outputs/paper_v2_long_{exp_num}_{name}",
        run_name="paper-v2-long-{name}", save_total_limit=3,
        no_answer_threshold=1.5, use_any_positive_logic=True,
        device=device, fp16=has_cuda,
        warmup_freeze_base_epochs=0, warmup_disable_global_softmax_epochs={warmup_gs},
        warmup_disable_any_positive_epochs=0, push_to_hub_on_save=False,
    )

def main():
    print("\\n" + "=" * 80)
    print("📊 EXPERIMENT {exp_num}: LONG SQUAD V2 - {title_upper}")
    print("=" * 80 + "\\n")
    config = create_config()
    trainer = XLNetRecurrentTrainer(config)
    try:
        trainer.train()
        print(f"\\n✅ Completed: {{config.output_dir}}")
    except Exception as e:
        print(f"\\n❌ Failed: {{e}}")
        raise

if __name__ == "__main__":
    main()
'''

# Define all experiments
squad_experiments = [
    # Already created: 01_baseline_squad_no_memory
    # Already created: 02_main_squad_8tokens
    {
        "num": "03", "name": "ablation_no_gating",
        "title": "Ablation (No Gating)",
        "purpose": "Show importance of gating mechanism",
        "mem_tokens": 8, "mem_update": "none", "mem_init": "learned",
        "global_softmax": "True", "warmup_gs": 1,
        "memory_desc": "8 tokens WITHOUT gating"
    },
    {
        "num": "04", "name": "ablation_4tokens",
        "title": "Ablation (4 Tokens)",
        "purpose": "Test lower bound of memory capacity",
        "mem_tokens": 4, "mem_update": "gated", "mem_init": "learned",
        "global_softmax": "True", "warmup_gs": 1,
        "memory_desc": "4 tokens"
    },
    {
        "num": "05", "name": "ablation_16tokens",
        "title": "Ablation (16 Tokens)",
        "purpose": "Test scalability to more memory",
        "mem_tokens": 16, "mem_update": "gated", "mem_init": "learned",
        "global_softmax": "True", "warmup_gs": 1,
        "memory_desc": "16 tokens"
    },
    {
        "num": "06", "name": "ablation_32tokens",
        "title": "Ablation (32 Tokens)",
        "purpose": "Test upper bound of memory capacity",
        "mem_tokens": 32, "mem_update": "gated", "mem_init": "learned",
        "global_softmax": "True", "warmup_gs": 1,
        "memory_desc": "32 tokens"
    },
]

long_squad_experiments = [
    {
        "num": "07", "name": "baseline_no_memory",
        "title": "Baseline (No Memory)",
        "purpose": "Baseline on long documents",
        "progressive": "[2, 4, 6]", "max_segs": 6,
        "mem_tokens": 0, "mem_update": "none", "mem_init": "zeros",
        "global_softmax": "False", "warmup_gs": 0,
        "memory_desc": "None (baseline)"
    },
    {
        "num": "08", "name": "main_8tokens",
        "title": "Main (8 Tokens)",
        "purpose": "Main result on long documents",
        "progressive": "[2, 4, 6]", "max_segs": 6,
        "mem_tokens": 8, "mem_update": "gated", "mem_init": "learned",
        "global_softmax": "True", "warmup_gs": 1,
        "memory_desc": "8 tokens with gating"
    },
    {
        "num": "09", "name": "ablation_no_progressive",
        "title": "Ablation (No Progressive)",
        "purpose": "Show benefit of progressive training",
        "progressive": "[6]", "max_segs": 6,
        "mem_tokens": 8, "mem_update": "gated", "mem_init": "learned",
        "global_softmax": "True", "warmup_gs": 1,
        "memory_desc": "8 tokens, no progressive"
    },
    {
        "num": "10", "name": "ablation_no_gating",
        "title": "Ablation (No Gating)",
        "purpose": "Show importance of gating on long docs",
        "progressive": "[2, 4, 6]", "max_segs": 6,
        "mem_tokens": 8, "mem_update": "none", "mem_init": "learned",
        "global_softmax": "True", "warmup_gs": 1,
        "memory_desc": "8 tokens WITHOUT gating"
    },
    {
        "num": "11", "name": "segments_4seg",
        "title": "Medium Documents (4 segments)",
        "purpose": "Performance on medium-length documents",
        "progressive": "[2, 4]", "max_segs": 4,
        "mem_tokens": 8, "mem_update": "gated", "mem_init": "learned",
        "global_softmax": "True", "warmup_gs": 1,
        "memory_desc": "8 tokens"
    },
    {
        "num": "12", "name": "segments_6seg",
        "title": "Long Documents (6 segments)",
        "purpose": "Performance on long documents",
        "progressive": "[2, 4, 6]", "max_segs": 6,
        "mem_tokens": 8, "mem_update": "gated", "mem_init": "learned",
        "global_softmax": "True", "warmup_gs": 1,
        "memory_desc": "8 tokens"
    },
]

# Create Standard SQuAD v2 scripts
base_path = Path("scripts/paper_experiments_v2/squad")
base_path.mkdir(parents=True, exist_ok=True)

for exp in squad_experiments:
    script_content = SQUAD_TEMPLATE.format(
        exp_num=exp["num"],
        name=exp["name"],
        title=exp["title"],
        title_upper=exp["title"].upper(),
        separator="=" * len(f"Standard SQuAD v2 - {exp['title']}"),
        purpose=exp["purpose"],
        mem_tokens=exp["mem_tokens"],
        mem_update=exp["mem_update"],
        mem_init=exp["mem_init"],
        global_softmax=exp["global_softmax"],
        warmup_gs=exp["warmup_gs"],
        memory_desc=exp["memory_desc"]
    )

    script_path = base_path / f"{exp['num']}_{exp['name']}.py"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    print(f"✅ Created: {script_path}")

# Create Long SQuAD v2 scripts
base_path = Path("scripts/paper_experiments_v2/long_squad")
base_path.mkdir(parents=True, exist_ok=True)

for exp in long_squad_experiments:
    script_content = LONG_SQUAD_TEMPLATE.format(
        exp_num=exp["num"],
        name=exp["name"],
        title=exp["title"],
        title_upper=exp["title"].upper(),
        separator="=" * len(f"Long SQuAD v2 - {exp['title']}"),
        purpose=exp["purpose"],
        progressive=exp["progressive"],
        max_segs=exp["max_segs"],
        mem_tokens=exp["mem_tokens"],
        mem_update=exp["mem_update"],
        mem_init=exp["mem_init"],
        global_softmax=exp["global_softmax"],
        warmup_gs=exp["warmup_gs"],
        memory_desc=exp["memory_desc"]
    )

    script_path = base_path / f"{exp['num']}_{exp['name']}.py"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    print(f"✅ Created: {script_path}")

print("\n🎉 All experiment scripts created successfully!")
print(f"   Standard SQuAD v2: {len(squad_experiments) + 2} scripts")
print(f"   Long SQuAD v2: {len(long_squad_experiments)} scripts")
print(f"   Total: {len(squad_experiments) + len(long_squad_experiments) + 2} scripts")
