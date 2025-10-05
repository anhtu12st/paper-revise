# Usage Examples and Troubleshooting Guide

Complete examples for using MemXLNet-QA, from basic evaluation to advanced training scenarios.

## 🚀 Quick Start Examples

### Basic Model Loading and Evaluation

```python
# Load a trained MemXLNet-QA model
from src.memxlnet_qa import MemXLNetForQA
from transformers import XLNetTokenizerFast

# Load model and tokenizer (with memory tokens)
model = MemXLNetForQA.from_pretrained("outputs/xlnet-squad-phase2-1/stage_2_segs_2/best_model")
tokenizer = XLNetTokenizerFast.from_pretrained("outputs/xlnet-squad-phase2-1/stage_2_segs_2/best_model")

print(f"Model loaded with {model.mem_token_count} memory tokens")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
```

### Simple Question Answering

```python
import torch
from src.data import create_evaluation_dataloader

# Create evaluation pipeline
eval_dataset, eval_dataloader = create_evaluation_dataloader(
    dataset_name="squad_v2",
    split="validation",
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=64,
    batch_size=8,
    max_examples=100,  # Limit for quick testing
    use_time_step_major=True
)

# Simple evaluation without memory (for comparison)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for time_step_batches in eval_dataloader:
        # Process first batch only for demo
        batch = time_step_batches[0]

        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass without memory
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_state=None  # No memory for simple example
        )

        print(f"Start logits shape: {outputs['start_logits'].shape}")
        print(f"End logits shape: {outputs['end_logits'].shape}")
        break  # Just show one batch
```

### Memory-Enabled Evaluation

```python
# Evaluation with memory state propagation
def evaluate_with_memory_simple(model, eval_dataloader, device, max_docs=10):
    """Simple memory-enabled evaluation example."""
    model.eval()
    memory_bank = {}
    processed_docs = 0

    with torch.no_grad():
        for time_step_batches in eval_dataloader:
            if processed_docs >= max_docs:
                break

            for batch in time_step_batches:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                example_ids = batch['example_ids']
                document_mask = batch['document_mask'].to(device)

                # Get memory states for this batch
                batch_memory = []
                for example_id in example_ids:
                    if example_id in memory_bank:
                        batch_memory.append(memory_bank[example_id])
                    else:
                        # Initialize new memory
                        init_memory = model.get_initial_memory(1, device)
                        batch_memory.append(init_memory)

                # Stack memories
                memory_state = torch.cat(batch_memory, dim=0)

                # Forward pass with memory
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    memory_state=memory_state
                )

                # Update memory bank
                new_memory = outputs["new_memory_state"]
                for i, example_id in enumerate(example_ids):
                    if document_mask[i]:  # Only active documents
                        memory_bank[example_id] = new_memory[i:i+1]

            processed_docs += len(set(time_step_batches[0]['example_ids']))

    print(f"✅ Processed {processed_docs} documents with memory")
    print(f"Memory bank contains {len(memory_bank)} document states")

# Run memory evaluation
evaluate_with_memory_simple(model, eval_dataloader, device, max_docs=5)
```

## 🔧 Training Examples

### Basic Training Setup

```python
from src.train import TrainingConfig, XLNetRecurrentTrainer

# Basic training configuration
config = TrainingConfig(
    # Model and data
    model_name="xlnet-base-cased",
    dataset_name="squad_v2",
    max_seq_length=384,
    doc_stride=64,

    # Memory system (disabled for basic example)
    memory_num_tokens=0,  # Disable memory for basic training

    # Training parameters
    num_epochs=1,
    train_batch_size=16,
    eval_batch_size=16,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    max_grad_norm=1.0,

    # Limit data for quick testing
    max_train_samples=1000,
    max_eval_samples=100,

    # Output
    output_dir="./outputs/basic-training",
    run_name="basic-example",

    # Evaluation frequency
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
)

# Initialize and run trainer
trainer = XLNetRecurrentTrainer(config)
trainer.train()
```

### Memory-Enabled Training

```python
# Memory-enabled training configuration
memory_config = TrainingConfig(
    # Model and data
    model_name="xlnet-base-cased",
    dataset_name="squad_v2",
    max_seq_length=384,
    doc_stride=64,

    # Memory system - ENABLED
    memory_num_tokens=8,           # Enable 8 memory tokens
    memory_update="gated",         # Use gated updates
    memory_init="learned",         # Learnable initial memory
    memory_impl="token",           # Token-based implementation

    # Progressive training
    progressive_segments=[1, 2],   # Start with 1 segment, then 2
    max_n_segs=2,                  # Maximum 2 segments per document
    bptt_horizon=4,                # Backprop through 4 time steps

    # Training parameters
    num_epochs=2,
    train_batch_size=8,            # Smaller batch for memory
    eval_batch_size=8,
    learning_rate=3e-5,
    warmup_ratio=0.1,

    # Warmup behavior
    warmup_freeze_base_epochs=0.5,          # Freeze base for half epoch
    warmup_disable_global_softmax_epochs=1, # Disable global softmax initially
    warmup_disable_any_positive_epochs=1,   # Disable any-positive initially

    # Advanced features
    use_global_softmax=True,       # Enable global span selection
    use_any_positive_logic=True,   # Enable any-positive logic
    no_answer_threshold=1.5,       # SQuAD v2 threshold

    # Limited data for example
    max_train_samples=500,
    max_eval_samples=50,

    # Output
    output_dir="./outputs/memory-training",
    run_name="memory-example",
)

# Initialize memory-enabled trainer
memory_trainer = XLNetRecurrentTrainer(memory_config)
memory_trainer.train()
```

### Custom Memory Configuration

```python
# Advanced memory configuration
advanced_config = TrainingConfig(
    # Base configuration
    model_name="xlnet-base-cased",
    dataset_name="squad_v2",

    # Advanced memory system
    memory_num_tokens=16,          # More memory tokens
    memory_update="gated",         # Gated updates
    memory_init="learned",         # Learnable initialization

    # Progressive curriculum learning
    progressive_segments=[1, 2, 4, 6],  # Curriculum: 1→2→4→6 segments
    max_n_segs=6,                       # Up to 6 segments
    bptt_horizon=8,                     # Longer backprop horizon

    # Training parameters
    num_epochs=4,                  # More epochs for complex training
    train_batch_size=12,
    eval_batch_size=8,
    learning_rate=2e-5,            # Lower learning rate
    weight_decay=0.01,
    warmup_ratio=0.15,             # Longer warmup

    # Gradient control
    max_grad_norm=1.0,
    gradient_accumulation_steps=2,  # Accumulate gradients

    # Mixed precision
    fp16=True,                     # Enable FP16

    # Output and evaluation
    output_dir="./outputs/advanced-memory",
    run_name="advanced-memory-training",
    eval_steps=1000,
    save_steps=2000,
    logging_steps=200,
    save_total_limit=3,
)

print("Advanced configuration created:")
print(f"- Memory tokens: {advanced_config.memory_num_tokens}")
print(f"- Progressive segments: {advanced_config.progressive_segments}")
print(f"- Max segments: {advanced_config.max_n_segs}")
print(f"- BPTT horizon: {advanced_config.bptt_horizon}")
```

## 🎯 Data Processing Examples

### Custom Dataset Processing

```python
from src.data import SquadLikeQADataset, configure_memory_tokens
from transformers import XLNetTokenizerFast

# Setup tokenizer with memory tokens
tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
mem_config = configure_memory_tokens(tokenizer, memory_num_tokens=4)

print("Memory token configuration:")
print(f"Read tokens: {mem_config['mem_read_ids']}")
print(f"Write tokens: {mem_config['mem_write_ids']}")

# Create custom dataset
custom_dataset = SquadLikeQADataset(
    split="validation",
    tokenizer=tokenizer,
    max_seq_length=512,        # Longer sequences
    doc_stride=128,            # Larger stride
    max_examples=200,          # Limited examples
    dataset_name="squad_v2",
    max_n_segs=4              # Up to 4 segments per document
)

print(f"Dataset created with {len(custom_dataset)} features")
print(f"Documents: {len(custom_dataset.get_all_documents())}")

# Examine document structure
for doc_id in custom_dataset.get_all_documents()[:3]:
    segments = custom_dataset.get_document_segments(doc_id)
    print(f"Document {doc_id}: {len(segments)} segments at indices {segments}")
```

### Time-Step-Major DataLoader Example

```python
from src.data import TimeStepMajorDataLoader

# Create time-step-major dataloader
tsm_dataloader = TimeStepMajorDataLoader(
    dataset=custom_dataset,
    batch_size=4,              # 4 documents per batch
    shuffle=True,
    max_segments=3             # Process up to 3 segments
)

print(f"Time-step-major dataloader created: {len(tsm_dataloader)} batches")

# Examine structure
for i, time_step_batches in enumerate(tsm_dataloader):
    print(f"\nDocument batch {i+1}:")
    print(f"  Time steps: {len(time_step_batches)}")

    for t, batch in enumerate(time_step_batches):
        print(f"  Time step {t}: {len(batch['example_ids'])} examples")
        print(f"    Example IDs: {batch['example_ids']}")
        print(f"    Document mask: {batch['document_mask'].tolist()}")

    if i >= 1:  # Show only first 2 document batches
        break
```

### Cache Management Example

```python
from src.data import process_and_cache_dataset, ChunkedCacheManager

# Process and cache dataset
cache_dir = "./cache/examples"
total_features = process_and_cache_dataset(
    dataset_name="squad_v2",
    split="validation",
    cache_dir=cache_dir,
    max_examples=100,          # Small dataset for example
    max_seq_length=384,
    doc_stride=64,
    streaming_chunk_size=1000,
    max_memory_gb=2.0,
    use_streaming=False,
    tokenizer=tokenizer,       # Include memory tokens
    max_n_segs=3
)

print(f"✅ Cached {total_features} features to {cache_dir}")

# Examine cache
cache_manager = ChunkedCacheManager(cache_dir)
total_chunks = cache_manager.get_total_chunks("squad_v2_mem8", "validation")
print(f"Cache contains {total_chunks} chunks")

# Load first chunk
if total_chunks > 0:
    chunk_data = cache_manager.load_chunk("squad_v2_mem8", "validation", 0)
    print(f"First chunk: {len(chunk_data)} features")
    if chunk_data:
        sample_feature = chunk_data[0]
        print(f"Sample feature keys: {list(sample_feature.keys())}")
```

## 🔍 Evaluation Examples

### Complete Evaluation Pipeline

```python
from src.evaluate import main as evaluate_main
import tempfile
import json

# Create temporary config for evaluation
eval_config = {
    "model_name": "outputs/xlnet-squad-phase2-1/stage_2_segs_2/best_model",
    "dataset_name": "squad_v2",
    "eval_split": "validation",
    "max_seq_length": 384,
    "doc_stride": 64,
    "eval_batch_size": 8,
    "max_eval_samples": 100,  # Limit for quick evaluation
    "max_n_segs": 2,
    "use_time_step_major": True,
    "no_answer_threshold": 1.5,
    "cache_dir": "./.cache"
}

# Save config to temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(eval_config, f, indent=2)
    temp_config_path = f.name

print(f"Temporary config saved to: {temp_config_path}")

# Run evaluation
try:
    results = evaluate_main([temp_config_path])
    print("✅ Evaluation completed successfully")
except Exception as e:
    print(f"❌ Evaluation failed: {e}")
finally:
    import os
    os.unlink(temp_config_path)  # Clean up
```

### Custom Evaluation Function

```python
def custom_evaluate(model, tokenizer, max_examples=50):
    """Custom evaluation function with detailed output."""
    from src.data import create_evaluation_dataloader
    import torch

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create dataloader
    eval_dataset, eval_dataloader = create_evaluation_dataloader(
        dataset_name="squad_v2",
        split="validation",
        tokenizer=tokenizer,
        batch_size=4,
        max_examples=max_examples,
        use_time_step_major=True
    )

    # Track results
    memory_bank = {}
    total_predictions = 0

    with torch.no_grad():
        for doc_batch_idx, time_step_batches in enumerate(eval_dataloader):
            print(f"Processing document batch {doc_batch_idx + 1}")

            for time_step_idx, batch in enumerate(time_step_batches):
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                example_ids = batch['example_ids']

                # Get/initialize memory
                batch_memory = []
                for example_id in example_ids:
                    if example_id in memory_bank:
                        batch_memory.append(memory_bank[example_id])
                    else:
                        init_mem = model.get_initial_memory(1, device)
                        batch_memory.append(init_mem)

                memory_state = torch.cat(batch_memory, dim=0)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    memory_state=memory_state
                )

                # Update memory bank
                new_memory = outputs["new_memory_state"]
                for i, example_id in enumerate(example_ids):
                    if batch['document_mask'][i]:
                        memory_bank[example_id] = new_memory[i:i+1]

                # Get predictions
                start_logits = outputs["start_logits"]
                end_logits = outputs["end_logits"]

                # Find best spans (simplified)
                start_preds = torch.argmax(start_logits, dim=1)
                end_preds = torch.argmax(end_logits, dim=1)

                total_predictions += len(start_preds)

                print(f"  Time step {time_step_idx}: {len(example_ids)} examples")

            if doc_batch_idx >= 4:  # Limit for example
                break

    print(f"✅ Custom evaluation completed:")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Documents with memory: {len(memory_bank)}")

    return {"total_predictions": total_predictions, "memory_documents": len(memory_bank)}

# Run custom evaluation
if 'model' in locals() and 'tokenizer' in locals():
    results = custom_evaluate(model, tokenizer, max_examples=20)
    print(f"Results: {results}")
```

## 🐛 Troubleshooting Examples

### Common Issues and Solutions

#### 1. Import Errors

```python
# Test all critical imports
def test_imports():
    """Test all critical imports and report issues."""
    import_tests = [
        ("src.train", ["TrainingConfig", "XLNetRecurrentTrainer"]),
        ("src.data", ["SquadLikeQADataset", "TimeStepMajorDataLoader", "create_evaluation_dataloader"]),
        ("src.memxlnet_qa", ["MemXLNetForQA"]),
        ("src.evaluate", ["main"]),
    ]

    for module_name, classes in import_tests:
        try:
            module = __import__(module_name, fromlist=classes)
            for class_name in classes:
                getattr(module, class_name)
            print(f"✅ {module_name}: {', '.join(classes)}")
        except ImportError as e:
            print(f"❌ {module_name}: Import failed - {e}")
        except AttributeError as e:
            print(f"❌ {module_name}: Missing class - {e}")

test_imports()
```

#### 2. Memory Issues

```python
# Memory usage diagnostics
def diagnose_memory_usage():
    """Diagnose memory usage issues."""
    import torch
    import psutil
    import os

    print("=== Memory Diagnostics ===")

    # System memory
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.total / (1024**3):.1f}GB")
    print(f"Available: {memory.available / (1024**3):.1f}GB")
    print(f"Used: {memory.percent:.1f}%")

    # GPU memory (if available)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            total = props.total_memory / (1024**3)

            print(f"GPU {i} ({props.name}):")
            print(f"  Total: {total:.1f}GB")
            print(f"  Allocated: {allocated:.2f}GB")
            print(f"  Cached: {cached:.2f}GB")
    else:
        print("No CUDA devices available")

    # Recommendations
    print("\n=== Recommendations ===")
    if memory.percent > 80:
        print("⚠️  High system memory usage")
        print("  - Reduce batch_size")
        print("  - Reduce max_train_samples")
        print("  - Enable streaming processing")

    print("✅ Memory diagnostics complete")

diagnose_memory_usage()
```

#### 3. Model Loading Issues

```python
# Model loading diagnostics
def diagnose_model_loading(checkpoint_path):
    """Diagnose model loading issues."""
    import os
    import torch

    print(f"=== Model Loading Diagnostics ===")
    print(f"Checkpoint path: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print("❌ Checkpoint directory does not exist")
        return

    # Check for required files
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    optional_files = [
        "memxlnet_config.json",
        "memxlnet_state.pt"
    ]

    print("\n=== Required Files ===")
    for file in required_files:
        file_path = os.path.join(checkpoint_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024**2)
            print(f"✅ {file} ({size:.1f}MB)")
        else:
            print(f"❌ {file} - MISSING")

    print("\n=== Memory-Specific Files ===")
    for file in optional_files:
        file_path = os.path.join(checkpoint_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024**2)
            print(f"✅ {file} ({size:.1f}MB)")
        else:
            print(f"⚠️  {file} - Missing (OK for non-memory models)")

    # Try loading
    print("\n=== Loading Test ===")
    try:
        from src.memxlnet_qa import MemXLNetForQA
        model = MemXLNetForQA.from_pretrained(checkpoint_path)
        print(f"✅ Model loaded successfully")
        print(f"  Memory tokens: {model.mem_token_count}")
        print(f"  Memory update: {model.memory_update}")
        print(f"  Memory init: {model.memory_init}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("  Try using the base model instead:")
        print(f"  XLNetForQuestionAnsweringSimple.from_pretrained('{checkpoint_path}')")

# Test with actual checkpoint path
checkpoint_path = "outputs/xlnet-squad-phase2-1/stage_2_segs_2/best_model"
if os.path.exists(checkpoint_path):
    diagnose_model_loading(checkpoint_path)
else:
    print(f"Checkpoint not found: {checkpoint_path}")
    print("Available checkpoints:")
    outputs_dir = "outputs"
    if os.path.exists(outputs_dir):
        for item in os.listdir(outputs_dir):
            item_path = os.path.join(outputs_dir, item)
            if os.path.isdir(item_path):
                print(f"  {item}")
```

#### 4. Data Processing Issues

```python
# Data processing diagnostics
def diagnose_data_processing():
    """Diagnose data processing issues."""
    print("=== Data Processing Diagnostics ===")

    # Test tokenizer
    try:
        from transformers import XLNetTokenizerFast
        tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
        print(f"✅ Base tokenizer loaded: {len(tokenizer)} tokens")

        # Test memory token addition
        from src.data import configure_memory_tokens
        mem_config = configure_memory_tokens(tokenizer, 4)
        print(f"✅ Memory tokens added: {len(tokenizer)} tokens")
        print(f"  Read IDs: {mem_config['mem_read_ids']}")
        print(f"  Write IDs: {mem_config['mem_write_ids']}")

    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return

    # Test dataset creation
    try:
        from src.data import SquadLikeQADataset
        dataset = SquadLikeQADataset(
            split="validation",
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=64,
            max_examples=10,  # Small test
            dataset_name="squad_v2",
            max_n_segs=2
        )
        print(f"✅ Dataset created: {len(dataset)} features")
        print(f"  Documents: {len(dataset.get_all_documents())}")

    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return

    # Test time-step-major dataloader
    try:
        from src.data import TimeStepMajorDataLoader
        dataloader = TimeStepMajorDataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            max_segments=2
        )
        print(f"✅ TimeStepMajorDataLoader created: {len(dataloader)} batches")

        # Test iteration
        for time_step_batches in dataloader:
            print(f"  Time steps: {len(time_step_batches)}")
            break

    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")

    print("✅ Data processing diagnostics complete")

diagnose_data_processing()
```

#### 5. Configuration Issues

```python
# Configuration diagnostics
def diagnose_configuration():
    """Diagnose configuration issues."""
    print("=== Configuration Diagnostics ===")

    try:
        from src.train import TrainingConfig

        # Test basic configuration
        config = TrainingConfig()
        print("✅ Basic configuration created")
        print(f"  Model: {config.model_name}")
        print(f"  Memory tokens: {config.memory_num_tokens}")

        # Test field aliases
        config1 = TrainingConfig(model_name="test-model")
        config2 = TrainingConfig(model_name_or_path="test-model")
        assert config1.model_name == config2.model_name
        print("✅ Field aliases working")

        # Test memory configuration
        memory_config = TrainingConfig(
            memory_num_tokens=8,
            memory_update="gated",
            memory_init="learned"
        )
        print("✅ Memory configuration created")
        print(f"  Tokens: {memory_config.memory_num_tokens}")
        print(f"  Update: {memory_config.memory_update}")
        print(f"  Init: {memory_config.memory_init}")

        # Test progressive segments
        progressive_config = TrainingConfig(
            progressive_segments=[1, 2, 4],
            max_n_segs=4
        )
        print("✅ Progressive configuration created")
        print(f"  Segments: {progressive_config.progressive_segments}")
        print(f"  Max segments: {progressive_config.max_n_segs}")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

    print("✅ Configuration diagnostics complete")

diagnose_configuration()
```

## 🔧 Performance Optimization Examples

### Memory Optimization

```python
# Memory-optimized configuration
memory_optimized_config = TrainingConfig(
    # Reduce memory usage
    train_batch_size=8,           # Smaller batches
    eval_batch_size=4,
    max_seq_length=256,           # Shorter sequences
    gradient_accumulation_steps=4, # Accumulate gradients

    # Enable optimizations
    fp16=True,                    # Mixed precision
    use_streaming=True,           # Streaming processing
    streaming_chunk_size=500,     # Smaller chunks

    # Limit data
    max_train_samples=5000,
    max_eval_samples=500,

    # Memory system
    memory_num_tokens=4,          # Fewer memory tokens
    max_n_segs=2,                # Fewer segments

    print("Memory-optimized configuration:")
    print(f"  Batch size: {memory_optimized_config.train_batch_size}")
    print(f"  Sequence length: {memory_optimized_config.max_seq_length}")
    print(f"  FP16: {memory_optimized_config.fp16}")
    print(f"  Memory tokens: {memory_optimized_config.memory_num_tokens}")
)
```

### Speed Optimization

```python
# Speed-optimized configuration
speed_optimized_config = TrainingConfig(
    # Larger batches for speed
    train_batch_size=32,
    eval_batch_size=16,

    # Fewer evaluation steps
    eval_steps=2000,
    logging_steps=500,
    save_steps=5000,

    # Disable expensive features during development
    use_global_softmax=False,
    use_any_positive_logic=False,

    # Shorter training
    num_epochs=1,
    warmup_ratio=0.05,           # Shorter warmup

    # Limited segments
    max_n_segs=1,                # Single segment for speed
    progressive_segments=[1],     # No progression

    print("Speed-optimized configuration:")
    print(f"  Batch size: {speed_optimized_config.train_batch_size}")
    print(f"  Eval steps: {speed_optimized_config.eval_steps}")
    print(f"  Global softmax: {speed_optimized_config.use_global_softmax}")
    print(f"  Progressive segments: {speed_optimized_config.progressive_segments}")
)
```

This comprehensive guide provides practical examples for all major use cases of MemXLNet-QA, from basic evaluation to advanced training scenarios, along with detailed troubleshooting steps for common issues.