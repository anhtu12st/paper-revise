MemXLNet-QA: Long-Context Question Answering with XLNet

Abstract

Long-context question answering (QA) remains challenging for encoder-only language models due to quadratic attention cost and limited positional capacity. We present MemXLNet-QA, a simple and effective extension to XLNet that adds explicit read/write memory tokens and a lightweight gated update mechanism. Our approach operates in a document-as-stream regime: documents are split into chunks and processed sequentially with a persistent per-document memory state. We pair this with time-step-major batching, robust no-answer calibration for SQuAD v2, and progressive training that increases the number of segments per document. On SQuAD v2, MemXLNet-QA maintains strong answer quality while scaling to longer inputs. Results are intentionally omitted in this draft and will be filled in later.

1. Introduction

Modern QA systems often face inputs far exceeding typical transformer context windows. While XLNet is built on Transformer-XL and exposes segment-level recurrence, canonical fine-tuning pipelines commonly ignore memory, treating each chunk independently. This leads to suboptimal long-context performance and unstable no-answer calibration for datasets like SQuAD v2.

We revisit XLNet’s recurrence and introduce explicit memory tokens, enabling the model to read from and write to a compact external state across chunks within the same document. Our contributions:

A minimal wrapper around XLNet (MemXLNetForQA) that injects memory read tokens and updates memory via write tokens using a gated mechanism.

A time-step-major training loop that preserves per-document memory and computes loss coherently over entire documents.

Robust SQuAD v2 no-answer handling, including context-only span selection and a conservative null-score anchor resilient to special/memory tokens.

Progressive training over segments-per-document and optional document-level “global softmax” that aggregates span logits across chunks.

2. Related Work

Transformer-XL and XLNet: segment-level recurrence and permutation language modeling for long-range dependencies.

Long-context transformers: local/global attention (e.g., Longformer, BigBird), memory augmentation, and retrieval-augmented methods.

QA calibration and SQuAD v2: no-answer scoring and thresholding; document-level span aggregation strategies.

3. Method

3.1 Architecture

MemXLNet-QA wraps XLNetForQuestionAnsweringSimple with an explicit memory interface (src/memxlnet_qa.py). The tokenizer is extended with additional special tokens: a small set of read tokens [MEM_R] and write tokens [MEM_W]. On each chunk:

Read: we prepend one or more [MEM_R] tokens whose embeddings are the current memory state; the model consumes these alongside the regular sequence.

Write: we reserve [MEM_W] positions whose hidden states summarize the chunk; a gated updater (e.g., sigmoid gate over a linear projection) merges them into the new memory state.

The memory state is persisted per document in the trainer and is reset between documents.

3.2 Data processing

We split documents into overlapping chunks (length max_seq_length, stride doc_stride) and create time-step-major batches. The collator in src/data.py optionally injects memory tokens and carefully realigns:

token_type_ids to mark context tokens (context-only span selection),

offset_mapping for exact text extraction,

start/end labels when trimming or inserting tokens.

3.3 Training objective

For each document batch, we compute cross-entropy over start and end indices. Two variants are supported:

Per-segment loss: average CE across chunks.

Document-level global softmax: concatenate chunk logits and compute CE over the flattened sequence.

We use AdamW with linear warmup and gradient clipping. A short warmup phase can freeze the base transformer and disable global-softmax/any-positive logic to stabilize calibration.

3.4 Prediction extraction and calibration (SQuAD v2)

At evaluation, we aggregate logits per document and extract spans with context-only masking (tokens with token_type_ids == 1 and valid offsets). No-answer (null) scoring uses a robust anchor that considers:

the true CLS positions (from input_ids),

the last index, and

the last non-context/invalid-offset token,

selecting the maximum start+end among these candidates. We compare the best context span score against the null score with a tunable no_answer_threshold. We also support:

Any-positive logic: predict an answer if any chunk is sufficiently confident;

Global softmax: select the best span across concatenated document logits.

4. Datasets and Preprocessing

Dataset: SQuAD v2 train/validation splits.

Chunking: max_seq_length tokens with doc_stride overlap.

Caching and streaming: chunked cache for large-scale processing (src/data.py).

Tokenization: XLNet tokenizer with additional memory special tokens; embeddings are resized accordingly.

5. Training Setup

Base model: XLNetForQuestionAnsweringSimple.

Wrapper: MemXLNetForQA with gated memory updates.

Optimizer/scheduler: AdamW + linear warmup; max grad norm.

Mixed precision: optional FP16.

Progressive segments: train over increasing segments per document (e.g., [1 → 3 → 6]).

Warmup behavior: early-epoch base freeze; disable global-softmax/any-positive initially.

Key implementation details live in src/train.py (trainer), src/data.py (dataset/collator), and src/memxlnet_qa.py (wrapper). A diagnostic tool for calibration and threshold sweeps is provided in scripts/debug_eval.py.

6. Evaluation Protocol

Metrics: SQuAD v2 Exact Match (EM) and F1, with splits for HasAnswer and NoAnswer.

Extraction: context-only spans; robust null score; threshold sweep to select no_answer_threshold.

Ablations: any-positive on/off, global softmax on/off, memory token count, stride, segments-per-document, and warmup settings.
