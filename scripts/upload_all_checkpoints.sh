#!/bin/bash
###############################################################################
# Upload All Checkpoints to HuggingFace Hub
###############################################################################
#
# This script uploads all checkpoints from outputs/ to HuggingFace Hub
# following the standardized naming convention.
#
# USAGE:
#   1. Set your HF_TOKEN:
#      export HF_TOKEN='your_token_here'
#
#   2. Edit HUB_USERNAME below (line 21)
#
#   3. Run the script:
#      bash scripts/upload_all_checkpoints.sh
#
# NAMING CONVENTION:
#   {username}/memxlnet-squad-phase2-mem{N}
#   Revisions: stage-{stage}-segs-{segments}
#
###############################################################################

# Configuration
HUB_USERNAME="anhtu12st"  # CHANGE THIS to your username!
MEMORY_TOKENS=16          # All your models use 16 memory tokens

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN not set!"
    echo "   Set with: export HF_TOKEN='your_token'"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  Upload All Checkpoints to HuggingFace Hub                             ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Hub Username: $HUB_USERNAME"
echo "Memory Tokens: $MEMORY_TOKENS"
echo ""

# Function to upload a checkpoint
upload_checkpoint() {
    local checkpoint_path="$1"
    local hub_id="$2"
    local revision="$3"
    local description="$4"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 $description"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Checkpoint: $checkpoint_path"
    echo "   Hub ID: $hub_id"
    echo "   Revision: $revision"
    echo ""

    if [ ! -d "$checkpoint_path" ]; then
        echo "   ⚠️  Checkpoint not found, skipping..."
        echo ""
        return
    fi

    uv run python scripts/upload_checkpoint_to_hub.py \
        --checkpoint-path "$checkpoint_path" \
        --hub-id "$hub_id" \
        --revision "$revision" \
        --private \
        --yes \
        --commit-message "Upload $description"

    if [ $? -eq 0 ]; then
        echo "   ✅ Upload successful!"
    else
        echo "   ❌ Upload failed!"
    fi
    echo ""
}

###############################################################################
# STANDARD SQUAD V2 CHECKPOINTS (phase2-mem16)
###############################################################################

BASE_HUB_ID="${HUB_USERNAME}/memxlnet-squad-phase2-mem16"

echo "════════════════════════════════════════════════════════════════════════"
echo "  Standard SQuAD v2 Checkpoints (phase2-mem16)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Stage 1, 1 segment
upload_checkpoint \
    "outputs/xlnet-squad-phase2-1/stage_1_segs_1/best_model" \
    "$BASE_HUB_ID" \
    "stage-1-segs-1" \
    "Phase 2, Stage 1, 1 segment"

# Stage 1, 2 segments
upload_checkpoint \
    "outputs/xlnet-squad-phase2-1/stage_1_segs_2/best_model" \
    "$BASE_HUB_ID" \
    "stage-1-segs-2" \
    "Phase 2, Stage 1, 2 segments"

# Stage 2, 2 segments (best model)
upload_checkpoint \
    "outputs/xlnet-squad-phase2-1/stage_2_segs_2/best_model" \
    "$BASE_HUB_ID" \
    "stage-2-segs-2-best" \
    "Phase 2, Stage 2, 2 segments (best)"

# Stage 2, 2 segments (final model)
upload_checkpoint \
    "outputs/xlnet-squad-phase2-1/stage_2_segs_2/final_model" \
    "$BASE_HUB_ID" \
    "stage-2-segs-2-final" \
    "Phase 2, Stage 2, 2 segments (final)"

# Stage 2, 2 segments (checkpoint-10000)
upload_checkpoint \
    "outputs/xlnet-squad-phase2-1/stage_2_segs_2/checkpoint-10000" \
    "$BASE_HUB_ID" \
    "stage-2-segs-2-ckpt-10k" \
    "Phase 2, Stage 2, 2 segments (checkpoint 10k)"

# Upload best overall as main/default revision
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Setting default revision (best model)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Upload best model to main branch (no revision = main/default)
uv run python scripts/upload_checkpoint_to_hub.py \
    --checkpoint-path "outputs/xlnet-squad-phase2-1/stage_2_segs_2/best_model" \
    --hub-id "$BASE_HUB_ID" \
    --private \
    --yes \
    --commit-message "Upload best model as default"

echo ""

###############################################################################
# LONG-CONTEXT SQUAD CHECKPOINTS (phase2-long-mem16)
###############################################################################

LONG_HUB_ID="${HUB_USERNAME}/memxlnet-squad-phase2-long-mem16"

echo "════════════════════════════════════════════════════════════════════════"
echo "  Long-Context SQuAD Checkpoints (phase2-long-mem16)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Stage 1, 3 segments
upload_checkpoint \
    "outputs/xlnet-long-squad-phase2-1/stage_1_segs_3/best_model" \
    "$LONG_HUB_ID" \
    "stage-1-segs-3" \
    "Phase 2 Long, Stage 1, 3 segments"

# Stage 2, 6 segments
upload_checkpoint \
    "outputs/xlnet-long-squad-phase2-1/stage_2_segs_6/best_model" \
    "$LONG_HUB_ID" \
    "stage-2-segs-6" \
    "Phase 2 Long, Stage 2, 6 segments"

# Stage 1, 12 segments (if exists)
upload_checkpoint \
    "outputs/xlnet-long-squad-phase2-1/stage_1_segs_12/best_model" \
    "$LONG_HUB_ID" \
    "stage-1-segs-12" \
    "Phase 2 Long, Stage 1, 12 segments"

# Upload best long model as default
uv run python scripts/upload_checkpoint_to_hub.py \
    --checkpoint-path "outputs/xlnet-long-squad-phase2-1/stage_2_segs_6/best_model" \
    --hub-id "$LONG_HUB_ID" \
    --private \
    --yes \
    --commit-message "Upload best long model as default"

echo ""

###############################################################################
# FULL-DATASET LONG-CONTEXT CHECKPOINTS (phase2-long-full-mem16)
###############################################################################

FULL_LONG_HUB_ID="${HUB_USERNAME}/memxlnet-squad-phase2-long-full-mem16"

echo "════════════════════════════════════════════════════════════════════════"
echo "  Full-Dataset Long-Context Checkpoints (phase2-long-full-mem16)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Stage 1, 24 segments
upload_checkpoint \
    "outputs/xlnet-long-squad-full-12-24/stage_1_segs_24/final_model" \
    "$FULL_LONG_HUB_ID" \
    "stage-1-segs-24-final" \
    "Phase 2 Long Full, Stage 1, 24 segments (final)"

# Upload as default
uv run python scripts/upload_checkpoint_to_hub.py \
    --checkpoint-path "outputs/xlnet-long-squad-full-12-24/stage_1_segs_24/final_model" \
    --hub-id "$FULL_LONG_HUB_ID" \
    --private \
    --yes \
    --commit-message "Upload 24-segment model as default"

echo ""

###############################################################################
# SUMMARY
###############################################################################

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  Upload Complete!                                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Repositories Created:"
echo ""
echo "   1. ${HUB_USERNAME}/memxlnet-squad-phase2-mem16"
echo "      Standard SQuAD v2 training with progressive stages"
echo "      Revisions: stage-1-segs-1, stage-1-segs-2, stage-2-segs-2-best, etc."
echo "      https://huggingface.co/${HUB_USERNAME}/memxlnet-squad-phase2-mem16"
echo ""
echo "   2. ${HUB_USERNAME}/memxlnet-squad-phase2-long-mem16"
echo "      Long-context SQuAD v2 training (up to 12 segments)"
echo "      Revisions: stage-1-segs-3, stage-2-segs-6, stage-1-segs-12"
echo "      https://huggingface.co/${HUB_USERNAME}/memxlnet-squad-phase2-long-mem16"
echo ""
echo "   3. ${HUB_USERNAME}/memxlnet-squad-phase2-long-full-mem16"
echo "      Full-dataset long-context training (24 segments)"
echo "      Revisions: stage-1-segs-24-final"
echo "      https://huggingface.co/${HUB_USERNAME}/memxlnet-squad-phase2-long-full-mem16"
echo ""
echo "💡 Usage Examples:"
echo ""
echo "   # Evaluate standard model"
echo "   uv run python scripts/evaluate_cls_fix.py --model-id ${HUB_USERNAME}/memxlnet-squad-phase2-mem16"
echo ""
echo "   # Evaluate specific stage"
echo "   uv run python scripts/evaluate_cls_fix.py --model-id ${HUB_USERNAME}/memxlnet-squad-phase2-mem16 --revision stage-1-segs-1"
echo ""
echo "   # Evaluate long-context model"
echo "   uv run python scripts/evaluate_cls_fix.py --model-id ${HUB_USERNAME}/memxlnet-squad-phase2-long-mem16 --revision stage-2-segs-6"
echo ""
echo "   # Load in Python"
echo "   from memxlnet.models import MemXLNetForQA"
echo "   model = MemXLNetForQA.from_pretrained('${HUB_USERNAME}/memxlnet-squad-phase2-mem16')"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
