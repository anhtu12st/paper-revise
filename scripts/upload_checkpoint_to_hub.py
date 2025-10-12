#!/usr/bin/env python3
"""
Upload MemXLNet Checkpoint to HuggingFace Hub
==============================================

This script uploads a local MemXLNet checkpoint to HuggingFace Hub for
cross-server accessibility and reproducibility.

HUGGINGFACE NAMING CONVENTION:
------------------------------
Model repositories follow the pattern:
  {username}/memxlnet-squad-{variant}

Common variants:
  - memxlnet-squad-phase2-mem16  (Phase 2 trained, 16 memory tokens)
  - memxlnet-squad-phase2-mem8   (Phase 2 trained, 8 memory tokens)
  - memxlnet-squad-baseline      (No memory tokens)
  - memxlnet-squad-cls-fixed     (With CLS position bug fix)

Revisions/Tags (optional):
  - best-model     (default, latest best checkpoint)
  - stage-1-segs-1 (training stage checkpoints)
  - stage-1-segs-2
  - v1.0, v2.0     (version tags)

SETUP:
------
1. Set HF_TOKEN environment variable:
   export HF_TOKEN='your_huggingface_token'

2. Run upload:
   python scripts/upload_checkpoint_to_hub.py \\
       --checkpoint-path outputs/xlnet-squad-phase2-1/stage_1_segs_1/best_model \\
       --hub-id anhtu12st/memxlnet-squad-phase2-mem16 \\
       --revision stage-1-segs-1

USAGE:
------
# Basic upload (creates or updates repository)
python scripts/upload_checkpoint_to_hub.py \\
    --checkpoint-path outputs/my-model/best_model \\
    --hub-id username/memxlnet-squad-phase2-mem16

# Upload with revision/tag
python scripts/upload_checkpoint_to_hub.py \\
    --checkpoint-path outputs/my-model/stage_1/best_model \\
    --hub-id username/memxlnet-squad-phase2-mem16 \\
    --revision stage-1-segs-1

# Create private repository
python scripts/upload_checkpoint_to_hub.py \\
    --checkpoint-path outputs/my-model/best_model \\
    --hub-id username/memxlnet-squad-phase2-mem16 \\
    --private

# Dry run (check files without uploading)
python scripts/upload_checkpoint_to_hub.py \\
    --checkpoint-path outputs/my-model/best_model \\
    --hub-id username/memxlnet-squad-phase2-mem16 \\
    --dry-run

# Non-interactive mode (skip confirmation prompt)
python scripts/upload_checkpoint_to_hub.py \\
    --checkpoint-path outputs/my-model/best_model \\
    --hub-id username/memxlnet-squad-phase2-mem16 \\
    --private \\
    --yes
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memxlnet.models.memxlnet_qa import MemXLNetForQA

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# ============================================================================

# Your HuggingFace username (used for examples)
HUB_USERNAME = "anhtu12st"  # Change this to your username!


# ============================================================================
# UPLOAD LOGIC
# ============================================================================


def validate_checkpoint(checkpoint_path: str) -> dict:
    """Validate checkpoint exists and contains required files."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint path must be a directory: {checkpoint_path}")

    # Check for required files
    required_files = {
        "memxlnet_config.json": "MemXLNet configuration",
        "memxlnet_state.pt": "Model weights",
        "config.json": "Base XLNet configuration",
    }

    optional_files = {
        "training_config.json": "Training configuration",
        "tokenizer_config.json": "Tokenizer configuration",
        "tokenizer.json": "Tokenizer data",
        "special_tokens_map.json": "Special tokens mapping",
        "spiece.model": "SentencePiece model",
    }

    found_files = {}
    missing_required = []

    # Check required files
    for filename, description in required_files.items():
        filepath = checkpoint_path / filename
        if filepath.exists():
            found_files[filename] = description
        else:
            missing_required.append(filename)

    # Check optional files
    for filename, description in optional_files.items():
        filepath = checkpoint_path / filename
        if filepath.exists():
            found_files[filename] = description

    if missing_required:
        raise ValueError(f"Missing required files: {', '.join(missing_required)}")

    return found_files


def load_and_validate_model(checkpoint_path: str):
    """Load model to verify it's valid."""
    try:
        model = MemXLNetForQA.from_pretrained(checkpoint_path)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model from checkpoint: {e}")


def upload_checkpoint(
    checkpoint_path: str,
    hub_id: str,
    revision: str = None,
    private: bool = False,
    commit_message: str = None,
    dry_run: bool = False,
):
    """Upload checkpoint to HuggingFace Hub."""
    from huggingface_hub import HfApi, create_repo

    # Validate checkpoint
    print("=" * 80)
    print("🔍 Validating checkpoint...")
    print("=" * 80)
    print()

    files = validate_checkpoint(checkpoint_path)
    print(f"✅ Checkpoint valid: {checkpoint_path}")
    print(f"   Found {len(files)} files:")
    for filename, description in files.items():
        print(f"   • {filename:30s} - {description}")
    print()

    # Load and validate model
    print("🔧 Loading model to verify...")
    try:
        model = load_and_validate_model(checkpoint_path)
        print("✅ Model loaded successfully:")
        print(f"   • Memory tokens: {model.mem_token_count}")
        print(f"   • Memory update: {model.memory_update}")
        print(f"   • Memory init: {model.memory_init}")
        print(f"   • Use differentiable memory: {model.use_differentiable_memory}")
        print()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    if dry_run:
        print("✅ DRY RUN: Checkpoint validation passed")
        print("   No files uploaded (use without --dry-run to upload)")
        return 0

    # Get HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN not found in environment!")
        print("   Set with: export HF_TOKEN='your_token'")
        return 1

    # Create API client
    api = HfApi(token=hf_token)

    # Create or get repository
    print("=" * 80)
    print(f"📦 Preparing repository: {hub_id}")
    print("=" * 80)
    print()

    try:
        repo_url = create_repo(
            repo_id=hub_id,
            token=hf_token,
            private=private,
            exist_ok=True,  # Don't fail if repo exists
            repo_type="model",
        )
        privacy_status = "🔒 PRIVATE" if private else "🌐 PUBLIC"
        print(f"✅ Repository ready: {repo_url}")
        print(f"   Privacy: {privacy_status}")
        print()
    except Exception as e:
        print(f"❌ Failed to create/access repository: {e}")
        return 1

    # Upload files
    print("=" * 80)
    print("📤 Uploading checkpoint files...")
    print("=" * 80)
    print()

    if commit_message is None:
        revision_info = f" (revision: {revision})" if revision else ""
        commit_message = f"Upload checkpoint from {checkpoint_path}{revision_info}"

    try:
        # Strategy: Always upload to main first, then create branch if revision specified
        if revision:
            print(f"🔀 Strategy: Upload to main, then create branch '{revision}'")
            print()

        # Step 1: Upload to main branch
        print("📤 Step 1: Uploading to main branch...")
        upload_result = api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=hub_id,
            repo_type="model",
            revision=None,  # Always upload to main first
            commit_message=commit_message,
            token=hf_token,
        )
        print("✅ Upload to main completed!")
        print(f"   Commit: {upload_result}")
        print()

        # Step 2: Create branch if revision specified
        if revision:
            print(f"🏷️  Step 2: Creating branch '{revision}'...")
            try:
                # Create a branch from the latest main commit
                api.create_branch(
                    repo_id=hub_id,
                    repo_type="model",
                    branch=revision,
                    token=hf_token,
                    exist_ok=True,  # Don't fail if branch already exists
                )
                print(f"✅ Branch '{revision}' created!")
                print()
            except Exception as branch_error:
                print(f"⚠️  Branch creation note: {branch_error}")
                print("   Files are on main branch. You can create the branch manually via Hub UI.")
                print()

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return 1

    # Success summary
    print("=" * 80)
    print("🎉 Checkpoint uploaded to HuggingFace Hub!")
    print("=" * 80)
    print()
    print("📍 Location:")
    print(f"   • Repository: {hub_id}")
    if revision:
        print(f"   • Revision: {revision}")
    print(f"   • URL: https://huggingface.co/{hub_id}")
    print()
    print("💡 Usage:")
    print("   from memxlnet.models import MemXLNetForQA")
    print(f"   model = MemXLNetForQA.from_pretrained('{hub_id}'{', revision=' + repr(revision) if revision else ''})")
    print()
    print("🔍 Evaluation:")
    print(f"   python scripts/evaluate_cls_fix.py --model-id {hub_id}{' --revision ' + revision if revision else ''}")
    print()
    print("=" * 80)

    return 0


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main upload entry point."""
    parser = argparse.ArgumentParser(
        description="Upload MemXLNet checkpoint to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to local checkpoint directory",
    )
    parser.add_argument(
        "--hub-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/memxlnet-squad-phase2-mem16')",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision/tag name (e.g., 'stage-1-segs-1', 'v1.0'). Default: main branch",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository (default: public)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message (default: auto-generated)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint without uploading",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt (non-interactive mode)",
    )

    args = parser.parse_args()

    # Print header
    print()
    print("=" * 80)
    print("🚀 MemXLNet Checkpoint Upload to HuggingFace Hub")
    print("=" * 80)
    print()

    # Validate inputs
    if not args.hub_id:
        print("❌ Error: --hub-id is required!")
        print("   Example: --hub-id anhtu12st/memxlnet-squad-phase2-mem16")
        return 1

    # Show configuration
    print("📋 Configuration:")
    print(f"   • Checkpoint: {args.checkpoint_path}")
    print(f"   • Hub ID: {args.hub_id}")
    if args.revision:
        print(f"   • Revision: {args.revision}")
    privacy_status = "🔒 PRIVATE" if args.private else "🌐 PUBLIC"
    print(f"   • Privacy: {privacy_status}")
    if args.dry_run:
        print("   • Mode: DRY RUN (validation only)")
    print()

    # Confirm
    if not args.dry_run and not args.yes:
        response = input("Continue with upload? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            print("Aborted.")
            return 1
        print()
    elif args.yes:
        print("✅ Auto-confirming (--yes flag set)")
        print()

    # Upload
    return upload_checkpoint(
        checkpoint_path=args.checkpoint_path,
        hub_id=args.hub_id,
        revision=args.revision,
        private=args.private,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
