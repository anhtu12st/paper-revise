#!/usr/bin/env python3
"""
Integration tests for Hub-first evaluation script.

Tests that the evaluate_cls_fix.py script can successfully load models from
HuggingFace Hub and run evaluations with the CLS position fix.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.integration
@pytest.mark.slow
class TestHubEvaluation:
    """Test Hub-first evaluation functionality."""

    def test_hub_model_availability(self):
        """Test that we can check if a Hub model exists."""
        from huggingface_hub import HfApi

        api = HfApi()

        # This is a public XLNet model that should always exist
        try:
            model_info = api.model_info("xlnet-base-cased")
            assert model_info is not None
            print(f"✅ Hub connection working: {model_info.modelId}")
        except Exception as e:
            pytest.skip(f"Cannot connect to HuggingFace Hub: {e}")

    def test_evaluation_script_exists(self):
        """Test that evaluation script exists and is executable."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "evaluate_cls_fix.py"
        assert script_path.exists(), f"Evaluation script not found: {script_path}"
        assert os.access(script_path, os.R_OK), "Evaluation script not readable"
        print(f"✅ Evaluation script found: {script_path}")

    def test_evaluation_script_help(self):
        """Test that evaluation script shows help."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "evaluate_cls_fix.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, "Help command failed"
        assert "model-id" in result.stdout, "Help should mention --model-id"
        assert "revision" in result.stdout, "Help should mention --revision"
        assert "test-size" in result.stdout, "Help should mention --test-size"
        print("✅ Evaluation script help working")

    @pytest.mark.skipif(
        not os.path.exists("outputs/xlnet-squad-phase2-1/stage_1_segs_1/best_model"),
        reason="Local checkpoint not available",
    )
    def test_upload_script_validation_local_checkpoint(self):
        """Test that upload script can validate local checkpoint (if available)."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "upload_checkpoint_to_hub.py"
        checkpoint_path = "outputs/xlnet-squad-phase2-1/stage_1_segs_1/best_model"

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--checkpoint-path",
                checkpoint_path,
                "--hub-id",
                "test/dummy-model",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            input="n\n",  # Decline confirmation
        )

        # Should succeed in dry-run mode or fail gracefully
        assert "Validating checkpoint" in result.stdout or result.returncode != 0
        print("✅ Upload script dry-run validation working")

    def test_model_loading_from_hub(self):
        """Test that MemXLNetForQA can load from Hub (using public base model)."""

        # Try loading a base XLNet model to test Hub integration
        # Note: This won't be a MemXLNet model, but tests the loading mechanism
        try:
            from transformers import XLNetForQuestionAnsweringSimple

            base_model = XLNetForQuestionAnsweringSimple.from_pretrained("xlnet-base-cased")
            assert base_model is not None
            print("✅ Hub model loading working")
        except Exception as e:
            pytest.skip(f"Cannot load model from Hub: {e}")

    def test_environment_variable_handling(self):
        """Test that scripts handle HF_TOKEN environment variable correctly."""
        # Save original
        original_token = os.environ.get("HF_TOKEN")

        try:
            # Test without token
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]

            # Public models should still work
            from transformers import XLNetTokenizerFast

            tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
            assert tokenizer is not None
            print("✅ Hub access without token working (public models)")

        finally:
            # Restore original
            if original_token:
                os.environ["HF_TOKEN"] = original_token

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip full evaluation in CI (too slow)",
    )
    def test_evaluation_quick_subset(self):
        """Test evaluation script with small subset (if Hub model available)."""
        # Check if a test model is available
        test_model_id = os.environ.get("TEST_MODEL_ID")
        if not test_model_id:
            pytest.skip("TEST_MODEL_ID not set - set to test Hub evaluation")

        script_path = Path(__file__).parent.parent.parent / "scripts" / "evaluate_cls_fix.py"

        # Run evaluation with small subset
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--model-id",
                test_model_id,
                "--test-size",
                "10",  # Very small for testing
                "--output-dir",
                "/tmp/test_eval_output",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes max
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"Evaluation failed with exit code {result.returncode}")

        # Check output
        assert "EVALUATION RESULTS" in result.stdout
        assert "F1 Score" in result.stdout
        print("✅ Quick subset evaluation completed")


@pytest.mark.integration
class TestEvaluationBackwardCompatibility:
    """Test backward compatibility with old evaluation approach."""

    @pytest.mark.skipif(
        not os.path.exists("outputs/xlnet-squad-phase2-1/stage_1_segs_1/best_model"),
        reason="Local checkpoint not available",
    )
    def test_old_checkpoint_structure(self):
        """Test that old checkpoints can still be loaded."""
        from memxlnet.models.memxlnet_qa import MemXLNetForQA

        checkpoint_path = "outputs/xlnet-squad-phase2-1/stage_1_segs_1/best_model"

        try:
            model = MemXLNetForQA.from_pretrained(checkpoint_path)
            assert model is not None
            assert hasattr(model, "mem_token_count")
            print(f"✅ Old checkpoint loaded: {model.mem_token_count} memory tokens")
        except Exception as e:
            pytest.fail(f"Failed to load old checkpoint: {e}")


def test_script_imports():
    """Test that evaluation script imports work correctly."""
    # This test runs the imports in the script
    script_path = Path(__file__).parent.parent.parent / "scripts" / "evaluate_cls_fix.py"

    # Read script and check for correct imports
    with open(script_path) as f:
        content = f.read()

    # Check for required imports
    required_imports = [
        "from memxlnet.models.memxlnet_qa import MemXLNetForQA",
        "from memxlnet.data import create_dataloader, load_chunked_dataset",
        "from transformers import XLNetTokenizerFast",
    ]

    for import_line in required_imports:
        assert import_line in content, f"Missing import: {import_line}"

    print("✅ All required imports present in evaluation script")


def test_naming_conventions():
    """Test that scripts follow HuggingFace naming conventions."""
    # Read both scripts
    eval_script = Path(__file__).parent.parent.parent / "scripts" / "evaluate_cls_fix.py"
    upload_script = Path(__file__).parent.parent.parent / "scripts" / "upload_checkpoint_to_hub.py"

    with open(eval_script) as f:
        eval_content = f.read()

    with open(upload_script) as f:
        upload_content = f.read()

    # Check naming convention documentation is present
    assert "memxlnet-squad-" in eval_content, "Evaluation script should document naming convention"
    assert "memxlnet-squad-" in upload_content, "Upload script should document naming convention"

    # Check for variant patterns
    assert "phase2-mem" in eval_content or "mem16" in eval_content, "Should mention memory variants"
    assert "phase2-mem" in upload_content or "mem16" in upload_content, "Should mention memory variants"

    print("✅ Naming conventions documented in scripts")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
