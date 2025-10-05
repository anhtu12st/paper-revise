"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root):
    """Return test data directory."""
    return project_root / "tests" / "fixtures"


@pytest.fixture
def cache_dir(tmp_path):
    """Return temporary cache directory for tests."""
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


@pytest.fixture
def example_config():
    """Return example training configuration for testing."""
    from memxlnet.training import TrainingConfig

    return TrainingConfig(
        model_name="xlnet-base-cased",
        memory_num_tokens=4,
        max_train_samples=10,
        max_eval_samples=5,
        num_epochs=1,
        train_batch_size=2,
        eval_batch_size=2,
    )
