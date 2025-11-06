"""
Unit tests for RBS hybrid training configuration.
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from rbsqa.configs.hybrid_training_config import (
    RBSTrainingConfig,
    create_balanced_config,
    create_quick_debug_config
)


class TestRBSTrainingConfig:
    """Test cases for RBSTrainingConfig class."""

    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = RBSTrainingConfig()

        assert config.num_epochs == 10
        assert config.batch_size == 8
        assert config.learning_rate == 5e-5
        assert config.use_rl_training is True
        assert config.rl_start_epoch == 2
        assert config.memory_num_tokens == 16
        assert config.num_memory_experts == 4
        assert config.use_rbs_mode is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = RBSTrainingConfig()
        config.validate()  # Should not raise

        # Invalid RL start epoch
        config = RBSTrainingConfig(rl_start_epoch=15, num_epochs=10, use_rl_training=True)
        with pytest.raises(ValueError, match="rl_start_epoch.*must be less than"):
            config.validate()

        # Invalid number of experts
        config = RBSTrainingConfig(num_memory_experts=0)
        with pytest.raises(ValueError, match="num_memory_experts must be between"):
            config.validate()

        config = RBSTrainingConfig(num_memory_experts=10)
        with pytest.raises(ValueError, match="num_memory_experts must be between"):
            config.validate()

        # Invalid memory tokens
        config = RBSTrainingConfig(memory_num_tokens=0)
        with pytest.raises(ValueError, match="memory_num_tokens must be positive"):
            config.validate()

        # Invalid batch size
        config = RBSTrainingConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

        # Invalid learning rate
        config = RBSTrainingConfig(learning_rate=0)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            config.validate()

        # Invalid belief state threshold
        config = RBSTrainingConfig(belief_state_threshold=1.5)
        with pytest.raises(ValueError, match="belief_state_threshold must be between"):
            config.validate()

        config = RBSTrainingConfig(belief_state_threshold=-0.1)
        with pytest.raises(ValueError, match="belief_state_threshold must be between"):
            config.validate()

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = RBSTrainingConfig(
            num_epochs=5,
            batch_size=16,
            learning_rate=1e-4
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['num_epochs'] == 5
        assert config_dict['batch_size'] == 16
        assert config_dict['learning_rate'] == 1e-4

        # Check that all fields are present
        expected_fields = set(RBSTrainingConfig.__dataclass_fields__.keys())
        actual_fields = set(config_dict.keys())
        assert expected_fields == actual_fields

    def test_config_save_load(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")

            # Create and save config
            original_config = RBSTrainingConfig(
                num_epochs=15,
                batch_size=4,
                learning_rate=2e-5,
                use_rl_training=False
            )
            original_config.save(config_path)

            # Load config
            loaded_config = RBSTrainingConfig.load(config_path)

            # Verify loaded config matches original
            assert loaded_config.num_epochs == original_config.num_epochs
            assert loaded_config.batch_size == original_config.batch_size
            assert loaded_config.learning_rate == original_config.learning_rate
            assert loaded_config.use_rl_training == original_config.use_rl_training

            # Verify the file was created and contains valid JSON
            assert os.path.exists(config_path)
            with open(config_path, 'r') as f:
                saved_data = json.load(f)
            assert saved_data['num_epochs'] == 15

    def test_from_args(self):
        """Test creating config from command line arguments."""
        from argparse import Namespace

        args = Namespace(
            num_epochs=20,
            batch_size=32,
            learning_rate=3e-5,
            use_rl_training=False,
            output_dir="./test_output",
            memory_num_tokens=8,
            num_memory_experts=2
        )

        config = RBSTrainingConfig.from_args(args)

        assert config.num_epochs == 20
        assert config.batch_size == 32
        assert config.learning_rate == 3e-5
        assert config.use_rl_training is False
        assert config.output_dir == "./test_output"
        assert config.memory_num_tokens == 8
        assert config.num_memory_experts == 2

        # Test with missing args (should use defaults)
        args = Namespace(
            num_epochs=25,
            batch_size=16,
            # Other fields missing
        )

        config = RBSTrainingConfig.from_args(args)

        assert config.num_epochs == 25  # From args
        assert config.batch_size == 16   # From args
        assert config.learning_rate == 5e-5  # Default value


class TestConfigFactoryFunctions:
    """Test configuration factory functions."""

    def test_create_balanced_config(self):
        """Test creating balanced configuration."""
        config = create_balanced_config(
            num_epochs=8,
            batch_size=12,
            memory_num_tokens=12,
            num_memory_experts=3,
            use_rl_training=True,
            custom_param="test"  # This should be ignored
        )

        assert config.num_epochs == 8
        assert config.batch_size == 12
        assert config.memory_num_tokens == 12
        assert config.num_memory_experts == 3
        assert config.use_rl_training is True
        assert config.learning_rate == 5e-5
        assert config.rl_learning_rate == 1e-4
        assert config.lambda_cost == 0.01
        assert config.belief_state_threshold == 0.7

        # Test RL start epoch calculation
        config = create_balanced_config(num_epochs=12)
        assert config.rl_start_epoch == 4  # max(2, 12 // 3)

        config = create_balanced_config(num_epochs=2)
        assert config.rl_start_epoch == 2  # max(2, 2 // 3)

    def test_create_quick_debug_config(self):
        """Test creating debug configuration."""
        config = create_quick_debug_config(
            memory_num_tokens=6,
            num_memory_experts=3
        )

        assert config.num_epochs == 2
        assert config.batch_size == 2
        assert config.memory_num_tokens == 6
        assert config.num_memory_experts == 3
        assert config.use_rl_training is True
        assert config.rl_start_epoch == 1
        assert config.eval_frequency == 1
        assert config.save_frequency == 1
        assert config.logging_steps == 1
        assert config.warmup_steps == 1
        assert config.dataloader_num_workers == 0
        assert config.use_wandb is False

    def test_create_balanced_config_with_kwargs(self):
        """Test balanced config with keyword arguments."""
        config = create_balanced_config(
            num_epochs=10,
            learning_rate=1e-4,
            rl_weight=0.2,
            lambda_cost=0.02
        )

        assert config.num_epochs == 10
        assert config.learning_rate == 1e-4  # Overridden
        assert config.rl_weight == 0.2      # Overridden
        assert config.lambda_cost == 0.02   # Overridden
        assert config.batch_size == 8       # Default

    def test_create_quick_debug_config_with_kwargs(self):
        """Test debug config with keyword arguments."""
        config = create_quick_debug_config(
            num_epochs=3,
            batch_size=4,
            custom_param="ignored"
        )

        assert config.num_epochs == 3       # Overridden
        assert config.batch_size == 4       # Overridden
        assert config.memory_num_tokens == 4  # Default for debug


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_config_save_directory(self):
        """Test saving config when directory doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "nested", "dir", "config.json")

            config = RBSTrainingConfig(test_param=42)
            config.save(config_path)

            assert os.path.exists(config_path)

    def test_invalid_json_load(self):
        """Test loading invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "invalid.json")

            # Write invalid JSON
            with open(config_path, 'w') as f:
                f.write("invalid json content")

            with pytest.raises(json.JSONDecodeError):
                RBSTrainingConfig.load(config_path)

    def test_missing_config_file(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            RBSTrainingConfig.load("/non/existent/path.json")

    def test_config_with_extra_fields_in_json(self):
        """Test loading JSON with extra fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "extra_fields.json")

            # Create JSON with extra fields
            data = {
                "num_epochs": 10,
                "batch_size": 8,
                "extra_field": "should be ignored",
                "another_extra": 123
            }

            with open(config_path, 'w') as f:
                json.dump(data, f)

            # Should load without error and ignore extra fields
            config = RBSTrainingConfig.load(config_path)
            assert config.num_epochs == 10
            assert config.batch_size == 8
            assert not hasattr(config, "extra_field")

    def test_config_post_init_validation(self):
        """Test that validation is called during __post_init__."""
        # This should raise during construction
        with pytest.raises(ValueError, match="num_memory_experts must be between"):
            RBSTrainingConfig(num_memory_experts=0)

    def test_config_immutability_of_loaded(self):
        """Test that loaded config can be modified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")

            # Save config
            original = RBSTrainingConfig(num_epochs=5)
            original.save(config_path)

            # Load and modify
            loaded = RBSTrainingConfig.load(config_path)
            loaded.num_epochs = 15

            assert loaded.num_epochs == 15
            assert original.num_epochs == 5  # Original unchanged