import json
import sys

from memxlnet.training.trainer import TrainingConfig, XLNetRecurrentTrainer


def main(config_path: str, model_path: str | None = None):
    """
    Evaluate a trained model using the saved training configuration.

    Args:
        config_path: Path to training_config.json
        model_path: Optional path to model directory (if different from config)
    """
    with open(config_path) as f:
        data = json.load(f)

    cfg = TrainingConfig(**data)

    # Override model path if provided
    if model_path is not None:
        cfg.model_name = model_path

    print("🔬 Starting Evaluation")
    print("=" * 50)
    print(f"Model: {cfg.model_name}")
    print(f"Dataset: {cfg.dataset_name} ({cfg.eval_split})")
    print(f"Max eval samples: {cfg.max_eval_samples or 'All'}")
    print(f"Batch size: {cfg.eval_batch_size}")
    print("=" * 50)

    trainer = XLNetRecurrentTrainer(cfg)
    train_loader, eval_loader, eval_dataset = trainer.prepare_data()

    print(f"📊 Evaluation dataset loaded: {len(eval_loader)} batches")

    # Run evaluation
    print("\n🚀 Running evaluation...")
    metrics = trainer.evaluate(eval_loader, eval_dataset)

    print("\n📈 Results:")
    print("=" * 30)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("=" * 30)

    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluate <path-to-training_config.json> [model_path]")
        print("  config_path: Path to training_config.json file")
        print("  model_path:  Optional path to model directory (overrides config)")
        raise SystemExit(2)

    config_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    main(config_path, model_path)
