"""CLI entry point for training the churn prediction model."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the multi-modal churn prediction model."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to configuration file.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed data directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Path to output directory for checkpoints and logs.",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function.

    Loads configuration, initializes model and data,
    and runs the training loop.
    """
    pass


if __name__ == "__main__":
    main()
