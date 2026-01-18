"""CLI entry point for evaluating the churn prediction model."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the multi-modal churn prediction model."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to configuration file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint to evaluate.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed data directory.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Data split to evaluate on.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results"),
        help="Path to output directory for evaluation results.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for predictions.",
    )

    return parser.parse_args()


def main() -> None:
    """Main evaluation function.

    Loads model checkpoint, runs inference on test data,
    and computes evaluation metrics.
    """
    pass


if __name__ == "__main__":
    main()
