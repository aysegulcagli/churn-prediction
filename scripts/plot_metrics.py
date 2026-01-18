"""Generate visual evaluation reports after training."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_training_loss(losses: list[float], save_path: Path) -> None:
    """Plot training loss over epochs.

    Args:
        losses: List of loss values per epoch.
        save_path: Path to save the figure.
    """
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, marker="o", linewidth=2, markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    """Plot ROC curve with AUC score.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        save_path: Path to save the figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    """Plot Precision-Recall curve with average precision.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        save_path: Path to save the figure.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR Curve (AP = {ap:.3f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    """Generate evaluation plots with placeholder data."""
    figures_dir = Path("artifacts/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    training_losses = [0.65, 0.52, 0.43, 0.38, 0.34, 0.31, 0.29, 0.27, 0.26, 0.25]

    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10)
    y_prob = np.clip(y_true * 0.6 + np.random.rand(100) * 0.4, 0, 1)

    plot_training_loss(training_losses, figures_dir / "train_loss.png")
    plot_roc_curve(y_true, y_prob, figures_dir / "roc_curve.png")
    plot_pr_curve(y_true, y_prob, figures_dir / "pr_curve.png")

    print(f"Figures saved to {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
