"""Evaluate tabular baseline model and generate plots."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ChurnDataset
from src.evaluation.metrics import roc_auc, pr_auc, f1_score
from src.models.tabular_baseline import TabularBaseline

# Plotting imports
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


def main() -> None:
    """Main evaluation and plotting function for tabular baseline."""
    # Fixed paths
    user_logs_path = "data/raw/user_logs.csv"
    labels_path = "data/raw/train_v2.csv"
    checkpoint_path = "artifacts/tabular_baseline.pt"
    figures_dir = Path("figures")

    # Create figures directory if needed
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please train the tabular baseline first.")
        sys.exit(1)

    # Load VAL dataset with aggregated=True
    print("=" * 60, flush=True)
    print("Evaluating TABULAR BASELINE on VAL split", flush=True)
    print("=" * 60, flush=True)
    print("Loading VAL dataset (aggregated=True)...", flush=True)

    val_dataset = ChurnDataset.from_kkbox(
        user_logs_path=user_logs_path,
        labels_path=labels_path,
        split="val",
        aggregated=True,
    )
    print(f"Val dataset size: {len(val_dataset)}", flush=True)

    # Shape validation: fail fast if mismatch
    sample = val_dataset[0]
    if "features" not in sample:
        print("ERROR: Dataset does not contain 'features' key.")
        print("This indicates aggregated=True was not applied correctly.")
        sys.exit(1)

    feature_dim = sample["features"].shape[0]
    expected_dim = 7
    if feature_dim != expected_dim:
        print(f"ERROR: Shape mismatch detected!")
        print(f"  Expected feature dimension: {expected_dim}")
        print(f"  Actual feature dimension: {feature_dim}")
        print("This may be caused by a stale cache. Delete data/cache/kkbox_val.pt and retry.")
        sys.exit(1)

    # Create DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
    )

    # Instantiate model
    model = TabularBaseline(input_dim=7)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from: {checkpoint_path}")

    # Collect predictions
    all_labels = []
    all_probs = []
    total_val_batches = len(val_loader)

    print("Running inference...", flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            labels = batch["label"].numpy()
            features = batch["features"].to(device)

            # Shape check on first batch
            if batch_idx == 0:
                if features.shape[1] != 7:
                    print(f"ERROR: Batch features have shape {features.shape}, expected (*, 7)")
                    sys.exit(1)

            logits = model(features)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

            all_labels.append(labels)
            all_probs.append(probs)

            # Progress every 25%
            if (batch_idx + 1) % max(1, total_val_batches // 4) == 0:
                print(f"  Val progress: batch {batch_idx + 1}/{total_val_batches}", flush=True)

    # Concatenate all batches
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    # Compute standard metrics
    y_pred_05 = (y_prob >= 0.5).astype(int)

    auc_roc = roc_auc(y_true, y_prob)
    auc_pr = pr_auc(y_true, y_prob)
    f1_at_05 = f1_score(y_true, y_pred_05)

    # Threshold sweep for best F1
    thresholds = np.arange(0.05, 0.96, 0.01)
    f1_scores = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1_t = f1_score(y_true, y_pred_t)
        f1_scores.append(f1_t)

    f1_scores = np.array(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # Print results
    print()
    print("=" * 60)
    print("Validation Results: Tabular Baseline")
    print("=" * 60)
    print(f"ROC-AUC:           {auc_roc:.4f}")
    print(f"PR-AUC:            {auc_pr:.4f}")
    print(f"F1 @ t=0.5:        {f1_at_05:.4f}")
    print(f"Best F1:           {best_f1:.4f} (threshold = {best_threshold:.2f})")
    print("=" * 60)

    # --- Plot 1: ROC and PR curves ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    axes[0].plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {auc_roc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve - Tabular Baseline (Val)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR curve
    axes[1].plot(recall, precision, color="green", lw=2, label=f"PR (AUC = {auc_pr:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve - Tabular Baseline (Val)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    roc_pr_path = figures_dir / "tabular_roc_pr_val.png"
    plt.savefig(roc_pr_path, dpi=150)
    plt.close()
    print(f"Saved: {roc_pr_path}")

    # --- Plot 2: Threshold vs F1 ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, f1_scores, color="purple", lw=2)
    ax.axvline(x=best_threshold, color="red", linestyle="--", lw=1,
               label=f"Best threshold = {best_threshold:.2f}")
    ax.axvline(x=0.5, color="gray", linestyle=":", lw=1, label="Default threshold = 0.5")
    ax.scatter([best_threshold], [best_f1], color="red", s=100, zorder=5)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("Threshold vs F1 Score - Tabular Baseline (Val)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    threshold_f1_path = figures_dir / "tabular_threshold_f1_val.png"
    plt.savefig(threshold_f1_path, dpi=150)
    plt.close()
    print(f"Saved: {threshold_f1_path}")

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
