"""CLI entry point for evaluating the churn prediction model."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChurnDataset
from src.evaluation.metrics import roc_auc, pr_auc, f1_score
from src.models.tabular_baseline import TabularBaseline
from src.models.time_series_encoder import TimeSeriesEncoder
from src.models.time_series_only import TimeSeriesOnly


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the churn prediction model."
    )

    parser.add_argument(
        "--user-logs",
        type=Path,
        required=True,
        help="Path to user_logs.csv file.",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to train_v2.csv file.",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["tabular_baseline", "time_series_only"],
        required=True,
        help="Model type to evaluate.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint.",
    )

    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine if aggregated based on model type
    aggregated = args.model_type == "tabular_baseline"

    # Load VAL dataset (using VAL instead of TEST for faster evaluation)
    print("=" * 60, flush=True)
    print("NOTE: Evaluating on VAL split (not TEST) for efficiency", flush=True)
    print("=" * 60, flush=True)
    print("Loading VAL dataset (cache-aware)...", flush=True)
    val_dataset = ChurnDataset.from_kkbox(
        user_logs_path=str(args.user_logs),
        labels_path=str(args.labels),
        split="val",
        aggregated=aggregated,
    )
    print(f"Val dataset size: {len(val_dataset)}", flush=True)

    # Create DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
    )

    # Instantiate model architecture
    if args.model_type == "tabular_baseline":
        model = TabularBaseline(input_dim=7)
    else:
        encoder = TimeSeriesEncoder(
            input_size=6,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
        )
        model = TimeSeriesOnly(encoder)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from: {args.checkpoint}")

    # Collect predictions
    all_labels = []
    all_probs = []
    total_val_batches = len(val_loader)

    print("Running inference...", flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            labels = batch["label"].numpy()

            if aggregated:
                features = batch["features"].to(device)
                logits = model(features)
            else:
                time_series = batch["time_series"].to(device)
                input_ids = torch.zeros(time_series.shape[0], 1, dtype=torch.long, device=device)
                attention_mask = torch.zeros(time_series.shape[0], 1, dtype=torch.long, device=device)
                logits = model(time_series, input_ids, attention_mask)

            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

            all_labels.append(labels)
            all_probs.append(probs)

            # Progress every 25%
            if (batch_idx + 1) % max(1, total_val_batches // 4) == 0:
                print(f"  Val progress: batch {batch_idx + 1}/{total_val_batches}", flush=True)

    # Concatenate all batches
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    # Compute metrics
    y_pred = (y_prob >= 0.5).astype(int)

    auc_roc = roc_auc(y_true, y_prob)
    auc_pr = pr_auc(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)

    # Print results
    print()
    print("=" * 60)
    print(f"Validation Results: {args.model_type}")
    print("=" * 60)
    print(f"AUC-ROC:  {auc_roc:.4f}")
    print(f"AUC-PR:   {auc_pr:.4f}")
    print(f"F1 @ 0.5: {f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
