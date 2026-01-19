"""CLI entry point for training the churn prediction model."""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.dataset import ChurnDataset
from src.models.tabular_baseline import TabularBaseline
from src.models.time_series_encoder import TimeSeriesEncoder
from src.models.time_series_only import TimeSeriesOnly
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the churn prediction model."
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
        help="Model type to train.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the checkpoint.",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine if aggregated based on model type
    aggregated = args.model_type == "tabular_baseline"

    # Load datasets
    print(f"Loading TRAIN split (aggregated={aggregated})...")
    train_dataset = ChurnDataset.from_kkbox(
        user_logs_path=str(args.user_logs),
        labels_path=str(args.labels),
        split="train",
        aggregated=aggregated,
    )
    print(f"Train dataset size: {len(train_dataset)}")

    print(f"Loading VAL split (aggregated={aggregated})...")
    val_dataset = ChurnDataset.from_kkbox(
        user_logs_path=str(args.user_logs),
        labels_path=str(args.labels),
        split="val",
        aggregated=aggregated,
    )
    print(f"Val dataset size: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
    )

    # Instantiate model
    if args.model_type == "tabular_baseline":
        model = TabularBaseline(input_dim=7)
        lr = 0.01
        weight_decay = 0.0
        use_gradient_clipping = False
    else:
        encoder = TimeSeriesEncoder(
            input_size=6,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
        )
        model = TimeSeriesOnly(encoder)
        lr = 0.001
        weight_decay = 1e-4
        use_gradient_clipping = True

    model = model.to(device)
    print(f"Model: {args.model_type}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training configuration
    max_epochs = 50
    patience = 5
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    print(f"Starting training (max_epochs={max_epochs}, patience={patience})...")
    print("-" * 60)

    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs} started", flush=True)

        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            labels = batch["label"].float().to(device)
            optimizer.zero_grad()

            if aggregated:
                features = batch["features"].to(device)
                logits = model(features)
            else:
                time_series = batch["time_series"].to(device)
                input_ids = torch.zeros(time_series.shape[0], 1, dtype=torch.long, device=device)
                attention_mask = torch.zeros(time_series.shape[0], 1, dtype=torch.long, device=device)
                logits = model(time_series, input_ids, attention_mask)

            loss = criterion(logits.squeeze(-1), labels)

            # NaN safety check
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected, skipping batch", flush=True)
                continue

            loss.backward()

            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            num_train_batches += 1

            # Progress every 10%
            if (batch_idx + 1) % max(1, total_train_batches // 10) == 0:
                print(f"  Train progress: batch {batch_idx + 1}/{total_train_batches}", flush=True)

        avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else float("nan")

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                labels = batch["label"].float().to(device)

                if aggregated:
                    features = batch["features"].to(device)
                    logits = model(features)
                else:
                    time_series = batch["time_series"].to(device)
                    input_ids = torch.zeros(time_series.shape[0], 1, dtype=torch.long, device=device)
                    attention_mask = torch.zeros(time_series.shape[0], 1, dtype=torch.long, device=device)
                    logits = model(time_series, input_ids, attention_mask)

                loss = criterion(logits.squeeze(-1), labels)
                val_loss += loss.item()
                num_val_batches += 1

                # Progress every 25%
                if (batch_idx + 1) % max(1, total_val_batches // 4) == 0:
                    print(f"  Val progress: batch {batch_idx + 1}/{total_val_batches}", flush=True)

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float("nan")

        print(f"Epoch {epoch + 1:02d}/{max_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best val loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    # Save checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": args.model_type,
            "model_state_dict": model.state_dict(),
        },
        args.output,
    )
    print(f"Checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
