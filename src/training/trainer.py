"""Training orchestration for churn prediction model."""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    """Handles model training, validation, and logging.

    Attributes:
        model: The fusion model to train.
        optimizer: Optimizer for parameter updates.
        criterion: Loss function for training.
        device: Device to run training on (cpu/cuda).
        config: Training configuration dictionary.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: dict[str, Any],
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Model to train.
            optimizer: Optimizer instance.
            criterion: Loss function.
            device: Device for computation.
            config: Training configuration containing hyperparameters.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Run one training epoch.

        Args:
            train_loader: DataLoader for training data.

        Returns:
            Dictionary containing training metrics (loss, accuracy, etc.).
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            time_series = batch["time_series"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].float().to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(time_series, input_ids, attention_mask)
            loss = self.criterion(logits.squeeze(-1), labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"loss": avg_loss}

    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Run validation.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                time_series = batch["time_series"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].float().to(self.device)

                logits = self.model(time_series, input_ids, attention_mask)
                loss = self.criterion(logits.squeeze(-1), labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"loss": avg_loss}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> dict[str, list[float]]:
        """Full training loop with validation.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Number of epochs to train.

        Returns:
            Dictionary containing training history.
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])

        return history

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save the checkpoint.
            epoch: Current epoch number.
        """
        pass

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Epoch number from which to resume.
        """
        pass
