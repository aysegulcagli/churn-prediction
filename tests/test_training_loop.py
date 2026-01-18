"""End-to-end smoke test for training loop."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.fusion_model import FusionModel
from src.models.time_series_encoder import TimeSeriesEncoder
from src.training.trainer import Trainer


class MockTextEncoder(nn.Module):
    """Mock text encoder for testing."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(32, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        dummy = torch.randn(batch_size, 32, device=input_ids.device)
        return self.linear(dummy)


class DummyDataset(Dataset):
    """Dummy dataset for smoke testing."""

    def __init__(self, num_samples: int = 20) -> None:
        self.num_samples = num_samples
        self.seq_len = 30
        self.ts_features = 10
        self.text_len = 32

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "time_series": torch.randn(self.seq_len, self.ts_features),
            "input_ids": torch.randint(0, 1000, (self.text_len,)),
            "attention_mask": torch.ones(self.text_len, dtype=torch.long),
            "label": torch.randint(0, 2, ()).float(),
        }


class TestTrainingLoop:
    """Smoke tests for training loop."""

    def test_fit_runs_one_epoch(self) -> None:
        """Test that fit runs for one epoch and returns history."""
        ts_encoder = TimeSeriesEncoder(
            input_size=10,
            hidden_size=32,
            num_layers=1,
        )
        text_encoder = MockTextEncoder(hidden_size=64)

        model = FusionModel(
            time_series_encoder=ts_encoder,
            text_encoder=text_encoder,
            fusion_hidden_size=64,
            dropout=0.1,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device("cpu")
        config = {"epochs": 1}

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
        )

        train_dataset = DummyDataset(num_samples=20)
        val_dataset = DummyDataset(num_samples=10)

        train_loader = DataLoader(train_dataset, batch_size=4)
        val_loader = DataLoader(val_dataset, batch_size=4)

        history = trainer.fit(train_loader, val_loader, epochs=1)

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 1
        assert len(history["val_loss"]) == 1
        assert isinstance(history["train_loss"][0], float)
        assert isinstance(history["val_loss"][0], float)
