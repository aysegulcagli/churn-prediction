"""Unit tests for FusionModel."""

import pytest
import torch
import torch.nn as nn

from src.models.time_series_encoder import TimeSeriesEncoder


class MockTextEncoder(nn.Module):
    """Mock text encoder for testing without loading pretrained model."""

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(64, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        dummy = torch.randn(batch_size, 64)
        return self.linear(dummy)


class TestFusionModel:
    """Tests for FusionModel class."""

    def test_output_shape(self) -> None:
        """Test that forward returns correct output shape."""
        from src.models.fusion_model import FusionModel

        batch_size = 4
        seq_len = 30
        ts_input_size = 10
        ts_hidden_size = 64
        text_hidden_size = 128
        fusion_hidden_size = 256

        ts_encoder = TimeSeriesEncoder(
            input_size=ts_input_size,
            hidden_size=ts_hidden_size,
            num_layers=2,
        )
        text_encoder = MockTextEncoder(hidden_size=text_hidden_size)

        model = FusionModel(
            time_series_encoder=ts_encoder,
            text_encoder=text_encoder,
            fusion_hidden_size=fusion_hidden_size,
            dropout=0.3,
        )

        time_series = torch.randn(batch_size, seq_len, ts_input_size)
        input_ids = torch.randint(0, 1000, (batch_size, 64))
        attention_mask = torch.ones(batch_size, 64, dtype=torch.long)

        output = model(time_series, input_ids, attention_mask)

        assert output.shape == (batch_size, 1)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        from src.models.fusion_model import FusionModel

        batch_size = 2
        seq_len = 20
        ts_input_size = 5
        ts_hidden_size = 32
        text_hidden_size = 64

        ts_encoder = TimeSeriesEncoder(
            input_size=ts_input_size,
            hidden_size=ts_hidden_size,
        )
        text_encoder = MockTextEncoder(hidden_size=text_hidden_size)

        model = FusionModel(
            time_series_encoder=ts_encoder,
            text_encoder=text_encoder,
            fusion_hidden_size=128,
        )

        time_series = torch.randn(batch_size, seq_len, ts_input_size)
        input_ids = torch.randint(0, 1000, (batch_size, 64))
        attention_mask = torch.ones(batch_size, 64, dtype=torch.long)

        output = model(time_series, input_ids, attention_mask)
        loss = output.sum()
        loss.backward()

        assert ts_encoder.lstm.weight_ih_l0.grad is not None
