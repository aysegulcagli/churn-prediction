"""Unit tests for TimeSeriesEncoder."""

import torch

from src.models.time_series_encoder import TimeSeriesEncoder


class TestTimeSeriesEncoder:
    """Tests for TimeSeriesEncoder class."""

    def test_output_shape(self) -> None:
        """Test that forward returns correct output shape."""
        batch_size = 8
        seq_len = 30
        input_size = 10
        hidden_size = 128

        encoder = TimeSeriesEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
        )

        x = torch.randn(batch_size, seq_len, input_size)
        output = encoder(x)

        assert output.shape == (batch_size, hidden_size)

    def test_single_layer_no_dropout(self) -> None:
        """Test encoder with single layer works without dropout error."""
        batch_size = 4
        seq_len = 20
        input_size = 5
        hidden_size = 64

        encoder = TimeSeriesEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.2,
        )

        x = torch.randn(batch_size, seq_len, input_size)
        output = encoder(x)

        assert output.shape == (batch_size, hidden_size)

    def test_different_sequence_lengths(self) -> None:
        """Test encoder handles different sequence lengths."""
        batch_size = 4
        input_size = 5
        hidden_size = 64

        encoder = TimeSeriesEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
        )

        for seq_len in [10, 30, 50]:
            x = torch.randn(batch_size, seq_len, input_size)
            output = encoder(x)
            assert output.shape == (batch_size, hidden_size)
