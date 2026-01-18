"""Time series encoder for behavioral sequence data."""

import torch
import torch.nn as nn


class TimeSeriesEncoder(nn.Module):
    """Encodes time series behavioral data into a fixed-size representation.

    Uses LSTM/GRU to capture temporal patterns in user activity sequences.

    Attributes:
        hidden_size: Dimension of the hidden state.
        num_layers: Number of recurrent layers.
        dropout: Dropout probability between layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the time series encoder.

        Args:
            input_size: Number of features in the input time series.
            hidden_size: Dimension of the hidden state.
            num_layers: Number of stacked recurrent layers.
            dropout: Dropout probability for regularization.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode time series input.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Encoded representation of shape (batch_size, hidden_size).
        """
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]
