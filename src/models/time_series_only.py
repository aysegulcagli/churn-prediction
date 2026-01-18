"""Time-series-only baseline model for churn prediction."""

import torch
import torch.nn as nn

from src.models.time_series_encoder import TimeSeriesEncoder


class TimeSeriesOnly(nn.Module):
    """Baseline model using only time series data.

    Wraps the TimeSeriesEncoder with a classification head.
    Ignores text inputs to enable ablation comparison with FusionModel.
    """

    def __init__(self, time_series_encoder: TimeSeriesEncoder) -> None:
        """Initialize the model.

        Args:
            time_series_encoder: Pretrained or initialized time series encoder.
        """
        super().__init__()
        self.time_series_encoder = time_series_encoder
        self.classifier = nn.Linear(time_series_encoder.hidden_size, 1)

    def forward(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            time_series: Time series input of shape (batch_size, seq_len, num_features).
            input_ids: Text token IDs (ignored).
            attention_mask: Text attention mask (ignored).

        Returns:
            Logits of shape (batch_size, 1).
        """
        del input_ids, attention_mask  # Explicitly unused

        ts_embedding = self.time_series_encoder(time_series)
        logits = self.classifier(ts_embedding)

        return logits
