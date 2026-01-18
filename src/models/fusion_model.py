"""Fusion model for multi-modal churn prediction."""

import torch
import torch.nn as nn

from src.models.text_encoder import TextEncoder
from src.models.time_series_encoder import TimeSeriesEncoder


class FusionModel(nn.Module):
    """Multi-modal fusion model for churn prediction.

    Combines representations from time series and text encoders
    to predict customer churn probability.

    Attributes:
        time_series_encoder: Encoder for behavioral sequence data.
        text_encoder: Encoder for customer text data.
        fusion_hidden_size: Dimension of the fusion layer.
        dropout: Dropout probability in fusion layers.
    """

    def __init__(
        self,
        time_series_encoder: TimeSeriesEncoder,
        text_encoder: TextEncoder,
        fusion_hidden_size: int = 256,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the fusion model.

        Args:
            time_series_encoder: Pretrained or initialized time series encoder.
            text_encoder: Pretrained or initialized text encoder.
            fusion_hidden_size: Hidden dimension for fusion layers.
            dropout: Dropout probability for regularization.
        """
        super().__init__()
        self.time_series_encoder = time_series_encoder
        self.text_encoder = text_encoder

        ts_hidden_size = time_series_encoder.hidden_size
        text_hidden_size = text_encoder.hidden_size
        combined_size = ts_hidden_size + text_hidden_size

        self.fusion = nn.Sequential(
            nn.Linear(combined_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, 1),
        )

    def forward(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the fusion model.

        Args:
            time_series: Time series input of shape (batch_size, seq_len, num_features).
            input_ids: Text token IDs of shape (batch_size, max_length).
            attention_mask: Text attention mask of shape (batch_size, max_length).

        Returns:
            Churn probability of shape (batch_size, 1).
        """
        ts_embedding = self.time_series_encoder(time_series)
        text_embedding = self.text_encoder(input_ids, attention_mask)

        combined = torch.cat([ts_embedding, text_embedding], dim=1)
        logits = self.fusion(combined)

        return logits
