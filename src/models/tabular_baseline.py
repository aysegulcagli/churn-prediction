"""Tabular baseline model for churn prediction."""

import torch
import torch.nn as nn


class TabularBaseline(nn.Module):
    """Simple logistic regression baseline using aggregated features.

    A minimal baseline that takes pre-computed tabular features
    and outputs churn logits via a single linear layer.
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize the baseline model.

        Args:
            input_dim: Dimension of the input feature vector.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (batch_size, input_dim).

        Returns:
            Logits of shape (batch_size, 1).
        """
        return self.linear(x)
