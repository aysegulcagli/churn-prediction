"""Dataset classes for multi-modal churn prediction."""

from typing import Any

import torch
from torch.utils.data import Dataset


class ChurnDataset(Dataset):
    """Multi-modal dataset for churn prediction.

    Handles time series behavioral data, text data, and churn labels.

    Attributes:
        time_series_data: Tensor of shape (num_samples, sequence_length, num_features).
        text_data: List of text strings or tokenized text tensors.
        labels: Tensor of binary churn labels (0 = retained, 1 = churned).
    """

    def __init__(
        self,
        time_series_data: torch.Tensor,
        text_data: list[str],
        labels: torch.Tensor,
    ) -> None:
        """Initialize the dataset.

        Args:
            time_series_data: Time series features for each user.
            text_data: Text data (support tickets, feedback) for each user.
            labels: Binary churn labels.

        Raises:
            ValueError: If data lengths do not match.
        """
        num_samples = time_series_data.shape[0]

        if len(text_data) != num_samples:
            raise ValueError(
                f"Time series has {num_samples} samples but text has {len(text_data)}"
            )

        if labels.shape[0] != num_samples:
            raise ValueError(
                f"Time series has {num_samples} samples but labels has {labels.shape[0]}"
            )

        self.time_series_data = time_series_data
        self.text_data = text_data
        self.labels = labels
        self.num_samples = num_samples

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
                - time_series: Time series tensor for the sample.
                - text: Text data for the sample.
                - label: Churn label.
        """
        return {
            "time_series": self.time_series_data[idx],
            "text": self.text_data[idx],
            "label": self.labels[idx],
        }
