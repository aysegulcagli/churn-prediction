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
        aggregated: If True, return aggregated features instead of raw time series.
    """

    def __init__(
        self,
        time_series_data: torch.Tensor,
        text_data: list[str],
        labels: torch.Tensor,
        aggregated: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            time_series_data: Time series features for each user.
            text_data: Text data (support tickets, feedback) for each user.
            labels: Binary churn labels.
            aggregated: If True, return aggregated tabular features.

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
        self.aggregated = aggregated

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
            Dictionary containing either:
                - time_series, text, label (when aggregated=False)
                - features, label (when aggregated=True)
        """
        if self.aggregated:
            time_series = self.time_series_data[idx]

            mean = time_series.mean(dim=0)
            std = time_series.std(dim=0)
            min_val = time_series.min(dim=0).values
            max_val = time_series.max(dim=0).values
            last_val = time_series[-1]

            features = torch.cat([mean, std, min_val, max_val, last_val])

            return {
                "features": features,
                "label": self.labels[idx],
            }

        return {
            "time_series": self.time_series_data[idx],
            "text": self.text_data[idx],
            "label": self.labels[idx],
        }
