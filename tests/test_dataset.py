"""Unit tests for ChurnDataset."""

import pytest
import torch

from src.data.dataset import ChurnDataset


class TestChurnDataset:
    """Tests for ChurnDataset class."""

    def test_len_returns_correct_count(self) -> None:
        """Test that __len__ returns the number of samples."""
        num_samples = 10
        seq_len = 30
        num_features = 5

        time_series = torch.randn(num_samples, seq_len, num_features)
        text_data = [f"Sample text {i}" for i in range(num_samples)]
        labels = torch.randint(0, 2, (num_samples,))

        dataset = ChurnDataset(time_series, text_data, labels)

        assert len(dataset) == num_samples

    def test_getitem_returns_correct_structure(self) -> None:
        """Test that __getitem__ returns expected dictionary keys."""
        num_samples = 5
        seq_len = 30
        num_features = 5

        time_series = torch.randn(num_samples, seq_len, num_features)
        text_data = [f"Sample text {i}" for i in range(num_samples)]
        labels = torch.randint(0, 2, (num_samples,))

        dataset = ChurnDataset(time_series, text_data, labels)
        sample = dataset[0]

        assert "time_series" in sample
        assert "text" in sample
        assert "label" in sample

    def test_getitem_returns_correct_shapes(self) -> None:
        """Test that __getitem__ returns tensors with correct shapes."""
        num_samples = 5
        seq_len = 30
        num_features = 5

        time_series = torch.randn(num_samples, seq_len, num_features)
        text_data = [f"Sample text {i}" for i in range(num_samples)]
        labels = torch.randint(0, 2, (num_samples,))

        dataset = ChurnDataset(time_series, text_data, labels)
        sample = dataset[2]

        assert sample["time_series"].shape == (seq_len, num_features)
        assert isinstance(sample["text"], str)
        assert sample["label"].shape == ()

    def test_mismatched_lengths_raises_error(self) -> None:
        """Test that mismatched data lengths raise ValueError."""
        time_series = torch.randn(10, 30, 5)
        text_data = [f"Sample text {i}" for i in range(8)]
        labels = torch.randint(0, 2, (10,))

        with pytest.raises(ValueError):
            ChurnDataset(time_series, text_data, labels)

    def test_mismatched_labels_raises_error(self) -> None:
        """Test that mismatched label length raises ValueError."""
        time_series = torch.randn(10, 30, 5)
        text_data = [f"Sample text {i}" for i in range(10)]
        labels = torch.randint(0, 2, (7,))

        with pytest.raises(ValueError):
            ChurnDataset(time_series, text_data, labels)
