"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    roc_auc,
    compute_all_metrics,
)


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_accuracy_perfect(self) -> None:
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert accuracy(y_true, y_pred) == 1.0

    def test_accuracy_half_correct(self) -> None:
        """Test accuracy with 50% correct predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert accuracy(y_true, y_pred) == 0.5

    def test_precision_perfect(self) -> None:
        """Test precision with no false positives."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert precision(y_true, y_pred) == 1.0

    def test_precision_with_false_positives(self) -> None:
        """Test precision with false positives."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 0, 1, 1])
        assert precision(y_true, y_pred) == pytest.approx(2 / 3)

    def test_recall_perfect(self) -> None:
        """Test recall with no false negatives."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert recall(y_true, y_pred) == 1.0

    def test_recall_with_false_negatives(self) -> None:
        """Test recall with false negatives."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1])
        assert recall(y_true, y_pred) == 0.5

    def test_f1_score_perfect(self) -> None:
        """Test F1 score with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert f1_score(y_true, y_pred) == 1.0

    def test_f1_score_balanced(self) -> None:
        """Test F1 score calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        expected_precision = 0.5
        expected_recall = 0.5
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        assert f1_score(y_true, y_pred) == pytest.approx(expected_f1)

    def test_roc_auc_perfect(self) -> None:
        """Test ROC-AUC with perfect separation."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        assert roc_auc(y_true, y_prob) == 1.0

    def test_roc_auc_random(self) -> None:
        """Test ROC-AUC with random predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        assert roc_auc(y_true, y_prob) == 0.5

    def test_compute_all_metrics(self) -> None:
        """Test compute_all_metrics returns all expected keys."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])

        metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
