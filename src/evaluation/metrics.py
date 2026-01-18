"""Evaluation metrics for churn prediction."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score as sklearn_f1_score,
    roc_auc_score,
    average_precision_score,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Accuracy score between 0 and 1.
    """
    return float(accuracy_score(y_true, y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate precision score.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Precision score between 0 and 1.
    """
    return float(precision_score(y_true, y_pred, zero_division=0))


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate recall score.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Recall score between 0 and 1.
    """
    return float(recall_score(y_true, y_pred, zero_division=0))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        F1 score between 0 and 1.
    """
    return float(sklearn_f1_score(y_true, y_pred, zero_division=0))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate area under ROC curve.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        ROC-AUC score between 0 and 1.
    """
    return float(roc_auc_score(y_true, y_prob))


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate area under precision-recall curve.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        PR-AUC score between 0 and 1.
    """
    return float(average_precision_score(y_true, y_prob))


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for positive class.
        threshold: Classification threshold for converting probabilities to labels.

    Returns:
        Dictionary containing all metric scores.
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc(y_true, y_prob),
        "pr_auc": pr_auc(y_true, y_prob),
    }
