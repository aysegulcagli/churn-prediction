"""Evaluation metrics for churn prediction."""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Accuracy score between 0 and 1.
    """
    pass


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate precision score.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Precision score between 0 and 1.
    """
    pass


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate recall score.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Recall score between 0 and 1.
    """
    pass


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        F1 score between 0 and 1.
    """
    pass


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate area under ROC curve.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        ROC-AUC score between 0 and 1.
    """
    pass


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate area under precision-recall curve.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        PR-AUC score between 0 and 1.
    """
    pass


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
    pass
