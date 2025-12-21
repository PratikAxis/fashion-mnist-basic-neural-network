"""Loss functions for the NumPy-only network."""

import numpy as np


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Compute mean cross-entropy loss for one-hot labels.

    y_true: (batch, num_classes) one-hot
    y_pred: (batch, num_classes) probabilities
    """
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]
    return float(loss)


def predict_classes(probs: np.ndarray) -> np.ndarray:
    return np.argmax(probs, axis=1)
