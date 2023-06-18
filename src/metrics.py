"""
This module contains various metric functions.
"""

import numpy as np
from numpy.typing import NDArray


# region Confusion matrix
def get_new_confusion_matrix(num_classes: int) -> NDArray:
    """
    Create a new confusion matrix with the given number of classes,
    where the rows represent the predicted class amd the columns
    represent the actual class

    Args:
        num_classes: The number of classes to include

    Returns:
        The confusion matrix.
    """
    if num_classes < 1:
        raise ValueError(
            "The number of classes must be greater than 0. Got: "
            f"{num_classes}")
    return np.zeros((num_classes, num_classes))


def add_to_confusion_matrix(
    confusion_matrix: NDArray,
    predictions: list[int],
    actual: list[int]
) -> None:
    """
    Add the predictions to the given confusion matrix.

    Args:
        confusion_matrix: The confusion matrix to add to
        predictions: The predicted classes
        actual: The actual classes
    """
    if len(predictions) != len(actual):
        raise ValueError(
            "The length of predictions and actual does not match."
        )

    for pred, act in zip(predictions, actual):
        confusion_matrix[pred, act] += 1
# endregion Confusion matrix


# region Metrics
SINGLE_VALUE_METRICS = {"accuracy", "loss"}


def accuracy(confusion_matrix: NDArray) -> float:
    """
    The accuracy for the given confusion matrix.

    Args:
        confusion_matrix: The confusion matrix

    Returns:
        The accuracy.
    """
    return confusion_matrix.diagonal().sum() / confusion_matrix.sum()


def precision(confusion_matrix: NDArray) -> NDArray:
    """
    The precision for all classes in the confusion matrix.

    Args:
        confusion_matrix: The confusion matrix

    Returns:
        The precision.
    """
    correct_predictions = confusion_matrix.diagonal().astype(float)
    actual = confusion_matrix.sum(axis=1, dtype=float)
    return np.divide(
        correct_predictions,
        actual,
        out=np.zeros_like(correct_predictions),
        where=actual != 0
    )


def recall(confusion_matrix: NDArray) -> NDArray:
    """
    The recall for all classes in the confusion matrix.

    Args:
        confusion_matrix: The confusion matrix

    Returns:
        The recall.
    """
    correct_predictions = confusion_matrix.diagonal().astype(float)
    predicted = confusion_matrix.sum(axis=0, dtype=float)
    return np.divide(
        correct_predictions,
        predicted,
        out=np.zeros_like(correct_predictions),
        where=predicted != 0
    )


def f1_score(confusion_matrix: NDArray) -> NDArray:
    """
    The f1 score for all classes in the confusion matrix.

    Args:
        confusion_matrix: The confusion matrix

    Returns:
        The f1 scores.
    """
    precision_ = precision(confusion_matrix)
    recall_ = recall(confusion_matrix)
    return np.divide(
        2 * precision_ * recall_,
        precision_ + recall_,
        out=np.zeros_like(precision_),
        where=precision_ + recall_ != 0
    )
# endregion Metrics
