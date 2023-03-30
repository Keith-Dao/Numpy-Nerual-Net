"""
This module contains the cross entropy loss class.
"""

from typing import Callable
import numpy as np
from numpy.typing import NDArray

from src import utils


class CrossEntropyLoss:
    """
    Cross entropy loss.
    """
    REDUCTIONS: dict[str, Callable[..., float]] = {
        "mean": np.mean,
        "sum": np.sum
    }

    def __init__(
        self,
        reduction: str = "mean"
    ) -> None:
        """
        Cross entropy loss init.

        Args:
            reduction: Name of the reduction method (mean, sum)
        """
        self._probabilities: NDArray | None = None
        self._labels: NDArray | None = None
        self.reduction: str = reduction

    # region Forward pass
    def forward(self, logits: NDArray, labels: NDArray) -> float:
        """
        Calculate the cross entropy loss.

        Args:
            logits: The logits to calculate the cross entropy loss on
            labels: The one-hot encoded labels
        Returns:
            The cross entropy loss.
        """
        self._probabilities = utils.softmax(logits)
        self._labels = labels

        return CrossEntropyLoss.REDUCTIONS[self.reduction](
            -np.sum(labels * utils.log_softmax(logits), axis=-1),
            axis=None
        )
    # endregion Forward pass

    # region Backward pass
    def backward(self) -> NDArray:
        """
        Perform the backward pass using the previous forward inputs.

        Returns:
            The gradients with respect to the predictions.
        """
        if self._probabilities is None or self._labels is None:
            raise RuntimeError("forward must be called before backward.")

        # Rescale the gradients
        batch_size = self._probabilities.shape[0]
        if self.reduction == "sum" or len(self._probabilities.shape) == 1:
            batch_size = 1

        return (self._probabilities - self._labels) / batch_size

    # endregion Backward pass

    # region Built-ins
    def __call__(self, logits: NDArray, labels: NDArray) -> float:
        """
        Calculate the cross entropy loss.

        Args:
            logits: The logits to calculate the cross entropy loss on
            labels: The one-hot encoded labels
        Returns:
            The cross entropy loss.
        """
        return self.forward(logits, labels)
    # endregion Built-ins
