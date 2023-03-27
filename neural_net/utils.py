"""
This module contains various utility functions.
"""

import numpy as np
from numpy.typing import NDArray


def softmax(in_: NDArray) -> NDArray:
    """
    The softmax function.

    Args:
        in_: The input vector or matrix
    Returns:
        The row wise probability vector of the given input.
    """
    exp = np.exp(in_ - np.max(in_, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)
