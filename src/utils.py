"""
This module contains various utility functions.
"""
import random
from typing import Any, Type

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


def log_softmax(in_: NDArray) -> NDArray:
    """
    The log softmax function.

    Args:
        in_: The input vector or matrix

    Returns:
        The log softmax of the given input.
    """
    in_ -= np.max(in_, axis=-1, keepdims=True)  # For numerical stability
    return in_ - np.log(np.exp(in_).sum(axis=-1, keepdims=True))


def shuffle(data: list | NDArray, inplace: bool = False) -> list | NDArray:
    """
    Shuffles the data.

    Args:
        data: The data to be shuffled
        inplace: Whether or not the shuffle in place

    Returns:
        The shuffled data.
    """
    if not inplace:
        data = data.copy()

    for i in range(len(data) - 1, 0, -1):
        j = random.randint(0, i - 1)
        data[i], data[j] = data[j], data[i]
    return data


def one_hot_encode(labels: list[int], classes: int) -> NDArray:
    """
    One hot encode the labels.

    Args:
        labels: The label of each example
        classes: The total number of classes in the dataset

    Returns:
        The one hot encoded labels.
    """
    encoded = np.zeros((len(labels), classes))
    rows, cols = zip(*enumerate(labels))
    encoded[rows, cols] = 1
    return encoded


def check_type(
    value: Any,
    types: tuple[Type, ...] | Type,
    variable_name: str
) -> None:
    """
    Checks the value is one of the given types. Otherwise raise a TypeError.

    Args:
        value: Value to check
        types: The expected types
        variable_name: The name of the variable containing the value
    """
    if not isinstance(value, types):
        raise TypeError(
            f"Invalid type for {variable_name}. Expected"
            f" {' or '.join(type_.__name__ for type_ in types)},"
            f" got {type(value).__name__}."
        )
