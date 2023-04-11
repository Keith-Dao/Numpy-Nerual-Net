"""
This module contains various utility functions.
"""
from collections.abc import Iterable
import pathlib
import random
from typing import Any, Type

import colorama
import numpy as np
from numpy.typing import NDArray
from PIL import Image


# region Mathematical functions
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
# endregion Mathematical functions


# region Array functions
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


def image_to_array(image_path: str | pathlib.Path) -> NDArray:
    """
    Opens the provided image path and returns it as a NumPy array.

    Args:
        image_path: The path to the image

    Returns:
        The NumPy array representation of the image.
    """
    check_type(image_path, (str, pathlib.Path), "image_path")
    return np.array(Image.open(image_path))


def normalise_array(
    data: NDArray,
    from_: tuple[float, float],
    to_: tuple[float, float]
) -> NDArray:
    """
    Normalise the given array from the current range to the new range.

    Args:
        data: The array to normalise
        from_: The current range of the data
        to_: The desired range of the data

    Returns:
        The data normalised to the new range.
    """
    from_min, from_max = from_
    if from_min >= from_max:
        raise ValueError(
            "The first value of from_ must be less than the second value."
        )
    to_min, to_max = to_
    if to_min >= to_max:
        raise ValueError(
            "The first value of to_ must be less than the second value."
        )
    return (
        (data - from_min) * (to_max - to_min) / (from_max - from_min)
        + to_min
    )


def normalise_image(data: NDArray) -> NDArray:
    """
    Normalise a standard PIL image array to [-1, 1].

    Args:
        data: The NumPy array representation of a PIL image

    Returns:
        The image normalised to [-1, 1].
    """
    return normalise_array(data, (0, 255), (-1, 1))
# endregion Array functions


# region Error functions
def print_warning(message: str) -> None:  # pragma: no cover
    """
    Print the provided message as a warning message.

    Args:
        message: Message to print.
    """
    colorama.just_fix_windows_console()
    print(f"{colorama.Fore.YELLOW}{message}{colorama.Style.RESET_ALL}")


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
            f""" {
                ' or '.join(type_.__name__ for type_ in types)
                if isinstance(types, Iterable)
                else types.__name__
            }"""
            f", got {type(value).__name__}."
        )
# endregion Error functions
