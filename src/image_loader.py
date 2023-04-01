"""
This module contains the image loader.
"""
import pathlib
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from src import utils


class DatasetIterator:
    """
    Dataset iterator.
    """

    def __init__(
        self,
        data: list[pathlib.Path],
        preprocessing: list[Callable[..., NDArray]],
        label_processor: Callable[[str], int],
        batch_size: int,
        **kwargs
    ) -> None:
        """
        Dataset iterator init.

        Args:
            data: The data to iterate through
            preprocessing: The preprocessing steps for the data
            label_processor: The processor for the label into an int
            batch_size: The batch size of the iterator
        Keyword args:
            drop_last: Whether or not to drop the last batch if it does not
                match the batch size
            shuffle: Whether or not to shuffle the data
        """
        if not isinstance(batch_size, int):
            raise TypeError(
                f"batch_size must be an int, got {type(batch_size).__name__}."
            )

        if batch_size < 1:
            raise ValueError(
                f"batch_size must be greater than 1, got {batch_size}."
            )

        self._data = data.copy()
        if kwargs.get("shuffle", True):
            self._data = utils.shuffle(self._data, inplace=True)
        self._label_processor = label_processor
        self._preprocessing = preprocessing
        self._batch_size = batch_size
        self._i = 0
        self._drop_last = kwargs.get("drop_last", False)

    # region Built-ins
    def __iter__(self):
        return self

    def __next__(self) -> tuple[NDArray, list[int]]:
        if self._i == len(self._data):
            raise StopIteration

        if self._drop_last and self._i + self._batch_size > len(self._data):
            raise StopIteration

        def get_next_item():
            filepath = self._data[self._i]
            self._i += 1

            data = filepath
            for step in self._preprocessing:
                data = step(data)
            if not isinstance(data, np.ndarray):
                raise ValueError(
                    "The preprocessing steps must result in a NumPy array."
                )

            label = self._label_processor(filepath.parent.name)
            if not isinstance(label, int):
                raise ValueError(
                    "The label processor must result in an int."
                )

            return data, label

        steps = min(len(self._data) - self._i, self._batch_size)
        data, labels = zip(*(get_next_item() for _ in range(steps)))
        data = np.array(data)
        labels = list(labels)

        return data, labels  # pyright: ignore [reportGeneralTypeIssues]

    def __len__(self) -> int:
        return (
            len(self._data) + (
                self._batch_size - 1 if not self._drop_last else 0
            )
        ) // self._batch_size
    # endregion Built-ins
