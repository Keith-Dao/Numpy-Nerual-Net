"""
This module contains the image loader.
"""
import itertools
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
        utils.check_type(batch_size, int, "batch_size")
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


class ImageLoader:
    """
    Image loader.
    """

    def __init__(
        self,
        folder_path: str,
        preprocessing: list[Callable[..., NDArray]],
        label_processor: Callable[[str], int],
        train_test_split: float = 1,
        **kwargs
    ) -> None:
        """
        Image loader init.

        Args:
            folder_path: Path to the root of the data folder.
            preprocessing: The preprocessing steps for the data
            label_processor: The processor for the label into an int
            train_test_split: The proportion of data to be used for training.
                                Must be in the range [0, 1].
        Keyword args:
            file_formats: File formats to accept
            shuffle: Whether or not to shuffle the dataset before splitting
        """
        utils.check_type(train_test_split, (int, float), "train_test_split")

        if not 0 <= train_test_split <= 1:
            raise ValueError(
                "train_test_split must be in the range [0, 1], got"
                f" {train_test_split}."
            )

        path: pathlib.Path = pathlib.Path(folder_path)
        file_formats = kwargs.get("file_formats", [".png"])
        files: list[pathlib.Path] = [
            file_path
            for file_path in path.glob("**/*.*")
            if file_path.suffix in file_formats
        ]
        if not files:
            raise ValueError(
                f"No matching files were found at {path} with the extensions"
                f" {file_formats}")

        if kwargs.get("shuffle", True):
            utils.shuffle(files, inplace=True)
        training_size = int(len(files) * train_test_split)
        self._train = files[:training_size]
        self._test = files[training_size:]

        self._preprocessing = preprocessing
        self._label_processor = label_processor

        self._classes = []

    # region Iterator
    def _get_iter(
        self,
        dataset: str,
        batch_size: int,
        **kwargs
    ) -> DatasetIterator:
        """
        Gets a dataset iterator.

        Args:
            dataset: The name of the dataset to use.
            batch_size: The size of each batch

        Keyword args:
            drop_last: Whether or not to drop the last batch if it does not
                match the batch size
            shuffle: Whether or not to shuffle the data

        Returns:
            Dataset iterator.
        """
        if dataset not in ["train", "test"]:
            raise ValueError(
                f"Selected dataset does not exist, got f{dataset}.")

        return DatasetIterator(
            getattr(self, f"_{dataset}"),
            preprocessing=self._preprocessing,
            label_processor=self._label_processor,
            batch_size=batch_size,
            **kwargs
        )
    # endregion Iterator

    # region Labels
    @property
    def classes(self) -> list[str]:
        """
        Get all the classes in the dataset.

        NOTE: The classes are not collected until this is first called.
        However, the classes are stored after so subsequent calls will be
        faster.

        Returns:
            All the classes in the dataset.
        """
        if self._classes:
            return self._classes

        classes: set[str] = set()
        for path in itertools.chain(self._train, self._test):
            classes.add(path.parent.name)
        self._classes = list(sorted(classes))
        return self._classes

    # endregion Labels

    # region Built-ins
    def __call__(
        self,
        dataset: str,
        batch_size: int,
        **kwargs
    ) -> DatasetIterator:
        """
        Gets a dataset iterator.

        Args:
            dataset: The name of the dataset to use.
            batch_size: The size of each batch

        Keyword args:
            drop_last: Whether or not to drop the last batch if it does not
                match the batch size
            shuffle: Whether or not to shuffle the data

        Returns:
            Dataset iterator.
        """
        return self._get_iter(
            dataset,
            batch_size,
            **kwargs
        )
    # endregion Built-ins
