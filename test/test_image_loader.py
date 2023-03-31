"""
This module tests the image loader module.
"""
from collections import Counter
import pathlib
from typing import Callable

import pytest
import numpy as np
from numpy.typing import NDArray

from src.image_loader import DatasetIterator


# pylint: disable=protected-access, invalid-name, too-many-public-methods
# pyright: reportGeneralTypeIssues=false
class TestDatasetIterator:
    """
    Dataset iterator tester
    """

    # region Fixtures
    @pytest.fixture
    def data(self) -> tuple[list[pathlib.Path], NDArray, list[int]]:
        """
        Dummy filepaths, data and classes.

        Returns:
            A tuple with dummy filepaths and the associated data and
            classes.
        """
        return (
            [
                pathlib.Path("data/0/145256656.png"),
                pathlib.Path("data/1/451584859.png"),
                pathlib.Path("data/0/389658641.png"),
            ],
            np.array([
                [[1, 4, 5], [2, 5, 6], [6, 5, 6]],
                [[4, 5, 1], [5, 8, 4], [8, 5, 9]],
                [[3, 8, 9], [6, 5, 8], [6, 4, 1]]
            ]),
            [0, 1, 0]
        )

    @pytest.fixture
    def preprocessing(self) -> list[Callable[..., NDArray]]:
        return [
            lambda path: np.array([int(c) for c in path.stem]).reshape(3, 3)
        ]

    @pytest.fixture
    def label_processor(self) -> Callable[[str], int]:
        return lambda label: int(label)

    @pytest.fixture
    def iterator(
        self,
        data,
        preprocessing,
        label_processor,
        request
    ) -> DatasetIterator:
        batch_size, drop_last = request.param
        return DatasetIterator(
            data[0],
            preprocessing,
            label_processor,
            batch_size,
            drop_last,
            shuffle=False
        )
    # endregion Fixtures

    # region Init tests
    @pytest.mark.parametrize("batch_size", [1, 2, 100])
    @pytest.mark.parametrize("drop_last", [True, False])
    def test_init(
        self,
        data,
        preprocessing,
        label_processor,
        batch_size,
        drop_last
    ):
        """
        Test a valid DatasetIterator init.
        """
        iterator = DatasetIterator(
            data[0],
            preprocessing,
            label_processor,
            batch_size,
            drop_last,
            shuffle=False
        )
        assert iterator._data is not data[0]
        assert iterator._data == data[0]
        assert iterator._preprocessing == preprocessing
        assert iterator._label_processor == label_processor
        assert iterator._batch_size == batch_size
        assert iterator._drop_last == drop_last

    @pytest.mark.parametrize("batch_size", [
        "batch_size", 0.156, [1]
    ])
    def test_init_batch_size_with_wrong_type(
        self,
        data,
        preprocessing,
        label_processor,
        batch_size
    ):
        """
        Test a DatasetIterator init with the incorrect type for batch_size.
        """
        with pytest.raises(TypeError):
            DatasetIterator(
                data[0],
                preprocessing,
                label_processor,
                batch_size
            )

    @pytest.mark.parametrize("batch_size", [
        0, -1, -4354
    ])
    def test_init_batch_size_with_invalid_value(
        self,
        data,
        preprocessing,
        label_processor,
        batch_size
    ):
        """
        Test a DatasetIterator init with the incorrect type for batch_size.
        """
        with pytest.raises(ValueError):
            DatasetIterator(
                data[0],
                preprocessing,
                label_processor,
                batch_size
            )

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_init_shuffle(
        self,
        data,
        preprocessing,
        label_processor,
        shuffle
    ):
        iterator = DatasetIterator(
            data[0],
            preprocessing,
            label_processor,
            1,
            shuffle=shuffle
        )
        assert Counter(iterator._data) == Counter(data[0])
        assert shuffle == (iterator._data != data[0]), \
            f"The data should {'' if shuffle else 'not '}be shuffled."

    def test_init_with_invalid_preprocessing(self, data, label_processor):
        """
        Test init with invalid preprocessing functions.
        """
        def invalid_preprocessing(*_) -> int:
            return 0

        iterator = DatasetIterator(
            data[0],
            [invalid_preprocessing],
            label_processor,
            1
        )
        # The function return type can only be checked at runtime
        with pytest.raises(ValueError):
            next(iterator)

    def test_init_with_invalid_label_processor(self, data, preprocessing):
        """
        Test init with invalid label_processor function.
        """
        def invalid_label_processor(*_) -> str:
            return "NOT VALID"

        iterator = DatasetIterator(
            data[0],
            preprocessing,
            invalid_label_processor,
            1
        )
        # The function return type can only be checked at runtime
        with pytest.raises(ValueError):
            next(iterator)
    # endregion Init tests

    # region Iterator tests
    @pytest.mark.parametrize("iterator", [
        (1, False),
        (1, True),
        (2, False),
        (2, True),
        (3, False),
        (3, True),
        (4, False),
        (4, True),
    ], indirect=["iterator"])
    def test_iter(self, iterator, data):
        """
        Test the iterator yields the correct data.
        """
        _, true_data, true_labels = data
        batch_size = iterator._batch_size
        total_batches = len(iterator)  # Tested in test_length

        for batch, (data, labels) in enumerate(iterator):
            assert batch < total_batches, \
                f"Expected {total_batches} batches, got at least {batch + 1}."
            assert np.array_equal(
                data,
                true_data[batch * batch_size: (batch + 1) * batch_size]
            )
            assert labels == true_labels[
                batch * batch_size: (batch + 1) * batch_size]

        with pytest.raises(StopIteration):
            next(iterator)
    # endregion Iterator tests

    # region Length tests
    @pytest.mark.parametrize("iterator, length", [
        ((1, False), 3),
        ((1, True), 3),
        ((2, False), 2),
        ((2, True), 1),
        ((3, False), 1),
        ((3, True), 1),
        ((4, False), 1),
        ((4, True), 0),
    ], indirect=["iterator"])
    def test_length(self, iterator, length):
        """
        Test the length method returns the correct length for the iterator.
        """
        assert len(iterator) == length
    # endregion Length tests
