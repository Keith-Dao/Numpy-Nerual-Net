"""
This module tests the image loader module.
"""
from collections import Counter
import shutil
from typing import Callable

from PIL import Image
import pytest
import numpy as np
from numpy.typing import NDArray

from src.image_loader import DatasetIterator, ImageLoader


# pylint: disable=protected-access, invalid-name, too-many-public-methods
# pylint: disable=redefined-outer-name, too-many-arguments
# pyright: reportGeneralTypeIssues=false
class TestFixtures:
    """
    Fixtures to be used for tests.
    """
    @pytest.fixture(scope="class")
    def data(self) -> tuple[NDArray, list[int]]:
        """
        Dummy filepaths, data and classes.

        Returns:
            A tuple with the data and associated class.
        """
        return (
            np.array([
                [[1, 4, 5], [2, 5, 6], [6, 5, 6]],
                [[4, 5, 1], [5, 8, 4], [8, 5, 9]],
                [[3, 8, 9], [6, 5, 8], [6, 4, 1]]
            ], dtype=np.uint8),
            [0, 0, 1]
        )

    @pytest.fixture(scope="class")
    def dummy_folder(self, tmp_path_factory, data):
        """
        Creates dummy files in a temporary path.
        """
        dirs = [
            tmp_path_factory.mktemp("0", numbered=False),
            tmp_path_factory.mktemp("1", numbered=False)
        ]

        for i, (x, label) in enumerate(zip(*data)):
            image = Image.fromarray(x)
            path = tmp_path_factory.getbasetemp() / str(label) \
                / f"{i}.png"
            image.save(path)

        yield tmp_path_factory.getbasetemp()
        for dir_ in dirs:
            shutil.rmtree(str(dir_))

    @pytest.fixture(scope="class")
    def dummy_files(self, dummy_folder):
        """
        Gets all the dummy file paths.
        """
        return list(
            sorted(
                dummy_folder.glob("**/*.png"),
                key=lambda path: path.name
            )
        )

    @pytest.fixture
    def preprocessing(self) -> list[Callable[..., NDArray]]:
        """
        Simple preprocessing steps.
        """
        return [
            lambda path: np.array(Image.open(path))
        ]

    @pytest.fixture
    def label_processor(self) -> Callable[[str], int]:
        """
        Generic label processor.
        """
        return int


class TestDatasetIterator(TestFixtures):
    """
    Dataset iterator tester
    """

    # region Fixtures
    @pytest.fixture
    def iterator(
        self,
        dummy_files,
        preprocessing,
        label_processor,
        request
    ) -> DatasetIterator:
        """
        Dataset iterator using the global fixtures.
        """
        batch_size, drop_last = request.param
        return DatasetIterator(
            dummy_files,
            preprocessing,
            label_processor,
            batch_size,
            drop_last=drop_last,
            shuffle=False
        )
    # endregion Fixtures

    # region Init tests
    @pytest.mark.parametrize("batch_size", [1, 2, 100])
    @pytest.mark.parametrize("drop_last", [True, False])
    def test_init(
        self,
        dummy_files,
        preprocessing,
        label_processor,
        batch_size,
        drop_last
    ):
        """
        Test a valid DatasetIterator init.
        """
        iterator = DatasetIterator(
            dummy_files,
            preprocessing,
            label_processor,
            batch_size,
            drop_last=drop_last,
            shuffle=False
        )
        assert iterator._data is not dummy_files
        assert iterator._data == dummy_files
        assert iterator._preprocessing == preprocessing
        assert iterator._label_processor == label_processor
        assert iterator._batch_size == batch_size
        assert iterator._drop_last == drop_last

    @pytest.mark.parametrize("batch_size", [
        "batch_size", 0.156, [1]
    ])
    def test_init_batch_size_with_wrong_type(
        self,
        dummy_files,
        preprocessing,
        label_processor,
        batch_size
    ):
        """
        Test a DatasetIterator init with the incorrect type for batch_size.
        """
        with pytest.raises(TypeError):
            DatasetIterator(
                dummy_files,
                preprocessing,
                label_processor,
                batch_size
            )

    @pytest.mark.parametrize("batch_size", [
        0, -1, -4354
    ])
    def test_init_batch_size_with_invalid_value(
        self,
        dummy_files,
        preprocessing,
        label_processor,
        batch_size
    ):
        """
        Test a DatasetIterator init with the incorrect type for batch_size.
        """
        with pytest.raises(ValueError):
            DatasetIterator(
                dummy_files,
                preprocessing,
                label_processor,
                batch_size
            )

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_init_shuffle(
        self,
        dummy_files,
        preprocessing,
        label_processor,
        shuffle
    ):
        """
        Tests the shuffle for the dataset iterator.
        """
        iterator = DatasetIterator(
            dummy_files,
            preprocessing,
            label_processor,
            1,
            shuffle=shuffle
        )
        assert Counter(iterator._data) == Counter(dummy_files)
        assert shuffle == (iterator._data != dummy_files), \
            f"The data should {'' if shuffle else 'not '}be shuffled."

    def test_init_with_invalid_preprocessing(
        self,
        dummy_files,
        label_processor
    ):
        """
        Test init with invalid preprocessing functions.
        """
        def invalid_preprocessing(*_) -> int:
            return 0

        iterator = DatasetIterator(
            dummy_files,
            [invalid_preprocessing],
            label_processor,
            1
        )
        # The function return type can only be checked at runtime
        with pytest.raises(ValueError):
            next(iterator)

    def test_init_with_invalid_label_processor(
        self,
        dummy_files,
        preprocessing
    ):
        """
        Test init with invalid label_processor function.
        """
        def invalid_label_processor(*_) -> str:
            return "NOT VALID"

        iterator = DatasetIterator(
            dummy_files,
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
        true_data, true_labels = data
        batch_size = iterator._batch_size
        total_batches = len(iterator)  # Tested in test_length

        for batch, (data_, labels) in enumerate(iterator):
            assert batch < total_batches, \
                f"Expected {total_batches} batches, got at least {batch + 1}."
            assert np.array_equal(
                data_,
                true_data[batch * batch_size: (batch + 1) * batch_size]
            )
            assert labels == true_labels[
                batch * batch_size: (batch + 1) * batch_size]
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


class TestImageLoader(TestFixtures):
    """
    Image loader tester.
    """
    # region Fixtures
    @pytest.fixture
    def loader(
        self,
        dummy_folder,
        preprocessing,
        label_processor,
        request
    ):
        """
        Image loader.
        """
        train_test_split = request.param
        return ImageLoader(
            str(dummy_folder),
            preprocessing,
            label_processor,
            train_test_split,
            file_formats=[".png"],
            shuffle=False
        )
    # endregion Fixtures

    # region Init tests
    def test_init(
        self,
        dummy_folder,
        preprocessing,
        label_processor
    ):
        """
        Test a valid image loader init.
        """
        image_loader = ImageLoader(
            str(dummy_folder),
            preprocessing,
            label_processor,
            .7,
            file_formats=[".png"],
            shuffle=False
        )

        # Fixture order is different
        dummy_files = list(dummy_folder.glob("**/*.png"))

        assert image_loader._train == dummy_files[:2]
        assert image_loader._test == dummy_files[2:]
        assert image_loader._preprocessing == preprocessing
        assert image_loader._label_processor == label_processor

    @pytest.mark.parametrize("split", [
        1.1,
        -0.2,
        100
    ])
    def test_init_split_value_error(
        self,
        dummy_folder,
        preprocessing,
        label_processor,
        split
    ):
        """
        Tests the image loader init when an invalid value for
        train_test_split is provided.
        """
        with pytest.raises(ValueError):
            ImageLoader(
                str(dummy_folder),
                preprocessing,
                label_processor,
                split,
                file_formats=[".png"]
            )

    @pytest.mark.parametrize("split", [
        "test",
        []
    ])
    def test_init_split_type_error(
        self,
        dummy_folder,
        preprocessing,
        label_processor,
        split
    ):
        """
        Tests the image loader init when an invalid type for
        train_test_split is provided.
        """
        with pytest.raises(TypeError):
            ImageLoader(
                str(dummy_folder),
                preprocessing,
                label_processor,
                split,
                file_formats=[".png"]
            )

    def test_init_shuffle(
        self,
        dummy_folder,
        preprocessing,
        label_processor
    ):
        """
        Test image loader init with shuffle.
        """
        image_loader = ImageLoader(
            str(dummy_folder),
            preprocessing,
            label_processor,
            1,
            file_formats=[".png"],
            shuffle=True
        )

        # Fixture order is different
        dummy_files = list(dummy_folder.glob("**/*.png"))
        assert image_loader._train != dummy_files
        assert Counter(image_loader._train) == Counter(dummy_files)

    def test_init_bad_path(
        self,
        preprocessing,
        label_processor
    ):
        """
        Test image loader init with a bad path.
        """
        with pytest.raises(ValueError):
            ImageLoader(
                "testing",
                preprocessing,
                label_processor,
                1,
                file_formats=[".png"],
                shuffle=True
            )

    @pytest.mark.parametrize("loader, train_size", [
        (0, 0),
        (0.3, 0),
        (1/3, 1),
        (0.4, 1),
        (0.6, 1),
        (2/3, 2),
        (0.7, 2),
        (0.9, 2),
        (1, 3)
    ], indirect=["loader"])
    def test_init_dataset_split(
        self,
        loader,
        train_size
    ):
        """
        Test the dataset is correctly splitting the data.
        """
        assert len(loader._train) == train_size
        assert len(loader._test) == 3 - train_size
    # endregion Init tests

    # region Iterator tests
    @pytest.mark.parametrize("dataset", [
        "train", "test"
    ])
    @pytest.mark.parametrize("batch_size", [
        1, 2, 3
    ])
    @pytest.mark.parametrize("loader", [
        0, 1/3, 2/3, 1
    ], indirect=["loader"])
    def test_iter(
        self,
        loader,
        data,
        batch_size,
        dataset
    ):
        """
        Test creating an iterator for a dataset.
        """
        true_x, true_labels = data
        for i, (x, label) in enumerate(
            loader(dataset, batch_size, shuffle=False)
        ):
            paths = getattr(loader, f"_{dataset}")[
                i * batch_size: (i + 1) * batch_size]
            true_idxs = [int(path.stem) for path in paths]
            assert np.array_equal(x, true_x[true_idxs])
            assert label == [true_labels[idx] for idx in true_idxs]

    @pytest.mark.parametrize("loader", [1], indirect=["loader"])
    def test_invalid_iter(
        self,
        loader
    ):
        """
        Test attempting to create an iterator for an invalid dataset.
        """
        with pytest.raises(ValueError):
            loader("invalid", 1)

    # endregion Iterator tests

    # region Classes tests
    @pytest.mark.parametrize("loader", [1], indirect=["loader"])
    def test_classes(
        self,
        loader,
        data
    ):
        """
        Test the classes property.
        """
        assert loader._classes == []
        _, classes = data
        classes = list(sorted({str(x) for x in classes}))
        assert loader.classes == classes
        assert loader._classes == classes
        assert loader.classes == classes
    # endregion Classes tests
