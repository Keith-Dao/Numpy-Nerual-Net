"""
This module tests the utils module.
"""
from collections import Counter
import pathlib

import numpy as np
from PIL import Image
import pytest

from src.utils import (
    check_type,
    flatten,
    is_yes,
    logits_to_prediction,
    normalise_image,
    one_hot_encode,
    softmax,
    log_softmax,
    shuffle,
    image_to_array,
    normalise_array
)
from . import FLOAT_TOLERANCE


# pylint: disable=invalid-name, too-few-public-methods
class TestSoftmax:
    """
    Softmax function tester.
    """
    @pytest.mark.parametrize("x, true_p", [
        (np.array([1, 1, 1]), np.array([0.33333333, 0.33333333, 0.33333333])),
        (np.array([1, 0, 0]), np.array([0.57611688, 0.21194156, 0.21194156])),
        (
            np.array([-1, -1, -1]),
            np.array([0.33333333, 0.33333333, 0.33333333])
        ),
        (np.array([999, 0, 0]), np.array([1, 0, 0])),
        (
            np.array([
                [1, 1, 1],
                [1, 0, 0],
                [999, 0, 0]
            ]),
            np.array([
                [0.33333333, 0.33333333, 0.33333333],
                [0.57611688, 0.21194156, 0.21194156],
                [1.,         0.,         0.]
            ])
        )
    ])
    def test_softmax(self, x, true_p):
        """
        Tests the softmax function.
        """
        assert np.allclose(softmax(x), true_p, atol=FLOAT_TOLERANCE)


class TestLogSoftmax:
    """
    Log softmax function tester.
    """
    @pytest.mark.parametrize("x, true_p", [
        (
            np.array([1, 1, 1]),
            np.array([-1.09861229, -1.09861229, -1.09861229])
        ),
        (
            np.array([1, 0, 0]),
            np.array([-0.55144471, - 1.55144471, - 1.55144471])
        ),
        (
            np.array([-1, -1, -1]),
            np.array([-1.09861229, - 1.09861229, - 1.09861229])),
        (
            np.array([999, 0, 0]),
            np.array([0., - 999., - 999.])),
        (
            np.array([
                [1, 1, 1],
                [1, 0, 0],
                [999, 0, 0]
            ]),
            np.array([
                [-1.09861229e+00, - 1.09861229e+00, - 1.09861229e+00],
                [-5.51444714e-01, - 1.55144471e+00, - 1.55144471e+00],
                [0.00000000e+00, - 9.99000000e+02, - 9.99000000e+02]
            ])
        )
    ])
    def test_log_softmax(self, x, true_p):
        """
        Tests the log softmax function.
        """
        assert np.allclose(log_softmax(x), true_p, atol=FLOAT_TOLERANCE)


class TestShuffle:
    """
    Shuffle function tester.
    """
    @pytest.mark.parametrize("data", [
        list(range(10)),
        np.arange(10),
    ])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_shuffle(self, data, inplace):
        """
        Test the shuffle function.
        """
        data_copy = data.copy()
        shuffled = shuffle(data, inplace=inplace)
        assert inplace == (data is shuffled), \
            f"Shuffle should be {'' if inplace else 'not '}done inplace"
        assert Counter(data_copy) == Counter(shuffled)


class TestOneHotEncode:
    """
    One hot encode tester.
    """
    @pytest.mark.parametrize("labels, classes, encoded", [
        ([0, 1, 2], 3, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        ([1, 0, 2], 3, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])),
        ([3], 10, np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]))
    ])
    def test_one_hot_encode(self, labels, classes, encoded):
        """
        Test the one hot encode function.
        """
        assert np.array_equal(one_hot_encode(labels, classes), encoded)


class TestImageToArray:
    """
    Image to array tester.
    """
    @pytest.fixture(scope="class")
    def image_file(self, tmp_path_factory) -> pathlib.Path:
        """
        Dummy image file.
        """
        image_data = np.arange(100, dtype=np.uint8).reshape(10, 10)
        image_path = tmp_path_factory.mktemp("data") / "test.png"
        Image.fromarray(image_data).save(image_path)
        return image_path

    def test_image_to_array(self, image_file):
        """
        Tests the image to array function.
        """
        assert np.array_equal(
            image_to_array(image_file),
            np.arange(100).reshape(10, 10)
        )

    @pytest.mark.parametrize("image_file", [
        1, 1.23, []
    ])
    def test_image_to_array_with_invalid_type(self, image_file):
        """
        Tests the image to array function with an invalid type.
        """
        with pytest.raises(TypeError):
            image_to_array(image_file)


class TestNormaliseArray:
    """
    Normalise array tester.
    """
    @pytest.mark.parametrize("data, from_, to_, expected", [
        (np.array([0, 127.5, 255]), (0, 255), (0, 1), np.array([0, 0.5, 1])),
        (np.array([0, 127.5, 255]), (0, 255), (-1, 1), np.array([-1, 0, 1])),
        (np.array([0, 127.5, 255]), (0, 255), (-2, 2), np.array([-2, 0, 2])),
        (np.array([0, 127.5, 255]), (0, 255), (-2, 3), np.array([-2, 0.5, 3])),
    ])
    def test_normalise_array(self, data, from_, to_, expected):
        """
        Test normalise array.
        """
        assert np.array_equal(
            normalise_array(data, from_, to_),
            expected
        )

    @pytest.mark.parametrize("data", [np.array([0, 1, 2])])
    @pytest.mark.parametrize("from_, to_", [
        ((255, 0), (0, 1)),
        ((0, 0), (0, 1)),
        ((0, 255), (1, -1)),
        ((0, 255), (-1, -1)),
    ])
    def test_normalise_array_invalid_ranges(self, data, from_, to_):
        """
        Test normalise array with invalid ranges.
        """
        with pytest.raises(ValueError):
            normalise_array(data, from_, to_)


class TestNormaliseImage:
    """
    Normalise image tester.
    """

    def test_normalise_image(self):
        """
        Test normalise image.
        """
        assert np.array_equal(
            normalise_image(np.array([0, 127.5, 255])),
            np.array([-1, 0, 1])
        )


class TestFlatten:
    """
    Flatten tester.
    """
    @pytest.mark.parametrize("data, expected", [
        (np.array([1, 2, 3]), np.array([1, 2, 3])),
        (np.array([[1], [2], [3]]), np.array([1, 2, 3])),
        (
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            np.array([1, 2, 3, 4, 5, 6, 7, 8])
        )
    ])
    def test_flatten(self, data, expected):
        """
        Test flatten.
        """
        assert np.array_equal(flatten(data), expected)


class TestLogitsToPredictions:
    """
    Logits to predictions tester.
    """
    @pytest.mark.parametrize("logits, expected", [
        (np.array([1, 2, 3]), [2]),
        (np.array([[5, 2, 1], [2, 24, 1]]), [0, 1])
    ])
    def test_logits_to_predictions(self, logits, expected):
        """
        Tests the logits_to_prediction method.
        """
        assert logits_to_prediction(logits) == expected


class TestCheckType:
    """
    Check type tester.
    """
    @pytest.mark.parametrize("value, types, exception", [
        (1, (int, float), None),
        (1.23, (int, float), None),
        ("a", int, TypeError),
        ("a", (int, float), TypeError),
        (1, int, None)
    ])
    def test_check_type(self, value, types, exception):
        """
        Test the check type function.
        """
        if exception is None:
            check_type(value, types, "test")
            return

        with pytest.raises(exception):
            check_type(value, types, "test")


class TestIsYes:
    """
    Is yes tester.
    """
    @pytest.mark.parametrize("response, expected", [
        ("y", True),
        ("Y", True),
        ("n", False),
        ("N", False)
    ])
    def test_is_yes(self, response, expected):
        """
        Test the yes no prompt with valid responses.
        """
        assert is_yes(response) is expected

    @pytest.mark.parametrize("responses, expected", [
        ("y", True),
        ("q209fl;y", True),
        ("1wea42\]Y", True),
        ("43290\t2n", False),
        ("5325092-=N", False)
    ])
    def test_is_yes_retry(self, responses, expected, monkeypatch):
        """
        Test the yes no prompt with invalid characters that lead to a valid
        response.
        """
        response_iter = iter(responses)
        monkeypatch.setattr('builtins.input', lambda _: next(response_iter))
        assert is_yes("INVALID") is expected
