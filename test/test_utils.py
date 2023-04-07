"""
This module tests the utils module.
"""
from collections import Counter

import numpy as np
import pytest

from src.utils import check_type, one_hot_encode, softmax, log_softmax, shuffle
from . import FLOAT_TOLERANCE


# pylint: disable=invalid-name, too-few-public-methods
class TestSoftmax:
    """
    Softmax function tester.
    """
    @pytest.mark.parametrize("x, true_p", [
        (np.array([1, 1, 1]), np.array([0.3333, 0.3333, 0.3333])),
        (np.array([1, 0, 0]), np.array([0.5761, 0.2119, 0.2119])),
        (np.array([-1, -1, -1]), np.array([0.3333, 0.3333, 0.3333])),
        (np.array([999, 0, 0]), np.array([1, 0, 0])),
        (
            np.array([
                [1, 1, 1],
                [1, 0, 0],
                [999, 0, 0]
            ]),
            np.array([
                [0.3333, 0.3333, 0.3333],
                [0.5761, 0.2119, 0.2119],
                [1, 0, 0]
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
        (np.array([1, 1, 1]), np.array([-1.0986, -1.0986, -1.0986])),
        (np.array([1, 0, 0]), np.array([-0.5514, -1.5514, -1.5514])),
        (np.array([-1, -1, -1]), np.array([-1.0986, -1.0986, -1.0986])),
        (np.array([999, 0, 0]), np.array([0, -999, -999])),
        (
            np.array([
                [1, 1, 1],
                [1, 0, 0],
                [999, 0, 0]
            ]),
            np.array([
                [-1.0986, -1.0986, -1.0986],
                [-0.5514, -1.5514, -1.5514],
                [0, -999, -999]
            ])
        )
    ])
    def test_softmax(self, x, true_p):
        """
        Tests the log softmax function.
        """
        assert np.allclose(log_softmax(x), true_p, atol=FLOAT_TOLERANCE)


class TestShuffle:
    """
    Shuffle function tester.
    """
    @pytest.mark.parametrize("data, equal", [
        (list(range(10)), lambda a, b: a == b),
        (np.arange(10), np.array_equal)
    ])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_shuffle(self, data, equal, inplace):
        """
        Test the shuffle function.
        """
        data_copy = data.copy()
        shuffled = shuffle(data, inplace=inplace)
        assert not inplace == equal(data, data_copy), \
            f"Shuffle should be {'' if inplace else 'not '}done inplace"
        assert not equal(data_copy, shuffled)
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
