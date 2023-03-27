"""
This module tests the utils module.
"""
import numpy as np
import pytest

from neural_net.utils import softmax, log_softmax
from neural_net.test import FLOAT_TOLERANCE


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
