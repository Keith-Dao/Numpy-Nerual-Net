"""
This module tests the utils module.
"""
import numpy as np
import pytest

from neural_net.utils import softmax


class TestSoftmax:
    TOLERANCE = 1e-4

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
        assert np.allclose(softmax(x), true_p, atol=1e-4)
