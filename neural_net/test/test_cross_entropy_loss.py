"""
This module tests the cross entropy loss class.
"""
import math

import numpy as np
from numpy._typing import NDArray
import pytest

from neural_net.cross_entropy_loss import CrossEntropyLoss
from neural_net.test import FLOAT_TOLERANCE


class TestCrossEntropyLoss:
    # region Fixtures
    @pytest.fixture
    def loss(self, request) -> CrossEntropyLoss:
        """
        Cross entropy loss.
        """
        return CrossEntropyLoss(request.param)

    @pytest.fixture
    def data_small_close(self) -> tuple[NDArray, NDArray]:
        """
        A single example with logits close to the target label after
        applying softmax to the logits.

        Returns:
            Logits and one-hot encoded label.
        """
        return (
            np.array([1, 0, 0]),
            np.array([1, 0, 0])
        )

    @pytest.fixture
    def data_small_exact(self) -> tuple[NDArray, NDArray]:
        """
        A single example with logits that exactly match the target label after
        applying softmax to the logits.

        Returns:
            Logits and one-hot encoded label.
        """
        return (
            np.array([0, 999, 0]),
            np.array([0, 1, 0])
        )

    @pytest.fixture
    def data_small_far(self) -> tuple[NDArray, NDArray]:
        """
        A single example with logits that do not match the target label after
        applying softmax to the logits.

        Returns:
            Logits and one-hot encoded label.
        """
        return (
            np.array([0, 1, 1]),
            np.array([1, 0, 0])
        )

    @pytest.fixture
    def data_large(self) -> tuple[NDArray, NDArray]:
        """
        An example with logits of minibatch size 3.

        Returns:
            Logits and one-hot encoded labels.
        """
        return (
            np.array([
                [1, 0, 0],
                [999, 0, 0],
                [0, 1, 1]
            ]),
            np.array([
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]
            ])
        )
    # endregion Fixtures

    # region Forward pass tests
    @pytest.mark.parametrize("loss, data, result", [
        ("mean", "data_small_close", 0.5514),
        ("mean", "data_small_exact", 0),
        ("mean", "data_small_far", 1.862),
        ("sum", "data_large", 2.4134),
        ("mean", "data_large", 0.8045),
    ], indirect=["loss"])
    def test_forward(self, loss, data, result, request):
        """
        Test the forward pass for cross entropy loss.
        """
        logits, labels = request.getfixturevalue(data)
        assert math.isclose(
            loss(logits, labels),
            result,
            abs_tol=FLOAT_TOLERANCE
        )
    # endregion Forward pass tests

    # region Backward pass tests
    @pytest.mark.parametrize("loss, data, grad", [
        ("mean", "data_small_close", np.array([-0.4239,  0.2119,  0.2119])),
        ("mean", "data_small_exact", np.array([0, 0, 0])),
        ("mean", "data_small_far", np.array([-0.8446,  0.4223,  0.4223])),
        (
            "sum", "data_large", np.array([
                [-0.4239,  0.2119,  0.2119],
                [0, 0, 0],
                [-0.8446,  0.4223,  0.4223]
            ])
        ),
        (
            "mean", "data_large", np.array([
                [-0.1413,  0.0706,  0.0706],
                [0.0000,  0.0000,  0.0000],
                [-0.2815,  0.1408,  0.1408]
            ])
        ),
    ], indirect=["loss"])
    def test_backward(self, loss, data, grad, request):
        """
        Test the backward pass for cross entropy loss.
        """
        logits, labels = request.getfixturevalue(data)
        loss(logits, labels)
        assert np.allclose(loss.backward(), grad, atol=FLOAT_TOLERANCE)

    @pytest.mark.parametrize("loss, data", [
        ("mean", "data_small_close"),
        ("mean", "data_small_exact"),
        ("mean", "data_small_far"),
        ("sum", "data_large"),
        ("mean", "data_large"),
    ], indirect=["loss"])
    def test_backward_error(self, loss, data, request):
        """
        Test an error is raised when backward is called before forward.
        """
        logits, labels = request.getfixturevalue(data)
        with pytest.raises(RuntimeError):
            loss.backward()
        loss(logits, labels)
        loss.backward()
    # endregion Backward pass tests
