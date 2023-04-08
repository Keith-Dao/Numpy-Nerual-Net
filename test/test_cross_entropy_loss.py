"""
This module tests the cross entropy loss class.
"""
import math

import numpy as np
from numpy._typing import NDArray
import pytest

from src.cross_entropy_loss import CrossEntropyLoss
from . import FLOAT_TOLERANCE


class TestCrossEntropyLoss:
    """
    Cross entropy loss class tester.
    """
    # region Fixtures
    @pytest.fixture
    def loss(self, request) -> CrossEntropyLoss:
        """
        Cross entropy loss.
        """
        return CrossEntropyLoss(request.param)

    @pytest.fixture
    def data_small_close(self) -> tuple[NDArray, NDArray, list[int]]:
        """
        A single example with logits close to the target label after
        applying softmax to the logits.

        Returns:
            Logits, one-hot encoded label and ground truth labels.
        """
        return (
            np.array([1, 0, 0]),
            np.array([1, 0, 0]),
            [0]
        )

    @pytest.fixture
    def data_small_exact(self) -> tuple[NDArray, NDArray, list[int]]:
        """
        A single example with logits that exactly match the target label after
        applying softmax to the logits.

        Returns:
            Logits, one-hot encoded label and ground truth labels.
        """
        return (
            np.array([0, 999, 0]),
            np.array([0, 1, 0]),
            [1]
        )

    @pytest.fixture
    def data_small_far(self) -> tuple[NDArray, NDArray, list[int]]:
        """
        A single example with logits that do not match the target label after
        applying softmax to the logits.

        Returns:
            Logits, one-hot encoded label and ground truth labels
        """
        return (
            np.array([0, 1, 1]),
            np.array([1, 0, 0]),
            [0]
        )

    @pytest.fixture
    def data_large(self) -> tuple[NDArray, NDArray, list[int]]:
        """
        An example with logits of minibatch size 3.

        Returns:
            Logits, one-hot encoded labels and ground truth labels.
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
            ]),
            [0, 0, 0]
        )
    # endregion Fixtures

    # region Init tests
    @pytest.mark.parametrize("reduction", [
        "sum", "mean"
    ])
    def test_init(self, reduction):
        """
        Test cross-entropy loss init.
        """
        loss = CrossEntropyLoss(reduction)
        assert loss.reduction == reduction

    @pytest.mark.parametrize("reduction", [
        "test", "Invalid", 1, [], 1.23
    ])
    def test_init_with_invalid_reduction(self, reduction):
        """
        Test cross-entropy loss init with invalid reductions.
        """
        with pytest.raises(ValueError):
            CrossEntropyLoss(reduction)
    # endregion Init tests

    # region Load tests
    @pytest.mark.parametrize("loss", [
        "sum", "mean"
    ], indirect=["loss"])
    def test_from_dict(self, loss):
        """
        Test the from_dict method.
        """
        attributes = loss.to_dict()
        new_loss = CrossEntropyLoss.from_dict(attributes)
        assert loss.reduction == new_loss.reduction

    @pytest.mark.parametrize("loss", [
        "sum", "mean"
    ], indirect=["loss"])
    def test_from_dict_with_invalid_class(self, loss):
        """
        Test the from_dict method with an invalid class.
        """
        attributes = loss.to_dict()
        attributes["class"] = "test"
        with pytest.raises(ValueError):
            CrossEntropyLoss.from_dict(attributes)

    @pytest.mark.parametrize("loss", [
        "sum", "mean"
    ], indirect=["loss"])
    @pytest.mark.parametrize("reduction", [
        "test", "Invalid", 1, [], 1.23
    ])
    def test_from_dict_with_invalid_reduction(self, loss, reduction):
        """
        Test the from_dict method with an invalid reduction.
        """
        attributes = loss.to_dict()
        attributes["reduction"] = reduction
        with pytest.raises(ValueError):
            CrossEntropyLoss.from_dict(attributes)
    # endregion Load tests

    # region Save tests
    @pytest.mark.parametrize("loss, reduction", [
        ("sum", "sum"),
        ("mean", "mean")
    ], indirect=["loss"])
    def test_to_dict(self, loss, reduction):
        """
        Test the to_dict method.
        """
        assert loss.to_dict() == {
            "class": "CrossEntropyLoss",
            "reduction": reduction
        }
    # endregion Save tests

    # region Forward pass tests
    @pytest.mark.parametrize("loss, data, result", [
        ("mean", "data_small_close", 0.5514447139320511),
        ("mean", "data_small_exact", 0),
        ("mean", "data_small_far", 1.8619948040582512),
        ("sum", "data_large", 2.4134395179903025),
        ("mean", "data_large", 0.8044798393301008),
    ], indirect=["loss"])
    def test_forward(self, loss, data, result, request):
        """
        Test the forward pass for cross entropy loss.
        """
        logits, one_hot_encoded, labels = request.getfixturevalue(data)
        assert math.isclose(
            loss(logits, one_hot_encoded),
            result,
            abs_tol=FLOAT_TOLERANCE
        )
        assert math.isclose(
            loss(logits, labels),
            result,
            abs_tol=FLOAT_TOLERANCE
        )

    @pytest.mark.parametrize("loss", [
        "mean", "sum"
    ], indirect=["loss"])
    @pytest.mark.parametrize("data", [
        (np.array([1, 1, 1]), []),
        (np.array([]), [1, 1, 1]),
        (np.array([]), [])
    ])
    def test_forward_with_missing_data(self, loss, data):
        """
        Test the forward pass with missing data.
        """
        logits, targets = data
        with pytest.raises(ValueError):
            loss(logits, targets)

    @pytest.mark.parametrize("loss", [
        "mean", "sum"
    ], indirect=["loss"])
    @pytest.mark.parametrize("data", [
        (np.array([1, 1, 1]), [1.0]),
        (np.array([1, 1, 1]), [[1]]),
        (np.array([1, 1, 1]), ["test"]),
    ])
    def test_forward_with_wrong_label_type(self, loss, data):
        """
        Test the forward pass with the wrong label type.
        """
        logits, targets = data
        with pytest.raises(TypeError):
            loss(logits, targets)

    @pytest.mark.parametrize("loss", [
        "mean", "sum"
    ], indirect=["loss"])
    @pytest.mark.parametrize("data", [
        (np.array([1, 1, 1]), [1, 2]),
        (np.array([[1, 1, 1], [1, 1, 1]]), [1]),
        (np.array([[1, 1, 1]]), np.array([[0, 1, 0], [0, 0, 1]])),
        (np.array([[1, 1, 1], [1, 1, 1]]), np.array([[0, 1, 0]])),

    ])
    def test_forward_with_mismatched_shape(self, loss, data):
        """
        Test the forward pass with mismatched shapes.
        """
        logits, targets = data
        with pytest.raises(ValueError):
            loss(logits, targets)

    @pytest.mark.parametrize("loss", [
        "mean", "sum"
    ], indirect=["loss"])
    @pytest.mark.parametrize("data", [
        (np.array([1, 1, 1]), [3]),
        (np.array([[1, 1, 1], [1, 1, 1]]), [1, 4])

    ])
    def test_forward_with_invalid_labels(self, loss, data):
        """
        Test the forward pass with mismatched shapes.
        """
        logits, targets = data
        with pytest.raises(ValueError):
            loss(logits, targets)
    # endregion Forward pass tests

    # region Backward pass tests
    @pytest.mark.parametrize("loss, data, grad", [
        (
            "mean", "data_small_close",
            np.array([
                -0.42388312,  0.21194156, 0.21194156
            ])
        ),
        ("mean", "data_small_exact", np.array([0, 0, 0])),
        (
            "mean", "data_small_far",
            np.array([
                -0.8446376, 0.4223188,  0.4223188
            ])
        ),
        (
            "sum", "data_large", np.array([
                [-0.42388312,  0.21194156,  0.21194156],
                [0.,     0.,     0.],
                [-0.8446376,  0.4223188, 0.4223188]
            ])
        ),
        (
            "mean", "data_large", np.array([
                [-0.14129437, 0.07064719, 0.07064719],
                [0.,    0.,    0.],
                [-0.28154587, 0.14077293, 0.14077293]
            ])
        ),
    ], indirect=["loss"])
    def test_backward(self, loss, data, grad, request):
        """
        Test the backward pass for cross entropy loss.
        """
        logits, one_hot_encoded, labels = request.getfixturevalue(data)
        loss(logits, one_hot_encoded)
        assert np.allclose(loss.backward(), grad, atol=FLOAT_TOLERANCE)
        loss(logits, labels)
        assert np.allclose(loss.backward(), grad, atol=FLOAT_TOLERANCE)

    @ pytest.mark.parametrize("loss, data", [
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
        logits, one_hot_encoded, labels = request.getfixturevalue(data)
        with pytest.raises(RuntimeError):
            loss.backward()
        loss(logits, one_hot_encoded)
        loss.backward()
        loss(logits, labels)
        loss.backward()
    # endregion Backward pass tests

    # region Built-ins tests
    @ pytest.mark.parametrize("loss, other, result", [
        ("sum", CrossEntropyLoss("sum"), True),
        ("sum", CrossEntropyLoss("mean"), False),
        ("sum", {"reduction": "sum"}, False),
        ("sum", "sum", False),
        ("mean", CrossEntropyLoss("mean"), True),
        ("mean", CrossEntropyLoss("sum"), False),
        ("mean", {"reduction": "mean"}, False),
        ("mean", "mean", False),
    ], indirect=["loss"])
    def test_dunder_eq(self, loss, other, result):
        """
        Test the __eq__ method.
        """
        assert (loss == other) is result
    # endregion Built-ins tests
