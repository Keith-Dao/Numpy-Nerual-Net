"""
This module tests the activation functions module.
"""
import numpy as np
from numpy.typing import NDArray
import pytest

from src.activation_functions import NoActivation, ReLU


# pylint: disable=protected-access, invalid-name
class TestActivationFunctions:
    """
    Activation function tests.
    """

    # region Fixtures
    @pytest.fixture
    def no_activation(self) -> tuple[NoActivation, NDArray, NDArray, NDArray]:
        """
        Creates no activation function and the input and output values.
        """
        X = np.arange(-10, 10, dtype=float).reshape(5, 4)
        Y = X.copy()
        grad = np.ones_like(Y)

        return (
            NoActivation(),
            X,
            Y,
            grad
        )

    @pytest.fixture
    def relu(self) -> tuple[ReLU, NDArray, NDArray, NDArray]:
        """
        Creates a ReLU function and the input and output values.
        """
        X = np.arange(-10, 10, dtype=float).reshape(5, 4)
        Y = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [2, 3, 4, 5],
            [6, 7, 8, 9]
        ])
        grad = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])

        return (
            ReLU(),
            X,
            Y,
            grad
        )
    # endregion Fixtures

    # region Forward tests
    @pytest.mark.parametrize("data", ["no_activation", "relu"])
    def test_forward(self, data, request):
        """
        Test the forward pass of the activation function.
        """
        function, X, Y, _ = request.getfixturevalue(data)
        assert np.array_equal(function(X), Y)
    # endregion Forward tests

    # region Backward tests
    @pytest.mark.parametrize("data", ["no_activation", "relu"])
    def test_backward(self, data, request):
        """
        Test the backward pass of the activation function.
        """
        function, X, _, grad = request.getfixturevalue(data)
        function(X)
        assert np.array_equal(function.backward(), grad)

    @pytest.mark.parametrize("data", ["no_activation", "relu"])
    def test_backward_call_before_forward(self, data, request):
        """
        Test calling the backward pass before calling forward.
        """
        function, X, _, _ = request.getfixturevalue(data)
        with pytest.raises(ValueError):
            function.backward()
        function(X)
        function.backward()
    # endregion backward tests

    # region Built-ins tests
    @pytest.mark.parametrize("first, other, result", [
        ("no_activation", "no_activation", True),
        ("no_activation", "relu", False),
        ("relu", "relu", True),
        ("relu", "no_activation", False)
    ])
    def test_dunder_eq(self, first, other, result, request):
        """
        Test the __eq__ method.
        """
        function, *_ = request.getfixturevalue(first)
        other_function, *_ = request.getfixturevalue(other)
        assert (function == other_function) is result

    @pytest.mark.parametrize("data", ["no_activation", "relu"])
    @pytest.mark.parametrize("other", [
        "no_activation",
        "relu",
        1,
        1.23,
        [],
        "test"
    ])
    def test_dunder_eq_with_invalid_types(self, data, other, request):
        """
        Test the __eq__ method with invalid types.
        """
        function, *_ = request.getfixturevalue(data)
        assert (function == other) is False
    # endregion Built-ins tests
