"""
This module tests the activation functions module.
"""
import numpy as np
from numpy.typing import NDArray
import pytest

from neural_net.activation_functions import NoActivation, ReLU


class TestActivationFunctions:
    """
    Activation function tests.
    """
    # pylint: disable=protected-access, invalid-name

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
        Y = X * (X > 0)
        grad = X > 0

        return (
            ReLU(),
            X,
            Y,
            grad
        )

    # Forward test
    @pytest.mark.parametrize("data", ["no_activation", "relu"])
    def test_forward(self, data, request):
        """
        Test the forward pass of the activation function.
        """
        function, X, Y, _ = request.getfixturevalue(data)
        assert np.array_equal(function(X), Y)
    # End forward test

    # Backward test
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
    # End backward test
