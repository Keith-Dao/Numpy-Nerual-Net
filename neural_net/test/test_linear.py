"""
This module tests the linear layer module.
"""
import numpy as np
from numpy.typing import NDArray
import pytest

from neural_net.linear import Linear
from neural_net.activation_functions import NoActivation, ReLU


# pylint: disable=protected-access, invalid-name
class TestLinear:
    """
    Linear layer class tester.
    """
    # Fixtures
    @pytest.fixture
    def layer(self) -> Linear:
        """
        Create a linear layer with 3 input channels, 2 output channels
        and no activation function.
        """
        layer = Linear(3, 2)
        layer.load_params(
            np.arange(1, 7, dtype=float).reshape(2, 3),
            np.arange(1, 3, dtype=float)
        )
        return layer

    @pytest.fixture
    def layer_with_relu(self) -> Linear:
        """
        Create a linear layer with 3 input channels, 2 output channels
        and a ReLU activation function.
        """
        layer = Linear(3, 2, activation=ReLU)
        layer.load_params(
            np.arange(1, 7, dtype=float).reshape(2, 3),
            np.arange(1, 3, dtype=float)
        )
        return layer

    @pytest.fixture
    def small(self) -> tuple[NDArray, NDArray]:
        """
        Create a small input (1, 3) and output (1, 2) pair.
        """
        return (
            np.arange(1, 4, dtype=float).reshape(1, 3),
            np.array([[15, 34]], dtype=float)
        )

    @pytest.fixture
    def large(self) -> tuple[NDArray, NDArray]:
        """
        Create a large input (10, 3) and output (10, 2) pair.
        """
        return (
            np.arange(1, 31, dtype=float).reshape(10, 3),
            np.array([[15.,  34.],
                      [33.,  79.],
                      [51., 124.],
                      [69., 169.],
                      [87., 214.],
                      [105., 259.],
                      [123., 304.],
                      [141., 349.],
                      [159., 394.],
                      [177., 439.]], dtype=float)
        )

    @pytest.fixture
    def large_with_negatives(self) -> tuple[NDArray, NDArray]:
        """
        Create a large input (10, 3) and output (10, 2) pair that contains
        negative numbers.
        """
        return (
            np.arange(-10, 20, dtype=float).reshape(10, 3),
            np.array([[-51., -131.],
                      [-33.,  -86.],
                      [-15.,  -41.],
                      [3.,    4.],
                      [21.,   49.],
                      [39.,   94.],
                      [57.,  139.],
                      [75.,  184.],
                      [93.,  229.],
                      [111.,  274.]], dtype=float)
        )
    # End fixtures

    # Init tests
    def test_init_1(self):
        """
        Tests the layer init.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_)
        assert layer._weight.shape == (out_, in_)
        assert layer._bias.shape == (out_, )
        assert isinstance(layer._activation, NoActivation)

    def test_init_2(self):
        """
        Tests the layer init with ReLU.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_, activation=ReLU)
        assert layer._weight.shape == (out_, in_)
        assert layer._bias.shape == (out_, )
        assert isinstance(layer._activation, ReLU)

    def test_load_params(self):
        """
        Tests the load parameter method.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_, activation=ReLU)
        weight = np.arange(1, 7, dtype=float).reshape(2, 3)
        bias = np.arange(1, 3, dtype=float)
        layer.load_params(weight, bias)
        assert np.array_equal(layer._weight, weight)
        assert np.array_equal(layer._bias, bias)

    def test_set_eval(self, layer):
        """
        Test setting the evaluation mode.
        """
        assert not layer.eval

        layer._input = 1
        layer.eval = True
        assert layer.eval
        assert layer._input is None

        layer._input = 1
        layer.eval = False
        assert not layer.eval
        assert layer._input is None
    # End init tests

    # Forward pass tests
    @pytest.mark.parametrize("data", [
        "small", "large", "large_with_negatives"
    ])
    def test_forward_1(self, layer, data, request):
        """
        Tests the forward pass for the linear layer.
        """
        X, Y_true = request.getfixturevalue(data)
        Y = layer(X)
        assert np.array_equal(Y, Y_true)

    @pytest.mark.parametrize("data", [
        "small", "large", "large_with_negatives"
    ])
    def test_forward_2(self, layer_with_relu, data, request):
        """
        Tests the forward pass for the linear layer with ReLU.
        """
        X, Y_true = request.getfixturevalue(data)
        Y = layer_with_relu(X)
        Y_true = Y_true * (Y_true > 0)
        assert np.array_equal(Y, Y_true)
    # End forward pass tests

    # Backward pass tests
    @pytest.mark.parametrize("data", [
        "small", "large", "large_with_negatives"
    ])
    def test_backward_1(self, layer, data, request):
        """
        Test the backward pass for the linear layer.
        """
        X, Y_true = request.getfixturevalue(data)
        _ = layer(X)
        grad = np.ones_like(Y_true)

        input_grad, (weight_grad, bias_grad) = layer.backward(grad)
        assert np.array_equal(
            input_grad,
            grad @ layer._weight
        )
        assert np.array_equal(
            weight_grad,
            grad.T @ X
        )
        assert np.array_equal(
            bias_grad,
            grad.sum(axis=0)
        )

    @pytest.mark.parametrize("data", [
        "small", "large", "large_with_negatives"
    ])
    def test_backward_2(self, layer_with_relu, data, request):
        """
        Test the backward pass for the linear layer with ReLU and multiple
        inputs.
        """
        X, Y_true = request.getfixturevalue(data)
        _ = layer_with_relu(X)
        grad = np.ones_like(Y_true)
        true_grad = Y_true > 0

        input_grad, (weight_grad, bias_grad) = layer_with_relu.backward(grad)
        assert np.array_equal(
            input_grad,
            true_grad @ layer_with_relu._weight
        )
        assert np.array_equal(
            weight_grad,
            true_grad.T @ X
        )
        assert np.array_equal(
            bias_grad,
            true_grad.sum(axis=0)
        )

    @pytest.mark.parametrize("layer_, data", [
        (layer, data)
        for layer in ("layer", "layer_with_relu")
        for data in ("small", "large", "large_with_negatives")
    ])
    def test_backward_call_before_forward(self, layer_, data, request):
        """
        Test attempting to call backward before forward.
        """
        layer = request.getfixturevalue(layer_)
        X, Y_true = request.getfixturevalue(data)
        grad = np.ones_like(Y_true)
        with pytest.raises(RuntimeError):
            layer.backward(grad)
        _ = layer(X)
        layer.backward(grad)

    @pytest.mark.parametrize("layer_, data", [
        (layer, data)
        for layer in ("layer", "layer_with_relu")
        for data in ("small", "large", "large_with_negatives")
    ])
    def test_backward_call_with_eval(self, layer_, data, request):
        """
        Test attempting to call backward with layer set to evaluation mode.
        """
        layer = request.getfixturevalue(layer_)
        layer.eval = True
        X, Y_true = request.getfixturevalue(data)
        grad = np.ones_like(Y_true)
        with pytest.raises(ValueError):
            layer.backward(grad)
        layer.eval = False
        _ = layer(X)
        layer.backward(grad)

    @pytest.mark.parametrize("layer_, data", [
        (layer, data)
        for layer in ("layer", "layer_with_relu")
        for data in ("small", "large", "large_with_negatives")
    ])
    def test_update(self, layer_, data, request):
        """
        Test parameter update.
        """
        X, Y_true = request.getfixturevalue(data)
        layer_ = request.getfixturevalue(layer_)
        _ = layer_(X)
        grad = np.ones_like(Y_true)
        learning_rate = 1e-4
        _, (weight_grad, bias_grad) = layer_.backward(grad)

        assert np.array_equal(
            layer_._weight,
            np.arange(1, 7, dtype=float).reshape(2, 3)
        )
        assert np.array_equal(
            layer_._bias,
            np.arange(1, 3, dtype=float)
        )
        layer_.update(grad, learning_rate)
        assert np.array_equal(
            layer_._weight,
            np.arange(1, 7, dtype=float).reshape(
                2, 3) - learning_rate * weight_grad
        )
        assert np.array_equal(
            layer_._bias,
            np.arange(1, 3, dtype=float) - learning_rate * bias_grad
        )
    # End backward tests

    # Built in tests
    def test_hash(self):
        """
        Test the hash function.
        """
        next_id = Linear.idCounter
        layer = Linear(1, 1)
        assert hash(layer) == hash(f"Linear layer {next_id}")

        another_layer = Linear(1, 1)
        assert hash(layer) != hash(another_layer)
    # End built in tests
