"""
This module tests the linear layer module.
"""
import numpy as np
from numpy.typing import NDArray
import pytest

from neural_net.linear import Linear
from neural_net.activation_functions import NoActivation, ReLU


# pylint: disable=protected-access, invalid-name, too-many-public-methods
# pyright: reportGeneralTypeIssues=false
class TestLinear:
    """
    Linear layer class tester.
    """

    # region Fixtures
    @pytest.fixture
    def layer(self, request) -> Linear:
        """
        Create a linear layer with 3 input channels, 2 output channels.
        """
        layer = Linear(3, 2)

        params = {
            "weight": np.arange(1, 7, dtype=float).reshape(2, 3),
            "bias": np.arange(1, 3, dtype=float)
        }
        if hasattr(request, "param"):
            params["activation_function"] = request.param
        layer.load_params(**params)

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
    # endregion fixtures

    # region Init tests
    def test_init(self):
        """
        Tests the layer init.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_)
        assert layer._weight.shape == (out_, in_)
        assert layer._bias.shape == (out_, )
        assert isinstance(layer._activation, NoActivation)

    def test_init_with_relu(self):
        """
        Tests the layer init with ReLU.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_, activation=ReLU)
        assert layer._weight.shape == (out_, in_)
        assert layer._bias.shape == (out_, )
        assert isinstance(layer._activation, ReLU)

    def test_init_with_uniform_distribution(self):
        """
        Tests the layer init with uniform parameter distributions.

        Given that it is difficult to test that the the sampling is correct,
        this will only test that it initialises without a runtime error.
        """
        in_, out_ = 3, 2
        layer = Linear(
            in_,
            out_,
            weight_init=np.random.uniform,
            bias_init=np.random.uniform
        )
        assert layer._weight.shape == (out_, in_)
        assert layer._bias.shape == (out_, )

    def test_init_with_custom_param_init(self):
        """
        Tests the layer init with a custom parameter init.
        """
        in_, out_ = 3, 2

        def weight_init(*, size: tuple[int, int]) -> NDArray:
            return np.arange(1, 1 + size[0] * size[1]).reshape(size)

        def bias_init(*, size: int) -> NDArray:
            return np.arange(1, 1 + size)

        layer = Linear(
            in_,
            out_,
            weight_init=weight_init,
            bias_init=bias_init
        )
        assert np.array_equal(layer._weight, np.arange(1, 7).reshape(2, 3))
        assert np.array_equal(layer._bias, np.arange(1, 3))
    # endregion Init tests

    # region Evaluation mode tests
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

    @pytest.mark.parametrize("layer", [None, "ReLU"], indirect=["layer"])
    @pytest.mark.parametrize(
        "data",
        ["small", "large", "large_with_negatives"]
    )
    def test_backward_call_with_eval(self, layer, data, request):
        """
        Test attempting to call backward with layer set to evaluation mode.
        """
        layer.eval = True
        X, Y_true = request.getfixturevalue(data)
        grad = np.ones_like(Y_true)
        with pytest.raises(RuntimeError):
            layer.backward(grad)
        layer(X)
        layer.eval = False
        with pytest.raises(RuntimeError):
            layer.backward(grad)
        layer(X)
        layer.backward(grad)

    # endregion Evaluation mode tests

    # region Load tests
    def test_load_params_all(self):
        """
        Tests the load parameter method for all parameters.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_)
        weight = np.arange(1, 7, dtype=float).reshape(2, 3)
        bias = np.arange(1, 3, dtype=float)
        layer.load_params(weight=weight, bias=bias, activation_function="ReLU")
        assert np.array_equal(layer._weight, weight)
        assert np.array_equal(layer._bias, bias)
        assert isinstance(layer._activation, ReLU)

    @pytest.mark.parametrize("weight", [
        np.arange(1, 7, dtype=float).reshape(2, 3),
        [[1, 2, 3], [4, 5, 6]]
    ])
    def test_load_params_weight(self, weight):
        """
        Tests the load parameter method for the weight parameter with a list.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_)
        prev_weight = layer._weight
        prev_bias = layer._bias
        prev_activation = layer._activation

        layer.load_params(weight=weight)
        assert np.array_equal(layer._weight, weight)
        assert not np.array_equal(layer._weight, prev_weight)
        assert np.array_equal(layer._bias, prev_bias)
        assert layer._activation == prev_activation

    @pytest.mark.parametrize("weight", [
        np.arange(1, 7, dtype=float).reshape(3, 2),
        np.arange(1, 7, dtype=float),
        [1],
        [[1, 2, "a"]]
    ])
    def test_load_params_weight_error(self, weight):
        """
        Tests the load parameter method for the weight parameter with
        invalid values.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_)
        with pytest.raises(ValueError):
            layer.load_params(weight=weight)

    @pytest.mark.parametrize("bias", [np.arange(1, 3, dtype=float), [1, 2]])
    def test_load_params_bias(self, bias):
        """
        Tests the load parameter method for the bias parameter.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_)
        prev_weight = layer._weight
        prev_bias = layer._bias
        prev_activation = layer._activation

        layer.load_params(bias=bias)
        assert np.array_equal(layer._weight, prev_weight)
        assert np.array_equal(layer._bias, bias)
        assert not np.array_equal(layer._bias, prev_bias)
        assert layer._activation == prev_activation

    @pytest.mark.parametrize("bias", [
        np.arange(1, 3, dtype=float).reshape(2, 1),
        [[1, 2]],
        [1],
        [1, 2, "a"]
    ])
    def test_load_params_bias_shape_error(self, bias):
        """
        Tests the load parameter method for the bias parameter with
        invalid data.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_)
        with pytest.raises(ValueError):
            layer.load_params(bias=bias)

    def test_load_params_activation(self):
        """
        Tests the load parameter method for the activation function.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_)
        prev_weight = layer._weight
        prev_bias = layer._bias
        prev_activation = layer._activation
        layer.load_params(activation_function="ReLU")
        assert np.array_equal(layer._weight, prev_weight)
        assert np.array_equal(layer._bias, prev_bias)
        assert layer._activation != prev_activation
        assert isinstance(layer._activation, ReLU)

    @pytest.mark.parametrize("activation_function, exception", [
        ("ggopn@03m", ValueError),
        (ReLU, TypeError),
        (ReLU(), TypeError),
        (1, TypeError)
    ])
    def test_load_params_activation_error(
        self,
        activation_function,
        exception
    ):
        """
        Tests the load parameter method for the activation function
        with invalid inputs.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_)
        with pytest.raises(exception):
            layer.load_params(activation_function=activation_function)
    # endregion Load tests

    # region Save tests
    @pytest.mark.parametrize(
        "layer, activation",
        [
            (None, "NoActivation"),
            ("ReLU", "ReLU")
        ],
        indirect=["layer"]
    )
    def test_to_dict(self, layer, activation, request):
        """
        Tests the to dict method.
        """
        assert layer.to_dict() == {
            "weight": [[1, 2, 3], [4, 5, 6]],
            "bias": [1, 2],
            "activation": activation
        }
    # endregion Save tests

    # region Forward pass tests
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

    @pytest.mark.parametrize("layer", ["ReLU"], indirect=["layer"])
    @pytest.mark.parametrize("data", [
        "small", "large", "large_with_negatives"
    ])
    def test_forward_2(self, layer, data, request):
        """
        Tests the forward pass for the linear layer with ReLU.
        """
        X, Y_true = request.getfixturevalue(data)
        Y = layer(X)
        Y_true = Y_true * (Y_true > 0)
        assert np.array_equal(Y, Y_true)
    # endregion forward pass tests

    # region Backward pass tests
    @pytest.mark.parametrize("data", [
        "small", "large", "large_with_negatives"
    ])
    def test_backward_1(self, layer, data, request):
        """
        Test the backward pass for the linear layer.
        """
        X, Y_true = request.getfixturevalue(data)
        layer(X)
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

    @pytest.mark.parametrize("layer", ["ReLU"], indirect=["layer"])
    @pytest.mark.parametrize("data", [
        "small", "large", "large_with_negatives"
    ])
    def test_backward_2(self, layer, data, request):
        """
        Test the backward pass for the linear layer with ReLU and multiple
        inputs.
        """
        X, Y_true = request.getfixturevalue(data)
        layer(X)
        grad = np.ones_like(Y_true)
        true_grad = Y_true > 0

        input_grad, (weight_grad, bias_grad) = layer.backward(grad)
        assert np.array_equal(
            input_grad,
            true_grad @ layer._weight
        )
        assert np.array_equal(
            weight_grad,
            true_grad.T @ X
        )
        assert np.array_equal(
            bias_grad,
            true_grad.sum(axis=0)
        )

    @pytest.mark.parametrize("layer", [None, "ReLU"], indirect=["layer"])
    @pytest.mark.parametrize(
        "data",
        ["small", "large", "large_with_negatives"],
    )
    def test_backward_call_before_forward(self, layer, data, request):
        """
        Test attempting to call backward before forward.
        """
        X, Y_true = request.getfixturevalue(data)
        grad = np.ones_like(Y_true)
        with pytest.raises(RuntimeError):
            layer.backward(grad)
        layer(X)
        layer.backward(grad)

    @pytest.mark.parametrize("layer", [None, "ReLU"], indirect=["layer"])
    @pytest.mark.parametrize(
        "data",
        ["small", "large", "large_with_negatives"]
    )
    def test_update(self, layer, data, request):
        """
        Test parameter update.
        """
        X, Y_true = request.getfixturevalue(data)
        layer(X)
        grad = np.ones_like(Y_true)
        learning_rate = 1e-4
        _, (weight_grad, bias_grad) = layer.backward(grad)

        assert np.array_equal(
            layer._weight,
            np.arange(1, 7, dtype=float).reshape(2, 3)
        )
        assert np.array_equal(
            layer._bias,
            np.arange(1, 3, dtype=float)
        )
        layer.update(grad, learning_rate)
        assert np.array_equal(
            layer._weight,
            np.arange(1, 7, dtype=float).reshape(
                2, 3) - learning_rate * weight_grad
        )
        assert np.array_equal(
            layer._bias,
            np.arange(1, 3, dtype=float) - learning_rate * bias_grad
        )
    # endregion backward tests
