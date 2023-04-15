"""
This module tests the linear layer module.
"""
import numpy as np
from numpy.typing import NDArray
import pytest

from src.activation_functions import NoActivation, ReLU
from src.cross_entropy_loss import CrossEntropyLoss
from src.linear import Linear


# pylint: disable=protected-access, invalid-name, too-many-public-methods
# pylint: disable=unused-argument
# pyright: reportGeneralTypeIssues=false
class TestLinear:
    """
    Linear layer class tester.
    """

    # region Fixtures
    @pytest.fixture
    def activation(self) -> None:
        """
        Default activation to None.
        """
        return None

    @pytest.fixture
    def layer(self, activation) -> Linear:
        """
        Create a linear layer with 3 input channels, 2 output channels.
        """
        layer = Linear(3, 2)

        params = {
            "weight": np.arange(1, 7, dtype=float).reshape(2, 3),
            "bias": np.arange(1, 3, dtype=float)
        }
        if activation:
            params["activation_function"] = activation
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
    def small_grad(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Get the gradients for the small input and layers with no activation or
        ReLU.

        The gradients are with respect to the output, input, weight and bias
        respectively.
        """
        return (
            np.ones((1, 2)),
            np.array([[5, 7, 9]]),
            np.array([[1, 2, 3], [1, 2, 3]]),
            np.array([1, 1])
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
    def large_grad(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Get the gradients for the large input and layers with no activation or
        ReLU.

        The gradients are with respect to the output, input, weight and bias
        respectively.
        """
        return (
            np.ones((10, 2)),
            np.array([
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.]
            ]),
            np.array([[145., 155., 165.], [145., 155., 165.]]),
            np.array([10, 10])
        )

    @pytest.fixture
    def large_with_negatives(self, activation) -> tuple[NDArray, NDArray]:
        """
        Create a large input (10, 3) and output (10, 2) pair that contains
        negative numbers.
        """
        output = np.array([
            [-51., -131.],
            [-33.,  -86.],
            [-15.,  -41.],
            [3.,    4.],
            [21.,   49.],
            [39.,   94.],
            [57.,  139.],
            [75.,  184.],
            [93.,  229.],
            [111.,  274.]
        ], dtype=float)
        if activation == "ReLU":
            output = np.array([
                [0., 0.],
                [0.,  0.],
                [0.,  0.],
                [3.,    4.],
                [21.,   49.],
                [39.,   94.],
                [57.,  139.],
                [75.,  184.],
                [93.,  229.],
                [111.,  274.]
            ], dtype=float)

        return (
            np.arange(-10, 20, dtype=float).reshape(10, 3),
            output
        )

    @pytest.fixture
    def large_with_negatives_grad(
        self,
        activation
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Get the gradients for the large with negatives input and layers with
        no activation or ReLU.

        The gradients are with respect to the output, input, weight and bias
        respectively.
        """
        X_grad = np.array([
            [5., 7., 9.],
            [5., 7., 9.],
            [5., 7., 9.],
            [5., 7., 9.],
            [5., 7., 9.],
            [5., 7., 9.],
            [5., 7., 9.],
            [5., 7., 9.],
            [5., 7., 9.],
            [5., 7., 9.]
        ])
        W_grad = np.array([[35., 45., 55.], [35., 45., 55.]])
        B_grad = np.array([10, 10])
        if activation == "ReLU":
            X_grad = np.array([
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.],
                [5., 7., 9.]
            ])
            W_grad = np.array([[56., 63., 70.], [56., 63., 70.]])
            B_grad = np.array([7, 7])

        return (
            np.ones((10, 2)),
            X_grad,
            W_grad,
            B_grad
        )
    # endregion fixtures

    # region Init tests
    def test_init(self):
        """
        Tests the layer init.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_)
        assert layer.weight.shape == (out_, in_)
        assert layer.bias.shape == (out_, )
        assert isinstance(layer.activation, NoActivation)

    def test_init_with_relu(self):
        """
        Tests the layer init with ReLU.
        """
        in_, out_ = 3, 2
        layer = Linear(in_, out_, activation=ReLU)
        assert layer.weight.shape == (out_, in_)
        assert layer.bias.shape == (out_, )
        assert isinstance(layer.activation, ReLU)

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
        assert layer.weight.shape == (out_, in_)
        assert layer.bias.shape == (out_, )

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
        assert np.array_equal(layer.weight, np.arange(1, 7).reshape(2, 3))
        assert np.array_equal(layer.bias, np.arange(1, 3))

    def test_init_with_mismatched_weight_shape(self):
        """
        Test the layer init with a weight init function that
        provides an mismatched shape.
        """
        in_, out_ = 3, 2

        def weight_init(**_) -> NDArray:
            return np.ones((10, 10))

        with pytest.raises(ValueError):
            Linear(in_, out_, weight_init=weight_init)

    def test_init_with_mismatched_bias_shape(self):
        """
        Test the layer init with a bias init function that
        provides an mismatched shape.
        """
        in_, out_ = 3, 2

        def bias_init(**_) -> NDArray:
            return np.ones(10)

        with pytest.raises(ValueError):
            Linear(in_, out_, bias_init=bias_init)

    # endregion Init tests

    # region Properties tests
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

    @pytest.mark.parametrize("eval_", [
        1, 1.123, [], {}, "True"
    ])
    def test_set_eval_with_invalid_type(self, layer, eval_):
        """
        Test setting eval with an invalid type.
        """
        with pytest.raises(TypeError):
            layer.eval = eval_

    @pytest.mark.parametrize("activation", [None, "ReLU"])
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

    # region Weight tests
    @pytest.mark.parametrize("weight", [
        np.ones(shape=(2, 3)),
        np.zeros(shape=(2, 3)),
        np.array([[5.23, 231, 129], [616.1, -123, 4]])
    ])
    def test_set_weight(self, layer, weight):
        """
        Test setting the weight.
        """
        layer.weight = weight
        assert np.array_equal(layer.weight, weight)

    @pytest.mark.parametrize("weight", [
        1,
        1.23,
        "Test",
        [[1, 2, 3], [4, 5, 6]],
        {1, 2, 3},
        range(6)
    ])
    def test_set_weight_with_invalid_type(self, layer, weight):
        """
        Test setting the weight with an invalid type.
        """
        with pytest.raises(TypeError):
            layer.weight = weight

    @pytest.mark.parametrize("weight", [
        np.ones(shape=(3, 2)),
        np.zeros(shape=(6, )),
        np.array([[5.23, 231,], [129, 616.1], [-123, 4]])
    ])
    def test_set_weight_with_wrong_shape(self, layer, weight):
        """
        Test setting the weight with the wrong shape.
        """
        with pytest.raises(ValueError):
            layer.weight = weight
    # endregion Weight tests

    # region Bias tests
    @pytest.mark.parametrize("bias", [
        np.ones(shape=(2, )),
        np.zeros(shape=(2, )),
        np.array([616.1, -123])
    ])
    def test_set_bias(self, layer, bias):
        """
        Test setting the bias.
        """
        layer.bias = bias
        assert np.array_equal(layer.bias, bias)

    @pytest.mark.parametrize("bias", [
        1,
        1.23,
        "Test",
        [[1, 2, 3], [4, 5, 6]],
        {1, 2, 3},
        range(6)
    ])
    def test_set_bias_with_invalid_type(self, layer, bias):
        """
        Test setting the bias with an invalid type.
        """
        with pytest.raises(TypeError):
            layer.bias = bias

    @pytest.mark.parametrize("bias", [
        np.ones(shape=(2, 1)),
        np.zeros(shape=(1, 2)),
        np.array([[5.23, 231,], [129, 616.1], [-123, 4]])
    ])
    def test_set_bias_with_wrong_shape(self, layer, bias):
        """
        Test setting the bias with the wrong shape.
        """
        with pytest.raises(ValueError):
            layer.bias = bias
    # endregion Bias tests

    # region Activation function tests
    @pytest.mark.parametrize("activation_function", [
        NoActivation, ReLU
    ])
    def test_activation_function(self, layer, activation_function):
        """
        Test setting the activation function.
        """
        layer.activation = activation_function
        assert layer.activation == activation_function()

    @pytest.mark.parametrize("activation_function", [
        1,
        1.23,
        [],
        "Invalid",
        ReLU(),
        NoActivation(),
        CrossEntropyLoss
    ])
    def test_activation_function_with_invalid_type(
        self,
        layer,
        activation_function
    ):
        """
        Test setting the activation function with an invalid type.e
        """
        with pytest.raises(TypeError):
            layer.activation = activation_function
    # endregion Activation function tests
    # endregion Properties tests

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
        assert np.array_equal(layer.weight, weight)
        assert np.array_equal(layer.bias, bias)
        assert isinstance(layer.activation, ReLU)

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
        prev_weight = layer.weight
        prev_bias = layer.bias
        prev_activation = layer.activation

        layer.load_params(weight=weight)
        assert np.array_equal(layer.weight, weight)
        assert not np.array_equal(layer.weight, prev_weight)
        assert np.array_equal(layer.bias, prev_bias)
        assert layer.activation == prev_activation

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
        prev_weight = layer.weight
        prev_bias = layer.bias
        prev_activation = layer.activation

        layer.load_params(bias=bias)
        assert np.array_equal(layer.weight, prev_weight)
        assert np.array_equal(layer.bias, bias)
        assert not np.array_equal(layer.bias, prev_bias)
        assert layer.activation == prev_activation

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
        prev_weight = layer.weight
        prev_bias = layer.bias
        prev_activation = layer.activation
        layer.load_params(activation_function="ReLU")
        assert np.array_equal(layer.weight, prev_weight)
        assert np.array_equal(layer.bias, prev_bias)
        assert layer.activation != prev_activation
        assert isinstance(layer.activation, ReLU)

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

    @pytest.mark.parametrize("activation", ["NoActivation", "ReLU"])
    def test_from_dict(self, layer, activation):
        """
        Test the from_dict method.
        """
        attributes = layer.to_dict()
        new_layer = Linear.from_dict(attributes)
        assert np.array_equal(new_layer.weight, layer.weight)
        assert np.array_equal(new_layer.bias, layer.bias)
        assert isinstance(new_layer.activation, type(layer.activation))

    @pytest.mark.parametrize("activation", ["NoActivation", "ReLU"])
    def test_from_dict_with_invalid_class(self, layer, activation):
        """
        Test the from_dict method with an invalid class
        """
        attributes = layer.to_dict()
        attributes["class"] = "test"
        with pytest.raises(ValueError):
            Linear.from_dict(attributes)
    # endregion Load tests

    # region Save tests
    @pytest.mark.parametrize("activation", ["NoActivation", "ReLU"])
    def test_to_dict(self, layer, activation):
        """
        Tests the to_dict method.
        """
        assert layer.to_dict() == {
            "class": "Linear",
            "weight": [[1, 2, 3], [4, 5, 6]],
            "bias": [1, 2],
            "activation_function": activation
        }
    # endregion Save tests

    # region Forward pass tests
    @pytest.mark.parametrize("activation", [None, "ReLU"])
    @pytest.mark.parametrize("data", [
        "small", "large",  "large_with_negatives"
    ])
    def test_forward(self, layer, data, request):
        """
        Tests the forward pass for the linear layer.
        """
        X, Y_true = request.getfixturevalue(data)
        Y = layer(X)
        assert np.array_equal(Y, Y_true)
    # endregion forward pass tests

    # region Backward pass tests
    @pytest.mark.parametrize("activation", [None, "ReLU"])
    @pytest.mark.parametrize("data, grads", [
        ("small", "small_grad"),
        ("large", "large_grad"),
        ("large_with_negatives", "large_with_negatives_grad")
    ])
    def test_backward(self, layer, data, grads, request):
        """
        Test the backward pass for the linear layer.
        """
        X, _ = request.getfixturevalue(data)
        layer(X)
        grad, true_input_grad, true_weight_grad, true_bias_grad = \
            request.getfixturevalue(grads)

        input_grad, (weight_grad, bias_grad) = layer.backward(grad)
        assert np.array_equal(input_grad, true_input_grad)
        assert np.array_equal(weight_grad, true_weight_grad)
        assert np.array_equal(bias_grad, true_bias_grad)

    @pytest.mark.parametrize("activation", [None, "ReLU"])
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

    @pytest.mark.parametrize("activation", [None, "ReLU"])
    @pytest.mark.parametrize("data, grads", [
        ("small", "small_grad"),
        ("large", "large_grad"),
        ("large_with_negatives", "large_with_negatives_grad")
    ])
    def test_update(self, layer, data, grads, request):
        """
        Test parameter update.
        """
        X, _ = request.getfixturevalue(data)
        layer(X)
        grad, _, true_weight_grad, true_bias_grad = request.getfixturevalue(
            grads)
        learning_rate = 1e-4

        layer.backward(grad)
        assert np.array_equal(
            layer.weight,
            np.arange(1, 7, dtype=float).reshape(2, 3)
        )
        assert np.array_equal(
            layer.bias,
            np.arange(1, 3, dtype=float)
        )

        layer.update(grad, learning_rate)
        assert np.array_equal(
            layer.weight,
            np.arange(1, 7, dtype=float).reshape(2, 3)
            - learning_rate * true_weight_grad
        )
        assert np.array_equal(
            layer.bias,
            np.arange(1, 3, dtype=float) - learning_rate * true_bias_grad
        )
    # endregion backward tests

    # region Built-ins tests
    @pytest.mark.parametrize("layer, other, result", [
        # same
        (
            Linear(
                3, 2,
                weight_init=lambda size: np.ones(shape=size),
                bias_init=lambda size: np.ones(shape=size)
            ),
            Linear(
                3, 2,
                weight_init=lambda size: np.ones(shape=size),
                bias_init=lambda size: np.ones(shape=size)
            ),
            True
        ),
        # shape different
        (
            Linear(
                3, 2,
                weight_init=lambda size: np.ones(shape=size),
                bias_init=lambda size: np.ones(shape=size)
            ),
            Linear(
                2, 3,
                weight_init=lambda size: np.ones(shape=size),
                bias_init=lambda size: np.ones(shape=size)
            ),
            False
        ),
        # weight different
        (
            Linear(
                3, 2,
                weight_init=lambda size: np.ones(shape=size),
                bias_init=lambda size: np.ones(shape=size)
            ),
            Linear(
                3, 2,
                weight_init=lambda size: np.zeros(shape=size),
                bias_init=lambda size: np.ones(shape=size)
            ),
            False
        ),
        # bias different
        (
            Linear(
                3, 2,
                weight_init=lambda size: np.ones(shape=size),
                bias_init=lambda size: np.ones(shape=size)
            ),
            Linear(
                3, 2,
                weight_init=lambda size: np.ones(shape=size),
                bias_init=lambda size: np.zeros(shape=size)
            ),
            False
        ),
        # values different
        (
            Linear(3, 2),
            Linear(3, 2),
            False
        ),
        # activation different
        (
            Linear(
                3, 2,
                weight_init=lambda size: np.ones(shape=size),
                bias_init=lambda size: np.ones(shape=size),
                activation=NoActivation
            ),
            Linear(
                3, 2,
                weight_init=lambda size: np.ones(shape=size),
                bias_init=lambda size: np.ones(shape=size),
                activation=ReLU
            ),
            False
        ),
        # invalid types
        (Linear(3, 2), 1, False),
        (Linear(3, 2), 1.23, False),
        (Linear(3, 2), [], False),
        (Linear(3, 2), "TEST", False),
        (Linear(3, 2), {}, False),
    ])
    def test_dunder_eq(self, layer, other, result):
        """
        Tests the __eq__ method.
        """
        assert (layer == other) is result
    # endregion Built-ins tests
