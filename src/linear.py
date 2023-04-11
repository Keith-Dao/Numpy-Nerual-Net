"""
This module contains the linear layer.
"""
from __future__ import annotations
from typing import Any, Callable, Type

import numpy as np
from numpy.typing import NDArray

from src import activation_functions as act, utils


# pylint: disable=too-many-instance-attributes
class Linear:
    """
    A linear layer.
    """

    # region Setup
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        weight_init: Callable[..., NDArray] | None = None,
        bias_init: Callable[..., NDArray] | None = None,
        activation: Type[act.ActivationFunction] = act.NoActivation,
    ) -> None:
        """
        Linear layer init.

        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            weight_init: Function to initialise the weight matrix that takes
                         the parameter size
            bias_init: Function to initialise the bias vector that takes
                       the parameter size
            activation: The activation function class to use
        """
        # Forward pass
        self.in_channels = in_channels
        self.out_channels = out_channels

        distribution_limit = np.sqrt(1 / in_channels)
        self._weight: NDArray = np.zeros(shape=(out_channels, in_channels))
        self.weight = weight_init(size=(out_channels, in_channels)) \
            if weight_init \
            else np.random.uniform(
                distribution_limit,
                -distribution_limit,
                size=(out_channels, in_channels))
        self._bias: NDArray = np.zeros(shape=(out_channels, ))
        self.bias = bias_init(size=out_channels) \
            if bias_init \
            else np.random.uniform(
                distribution_limit,
                -distribution_limit,
                size=out_channels)
        self.activation = activation
        self._eval: bool = False

        # Backward pass
        self._input: NDArray | None = None
    # endregion Setup

    # region Properties
    # region Evaluation mode
    @property
    def eval(self):
        """
        Layer's evaluation mode.
        """
        return self._eval

    @eval.setter
    def eval(self, eval_: bool) -> None:
        utils.check_type(eval_, bool, "eval")

        if eval_ != self._eval:
            self._input = None
        self._eval = eval_
    # endregion Evaluation mode

    # region Weights
    @property
    def weight(self) -> NDArray:
        """
        Layer's weight matrix.
        """
        return self._weight

    @weight.setter
    def weight(self, new_weight: NDArray) -> None:
        utils.check_type(new_weight, np.ndarray, "weight")

        if self.weight.shape != new_weight.shape:
            raise ValueError(
                f"Invalid shape for new weight. Expected {self._weight.shape},"
                f" got {new_weight.shape}."
            )

        self._weight = new_weight
    # endregion Weights

    # region Bias
    @property
    def bias(self) -> NDArray:
        """
        Layer's bias vector.
        """
        return self._bias

    @bias.setter
    def bias(self, new_bias: NDArray) -> None:
        utils.check_type(new_bias, np.ndarray, "bias")

        if self._bias.shape != new_bias.shape:
            raise ValueError(
                f"Invalid weight shape. Expected {self._bias.shape}"
                f", got {new_bias.shape}."
            )

        self._bias = new_bias
    # endregion Bias

    # region Activation function
    @property
    def activation(self) -> act.ActivationFunction:
        """
        Layer's activation function.
        """
        return self._activation

    @activation.setter
    def activation(self, new_activation: Type[act.ActivationFunction]) -> None:
        # Do not allow an actual activation function instance to prevent
        # another layer from using the same instance.
        if not isinstance(new_activation, type):
            raise TypeError(
                "Invalid type for activation. Expected"
                " type[ActivationFunction],"
                f" got {type(new_activation).__name__}."
            )

        activation: act.ActivationFunction = new_activation()
        utils.check_type(activation, act.ActivationFunction, "activation")

        self._activation = activation
    # endregion Activation function
    # endregion Properties

    # region Load
    def _load_activation(self, activation_function: str) -> None:
        """
        Loads the activation function for the layer.

        Args:
            activation_function: The name of the activation function class
        """
        utils.check_type(activation_function, str, "activation_function")
        try:
            self.activation = getattr(act, activation_function)
        except AttributeError as exc:
            raise ValueError(
                f"{activation_function} is not a valid activation function."
            ) from exc

    def load_params(
        self,
        *,
        weight: NDArray | list[list[float]] | None = None,
        bias: NDArray | list[float] | None = None,
        activation_function: str | None = None
    ) -> None:
        """
        Load parameters for the layer.

        Args:
            weight: The weight values
            bias: The bias values
        """
        if weight is not None:
            self.weight = np.array(weight)

        if bias is not None:
            self.bias = np.array(bias)

        if activation_function is not None:
            self._load_activation(activation_function)

    @classmethod
    def from_dict(cls, attributes: dict[str, Any]) -> Linear:
        """
        Create a linear instance from an attributes dictionary.

        Args:
            attributes: The attributes of the linear instance

        Returns:
            A linear instance with the provided attributes.
        """
        if cls.__name__ != attributes["class"]:
            raise ValueError(
                f"Invalid class value in attributes. Expected {cls.__name__},"
                f" got {attributes['class']}."
            )

        out_channels, in_channels = np.array(attributes["weight"]).shape
        linear = cls(in_channels, out_channels)
        linear.load_params(**{
            key: value for key, value in attributes.items() if key != "class"
        })
        return linear

    # endregion Load

    # region Save
    def to_dict(self) -> dict[str, Any]:
        """
        Get all relevant attributes in a serialisable format.

        Attributes includes:
            - weight -- weights as a two-dimensional list
            - bias -- bias as a list
            - activation_function -- name of the activation function as a
                                     string

        Returns:
            Attributes listed above as a dictionary.
        """
        return {
            "class": type(self).__name__,
            "weight": self.weight.tolist(),
            "bias": self.bias.tolist(),
            "activation_function": type(self.activation).__name__
        }
    # endregion Save

    # region Forward pass
    def forward(self, input_) -> NDArray:
        """
        Perform the forward pass for the layer.

        Args:
            inputs: The inputs required to perform the forward pass

        Returns:
            Result from the forward pass.
        """
        self._input = input_ if not self.eval else None
        output_: NDArray = input_ @ self.weight.T + self.bias
        return self.activation(output_)
    # endregion Forward pass

    # region Backward pass
    def backward(self, grad: NDArray) -> tuple[NDArray, tuple[NDArray, ...]]:
        """
        Perform the backward pass for the layer.

        Args:
            grad: The propagated gradients

        Returns:
            The gradient with respect to the input and
            the gradients with respect to each parameter.
        """
        if self.eval:
            raise RuntimeError(
                "Backward pass is not available for layers set in evaluation"
                " mode."
            )

        if self._input is None:
            raise RuntimeError(
                "Backward pass is not available for layers that have not been"
                " through the forward pass with evaluation mode turned off."
            )

        # Calculate grad
        grad = grad * self.activation.backward()
        weight_grad = grad.T @ self._input  # dE/dw
        bias_grad = grad.sum(axis=0)  # dE/dB
        input_grad = grad @ self.weight  # dE/dX

        return (input_grad), (weight_grad, bias_grad)

    def update(self, grad: NDArray, learning_rate: float) -> NDArray:
        """
        Update the parameters of the layer.

        Args:
            grad: The propagated gradients
            learning_rate: The learning rate

        Returns:
            The gradient with respect to the input.
        """
        input_grad, (weight_grad, bias_grad) = self.backward(grad)
        self.weight -= learning_rate * weight_grad
        self.bias -= learning_rate * bias_grad
        return input_grad
    # endregion Backward pass

    # region Built-ins
    def __call__(self, input_: NDArray) -> NDArray:
        return self.forward(input_)

    def __eq__(self, other: object) -> bool:
        """
        Checks if another is equal to this.

        Args:
            other: Object to compare with

        Returns:
            True if all attributes are equal, otherwise False.
        """
        return (
            isinstance(other, type(self))
            and np.array_equal(self.weight, other.weight)
            and np.array_equal(self.bias, other.bias)
            and self.activation == other.activation
        )
    # endregion built-ins
