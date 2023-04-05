"""
This module contains the linear layer.
"""
from __future__ import annotations
from typing import Any, Callable, Type

import numpy as np
from numpy.typing import NDArray

from src import activation_functions as act


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
        weight_init: Callable[..., NDArray] = np.random.normal,
        bias_init: Callable[..., NDArray] = np.random.normal,
        activation: Type[act.ActivationFunction] = act.NoActivation,
    ) -> None:
        """
        Linear layer init.

        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            weight_init: Function to initialise the weight matrix
            bias_init: Function to initialise the bias vector
            activation: The activation function to use
        """
        # Forward pass
        self._weight: NDArray = weight_init(size=(out_channels, in_channels))
        if self._weight.shape != (out_channels, in_channels):
            raise ValueError(
                f"Invalid weight shape. Expected {(out_channels, in_channels)}"
                f", got {self._weight.shape}"
            )
        self._bias: NDArray = bias_init(size=out_channels)
        if self._bias.shape != (out_channels, ):
            raise ValueError(
                f"Invalid weight shape. Expected {(out_channels, )}"
                f", got {self._bias.shape}"
            )
        self._activation: act.ActivationFunction = activation()
        self._eval: bool = False

        # Backward pass
        self._input: NDArray | None = None
    # endregion Setup

    # region Evaluation mode
    @property
    def eval(self):
        """
        Layer evaluation mode.
        """
        return self._eval

    @eval.setter
    def eval(self, eval_: bool) -> None:
        if eval_ != self._eval:
            self._input = None
        self._eval = eval_
    # endregion Evaluation mode

    # region Load
    def _load_weight(self, weight: list[list[float]] | NDArray) -> None:
        """
        Loads weight for the layer.

        Args:
            weight: The weight values
        """
        new_weight = np.array(weight)
        if new_weight.shape != self._weight.shape:
            raise ValueError(
                f"The new weight has a shape of ${new_weight.shape},"
                f" expected {self._weight.shape}."
            )
        self._weight = new_weight

    def _load_bias(self, bias: list[float] | NDArray) -> None:
        """
        Loads bias for the layer.

        Args:
            bias: The bias values
        """
        new_bias = np.array(bias)
        if new_bias.shape != self._bias.shape:
            raise ValueError(
                f"The new bias has a shape of ${new_bias.shape},"
                f" expected {self._bias.shape}."
            )
        self._bias = new_bias

    def _load_activation(self, activation_function: str) -> None:
        """
        Loads the activation function for the layer.

        Args:
            activation_function: The name of the activation function class
        """
        if not isinstance(activation_function, str):
            raise TypeError(
                f"activation_function is of type"
                f" {type(activation_function).__name__}, expected str."
            )
        try:
            self._activation = getattr(act, activation_function)()
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
            self._load_weight(weight)

        if bias is not None:
            self._load_bias(bias)

        if activation_function is not None:
            self._load_activation(activation_function)
    # endregion Load

    # region Save
    def to_dict(self) -> dict[str, Any]:
        """
        Get all relevant attributes in a serialisable format.

        Attributes includes:
            - weight -- weights as a two-dimensional list
            - bias -- bias as a list
            - activation -- name of the activation function as a string

        Returns:
            Attributes listed above as a dictionary.
        """
        return {
            "class": type(self).__name__,
            "weight": self._weight.tolist(),
            "bias": self._bias.tolist(),
            "activation": type(self._activation).__name__
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
        output_: NDArray = input_ @ self._weight.T + self._bias
        return self._activation(output_)
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
        grad = grad * self._activation.backward()
        weight_grad = grad.T @ self._input  # dE/dw
        bias_grad = grad.sum(axis=0)  # dE/dB
        input_grad = grad @ self._weight  # dE/dX

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
        self._weight -= learning_rate * weight_grad
        self._bias -= learning_rate * bias_grad
        return input_grad
    # endregion Backward pass

    # region Built-ins
    def __call__(self, input_: NDArray) -> NDArray:
        return self.forward(input_)
    # endregion built-ins
