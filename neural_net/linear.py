"""
This module contains the linear layer.
"""
from __future__ import annotations
from typing import Type

import numpy as np
from numpy.typing import NDArray

from . import activation_functions as act


class Linear:
    """
    A linear layer.
    """
    idCounter = 0

    # Setup
    def __init__(
        self,
        in_: int,
        out_: int,
        *,
        activation: Type[act.ActivationFunction] = act.NoActivation,
    ) -> None:
        self.name = f"Linear layer {Linear.idCounter}"
        Linear.idCounter += 1

        # Forward pass
        self._weight: NDArray = np.ndarray(
            (out_, in_),
            dtype=np.float32
        )
        self._bias: NDArray = np.ndarray(out_, dtype=np.float32)
        self._activation: act.ActivationFunction = activation()
        self._eval: bool = False

        # Backward pass
        self._input: NDArray | None = None

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

    def load_params(self, weight: NDArray, bias: NDArray) -> None:
        """
        Load parameters for the layer.

        Args:
            weight: The weight values
            bias: The bias values
        """
        self._weight = weight
        self._bias = bias
    # End setup

    # Forward pass
    def forward(self, input_) -> NDArray:
        """
        Perform the forward pass for the layer.

        Args:
            inputs: The inputs required to perform the forward pass.
        Returns:
            Result from the forward pass.
        """
        self._input = input_ if not self.eval else None
        output_: NDArray = input_ @ self._weight.T + self._bias
        return self._activation(output_)
    # End forward pass

    # Backward pass
    def backward(self, grad: NDArray) -> tuple[NDArray, tuple[NDArray, ...]]:
        """
        Perform the backward pass for the layer.

        Args:
            grad: The propagated gradients

        Returns:
            The gradient with respect to the input and 
            the gradients with respect to each parameter.
        """
        if self._input is None:
            raise ValueError(
                "forward must be called before backward."
                if not self.eval
                else "backward cannot be called on layers set in eval mode."
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
    # End backward pass

    # Built-ins
    def __call__(self, input_: NDArray) -> NDArray:
        return self.forward(input_)

    def __hash__(self) -> int:
        return hash(self.name)
    # End built-ins
