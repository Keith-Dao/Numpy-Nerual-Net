"""
This module contains all the activation functions used in the neural network.
"""
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class ActivationFunction(ABC):
    """
    Abstract activation function.
    """

    # region Abstract methods
    @abstractmethod
    def forward(self, input_: NDArray) -> NDArray:
        """
        Perform the forward pass.

        Args:
            input: Input to the function

        Returns:
            Result of applying the activation function to the input.
        """
        self._input = input_

    @abstractmethod
    def backward(self) -> NDArray:
        """
        Perform the backward pass.

        Returns:
            The gradient with respect to the input.
        """
        if self._input is None:
            raise ValueError("backward cannot be called before forward.")
    # endregion Abstract methods

    # region Setup
    def __init__(self) -> None:
        super().__init__()
        self._input: NDArray | None = None
    # endregion Setup

    # region Built-ins
    def __call__(self, input_: NDArray) -> NDArray:
        return self.forward(input_)
    # endregion Built-ins


class NoActivation(ActivationFunction):
    """
    No activation function.
    """

    def forward(self, input_: NDArray) -> NDArray:
        super().forward(input_)
        return input_

    def backward(self) -> NDArray:
        super().backward()
        return np.ones_like(self._input)


class ReLU(ActivationFunction):
    """
    ReLU activation function.

    f(x) = x if x > 0 else 0
    """

    def forward(self, input_: NDArray) -> NDArray:
        super().forward(input_)
        return (input_ > 0).astype(input_.dtype) * input_

    def backward(self) -> NDArray:
        super().backward()
        return (self._input > 0).astype(self._input.dtype)  # type: ignore
