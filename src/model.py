"""
This module contains the model class.
"""
from __future__ import annotations
import json
import pathlib
import pickle
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src import cross_entropy_loss, image_loader, linear, utils


# pylint: disable=too-many-instance-attributes
class Model:
    """
    The neural network model.
    """
    SAVE_METHODS: dict[str, tuple[ModuleType, bool]] = {
        # file extension : (module, is binary file)
        ".pkl": (pickle, True),
        ".json": (json, False)
    }

    def __init__(
        self,
        layers: list[linear.Linear],
        loss: cross_entropy_loss.CrossEntropyLoss,
        **kwargs
    ) -> None:
        """
        Model init.

        Args:
            layers: List of sequential layers in the neural network
            loss: The loss used to update this model

        Keyword args:
            total_epochs: The total epochs of the model
            train_history: The training loss history
            validation_history: The validation loss history
        """
        self._eval = False
        self.layers = layers
        self.loss = loss
        self.total_epochs = kwargs.get("total_epochs", 0)
        self.train_history = kwargs.get("train_history", [])
        self.validation_history = kwargs.get("validation_history", [])

    # region Static methods
    @staticmethod
    def calculate_mean_epoch_loss(
        history: list[float],
        minibatches: int
    ) -> float:
        """
        Calculate the mean loss for the previous epoch.

        Args:
            history: The loss values for the computed minibatches
            minibatches: The number of minibatches in an epoch

        Returns:
            The mean epoch loss for the previous epoch.
        """
        return sum(history[-minibatches:]) / minibatches
    # endregion Static methods

    # region Properties
    # region Evaluation mode
    @property
    def eval(self) -> bool:
        """
        Model's evaluation mode.
        """
        return self._eval

    @eval.setter
    def eval(self, eval_: bool) -> None:
        utils.check_type(eval_, bool, "eval")
        if self._eval == eval_:
            return

        for layer in self.layers:
            layer.eval = eval_
        self._eval = eval_
    # endregion Evaluation mode

    # region Layers
    @property
    def layers(self) -> list[linear.Linear]:
        """
        Model's layers.
        """
        return self._layers

    @layers.setter
    def layers(self, layers: list[linear.Linear]) -> None:
        utils.check_type(layers, list, "layers")
        if not layers:
            raise ValueError("layers cannot be an empty list.")

        if any(not isinstance(layer, linear.Linear) for layer in layers):
            raise TypeError(
                "Invalid type for layers. Expected all list elements to be"
                " Linear."
            )

        self._layers = layers
    # endregion Layers

    # region Loss
    @property
    def loss(self) -> cross_entropy_loss.CrossEntropyLoss:
        """
        The model's loss.
        """
        return self._loss

    @loss.setter
    def loss(self, loss: cross_entropy_loss.CrossEntropyLoss) -> None:
        utils.check_type(loss, cross_entropy_loss.CrossEntropyLoss, "loss")

        self._loss = loss
    # endregion Loss

    # region Total epochs
    @property
    def total_epochs(self) -> int:
        """
        Total epochs the model has trained for.
        """
        return self._total_epochs

    @total_epochs.setter
    def total_epochs(self, epochs: int) -> None:
        utils.check_type(epochs, int, "total_epochs")
        if epochs < 0:
            raise ValueError("total_epochs must be > 0.")

        self._total_epochs = epochs
    # endregion Total epochs

    # region Train history
    @property
    def train_history(self) -> list[float]:
        """
        Model's training loss history.
        """
        return self._train_history

    @train_history.setter
    def train_history(self, train_history: list[float]) -> None:
        utils.check_type(train_history, list, "train_history")
        if any(not isinstance(loss, (float, int)) for loss in train_history):
            raise TypeError(
                "Invalid type for train_history. Expected all list elements to"
                " be a number."
            )

        self._train_history = train_history
    # endregion Train history

    # region Validation history
    @property
    def validation_history(self) -> list[float]:
        """
        Model's validation loss history.
        """
        return self._validation_history

    @validation_history.setter
    def validation_history(self, validation_history: list[float]) -> None:
        utils.check_type(validation_history, list, "validation_history")
        if any(
            not isinstance(loss, (float, int))
            for loss in validation_history
        ):
            raise TypeError(
                "Invalid type for validation_history. Expected all list"
                " elements to be a number."
            )

        self._validation_history = validation_history
    # endregion Validation history
    # endregion Properties

    # region Load
    @classmethod
    def from_dict(cls, attributes: dict[str, Any]) -> Model:
        """
        Create a model instance from an attributes dictionary.

        Args:
            attributes: The attributes of the model instance

        Returns:
            A model instance with the provided attributes.
        """
        if cls.__name__ != attributes["class"]:
            raise ValueError(
                f"Invalid class value in attributes. Expected {cls.__name__},"
                f" got {attributes['class']}."
            )

        layers = [
            getattr(linear, layer_attributes["class"])
            .from_dict(layer_attributes)
            for layer_attributes in attributes["layers"]
        ]
        loss = getattr(cross_entropy_loss, attributes["loss"]["class"]) \
            .from_dict(attributes["loss"])
        return cls(layers, loss, **{
            key: val
            for key, val in attributes.items()
            if key not in {"layers", "loss"}
        })

    @classmethod
    def load(
        cls,
        file_path: str | pathlib.Path
    ) -> Model:
        """
        Load a model from the given file.

        NOTE: The format of the file would be inferred by the provided
        file extension i.e. .pkl would be a pickle file while .json
        would be a JSON file.

        Args:
            file_path: Path of the file to load from

        Returns:
            A new model object with the attributes specified in the file.
        """
        utils.check_type(file_path, (str, pathlib.Path), "file_path")
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)

        if file_path.suffix not in Model.SAVE_METHODS:
            raise ValueError(
                f"File format {file_path.suffix} not supported."
                f" Select from {' or '.join(Model.SAVE_METHODS.keys())}."
            )

        module, is_binary = Model.SAVE_METHODS[file_path.suffix]
        write_mode = "rb" if is_binary else "r"
        encoding = None if is_binary else "UTF-8"
        with open(file_path, write_mode, encoding=encoding) as load_file:
            attributes = module.load(load_file)
        return cls.from_dict(attributes)
    # endregion Load

    # region Save
    def to_dict(self) -> dict:
        """
        Get all the relevant attributes in a serialisable format.

        Attributes:
            - layers -- list of the serialised layers in sequential order
            - loss -- the loss function for the model
            - epochs -- the total number of epochs the model has trained for
            - train_history -- the training loss history for the model
            - validation_history -- the validation loss history for the model

        Returns:
            Attributes listed above as a dictionary.
        """
        return {
            "class": type(self).__name__,
            "layers": [
                layer.to_dict()
                for layer in self.layers
            ],
            "loss": self.loss.to_dict(),
            "epochs": self.total_epochs,
            "train_history": self.train_history,
            "validation_history": self.validation_history
        }

    def save(self, save_path: str | pathlib.Path):
        """
        Save the model attributes to the provided save path.

        NOTE: The format of the file would be inferred by the provided
        file extension i.e. .pkl would be a pickle file while .json
        would be a JSON file.

        Args:
            save_path: Path to save the model attributes.
        """
        utils.check_type(save_path, (str, pathlib.Path), "save_path")
        if isinstance(save_path, str):
            save_path = pathlib.Path(save_path)

        if save_path.suffix not in Model.SAVE_METHODS:
            raise ValueError(
                f"File format {save_path.suffix} not supported."
                f"Select from {' or '.join(Model.SAVE_METHODS.keys())}."
            )

        module, is_binary = Model.SAVE_METHODS[save_path.suffix]
        write_mode = "wb" if is_binary else "w"
        encoding = None if is_binary else "UTF-8"
        with open(save_path, write_mode, encoding=encoding) as save_file:
            module.dump(self.to_dict(), save_file)
    # endregion Save

    # region Forward pass
    def forward(self, input_: NDArray) -> NDArray:
        """
        Perform the forward pass.

        Args:
            input_: The input values to the model

        Returns:
            The output values of the model.
        """
        out = input_
        for layer in self.layers:
            out = layer(out)
        return out
    # endregion Forward pass

    # region Train
    def _train_step(
        self,
        data: NDArray,
        labels: list[int],
        learning_rate: float
    ) -> float:
        """
        Perform a training step for one minibatch.

        Args:
            data: The minibatch data
            labels: The ground truth labels for the inputs
            learning_rate: The learning rate

        Returns:
            The output loss.
        """
        output = self(data)
        loss = self.loss(output, labels)
        grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.update(grad, learning_rate)
        return loss

    def train(
        self,
        data_loader: image_loader.ImageLoader,
        learning_rate: float,
        batch_size: int,
        epochs: int
    ) -> None:
        """
        Train the model for the given number of epochs.

        Args:
            data_loader: The data loader
            learning_rate: The learning rate
            batch_size: The batch size
            epochs: The number of epochs to train for
        """
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}:")
            # Training
            training_data = data_loader("train", batch_size=batch_size)
            self.train_history.extend([
                self._train_step(
                    data,
                    labels,
                    learning_rate
                )
                for data, labels in tqdm(training_data)
            ])
            print(f"""Loss: {
                Model.calculate_mean_epoch_loss(
                    self.train_history,
                    len(training_data)
                )
            }""")

            # Validation
            validation_data = data_loader("test", batch_size=batch_size)
            if len(validation_data) == 0:
                continue

            self.eval = True
            self.validation_history.extend([
                self.loss(
                    self(data),
                    labels
                )
                for data, labels in validation_data
            ])
            print(
                f"""Validation Loss: {
                    Model.calculate_mean_epoch_loss(
                        self.validation_history,
                        len(validation_data)
                    )
                }"""
            )
            self.eval = False
        self.total_epochs += epochs
    # endregion Train

    # region Visualisation
    def generate_history_graph(self) -> None:  # pragma: no cover
        """
        Generates the model's history data.
        """
        train_points = np.linspace(
            0,
            self.total_epochs,
            len(self.train_history)
        )
        validation_points = np.linspace(
            0,
            self.total_epochs,
            len(self.validation_history)
        )

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.plot(
            train_points,
            self.train_history,
            "-c",
            label="Training Loss"
        )
        axis.plot(
            validation_points,
            self.validation_history,
            "-r",
            label="Validation Loss"
        )
        axis.legend(loc="upper right")
        axis.set_xlabel("Epoch")

    def display_history_graph(self) -> None:  # pragma: no cover
        """
        Generates and displays the model's history graph.
        """
        self.generate_history_graph()
        plt.show()
    # endregion Visualisation

    # region Built-ins

    def __call__(self, input_: NDArray) -> NDArray:
        """
        Perform the forward pass.

        Args:
            input_: The input values to the model

        Returns:
            The output values of the model.
        """
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
            and self.layers == other.layers
            and self.loss == other.loss
        )
    # endregion Built-ins
