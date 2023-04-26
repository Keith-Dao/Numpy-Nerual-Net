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
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from tabulate import tabulate
from tqdm import tqdm

from src import (
    cross_entropy_loss,
    image_loader,
    linear,
    metrics,
    utils
)


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
            train_metrics: The training metrics to store or
                the training history metrics
            validation_metrics: The validation metrics to store or
                the validation history metrics
            classes: The classes used to train the model
        """
        self._eval = False
        self.layers = layers
        self.loss = loss
        self.total_epochs = kwargs.get("total_epochs", 0)

        # Metrics
        self.train_metrics = kwargs.get("train_metrics") or []
        self.validation_metrics = kwargs.get("validation_metrics") or []
        self.classes = kwargs.get("classes")

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

    # region Train metrics
    @property
    def train_metrics(self) -> dict[str, list[float]]:
        """
        The metrics to store when training the model.
        """
        return self._train_metrics

    @train_metrics.setter
    def train_metrics(self, train_metrics) -> None:
        utils.check_type(train_metrics, (dict, list), "train_metrics")

        if isinstance(train_metrics, dict):
            self._train_metrics = train_metrics
        else:
            self._train_metrics = Model.metrics_list_to_dict(train_metrics)

        if any(
            metric != "loss" and not hasattr(metrics, metric)
            for metric in self.train_metrics
        ):
            raise ValueError(
                "An invalid metric was provided to train_metrics."
            )
        if any(
            not isinstance(value, list)
            for value in self.train_metrics.values()
        ):
            raise ValueError(
                "All train metric histories must be a list."
            )
    # endregion Train metrics

    # region Validation metrics
    @property
    def validation_metrics(self) -> dict[str, list[float]]:
        """
        The metrics to store when validating the model.
        """
        return self._validation_metrics

    @validation_metrics.setter
    def validation_metrics(self, validation_metrics) -> None:
        utils.check_type(
            validation_metrics,
            (dict, list),
            "validation_metrics"
        )

        if isinstance(validation_metrics, dict):
            self._validation_metrics = validation_metrics
        else:
            self._validation_metrics = Model.metrics_list_to_dict(
                validation_metrics
            )

        if any(
            metric != "loss" and not hasattr(metrics, metric)
            for metric in self.validation_metrics
        ):
            raise ValueError(
                "An invalid metric was provided to validation_metrics."
            )
        if any(
            not isinstance(value, list)
            for value in self.validation_metrics.values()
        ):
            raise ValueError(
                "All validation metric histories must be a list."
            )
    # endregion Validation metrics
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
            - total_epochs -- the total number of epochs the model has trained
                for
            - train_metrics -- the history of the training metrics for the
                model
            - validation_metrics -- the history of the validation metrics for
                the model

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
            "total_epochs": self.total_epochs,
            "train_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
            "classes": self.classes
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
                f"File format {save_path.suffix} is not supported."
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

    def get_loss_with_confusion_matrix(
        self,
        input_: NDArray,
        confusion_matrix: NDArray,
        labels: list[int]
    ) -> float:
        """
        Perform the forward pass and store the predictions to the
        confusion matrix.

        Args:
            input_: The input values to the model
            confusion_matrix: The confusion matrix where the rows
                represent the predicted class and the columns
                represent the actual class
            labels: The ground truth labels for the inputs

        Returns:
            The loss of the forward pass.
        """
        logits = self(input_)
        metrics.add_to_confusion_matrix(
            confusion_matrix,
            utils.logits_to_prediction(logits),
            labels
        )
        return self.loss(logits, labels)

    def predict(self, input_: NDArray) -> list[str]:
        """
        Using the provided input, predict the classes.

        Args:
            input_: The input values to the model

        Returns:
            The predicted classes.
        """
        if self.classes is None:
            raise ValueError("Model is missing the classes.")

        logits = self(input_)
        return [
            self.classes[label]
            for label in utils.logits_to_prediction(logits)
        ]
    # endregion Forward pass

    # region Train

    def _train_step(
        self,
        data: NDArray,
        labels: list[int],
        learning_rate: float,
        confusion_matrix: NDArray
    ) -> float:
        """
        Perform a training step for one minibatch.

        Args:
            data: The minibatch data
            labels: The ground truth labels for the inputs
            learning_rate: The learning rate
            confusion_matrix: The confusion matrix where the rows
                represent the predicted class and the columns
                represent the actual class

        Returns:
            The output loss.
        """
        loss = self.get_loss_with_confusion_matrix(
            data,
            confusion_matrix,
            labels
        )
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
        self.classes = data_loader.classes
        num_classes = len(self.classes)
        for epoch in range(1, epochs + 1):
            # Training
            training_data = data_loader("train", batch_size=batch_size)
            confusion_matrix = metrics.get_new_confusion_matrix(num_classes)
            total_training_loss = sum(
                self._train_step(
                    data,
                    labels,
                    learning_rate,
                    confusion_matrix
                )
                for data, labels in tqdm(
                    training_data,
                    desc=f"Training epoch {epoch}/{epochs}"
                )
            )
            training_loss = total_training_loss / len(training_data)
            Model.store_metrics(
                self.train_metrics,
                confusion_matrix,
                training_loss
            )
            Model.print_metrics(
                self.train_metrics,
                self.classes
            )

            # Validation
            validation_data = data_loader("test", batch_size=batch_size)
            if len(validation_data) == 0:
                continue

            validation_loss, confusion_matrix = self.test(
                validation_data,
                num_classes,
                f"Validation epoch {epoch}/{epochs}"
            )
            Model.store_metrics(
                self.validation_metrics,
                confusion_matrix,
                validation_loss
            )
            Model.print_metrics(
                self.validation_metrics,
                self.classes
            )
        self.total_epochs += epochs
    # endregion Train

    # region Test
    def test(
        self,
        data_loader: image_loader.DatasetIterator,
        num_classes: int,
        tqdm_description: str = ""
    ) -> tuple[float, NDArray]:
        """
        Perform test on the model with the given data loader.

        Args:
            data_loader: Loader with data to test on
            num_classes: The number of classes in the dataset
            tqdm_description: The description to display on the progress bar

        Returns:
            The mean loss and confusion matrix of the test results.
        """
        self.eval = True
        confusion_matrix = metrics.get_new_confusion_matrix(
            num_classes
        )
        loss = sum(
            self.get_loss_with_confusion_matrix(
                data,
                confusion_matrix,
                labels
            )
            for data, labels in tqdm(
                data_loader,
                desc=tqdm_description
            )
        ) / len(data_loader)
        self.eval = False
        return loss, confusion_matrix
    # endregion Test

    # region Metrics
    @staticmethod
    def metrics_list_to_dict(metrics_: list[str]) -> dict[str, list]:
        """
        Convert a list of metrics to a dictionary to store the metric
        history.

        Args:
            metrics_: The metrics to store

        Returns:
            A dictionary to store the history of each metric that
            need to be stored.
        """
        return {
            metric: []
            for metric in metrics_
        }

    @staticmethod
    def store_metrics(
        metrics_: dict[str, list],
        confusion_matrix: NDArray,
        loss: float
    ) -> None:  # pragma: no cover
        """
        Store the metrics.

        Args:
            metrics_: The dictionary storing the metrics history
            confusion_matrix: The confusion matrix to use for the metrics
            loss: The loss
        """
        for metric in metrics_.keys():
            metrics_[metric].append(
                loss
                if metric == "loss"
                else getattr(metrics, metric)(confusion_matrix).tolist()
            )

    @staticmethod
    def print_metrics(
        metrics_: dict[str, list],
        classes: list[str]
    ) -> None:  # pragma: no cover
        """
        Print the tracked metrics.

        Args:
            metrics_: The dictionary storing the metrics history
            classes: The classes of the dataset
        """
        multiclass_headers, multiclass_data = ["Class"], [classes]
        singular_headers, singular_data = [], []

        for metric in metrics_:
            if metric in metrics.SINGLE_VALUE_METRICS:
                headers = singular_headers
                data = singular_data
            else:
                headers = multiclass_headers
                data = multiclass_data

            headers.append(" ".join(metric.split("_")).capitalize())
            data.append(metrics_[metric][-1])

        float_format = ".4f"
        if singular_headers:
            print(tabulate(
                [singular_data],
                headers=singular_headers,
                floatfmt=float_format
            ), end="\n\n")

        if len(multiclass_headers) > 1:
            multiclass_data = list(zip(*multiclass_data))
            print(tabulate(
                multiclass_data,
                headers=multiclass_headers,
                floatfmt=float_format
            ), end="\n\n")
    # endregion Metrics

    # region Visualisation
    def _plot_metric(
        self,
        dataset: str,
        metric: str,
        axis: plt.Axes
    ) -> None:  # pragma: no cover
        """
        Plot the metric.

        Args:
            dataset: The dataset type to plot
            metric: The metric to plot
            axis: The axis to plot the metric on
        """
        if self.classes is None:
            raise ValueError("Cannot plot metrics. Missing classes.")
        if metric not in metrics.SINGLE_VALUE_METRICS:
            raise ValueError(f"Plotting {metric} is not supported.")

        metrics_ = getattr(self, f"{dataset}_metrics")
        if metric not in metrics_:
            return

        history = metrics_[metric]
        if not history:
            return

        axis.plot(
            range(1, len(history) + 1),
            history,
            ".-",
            label=dataset.capitalize()
        )

    def _generate_history_graph(self, metric: str) -> None:  # pragma: no cover
        """
        Generates the model's history data.

        Args:
            metric: The metric to generate a history graph for
            classes: The classes being displayed
        """
        if metric not in metrics.SINGLE_VALUE_METRICS:
            return
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

        self._plot_metric("train", metric, axis)
        self._plot_metric("validation", metric, axis)

        axis.legend()
        axis.set_xlabel("Epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlim(0, self.total_epochs + 1)

        metric_name = " ".join(metric.split("_")).capitalize()
        axis.set_title(metric_name)
        axis.set_ylabel(metric_name)
        axis.grid(which="major", alpha=0.5)

    def display_history_graphs(self) -> None:  # pragma: no cover
        """
        Generates and displays the model's history graphs.

        Args:
            classes: The classes that are being displayed
        """
        visualizable_metrics = (
            (
                set(self.train_metrics.keys())
                | set(self.validation_metrics.keys())
            )
            & metrics.SINGLE_VALUE_METRICS
        )
        for metric in visualizable_metrics:
            self._generate_history_graph(metric)

        graphed_metrics = utils.join_with_different_last(
            visualizable_metrics,
            ", ",
            " and "
        )
        show_graph = input(
            "Would you like to view the history graphs for"
            f" {graphed_metrics}? [y/n]: "
        )
        if utils.is_yes(show_graph):
            plt.show()
        plt.close("all")
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
