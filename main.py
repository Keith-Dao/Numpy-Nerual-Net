"""
This module is the main driver file.
"""
import argparse
import pathlib
import sys
from typing import Any

import yaml

from src import (
    activation_functions as act,
    cross_entropy_loss as cel,
    image_loader,
    linear,
    model,
    utils
)

# region Constants
DEFAULT_CONFIG_PATH = "config.yaml"
# endregion Constants


# region Argparse
class DefaultConfigPathAction(argparse.Action):
    """
    Argparse action to print a message when defaulting.
    """

    def __call__(self, _, namespace, values, *__):
        if values is None:
            values = DEFAULT_CONFIG_PATH
            utils.print_warning(
                "Config file path was not provided. Defaulting to"
                f" {DEFAULT_CONFIG_PATH}."
            )
        setattr(namespace, self.dest, values)


def get_config_filepath() -> str:
    """
    Gets the config filepath.

    Args:
        default_config_path: The config path to use if none is provided
    Returns:
        The config filepath.
    """
    parser = argparse.ArgumentParser(
        prog="Neural Net",
        description="Neural network for classifying images of digits."
    )
    parser.add_argument(
        "config_file",
        help="Path to the config file.",
        nargs="?",
        action=DefaultConfigPathAction
    )

    return getattr(parser.parse_args(), "config_file")
# endregion Argparse


# region Parse config
def get_config() -> dict[str, Any]:
    """
    Get the config values from the config file.

    Returns:
        A dictionary with the parsed config files.
    """
    with open(
        pathlib.Path(get_config_filepath()),
        "r",
        encoding=sys.getdefaultencoding()
    ) as file:
        return yaml.safe_load(file)
# endregion Parse config


# region Load model
def get_model(config: dict[str, Any]) -> model.Model:
    """
    Loads the model using the file provided in the config,
    or use the default model is none is provided.

    Args:
        config: The configuration values from the config file

    Returns:
        The loaded model or default model if no model
        file is provided.
    """
    model_path = config.get("model_path", None)
    if model_path is not None:
        return model.Model.load(model_path)

    # Load the default model
    layers = [
        linear.Linear(784, 250, activation=act.ReLU),
        linear.Linear(250, 250, activation=act.ReLU),
        linear.Linear(250, 10)
    ]
    loss = cel.CrossEntropyLoss()
    return model.Model(
        layers,
        loss
    )
# endregion Load model


# region Train
def train_model(model: model.Model, config: dict[str, Any]) -> None:
    """
    Train the model based on the config values.

    Args:
        model: The model to train
        config: The configuration values from the config file
    """
    # Epochs
    if "epochs" not in config or config["epochs"] == 0:
        utils.print_warning(
            "No value for epochs was provided or was 0. Skipping training."
        )
        return
    epochs = config["epochs"]
    utils.check_type(epochs, int, "epochs")
    if epochs < 0:
        raise ValueError("epochs cannot be negative.")

    # Learning rate
    if "learning_rate" not in config:
        utils.print_warning(
            "Value of learning_rate not found, defaulting to 1e-4.")
    learning_rate = config.get("learning_rate", 1e-4)
    utils.check_type(learning_rate, float, "learning_rate")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0.")

    # Batch size
    if "batch_size" not in config:
        utils.print_warning("Value of batch_size not found, defaulting to 1.")
    batch_size = config.get("batch_size", 1)
    utils.check_type(batch_size, int, "batch_size")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    # Training images
    if "train_path" not in config:
        utils.print_warning(
            "No value for train_path was provided. Skipping training."
        )
        return
    if "train_validation_split" not in config:
        utils.print_warning(
            "No value for train_validation_split was provided."
            " Defaulting to 0.7."
        )
    if "file_formats" not in config:
        utils.print_warning(
            "No value for file_formats was provided."
            " Defaulting to only accept .png"
        )
    train_validation_split = config.get("train_validation_split", 0.7)
    file_formats = config.get("file_formats", [".png"])

    loader = image_loader.ImageLoader(
        config["train_path"],
        [
            utils.image_to_array,
            utils.normalise_image,
            utils.flatten
        ],
        file_formats,
        train_validation_split,
    )

    model.train(
        loader,
        learning_rate,
        batch_size,
        epochs
    )
# endregion Train


def main():
    """
    Sets up the environment based on the config file.

    Performs training if the number of epochs and train data are provided.

    If test data is provided, perform inferencing.

    Prompts the user with saving the model.
    """
    config = get_config()
    model = get_model(config)
    train_model(model, config)
    model.display_history_graph()


if __name__ == "__main__":
    main()
