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


def main():
    """
    Sets up the environment based on the config file.

    Performs training if the number of epochs and train data are provided.

    If test data is provided, perform inferencing.

    Prompts the user with saving the model.
    """
    config = get_config()
    model = get_model(config)
    print(model)


if __name__ == "__main__":
    main()
