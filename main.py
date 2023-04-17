"""
This module is the main driver file.
"""
import argparse
import pathlib
import readline
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
        loss,
        train_metrics=config.get("train_metrics"),
        validation_metrics=config.get("validation_metrics")
    )
# endregion Load model


# region Load image loader
def get_image_loader(
    config: dict[str, Any]
) -> image_loader.ImageLoader | None:
    """
    Creates an image loader.

    Args:
        config: The configuration values from the config file

    Returns:
        An image loader for the training data, if the data is provided.
        Else, None is returned.
    """
    if config.get("train_path") is None:
        utils.print_warning(
            "No value for train_path was provided. Skipping training."
        )
        return None

    if config.get("train_validation_split") is None:
        utils.print_warning(
            "No value for train_validation_split was provided."
            " Defaulting to 0.7."
        )
        train_validation_split = 0.7
    else:
        train_validation_split = config["train_validation_split"]

    if config.get("file_formats") is None:
        utils.print_warning(
            "No value for file_formats was provided."
            " Defaulting to only accept .png"
        )
        file_formats = [".png"]
    else:
        file_formats = config["file_formats"]

    return image_loader.ImageLoader(
        config["train_path"],
        [
            utils.image_to_array,
            utils.normalise_image,
            utils.flatten
        ],
        file_formats,
        train_validation_split,
    )
# endregion Load image loader


# region Train
def train_model(
    model: model.Model,
    config: dict[str, Any]
) -> bool:
    """
    Train the model based on the config values.

    Args:
        model: The model to train
        config: The configuration values from the config file

    Returns:
        Boolean if the model was trained.
    """
    # Epochs
    if not config.get("epochs"):
        utils.print_warning(
            "No value for epochs was provided or was 0. Skipping training."
        )
        return False
    epochs = config["epochs"]

    # Learning rate
    if config.get("learning_rate") is None:
        utils.print_warning(
            "Value of learning_rate not found, defaulting to 1e-4."
        )
        learning_rate = 1e-4
    else:
        learning_rate = config["learning_rate"]
    utils.check_type(learning_rate, (float, int), "learning_rate")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0.")

    # Batch size
    if config.get("batch_size") is None:
        utils.print_warning("Value of batch_size not found, defaulting to 1.")
        batch_size = 1
    else:
        batch_size = config["batch_size"]

    # Training images
    loader = get_image_loader(config)
    if loader is None:
        return False

    model.train(
        loader,
        learning_rate,
        batch_size,
        epochs
    )
    return True
# endregion Train


# region Save prompt
def prompt_save(model: model.Model) -> None:
    """
    Prompt model save.
    """
    readline.set_auto_history(True)

    def is_yes(response: str) -> bool:
        while response not in ["y", "n"]:
            response = input(
                "Please enter either y for yes or n for no: ").lower()
        return response == "y"

    response = input("Would you like to save the model? [y/n]: ").lower()
    if not is_yes(response):
        return

    def is_valid_path(path: pathlib.Path) -> bool:
        if path.suffix not in model.SAVE_METHODS.keys():
            utils.print_error(
                f"File format \"{save_path.suffix}\" is not supported."
                f" Select from {' or '.join(model.SAVE_METHODS.keys())}."
            )
            return False

        if path.exists():
            response = input(
                "The current file already exists. Would you like to overwrite"
                " it? [y/n]: "
            )
            return is_yes(response)

        return True

    save_path = pathlib.Path(input(
        "Where would you like to save the model file?"
        " Enter a file path with the one of the following extensions"
        f" ({', '.join(model.SAVE_METHODS.keys())}): "
    ))
    while not is_valid_path(save_path):
        save_path = pathlib.Path(
            input("Please enter the location to save the model file: ")
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"Model successfully saved at {save_path.resolve()}.")
# endregion Save prompt


def main():
    """
    Sets up the environment based on the config file.

    Performs training if the number of epochs and train data are provided.

    If test data is provided, perform inferencing.

    Prompts the user with saving the model.
    """
    config = get_config()
    model = get_model(config)
    trained = train_model(model, config)
    if trained:
        model.display_history_graphs()
        prompt_save(model)


if __name__ == "__main__":
    main()
