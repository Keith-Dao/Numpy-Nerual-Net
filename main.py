"""
This module is the main driver file.
"""
import argparse
import pathlib
import readline
import sys
from typing import Any

import matplotlib.pyplot as plt
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


def get_args() -> argparse.Namespace:
    """
    Gets the parsed args.

    Args:
        default_config_path: The config path to use if none is provided

    Returns:
        The command line args.
    """
    parser = argparse.ArgumentParser(
        description="Neural network for classifying images of digits."
    )
    parser.add_argument(
        "config_file",
        help="Path to the config file.",
        nargs="?",
        action=DefaultConfigPathAction
    )
    parser.add_argument(
        "-p",
        "--prediction-mode",
        help="Skip to the prediction mode.",
        action="store_true"
    )

    return parser.parse_args()
# endregion Argparse


# region Parse config
def get_config(config_path: str) -> dict[str, Any]:
    """
    Get the config values from the config file.

    Args:
        config_path: Path to the config file.

    Returns:
        A dictionary with the parsed config files.
    """
    with open(
        pathlib.Path(config_path),
        "r",
        encoding=sys.getdefaultencoding()
    ) as file:
        return yaml.safe_load(file)


def get_train_validation_split(
    config: dict[str, Any],
) -> float:
    """
    Get the train validation split in the config file.

    Args:
        config: The configuration values from the config file
        dataset: The dataset to load

    Returns:
        The train validation split.
    """
    if config.get("train_validation_split") is None:
        utils.print_warning(
            "No value for train_validation_split was provided."
            " Defaulting to 0.7."
        )
        return 0.7
    return config["train_validation_split"]


def get_file_formats(config: dict[str, Any]) -> list[str]:
    """
    Get the file formats listed in the config file.

    Args:
        config: The configuration values from the config file

    Returns:
        The list of valid file formats.
    """
    if config.get("file_formats") is None:
        utils.print_warning(
            "No value for file_formats was provided."
            " Defaulting to only accept .png"
        )
        return [".png"]
    return config["file_formats"]


def get_batch_size(config: dict[str, Any]) -> int:
    """
    Get the batch size from the config file.

    Args:
        config: The configuration values from the config file

    Returns:
        The batch size.
    """
    if config.get("batch_size") is None:
        utils.print_warning("Value of batch_size not found, defaulting to 1.")
        return 1
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be greater than 0.")
    return config["batch_size"]
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
    utils.print_warning("No model file was provided. Loading untrained model.")
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


# region Image loader
def get_image_loader(
    config: dict[str, Any],
    dataset: str
) -> image_loader.ImageLoader | None:
    """
    Creates an image loader.

    Args:
        config: The configuration values from the config file
        dataset: The dataset to load

    Returns:
        An image loader for the training data, if the data is provided.
        Else, None is returned.
    """
    if config.get(f"{dataset}_path") is None:
        utils.print_warning(
            f"No value for {dataset}_path was provided. Skipping {dataset}ing."
        )
        return None

    train_validation_split = (
        0
        if dataset == "test"
        else get_train_validation_split(config)
    )
    file_formats = get_file_formats(config)

    return image_loader.ImageLoader(
        config[f"{dataset}_path"],
        image_loader.ImageLoader.STANDARD_PREPROCESSING,
        file_formats,
        train_validation_split,
    )
# endregion Image loader


# region Train
def train_model(
    model_: model.Model,
    config: dict[str, Any]
) -> bool:
    """
    Train the model based on the config values.

    Args:
        model_: The model to train
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
    batch_size = get_batch_size(config)

    # Training images
    loader = get_image_loader(config, "train")
    if loader is None:
        return False

    model_.train(
        loader,
        learning_rate,
        batch_size,
        epochs
    )
    return True
# endregion Train


# region Save prompt
def prompt_save(model_: model.Model) -> None:
    """
    Prompt model save.

    Args:
        model_: The model to save
    """
    response = input("Would you like to save the model? [y/n]: ").lower()
    if not utils.is_yes(response):
        return

    def is_valid_path(path: pathlib.Path) -> bool:
        if path.suffix not in model_.SAVE_METHODS.keys():
            utils.print_error(
                f"File format \"{path.suffix}\" is not supported."
                f" Select from {' or '.join(model_.SAVE_METHODS.keys())}."
            )
            return False

        if path.exists():
            response = input(
                "The current file already exists. Would you like to overwrite"
                " it? [y/n]: "
            )
            return utils.is_yes(response)

        return True

    stop_code = "CANCEL"
    extensions = utils.join_with_different_last(
        model_.SAVE_METHODS.keys(),
        ", ",
        " or "
    )
    enter_path_prompt = (
        "Enter a file path with the one of the following extensions"
        f" ({extensions}) or type {stop_code} to cancel saving: "
    )
    save_path = utils.get_path_input(
        f"Where would you like to save the model file? {enter_path_prompt}",
        stop_code
    )
    while save_path is not None and not is_valid_path(save_path):
        save_path = utils.get_path_input(enter_path_prompt, stop_code)
    if save_path is None:
        print("Model was not saved.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model_.save(save_path)
    print(f"Model successfully saved at {save_path.resolve()}.")
# endregion Save prompt


# region Test
def test_model(model_: model.Model, config: dict[str, Any]) -> None:
    """
    Tests the model, if a test set is provided.

    Args:
        model_: The model to test
        config: The configuration values from the config file
    """
    test_image_loader = get_image_loader(config, "test")
    if test_image_loader is None:
        return

    if (config.get("test_metrics") or []) == []:
        utils.print_warning(
            "No metrics were provided in test_metrics. Skipping testing."
        )
        return
    utils.check_type(config["test_metrics"], (list), "test_metrics")
    metric_history = model.Model.metrics_list_to_dict(config["test_metrics"])

    batch_size = get_batch_size(config)

    test_loss, confusion_matrix = model_.test(
        test_image_loader("test", batch_size),
        "Testing"
    )
    model.Model.store_metrics(metric_history, confusion_matrix, test_loss)
    model.Model.print_metrics(metric_history, test_image_loader.classes)
# endregion Test


# region Train and test
def train_and_test(model_: model.Model, config: dict[str, Any]) -> None:
    """
    Train and test the model.

    Args:
        model_: The model to train and test
        config: The configuration values from the config file
    """
    if train_model(model_, config):
        model_.display_history_graphs()
        prompt_save(model_)
    test_model(model_, config)
# endregion Train and test


# region Predict
def start_prediction(
    model_: model.Model,
    config: dict[str, Any]
) -> None:
    """
    Start mode that allows users to choose files to predict with the model.

    Args:
        model_: The model to use to predict
        config: The configuration values from the config file
    """
    if model_.classes is None:
        utils.print_error(
            "Prediction is not available for untrained models. Please train"
            " the model first or load a pre-trained model."
        )
        return

    if not utils.is_yes(input("Would you like to predict images? [y/n]: ")):
        return

    model_.eval = True
    file_formats = get_file_formats(config)
    preprocessing = image_loader.ImageLoader.STANDARD_PREPROCESSING
    stop_code = "QUIT"
    while True:
        filepath = utils.get_path_input(
            f"Please enter the path to the image or {stop_code} to exit: ",
            stop_code
        )
        if filepath is None:
            break

        if filepath.suffix not in file_formats:
            utils.print_error("Invalid file format.")
            continue

        data = utils.image_to_array(filepath)
        for preprocess in preprocessing:
            data = preprocess(data)

        prediction = model_.predict(
            data  # pyright: ignore [reportGeneralTypeIssues]
        )[0]
        print(f"Predicted: {prediction}")

    model_.eval = False
# endregion Predict


def main():
    """
    Sets up the environment based on the config file.

    Performs training if the number of epochs and train data are provided.

    If test data is provided, perform inferencing.

    Prompts the user with saving the model.
    """
    readline.set_auto_history(True)
    args = get_args()
    config = get_config(args.config_file)
    model_ = get_model(config)
    if not args.prediction_mode:
        train_and_test(model_, config)
    start_prediction(model_, config)
    if plt.get_fignums():
        input("Hit enter to close all opened graphs: ")


if __name__ == "__main__":
    main()
