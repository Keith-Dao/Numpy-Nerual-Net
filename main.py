"""
This module is the main driver file.
"""
import argparse
import pathlib
import sys

import yaml

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
            print(
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


def main():
    """
    Sets up the environment based on the config file.

    Performs training if the number of epochs and train data are provided.

    If test data is provided, perform inferencing.

    Prompts the user with saving the model.
    """
    with open(
        pathlib.Path(get_config_filepath()),
        "r",
        encoding=sys.getdefaultencoding()
    ) as file:
        config = yaml.safe_load(file)
    print(config)


if __name__ == "__main__":
    main()
