import json
from dataclasses import dataclass


@dataclass
class FilePathsConfig:
    """Dataclass for the file paths configuration."""

    dialog_acts_path: str
    all_dialogs_path: str
    restaurant_info_path: str
    extended_restaurant_info_path: str
    evaluation_results_path: str
    baseline_rules_path: str
    additional_requirement_rules_path: str
    dialog_config_path: str


def load_file_paths_configuration(configuration_file_path: str) -> FilePathsConfig:
    """creates a dataclass from a dict containing the configuration data.

    Args:
        configuration_file_path (str): file path to the configuration data

    Returns:
        FilePathsConfig: configuration data as instance of dataclass
    """
    with open(configuration_file_path) as f:
        data = json.load(f)

    return FilePathsConfig(**data)


def load_configuration(configuration_file_path: str) -> list[bool]:
    """Loads the configuration for the dialog manager

    Args:
        configuration_file_path (str): path to the configuration data

    Returns:
        list[str]: contains the parsed configuration data as a list of boolean values
    """
    with open(configuration_file_path) as f:
        data = json.load(f)

    configuration = [
        data["text_to_speech"],
        data["delay"],
        data["allow_feedback"],
        data["allow_preference_change"],
    ]

    return configuration
