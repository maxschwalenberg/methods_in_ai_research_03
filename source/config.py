import json
from dataclasses import dataclass


@dataclass
class FilePathsConfig:
    dialog_acts_path: str
    all_dialogs_path: str
    restaurant_info_path: str
    extended_restaurant_info_path: str
    evaluation_results_path: str
    baseline_rules_path: str
    additional_requirement_rules_path: str
    dialog_config_path: str


def load_file_paths_configuration(configuration_file_path: str):
    with open(configuration_file_path) as f:
        data = json.load(f)

    return FilePathsConfig(**data)


def load_configuration(configuration_file_path: str):
    with open(configuration_file_path) as f:
        data = json.load(f)

    configuration = [
        data["text_to_speech"],
        data["delay"],
        data["allow_feedback"],
        data["allow_preference_change"],
    ]

    return configuration


load_file_paths_configuration("output/data/file_paths_config.json")
