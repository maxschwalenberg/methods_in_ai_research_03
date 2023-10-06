import json


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
