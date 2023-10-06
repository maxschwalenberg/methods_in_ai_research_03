import json


def setup():
    # this function allows the user to set the configurations for the dialog system

    configuration = [False, False, False, False]

    user_utterance = ""
    while user_utterance != "0":
        message = (
            f"System: Hello, please choose your configuration:\n"
            f"Start: 0\n"
            f"Enable Text to speech: 1 \t status: {configuration[0]}\n"
            f"Enable Response delay: 2 \t status {configuration[1]}\n"
            f"Enable User feedback: 3 \t status: {configuration[2]}\n"
            f"Allow preference change: 4 \t status: {configuration[3]}"
        )
        print(message)
        user_utterance = input("Config choice: ")
        try:
            switch = int(user_utterance)
        except:
            print(f"Please enter a number!")
            continue

        if switch <= 4 and switch >= 1:
            configuration[switch - 1] = not configuration[switch - 1]

    print("\n\n")
    return configuration


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
