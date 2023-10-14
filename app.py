from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pymongo
from pymongo.database import Database
import pickle
from source.dialog_management import *
from source.config import (
    FilePathsConfig,
    load_configuration,
    load_file_paths_configuration,
)
from source.datacreator import Datacreator
from source.ml_model import DecisionTreeModel
from source.model import Model
import json

app = Flask(__name__)
app.secret_key = "topsecret"


client: pymongo.MongoClient = pymongo.MongoClient("127.0.0.1", 27017)
db: Database = client.flask_db


model_path = "output/data/decision_tree.rf"


def load_ml_model(path):
    loaded_model = pickle.load(open(path, "rb"))

    return loaded_model


def setup():
    filenames_config = load_file_paths_configuration(
        "output/data/file_paths_config.json"
    )
    configuration = load_configuration(filenames_config.dialog_config_path)

    return configuration, filenames_config


class DialogManagementWrapper(DialogManagement):
    def __init__(
        self,
        classifier: Model,
        configuration,
        file_paths_config: FilePathsConfig,
        current_state: str,
        extracted_preference: dict,
        extracted_preferences_old: dict,
        suggestions,
        previous_suggestion_index,
        request_utterance,
        debug=False,
    ) -> None:
        super().__init__(classifier, configuration, file_paths_config, debug)

        info = Info(
            self.keyword_dict,
            extracted_preference,
            extracted_preferences_old,
            configuration,
            file_paths_config,
        )

        # initialize the welcome state with no known preferences
        state_class = globals()[current_state]

        if current_state == "Contradiction":
            instance = state_class(info, "explanation string")
        elif current_state == "GiveDetails":
            instance = state_class(
                info,
                suggestions,
                previous_suggestion_index,
                request_utterance,
            )
        elif current_state == "Suggestion":
            instance = state_class(info, previous_suggestion_index=previous_suggestion_index)

        else:
            instance = state_class(info)

        self.current_state = instance

    def generate_feedback_string(self):
        if self.current_state.info.allow_feedback:
            self.current_state.give_preferences_feedback()
        else:
            self.current_state.feedback_string = ""

    def transition(self, user_utterance):
        # self.dialog() executes system response
        # optionally do delay
        if self.current_state.info.delay:
            time.sleep(random.uniform(0.5, 2))
        # self.dialog()
        classified_response = self.classifier.predict_single_sentence(user_utterance)
        print(f"{classified_response}\t{user_utterance}")
        self.current_state.user_utterance = user_utterance

        if type(self.current_state).__name__ == "Suggestion":
            self.current_state.restaurant_lookup.lookup(self.current_state.info.extracted_preferences)

        new_state = self.current_state.transition(classified_response)
        return new_state


parcipant_info_file = "completed_forms_data.json"



@app.route("/")
def chatbot():
    decision_tree = load_ml_model("output/data/decision_tree.rf")
    configuration, filenames_config = setup()
    dialog = DialogManagementWrapper(
        decision_tree,
        configuration,
        filenames_config,
        "Welcome",
        {},
        {},
        None,
        None,
        None,
        debug=False,
    )

    session["current_state"] = type(dialog.current_state).__name__
    session["extracted_preferences"] = dialog.current_state.info.extracted_preferences
    session[
        "extracted_preferences_old"
    ] = dialog.current_state.info.extracted_preferences_old
    session["suggestions"] = None
    session["request_utterance"] = None
    session["previous_suggestion_index"] = None

    dialog.generate_feedback_string()
    return_message = dialog.current_state.dialog(return_message=True)

    return_message = return_message.replace("System: ", "")


    with open(parcipant_info_file) as f:
        data = json.load(f)

    participant_number = random.randint(0, 999)
    while participant_number in (data["assigned_numbers"] + data["completed_numbers"]):
        participant_number = random.randint(0, 999)

    if data["started_with_word_delay"] > data["started_without_word_delay"]:
        word_delay = False
    
    else:
        word_delay = True
    
    with open(parcipant_info_file, "w") as f:
        data["assigned_numbers"].append(participant_number)

        if len(data["assigned_numbers"]) > 20:
            data["assigned_numbers"].pop()

        json.dump(data, f)

    return render_template("chatbot.html", first_message=return_message, word_delay=word_delay, participant_number=participant_number)


@app.route("/thanks", methods=["GET"])
def thanks():
    return render_template("thanks.html")


@app.route("/api/restart_dialog", methods=["GET"])
def restart_dialog():
    decision_tree = load_ml_model("output/data/decision_tree.rf")
    configuration, filenames_config = setup()
    dialog = DialogManagementWrapper(
        decision_tree,
        configuration,
        filenames_config,
        "Welcome",
        {},
        {},
        None,
        None,
        None,
        debug=False,
    )

    session["current_state"] = type(dialog.current_state).__name__
    session["extracted_preferences"] = dialog.current_state.info.extracted_preferences
    session[
        "extracted_preferences_old"
    ] = dialog.current_state.info.extracted_preferences_old
    session["suggestions"] = None
    session["request_utterance"] = None
    session["previous_suggestion_index"] = None

    dialog.generate_feedback_string()
    return_message = dialog.current_state.dialog(return_message=True)

    return_message = return_message.replace("System: ", "")

    return jsonify({"response": return_message})


@app.route("/api/user_input", methods=["POST"])
def api_return_response():
    data = request.get_json()
    decision_tree = load_ml_model("output/data/decision_tree.rf")
    configuration, filenames_config = setup()

    current_state = session["current_state"]
    extracted_preferences = session["extracted_preferences"]
    extracted_preferences_old = session["extracted_preferences_old"]

    # reconstruct dialog state
    dialog = DialogManagementWrapper(
        decision_tree,
        configuration,
        filenames_config,
        current_state,
        extracted_preferences,
        extracted_preferences_old,
        session["suggestions"],
        session["previous_suggestion_index"],
        session["request_utterance"],
        debug=False,
    )

    if current_state != "GiveDetails":
        session["request_utterance"] = data["utterance"]


    if type(current_state).__name__ == "Suggestion":
        # to execute lookup function
        print("Fuck")
        dialog.current_state.restaurant_lookup.lookup(dialog.current_state.info.extracted_preferences)

    new_state: State = dialog.transition(data["utterance"])
    

    try:
        session["suggestions"] = new_state.suggestions
        session["previous_suggestion_index"] = new_state.previous_suggestion_index
    except:
        pass


    dialog = DialogManagementWrapper(
        decision_tree,
        configuration,
        filenames_config,
        type(new_state).__name__,
        new_state.info.extracted_preferences,
        new_state.info.extracted_preferences_old,
        session["suggestions"],
        session["previous_suggestion_index"],
        session["request_utterance"],
        debug=False,
    )
    print(f"State Name: {type(new_state).__name__}")
    try:
        dialog.generate_feedback_string()
        return_message = dialog.current_state.dialog(return_message=True)
    except:
        new_state: State = dialog.transition("")

        try:
            session["suggestions"] = new_state.suggestions
            session["previous_suggestion_index"] = new_state.previous_suggestion_index
        except:
            pass

        print(new_state)
        dialog = DialogManagementWrapper(
            decision_tree,
            configuration,
            filenames_config,
            type(new_state).__name__,
            new_state.info.extracted_preferences,
            new_state.info.extracted_preferences_old,
            session["suggestions"],
            session["previous_suggestion_index"],
            session["request_utterance"],
            debug=False,
        )

        dialog.generate_feedback_string()
        return_message = dialog.current_state.dialog(return_message=True)

    print(new_state)
    print(return_message)
    if new_state is None:
        pass
    else:
        session["current_state"] = type(dialog.current_state).__name__
        session[
            "extracted_preferences"
        ] = dialog.current_state.info.extracted_preferences
        session[
            "extracted_preferences_old"
        ] = dialog.current_state.info.extracted_preferences_old

    return_message = return_message.replace("System: ", "")

    return_data = {}
    if session["current_state"] == "Goodbye":
        return_data["dialog_finished"] = True
    else:
        return_data["dialog_finished"] = False

    return_data["response"] = return_message
    print(return_data)

    return jsonify(return_data), 200


@app.route("/api/forms_completed", methods=["POST"])
def forms_completed():
    participant_data = request.get_json()

    with open(parcipant_info_file) as f:
        data = json.load(f)

    with open(parcipant_info_file, "w") as f:
        if participant_data["word_delay"]:
            data["started_with_word_delay"] += 1
        else:
            data["started_without_word_delay"] += 1

        try:
            data["completed_numbers"].append(participant_data["participant_number"])
        except:
            pass
        
        try:
            data["assigned_numbers"].remove(participant_data["participant_number"])
        except:
            pass

        json.dump(data, f)

        
    


    return "dwadw"