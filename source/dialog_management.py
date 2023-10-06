import csv
import random
import re
import copy
import time
from Levenshtein import distance as levdistance
import pyttsx3

from source.model import Model
from source.restaurant_lookup import RestaurantLookup
from source.config import FilePathsConfig

# for speech
engine = pyttsx3.init()


class DialogManagement:
    """The dialog system which handles and keeps track of
    the conversation.
    """

    def __init__(
        self,
        classifier: Model,
        configuration,
        file_paths_config: FilePathsConfig,
        debug=False,
    ) -> None:
        self.classifier = classifier

        self.fetchKeywords(file_paths_config.extended_restaurant_info_path)

        # initialize info instance
        info = Info(self.keyword_dict, {}, {}, configuration, file_paths_config)

        # initialize the welcome state with no known preferences
        self.current_state = Welcome(info)

        # optionally enable debugging --> print classifications for each user input
        self.debug = debug

    def run_dialog(self):
        # while the current state is not goodbye we can go to a new state which is based on the running of classifier ml model
        while not isinstance(self.current_state, Goodbye):
            (
                new_state,
                classified_response,
            ) = self.current_state.run(self.classifier)
            self.current_state = new_state

            if self.debug:
                print(f"Classified `{classified_response}`")

        # run the goodbye state
        self.current_state.run()

    def fetchKeywords(self, filename):
        file = open(filename)
        file = csv.DictReader(file)
        keyword_names = file.fieldnames[
            1:4
        ]  # incongruent with preference_extraction.py
        keyword_dict = {key: set() for key in keyword_names}

        for row in file:
            for keyword in keyword_names:
                if (row[keyword]) != "":
                    keyword_dict[keyword].add(row[keyword])

        self.keyword_dict = keyword_dict


def text_to_speech(message: str):
    # remove system string
    if "System:" in message:
        message = message.replace("System:", "")

    engine.say(message)
    # play the speech
    engine.runAndWait()


# need to add postcode?
def pattern_match_request(data):
    data = data.lower()

    if re.findall("phone", data) != []:
        return "phone"
    elif re.match("address", data) != []:
        return "address"
    else:
        return None


def patternMatchKeywordExtraction(data, keyword_dict, context : str):
    data = data.lower()
    temp = None
    result = {}

    # Keyword search
    for key, values in keyword_dict.items():
        for value in values:
            if temp := (re.findall(r"\b" + value + r"\b", data)):
                result[key] = value

    # Pattern search to find the possible misspelled keywords

    # Check for food keyword pattern (* food)
    # Example: "I want to eat (chinese) food"
    if (temp := re.findall("(\w+) food", data)) and ("food" not in result):
        for keyword in keyword_dict["food"]:
            if levdistance(temp[0], keyword) <= 2:
                result["food"] = keyword

    # Check for area keyword pattern (in the *)
    # Example: "I would like a restaurant in the (south) side"
    if (temp := re.findall(r"in the (\w+)", data)) and ("area" not in result):
        for keyword in keyword_dict["area"]:
            if levdistance(temp[0], keyword) <= 2:
                result["area"] = keyword

    # Check for price range keyword pattern (* priced)
    # Example: "I want the restaurant to be (cheap) priced"
    if (temp := re.findall(r"(\w+) priced", data)) and ("pricerange" not in result):
        for keyword in keyword_dict["pricerange"]:
            if levdistance(temp[0], keyword) <= 2:
                result["pricerange"] = keyword

    # Check for food/price range keyword pattern (* restaurant)
    # Example: "I want a (Chinese)/(Cheap) restaurant"
    if (temp := re.findall("(\w+) restaurant", data)) and (
        "food" or "pricerange" not in result
    ):
        for key, values in keyword_dict.items():
            for value in values:
                if levdistance(temp[0], value) <= 2:
                    result[key] = value

    # Check for any keyword pattern (any *)
    # Example: "I want any (food)/(area)/(pricerange)"
    if (re.search(r"\b" + "any" + r"\b", data) and (len(result) != 3)):
        if (temp := re.findall("any (\w+)", data)):
            for word in temp:
                for key in keyword_dict.keys():
                    if levdistance(word, key) <= 2:
                        result[key] = "Any"
        elif (context in keyword_dict.keys()):
             result[context] = "Any"

    return result


def additional_keyword_extraction(data):
    result = {}
    add_list = [
        "touristic",
        "assigned seats",
        "children",
        "romantic",
    ]
    data = data.split()
    for word in data:
        for keyword in add_list:
            if levdistance(word, keyword) <= 2:
                result["additional_requirement"] = keyword
                break
    return result


class Info:
    def __init__(
        self,
        keyword_dict,
        extracted_preferences: dict,
        extracted_preferences_old: dict,
        configuration,
        file_paths_config: FilePathsConfig,
    ) -> None:
        self.keyword_dict = keyword_dict
        self.extracted_preferences = extracted_preferences
        self.extracted_preferences_old = extracted_preferences_old
        self.file_paths_config = file_paths_config

        # Configuration settings by the user
        (
            self.t2s,
            self.delay,
            self.allow_feedback,
            self.allow_preference_change,
        ) = configuration


class State:
    def __init__(self, info: Info) -> None:
        self.info = info
        self.keyword_dict = info.keyword_dict

        # create two dicts of extracted preferences. the old one always stores the extracted preferences of the previous state
        # that way the system can give feedback to the user if new preferences were detected
        self.user_utterance = ""

    def give_preferences_feedback(self):
        changed_values = {}

        for key in self.info.extracted_preferences:
            try:
                current_preference = self.info.extracted_preferences[key]
                old_preference = self.info.extracted_preferences_old[key]

                if current_preference != old_preference:
                    changed_values[key] = current_preference

            except:
                changed_values[key] = self.info.extracted_preferences[key]

        n_changed_values = len(list(changed_values.keys()))
        feedback_string = ""
        if n_changed_values > 0:
            feedback_string += "Okay"
            for key_i, key in enumerate(changed_values):
                if (key_i + 1) == n_changed_values and n_changed_values > 1:
                    feedback_string += " and "
                else:
                    feedback_string += ", "
                if key == "food":
                    feedback_string += (
                        f"the restaurant should serve {changed_values[key]} food"
                    )
                elif key == "area":
                    feedback_string += (
                        f"the restaurant should be in the {changed_values[key]}"
                    )
                elif key == "pricerange":
                    feedback_string += f"the price should be {changed_values[key]}"

            feedback_string += ". "

        self.feedback_string = feedback_string

    def run(self, classifier: Model):
        # enable/disable the feedback for the user that a preference was extracted
        if self.info.allow_feedback:
            self.give_preferences_feedback()
        else:
            self.feedback_string = ""

        # self.dialog() executes system response
        # optionally do delay
        if self.info.delay:
            time.sleep(random.uniform(0.5, 2))
        self.dialog()
        classified_response = classifier.predict_single_sentence(self.user_utterance)

        new_state = self.transition(classified_response)
        # print
        print(self.info.extracted_preferences)
        return new_state, classified_response

    def dialog(self):
        self.user_utterance = input("User: ")

    def transition(self, input):
        pass


class Welcome(State):
    def __init__(self, info: Info) -> None:
        super().__init__(info)

    def dialog(self):
        message = f"System: {self.feedback_string}Hello, welcome to the restaurant recommender system? You can ask for restaurants by area/price range/food type. How may I help you?"
        print(message)
        if self.info.t2s:
            text_to_speech(message)
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart" or input == "repeat":
            return Welcome(self.info)
        elif input == "bye":
            return Goodbye(self.info)
        elif input == "inform" or input == "hello":
            # extract preferences
            self.info.extracted_preferences_old = copy.deepcopy(
                self.info.extracted_preferences
            )
            self.info.extracted_preferences = patternMatchKeywordExtraction(
                self.user_utterance, self.keyword_dict, None
            )
            return AskForInformation(self.info)


class AskForInformation(State):
    def __init__(self, info: Info) -> None:
        super().__init__(info)

    def dialog(self):
        user_utterance = "temporary string"

        return user_utterance

    def transition(self, input):
        # analyze preferences and route accordingly
        if "area" not in self.info.extracted_preferences:
            return AskArea(self.info)
        elif "pricerange" not in self.info.extracted_preferences:
            return AskPrice(self.info)
        elif "food" not in self.info.extracted_preferences:
            return AskType(self.info)
        else:  # everything is filled
            return AskForAdditionalInformation(self.info)


class AskArea(State):
    def __init__(self, info: Info) -> None:
        super().__init__(info)

    def dialog(self):
        message = f"System: {self.feedback_string}Which area do you want to go?"
        print(message)
        if self.info.t2s:
            text_to_speech(message)

        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.info)
        elif input == "bye":
            return Goodbye(self.info)
        elif (
            input == "negate" or input == "repeat"
        ):  # if the user negate the ask, we should ask again
            return AskArea(self.info)
        elif (
            input == "deny"
        ):  # if deny we delete the preferences and we go back to ask for info
            extracted_preferences = patternMatchKeywordExtraction(
                self.user_utterance, self.keyword_dict, "area"
            )
            already_existing_keys = list(self.info.extracted_preferences.keys())
            for existing_key in already_existing_keys:
                if existing_key in list(extracted_preferences.keys()):
                    del extracted_preferences[existing_key]
            self.info.extracted_preferences.update(extracted_preferences)

            return AskForInformation(self.info)
        elif input == "inform":
            # extract preferences
            self.info.extracted_preferences_old = copy.deepcopy(
                self.info.extracted_preferences
            )

            # if we want to allow preferences to be overwritten
            if self.info.allow_preference_change:
                self.info.extracted_preferences.update(
                    patternMatchKeywordExtraction(
                        self.user_utterance, self.keyword_dict, "area"
                    )
                )
            # else, first delete all the entries of the extracted preferences that are already present in the dict
            else:
                extracted_preferences = patternMatchKeywordExtraction(
                    self.user_utterance, self.keyword_dict, "area"
                )
                already_existing_keys = list(self.info.extracted_preferences.keys())
                for existing_key in already_existing_keys:
                    if existing_key in list(extracted_preferences.keys()):
                        del extracted_preferences[existing_key]
                self.info.extracted_preferences.update(extracted_preferences)

            return AskForInformation(self.info)
        else:
            return AskForInformation(self.info)


class AskPrice(State):
    def __init__(self, info: Info) -> None:
        super().__init__(info)

    def dialog(self):
        message = (
            f"System: {self.feedback_string}How expensive should the restaurant be?"
        )
        print(message)
        if self.info.t2s:
            text_to_speech(message)

        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.info)
        elif input == "bye":
            return Goodbye(self.info)
        elif (
            input == "negate" or input == "repeat"
        ):  # if the user negate the ask, we should ask again
            return AskPrice(self.info)
        elif input == "inform":
            # extract preferences
            self.info.extracted_preferences_old = copy.deepcopy(
                self.info.extracted_preferences
            )

            # if we want to allow preferences to be overwritten
            if self.info.allow_preference_change:
                self.info.extracted_preferences.update(
                    patternMatchKeywordExtraction(
                        self.user_utterance, self.keyword_dict, "pricerange"
                    )
                )
            # else, first delete all the entries of the extracted preferences that are already present in the dict
            else:
                extracted_preferences = patternMatchKeywordExtraction(
                    self.user_utterance, self.keyword_dict, "pricerange"
                )
                already_existing_keys = list(self.info.extracted_preferences.keys())
                for existing_key in already_existing_keys:
                    if existing_key in list(extracted_preferences.keys()):
                        del extracted_preferences[existing_key]
                self.info.extracted_preferences.update(extracted_preferences)

            return AskForInformation(self.info)
        else:
            return AskForInformation(self.info)


class AskType(State):
    def __init__(self, info: Info) -> None:
        super().__init__(info)

    def dialog(self):
        message = f"System: {self.feedback_string}What type of food would you like?"
        print(message)
        if self.info.t2s:
            text_to_speech(message)

        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.info)
        elif input == "bye":
            return Goodbye(self.info)
        elif (
            input == "negate" or input == "negate"
        ):  # if the user negate the ask, we should ask again
            return AskType(self.info)
        elif input == "inform":
            # extract preferences
            self.info.extracted_preferences_old = copy.deepcopy(
                self.info.extracted_preferences
            )

            # if we want to allow preferences to be overwritten
            if self.info.allow_preference_change:
                self.info.extracted_preferences.update(
                    patternMatchKeywordExtraction(
                        self.user_utterance, self.keyword_dict, "food"
                    )
                )
            # else, first delete all the entries of the extracted preferences that are already present in the dict
            else:
                extracted_preferences = patternMatchKeywordExtraction(
                    self.user_utterance, self.keyword_dict, "food"
                )
                already_existing_keys = list(self.info.extracted_preferences.keys())
                for existing_key in already_existing_keys:
                    if existing_key in list(extracted_preferences.keys()):
                        del extracted_preferences[existing_key]
                self.info.extracted_preferences.update(extracted_preferences)

            return AskForInformation(self.info)
        else:
            return AskForInformation(self.info)


class AskForAdditionalInformation(State):
    def __init__(self, info: Info) -> None:
        super().__init__(info)

    def dialog(self):
        message = f"System: {self.feedback_string}Do you have additional requirements?"
        print(message)
        if self.info.t2s:
            text_to_speech(message)
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.info)
        elif input == "bye":
            return Goodbye(self.info)
        elif (
            # why negate or negate?
            input == "negate"
            or input == "negate"
        ):  # if the user negate the ask, we should ask again ?? Should this go to suggestion?
            return Suggestion(self.info)
        elif input == "inform":
            # if we want to allow preferences to be overwritten
            if self.info.allow_preference_change:
                self.info.extracted_preferences.update(
                    additional_keyword_extraction(self.user_utterance)
                )
            # else, first delete all the entries of the extracted preferences that are already present in the dict
            else:
                extracted_preferences = additional_keyword_extraction(
                    self.user_utterance
                )
                already_existing_keys = list(self.info.extracted_preferences.keys())
                for existing_key in already_existing_keys:
                    if existing_key in list(extracted_preferences.keys()):
                        del extracted_preferences[existing_key]
                self.info.extracted_preferences.update(extracted_preferences)

            return Suggestion(self.info)
        else:
            return AskForInformation(self.info)


class Suggestion(State):
    def __init__(
        self,
        info: Info,
        previous_suggestion_index=None,
    ) -> None:
        super().__init__(info)

        self.restaurant_lookup = RestaurantLookup(info.file_paths_config)
        self.previous_suggestion_index = previous_suggestion_index
        self.suggestions = None

    def dialog(self):
        self.suggestions = self.restaurant_lookup.lookup(
            self.info.extracted_preferences
        )
        if not self.suggestions.empty:
            random_index = 0
            if len(self.suggestions.values) > 1:
                while self.previous_suggestion_index == random_index:
                    random_index = random.randrange(0, (len(self.suggestions.values)))
            self.previous_suggestion_index = random_index

            message = f"System: {self.feedback_string}The best restaurant according to your preferences is: {self.suggestions.values[random_index][0]}."

            if "additional_requirement" in self.info.extracted_preferences:
                message += f" {self.restaurant_lookup.explain_inference(self.suggestions.iloc[random_index], self.info.extracted_preferences['additional_requirement'])}"

            print(message)
            if self.info.t2s:
                text_to_speech(message)

        else:
            message = "System: No restaurants found."
            print(message)
            if self.info.t2s:
                text_to_speech(message)
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        # after the suggestion it can negate/ back to ask again for parameters | it can ask for alternative
        # it can say bye | it can confirm
        if input == "restart":
            return Welcome(self.info)
        elif input == "bye" or input == "thankyou":
            return Goodbye(self.info)
        elif input == "negate":
            return Welcome(self.info)
        elif input == "reqalts" or input == "reqmore":
            return Suggestion(
                self.info,
                self.previous_suggestion_index,
            )
        elif input == "request":
            return GiveDetails(
                self.info,
                self.suggestions.values,
                self.previous_suggestion_index,
                self.user_utterance,
            )
        elif (
            input == "deny"
        ):  # if deny we delete the preferences and we go back to ask for info
            extracted_preferences = patternMatchKeywordExtraction(
                self.user_utterance, self.keyword_dict, None
            )
            already_existing_keys = list(self.info.extracted_preferences.keys())
            for existing_key in already_existing_keys:
                if existing_key in list(extracted_preferences.keys()):
                    del extracted_preferences[existing_key]
            self.info.extracted_preferences.update(extracted_preferences)

            return AskForInformation(self.info)
        else:
            return Suggestion(
                self.info,
                previous_suggestion_index=self.previous_suggestion_index,
            )  # if the user does not like the suggestion which is the next state?


class GiveDetails(State):
    def __init__(
        self,
        info: Info,
        suggestions,
        previous_suggestion_index,
        request_utterance,
    ) -> None:
        super().__init__(info)

        self.previous_suggestion_index = previous_suggestion_index
        self.suggestions = suggestions
        self.request_utterance = request_utterance

    def dialog(self):
        request_type = pattern_match_request(self.request_utterance)

        if request_type == "phone":
            message = f"System: {self.feedback_string}The phone number of this restaurant is: {self.suggestions[self.previous_suggestion_index][4]}"

        elif request_type == "address":
            message = f"System: {self.feedback_string}The address number of this restaurant is: {self.suggestions[self.previous_suggestion_index][5]}"

        else:
            message = "Sorry I can't understand this request"

        print(message)
        if self.info.t2s:
            text_to_speech(message)

        user_utterance = super().dialog()
        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.info)
        elif (
            input == "bye"
            or input == "thankyou"
            or input == "ack"
            or input == "confirm"
            or input == "affirm"
        ):
            return Goodbye(self.info)
        elif input == "repeat":
            return GiveDetails(
                self.info,
                self.suggestions,
                self.previous_suggestion_index,
                self.request_utterance,
            )
        elif input == "request":
            return GiveDetails(
                self.info,
                self.suggestions,
                self.previous_suggestion_index,
                self.request_utterance,
            )
        elif input == "negate" or input == "reqalts":
            return Suggestion(
                self.info,
                self.previous_suggestion_index,
            )


class Goodbye(State):
    def __init__(self, info: Info) -> None:
        super().__init__(info)

    def dialog(self):
        message = "System: Goodbye, have a nice day!"

        print(message)
        if self.info.t2s:
            text_to_speech(message)

    def run(self):
        self.dialog()
