import csv
import random
import re
from Levenshtein import distance as levdistance

from source.model import Model
from source.restaurant_lookup import RestaurantLookup

# TODO
# 1. create a rules dict what the system says in each state
# 2. create a state transitions dict to change state based on a classification result


class DialogManagement:
    def __init__(self, classifier: Model, debug=False) -> None:
        self.classifier = classifier

        self.know_preferences: dict = {}
        self.know_preferences = {
            "area": "west",
            "pricerange": "moderate",
            "food": "british",
        }

        filename = "data/restaurant_info.csv"
        self.restaurant_lookup = RestaurantLookup(filename)
        self.fetchKeywords(filename)
        self.current_state = Welcome(self.keyword_dict)

        # optionally enable debugging --> print classifications for each user input
        self.debug = debug

    def run_dialog(self):
        while not isinstance(self.current_state, Goodbye):
            (
                new_state,
                classified_response,
                extracted_preferences,
            ) = self.current_state.run(self.classifier)
            self.current_state = new_state

            # overwrite known preferences with new ones if there are any

            for preference_key in extracted_preferences:
                self.know_preferences[preference_key] = extracted_preferences[
                    preference_key
                ]

            if self.debug:
                print(f"Classified `{classified_response}`")

        # run the goodbye state
        self.current_state.run()

    def fetchKeywords(self, filename):
        file = open(filename)
        file = csv.DictReader(file)
        keyword_names = file.fieldnames[1:4]
        keyword_dict = {key: set() for key in keyword_names}

        for row in file:
            for keyword in keyword_names:
                if (row[keyword]) != "":
                    keyword_dict[keyword].add(row[keyword])

        self.keyword_dict = keyword_dict


def patternMatchKeywordExtraction(data, keyword_dict):
    data = data.lower()
    temp = None
    result = {}
    if temp := re.findall("(\w+) food", data):
        result["food"] = temp[0]
        # result.append(("food", temp[0]))
    if temp := re.findall("in the (\w+)", data):
        result["area"] = temp[0]
        # result.append(("area", temp[0]))
    if temp := re.findall("(\w+) priced", data):
        # result.append(("pricerange", temp[0]))
        result["pricerange"] = temp[0]

    if temp := re.findall("(\w+) restaurant", data):
        for key, values in keyword_dict.items():
            for value in values:
                if levdistance(temp[0], value) <= 2:
                    result[key] = value

    return result


class State:
    def __init__(self, keyword_dict) -> None:
        self.keyword_dict = keyword_dict
        self.extracted_preferences: dict = {}

    def run(self, classifier: Model):
        self.dialog()
        classified_response = classifier.predict_single_sentence(self.user_utterance)

        new_state, extracted_preferences = self.transition(classified_response)

        return new_state, classified_response, extracted_preferences

    def dialog(self):
        self.user_utterance = input("User: ")

    def transition(self, input):
        # if user wants to start over, it does not matter which is the current state
        # the same with bye
        #if input == "restart":
         #   return Welcome()
        #elif input == "bye":
         #   return Goodbye()
        #else:
        pass


class Welcome(State):
    def __init__(self, keyword_dict) -> None:
        super().__init__(keyword_dict)

    def dialog(self):
        print("System: Hello, how can I help you?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        extracted_preferences = {}
        if input == "restart" or input == "repeat":
            return Welcome(self.keyword_dict), extracted_preferences
        elif input == "bye":
            return Goodbye(self.keyword_dict), extracted_preferences
        elif input == "inform" or input == "hello":
            # extract preferences
            self.extracted_preferences = patternMatchKeywordExtraction(
                self.user_utterance, self.keyword_dict
            )
            return AskForInformation(self.keyword_dict), self.extracted_preferences


class AskForInformation(State):
    def __init__(self, keyword_dict) -> None:
        super().__init__(keyword_dict)

    def dialog(self):
        user_utterance = "temporary string"

        return user_utterance

    def transition(self, input):
        # analyze preferences and route accordingly
        if self.extracted_preferences["area"] == None:
            return AskArea(self.keyword_dict), self.extracted_preferences
        elif self.extracted_preferences["princerange"] == None:
            return AskPrice(self.keyword_dict), self.extracted_preferences
        elif self.extracted_preferences["food"] == None:
            return AskType(self.keyword_dict), self.extracted_preferences
        else: #everything is filled
            return Suggestion (self.keyword_dict), self.extracted_preferences

class AskArea(State):
    def __init__(self, keyword_dict) -> None:
        super().__init__(keyword_dict)

    def dialog(self):
        print("System: Which area do you want to go?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.keyword_dict), self.extracted_preferences
        elif input == "bye":
            return Goodbye(self.keyword_dict), self.extracted_preferences
        elif input == "negate" or input == "repeat": # if the user negate the ask, we should ask again
            return AskArea(self.keyword_dict), self.extracted_preferences
        elif input == "inform":
            # extract preferences
            self.extracted_preferences = patternMatchKeywordExtraction(
                self.user_utterance, self.keyword_dict
            )
            return AskForInformation(self.keyword_dict), self.extracted_preferences


class AskPrice(State):
    def __init__(self, keyword_dict) -> None:
        super().__init__(keyword_dict)

    def dialog(self):
        print("System: How expensive should the restaurant be?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.keyword_dict), self.extracted_preferences
        elif input == "bye":
            return Goodbye(self.keyword_dict), self.extracted_preferences
        elif input == "negate" or input == "repeat": # if the user negate the ask, we should ask again
            return AskPrice(self.keyword_dict), self.extracted_preferences
        elif input == "inform":
            # extract preferences
            self.extracted_preferences = patternMatchKeywordExtraction(
                self.user_utterance, self.keyword_dict
            )
            return AskForInformation(self.keyword_dict), self.extracted_preferences


class AskType(State):
    def __init__(self, keyword_dict) -> None:
        super().__init__(keyword_dict)

    def dialog(self):
        print("System: How expensive should the restaurant be?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.keyword_dict), self.extracted_preferences
        elif input == "bye":
            return Goodbye(self.keyword_dict), self.extracted_preferences
        elif input == "negate" or input == "negate": # if the user negate the ask, we should ask again
            return AskType(self.keyword_dict), self.extracted_preferences
        elif input == "inform":
            # extract preferences
            self.extracted_preferences = patternMatchKeywordExtraction(
                self.user_utterance, self.keyword_dict
            )
            return AskForInformation(self.keyword_dict), self.extracted_preferences


class Suggestion(State):
    def __init__(self, keyword_dict) -> None:
        super().__init__(keyword_dict)

    def dialog(self):
        print(
            "System: The best restaurant according to your preferences is this ",
            "---",
            ".",
        )
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        # after the suggestion it can negate/ back to ask again for parameters | it can ask for alternative
        # it can say by | it can confirm
        if input == "restart":
            return Welcome(self.keyword_dict), self.extracted_preferences
        elif input == "bye" or input == "thankyou":
            return Goodbye(self.keyword_dict), self.extracted_preferences
        elif input == "negate":
            return Welcome(self.keyword_dict), self.extracted_preferences
        elif input == "reqalts" or input == "reqmore" or input == "repeat" or input == "deny":
            return Suggestion(
                self.keyword_dict
            )  # if the user does not like the suggestion which is the next state?
        elif input == "affirm" or input == "request" or input == "ack" or input == "confirm":
            return GiveDetails(self.keyword_dict), self.extracted_preferences


class GiveDetails(State):
    def __init__(self, keyword_dict) -> None:
        super().__init__(keyword_dict)

    def dialog(self):
        print("Details: Address and Phone")
        # should we get the restaurant that we recommended to accesss to it details,
        # or store this restaurant as a self.restaurantsuggested
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.keyword_dict), self.extracted_preferences
        elif input == "bye" or input == "thankyou" or input == "ack" or input == "confirm" or input == "affirm":
            return Goodbye(self.keyword_dict), self.extracted_preferences
        elif input == "repeat":
            return GiveDetails(self.keyword_dict), self.extracted_preferences
        elif input == "negate" or input == "reqalts":
            return Suggestion(self.keyword_dict), self.extracted_preferences


class Goodbye(State):
    def __init__(self, keyword_dict) -> None:
        super().__init__(keyword_dict)

    def dialog(self):
        print("System: Goodbye, have a nice day!")

    def run(self):
        self.dialog()
