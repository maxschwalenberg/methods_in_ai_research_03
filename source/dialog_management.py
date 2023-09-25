import csv
import random
import re
import copy
from Levenshtein import distance as levdistance

from source.model import Model
from source.restaurant_lookup import RestaurantLookup


class DialogManagement:
    """The dialog system which handles and keeps track of
    the conversation.
    """

    def __init__(self, classifier: Model, debug=False) -> None:
        self.classifier = classifier

        # Gather the known keywords for the
        filename = "data/restaurant_info.csv"
        self.fetchKeywords(filename)

        # initialize the welcome state with no known preferences
        self.current_state = Welcome(self.keyword_dict, {}, {})

        # optionally enable debugging --> print classifications for each user input
        self.debug = debug

    def run_dialog(self):
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
        keyword_names = file.fieldnames[1:4]
        keyword_dict = {key: set() for key in keyword_names}

        for row in file:
            for keyword in keyword_names:
                if (row[keyword]) != "":
                    keyword_dict[keyword].add(row[keyword])

        self.keyword_dict = keyword_dict


def patternMatchRequest(data):
    data = data.lower()

    if re.findall("phone", data) != []:
        return "phone"
    elif re.match("address", data) != []:
        return "address"
    else:
        return None


def patternMatchKeywordExtraction(data, keyword_dict):
    data = data.lower()
    temp = None
    result = {}
    if temp := re.findall(r"in the (\w+)|(north|south|east|west)|(any)", data):
        area = ""
        for match in temp:
            if match[0]:
                area = match[0]
            elif match[1]:
                area = match[1]
            elif match[2]:
                area = "Any"

        result["area"] = area

    if temp := re.findall(r"(\w+) (priced|(expensive|cheap)|(any))", data):
        price_range = ""
        for match in temp:
            if match[0]:
                price_range = match[0]
                if match[1] and match[1].lower() == "any":
                    price_range = "Any"

        result["pricerange"] = price_range

    if temp := re.findall(r"(\w+) (food|(chinese|italian)|(any))", data):
        food_type = ""
        for match in temp:
            if match[0]:
                food_type = match[0]
                if match[1] and match[1].lower() == "any":
                    food_type = "Any"

        result["food"] = food_type

    if temp := re.findall("(\w+) restaurant", data):
        for key, values in keyword_dict.items():
            for value in values:
                if levdistance(temp[0], value) <= 2:
                    result[key] = value
    return result


class State:
    def __init__(
        self, keyword_dict, extracted_preferences: dict, extracted_preferences_old: dict
    ) -> None:
        self.keyword_dict = keyword_dict

        # create two dicts of extracted preferences. the old one always stores the extracted preferences of the previous state
        # that way the system can give feedback to the user if new preferences were detected
        self.extracted_preferences: dict = extracted_preferences
        self.extracted_preferences_old: dict = extracted_preferences_old
        self.user_utterance = ""

    def give_preferences_feedback(self):
        changed_values = {}

        for key in self.extracted_preferences:
            try:
                current_preference = self.extracted_preferences[key]
                old_preference = self.extracted_preferences_old[key]

                if current_preference != old_preference:
                    changed_values[key] = current_preference

            except:
                changed_values[key] = self.extracted_preferences[key]

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
                elif key == "price":
                    feedback_string += f"the price should be {changed_values[key]}"

            feedback_string += ". "

        self.feedback_string = feedback_string

    def run(self, classifier: Model):
        self.give_preferences_feedback()
        self.dialog()
        classified_response = classifier.predict_single_sentence(self.user_utterance)

        new_state = self.transition(classified_response)
        # print
        print(self.extracted_preferences)
        return new_state, classified_response

    def dialog(self):
        self.user_utterance = input("User: ")

    def transition(self, input):
        pass


class Welcome(State):
    def __init__(
        self, keyword_dict, extracted_preferences: dict, extracted_preferences_old: dict
    ) -> None:
        super().__init__(keyword_dict, extracted_preferences, extracted_preferences_old)

    def dialog(self):
        print(f"System: {self.feedback_string}Hello, how can I help you?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        extracted_preferences = {}
        if input == "restart" or input == "repeat":
            return Welcome(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif input == "bye":
            return Goodbye(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif input == "inform" or input == "hello":
            # extract preferences
            self.extracted_preferences_old = copy.deepcopy(self.extracted_preferences)
            self.extracted_preferences = patternMatchKeywordExtraction(
                self.user_utterance, self.keyword_dict
            )
            return AskForInformation(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )


class AskForInformation(State):
    def __init__(
        self, keyword_dict, extracted_preferences: dict, extracted_preferences_old: dict
    ) -> None:
        super().__init__(keyword_dict, extracted_preferences, extracted_preferences_old)

    def dialog(self):
        user_utterance = "temporary string"

        return user_utterance

    def transition(self, input):
        # analyze preferences and route accordingly
        if "area" not in self.extracted_preferences:
            return AskArea(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif "pricerange" not in self.extracted_preferences:
            return AskPrice(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif "food" not in self.extracted_preferences:
            return AskType(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        else:  # everything is filled
            return Suggestion(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )


class AskArea(State):
    def __init__(
        self, keyword_dict, extracted_preferences: dict, extracted_preferences_old: dict
    ) -> None:
        super().__init__(keyword_dict, extracted_preferences, extracted_preferences_old)

    def dialog(self):
        print(f"System: {self.feedback_string}Which area do you want to go?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif input == "bye":
            return Goodbye(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif (
            input == "negate" or input == "repeat"
        ):  # if the user negate the ask, we should ask again
            return AskArea(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif input == "inform":
            # extract preferences
            self.extracted_preferences_old = copy.deepcopy(self.extracted_preferences)

            self.extracted_preferences.update(
                patternMatchKeywordExtraction(self.user_utterance, self.keyword_dict)
            )
            return AskForInformation(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        else:
            return AskForInformation(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )


class AskPrice(State):
    def __init__(
        self, keyword_dict, extracted_preferences: dict, extracted_preferences_old: dict
    ) -> None:
        super().__init__(keyword_dict, extracted_preferences, extracted_preferences_old)

    def dialog(self):
        print(f"System: {self.feedback_string}How expensive should the restaurant be?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif input == "bye":
            return Goodbye(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif (
            input == "negate" or input == "repeat"
        ):  # if the user negate the ask, we should ask again
            return AskPrice(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif input == "inform":
            # extract preferences
            self.extracted_preferences_old = copy.deepcopy(self.extracted_preferences)
            self.extracted_preferences.update(
                patternMatchKeywordExtraction(self.user_utterance, self.keyword_dict)
            )
            return AskForInformation(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        else:
            return AskForInformation(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )


class AskType(State):
    def __init__(
        self, keyword_dict, extracted_preferences: dict, extracted_preferences_old: dict
    ) -> None:
        super().__init__(keyword_dict, extracted_preferences, extracted_preferences_old)

    def dialog(self):
        print(f"System: {self.feedback_string}What type of food would you like?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.keyword_dict)
        elif input == "bye":
            return Goodbye(self.keyword_dict)
        elif (
            input == "negate" or input == "negate"
        ):  # if the user negate the ask, we should ask again
            return AskType(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif input == "inform":
            # extract preferences
            self.extracted_preferences_old = copy.deepcopy(self.extracted_preferences)
            self.extracted_preferences.update(
                patternMatchKeywordExtraction(self.user_utterance, self.keyword_dict)
            )
            return AskForInformation(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        else:
            return AskForInformation(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )


class Suggestion(State):
    def __init__(
        self,
        keyword_dict,
        extracted_preferences: dict,
        extracted_preferences_old: dict,
        previous_suggestion_index=None,
    ) -> None:
        super().__init__(keyword_dict, extracted_preferences, extracted_preferences_old)

        filename = "data/restaurant_info.csv"
        self.restaurant_lookup = RestaurantLookup(filename)
        self.extracted_preferences = extracted_preferences
        self.previous_suggestion_index = previous_suggestion_index
        self.suggestions = None

    def dialog(self):
        self.suggestions = self.restaurant_lookup.lookup(self.extracted_preferences)
        if not self.suggestions.empty:
            random_index = 0
            while self.previous_suggestion_index == random_index:
                random_index = random.randrange(0, (len(self.suggestions.values) - 1))
            self.previous_suggestion_index = random_index
            print(
                f'System: {self.feedback_string}The best restaurant according to your preferences is: "',
                self.suggestions.values[random_index][0],
                '"',
            )
        else:
            print("System: No restaurants found.")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        # after the suggestion it can negate/ back to ask again for parameters | it can ask for alternative
        # it can say by | it can confirm
        if input == "restart":
            return Welcome(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif input == "bye" or input == "thankyou":
            return Goodbye(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif input == "negate":
            return Welcome(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif input == "reqalts" or input == "reqmore":
            return Suggestion(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
                self.previous_suggestion_index,
            )
        elif input == "request":
            return GiveDetails(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
                self.suggestions.values,
                self.previous_suggestion_index,
                self.user_utterance,
            )
        else:
            return Suggestion(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
                previous_suggestion_index=self.previous_suggestion_index,
            )  # if the user does not like the suggestion which is the next state?


class GiveDetails(State):
    def __init__(
        self,
        keyword_dict,
        extracted_preferences: dict,
        extracted_preferences_old: dict,
        suggestions,
        previous_suggestion_index,
        request_utterance,
    ) -> None:
        super().__init__(keyword_dict, extracted_preferences, extracted_preferences_old)

        self.previous_suggestion_index = previous_suggestion_index
        self.suggestions = suggestions
        self.request_utterance = request_utterance

    def dialog(self):
        request_type = patternMatchRequest(self.request_utterance)

        if request_type == "phone":
            print(
                f"System: {self.feedback_string}The phone number of this restaurant is:",
                self.suggestions[self.previous_suggestion_index][4],
            )
        elif request_type == "address":
            print(
                f"System: {self.feedback_string}The address number of this restaurant is:",
                self.suggestions[self.previous_suggestion_index][5],
            )
        else:
            print("Sorry I can't understand this request")

        user_utterance = super().dialog()
        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.keyword_dict)
        elif (
            input == "bye"
            or input == "thankyou"
            or input == "ack"
            or input == "confirm"
            or input == "affirm"
        ):
            return Goodbye(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
            )
        elif input == "repeat":
            return GiveDetails(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
                self.suggestions,
                self.previous_suggestion_index,
                self.request_utterance,
            )
        elif input == "request":
            return GiveDetails(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
                self.suggestions,
                self.previous_suggestion_index,
                self.request_utterance,
            )
        elif input == "negate" or input == "reqalts":
            return Suggestion(
                self.keyword_dict,
                self.extracted_preferences,
                self.extracted_preferences_old,
                self.previous_suggestion_index,
            )


class Goodbye(State):
    def __init__(
        self, keyword_dict, extracted_preferences: dict, extracted_preferences_old: dict
    ) -> None:
        super().__init__(keyword_dict, extracted_preferences, extracted_preferences_old)

    def dialog(self):
        print("System: Goodbye, have a nice day!")

    def run(self):
        self.dialog()
