import csv
import random
import re
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
        self.user_utterance = ""

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
            return AskForInformation(self.keyword_dict, self.extracted_preferences), self.extracted_preferences


class AskForInformation(State):
    def __init__(self, keyword_dict, extracted_preferences) -> None:
        super().__init__(keyword_dict)
        self.extracted_preferences = extracted_preferences

    def dialog(self):
        user_utterance = "temporary string"

        return user_utterance

    def transition(self, input):
        # analyze preferences and route accordingly
        if "area" not in self.extracted_preferences:
            return AskArea(self.keyword_dict, self.extracted_preferences), self.extracted_preferences
        elif "pricerange" not in self.extracted_preferences:
            return AskPrice(self.keyword_dict, self.extracted_preferences), self.extracted_preferences
        elif "food" not in self.extracted_preferences:
            return AskType(self.keyword_dict, self.extracted_preferences), self.extracted_preferences
        else: #everything is filled
            return Suggestion (self.keyword_dict, self.extracted_preferences), self.extracted_preferences

class AskArea(State):
    def __init__(self, keyword_dict, extracted_preferences) -> None:
        super().__init__(keyword_dict)
        self.extracted_preferences = extracted_preferences

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
            return AskArea(self.keyword_dict, self.extracted_preferences), self.extracted_preferences
        elif input == "inform":
            # extract preferences
            self.extracted_preferences.update(patternMatchKeywordExtraction(
                self.user_utterance, self.keyword_dict)
            )
            return AskForInformation(self.keyword_dict, self.extracted_preferences), self.extracted_preferences
        else:
            return AskForInformation(self.keyword_dict, self.extracted_preferences), self.extracted_preferences


class AskPrice(State):
    def __init__(self, keyword_dict, extracted_preferences) -> None:
        super().__init__(keyword_dict)
        self.extracted_preferences = extracted_preferences

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
            return AskPrice(self.keyword_dict, self.extracted_preferences), self.extracted_preferences
        elif input == "inform":
            # extract preferences
            self.extracted_preferences.update(patternMatchKeywordExtraction(
                self.user_utterance, self.keyword_dict)
            )
            return AskForInformation(self.keyword_dict, self.extracted_preferences), self.extracted_preferences
        else:
            return AskForInformation(self.keyword_dict, self.extracted_preferences), self.extracted_preferences


class AskType(State):
    def __init__(self, keyword_dict, extracted_preferences) -> None:
        super().__init__(keyword_dict)
        self.extracted_preferences = extracted_preferences

    def dialog(self):
        print("System: What type of food would you like?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.keyword_dict), self.extracted_preferences
        elif input == "bye":
            return Goodbye(self.keyword_dict), self.extracted_preferences
        elif input == "negate" or input == "negate": # if the user negate the ask, we should ask again
            return AskType(self.keyword_dict, self.extracted_preferences), self.extracted_preferences
        elif input == "inform":
            # extract preferences
            self.extracted_preferences.update(patternMatchKeywordExtraction(
                self.user_utterance, self.keyword_dict)
            )
            return AskForInformation(self.keyword_dict, self.extracted_preferences), self.extracted_preferences
        else:
            return AskForInformation(self.keyword_dict, self.extracted_preferences), self.extracted_preferences


class Suggestion(State):
    def __init__(self, keyword_dict, extracted_preferences, previous_suggestion_index = None) -> None:
        super().__init__(keyword_dict)
        filename = "data/restaurant_info.csv"
        self.restaurant_lookup = RestaurantLookup(filename)
        self.extracted_preferences = extracted_preferences
        self.previous_suggestion_index = previous_suggestion_index
        self.suggestions = None

    def dialog(self):
        self.suggestions = self.restaurant_lookup.lookup(self.extracted_preferences)
        if(self.suggestions.values != []):
            random_index = 0
            while(self.previous_suggestion_index == random_index):
                random_index = random.randrange(0, (len(self.suggestions.values) - 1))
            self.previous_suggestion_index = random_index
            print(
                "System: The best restaurant according to your preferences is: \"",
                self.suggestions.values[random_index][0],
                "\""
                
            )
        else:
            print("System: No restaurants found.")
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
        elif input == "reqalts" or input == "reqmore":
            return Suggestion(
                self.keyword_dict,
                self.extracted_preferences,
                self.previous_suggestion_index
            ), self.extracted_preferences
        elif input == "request":
            return GiveDetails(self.keyword_dict, 
                               self.suggestions.values,
                               self.previous_suggestion_index,
                               self.user_utterance), self.extracted_preferences
        else:
            return Suggestion(
                self.keyword_dict,
                self.extracted_preferences
            ), self.extracted_preferences  # if the user does not like the suggestion which is the next state?


class GiveDetails(State):
    def __init__(self, keyword_dict, suggestions, 
                 previous_suggestion_index, request_utterance) -> None:
        super().__init__(keyword_dict)
        self.previous_suggestion_index = previous_suggestion_index
        self.suggestions = suggestions
        self.request_utterance = request_utterance

    def dialog(self):
        request_type = patternMatchRequest(self.request_utterance)

        if(request_type == "phone"):
            print("System: The phone number of this restaurant is:"
                  , self.suggestions[self.previous_suggestion_index][4])
        elif(request_type == "address"):
             print("System: The address number of this restaurant is:"
                  , self.suggestions[self.previous_suggestion_index][5])
        else:
            print("Sorry I can't understand this request")

        user_utterance = super().dialog()
        return user_utterance

    def transition(self, input):
        if input == "restart":
            return Welcome(self.keyword_dict), self.extracted_preferences
        elif input == "bye" or input == "thankyou" or input == "ack" or input == "confirm" or input == "affirm":
            return Goodbye(self.keyword_dict), self.extracted_preferences
        elif input == "repeat":
            return GiveDetails(self.keyword_dict, self.prevous_suggestion_index), self.extracted_preferences
        elif input == "request":
            return GiveDetails(self.keyword_dict, 
                               self.suggestions,
                               self.previous_suggestion_index,
                               self.user_utterance), self.extracted_preferences
        elif input == "negate" or input == "reqalts":
            return Suggestion(self.keyword_dict,
                              self.extracted_preferences,
                              self.previous_suggestion_index
                              ), self.extracted_preferences


class Goodbye(State):
    def __init__(self, keyword_dict) -> None:
        super().__init__(keyword_dict)

    def dialog(self):
        print("System: Goodbye, have a nice day!")

    def run(self):
        self.dialog()
