import csv
import random

from source.model import Model
from source.restaurant_lookup import RestaurantLookup

# TODO
# 1. create a rules dict what the system says in each state
# 2. create a state transitions dict to change state based on a classification result


class DialogManagement:
    def __init__(self, classifier: Model, debug=False) -> None:
        self.classifier = classifier

        self.current_state = Welcome()
        self.know_preferences: dict = {}
        self.know_preferences = {
            "area": "west",
            "pricerange": "moderate",
            "type": "british",
        }

        self.restaurant_lookup = RestaurantLookup("data/restaurant_info.csv")

        # optionally enable debugging --> print classifications for each user input
        self.debug = debug

    def run_dialog(self):
        while not isinstance(self.current_state, Goodbye):
            new_state, classified_response = self.current_state.run(self.classifier)
            self.current_state = new_state
            #if the user denies, we should maintain the current state

            if self.debug:
                print(f"Classified `{classified_response}`")

        # run the goodbye state
        self.current_state.run()


class State:
    def run(self, classifier: Model):
        user_response = self.dialog()
        classified_response = classifier.predict_single_sentence(user_response)

        new_state = self.transition(classified_response)

        return new_state, classified_response

    def dialog(self):
        user_utterance = input("User: ")
        return user_utterance

    def transition(self, input):
        # if user wants to start over, it does not matter which is the current state
        # the same with bye
        if input == "restart":
            return Welcome()
        elif input == "bye":
            return Goodbye()
        else:
            pass


class Welcome(State):
    def dialog(self):
        print("System: Hello, how can I help you?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "bye":
            return Goodbye()

        elif input == "inform":
            # what do we do if the user already gives us information?


            # return None for now, which will result in an error
            return None

        else:
            # return AskPrice() in the default case
            return AskArea()


class AskArea(State):
    def dialog(self):
        print("System: Which area do you want to go?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "inform":
            return AskPrice()


class AskPrice(State):
    def dialog(self):
        print("System: How expensive should the restaurant be?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "inform":
            return AskType()


class AskType(State):
    def dialog(self):
        print("System: How expensive should the restaurant be?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "inform":
            return Suggestion()


class Suggestion(State):

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
        if input == "negate":
            return Welcome()
        elif input == "reqalts" or input == "reqmore": 
            return Suggestion() #if the user does not like the suggestion which is the next state?
        elif input == "thankyou":
            return Goodbye()
        elif input == "affirm" or input == "request":
            return GiveDetails()

class GiveDetails(State):
    def dialog(self):
        print("Details: Address and Phone")
        # should we get the restaurant that we recommended to accesss to it details,
        # or store this restaurant as a self.restaurantsuggested
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "thankyou":
            return Goodbye()
        elif input == "negate":
            return Suggestion

class Goodbye(State):
    def dialog(self):
        print("System: Goodbye, have a nice day!")

    def run(self):
        self.dialog()
