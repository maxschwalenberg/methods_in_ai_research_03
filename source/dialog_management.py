import csv
import random

from source.model import Model

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
            "type": "british"   
        }

        # optionally enable debugging --> print classifications for each user input
        self.debug = debug

    def run_dialog(self):
        while not isinstance(self.current_state, Goodbye):
            new_state, classified_response = self.current_state.run(self.classifier)
            self.current_state = new_state

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
            return AskPrice()

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
    def getRestaurant(self):
        # Open Database
        with open("restaurant_info.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            restaurants = list(reader)

        # Search restaurants according to the preference
        filtered_restaurants = []
        for restaurant in restaurants:
            if (restaurant["area"] == self.know_preferences["area"] and restaurant["pricerange"] == self.know_preferences["pricerange"] and
            restaurant["food"] == self.know_preferences["type"]):
                filtered_restaurants.append(restaurant)
        
        # Pick a random restaurant from the list
        if filtered_restaurants:
            selected_restaurant = random.choice(filtered_restaurants)
            restaurant_name = selected_restaurant["restaurantname"]
        else:
            restaurant_name = "None"
        return restaurant_name

    def dialog(self):
        restaurant = self.getRestaurant()
        if (restaurant == "None"):
            print("System: There are no restaurants according to your preferences.")
        else:
            print("System: The best restaurant according to your preferences is this ", restaurant, ".")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "inform":
            return Suggestion()

class Goodbye(State):
    def dialog(self):
        print("System: Goodbye, have a nice day!")

    def run(self):
        self.dialog()
