from source.model import Model

# TODO
# 1. create a rules dict what the system says in each state
# 2. create a state transitions dict to change state based on a classification result


class DialogManagement:
    def __init__(self, classifier: Model, debug=False) -> None:
        self.classifier = classifier

        self.current_state = Welcome()
        self.know_preferences: dict = {}

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


class AskPrice(State):
    def dialog(self):
        print("System: How expensive should the restaurant be?")
        user_utterance = super().dialog()

        return user_utterance

    def transition(self, input):
        if input == "inform":
            return AskPrice()


class Goodbye(State):
    def dialog(self):
        print("System: Goodbye, have a nice day!")

    def run(self):
        self.dialog()
