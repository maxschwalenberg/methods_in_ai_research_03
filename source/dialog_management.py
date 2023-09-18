from source.model import Model

# TODO
# 1. create a rules dict what the system says in each state
# 2. create a state transitions dict to change state based on a classification result


class DialogManagement:
    def __init__(self, classifier: Model) -> None:
        self.classifier = classifier

        # TODO: set first state
        self.current_state = "welcome"

    def start_dialog(self):
        while self.current_state != "goodbye":
            classification_result = self.handle_current_state()
            self.state_transition(classification_result)

    def handle_current_state(self):
        rules = {"welcome": "Hello, how can I help you?"}

        # check current state and print predefined system utterance
        # also receive the response of the user which is returned by the below function
        user_res = self.make_utterance_and_request(rules[self.current_state])
        classified_user_res = self.classify_user_utterance(user_res)

        return classified_user_res

    def state_transition(self, classification_result: str):
        if self.current_state == "welcome":
            if classification_result == "ack":
                self.current_state = "request_area"

    def classify_user_utterance(self, user_utteranec: str):
        # return the classification result of the users utterance
        return self.classifier.predict_single_sentence(user_utteranec)

    def make_utterance_and_request(self, utterance_string: str):
        user_answer = input(utterance_string)
        return user_answer
