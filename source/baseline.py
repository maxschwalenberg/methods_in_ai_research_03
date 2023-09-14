from source.datacreator import datacreator
from source.model import Model
import json

class Baseline(Model):

    def __init__(self, datacreator_instance: datacreator) -> None:
        super().__init__(datacreator_instance)


class RuleBasedBaseline (Model):

    def __init__(self, datacreator_instance: datacreator) -> None:
        super().__init__(datacreator_instance)
        self.rule_data = []
    
    def loadRulesFile(self, filename):
        file = open(filename).read()
        self.rule_data = json.loads(file)

    def predict(self, utterance):
        for rule in self.rule_data['rules']:
            if utterance in rule['keywords']:
                return rule['intent']
      

    def test(self):
        counter = 0.0
        correct_counter = 0.0
        ack_counter = 0.0
        ack_correct = 0.0
        for utterance, dialog_act in zip(self.datacreator_instance.x_train, self.datacreator_instance.y_train):
            for rule in self.rule_data['rules']:
                if (any(keyword in utterance for keyword in rule['keywords'])) and dialog_act in rule['intent'] :
                    correct_counter += 1
                    break
            counter += 1
        print(correct_counter / counter)
    