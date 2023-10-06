from source.datacreator import Datacreator
from source.model import Model
import json
import re


class RuleBasedBaseline(Model):

    def __init__(self, datacreator_instance: Datacreator, rulesfile : str) -> None:
        super().__init__(datacreator_instance)
        self.rule_data = []
        self.loadRulesFile(rulesfile)

    def load_rules_file(self, filename):
        file = open(filename).read()
        self.rule_data = json.loads(file)

    def predict_single_utterance(self):
        print("Enter utterance: ")
        utterance = input()
        for rule in self.rule_data["rules"]:
            for keyword in rule["utterances"]:
                if re.search(r"\b" + keyword + r"\b", utterance):
                    print(rule["intent"])
                    return
        print("null")

    def predict(self, input_list):
        result_list = []
        for utterance in input_list:
            has_found = False
            for rule in self.rule_data["rules"]:
                for keyword in rule["keywords"]:
                    if re.search(r"\b" + keyword + r"\b", utterance):
                        result_list.append(rule["intent"])
                        has_found = True
                        break
                if has_found:
                    break
            if not has_found:
                result_list.append("null")
        self.preds = result_list

    def develop(self):
        self.predict(self.datacreator_instance.x_test)
        self.evaluate()


class MajorityClassBaseline(Model):
    def __init__(self, datacreator_instance: Datacreator) -> None:
        super().__init__(datacreator_instance)
        self.majority = None
        self.findMajority(self.datacreator_instance.y_test)

    def find_majority(self, input_list):
        count_dict = {}
        for intent in input_list:
            if intent not in count_dict:
                count_dict[intent] = 1
            else:
                count_dict[intent] += 1
        self.majority = max(count_dict, key=count_dict.get)

    def predict_single_utterance(self):
        print("Enter utterance: ")

        # because we predict the majority anyways we can disregard the input
        _ = input()
        print(self.majority)

    def predict(self, input_list):
        self.preds = [self.majority] * len(input_list)

    def develop(self):
        self.predict(self.datacreator_instance.x_test)
        self.evaluate()
